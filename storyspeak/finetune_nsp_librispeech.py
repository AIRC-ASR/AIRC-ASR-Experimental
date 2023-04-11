import nltk
import torch
import yaml
from torch.nn.parallel import DataParallel
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer, OPTForCausalLM

from train_siamese_network import SiameseNetwork
from util import (custom_loss_unsupervised, generate_continuations,
                  ls_create_data_loaders)

nltk.download('wordnet')
import os

from loguru import logger


def contrastive_loss(output, label, margin=1.0):
    # convert the label to a float tensor
    label = label.float()
    
    # calculate the distance between the two embeddings
    distance = (1 - output) / 2
    
    # calculate the loss for similar pairs
    loss_similar = label * torch.pow(distance, 2)
    
    # calculate the loss for dissimilar pairs
    loss_dissimilar = (1 - label) * torch.pow(torch.clamp(margin - distance, min=0.0), 2)
    
    # calculate the total loss
    loss = loss_similar + loss_dissimilar
    
    return torch.mean(loss)



def train_nsp(model, siamese_net, optimizer, train_loader, val_loader, device, epochs, coherence_coeff, relevance_coeff, diversity_coeff, max_length, tokenizer, save_dir, max_grad_norm):
  for epoch in range(epochs):
    model.train()
    total_loss = 0
    num_batches = 0
    for batch in tqdm(train_loader, desc="Training Finetuned LibriSpeech", leave=False):
      input_ids = batch["utterance1_input_ids"].to(device)
      attention_masks = batch["utterance1_attention_mask"].to(device)
      labels = batch["label"].to(device)

      prompts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

      # Generate the continuations using the language model
      continuations = generate_continuations(model, tokenizer, prompts, device, max_length)

      # Reconstruct the continuation token IDs and attention masks
      batch_encodings = tokenizer.batch_encode_plus(continuations, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
      continuation_token_ids = batch_encodings["input_ids"]
      continuation_attention_masks = batch_encodings["attention_mask"]

      # Wrap the model with DataParallel
      if torch.cuda.device_count() > 1:
        siamese_net = DataParallel(siamese_net)

      # Run it through the siamese network and evaluate the loss
      siamese_output = siamese_net(
        input_ids1=input_ids,
        attention_mask1=attention_masks,
        input_ids2=continuation_token_ids,
        attention_mask2=continuation_attention_masks
      )

      # Calculate the contrastive loss function using the siamese network output
      current_loss = contrastive_loss(siamese_output, labels)
      current_loss.requires_grad_(True)
      total_loss += current_loss.item()

      num_batches += 1
      if num_batches * (input_ids.shape[0]) % 100 == 0:   
        logger.info("Batch {}/{} - LOSS: {:.4f}", num_batches, len(test_loader), total_loss / num_batches)

      optimizer.zero_grad()
      current_loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
      optimizer.step()

    # Find the final training loss
    total_loss /= num_batches

    val_loss, val_diversity_loss, val_coherence_loss, val_relevance_loss = \
    evaluate_nsp(model, siamese_net, val_loader, device, coherence_coeff, relevance_coeff, diversity_coeff, max_length, tokenizer)
    logger.info("VALIDATION - LOSS: {:.4f}, DIV: {:.4f}, COH: {:.4f}, REL: {:.4f}",
                 val_loss, val_diversity_loss, val_coherence_loss, val_relevance_loss)
  
    # Save checkpoint
    if epoch == epochs - 1:
      file_name = "final_finetune_nsp.pt"
    else:
      file_name = f"finetune_nsp_{epoch + 1}.pt"

    checkpoint_path = os.path.join(save_dir, file_name)
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict() if torch.cuda.device_count() > 1 else model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': total_loss,
        'val_loss': val_loss,
        'val_diversity_loss': val_diversity_loss,
        'val_coherence_loss': val_coherence_loss,
        'val_relevance_loss': val_relevance_loss
    }, checkpoint_path)
    logger.info(f"Checkpoint saved at {checkpoint_path}")


def evaluate_nsp(model, siamese_net, val_loader, device, coherence_coeff, relevance_coeff, diversity_coeff, max_length, tokenizer):
  model.eval()
  total_loss = 0
  num_batches = 0
  total_diversity_loss = 0
  total_coherence_loss = 0
  total_relevance_loss = 0
  with torch.no_grad():
    for batch in tqdm(val_loader, desc="Evaluating Finetuned LibriSpeech", leave=False):
      input_ids = batch["utterance1_input_ids"].to(device)
      attention_masks = batch["utterance1_attention_mask"].to(device)
      labels = batch["label"].to(device)

      prompts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

      # Generate the continuations using the language model
      continuations = generate_continuations(model, tokenizer, prompts, device, max_length)

      # Reconstruct the continuation token IDs and attention masks
      batch_encodings = tokenizer.batch_encode_plus(continuations, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
      continuation_token_ids = batch_encodings["input_ids"]
      continuation_attention_masks = batch_encodings["attention_mask"]

      # Run it through the siamese network and evaluate the loss
      siamese_output = siamese_net(
        input_ids1=input_ids,
        attention_mask1=attention_masks,
        input_ids2=continuation_token_ids,
        attention_mask2=continuation_attention_masks
      )

      # Calculate the contrastive loss function using the siamese network output
      current_loss = contrastive_loss(siamese_output, labels)
      current_loss.requires_grad_(True)
      total_loss += current_loss.item()

      coherence_loss, relevance_loss, diversity_loss = custom_loss_unsupervised(model, tokenizer, input_ids, device, max_length)
      total_coherence_loss += (coherence_coeff * coherence_loss)
      total_relevance_loss += (relevance_coeff * relevance_loss)
      total_diversity_loss += (diversity_coeff * diversity_loss)

      num_batches += 1
      if num_batches * (input_ids.shape[0]) % 50 == 0:   
        logger.info("Batch {}/{} - LOSS: DIV: {:.4f}, COH: {:.4f}, REL: {:.4f}",
          num_batches, len(val_loader),
          total_diversity_loss / num_batches,
          total_coherence_loss / num_batches,
          total_relevance_loss / num_batches
        )

  # Find the final evaluation loss
  total_loss /= num_batches
  total_diversity_loss /= num_batches
  total_coherence_loss /= num_batches
  total_relevance_loss /= num_batches

  return total_loss, total_diversity_loss, total_coherence_loss, total_relevance_loss


if __name__ == "__main__":
    # Load the hyperparameters from the config file
    CONFIG_FILE_PATH = "experiments/librispeech/finetuning/finetune1/config.yaml"
    with open(CONFIG_FILE_PATH, "r") as f:
      config = yaml.safe_load(f)

    (batch_size, learning_rate, num_epochs,
    max_seq_length, coherence_coeff, relevance_coeff,
    diversity_coeff, model_name, max_length, save_dir, max_grad_norm) = (config["batch_size"], config["learning_rate"], config["num_epochs"],
                              config["max_seq_length"], config["coherence_coeff"], config["relevance_coeff"],
                              config["diversity_coeff"], config["model_name"], config["max_length"],
                              config["save_dir"], config["max_grad_norm"])

    # Set up the device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    model = OPTForCausalLM.from_pretrained(model_name, output_hidden_states=True).to(device)
    if torch.cuda.device_count() > 1:
      model = DataParallel(model)

    # Set up the optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Load the common voice dataset
    # Set up the dataset and data loaders
    train_loader, val_loader, test_loader = ls_create_data_loaders(tokenizer, batch_size, max_seq_length)

    # Evaluate on the test set
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)

    # Get the fine-tuned Siamese model
    logger.info("Loading the fine-tuned Siamese model...")

    # Make dummy model to load the weights
    siamese_net = SiameseNetwork(model).to(device)
    siamese_net_optimizer = AdamW(siamese_net.parameters(), lr=learning_rate)

    # Get the checkpoint
    SIAMESE_MODEL_PATH = "checkpoints/final_finetune_siamese.pt"
    checkpoint = torch.load(SIAMESE_MODEL_PATH)

    # Load the weights
    siamese_net.load_state_dict(checkpoint['model_state_dict'])
    siamese_net_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    logger.info(f'Siamese model loaded, had trained for {checkpoint["epoch"]} epochs and val loss of {checkpoint["val_loss"]}!')

    logger.info("Finetuning the language model on LibriSpeech for Next Sentence Prediction...")
    train_nsp(model, siamese_net, optimizer, train_loader, val_loader, device,
          num_epochs, coherence_coeff, relevance_coeff, diversity_coeff,
          max_length, tokenizer, save_dir, max_grad_norm)
