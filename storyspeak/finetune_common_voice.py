import nltk
import torch
import yaml
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer, OPTForCausalLM

from util import custom_loss_unsupervised, cv_create_data_loaders
from baseline_common_voice import evaluate_cv
from torch.nn.parallel import DataParallel

nltk.download('wordnet')
import os

from loguru import logger


def train_cv(model, optimizer, train_loader, val_loader, device, epochs, coherence_coeff, relevance_coeff, diversity_coeff, max_length, tokenizer, save_dir, max_grad_norm):
  for epoch in range(epochs):
    model.train()
    total_loss = 0
    total_coherence_loss = 0
    total_relevance_loss = 0
    total_diversity_loss = 0
    num_batches = 0
    for batch in tqdm(train_loader, desc="Training Finetuned Common Voice", leave=False):
      input_ids = batch["input_ids"].to(device)
      coherence_loss, relevance_loss, diversity_loss = custom_loss_unsupervised(model, tokenizer, input_ids, device, max_length)
      total_loss += coherence_loss + relevance_loss + diversity_loss

      total_coherence_loss += (coherence_coeff * coherence_loss)
      total_relevance_loss += (relevance_coeff * relevance_loss)
      total_diversity_loss += (diversity_coeff * diversity_loss)

      current_loss = (coherence_coeff * coherence_loss) + (relevance_coeff * relevance_loss) + (diversity_coeff * diversity_loss)
      total_loss += current_loss
      current_loss = torch.tensor(current_loss, requires_grad=True)

      num_batches += 1
      if num_batches * (input_ids.shape[0]) % 100 == 0:   
        logger.info("Batch {}/{} - LOSS: {:.4f}, DIV: {:.4f}, COH: {:.4f}, REL: {:.4f}",
          num_batches, len(test_loader), total_loss / num_batches,
          total_diversity_loss / num_batches,
          total_coherence_loss / num_batches,
          total_relevance_loss / num_batches
        )

      optimizer.zero_grad()
      current_loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
      optimizer.step()
    
    total_loss /= num_batches
    total_diversity_loss /= num_batches
    total_coherence_loss /= num_batches
    total_relevance_loss /= num_batches
  
    val_loss, val_diversity_loss, val_coherence_loss, val_relevance_loss = \
    evaluate_cv(model, val_loader, device, coherence_coeff, relevance_coeff, diversity_coeff, max_length, tokenizer)
    logger.info("VALIDATION - LOSS: {:.4f}, DIV: {:.4f}, COH: {:.4f}, REL: {:.4f}",
                 val_loss, val_diversity_loss, val_coherence_loss, val_relevance_loss)
    # Save checkpoint
    if epoch == epochs - 1:
      file_name = "final_finetune_cv.pt"
    else:
      file_name = f"finetune_cv_{epoch + 1}.pt"

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


if __name__ == "__main__":
    # Load the hyperparameters from the config file
    CONFIG_FILE_PATH = "experiments/commonvoice/finetuning/finetune1/config.yaml"
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
    model = OPTForCausalLM.from_pretrained(model_name).to(device)
    if torch.cuda.device_count() > 1:
      model = DataParallel(model)

    # Set up the optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Load the common voice dataset
    # Set up the dataset and data loaders
    train_loader, val_loader, test_loader = cv_create_data_loaders(tokenizer, batch_size, max_seq_length)

    # Evaluate on the test set
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)

    logger.info("Finetuning the language model on Common Voice for coherency, diversity and relevancy...")
    train_cv(model, optimizer, train_loader, val_loader, device,
          num_epochs, coherence_coeff, relevance_coeff, diversity_coeff,
          max_length, tokenizer, save_dir, max_grad_norm)

    logger.info("Evaluating the Common Voice finetuned model on the test set...")
    total_loss, total_diversity_loss, total_coherence_loss, total_relevance_loss = \
      evaluate_cv(model, test_loader, device, coherence_coeff, relevance_coeff, diversity_coeff, max_length, tokenizer, save_dir)
    logger.info("FINAL FINETUNED COMMON VOICE - LOSS: {:.4f}, DIV: {:.4f}, COH: {:.4f}, REL: {:.4f}",
                 total_loss, total_diversity_loss, total_coherence_loss, total_relevance_loss)
