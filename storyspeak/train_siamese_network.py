import nltk
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer, OPTForCausalLM

from util import ls_create_data_loaders

nltk.download('wordnet')
import os

from loguru import logger


class SiameseNetwork(nn.Module):
  def __init__(self, opt_model):
    super(SiameseNetwork, self).__init__()
    self.opt_model = opt_model
    self.fc1 = nn.Linear(opt_model.config.hidden_size, 64)
    self.fc2 = nn.Linear(64, 1)
      
  def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
      with torch.no_grad():
        outputs1 = self.opt_model(input_ids1, attention_mask1)
        last_hidden_state1 = outputs1.hidden_states[-1]
        pooled_output1 = last_hidden_state1[:, 0]
        # or use a pooling method to get a fixed-size vector from the last hidden state

        outputs2 = self.opt_model(input_ids2, attention_mask2)
        last_hidden_state2 = outputs2.hidden_states[-1]
        pooled_output2 = last_hidden_state2[:, 0]
        # or use a pooling method to get a fixed-size vector from the last hidden state

      output = torch.cat((pooled_output1, pooled_output2), dim=1)
      output = self.fc1(output)
      output = F.relu(output)
      output = self.fc2(output)
      return output


def train_siamese(model, optimizer, train_loader, val_loader, device, epochs, save_dir, criterion):
  for epoch in range(epochs):
    model.train()
    total_loss = 0
    num_batches = 0
    for batch in tqdm(train_loader, desc="Training Siamese Network", leave=False):
      utterance1_input_ids = batch["utterance1_input_ids"].to(device)
      utterance1_attention_mask = batch["utterance1_attention_mask"].to(device)
      utterance2_input_ids = batch["utterance2_input_ids"].to(device)
      utterance2_attention_mask = batch["utterance2_attention_mask"].to(device)
      labels = batch["label"].to(device)

      # Forward pass
      outputs = model(
          input_ids1=utterance1_input_ids,
          attention_mask1=utterance1_attention_mask,
          input_ids2=utterance2_input_ids,
          attention_mask2=utterance2_attention_mask,
      )
      outputs = outputs.squeeze() # remove extra dimension
      loss = criterion(outputs, labels.float())
    
      # Backward pass and optimization
      loss.backward()
      optimizer.step()

      total_loss += loss.item()
      num_batches += 1
    
    avg_loss = total_loss / num_batches
    logger.info("TRAIN - LOSS: {:.4f}", avg_loss)

    # Evaluate on validation set
    val_loss = evaluate_siamese(model, val_loader, device, epoch, criterion) 
    logger.info("VALIDATION - LOSS: {:.4f}", val_loss)

    # Save checkpoint
    if epoch == epochs - 1:
      file_name = "final_finetune_siamese.pt"
    else:
      file_name = f"finetune_siamese_{epoch + 1}.pt"

    checkpoint_path = os.path.join(save_dir, file_name)
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': total_loss,
        'val_loss': val_loss,
    }, checkpoint_path)
    logger.info(f"Checkpoint saved at {checkpoint_path}")


def evaluate_siamese(model, val_loader, device, epoch, criterion):
  # Evaluation on validation set
  model.eval()
  total_val_loss = 0
  num_val_batches = 0
  avg_val_loss = 0
  with torch.no_grad():
    for val_batch in tqdm(val_loader, desc="Evaluating Siamese Network", leave=False):
      val_utterance1_input_ids = val_batch["utterance1_input_ids"].to(device)
      val_utterance1_attention_mask = val_batch["utterance1_attention_mask"].to(device)
      val_utterance2_input_ids = val_batch["utterance2_input_ids"].to(device)
      val_utterance2_attention_mask = val_batch["utterance2_attention_mask"].to(device)
      val_labels = val_batch["label"].to(device)

      val_outputs = model(
        input_ids1=val_utterance1_input_ids,
        attention_mask1=val_utterance1_attention_mask,
        input_ids2=val_utterance2_input_ids,
        attention_mask2=val_utterance2_attention_mask,
      )
      val_outputs = val_outputs.squeeze() # remove extra dimension
      val_loss = criterion(val_outputs, val_labels.float())

      preds = torch.round(torch.sigmoid(val_outputs))
      accuracy = (preds == val_labels).float().mean()
      logger.info(f"Epoch: {epoch + 1}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {accuracy:.4f}")

      total_val_loss += val_loss.item()
      num_val_batches += 1

    avg_val_loss = total_val_loss / num_val_batches
    logger.info(f"Epoch: {epoch + 1}, Validation Loss: {avg_val_loss:.4f}")

  return avg_val_loss


if __name__ == '__main__':
  # Load the hyperparameters from the config file
  CONFIG_FILE_PATH = "experiments/siamese/finetuning/finetune1/config.yaml"
  with open(CONFIG_FILE_PATH, "r") as f:
    config = yaml.safe_load(f)

  (batch_size, learning_rate, num_epochs,
  max_seq_length, model_name, max_length, save_dir) = (
    config["batch_size"], config["learning_rate"], config["num_epochs"],
    config["max_seq_length"], config["model_name"], config["max_length"], config["save_dir"])

  # Set up the device for training
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # Create the model and tokenizer
  tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
  opt_model = OPTForCausalLM.from_pretrained(model_name, output_hidden_states=True).to(device)

  # create instance of SiameseNetwork
  siamese_net = SiameseNetwork(opt_model).to(device)

  # Load the librispeech dataset
  # Set up the dataset and data loaders
  train_loader, val_loader, test_loader = ls_create_data_loaders(tokenizer, batch_size, max_seq_length)

  # Set up the optimizer and loss function
  optimizer = torch.optim.Adam(siamese_net.parameters(), lr=learning_rate)
  criterion = torch.nn.CrossEntropyLoss()

  # Train the model
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  logger.info("Training the Siamese Network for Next Sentence Prediction...")
  train_siamese(siamese_net, optimizer, train_loader, val_loader, device, num_epochs, save_dir, criterion)

  # Evaluate the model on the test set
  logger.info("Evaluating the Siamese Network on the test set...")
  test_Loss = evaluate_siamese(siamese_net, test_loader, device, num_epochs, criterion)
  logger.info("Test Loss: {:.4f}", test_Loss)
