import nltk
import torch
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer, OPTForCausalLM

from util import custom_loss_unsupervised, cv_create_data_loaders

nltk.download('wordnet')
import os

from loguru import logger


def evaluate_cv(model, test_loader, device, coherence_coeff, relevance_coeff, diversity_coeff, max_length, tokenizer, save_dir):
  '''
  This function evaluates the common voice baseline model on the test set.
  It computes the loss on the test set and saves the model.
  It also computes the coherence, relevance and diversity loss on the test set.

  '''
  model.eval()
  total_loss = 0
  total_coherence_loss = 0
  total_relevance_loss = 0
  total_diversity_loss = 0
  num_batches = 0
  with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating Common Voice Model", leave=False):
      input_ids = batch["input_ids"].to(device)

      coherence_loss, relevance_loss, diversity_loss = custom_loss_unsupervised(model, tokenizer, input_ids, device, max_length)
      total_loss += coherence_loss + relevance_loss + diversity_loss

      total_coherence_loss += (coherence_coeff * coherence_loss)
      total_relevance_loss += (relevance_coeff * relevance_loss)
      total_diversity_loss += (diversity_coeff * diversity_loss)

      num_batches += 1
      if num_batches * (input_ids.shape[0]) % 100 == 0:   
        logger.info("Batch {}/{} - LOSS: DIV: {:.4f}, COH: {:.4f}, REL: {:.4f}",
          num_batches, len(test_loader),
          total_diversity_loss / num_batches,
          total_coherence_loss / num_batches,
          total_relevance_loss / num_batches
        )

  total_loss /= num_batches
  total_diversity_loss /= num_batches
  total_coherence_loss /= num_batches
  total_relevance_loss /= num_batches

  file_name = f"final_cv_baseline.pt"
  checkpoint_path = os.path.join(save_dir, file_name)
  torch.save({
      'model_state_dict': model.state_dict() if torch.cuda.device_count() > 1 else model.module.state_dict(),
      'test_loss': total_loss,
      'test_diversity_loss': total_diversity_loss,
      'test_coherence_loss': total_coherence_loss,
      'test_relevance_loss': total_relevance_loss
  }, checkpoint_path)
  logger.info(f"Baseline Common Voice model saved at {checkpoint_path}")

  return total_loss, total_diversity_loss, total_coherence_loss, total_relevance_loss


if __name__ == "__main__":
    # Load the hyperparameters from the config file
    CONFIG_FILE_PATH = "experiments/commonvoice/baselines/baseline1.yaml"
    with open(CONFIG_FILE_PATH, "r") as f:
      config = yaml.safe_load(f)

    (model_name, batch_size, max_seq_length,
     coherence_coeff, relevance_coeff, diversity_coeff,
     max_length, save_dir) = (
      config["model_name"], config["batch_size"], config["max_seq_length"],
      config["coherence_coeff"], config["relevance_coeff"], config["diversity_coeff"],
      config["max_length"], config["save_dir"])

    # Set up the device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    model = OPTForCausalLM.from_pretrained(model_name).to(device)
    if torch.cuda.device_count() > 1:
      model = torch.nn.DataParallel(model)

    # Load the common voice dataset
    # Set up the dataset and data loaders
    train_loader, val_loader, test_loader = cv_create_data_loaders(tokenizer, batch_size, max_seq_length)

    # Evaluate on the test set
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)

    logger.info("Evaluating the Common Voice baseline model on the test set...")
    total_loss, total_diversity_loss, total_coherence_loss, total_relevance_loss = \
      evaluate_cv(model, test_loader, device, coherence_coeff, relevance_coeff, diversity_coeff, max_length, tokenizer, save_dir)

    logger.info("FINAL COMMON VOICE BASELINE - LOSS: {:.4f}, DIV: {:.4f}, COH: {:.4f}, REL: {:.4f}",
                 total_loss, total_diversity_loss, total_coherence_loss, total_relevance_loss)
