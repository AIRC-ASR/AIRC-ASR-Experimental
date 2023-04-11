import nltk
import torch
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer, OPTForCausalLM

from util import custom_loss_unsupervised, ls_create_data_loaders
from torch.nn.parallel import DataParallel

nltk.download('wordnet')
import os

from loguru import logger



def evaluate_cv(model, test_loader, device, coherence_coeff, relevance_coeff, diversity_coeff, max_length, tokenizer, save_dir):
    model.eval()
    total_loss = 0
    total_coherence_loss = 0
    total_relevance_loss = 0
    total_diversity_loss = 0
    num_batches = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating LibriSpeech Model", leave=False):
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

    file_name = f"final_ls_baseline.pt"
    checkpoint_path = os.path.join(save_dir, file_name)
    torch.save({
        'model_state_dict': model.state_dict() if torch.cuda.device_count() > 1 else model.module.state_dict(),
        'test_loss': total_loss,
        'test_diversity_loss': total_diversity_loss,
        'test_coherence_loss': total_coherence_loss,
        'test_relevance_loss': total_relevance_loss
    }, checkpoint_path)
    logger.info(f"Baseline LibriSpeech model saved at {checkpoint_path}")

    return total_loss, total_diversity_loss, total_coherence_loss, total_relevance_loss


if __name__ == '__main__':
  # Load the hyperparameters from the config file
  CONFIG_FILE_PATH = "experiments/librispeech/baselines/baseline1.yaml"
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

  # Create the model, tokenizer and loss function
  tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
  model = OPTForCausalLM.from_pretrained(model_name).to(device)
  if torch.cuda.device_count() > 1:
    model = DataParallel(model)
  criterion = torch.nn.CrossEntropyLoss()

  # Load the librispeech dataset
  # Set up the dataset and data loaders
  train_dataset, val_dataset, test_dataset = ls_create_data_loaders(tokenizer, batch_size, max_seq_length)
  print('train_dataset', len(train_dataset), 'val_dataset', len(val_dataset), 'test_dataset', len(test_dataset))