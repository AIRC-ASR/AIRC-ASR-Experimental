import nltk
import torch
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer, OPTForCausalLM

from util import custom_loss_supervised, wp_create_data_loaders

nltk.download('wordnet')
import os

from jiwer import wer
from loguru import logger
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from torch.nn.parallel import DataParallel


def evaluate_wp(model, test_loader, device, coherence_coeff, relevance_coeff, diversity_coeff, max_length, tokenizer, save_dir):
    model.eval()
    total_loss = 0
    total_coherence_loss = 0
    total_relevance_loss = 0
    total_diversity_loss = 0
    total_wer = 0
    total_bleu = 0
    total_rouge = 0
    total_meteor = 0
    num_batches = 0
    smoothing = SmoothingFunction().method1
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Wrting Prompts Model", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            loss = outputs.loss.item()
            coherence_loss, relevance_loss, diversity_loss = custom_loss_supervised(model, tokenizer, input_ids, device, max_length, outputs)

            total_loss += loss + coherence_loss + relevance_loss + diversity_loss

            total_coherence_loss += (coherence_coeff * coherence_loss)
            total_relevance_loss += (relevance_coeff * relevance_loss)
            total_diversity_loss += (diversity_coeff * diversity_loss)

            # Calculate Word Error Rate
            predicted_transcripts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs.logits.argmax(dim=-1)]
            ground_truth_transcripts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in labels]
            total_wer += sum([wer(gt, pred) for gt, pred in zip(ground_truth_transcripts, predicted_transcripts)]) / len(predicted_transcripts)

            # Calculate BLEU Score
            predicted_tokens = [tokenizer.decode(ids, skip_special_tokens=True).split() for ids in outputs.logits.argmax(dim=-1)]
            ground_truth_tokens = [[tokenizer.decode(id, skip_special_tokens=True).split()] for id in labels]
            total_bleu += corpus_bleu(ground_truth_tokens, predicted_tokens, smoothing_function=smoothing)

            # Calculate ROUGE Score
            rouge = Rouge()
            scores = rouge.get_scores(predicted_transcripts, ground_truth_transcripts, avg=True)
            total_rouge += scores["rouge-l"]["f"]

            # Calculate METEOR Score
            predicted_tokens = [word_tokenize(transcript.lower()) for transcript in predicted_transcripts]
            ground_truth_tokens = [[word_tokenize(transcript.lower())] for transcript in ground_truth_transcripts]
            total_meteor += sum((meteor_score(reference, predicted) for reference, predicted in zip(ground_truth_tokens, predicted_tokens)))

            num_batches += 1
            if num_batches * (input_ids.shape[0]) % 100 == 0:   
              logger.info("Batch {}/{} - LOSS: {:.4f}, WER: {:.4f}, BLEU: {:.4f}, ROUGE-L: {:.4f}, METEOR: {:.4f}, DIV: {:.4f}, COH: {:.4f}, REL: {:.4f}",
                 num_batches, len(test_loader), total_loss / num_batches,
                 total_wer / num_batches, total_bleu / num_batches,
                 total_rouge / num_batches, total_meteor / num_batches,
                 total_diversity_loss / num_batches,
                 total_coherence_loss / num_batches,
                 total_relevance_loss / num_batches
              )


    total_loss /= num_batches
    total_wer /= num_batches
    total_bleu /= num_batches
    total_rouge /= num_batches
    total_meteor /= num_batches
    total_diversity_loss /= num_batches
    total_coherence_loss /= num_batches
    total_relevance_loss /= num_batches

    file_name = f"final_wp_baseline.pt"
    checkpoint_path = os.path.join(save_dir, file_name)
    torch.save({
        'model_state_dict': model.state_dict() if torch.cuda.device_count() > 1 else model.module.state_dict(),
        'test_loss': total_loss,
        'test_wer': total_wer,
        'test_bleu': total_bleu,
        'test_rouge': total_rouge,
        'test_meteor': total_meteor,
        'test_diversity_loss': total_diversity_loss,
        'test_coherence_loss': total_coherence_loss,
        'test_relevance_loss': total_relevance_loss
    }, checkpoint_path)
    logger.info(f"Baseline Writing Prompts model saved at {checkpoint_path}")

    return total_loss, total_wer, total_bleu, total_rouge, total_meteor, total_diversity_loss, total_coherence_loss, total_relevance_loss


if __name__ == "__main__":
    # Load the hyperparameters from the config file
    CONFIG_FILE_PATH = "experiments/writingprompts/baselines/baseline1.yaml"
    with open(CONFIG_FILE_PATH, "r") as f:
      config = yaml.safe_load(f)

    (train_file, test_file, valid_file, model_name, batch_size,
     max_seq_length, coherence_coeff, relevance_coeff, diversity_coeff, max_length, save_dir) = (
      config["train_file"], config["test_file"], config["valid_file"],
      config["model_name"], config["batch_size"], config["max_seq_length"],
      config["coherence_coeff"], config["relevance_coeff"], config["diversity_coeff"],
      config["max_length"], config["save_dir"])

    # Set up the device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = OPTForCausalLM.from_pretrained(model_name).to(device)
    if torch.cuda.device_count() > 1:
      model = DataParallel(model)

    # Load the writing prompts dataset
    # Set up the dataset and data loaders
    train_loader, val_loader, test_loader = wp_create_data_loaders(train_file, test_file, valid_file, tokenizer, batch_size, max_seq_length)

    # Evaluate on the test set
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)

    logger.info("Evaluating the Writing Prompts baseline model on the test set...")
    total_loss, total_wer, total_bleu, total_rouge, total_meteor, total_diversity_loss, total_coherence_loss, total_relevance_loss = \
      evaluate_wp(model, test_loader, device, coherence_coeff, relevance_coeff, diversity_coeff, max_length, tokenizer, save_dir)

    logger.info("FINAL WRITING PROMPTS BASELINE - LOSS: {:.4f}, WER: {:.4f}, {:.4f}, BLEU: {:.4f}, ROUGE-L: {:.4f}, METEOR: {:.4f}, DIV: {:.4f}, COH: {:.4f}, REL: {:.4f}",
                 total_loss, total_wer, total_bleu, total_rouge, total_meteor, total_diversity_loss, total_coherence_loss, total_relevance_loss)
