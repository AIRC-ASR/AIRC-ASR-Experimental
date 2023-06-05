import nltk
import torch
from nltk import ngrams
from nltk.corpus import gutenberg
from pyphones import Pyphones
from transformers import AutoTokenizer, OPTForCausalLM
from transformers import BertForMaskedLM, BertTokenizer

nltk.download('gutenberg')

if __name__ == '__main__':
  py = Pyphones("see")
  candidate_words = py.get_the_homophones_as_list()
  print('Candidate Words:', candidate_words)

  # Step 1: Load BERT model and tokenizer
  model_name = 'bert-large-uncased'
  model = BertForMaskedLM.from_pretrained(model_name)
  tokenizer = BertTokenizer.from_pretrained(model_name)

  # Step 2: Tokenize sentence with [MASK] token
  sentence = "she could [MASK] the sound of waves crashing against the shore"
  tokenized_sentence = tokenizer.tokenize(sentence)
  masked_index = tokenized_sentence.index('[MASK]')

  # Step 3: Convert tokenized sentence to input features
  indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_sentence)
  segments_ids = [0] * len(indexed_tokens)
  tokens_tensor = torch.tensor([indexed_tokens])
  segments_tensor = torch.tensor([segments_ids])

  # Step 4: Pass input features through BERT model
  model.eval()
  with torch.no_grad():
      outputs = model(tokens_tensor, segments_tensor)
      predictions = outputs[0][0, masked_index]

  # Step 5: Retrieve predicted probabilities for each candidate word
  predicted_probabilities = {word: predictions[tokenizer.convert_tokens_to_ids([word])[0]].item() for word in candidate_words}

  # Step 6: Select word with highest probability
  predicted_word = max(predicted_probabilities, key=predicted_probabilities.get)

  print("Predicted missing word:", predicted_word)
