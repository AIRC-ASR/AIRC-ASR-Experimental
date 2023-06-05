import Levenshtein
import fuzzy
from transformers import pipeline

if __name__ == '__main__':
  unmasker = pipeline('fill-mask', model='bert-base-uncased')
  sentence = "she could [MASK] the sound of waves crashing against the shore"
  unmasker_output = unmasker(sentence)
  predicted_words = [output['token_str'] for output in unmasker_output]
  print('Predicted Words:', predicted_words)

  replaced_word = 'tree'
  levenshtein_distances = [Levenshtein.distance(replaced_word, word) for word in predicted_words]
  fuzzy_distances = [Levenshtein.distance(fuzzy.nysiis(replaced_word), fuzzy.nysiis(word)) for word in predicted_words]

  levenshtein_weight = 0.6
  phonetic_weight = 0.4
  weighted_scores = [levenshtein_weight * levenshtein_distance + phonetic_weight * fuzzy_distance for levenshtein_distance, fuzzy_distance in zip(levenshtein_distances, fuzzy_distances)]
  weights_and_words = list(zip(weighted_scores, predicted_words))
  print('Weights and Words:', weights_and_words)

  sorted_words = [word for _, word in sorted(zip(weighted_scores, predicted_words))]
  print('Sorted Words:', sorted_words)

  best_word = sorted_words[0]
  print('Best Word:', best_word)