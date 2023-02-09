from transformers import GPT2Tokenizer

if __name__ == '__main__':
  tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
  result = tokenizer("Hello world. dl. BOYYYYY . <|endoftext|>")['input_ids']

  def split_tokenized_sentences(input_ids):
    '''This functions splits a list of input IDs on a period, "." into a list of lists
      where each sublist represents a sentence.'''
    sentences = []
    sentence = []
    PERIOD_INPUT_ID = 13

    for input_id in input_ids:
      # Split each sentence into its own list based on the "." input ID
      sentence.append(input_id)
      if input_id == PERIOD_INPUT_ID:
        sentences.append(sentence.copy())
        sentence = []

    # Add on the last sentence remaining in sentence
    sentences.append(sentence)
    return sentences
