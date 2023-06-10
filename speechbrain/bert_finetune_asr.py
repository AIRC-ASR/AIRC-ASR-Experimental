import os
import json
from transformers import BertTokenizer, BertForMaskedLM
from transformers import LineByLineTextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

with open('training_examples.json', encoding='utf-8') as json_file:
  training_examples = json.load(json_file)

vocab = set()
labels = []
for training_example in training_examples:
  sentence, label = training_example['sentence'], training_example['label']
  sentence = sentence.lower().strip()
  label = label.lower().strip()
  labels.append(label)

  # Collect unique words or tokens from the training data
  for word in label.split(";"):
    vocab.add(word.strip())

librispeech_vocab_file = 'librispeech_vocab.txt'
# Write the vocabulary to a text file
with open(librispeech_vocab_file, 'w') as f:
  f.write('\n'.join(sorted(vocab)))


# Load the pre-trained BERT model and tokenizer
model_name = 'bert-large-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)

# Extend the tokenizer vocabulary with the LibriSpeech dataset
tokenizer.add_tokens(librispeech_vocab_file)

# Write the training data to a text file
train_data_file = 'librispeech_train.txt'
with open(train_data_file, 'w') as f:
  for label in labels:
    f.write(label + '\n')

# Load the LibriSpeech training data
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=train_data_file,
    block_size=128  # Adjust the block size as per your requirement
)

# Define the data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15  # Set the masking probability as per your requirement
)

# Set up the training arguments
training_args = TrainingArguments(
    output_dir='./bert_mlm_output',
    overwrite_output_dir=True,
    num_train_epochs=3,  # Adjust the number of training epochs as per your requirement
    per_device_train_batch_size=8,  # Adjust the batch size as per your GPU memory
    save_steps=500,
    save_total_limit=2,
)

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# Start the training
trainer.train()

# Save the fine-tuned model
model.save_pretrained('./fine-tuned_bert_model')
tokenizer.save_pretrained('./fine-tuned_bert_model')