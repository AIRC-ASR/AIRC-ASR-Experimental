import jiwer
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained language model
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Define training data
reference_sentences = [
    "CUT AS MANY NICE EVEN SLICES AS MAY BE REQUIRED RATHER MORE THAN ONE QUARTER INCH IN THICKNESS AND TOAST THEM BEFORE A VERY BRIGHT FIRE WITHOUT ALLOWING THE BREAD TO BLACKEN WHICH SPOILS THE APPEARANCE AND FLAVOUR OF ALL TOAST",
    "BUT THE RASHNESS OF THESE CONCESSIONS HAS ENCOURAGED A MILDER SENTIMENT OF THOSE OF THE DOCETES WHO TAUGHT NOT THAT CHRIST WAS A PHANTOM BUT THAT HE WAS CLOTHED WITH AN IMPASSIBLE AND INCORRUPTIBLE BODY",
    # Add more reference sentences here
]

hypothesis_sentences = [
    "CUT AS MANY NICE EVEN SLICES AS MAY BE REQUIRED RATHER MORE THAN ONE QUARTER INCH IN THICKNESS AND TOAST THEM BEFORE A VERY BRIGHT FIRE WITHOUT ALLOWING THE BREAD TO BLACKEN WHICH SPOILS THE APPEARANCE AND FLAVOR OF ALL TOAST",
    "BUT THE RASHNESS OF THESE CONCESSIONS HAS ENCOURAGED A MILDER SENTIMENT OF THOSE OF THE DOCITS WHO TAUGHT NOT THAT CHRIST WAS A PHANTOM BUT THAT HE WAS CLOTHED WITH AN IMPASSIBLE AND INCORRUPTIBLE BODY",
    # Add more hypothesis sentences here
]

# Tokenize and truncate/pad the training data
max_length = 128  # Maximum sequence length

tokenized_data = []
for reference, hypothesis in zip(reference_sentences, hypothesis_sentences):
    reference_tokens = tokenizer.encode(reference, add_special_tokens=True)
    hypothesis_tokens = tokenizer.encode(hypothesis, add_special_tokens=True)

    # Truncate or pad sequences to a fixed length
    reference_tokens = reference_tokens[:max_length]
    hypothesis_tokens = hypothesis_tokens[:max_length]

    tokenized_data.append((reference_tokens, hypothesis_tokens))

# Prepare input tensors for training
input_ids = [torch.tensor(reference_tokens).unsqueeze(0) for reference_tokens, _ in tokenized_data]
labels = [torch.tensor(hypothesis_tokens).unsqueeze(0) for _, hypothesis_tokens in tokenized_data]

# Pad sequences to the same length
input_ids = pad_sequence(input_ids, batch_first=True)
labels = pad_sequence(labels, batch_first=True)

# Set training parameters
epochs = 5
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
model.train()
for epoch in range(epochs):
    total_loss = 0
    for inputs, labels in zip(input_ids, labels):
        outputs = model(inputs, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_loss = total_loss / len(input_ids)
    print(f"Epoch {epoch + 1} - Average Loss: {average_loss:.4f}")

# Evaluation example
model.eval()
with torch.no_grad():
    reference = "CUT AS MANY NICE EVEN SLICES AS MAY BE REQUIRED RATHER MORE THAN ONE QUARTER INCH IN THICKNESS AND TOAST THEM BEFORE A VERY BRIGHT FIRE WITHOUT ALLOWING THE BREAD TO BLACKEN WHICH SPOILS THE APPEARANCE AND FLAVOUR OF ALL TOAST"
    hypothesis = "CUT AS MANY NICE EVEN SLICES AS MAY BE REQUIRED RATHER MORE THAN ONE QUARTER INCH IN THICKNESS AND TOAST THEM BEFORE A VERY BRIGHT FIRE WITHOUT ALLOWING THE BREAD TO BLACKEN WHICH SPOILS THE APPEARANCE AND FLAVOR OF ALL TOAST"

    tokenized_reference = tokenizer.encode(reference, add_special_tokens=True)
    tokenized_hypothesis = tokenizer.encode(hypothesis, add_special_tokens=True)

    # Truncate or pad sequences to a fixed length
    tokenized_reference = tokenized_reference[:max_length]
    tokenized_hypothesis = tokenized_hypothesis[:max_length]

    input_tensor = torch.tensor(tokenized_reference).unsqueeze(0)
    label_tensor = torch.tensor(tokenized_hypothesis).unsqueeze(0)

    # Calculate Word Error Rate (WER)
    wer = jiwer.wer(input_tensor, label_tensor)
    print(f"WER: {wer:.2%}")
