import re
import random
from math import isnan

import torch
import torch.nn.functional as F
import torchaudio.datasets as dsets
from datasets import load_dataset
from torch.utils.data import DataLoader, random_split


class WritingPromptsDataset(torch.utils.data.Dataset):
    def __init__(self, source_file_path, target_file_path, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        with open(source_file_path, 'r') as f:
            self.source_data = f.readlines()
        with open(target_file_path, 'r') as f:
            self.target_data = f.readlines()

    def __getitem__(self, idx):
        source_text = self.source_data[idx].strip()
        target_text = self.target_data[idx].strip()
        source_encoding = self.tokenizer(source_text, max_length=self.max_seq_length, padding='max_length', truncation=True, return_tensors='pt')
        target_encoding = self.tokenizer(target_text, max_length=self.max_seq_length, padding='max_length', truncation=True, return_tensors='pt')
        return {'input_ids': source_encoding['input_ids'].squeeze(0),
                'attention_mask': source_encoding['attention_mask'].squeeze(0),
                'labels': target_encoding['input_ids'].squeeze(0)}

    def __len__(self):
        return len(self.source_data)


def wp_create_data_loaders(train_file, test_file, valid_file, tokenizer, batch_size, max_seq_length):
    """
    Creates PyTorch Writing Prompts DataLoader objects for the training, testing, and validation datasets.

    Args:
        train_file (str): Path to the training data file.
        test_file (str): Path to the testing data file.
        valid_file (str): Path to the validation data file.
        tokenizer (PreTrainedTokenizer): The tokenizer to use for tokenizing the input text.
        batch_size (int): The batch size to use for each DataLoader.
        max_seq_length (int): The maximum sequence length to use for each input sequence.

    Returns:
        A tuple of three DataLoader objects for the training, testing, and validation datasets.
    """
    train_dataset = WritingPromptsDataset(source_file_path=train_file, target_file_path=train_file, tokenizer=tokenizer, max_seq_length=max_seq_length)
    test_dataset = WritingPromptsDataset(source_file_path=test_file, target_file_path=test_file, tokenizer=tokenizer, max_seq_length=max_seq_length)
    val_dataset = WritingPromptsDataset(source_file_path=valid_file, target_file_path=valid_file, tokenizer=tokenizer, max_seq_length=max_seq_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader



def remove_special_characters(batch):
    chars_to_remove_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\']'
    if 'sentence' in batch:
        batch["sentence"] = re.sub(chars_to_remove_regex, '', batch["sentence"]).lower()
    elif 'utterance' in batch:
        batch["utterance"] = re.sub(chars_to_remove_regex, '', batch["utterance"]).lower()

    return batch


class CommonVoiceDataset(torch.utils.data.Dataset):
    # NOTE: This is unsupervised learning, so we don't need to pass in a target file path
    def __init__(self, tokenizer, max_seq_length, split):
        if split not in ['train', 'test', 'validation']:
            raise ValueError("Invalid split value")

        self.source_data = load_dataset("common_voice", "en", split=split)

        # This removes punctuation and special characters from the text
        self.source_data = self.source_data.map(remove_special_characters, desc="Preprocessing dataset")

        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __getitem__(self, idx):
        source_text = self.source_data[idx]['sentence'].strip()
        source_encoding = self.tokenizer(source_text, max_length=self.max_seq_length, padding='max_length', truncation=True, return_tensors='pt')
        return {'input_ids': source_encoding['input_ids'].squeeze(0),
                'attention_mask': source_encoding['attention_mask'].squeeze(0)}

    def __len__(self):
        return len(self.source_data)

def cv_create_data_loaders(tokenizer, batch_size, max_seq_length):
    """
    Creates PyTorch Common Voice DataLoader objects for the training, testing, and validation datasets.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer to use for tokenizing the input text.
        batch_size (int): The batch size to use for each DataLoader.
        max_seq_length (int): The maximum sequence length to use for each input sequence.

    Returns:
        A tuple of three DataLoader objects for the training, testing, and validation datasets.
    """
    train_dataset = CommonVoiceDataset(tokenizer, max_seq_length, 'train')
    test_dataset = CommonVoiceDataset(tokenizer, max_seq_length, 'test')
    val_dataset = CommonVoiceDataset(tokenizer, max_seq_length, 'validation')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


class LibriSpeechDataset(torch.utils.data.Dataset):
    # NOTE: This is unsupervised learning, so we don't need to pass in a target file path
    def __init__(self, tokenizer, max_seq_length, url):
        if url not in [
            "dev-clean",
            "dev-other",
            "test-clean",
            "test-other",
            "train-clean-100",
            "train-clean-360",
            "train-other-500",
        ]:
            raise ValueError("Invalid url value")

        self.source_data = dsets.LIBRISPEECH(root='./librispeech', url=url, download=True)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __getitem__(self, idx):
        utterance1 = self.source_data[idx][2].strip()
        utterance2 = ''
        label = 1
        if random.random() < 0.5:
            random_index = random.randint(0, len(self.source_data) - 1)
            utterance2 = self.source_data[random_index][2].strip()
            label = 0
        else:
            if idx + 1 < len(self.source_data):
                utterance2 = self.source_data[idx + 1][2].strip()

        utterance1_encoding = self.tokenizer(utterance1, max_length=self.max_seq_length, padding='max_length', truncation=True, return_tensors='pt')
        utterance2_encoding = self.tokenizer(utterance2, max_length=self.max_seq_length, padding='max_length', truncation=True, return_tensors='pt')

        return {'utterance1_input_ids': utterance1_encoding['input_ids'].squeeze(0),
                'utterance1_attention_mask': utterance1_encoding['attention_mask'].squeeze(0),
                'utterance2_input_ids': utterance2_encoding['input_ids'].squeeze(0),
                'utterance2_attention_mask': utterance2_encoding['attention_mask'].squeeze(0),
                'label': torch.tensor(label).squeeze(0)}

    def __len__(self):
        return len(self.source_data) // 2


def ls_create_data_loaders(tokenizer, batch_size, max_seq_length):
    """
    Creates PyTorch LibriSpeech DataLoader objects for the training, testing, and validation datasets.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer to use for tokenizing the input text.
        batch_size (int): The batch size to use for each DataLoader.
        max_seq_length (int): The maximum sequence length to use for each input sequence.

    Returns:
        A tuple of three DataLoader objects for the training, testing, and validation datasets.
    """
    train_dataset = LibriSpeechDataset(tokenizer, max_seq_length, 'train-clean-100')
    test_dataset = LibriSpeechDataset(tokenizer, max_seq_length, 'test-clean')

    # Create the validation dataset by splitting the training dataset
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def generate_continuation(model, tokenizer, prompt, device, max_length):
    # Tokenize the prompt and convert to tensor
    # Tokenize the text and convert to tensor
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    # Generate continuation
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            do_sample=True,
            temperature=1.0,
            top_p=0.9,
            top_k=0
        )
    # Convert output to string and remove prompt
    output_str = tokenizer.decode(output[0], skip_special_tokens=True)
    output_str = output_str.replace(prompt, "", 1).strip()
    return output_str


def generate_continuations(model, tokenizer, prompts, device, max_length):
    continuations = []
    for prompt in prompts:
        continuation = generate_continuation(model, tokenizer, prompt, device, max_length)
        continuations.append(continuation)
    return continuations

def coherence_score(model, tokenizer, device, continuation):
    '''This function calculates the coherence score of a generated continuation of a
    given text prompt by the fine-tuned OPT model. The coherence score is a measure
    of how coherent the generated text is with respect to the original text prompt.

    First, the function tokenizes the given text prompt and converts it to a PyTorch tensor
    to be used as input for the model. Then, the function uses the generate method of the
    model to generate a continuation of the input prompt. The generated continuation is a
    tensor of token ids representing the sequence of words generated by the model.

    Next, the function converts the generated continuation tensor to a string and removes
    the input prompt from it, so that only the generated text remains. The function then
    calculates the coherence score of the generated text by passing it as input to the
    fine-tuned OPT model and calculating the loss between the generated text and the original text.
    '''


    # Calculate coherence score
    continuation_input_ids = tokenizer(continuation, return_tensors="pt").input_ids.to(device)
    coherence_score = model.forward(input_ids=continuation_input_ids, labels=continuation_input_ids).loss.item()
    if isnan(coherence_score):
        coherence_score = 0.0

    return coherence_score


def custom_loss_supervised(model, tokenizer, input_ids, device, max_length, outputs):
    """
    Computes the coherence loss between the input prompt and generated text.

    Args:
        model (PreTrainedModel): The model used for generating text.
        tokenizer (PreTrainedTokenizer): The tokenizer used for tokenizing text.
        input_ids (torch.Tensor): The input IDs for the model, as a tensor of shape (batch_size, sequence_length).
        outputs (SequenceClassifierOutput): The model output, containing the predicted logits and the ground-truth labels.
        device (torch.device): The device on which to perform the computations.

    Returns:
        coherence_loss (torch.Tensor): The coherence loss between the input prompt and generated text, as a scalar tensor.
    """
    coherence_loss = 0
    for i in range(len(input_ids)):
        # Get the prompt
        prompt = tokenizer.decode(input_ids[i], skip_special_tokens=True)

        # Generate the continuation
        continuation = generate_continuation(model, tokenizer, prompt, device, max_length)
        next_token = tokenizer.decode(outputs.logits[i].argmax(dim=-1), skip_special_tokens=True)
        continuation = continuation + " " + next_token

        coherence_loss += coherence_score(model, tokenizer, device, continuation)

    if len(input_ids):
        coherence_loss /= len(input_ids)

    relevance_loss = relevance_loss_function_supervised(model, input_ids, outputs, device).item()
    diversity_loss = diversity_loss_function_supervised(tokenizer, device, outputs).item()

    return coherence_loss, relevance_loss, diversity_loss


def custom_loss_unsupervised(model, tokenizer, input_ids, device, max_length):
    """
    Computes the coherence loss between the input prompt and generated text.

    Args:
        model (PreTrainedModel): The model used for generating text.
        tokenizer (PreTrainedTokenizer): The tokenizer used for tokenizing text.
        input_ids (torch.Tensor): The input IDs for the model, as a tensor of shape (batch_size, sequence_length).
        outputs (SequenceClassifierOutput): The model output, containing the predicted logits and the ground-truth labels.
        device (torch.device): The device on which to perform the computations.

    Returns:
        coherence_loss (torch.Tensor): The coherence loss between the input prompt and generated text, as a scalar tensor.
    """
    coherence_loss = 0
    relevance_scores = torch.tensor([], device=device)
    diversity_loss = 0
    for i in range(len(input_ids)):
        # Get the prompt
        prompt = tokenizer.decode(input_ids[i], skip_special_tokens=True)

        # Generate the continuation
        continuation = generate_continuation(model, tokenizer, prompt, device, max_length)
        continuation_input_ids = tokenizer(continuation, return_tensors="pt").input_ids.to(device)

        coherence_loss += coherence_score(model, tokenizer, device, continuation)
        cos_sim = relevance_score_function_unsupervised(model, input_ids[i], continuation_input_ids.squeeze(0), device)
        relevance_scores = torch.cat((relevance_scores, cos_sim.unsqueeze(0)))
    
        diversity_loss += diversity_score_function_unsupervised(tokenizer, device, continuation).item()

    if len(input_ids):
        coherence_loss /= len(input_ids)

    relevance_loss = (1.0 - relevance_scores.mean()).item()

    return coherence_loss, relevance_loss, diversity_loss


def relevance_score_function_supervised(model, input_ids, outputs, device):
    """
    Computes the relevance score between the input prompt and generated text.

    Args:
        model (PreTrainedModel): The model used for generating text.
        input_ids (torch.Tensor): The input IDs for the model, as a tensor of shape (batch_size, sequence_length).
        outputs (SequenceClassifierOutput): The model output, containing the predicted logits and the ground-truth labels.
        device (torch.device): The device on which to perform the computations.

    Returns:
        relevance_scores (torch.Tensor): The relevance scores between the input prompt and generated text, as a tensor of shape (batch_size,).
    """
    # Convert logits to probabilities
    probs = torch.softmax(outputs.logits, dim=-1)
    # Get the index of the predicted class
    pred_class_idx = torch.argmax(probs, dim=-1)
    # Compute the relevance score using cosine similarity
    relevance_scores = torch.tensor([], device=device)
    for i in range(len(input_ids)):
        input_vec = model.get_input_embeddings()(input_ids[i].unsqueeze(0).to(device))[0]
        output_vec = model.get_input_embeddings()(pred_class_idx[i].unsqueeze(0).to(device))[0]
        cos_sim = F.cosine_similarity(input_vec, output_vec, dim=0)
        relevance_scores = torch.cat((relevance_scores, cos_sim.unsqueeze(0)))

    return relevance_scores


def relevance_loss_function_supervised(model, input_ids, outputs, device):
    """
    Computes the relevance loss between the input prompt and generated text.

    Args:
        model (PreTrainedModel): The model used for generating text.
        tokenizer (PreTrainedTokenizer): The tokenizer used for tokenizing text.
        input_ids (torch.Tensor): The input IDs for the model, as a tensor of shape (batch_size, sequence_length).
        outputs (SequenceClassifierOutput): The model output, containing the predicted logits and the ground-truth labels.
        device (torch.device): The device on which to perform the computations.

    Returns:
        relevance_loss (torch.Tensor): The relevance loss between the input prompt and generated text, as a scalar tensor.
    """
    relevance_scores = relevance_score_function_supervised(model, input_ids, outputs, device)
    relevance_loss = 1.0 - relevance_scores.mean()
    return relevance_loss


def relevance_score_function_unsupervised(model, input_ids, continuation_input_ids, device):
    """
    Computes the relevance score between the input prompt and generated text.

    Args:
        model (PreTrainedModel): The model used for generating text.
        input_ids (torch.Tensor): The input IDs for the model, as a tensor of shape (batch_size, sequence_length).
        outputs (SequenceClassifierOutput): The model output, containing the predicted logits and the ground-truth labels.
        device (torch.device): The device on which to perform the computations.

    Returns:
        relevance_scores (torch.Tensor): The relevance scores between the input prompt and generated text, as a tensor of shape (batch_size,).
    """
    # Compute the relevance score using cosine similarity
    input_vec = model.get_input_embeddings()(input_ids.unsqueeze(0).to(device))[0]
    output_vec = model.get_input_embeddings()(continuation_input_ids.unsqueeze(0).to(device))[0]

    # Pad or truncate the vectors to be the same size
    if input_vec.size(0) > output_vec.size(0):
        output_vec = F.pad(output_vec, (0, 0, 0, input_vec.size(0) - output_vec.size(0)))
    elif output_vec.size(0) > input_vec.size(0):
        output_vec = output_vec[:input_vec.size(0)]

    cos_sim = F.cosine_similarity(input_vec, output_vec, dim=0)

    return cos_sim


def relevance_loss_function_unsupervised(model, input_ids, outputs, device):
    """
    Computes the relevance loss between the input prompt and generated text.

    Args:
        model (PreTrainedModel): The model used for generating text.
        tokenizer (PreTrainedTokenizer): The tokenizer used for tokenizing text.
        input_ids (torch.Tensor): The input IDs for the model, as a tensor of shape (batch_size, sequence_length).
        outputs (SequenceClassifierOutput): The model output, containing the predicted logits and the ground-truth labels.
        device (torch.device): The device on which to perform the computations.

    Returns:
        relevance_loss (torch.Tensor): The relevance loss between the input prompt and generated text, as a scalar tensor.
    """
    relevance_scores = relevance_score_function_unsupervised(model, input_ids, outputs, device)
    relevance_loss = 1.0 - relevance_scores.mean()
    return relevance_loss



def diversity_loss_function_supervised(tokenizer, device, outputs):
    """
    Computes the diversity loss between the generated text in the batch.

    Args:
        model (PreTrainedModel): The model used for generating text.
        tokenizer (PreTrainedTokenizer): The tokenizer used for tokenizing text.
        input_ids (torch.Tensor): The input IDs for the model, as a tensor of shape (batch_size, sequence_length).
        outputs (SequenceClassifierOutput): The model output, containing the predicted logits and the ground-truth labels.
        device (torch.device): The device on which to perform the computations.

    Returns:
        diversity_loss (torch.Tensor): The diversity loss between the generated text in the batch, as a scalar tensor.
    """
    # Decode the generated text
    generated_text = tokenizer.batch_decode(outputs.logits.argmax(dim=-1), skip_special_tokens=True)
    # Split the text into individual tokens
    tokenized_text = [tokenizer.tokenize(text) for text in generated_text]
    # Compute the unique token count for each text
    unique_token_counts = [len(set(tokens)) for tokens in tokenized_text]
    # Compute the diversity loss
    diversity_loss = torch.mean(torch.tensor(unique_token_counts, device=device, dtype=torch.float32))
    if torch.isnan(diversity_loss):
        diversity_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

    return diversity_loss

def diversity_score_function_unsupervised(tokenizer, device, generated_text):
    """
    Computes the diversity loss between the generated text in the batch.

    Args:
        model (PreTrainedModel): The model used for generating text.
        tokenizer (PreTrainedTokenizer): The tokenizer used for tokenizing text.
        input_ids (torch.Tensor): The input IDs for the model, as a tensor of shape (batch_size, sequence_length).
        outputs (SequenceClassifierOutput): The model output, containing the predicted logits and the ground-truth labels.
        device (torch.device): The device on which to perform the computations.

    Returns:
        diversity_loss (torch.Tensor): The diversity loss between the generated text in the batch, as a scalar tensor.
    """
    # Split the text into individual tokens
    tokenized_text = [tokenizer.tokenize(text) for text in generated_text]
    # Compute the unique token count for each text
    unique_token_counts = [len(set(tokens)) for tokens in tokenized_text]
    # Compute the diversity loss
    diversity_loss = torch.mean(torch.tensor(unique_token_counts, device=device, dtype=torch.float32))
    if torch.isnan(diversity_loss):
        diversity_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

    return diversity_loss
