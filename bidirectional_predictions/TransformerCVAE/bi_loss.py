from concurrent.futures import ThreadPoolExecutor
from functools import partial
from bi_training_core import train_step, Device
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import os
import torch.nn.functional as F
import torch.nn as nn


# NOTE: This is the bidirectional running of the program
def get_sentence_encodings_and_masks(tokens, mask, tokenizer, curr_batch_size, curr_seq_len):
    '''This function takes the y_tokens and y_mask and returns the sentence encodings and masks.'''
    # This decodes each sentence of each story from input IDs to text and splits it into sentences
    tokens_text = tokenizer.decode(tokens[0, :][mask[0, :] == 1].tolist())
    sentences = tokens_text.split('.')

    # Shape: (number of stories, number of sentences, max sentence length)
    # Creates a tensor of zeros with the shape of the number of stories, number of sentences, and max sentence length
    # Each sentence is encoded (text maps to input IDs (integers)) and the mask is created (all ones for each character)
    with torch.no_grad():
        sentence_encodings = torch.zeros((curr_batch_size, len(sentences), curr_seq_len), dtype=torch.long).to(Device.device)
        sentence_masks = torch.zeros((curr_batch_size, len(sentences), curr_seq_len), dtype=torch.long).to(Device.device)
        sentence_directions = torch.zeros((curr_batch_size, len(sentences), 1), dtype=torch.int).to(Device.device)
    assert(len(sentence_encodings) == len(sentence_masks))

    # Since the bidirectional loss of each story is independent of the others, we can use multithreading to speed up the process
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # use a partial and map to run the function on all the sentences for all batches
        partial_get_sentence_encoding_and_mask = partial(get_sentence_encoding_and_mask, tokenizer=tokenizer)
        results = executor.map(partial_get_sentence_encoding_and_mask, sentences)

        # This loop takes the results from the multithreading and puts them into the tensor
        for i, result in enumerate(results):
            sentence_encodings[:, i, :len(result[0])] = torch.tensor(result[0], dtype=torch.long)
            sentence_masks[:, i, :len(result[1])] = result[1]
            sentence_directions[:, i] = result[2]

    return sentence_encodings, sentence_masks, sentence_directions

def get_sentence_encoding_and_mask(sentence, tokenizer):
    '''This function takes a sentence and returns the encoding and mask.'''
    sentence_encoding = tokenizer.encode(sentence + '.')
    sentence_direction = 0
    dir_idx = sentence.find("</")
    if dir_idx != -1:
        special_token = sentence[dir_idx+2: sentence.find("/>")]
        if special_token == 'SFWD':
            sentence_direction = 1
        elif special_token == 'SBKWD':
            sentence_direction = 2
  
    with torch.no_grad():
        sentence_mask = torch.ones(len(sentence_encoding), dtype=torch.long).to(Device.device)
    assert(len(sentence_encoding) == len(sentence_mask))

    return sentence_encoding, sentence_mask, sentence_direction

def bidirectional_loss(loss_type, VAE, optimizer, tokens_mask, tokens, mask, loss_fn,
    beta, model_type, tokenizer, curr_batch_size, curr_seq_len, input_tokens):
        '''This function runs the bidirectional training on different levels.
        loss_types designates the possible loss types: 
            "previous_sentence": The latest sentence needs to predict the previous one and vice versa.
            "previous_sentences" The latest sentence needs to predict the previous ones and vice versa.
            "prompt": The prompt predicts the target story and vice versa.
        All other arguments are the same as train_step()
        '''
        # This gets the sentence encodings and masks for a batch of stories
        sentence_encodings, sentence_masks, sentence_directions = get_sentence_encodings_and_masks(tokens, tokens_mask, tokenizer, curr_batch_size, curr_seq_len)

        assert(len(sentence_encodings) == len(sentence_masks))

        # This runs the bidirectional training on the different levels
        if loss_type == "previous_sentence":
            try:
                return bidirectional_two_sentences(VAE, optimizer, sentence_encodings, sentence_masks, mask, loss_fn, beta, model_type, curr_seq_len, input_tokens, sentence_directions)
            except RuntimeError as e:
                print(e)
                raise("RuntimeError: bidirectional_two_sentences()")
        elif loss_type == "all_previous_sentences":
            try:
                return bidirectional_all_previous_sentences(VAE, optimizer, sentence_encodings, sentence_masks, mask, loss_fn, beta, model_type, curr_batch_size, input_tokens)

            except RuntimeError as e:
                print(e)
                raise("RuntimeError: bidirectional_all_previous_sentences()")


def compute_previous_sentence_loss(VAE, optimizer, sentence_encoding, sentence_mask,
                                   prev_encodings, prev_masks, mask, loss_fn, beta, model_type, input_tokens):
    if prev_encodings is None:
        prev_encodings = torch.zeros((1, input_tokens.shape[1]), dtype=torch.long).to(Device.device)
        prev_masks = torch.zeros((1, input_tokens.shape[1]), dtype=torch.long).to(Device.device)
    else:
        prev_encodings = prev_encodings.clone().detach_()
        prev_masks = prev_masks.clone().detach_()

    y_sentence_encoding_padded = F.pad(sentence_encoding, (0, input_tokens.shape[1] - sentence_encoding.shape[1]))
    y_sentence_mask_padded = F.pad(sentence_mask, (0, input_tokens.shape[1] - sentence_mask.shape[1]))

    output_all_previous_sentences = train_step(VAE, optimizer, y_sentence_mask_padded,
                                                y_sentence_encoding_padded, prev_encodings, prev_masks,
                                                y_sentence_encoding_padded, prev_encodings, mask, loss_fn, beta, model_type)

    loss_all_previous_sentences, ce_loss_all_previous_sentences, kl_loss_sentence_all_previous_sentences = output_all_previous_sentences[-1]

    return loss_all_previous_sentences, ce_loss_all_previous_sentences, kl_loss_sentence_all_previous_sentences


def bidirectional_all_previous_sentences(VAE, optimizer, sentence_encodings, sentence_masks, mask,
                                          loss_fn, beta, model_type, curr_batch_size, input_tokens):
    total_loss_all_previous_sentences = 0
    total_ce_loss_all_previous_sentences = 0
    total_kl_loss_sentence_all_previous_sentences = 0

    for batch_idx in range(curr_batch_size):
        prev_encodings = None
        prev_masks = None
        batch_losses = []

        for idx in range(len(sentence_encodings[batch_idx])):
            y_sentence_encoding = sentence_encodings[batch_idx, idx][0: input_tokens.shape[1]].unsqueeze(0)
            y_sentence_mask = sentence_masks[batch_idx, idx][0: input_tokens.shape[1]].unsqueeze(0)

            loss, ce_loss, kl_loss = compute_previous_sentence_loss(VAE, optimizer, y_sentence_encoding, y_sentence_mask,
                                     prev_encodings, prev_masks, mask, loss_fn, beta, model_type, input_tokens)
            batch_losses.append((loss, ce_loss, kl_loss))

            if idx == 0:
                prev_encodings = torch.zeros((1, input_tokens.shape[1]), dtype=torch.long)
                prev_masks = torch.zeros((1, input_tokens.shape[1]), dtype=torch.long)
            else:
                prev_encodings[:, 0: input_tokens.shape[1] - y_sentence_encoding.shape[1]] = prev_encodings[:, y_sentence_encoding.shape[1]:]
                prev_masks[:, 0: input_tokens.shape[1] - y_sentence_mask.shape[1]] = prev_masks[:, y_sentence_mask.shape[1]:]

            prev_encodings[:, input_tokens.shape[1] - y_sentence_encoding.shape[1]:] = y_sentence_encoding.flatten()
            prev_masks[:, input_tokens.shape[1] - y_sentence_mask.shape[1]:] = y_sentence_mask.flatten()

        for loss, ce_loss, kl_loss in batch_losses:
            total_loss_all_previous_sentences += loss
            total_ce_loss_all_previous_sentences += ce_loss
            total_kl_loss_sentence_all_previous_sentences += kl_loss

    return total_loss_all_previous_sentences, total_ce_loss_all_previous_sentences, total_kl_loss_sentence_all_previous_sentences


def bidirectional_two_sentences(VAE, optimizer, sentence_encodings, sentence_masks, mask, 
                                loss_fn, beta, model_type, curr_seq_len, input_tokens, sentence_directions):
    '''This function finds the loss for the bidirectional training of the previous sentence.'''
    # Compute the number of pairwise comparisons to be made
    num_pairs = sentence_encodings.shape[0]

    def compute_loss(pair_idx):
        total_loss_sentence_b_a, total_loss_sentence_a_b, total_ce_loss_sentence_b_a, \
            total_ce_loss_sentence_a_b, total_kl_loss_sentence_b_a, total_kl_loss_sentence_a_b = 0, 0, 0, 0, 0, 0
        
        if len(sentence_encodings[pair_idx]) < 2:
            return 0, 0, 0, 0, 0, 0
        
        # Get sentence pair encodings and masks
        encoding_a = sentence_encodings[pair_idx][0][:min(curr_seq_len - 1, input_tokens.shape[1])].unsqueeze(0)
        mask_a = sentence_masks[pair_idx][0][:min(curr_seq_len - 1, input_tokens.shape[1])].unsqueeze(0)
        encoding_b = sentence_encodings[pair_idx][1][:min(curr_seq_len - 1, input_tokens.shape[1])].unsqueeze(0)
        mask_b = sentence_masks[pair_idx][1][:min(curr_seq_len - 1, input_tokens.shape[1])].unsqueeze(0)

        if sentence_directions[pair_idx, 1] == 0 or sentence_directions[pair_idx, 1] == 2:
            # Compute loss for Sentence B -> Sentence A
            output_sentence_b_a = train_step(VAE, optimizer, encoding_b, mask_b, encoding_a, mask_a,
                                            encoding_b, encoding_a, mask, loss_fn, beta, model_type)
            total_loss_sentence_b_a, total_ce_loss_sentence_b_a, total_kl_loss_sentence_b_a = output_sentence_b_a[-1]

        if sentence_directions[pair_idx, 1] == 0 or sentence_directions[pair_idx, 1] == 2:
            # Compute loss for Sentence A -> Sentence B
            output_sentence_a_b = train_step(VAE, optimizer, encoding_a, mask_a, encoding_b, mask_b,
                                            encoding_a, encoding_b, mask, loss_fn, beta, model_type)
            total_loss_sentence_a_b, total_ce_loss_sentence_a_b, total_kl_loss_sentence_a_b = output_sentence_a_b[-1]

        # Return the losses for this pair
        return (total_loss_sentence_b_a, total_loss_sentence_a_b, total_ce_loss_sentence_b_a,
                total_ce_loss_sentence_a_b, total_kl_loss_sentence_b_a, total_kl_loss_sentence_a_b)

    # Use multithreading to compute losses for all pairwise sentence combinations
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Start all the threads
        futures = [executor.submit(compute_loss, idx) for idx in range(num_pairs)]
        # Wait for all the threads to finish and get the results
        results = [f.result() for f in as_completed(futures)]

    # Sum the losses and normalize the losses for all the pairs
    losses = [sum(r[i] for r in results) / num_pairs for i in range(len(results[0]))]
    total_loss_sentence_b_a, total_loss_sentence_a_b, total_ce_loss_sentence_b_a, \
        total_ce_loss_sentence_a_b, total_kl_loss_sentence_b_a, total_kl_loss_sentence_a_b = losses

    # Return the total losses
    return (total_loss_sentence_b_a, total_loss_sentence_a_b, total_ce_loss_sentence_b_a,
            total_ce_loss_sentence_a_b, total_kl_loss_sentence_b_a, total_kl_loss_sentence_a_b)

