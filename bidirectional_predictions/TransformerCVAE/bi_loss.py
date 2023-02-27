from concurrent.futures import ThreadPoolExecutor
from functools import partial
from bi_training_core import train_step, Device
import torch
import os
import torch.nn.functional as F


# NOTE: This is the bidirectional running of the program
def get_sentence_encodings_and_masks(y_tokens, y_mask, tokenizer, curr_batch_size, curr_seq_len):
    '''This function takes the y_tokens and y_mask and returns the sentence encodings and masks.'''
    # This decodes each sentence of each story from input IDs to text and splits it into sentences
    y_tokens_text = tokenizer.decode(y_tokens[0, :][y_mask[0, :] == 1].tolist())
    y_sentences = y_tokens_text.split('.')

    # Shape: (number of stories, number of sentences, max sentence length)
    # Creates a tensor of zeros with the shape of the number of stories, number of sentences, and max sentence length
    # Each sentence is encoded (text maps to input IDs (integers)) and the mask is created (all ones for each character)
    with torch.no_grad():
        y_sentence_encodings = torch.zeros((curr_batch_size, len(y_sentences), curr_seq_len - 1), dtype=torch.long).to(Device.device)
        y_sentence_masks = torch.zeros((curr_batch_size, len(y_sentences), curr_seq_len - 1), dtype=torch.long).to(Device.device)
    assert(len(y_sentence_encodings) == len(y_sentence_masks))

    # Since the bidirectional loss of each story is independent of the others, we can use multithreading to speed up the process
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # use a partial and map to run the function on all the sentences for all batches
        partial_get_sentence_encoding_and_mask = partial(get_sentence_encoding_and_mask, tokenizer=tokenizer)
        results = executor.map(partial_get_sentence_encoding_and_mask, y_sentences)

        # This loop takes the results from the multithreading and puts them into the tensor
        for i, result in enumerate(results):
            y_sentence_encodings[:, i, :len(result[0])] = torch.tensor(result[0], dtype=torch.long)
            y_sentence_masks[:, i, :len(result[1])] = result[1]

    return y_sentence_encodings, y_sentence_masks

def get_sentence_encoding_and_mask(sentence, tokenizer):
    '''This function takes a sentence and returns the encoding and mask.'''
    sentence_encoding = tokenizer.encode(sentence + '.')
    with torch.no_grad():
        sentence_mask = torch.ones(len(sentence_encoding), dtype=torch.long).to(Device.device)
    assert(len(sentence_encoding) == len(sentence_mask))

    return sentence_encoding, sentence_mask

def bidirectional_loss(loss_type, VAE, optimizer, y_mask, y_tokens, mask, loss_fn,
    beta, model_type, tokenizer, curr_batch_size, curr_seq_len, input_tokens):
        '''This function runs the bidirectional training on different levels.
        loss_types designates the possible loss types: 
            "previous_sentence": The latest sentence needs to predict the previous one and vice versa.
            "previous_sentences" The latest sentence needs to predict the previous ones and vice versa.
            "prompt": The prompt predicts the target story and vice versa.
        All other arguments are the same as train_step()
        '''
        # This gets the sentence encodings and masks for a batch of stories
        y_sentence_encodings, y_sentence_masks = get_sentence_encodings_and_masks(y_tokens, y_mask, tokenizer, curr_batch_size, curr_seq_len)
        assert(len(y_sentence_encodings) == len(y_sentence_masks))

        # This runs the bidirectional training on the different levels
        if loss_type == "previous_sentence":
            try:
                return bidirectional_two_sentences(VAE, optimizer, y_sentence_encodings, y_sentence_masks, mask, loss_fn, beta, model_type, curr_batch_size, curr_seq_len, input_tokens)
            except RuntimeError as e:
                print(e)
                print("RuntimeError: bidirectional_two_sentences()")
                return 0, 0, 0, 0, 0, 0
        elif loss_type == "all_previous_sentences":
            try:
               result = bidirectional_all_previous_sentences(VAE, optimizer, y_sentence_encodings, y_sentence_masks, mask, loss_fn, beta, model_type, curr_batch_size, curr_seq_len, input_tokens)
               print('result', result)
               return result
            except RuntimeError as e:
                print(e)
                print("RuntimeError: bidirectional_all_previous_sentences()")
                return 0, 0, 0, 0, 0, 0


def bidirectional_all_previous_sentences(VAE, optimizer, y_sentence_encodings, y_sentence_masks, mask,
        loss_fn, beta, model_type, curr_batch_size, curr_seq_len, input_tokens):
    '''This function finds the loss for the bidirectional training on all previous sentences.
    So if the sentence is the 3rd sentence in the story, it will predict the first two sentences and vice versa.'''
    total_loss_all_previous_sentences = 0
    total_ce_loss_all_previous_sentences = 0
    total_kl_loss_sentence_all_previous_sentences = 0

    for batch_idx in range(curr_batch_size):
        with torch.no_grad():
            prev_encodings = torch.zeros((1, curr_seq_len), dtype=torch.long).to(Device.device)
            prev_masks = torch.zeros((1, curr_seq_len), dtype=torch.long).to(Device.device)
        for idx in range(len(y_sentence_encodings[batch_idx])): # [1, 38, 1024]
            y_sentence_encoding = y_sentence_encodings[batch_idx, idx][0: input_tokens.shape[1]].unsqueeze(0) # [1, 1023]
            y_sentence_mask = y_sentence_masks[batch_idx, idx][0: input_tokens.shape[1]].unsqueeze(0)
            assert(y_sentence_encoding.shape[1] == y_sentence_mask.shape[1])

            # This is the case where the sentence is the first sentence in the story
            # It does not have any previous sentences before it to predict
            if idx > 0:
                y_sentence_encoding_padded = F.pad(y_sentence_encoding, (0, curr_seq_len - y_sentence_encoding.shape[1]))
                y_sentence_mask_padded = F.pad(y_sentence_mask, (0, curr_seq_len - y_sentence_mask.shape[1]))
                assert(y_sentence_encoding_padded.shape[1] == y_sentence_mask_padded.shape[1])

                # SENTENCE LEVEL LOSS, Sentence B -> All Previous Sentences
                print('SHAPES', y_sentence_encoding_padded.shape, y_sentence_mask_padded.shape, prev_encodings.shape, prev_masks.shape)
                output_all_previous_sentences = train_step(VAE, optimizer, y_sentence_mask_padded,
                    y_sentence_encoding_padded, prev_encodings, prev_masks,
                    y_sentence_encoding_padded, prev_encodings, mask, loss_fn, beta, model_type
                )
                print('output_all_previous_sentences', output_all_previous_sentences)

                loss_all_previous_sentences, ce_loss_all_previous_sentences, kl_loss_sentence_all_previous_sentences = output_all_previous_sentences[-1]
                total_loss_all_previous_sentences += loss_all_previous_sentences
                total_ce_loss_all_previous_sentences += ce_loss_all_previous_sentences
                total_kl_loss_sentence_all_previous_sentences += kl_loss_sentence_all_previous_sentences

            prev_encodings[0, 0: input_tokens.shape[1]] = y_sentence_encoding.flatten()
            prev_masks[0, 0: input_tokens.shape[1]] = y_sentence_mask.flatten()
            assert(prev_encodings.shape[1] == prev_masks.shape[1] == curr_seq_len)
    
    return (
        total_loss_all_previous_sentences,
        total_ce_loss_all_previous_sentences,
        total_kl_loss_sentence_all_previous_sentences,
    )


def bidirectional_two_sentences(VAE, optimizer, y_sentence_encodings, y_sentence_masks, mask, 
        loss_fn, beta, model_type, curr_batch_size, curr_seq_len, input_tokens):
    '''This function finds the loss for the bidirectional training of the previous sentence.'''
    total_loss_sentence_b_a = 0
    total_loss_sentence_a_b = 0
    total_ce_loss_sentence_b_a = 0
    total_ce_loss_sentence_a_b = 0
    total_kl_loss_sentence_b_a = 0
    total_kl_loss_sentence_a_b = 0

    for batch_idx in range(curr_batch_size):
        for idx in range(len(y_sentence_encodings[batch_idx]) - 1):
            # Get the sentence encoding and mask for the current sentence and the next sentence
            y_sentence_encoding_a = y_sentence_encodings[batch_idx, idx][0: input_tokens.shape[1]].unsqueeze(0)       
            y_sentence_mask_a = y_sentence_masks[batch_idx, idx][0: input_tokens.shape[1]].unsqueeze(0)
            assert(y_sentence_encoding_a.shape[1] == y_sentence_mask_a.shape[1])

            y_sentence_encoding_b = y_sentence_encodings[batch_idx, idx + 1][0: input_tokens.shape[1]].unsqueeze(0)
            y_sentence_mask_b = y_sentence_masks[batch_idx, idx + 1][0: input_tokens.shape[1]].unsqueeze(0)
            assert(y_sentence_encoding_b.shape[1] == y_sentence_mask_b.shape[1])

            # SENTENCE LEVEL LOSS, Sentence B -> Sentence A
            print('BETTER SHAPES', y_sentence_encoding_b.shape, y_sentence_mask_b.shape, y_sentence_encoding_a.shape, y_sentence_mask_a.shape)
            output_sentence_b_a = train_step(VAE, optimizer, y_sentence_encoding_b, y_sentence_mask_b, y_sentence_encoding_a, y_sentence_mask_a,
                    y_sentence_encoding_b, y_sentence_encoding_a, mask, loss_fn, beta, model_type)
            loss_sentence_b_a, ce_loss_sentence_b_a, kl_loss_sentence_b_a = output_sentence_b_a[-1]

            total_loss_sentence_b_a += loss_sentence_b_a
            total_ce_loss_sentence_b_a += ce_loss_sentence_b_a
            total_kl_loss_sentence_b_a += kl_loss_sentence_b_a

            # # SENTENCE LEVEL LOSS, Sentence A -> Sentence B
            output_sentence_a_b = train_step(VAE, optimizer, y_sentence_encoding_a, y_sentence_mask_a, y_sentence_encoding_b, y_sentence_mask_b,
                    y_sentence_encoding_a, y_sentence_encoding_b, mask, loss_fn, beta, model_type)
            loss_sentence_a_b, ce_loss_sentence_a_b, kl_loss_sentence_a_b = output_sentence_a_b[-1]

            # Sum up the losses
            total_loss_sentence_a_b += loss_sentence_a_b
            total_ce_loss_sentence_a_b += ce_loss_sentence_a_b
            total_kl_loss_sentence_a_b += kl_loss_sentence_a_b

    return (
        total_loss_sentence_b_a,
        total_loss_sentence_a_b,
        total_ce_loss_sentence_b_a,
        total_ce_loss_sentence_a_b,
        total_kl_loss_sentence_b_a,
        total_kl_loss_sentence_a_b,
    )
