from concurrent.futures import ThreadPoolExecutor
from functools import partial
from bi_training_core import train_step, Device
import torch
import os
import torch.nn.functional as F


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
            return find_loss_bidirectional_two_sentences(sentence_encodings, sentence_masks, VAE, optimizer,
                mask, loss_fn, beta, model_type, curr_batch_size, curr_seq_len, input_tokens, sentence_directions)
        elif loss_type == "all_previous_sentences":
            return find_loss_bidirectional_all_previous_sentences(sentence_encodings, sentence_masks, VAE, optimizer,
                mask, loss_fn, beta, model_type, curr_batch_size, curr_seq_len, input_tokens)


def find_loss_bidirectional_two_sentences(sentence_encodings, sentence_masks,
        VAE, optimizer, mask, loss_fn, beta, model_type, curr_batch_size, curr_seq_len, input_tokens, sentence_directions):
    '''This function finds the loss for the bidirectional training of the previous sentence.
    So if the latest sentence is "I am a cat.", then the previous sentence is "I am a dog." and vice versa.'''
    total_loss_sentence_b_a = 0
    total_loss_sentence_a_b = 0
    total_ce_loss_sentence_b_a = 0
    total_ce_loss_sentence_a_b = 0
    total_kl_loss_sentence_b_a = 0
    total_kl_loss_sentence_a_b = 0

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # use a partial and map to run the function on all the sentences for all batches
        partial_bidirectional_two_sentences = partial(bidirectional_two_sentences, VAE=VAE, optimizer=optimizer,
            sentence_encodings=sentence_encodings, sentence_masks=sentence_masks, mask=mask, loss_fn=loss_fn, beta=beta, 
            model_type=model_type, curr_batch_size=curr_batch_size, curr_seq_len=curr_seq_len, input_tokens=input_tokens, sentence_directions=sentence_directions
        )
        try:
            results = executor.map(partial_bidirectional_two_sentences, range(len(sentence_encodings)))
        except RuntimeError as e:
            print(e)
            print("RuntimeError: find_loss_bidirectional_two_sentences()")
            return (
                total_loss_sentence_b_a,
                total_loss_sentence_a_b,
                total_ce_loss_sentence_b_a,
                total_ce_loss_sentence_a_b,
                total_kl_loss_sentence_b_a,
                total_kl_loss_sentence_a_b
            )

        # This loop takes the results from the multithreading and sums the losses for each batch
        for result in results:
            total_loss_sentence_b_a += result[0]
            total_loss_sentence_a_b += result[1]
            total_ce_loss_sentence_b_a += result[2]
            total_ce_loss_sentence_a_b += result[3]
            total_kl_loss_sentence_b_a += result[4]
            total_kl_loss_sentence_a_b += result[5]

    return (
        total_loss_sentence_b_a,
        total_loss_sentence_a_b,
        total_ce_loss_sentence_b_a,
        total_ce_loss_sentence_a_b,
        total_kl_loss_sentence_b_a,
        total_kl_loss_sentence_a_b
    )


def find_loss_bidirectional_all_previous_sentences(sentence_encodings, sentence_masks,
        VAE, optimizer, mask, loss_fn, beta, model_type, curr_batch_size, curr_seq_len, input_tokens):
    '''This function finds the loss for the bidirectional training on all previous sentences.
    So if the sentence is the 3rd sentence in the story, it will predict the first two sentences and vice versa.'''
    total_loss_all_previous_sentences = 0
    total_ce_loss_all_previous_sentences = 0
    total_kl_loss_sentence_all_previous_sentences = 0

    mask = torch.ones((1, curr_seq_len), dtype=torch.long).to(Device.device)
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # use a partial and map to run the function on all the sentences for all batches
        partial_bidirectional_all_previous_sentences = partial(bidirectional_all_previous_sentences, VAE=VAE, optimizer=optimizer,
            sentence_encodings=sentence_encodings, sentence_masks=sentence_masks, mask=mask,
            loss_fn=loss_fn, beta=beta, model_type=model_type, curr_batch_size=curr_batch_size, curr_seq_len=curr_seq_len, input_tokens=input_tokens
        )
        try:
            results = executor.map(partial_bidirectional_all_previous_sentences, range(len(sentence_encodings)))
        except RuntimeError as e:
            print(e)
            print("RuntimeError: find_loss_bidirectional_all_previous_sentences()")
            return (
                total_loss_all_previous_sentences,
                total_ce_loss_all_previous_sentences,
                total_kl_loss_sentence_all_previous_sentences
            )

        # This loop takes the results from the multithreading and sums the losses for each batch
        for result in results:
            total_loss_all_previous_sentences += result[0]
            total_ce_loss_all_previous_sentences += result[1]
            total_kl_loss_sentence_all_previous_sentences += result[2]

    return (
        total_loss_all_previous_sentences,
        total_ce_loss_all_previous_sentences,
        total_kl_loss_sentence_all_previous_sentences,
    )

def bidirectional_all_previous_sentences(idx, VAE, optimizer, sentence_encodings, sentence_masks, mask,
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
        for idx in range(len(sentence_encodings[batch_idx])):
            sentence_encoding = sentence_encodings[batch_idx, idx][0: input_tokens.shape[1]].unsqueeze(0)
            sentence_mask = sentence_masks[batch_idx, idx][0: input_tokens.shape[1]].unsqueeze(0)
            assert(sentence_encoding.shape[1] == sentence_mask.shape[1])

            # This is the case where the sentence is the first sentence in the story
            # It does not have any previous sentences before it to predict
            if idx > 0:
                sentence_encoding_padded = F.pad(sentence_encoding, (0, curr_seq_len - sentence_encoding.shape[1]))
                sentence_mask_padded = F.pad(sentence_mask, (0, curr_seq_len - sentence_mask.shape[1]))
                assert(sentence_encoding_padded.shape[1] == sentence_mask_padded.shape[1])

                # SENTENCE LEVEL LOSS, Sentence B -> All Previous Sentences
                output_all_previous_sentences = train_step(VAE, optimizer, sentence_mask_padded,
                    sentence_encoding_padded, prev_encodings, prev_masks,
                    sentence_encoding_padded, prev_encodings, mask, loss_fn, beta, model_type
                )

                loss_all_previous_sentences, ce_loss_all_previous_sentences, kl_loss_sentence_all_previous_sentences = output_all_previous_sentences[-1]
                total_loss_all_previous_sentences += loss_all_previous_sentences
                total_ce_loss_all_previous_sentences += ce_loss_all_previous_sentences
                total_kl_loss_sentence_all_previous_sentences += kl_loss_sentence_all_previous_sentences

            prev_encodings[0, 0: input_tokens.shape[1]] = sentence_encoding.flatten()
            prev_masks[0, 0: input_tokens.shape[1]] = sentence_mask.flatten()
            assert(prev_encodings.shape[1] == prev_masks.shape[1] == curr_seq_len)
    
    return (
        total_loss_all_previous_sentences,
        total_ce_loss_all_previous_sentences,
        total_kl_loss_sentence_all_previous_sentences,
    )

def bidirectional_two_sentences(idx, VAE, optimizer, sentence_encodings, sentence_masks, mask, 
        loss_fn, beta, model_type, curr_batch_size, curr_seq_len, input_tokens, sentence_directions):
    '''This function finds the loss for the bidirectional training of the previous sentence.'''
    total_loss_sentence_b_a = 0
    total_loss_sentence_a_b = 0
    total_ce_loss_sentence_b_a = 0
    total_ce_loss_sentence_a_b = 0
    total_kl_loss_sentence_b_a = 0
    total_kl_loss_sentence_a_b = 0

    for batch_idx in range(curr_batch_size):
        for idx in range(len(sentence_encodings[batch_idx]) - 1):
            # Get the sentence encoding and mask for the current sentence and the next sentence
            sentence_encoding_a = sentence_encodings[batch_idx, idx][0: input_tokens.shape[1]].unsqueeze(0)       
            sentence_mask_a = sentence_masks[batch_idx, idx][0: input_tokens.shape[1]].unsqueeze(0)
            assert(sentence_encoding_a.shape[1] == sentence_mask_a.shape[1])

            sentence_encoding_b = sentence_encodings[batch_idx, idx + 1][0: input_tokens.shape[1]].unsqueeze(0)
            sentence_mask_b = sentence_masks[batch_idx, idx + 1][0: input_tokens.shape[1]].unsqueeze(0)
            assert(sentence_encoding_b.shape[1] == sentence_mask_b.shape[1])

            if sentence_directions[batch_idx, idx] == 0 or sentence_directions[batch_idx, idx] == 2:
                # SENTENCE LEVEL LOSS, Sentence B -> Sentence A
                output_sentence_b_a = train_step(VAE, optimizer, sentence_encoding_b, sentence_mask_b, sentence_encoding_a, sentence_mask_a,
                        sentence_encoding_b, sentence_encoding_a, mask, loss_fn, beta, model_type)
                loss_sentence_b_a, ce_loss_sentence_b_a, kl_loss_sentence_b_a = output_sentence_b_a[-1]

                total_loss_sentence_b_a += loss_sentence_b_a
                total_ce_loss_sentence_b_a += ce_loss_sentence_b_a
                total_kl_loss_sentence_b_a += kl_loss_sentence_b_a

            if sentence_directions[batch_idx, idx] == 0 or sentence_directions[batch_idx, idx] == 1:
                # SENTENCE LEVEL LOSS, Sentence A -> Sentence B
                output_sentence_a_b = train_step(VAE, optimizer, sentence_encoding_a, sentence_mask_a, sentence_encoding_b, sentence_mask_b,
                        sentence_encoding_a, sentence_encoding_b, mask, loss_fn, beta, model_type)
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
        total_kl_loss_sentence_a_b
    )
