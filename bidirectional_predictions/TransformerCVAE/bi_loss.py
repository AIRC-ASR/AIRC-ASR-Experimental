from concurrent.futures import ThreadPoolExecutor
from functools import partial
from bi_training_core import train_step, Device
import torch
import os


# NOTE: This is the bidirectional running of the program
def get_sentence_encodings_and_masks(y_tokens, y_mask, tokenizer, batch_schedule, cur_b_schedule):
    '''This function takes the y_tokens and y_mask and returns the sentence encodings and masks.'''
    y_tokens_text = tokenizer.decode(y_tokens[0, :][y_mask[0, :] == 1].tolist())
    y_sentences = y_tokens_text.split('.')

    # Shape: (number of stories, number of sentences, max sentence length)
    y_sentence_encodings = torch.zeros((batch_schedule[cur_b_schedule][0], len(y_sentences), batch_schedule[cur_b_schedule][1]), dtype=torch.long).to(Device.device)
    y_sentence_masks = torch.zeros((batch_schedule[cur_b_schedule][0], len(y_sentences), batch_schedule[cur_b_schedule][1]), dtype=torch.long).to(Device.device)

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        partial_get_sentence_encoding_and_mask = partial(get_sentence_encoding_and_mask, tokenizer=tokenizer)
        results = executor.map(partial_get_sentence_encoding_and_mask, y_sentences)
        for i, result in enumerate(results):
            y_sentence_encodings[:, i, :len(result[0])] = torch.tensor(result[0])
            y_sentence_masks[:, i, :len(result[1])] = torch.tensor(result[1])

    return y_sentence_encodings, y_sentence_masks

def get_sentence_encoding_and_mask(sentence, tokenizer):
    '''This function takes a sentence and returns the encoding and mask.'''
    sentence_encoding = tokenizer.encode(sentence + '.')
    sentence_mask = torch.ones(len(sentence_encoding), dtype=torch.long).to(Device.device)
    assert(len(sentence_encoding) == len(sentence_mask))

    return sentence_encoding, sentence_mask

def bidirectional_loss(loss_type, VAE, optimizer, y_mask, y_tokens, x_mask, x_tokens,
        target_tokens, input_tokens, mask, loss_fn, beta, model_type, tokenizer, batch_schedule, cur_b_schedule):
        '''This function runs the bidirectional training on different levels.
        loss_types designates the possible loss types: 
            "previous_sentence": The latest sentence needs to predict the previous one and vice versa.
            "previous_sentences" The latest sentence needs to predict the previous ones and vice versa.
            "prompt": The prompt predicts the target story and vice versa.
        All other arguments are the same as train_step()
        '''
        y_sentence_encodings, y_sentence_masks = get_sentence_encodings_and_masks(y_tokens, y_mask, tokenizer, batch_schedule, cur_b_schedule)
        assert(len(y_sentence_encodings) == len(y_sentence_masks))

        if loss_type == "previous_sentence":
            return find_loss_bidirectional_two_sentences(y_sentence_encodings, y_sentence_masks, VAE, optimizer,
                y_sentence_encodings, y_sentence_masks, x_mask, x_tokens, target_tokens, input_tokens, mask, loss_fn, beta, model_type, batch_schedule, cur_b_schedule)
        elif loss_type == "all_previous_sentences":
            return find_loss_bidirectional_all_previous_sentences(y_sentence_encodings, y_sentence_masks, VAE, optimizer,
                y_sentence_encodings, y_sentence_masks, x_mask, x_tokens, target_tokens, input_tokens, mask, loss_fn, beta, model_type, batch_schedule, cur_b_schedule)


def find_loss_bidirectional_two_sentences(y_sentence_encodings, y_sentence_masks,
        VAE, optimizer, y_mask, y_tokens, x_mask, x_tokens, target_tokens, input_tokens, mask, loss_fn,
        beta, model_type, batch_schedule, cur_b_schedule):
    total_loss_sentence_b_a = 0
    total_loss_sentence_a_b = 0
    total_ce_loss_sentence_b_a = 0
    total_ce_loss_sentence_a_b = 0
    total_kl_loss_sentence_b_a = 0
    total_kl_loss_sentence_a_b = 0

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # use a partial and map to run the function on all the sentences for all batches
        partial_bidirectional_two_sentences = partial(bidirectional_two_sentences, device=Device.device, VAE=VAE, optimizer=optimizer,
            y_sentence_encodings=y_sentence_encodings, y_sentence_masks=y_sentence_masks, y_mask=y_mask, y_tokens=y_tokens, x_mask=x_mask, x_tokens=x_tokens,
            target_tokens=target_tokens, input_tokens=input_tokens, mask=mask, loss_fn=loss_fn, beta=beta, model_type=model_type, batch_schedule=batch_schedule, cur_b_schedule=cur_b_schedule)
        results = executor.map(partial_bidirectional_two_sentences, range(len(y_sentence_encodings)))
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


def bidirectional_two_sentences(idx, VAE, optimizer, y_sentence_encodings, y_sentence_masks, mask, 
        loss_fn, beta, model_type, batch_schedule, cur_b_schedule):
    total_loss_sentence_b_a = 0
    total_loss_sentence_a_b = 0
    total_ce_loss_sentence_b_a = 0
    total_ce_loss_sentence_a_b = 0
    total_kl_loss_sentence_b_a = 0
    total_kl_loss_sentence_a_b = 0

    for batch_idx in range(batch_schedule[cur_b_schedule][0]):
        for idx in range(len(y_sentence_encodings[batch_idx]) - 1):
            y_sentence_encoding_a = y_sentence_encodings[batch_idx, idx].unsqueeze(0)
            y_sentence_mask_a = y_sentence_masks[batch_idx, idx].unsqueeze(0)

            y_sentence_encoding_b = y_sentence_encodings[batch_idx, idx + 1].unsqueeze(0)
            y_sentence_mask_b = y_sentence_masks[batch_idx, idx + 1].unsqueeze(0)

            # SENTENCE LEVEL LOSS, Sentence B -> Sentence A
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


def find_loss_bidirectional_all_previous_sentences(y_sentence_encodings, y_sentence_masks,
    VAE, optimizer, mask, loss_fn, beta, model_type, batch_schedule, cur_b_schedule):
    total_loss_all_previous_sentences = 0
    total_ce_loss_all_previous_sentences = 0
    total_kl_loss_sentence_all_previous_sentences = 0

    for batch_idx in range(batch_schedule[cur_b_schedule][0]):
        for idx in range(len(y_sentence_encodings[batch_idx])):
            y_sentence_encoding = y_sentence_encodings[batch_idx, idx].unsqueeze(0)
            y_sentence_mask = y_sentence_masks[batch_idx, idx].unsqueeze(0)

            if idx > 0:
                # SENTENCE LEVEL LOSS, Sentence B -> All Previous Sentences
                output_all_previous_sentences = train_step(VAE, optimizer, y_sentence_encoding,
                    y_sentence_mask, y_sentence_encodings[batch_idx, :idx], y_sentence_masks[batch_idx, :idx],
                    y_sentence_encoding, y_sentence_encodings[batch_idx, :idx], mask, loss_fn, beta, model_type
                )

                loss_all_previous_sentences, ce_loss_all_previous_sentences, kl_loss_sentence_all_previous_sentences = output_all_previous_sentences[-1]
                total_loss_all_previous_sentences += loss_all_previous_sentences
                total_ce_loss_all_previous_sentences += ce_loss_all_previous_sentences
                total_kl_loss_sentence_all_previous_sentences += kl_loss_sentence_all_previous_sentences

    return (
        total_loss_all_previous_sentences,
        total_ce_loss_all_previous_sentences,
        total_kl_loss_sentence_all_previous_sentences,
    )
