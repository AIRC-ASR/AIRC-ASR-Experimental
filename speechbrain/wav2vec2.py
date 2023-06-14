import copy
import json
import sys
import numpy
import sentencepiece
import speechbrain
from speechbrain.pretrained import EncoderASR
from SoundsLike.SoundsLike import Search
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from itertools import *
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_io
import Levenshtein


def safe_join(fragments: list[str], joiner=""):
    """
    Concatenates a list of strings into on string, safely ignoring None types

    Parameters
    ----------
    fragments: list[str]
        The substrings to join
    joiner: str
        The string used to join the substrings

    Returns
    -------
    str
        The joined string
    """

    if fragments is None:
        return fragments
    fragments = list(filter(lambda frag: frag is not None, fragments))
    return joiner.join(fragments)


class CustomEncoder(EncoderASR):
    """Custom class to add behavior to the ASR encoder"""

    @staticmethod
    def filter_output(preds: torch.Tensor, scores: torch.Tensor, blank_id=-1):
        """
        Filters out duplicate tokens and blank tokens from the predictions

        Parameters
        ----------
        preds: torch.Tensor
            The predicted tokens
        scores: torch.Tensor
            The confidence scores of each of the tokens
        blank_id: int
            The id of the blank tokens to filter out

        Returns
        -------
        tuple[list, list]
            The filtered predictions and their corresponding scores
        """

        # Filter the repetitions
        preds_out = []
        scores_out = []
        curr = None
        for token in preds:
            if token != curr:
                if token != blank_id:
                    preds_out.append(token)
                    scores_out.append(scores[0])
                curr = token
            scores = scores[1:]

        return preds_out, scores_out

    @classmethod
    def merge(cls, scores: list[torch.Tensor], predictions: list[torch.Tensor], blank_id=-1, threshold: float = 3):
        """
        Takes in several, greedy predictions and merges the more likely predictions into the less likely ones, creating predictions of reasonable variation

        Parameters
        ----------
        scores: list[torch.Tensor]
            The confidence scores of each of the predictions
        predictions: list[torch.Tensor]
            The predictions
        blank_id: int
            The id of the blank tokens to filter out
        threshold: float
            The confidence difference threshold for replacing more likely tokens with less likely ones

        Returns
        -------
        tuple:
            The merged predictions and their scores
        """

        merged_preds = [torch.clone(predictions[0])]
        merged_scores = [torch.clone(scores[0])]
        for i in range(len(predictions)-1):
            merged_preds.append(torch.clone(predictions[i]))
            merged_scores.append(torch.clone(scores[i]))

            for j in range(len(merged_preds[i])):
                if abs(scores[i][j] - scores[i+1][j]) < threshold:
                    # Choose next most likely choice if the difference of the scores are within the threshold
                    merged_preds[i+1][j] = predictions[i+1][j]
                    merged_scores[i+1][j] = scores[i+1][j]

            merged_preds[i], merged_scores[i] = cls.filter_output(merged_preds[i], merged_scores[i], blank_id=blank_id)
        merged_preds[-1], merged_scores[-1] = cls.filter_output(merged_preds[-1], merged_scores[-1], blank_id=blank_id)

        return merged_preds, merged_scores

    @classmethod
    def decode_probs(cls, encoded: torch.Tensor, seq_lens: torch.Tensor, blank_id=-1, top_k=1):
        """
        Decodes the output of the encoder along with the confidence probabilities

        Parameters
        ----------
        encoded: torch.Tensor
            The encoded sound data
        seq_lens: torch.Tensor
            The ratio of the lengths of the sequences relative to the longest sequence
        blank_id: int
            The id of the blank tokens to filter out
        top_k: int
            How many prediction sequences to gather

        Returns
        -------
        tuple
            The predicted token ids along with the confidence probabilities
        """

        if isinstance(blank_id, int) and blank_id < 0:
            blank_id = encoded.shape[-1] + blank_id
        batch_max_len = encoded.shape[1]
        batch_preds = []
        batch_scores = []
        for seq, seq_len in zip(encoded, seq_lens):
            # Restore the size of the current sequence
            actual_size = int(torch.round(seq_len * batch_max_len))
            zipped_scores, zipped_predictions = torch.topk(seq.narrow(0, 0, actual_size), k=top_k, dim=1)
            scores, predictions = list(torch.split(zipped_scores, 1, dim=1)), list(torch.split(zipped_predictions, 1, dim=1))

            for i in range(len(predictions)):
                predictions[i] = torch.squeeze(predictions[i])
                scores[i] = torch.squeeze(scores[i])

            # Combine top predictions into sensible varieties
            preds, scores = cls.merge(scores, predictions, blank_id=blank_id, threshold=3)
            batch_preds.append(preds)
            batch_scores.append(scores)

        return batch_preds, batch_scores

    def _helper(self, predicted_words, predictions, scores, i, j, current_prob, confidence_threshold, current_token):
        leaves = []

        if current_prob >= confidence_threshold:
            if j + 1 < len(predictions[i]):
                leaves.extend(self._helper(predicted_words, predictions, scores, i, j + 1, torch.sigmoid(scores[i][j + 1]).item(), confidence_threshold, predicted_words[i][j + 1]))
            elif i + 1 < len(predicted_words):
                leaves.extend(self._helper(predicted_words, predictions, scores, i + 1, 0, torch.sigmoid(scores[i + 1][0]).item(), confidence_threshold, predicted_words[i + 1][0]))
            else:
                leaves.append((predicted_words, predictions, scores))
        else:
            leaves.append((predicted_words, predictions, scores))  # Append the current state as a leaf

            if i + 1 < len(predicted_words):
                second_best_token = predicted_words[i + 1][j] if j < len(predicted_words[i + 1]) else None
                second_best_prediction = predictions[i + 1][j].item() if j < len(predictions[i + 1]) else None
                second_best_score = scores[i + 1][j].item() if j < len(scores[i + 1]) else None

                if second_best_token is None:
                    if j + 1 < len(predictions[i]):
                        leaves.extend(self._helper(predicted_words, predictions, scores, i, j + 1, torch.sigmoid(scores[i][j + 1]).item(), confidence_threshold, predicted_words[i][j + 1]))
                    elif i + 1 < len(predicted_words):
                        leaves.extend(self._helper(predicted_words, predictions, scores, i + 1, 0, torch.sigmoid(scores[i + 1][0]).item(), confidence_threshold, predicted_words[i + 1][0]))
                    else:
                        leaves.append((predicted_words, predictions, scores))
                else:
                    predicted_words_branch = copy.deepcopy(predicted_words)
                    predictions_branch = copy.deepcopy(predictions)
                    scores_branch = copy.deepcopy(scores)

                    try:
                        predicted_words_branch[i][j] = second_best_token
                        predictions_branch[i][j] = second_best_prediction
                        scores_branch[i][j] = second_best_score
                    except IndexError:
                        print(f"IndexError: {i}, {j}, {len(predicted_words)}, {len(predictions)}, {len(scores)}, {len(predicted_words[i][0])}")

                    if j + 1 < len(predictions[i]) and j + 1 < len(predictions_branch[i]):
                        leaves.extend(self._helper(predicted_words, predictions, scores, i, j + 1, torch.sigmoid(scores[i][j + 1]).item(), confidence_threshold, predicted_words[i][j + 1]))
                        leaves.extend(self._helper(predicted_words_branch, predictions_branch, scores_branch, i, j + 1, torch.sigmoid(scores_branch[i][j + 1]).item(), confidence_threshold, predicted_words_branch[i][j + 1]))
                    elif i + 1 < len(predicted_words) and i + 1 < len(predicted_words_branch):
                        leaves.extend(self._helper(predicted_words, predictions, scores, i + 1, 0, torch.sigmoid(scores[i + 1][0]).item(), confidence_threshold, predicted_words[i + 1][0]))
                        leaves.extend(self._helper(predicted_words_branch, predictions_branch, scores_branch, i + 1, 0, torch.sigmoid(scores_branch[i + 1][0]).item(), confidence_threshold, predicted_words_branch[i + 1][0]))
                    else:
                        leaves.append((predicted_words, predictions, scores))

        return leaves



    def beam_search_decoding(self, predicted_words, predictions, scores, confidence_threshold):
        i = 0
        j = 0
        current_prob = torch.sigmoid(scores[i][j]).item()
        current_token = predicted_words[i][j]
        leaves = self._helper(predicted_words, predictions, scores, i, j, current_prob, confidence_threshold, current_token)
        
        sentence_options = []
        for idx, (predicted_words_, predictions_, scores_) in enumerate(leaves):
            sentence_ = "".join(predicted_words_[0])
            sentence_score = sum(scores_[0])
            sentence_options.append((idx, sentence_, sentence_score))

        print("BEAM SEARCH DECODING OUTPUTS")
        print("NUMBER OF OPTIONS: ", len(sentence_options))
        print("")

        for idx, sentence_, sentence_score in sentence_options:
            print(f"IDX: {idx}")
            print(f"SENTENCE: {sentence_}")
            print(f"SCORE: {sentence_score}")
            print("")

        best_option = max(sentence_options, key=lambda x: x[2])
        print("BEST OPTION")
        print(f"IDX: {best_option[0]}")
        print(f"SENTENCE: {best_option[1]}")
        print(f"SCORE: {best_option[2]}")
        print("")

        best_idx = best_option[0]

        predicted_words = leaves[best_idx][0]
        predictions = leaves[best_idx][1]
        scores = leaves[best_idx][2]

        return predicted_words, predictions, scores

    def tokenize_predictions(self, predictions):
        if isinstance(self.tokenizer, speechbrain.dataio.encoder.CTCTextEncoder):
            hypotheses = [[self.tokenizer.decode_ndim(token_seq)
                                for token_seq in predict_subset]
                                for predict_subset in predictions]
        elif isinstance(self.tokenizer, sentencepiece.SentencePieceProcessor):
            hypotheses = [[
                self.tokenizer.decode_ids(token_seq)
                for token_seq in predict_subset
            ] for predict_subset in predictions]
        else:
            sys.exit("The tokenizer must be sentencepiece or CTCTextEncoder")

        return hypotheses

    def scores_to_probs(self, scores):
        """
        Converts a list of scores to probabilities using softmax.
        """
        probs = []

        for score_list in scores[0]:
            score_tensor = torch.tensor(score_list)
            probs_list = torch.sigmoid(score_tensor).tolist()
            probs.append(probs_list)

        return [probs]

    def transcribe_batch(self, wavs, wav_lens, top_k=2, confidence_threshold=0.991):
        """
        Transcribes the input audio into a sequence of words with branching on low confidence tokens.

        Arguments
        ---------
        wavs : torch.Tensor
            Batch of waveforms [batch, time, channels] or [batch, time]
            depending on the model.
        wav_lens : torch.Tensor
            Lengths of the waveforms relative to the longest one in the
            batch, tensor of shape [batch]. The longest one should have
            relative length 1.0 and others len(waveform) / max_length.
            Used for ignoring padding.
        top_k : int
            Number of top predictions to consider at each step.
        confidence_threshold : float
            The threshold below which a token is considered low confidence.

        Returns
        -------
        list
            Each waveform in the batch transcribed.
        tensor
            Each predicted token id.
        """

        with torch.no_grad():
            # Encodes the input audio from a waveform into a sequence of tokens
            encoder_out = self.encode_batch(wavs, wav_lens)

            # Decodes the encoded audio into a sequence of tokens
            predictions, scores = self.decode_probs(encoder_out, wav_lens, blank_id=0, top_k=top_k)

            # Convert the list of scores to a list of probabilities
            probs = self.scores_to_probs(scores)
            print('PROBS', probs)

            # Converts the predicted token ids into a list of tokens for each sentence
            predicted_words = self.tokenize_predictions(predictions)

            # Uses beam search decoding to replace low confidence tokens with the next best prediction
            # Updates the predicted_words, predictions, and scores lists
            predicted_words, predictions, scores = self.beam_search_decoding(predicted_words[0], predictions[0], scores[0], confidence_threshold)

        return [predicted_words], [predictions], [scores]

# wavs, wav_lens, ref_text = zip(*[(x["speech"].numpy(), x["speech"].shape[0], x['text'].numpy().decode()) for idx, x in enumerate(ds[subset]) if idx<max_samples])

def parse_mistakes(ref_text: str, hyp_tokens: list[str], scores: list[torch.Tensor]):
    """
    Groups the reference and hypothesis by word and calculates which mistakes the hypothesis has made

    Parameters
    ----------
    ref_text: str
        The correct transcription
    hyp_tokens: list[str]
        The predicted transcription
    scores: list[torch.Tensor]
        The scores of the hypothesis

    Returns
    -------
    tuple
        A report of the errors and the reorganized reference and hypothesis
    """

    ref = ref_text.split()
    split_hyps = [[]]
    split_scores = [[]]

    # Group hypothesis and scores by words instead of by token
    for i, token in enumerate(hyp_tokens):
        if token == " ":
            split_scores.append([])
            split_hyps.append([])
            continue
        split_scores[-1].append(scores[i])
        split_hyps[-1].append(hyp_tokens[i])

    max_len = max(len(ref), len(split_hyps))
    report = {"ins": [], "del": [], "sub": [], "total": 0, "words": len(ref), "scores": split_scores}

    ref = ref + [[None]] * (max_len - len(ref))
    hyp = split_hyps + [[None]] * (max_len - len(split_hyps))

    # Tally which mistakes are made by the hypothesis, using the reference
    hyp_offset = 0
    for i in range(max_len):
        if i+1+hyp_offset >= len(hyp):
            hyp += [[None]]*(i+2+hyp_offset - len(hyp))

        if ref[i] == safe_join(hyp[i+hyp_offset]):
            continue
        report["total"] += 1
        if i < max_len-1 and ref[i] == safe_join(hyp[i+1+hyp_offset]):
            report["ins"].append(i)
            hyp_offset += 1
        elif i < max_len-1 and ref[i+1] == safe_join(hyp[i+hyp_offset]):
            report["del"].append(i)
            hyp_offset -= 1
        else:
            report["sub"].append(i)

    # if report["total"]:
    #     visualize_confidence(hyp_tokens, list(ref_text), scores, report)

    return report, ref, hyp


def summarize_reports(reports: list[dict], refs: list[list[str]], hyp: list[list[list[str]]]):
    """
    Summarizes the results of all of the reports

    Parameters
    ----------
    reports: list[dict]
        All of the reports from the samples
    refs: list[list[str]]
        The correct transcriptions of the samples
    hyp: list[list[list[str]]]
        The guessed transcriptions of the samples
    """
    summary = {"ins": 0, "del": 0, "sub": 0, "total": 0, "words": 0}

    # print("Mistakes:\nReference -> Hypothesis")
    for i in range(len(reports)):
        summary["ins"] += len(reports[i]["ins"])
        summary["del"] += len(reports[i]["del"])
        summary["sub"] += len(reports[i]["sub"])
        summary["total"] += reports[i]["total"]
        summary["words"] += reports[i]["words"]

        """def score_map():
            \"""Arranges each token with its corresponding score\"""

            merged_str = ""
            for j in range(len(hyp[i][idx: idx + 2])):
                for k in range(len(hyp[i][idx: idx + 2][j])):
                    if hyp[i][idx: idx + 2][j][k] is None:
                        continue
                    merged_str += f"{hyp[i][idx: idx + 2][j][k]}: {round(reports[i]['scores'][idx: idx + 2][j][k].item(), 2)}, "
                merged_str += "| "
            # print(merged_str, "\n")

        def join_words(words: list[list[str]]):
            \"""Safely joins a list of lists\"""

            joined = []
            for word in words:
                joined.append(safe_join(word))
            return safe_join(joined, " ")

        for idx in reports[i]["ins"]:
            print(f"Insertion: {safe_join(refs[i][idx : idx+2], ' ')} -> {join_words(hyp[i][idx : idx+2])}")
            score_map()

        for idx in reports[i]["del"]:
            print(f"Deletion: {safe_join(refs[i][idx : idx+2], ' ')} -> {join_words(hyp[i][idx : idx+2])}")
            score_map()

        for idx in reports[i]["sub"]:
            print(f"Substitution: {safe_join(refs[i][idx : idx+2], ' ')} -> {join_words(hyp[i][idx : idx+2])}")
            score_map()"""

    print("Total %WER {} [ {} errors / {} words, {} ins, {} del, {} sub ]".format(round(summary["total"] / summary["words"]*100, 2),
                                                                     summary["total"], summary["words"], summary['ins'], summary['del'], summary['sub']))

    # if summary["total"] > 0:
    #     print("Out of {} errors: {}% ins, {}% del, {}% sub".format(summary["total"], round(summary["ins"] / summary["total"]*100, 2),
    #                                                                round(summary["del"] / summary["total"]*100, 2),
    #                                                                round(summary["sub"] / summary["total"]*100, 2)))


def visualize_confidence(hyp_tokens: list[str], ref_tokens: list[str], token_scores: list[torch.Tensor], error_report: dict, max_score=16):
    """
    Visualizes the confidence scores for each token in a pyplot

    Parameters
    ----------
    hyp_tokens: list[str]
        The hypothesized transcription of the sample
    ref_tokens: list[str]
        The reference transcription of the sample
    token_scores: list[torch.Tensor]
        The confidence scores of the predictions
    error_report: dict
        A report of all the errors in transcription
    max_score: int
        The maximum score a token can have
    """

    token_str = ""
    score_str = ""
    for j, token in enumerate(hyp_tokens):
        token_str += token.ljust(7) + "|"
        score_str += str(round(token_scores[j].item(), 2)).ljust(7) + "|"

    token_scores = [min(round(score.item()/max_score, 2), max_score) for score in token_scores]

    max_len = max(len(ref_tokens), len(hyp_tokens))
    fig, ax1 = plt.subplots(figsize=(len(hyp_tokens) * 0.15, 6), dpi=80)
    ax2 = ax1.twiny()
    ax1.twinx()

    legend = []

    for err_type in [("ins", "yellow"), ("del", "orange"), ("sub", "red")]:
        legend.append(mpatches.Patch(color=err_type[1], label=err_type[0]))
        for error in error_report[err_type[0]]:
            place = 1
            for work_scores in error_report["scores"][:error]:
                place += len(work_scores)+1
            if error >= len(error_report["scores"]):
                error = len(error_report)-1
            ax1.axvspan(place, place+len(error_report["scores"][error]), alpha=.5, color=err_type[1])

    ax1.set_xlabel('Predicted Tokens')
    ax1.set_ylabel('Prediction Confidence')
    ax2.set_xlabel('Original Tokens')

    ax1.plot(range(len(token_scores)), token_scores)
    ax1.set_xticks(range(max_len), hyp_tokens + [""] * (max_len - len(hyp_tokens)))
    ax2.set_xticks(range(max_len), ref_tokens + [""] * (max_len - len(ref_tokens)))
    ax1.legend(handles=legend)
    ax1.set_title("Confidence Scores of Generated Tokens")
    fig.show()

    pass


def main(max_samples=500, batch_size=1, top_k=1, start_idx=0):
    PATTERN = """
Reference:
{ref}
Hypothesis(es):
{hyp}
======================
"""

    subset = "dev_clean"

    asr_model = CustomEncoder.from_hparams(source="speechbrain/asr-wav2vec2-librispeech", savedir="pretrained_models/asr-wav2vec2-librispeech")
    # asr_model.transcribe_file("raw/3660-172183-0001.flac")

    ds = tfds.load('librispeech', builder_kwargs={'config': 'lazy_decode'})
    ds_iter = iter(ds[subset])

    for i in range(start_idx):
        next(ds_iter)

    all_reports, all_refs, all_hyps = [], [], []

    for i in range(0, max_samples, batch_size):
        wavs, wav_lens, ref_texts = [], [], []

        # print(f"\nLoading batch {i // batch_size}...\n")

        # Process raw data into batches
        for _ in range(batch_size):
            sample = next(ds_iter)
            wav, wav_len, ref_text = sample["speech"].numpy(), sample["speech"].shape[0], sample['text'].numpy().decode()
            wavs.append(wav)
            wav_lens.append(wav_len)
            ref_texts.append(ref_text)

        max_len = max(wav_lens)
        wav_lens = [[wav_len / max_len] for wav_len in wav_lens]
        wavs = numpy.array([torch.nn.functional.pad(torch.Tensor(wav), (0, max_len - wav.shape[0]),
                                                    mode="constant", value=0.0).numpy() for wav in wavs])

        transcriptions, tokens, scores = asr_model.transcribe_batch(torch.Tensor(wavs), torch.Tensor(wav_lens), top_k=top_k)

        for i, transcripts in enumerate(transcriptions):
            hyp_strs = [safe_join(trans) for trans in transcripts]
            print(PATTERN.format(ref=ref_texts[i], hyp=json.dumps(hyp_strs).replace('", ', '\n').
                                 replace("[", "").replace("]", "").replace('"', "")))

            report, ref, hyp = parse_mistakes(ref_texts[i], transcripts[0], scores[i][0])
            all_reports.append(report)
            all_refs.append(ref)
            all_hyps.append(hyp)

            print("%WER {} [ {} errors / {} words, {} ins, {} del, {} sub ]".format(round(report["total"] / report["words"]*100, 2),
                                                                           report["total"], report["words"], len(report['ins']), len(report['del']), len(report['sub'])))

    summarize_reports(all_reports, all_refs, all_hyps)


if __name__ == "__main__":

    main(50, 2, 2, start_idx=400)
