import json
import sys
import numpy
import sentencepiece
import speechbrain
from speechbrain.pretrained import EncoderASR
import torch
from itertools import *
import tensorflow as tf
import tensorflow_datasets as tfds

import tensorflow_io


def safe_join(fragments: list[str], joiner=""):
    if fragments is None:
        return fragments
    fragments = list(filter(lambda frag: frag is not None, fragments))
    return joiner.join(fragments)


class CustomEncoder(EncoderASR):

    @staticmethod
    def filter_output(preds: torch.Tensor, scores: torch.Tensor, blank_id=-1):

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
    def decode_probs(cls, probabilities, seq_lens, blank_id=-1, top_k=1):

        if isinstance(blank_id, int) and blank_id < 0:
            blank_id = probabilities.shape[-1] + blank_id
        batch_max_len = probabilities.shape[1]
        batch_preds = []
        batch_scores = []
        for seq, seq_len in zip(probabilities, seq_lens):
            actual_size = int(torch.round(seq_len * batch_max_len))
            zipped_scores, zipped_predictions = torch.topk(seq.narrow(0, 0, actual_size), k=top_k, dim=1)
            scores, predictions = list(torch.split(zipped_scores, 1, dim=1)), list(torch.split(zipped_predictions, 1, dim=1))

            for i in range(len(predictions)):
                predictions[i] = torch.squeeze(predictions[i])
                scores[i] = torch.squeeze(scores[i])

            preds, scores = cls.merge(scores, predictions, blank_id=blank_id, threshold=3)
            batch_preds.append(preds)
            batch_scores.append(scores)
        return batch_preds, batch_scores

    def transcribe_batch(self, wavs, wav_lens, top_k=1):
        """Transcribes the input audio into a sequence of words

        The waveforms should already be in the model's desired format.
        You can call:
        ``normalized = EncoderASR.normalizer(signal, sample_rate)``
        to get a correctly converted signal in most cases.

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

        Returns
        -------
        list
            Each waveform in the batch transcribed.
        tensor
            Each predicted token id.
        """
        with torch.no_grad():
            wav_lens = wav_lens.to(self.device)
            encoder_out = self.encode_batch(wavs, wav_lens)
            predictions, scores = self.decode_probs(encoder_out, wav_lens, blank_id=0, top_k=top_k)
            if isinstance(self.tokenizer, speechbrain.dataio.encoder.CTCTextEncoder):
                predicted_words = [[self.tokenizer.decode_ndim(token_seq)
                                    for token_seq in predict_subset]
                                   for predict_subset in predictions]
            elif isinstance(self.tokenizer, sentencepiece.SentencePieceProcessor):
                predicted_words = [[
                    self.tokenizer.decode_ids(token_seq)
                    for token_seq in predict_subset
                ] for predict_subset in predictions]
            else:
                sys.exit("The tokenizer must be sentencepiece or CTCTextEncoder")

        return predicted_words, predictions, scores


# wavs, wav_lens, ref_text = zip(*[(x["speech"].numpy(), x["speech"].shape[0], x['text'].numpy().decode()) for idx, x in enumerate(ds[subset]) if idx<max_samples])

def parse_mistakes(ref_text: str, hyp_tokens: list[str], scores: list[torch.Tensor]):
    ref = ref_text.split()
    split_hyps = [[]]
    split_scores = [[]]

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

    return report, ref, hyp


def summarize_reports(reports: list[dict], refs: list[str], hyp: list[list[list[str]]]):
    summary = {"ins": 0, "del": 0, "sub": 0, "total": 0, "words": 0}

    print("Mistakes:\nReference -> Hypothesis")
    for i in range(len(reports)):
        summary["ins"] += len(reports[i]["ins"])
        summary["del"] += len(reports[i]["del"])
        summary["sub"] += len(reports[i]["sub"])
        summary["total"] += reports[i]["total"]
        summary["words"] += reports[i]["words"]

        def score_map():
            merged_str = ""
            for j in range(len(hyp[i][idx: idx + 2])):
                for k in range(len(hyp[i][idx: idx + 2][j])):
                    if hyp[i][idx: idx + 2][j][k] is None:
                        continue
                    merged_str += f"{hyp[i][idx: idx + 2][j][k]}: {round(reports[i]['scores'][idx: idx + 2][j][k].item(), 2)}, "
                merged_str += "| "
            print(merged_str, "\n")

        def join_words(words: list[list[str]]):
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
            score_map()

    print("Total %WER {} [ {} errors / {} words, {} ins, {} del, {} sub ]".format(round(summary["total"] / summary["words"]*100, 2),
                                                                     summary["total"], summary["words"], summary['ins'], summary['del'], summary['sub']))

    if summary["total"] > 0:
        print("Out of {} errors: {}% ins, {}% del, {}% sub".format(summary["total"], round(summary["ins"] / summary["total"]*100, 2),
                                                                   round(summary["del"] / summary["total"]*100, 2),
                                                                   round(summary["sub"] / summary["total"]*100, 2)))


def main(max_samples=500, batch_size=1, top_k=1):
    PATTERN = """
Reference:
{ref}
Hypothesis(es):
{hyp}

Top Scores:
{tokens}
{scores}

    ======================
"""

    subset = "dev_clean"

    asr_model = CustomEncoder.from_hparams(source="speechbrain/asr-wav2vec2-librispeech", savedir="pretrained_models/asr-wav2vec2-librispeech")
    # asr_model.transcribe_file("raw/3660-172183-0001.flac")

    ds = tfds.load('librispeech', builder_kwargs={'config': 'lazy_decode'})
    ds_iter = iter(ds[subset])

    all_reports = []
    all_refs = []
    all_hyps = []

    for i in range(0, max_samples, batch_size):
        wavs, wav_lens, ref_texts = [], [], []
        print(f"\nLoading batch {i // batch_size}...\n")
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
            report, ref, hyp = parse_mistakes(ref_texts[i], transcripts[0], scores[i][0])
            all_reports.append(report)
            all_refs.append(ref)
            all_hyps.append(hyp)

            wer_str = "%WER {} [ {} errors / {} words, {} ins, {} del, {} sub ]".format(round(report["total"] / report["words"]*100, 2),
                                                                           report["total"], report["words"], len(report['ins']), len(report['del']), len(report['sub']))
            print(wer_str)

            hyp_strs = [safe_join(trans) for trans in transcripts]
            token_str = ""
            score_str = ""
            for j, token in enumerate(transcripts[0]):
                token_str += token.ljust(7) + "|"
                score_str += str(round(scores[i][0][j].item(), 2)).ljust(7) + "|"
            print(PATTERN.format(ref=ref_texts[i], hyp=json.dumps(hyp_strs).replace('", ', '\n').
                                 replace("[", "").replace("]", "").replace('"', ""), tokens=token_str, scores=score_str))

    summarize_reports(all_reports, all_refs, all_hyps)


if __name__ == "__main__":

    main(50, 1, 2)
