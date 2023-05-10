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


class CustomEncoder(EncoderASR):

    @staticmethod
    def filter_output(string_pred, blank_id=-1):

        if isinstance(string_pred, list):
            # Filter the repetitions
            string_out = [i[0] for i in groupby(string_pred)]

            # Filter the blank symbol
            string_out = list(filter(lambda elem: elem != blank_id, string_out))
        else:
            raise ValueError("filter_ctc_out can only filter python lists")
        return string_out

    @classmethod
    def merge(cls, scores: list[torch.Tensor], predictions: list[torch.Tensor], blank_id=-1, threshold: float = 3):
        merged = [torch.clone(predictions[0])]
        for i in range(len(predictions)-1):
            merged.append(torch.clone(predictions[i]))
            for j in range(len(merged[i])):
                if abs(scores[i][j] - scores[i+1][j]) < threshold:
                    # Choose next most likely choice if the difference of the scores are within the threshold
                    merged[i+1][j] = predictions[i+1][j]

            merged[i] = cls.filter_output(merged[i].tolist(), blank_id=blank_id)
        merged[-1] = cls.filter_output(merged[-1].tolist(), blank_id=blank_id)

        return merged

    @classmethod
    def decode_probs(cls, probabilities, seq_lens, blank_id=-1, top_k=1):

        if isinstance(blank_id, int) and blank_id < 0:
            blank_id = probabilities.shape[-1] + blank_id
        batch_max_len = probabilities.shape[1]
        batch_outputs = []
        for seq, seq_len in zip(probabilities, seq_lens):
            actual_size = int(torch.round(seq_len * batch_max_len))
            zipped_scores, zipped_predictions = torch.topk(seq.narrow(0, 0, actual_size), k=top_k, dim=1)
            scores, predictions = list(torch.split(zipped_scores, 1, dim=1)), list(torch.split(zipped_predictions, 1, dim=1))

            for i in range(len(predictions)):
                predictions[i] = torch.squeeze(predictions[i])
                scores[i] = torch.squeeze(scores[i])

            batch_outputs.append(cls.merge(scores, predictions, blank_id=blank_id, threshold=3))
        return batch_outputs

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
            predictions = self.decode_probs(encoder_out, wav_lens, blank_id=0, top_k=top_k)
            if isinstance(self.tokenizer, speechbrain.dataio.encoder.CTCTextEncoder):
                predicted_words = [[
                    "".join(self.tokenizer.decode_ndim(token_seq))
                    for token_seq in predict_subset
                ] for predict_subset in predictions]
            elif isinstance(
                self.tokenizer, sentencepiece.SentencePieceProcessor
            ):
                predicted_words = [[
                    self.tokenizer.decode_ids(token_seq)
                    for token_seq in predict_subset
                ] for predict_subset in predictions]
            else:
                sys.exit(
                    "The tokenizer must be sentencepiece or CTCTextEncoder"
                )

        return predicted_words, predictions


# wavs, wav_lens, ref_text = zip(*[(x["speech"].numpy(), x["speech"].shape[0], x['text'].numpy().decode()) for idx, x in enumerate(ds[subset]) if idx<max_samples])

def parse_mistakes(ref_text: str, hyp_text: str):
    ref = ref_text.split()
    hyp = hyp_text.split()

    max_len = max(len(ref), len(hyp))
    report = {"ins": [], "del": [], "sub": [], "total": 0, "words": len(ref)}

    ref = ref + [None] * (max_len - len(ref))
    hyp = hyp + [None] * (max_len - len(hyp))

    hyp_offset = 0
    for i in range(max_len):
        if i+1+hyp_offset >= len(hyp):
            hyp += [None]*(i+2+hyp_offset - len(hyp))

        if ref[i] == hyp[i+hyp_offset]:
            continue
        report["total"] += 1
        if i < max_len-1 and ref[i] == hyp[i+1+hyp_offset]:
            report["ins"].append(i)
            hyp_offset += 1
        elif i < max_len-1 and ref[i+1] == hyp[i+hyp_offset]:
            report["del"].append(i)
            hyp_offset -= 1
        else:
            report["sub"].append(i)

    return report, ref, hyp


def summarize_reports(reports: list[dict], refs: list[str], hyp: list[str]):
    summary = {"ins": 0, "del": 0, "sub": 0, "total": 0, "words": 0}

    print("Mistakes:\nReference -> Hypothesis")
    for i in range(len(reports)):
        summary["ins"] += len(reports[i]["ins"])
        summary["del"] += len(reports[i]["del"])
        summary["sub"] += len(reports[i]["sub"])
        summary["total"] += reports[i]["total"]
        summary["words"] += reports[i]["words"]

        for idx in reports[i]["ins"]:
            print(f"Insertion: {refs[i][idx : idx+2]} -> {hyp[i][idx : idx+2]}")

        for idx in reports[i]["del"]:
            print(f"Deletion: {refs[i][idx : idx+2]} -> {hyp[i][idx : idx+2]}")

        for idx in reports[i]["sub"]:
            print(f"Substitution: {refs[i][idx : idx+2]} -> {hyp[i][idx : idx+2]}")

    print("Total %WER {} [ {} / {}, {} ins, {} del, {} sub ]".format(round(summary["total"] / summary["words"]*100, 2),
                                                                     summary["total"], summary["words"], summary['ins'], summary['del'], summary['sub']))


def main():
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

    max_samples = 500
    batch_size = 1

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

        transcriptions, tokens = asr_model.transcribe_batch(torch.Tensor(wavs), torch.Tensor(wav_lens), top_k=2)

        for i, transcript in enumerate(transcriptions):
            report, ref, hyp = parse_mistakes(ref_texts[i], transcript[0])
            all_reports.append(report)
            all_refs.append(ref)
            all_hyps.append(hyp)

            wer_str = "%WER {} [ {} / {}, {} ins, {} del, {} sub ]".format(round(report["total"] / report["words"]*100, 2),
                                                                           report["total"], report["words"], len(report['ins']), len(report['del']), len(report['sub']))
            print(wer_str)
            print(PATTERN.format(ref=ref_texts[i], hyp=json.dumps(transcript).replace('", ', '\n').
                                 replace("[", "").replace("]", "").replace('"', "")))

    summarize_reports(all_reports, all_refs, all_hyps)


if __name__ == "__main__":

    main()
