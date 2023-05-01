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
    def decode_probs(cls, probabilities, seq_lens, blank_id=-1, top_k=1):

        if isinstance(blank_id, int) and blank_id < 0:
            blank_id = probabilities.shape[-1] + blank_id
        batch_max_len = probabilities.shape[1]
        batch_outputs = []
        for seq, seq_len in zip(probabilities, seq_lens):
            actual_size = int(torch.round(seq_len * batch_max_len))
            scores, predictions = torch.topk(seq.narrow(0, 0, actual_size), k=top_k, dim=1)
            scores_split, predictions_split = torch.split(scores, 1, dim=1), torch.split(predictions, 1, dim=1)
            outs = []
            for i in range(len(predictions_split)):
                outs.append(cls.filter_output(torch.squeeze(predictions_split[i]).tolist(), blank_id=blank_id))
            batch_outputs.append(outs)
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


if __name__ == "__main__":
    asr_model = CustomEncoder.from_hparams(source="speechbrain/asr-wav2vec2-librispeech", savedir="pretrained_models/asr-wav2vec2-librispeech")
    # asr_model.transcribe_file("raw/3660-172183-0001.flac")

    ds = tfds.load('librispeech', builder_kwargs={'config': 'lazy_decode'})

    subset = "dev_clean"
    max_samples = 2

    wavs, wav_lens = zip(*[(x["speech"].numpy(), x["speech"].shape[0]) for idx, x in enumerate(ds[subset]) if idx<max_samples])
    max_len = max(wav_lens)
    wav_lens = [[wav_len/max_len] for wav_len in wav_lens]
    wavs = numpy.array([torch.nn.functional.pad(torch.Tensor(wav), (0, max_len-wav.shape[0]), mode="constant", value=0.0).numpy() for wav in wavs])

    transcriptions = asr_model.transcribe_batch(torch.Tensor(wavs), torch.Tensor(wav_lens), top_k=2)
    print(transcriptions)



