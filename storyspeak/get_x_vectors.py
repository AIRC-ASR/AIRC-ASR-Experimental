
import torchaudio
from speechbrain.pretrained import EncoderClassifier

# Relevant Resource: https://huggingface.co/speechbrain/spkrec-xvect-voxceleb
encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb")
speech_signal, sample_rate = torchaudio.load('test.wav')
embeddings = encoder.encode_batch(speech_signal)
print('Embeddings shape:', embeddings.shape, 'sample rate', sample_rate ,'Embeddings:', embeddings)