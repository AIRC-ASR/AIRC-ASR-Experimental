import torch
import torchaudio
import torch.nn as nn
from transformers import OPTForCausalLM, AutoConfig, AutoTokenizer
from speechbrain.pretrained import EncoderClassifier

from typing import List, Optional
from speechbrain.pretrained import EncoderClassifier

class SpeakerAwareOPT(OPTForCausalLM):
    def __init__(self, config, speaker_embeddings=None, model_name="facebook/opt-350m"):
        super().__init__(config)
        self.encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.speaker_embeddings = speaker_embeddings.flatten() if speaker_embeddings is not None else None
        self.embedding_size = self.speaker_embeddings.shape[0] if self.speaker_embeddings is not None else None
        print("Loaded speaker embeddings from", self.speaker_embeddings.shape)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        print("EMBEDDINGSS", self.speaker_embeddings)
        if self.speaker_embeddings is None:
            return super().forward(input_ids=input_ids, attention_mask=attention_mask,
                                      head_mask=head_mask, past_key_values=past_key_values,
                                      inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache,
                                      output_attentions=output_attentions,
                                      output_hidden_states=output_hidden_states, return_dict=return_dict)
        
        else:
          inputs_embeds = torch.cat([inputs_embeds, self.speaker_embeddings.unsqueeze(0)], dim=-1)
          outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask,
                                      head_mask=head_mask, past_key_values=past_key_values,
                                      inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache,
                                      output_attentions=output_attentions,
                                      output_hidden_states=output_hidden_states, return_dict=return_dict)
          return outputs

# Example usage
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m", padding_side='left')

# Initialize speaker embeddings
encoder = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-xvect-voxceleb", 
    savedir="pretrained_models/spkrec-xvect-voxceleb"
)
speech_signal, sample_rate = torchaudio.load('test.wav')
speaker_embeddings = encoder.encode_batch(speech_signal)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = AutoConfig.from_pretrained("facebook/opt-350m")
config.output_hidden_states = True
model = SpeakerAwareOPT(config, speaker_embeddings=speaker_embeddings, model_name="facebook/opt-350m")

# Prepare input text and speech signal
input_text = "This is a test sentence."
speech_signal, sample_rate = torchaudio.load('test.wav')

# Encode the speaker ID and concatenate it to the input tensor
speaker_embeddings = model.encoder.encode_batch(speech_signal)
inputs = tokenizer(input_text, return_tensors='pt')
# create an instance of the nn.Embedding class
embedding_layer = nn.Embedding(num_embeddings=100, embedding_dim=50)

# create an input tensor
# input_tensor = torch.LongTensor([[1, 2, 3], [4, 5, 6]])
# pass the input tensor through the embedding layer and call the wte() method
print('SHAPEEE', inputs['input_ids'].shape)
output_tensor = embedding_layer(inputs['input_ids'])
wte_output = embedding_layer.weight
print('wte_output', wte_output.shape)

speaker_embeddings = speaker_embeddings.flatten()
inputs_embeds = torch.cat([wte_output, speaker_embeddings], dim=-1)

# Run the model forward pass
outputs = model(inputs_embeds=inputs_embeds)
