from transformers import T5ForConditionalGeneration, T5Config, T5Tokenizer
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys
import numpy as np

# model setup
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
config = T5Config.from_pretrained("google/flan-t5-large")

my_input = "The jacket"
inputs = tokenizer(my_input, return_tensors="pt")
inputs_ids = inputs.input_ids

outputs = model.generate(
    input_ids=inputs_ids,
    attention_mask=inputs.attention_mask,
    output_hidden_states=True,
    return_dict_in_generate=True
)

tokens1 = tokenizer.convert_ids_to_tokens(inputs_ids[0])
print(f"These are the input IDS: {inputs_ids[0]}")
print(f"These are the tokens: {tokens1}")

encoder_hidden_states = outputs.encoder_hidden_states
decoder_hidden_states = outputs.decoder_hidden_states

seq_len = encoder_hidden_states[0].shape[1]
num_layers = len(encoder_hidden_states)

# for token_idx in range(seq_len):
#     print(f"token index: {token_idx}")
token_idx = 2

original_embedding = encoder_hidden_states[0][0, token_idx, :]
print(f"Original embedding: {original_embedding}")
cosine_similarities = []
for layer in encoder_hidden_states:
    cur_embedding = layer[0, token_idx, :]
    print(f"Current embedding: {cur_embedding}")
    sim = F.cosine_similarity(original_embedding, cur_embedding, dim=0).item()
    cosine_similarities.append(sim)

print(outputs[0][1:-1])
output_text = tokenizer.decode(outputs[0][0])
print(f"output: {output_text}")

plt.plot(cosine_similarities, marker='o')
plt.xlabel("Layer")
plt.ylabel("Cosine similarity to input embedding")
plt.show()

# the embeddingsare 1024, so just plot them as 32x32 grid like you do for attention heads