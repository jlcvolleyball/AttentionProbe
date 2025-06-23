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

encoder_embedding = model.encoder.embed_tokens
decoder_embedding = model.decoder.embed_tokens

my_input = "The jacket"
inputs = tokenizer(my_input, return_tensors="pt")
input_ids = inputs.input_ids

outputs = model.generate(
    input_ids=input_ids,
    attention_mask=inputs.attention_mask,
    output_hidden_states=True,
    return_dict_in_generate=True
)

tokens1 = tokenizer.convert_ids_to_tokens(input_ids[0])
print(f"These are the input IDS: {input_ids[0]}")
print(f"These are the tokens: {tokens1}")

encoder_hidden_states = outputs.encoder_hidden_states
decoder_hidden_states = outputs.decoder_hidden_states

input_embeddings = encoder_embedding(input_ids[0][1])
original_embedding = encoder_hidden_states[0][0, 1, :]