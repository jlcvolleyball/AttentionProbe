from transformers import T5ForConditionalGeneration, T5Config, T5Tokenizer
import torch
import sys
import numpy as np

input_text1 = "The woman hit the thief with the stick. Who had the stick?"

if len(sys.argv) > 1:
    user_input = sys.argv[1]
    input_text1 = user_input

# model setup
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
config = T5Config.from_pretrained("google/flan-t5-large")

inputs = tokenizer(input_text1,
                   padding=True,
                   return_tensors="pt",
                   return_attention_mask=True)
inputs_ids = inputs.input_ids

tokens1 = tokenizer.convert_ids_to_tokens(inputs_ids[0])
print(f"These are the input IDS: {inputs_ids[0]}")
print(f"These are the tokens: {tokens1}")

