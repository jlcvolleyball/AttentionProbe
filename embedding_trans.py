from transformers import T5ForConditionalGeneration, T5Config, T5Tokenizer
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys
import numpy as np
from matplotlib.widgets import TextBox

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

# global variables
user_prompt = ""
repeated_word = ""
repeated_word_count = 0
repeated_word_locs = []

def display_embeddings():
    fig, axs = plt.subplots(2, 5, figsize=(50, 8))
    fig.suptitle(f"Input Sentence: {user_prompt}, Repeated Token: {repeated_word}", fontsize=16)
    for i in range(repeated_word_count):
        axs[i].set_title(f"{i}th occurrence of {repeated_word}")
        cur_embedding = encoder_hidden_states[0][0, token_idx, :]
        reshaped_original_embedding1 = cur_embedding.numpy().reshape((32, 32))
        im1 = axs[0].imshow(reshaped_original_embedding1)

def generate_count():
    global repeated_word_count, repeated_word_locs
    count = 0
    for idx, token in enumerate(tokens1):
        if repeated_word in token:
            count += 1
            repeated_word_locs.append(idx)
    repeated_word_count = count

def check_word_validity(word):
    for token in tokens1:
        if word in token: return True
    return False

def main():
    global user_prompt, repeated_word
    user_prompt = input("My prompt: ")
    repeated_word = input("The repeated token: ")
    while not check_word_validity(repeated_word):
        print(f"Your repeated token, '{repeated_word}', is not a valid token. Please try again.")
        repeated_word = input("The repeated token: ")
    generate_count()
    display_embeddings()

if __name__ == '__main__':
    main()