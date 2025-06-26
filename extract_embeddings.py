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

# global variables
mode = None

# for embedding matrix visualizations
fig, axs = None, None
im1, im2 = None, None
cb1, cb2 = None, None
embedding1_layer_ax, embedding2_layer_ax, embedding1_layer, embedding2_layer = None, None, None, None
token_num_label_ax, token_num_label = None, None
emb1_layer_idx, emb2_layer_idx = 0, 0
token_idx = 0
min_embedding1_label_ax, min_embedding1_label, max_embedding1_label_ax, max_embedding1_label = None, None, None, None
min_embedding2_label_ax, min_embedding2_label, max_embedding2_label_ax, max_embedding2_label = None, None, None, None
max_embedding1, max_embedding2 = -float("inf"), -float("inf")
min_embedding1, min_embedding2 = float("inf"), float("inf")

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
def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def plot_embedding(plot_idx, layer_idx):
    global emb1_layer_idx, emb2_layer_idx
    if plot_idx == 0:
        emb1_layer_idx = layer_idx
    if plot_idx == 1:
        emb2_layer_idx = layer_idx
    matrix_embedding_visualizations()

def submit_emb2_idx(text):
    if not text.isdigit() or int(text) > 23:
        print("Error: You entered an invalid layer number. Program exit")
        return
    cur_layer_idx = int(text)
    plot_embedding(1, cur_layer_idx)

def submit_emb1_idx(text):
    if not text.isdigit() or int(text) > 23:
        print("Error: You entered an invalid layer number. Program exit")
        return
    cur_layer_idx = int(text)
    plot_embedding(0, cur_layer_idx)

def submit_token_num(text):
    global token_idx
    if not text.isdigit() or int(text) > seq_len-1:
        print("Error: You entered an invalid token number. Program exit")
        return
    cur_token_idx = int(text)
    token_idx = cur_token_idx
    matrix_embedding_visualizations()

def submit_emb1_min(text):
    global min_embedding1, im1
    if not is_float(text):
        print("Error: You entered an invalid range for min. Program exit")
        return
    min_embedding1 = round(float(text), 2)
    im1.set_clim(min_embedding1, max_embedding1)

def submit_emb1_max(text):
    global max_embedding1, im1
    if not is_float(text):
        print("Error: You entered an invalid range for max. Program exit")
        return
    max_embedding1 = round(float(text), 2)
    im1.set_clim(min_embedding1, max_embedding1)

def submit_emb2_min(text):
    global min_embedding2, im2
    if not is_float(text):
        print("Error: You entered an invalid range for min. Program exit")
        return
    min_embedding2 = round(float(text), 2)
    im2.set_clim(min_embedding2, max_embedding2)

def submit_emb2_max(text):
    global max_embedding2, im2
    if not is_float(text):
        print("Error: You entered an invalid range for max. Program exit")
        return
    max_embedding2 = round(float(text), 2)
    im2.set_clim(min_embedding2, max_embedding2)

def matrix_embedding_visualizations():
    global fig, axs
    global embedding1_layer_ax, embedding2_layer_ax, embedding1_layer, embedding2_layer
    global im1, im2
    global cb1, cb2
    global token_num_label_ax, token_num_label
    global min_embedding1_label_ax, min_embedding1_label, max_embedding1_label_ax, max_embedding1_label
    global min_embedding2_label_ax, min_embedding2_label, max_embedding2_label_ax, max_embedding2_label
    global max_embedding1, max_embedding2, min_embedding1, min_embedding2

    if cb1 is not None: cb1.remove()
    if cb2 is not None: cb2.remove()

    # default token index for now:
    if fig is None and axs is None:
        fig, axs = plt.subplots(1, 2, figsize=(50, 8))
        axs[0].plot([1, 2, 3], [4, 5, 6])
        axs[0].set_title("Embedding 1 Plot")
        axs[1].plot([1, 2, 3], [6, 5, 4])
        axs[1].set_title("Embedding 2 Plot")

        embedding1_layer_ax = fig.add_axes([0.30, 0.87, 0.05, 0.05])
        embedding2_layer_ax = fig.add_axes([0.70, 0.87, 0.05, 0.05])
        embedding1_layer = TextBox(embedding1_layer_ax, label='Layer ', initial="0")
        embedding2_layer = TextBox(embedding2_layer_ax, label='Layer ', initial="0")
        embedding1_layer.label.set_fontsize(16)
        embedding1_layer.text_disp.set_fontsize(16)
        embedding2_layer.label.set_fontsize(16)
        embedding2_layer.text_disp.set_fontsize(16)
        embedding1_layer.on_submit(submit_emb1_idx)
        embedding2_layer.on_submit(submit_emb2_idx)

        token_num_label_ax = fig.add_axes([0.55, 0.03, 0.05, 0.05])
        token_num_label = TextBox(token_num_label_ax, label='Token Number ', initial="0")
        token_num_label.label.set_fontsize(16)
        token_num_label.text_disp.set_fontsize(16)
        token_num_label.on_submit(submit_token_num)

        min_embedding1_label_ax = fig.add_axes([0.20, 0.15, 0.05, 0.05])
        min_embedding1_label = TextBox(min_embedding1_label_ax, label='Min ', initial="0")
        max_embedding1_label_ax = fig.add_axes([0.30, 0.15, 0.05, 0.05])
        max_embedding1_label = TextBox(max_embedding1_label_ax, label='Max ', initial="0")
        min_embedding1_label.on_submit(submit_emb1_min)
        max_embedding1_label.on_submit(submit_emb1_max)

        min_embedding2_label_ax = fig.add_axes([0.60, 0.15, 0.05, 0.05])
        min_embedding2_label = TextBox(min_embedding2_label_ax, label='Min ', initial="0")
        max_embedding2_label_ax = fig.add_axes([0.70, 0.15, 0.05, 0.05])
        max_embedding2_label = TextBox(max_embedding2_label_ax, label='Max ', initial="0")
        min_embedding2_label.on_submit(submit_emb2_min)
        max_embedding2_label.on_submit(submit_emb2_max)

    # updating title
    fig.suptitle(f"Current Token: {tokens1[token_idx]}", fontsize=16)

    # initialize to original embeddings
    original_embedding1 = encoder_hidden_states[emb1_layer_idx][0, token_idx, :]
    reshaped_original_embedding1 = original_embedding1.numpy().reshape((32, 32))
    im1 = axs[0].imshow(reshaped_original_embedding1)
    cb1 = fig.colorbar(im1, ax=axs[0], shrink=0.7, pad=0.1)
    original_embedding2 = encoder_hidden_states[emb2_layer_idx][0, token_idx, :]
    reshaped_original_embedding2 = original_embedding2.numpy().reshape((32, 32))
    im2 = axs[1].imshow(reshaped_original_embedding2)
    cb2 = fig.colorbar(im2, ax=axs[1], shrink=0.7, pad=0.1)

    # set the initial values for min and max labels
    min_embedding1, max_embedding1 = im1.get_clim()
    min_embedding2, max_embedding2 = im2.get_clim()
    min_embedding1_label.set_val(str(round(min_embedding1, 2)))
    max_embedding1_label.set_val(str(round(max_embedding1, 2)))
    min_embedding2_label.set_val(str(round(min_embedding2, 2)))
    max_embedding2_label.set_val(str(round(max_embedding2, 2)))

    plt.show()

def cosine_sim_lineplot():
    all_sims = []
    for cur_token in range(seq_len):
        original_embedding = encoder_hidden_states[0][0, cur_token, :]
        cosine_similarities = []
        for layer in encoder_hidden_states:
            cur_embedding = layer[0, cur_token, :]
            # print(f"Current embedding: {cur_embedding}")
            sim = F.cosine_similarity(original_embedding, cur_embedding, dim=0).item()
            cosine_similarities.append(sim)
        all_sims.append(cosine_similarities)

    print(outputs[0][1:-1])
    output_text = tokenizer.decode(outputs[0][0])
    print(f"output: {output_text}")

    for idx, cos_sim in enumerate(all_sims):
        plt.plot(cos_sim, marker='o', label=tokens1[idx])
    plt.xlabel("Layer")
    plt.ylabel("Cosine similarity to input embedding")
    plt.legend()
    plt.show()

def matrix_cosine_sim_visualization():
    all_sims = []
    cur_token = 2
    for layer_i in encoder_hidden_states:
        layer_i_embedding = layer_i[0, cur_token, :]
        layer_ij_sims = []
        for layer_j in encoder_hidden_states:
            layer_j_embedding = layer_j[0, cur_token, :]
            sim = F.cosine_similarity(layer_i_embedding, layer_j_embedding, dim=0).item()
            layer_ij_sims.append(sim)
        all_sims.append(layer_ij_sims)
    all_sims_np = np.array(all_sims)
    im = plt.imshow(all_sims_np)
    plt.colorbar()
    plt.show()

def main():
    global mode
    if len(sys.argv) == 3:
        flag = sys.argv[1]
        if flag == "-mode":
            mode = int(sys.argv[2])
    if mode == 0:
        cosine_sim_lineplot()
    if mode == 1:
        matrix_embedding_visualizations()
    if mode == 2:
        matrix_cosine_sim_visualization()

if __name__ == '__main__':
    main()

# the embeddingsare 1024, so just plot them as 32x32 grid like you do for attention heads