from transformers import T5ForConditionalGeneration, T5Config, T5Tokenizer
import torch
import sys
import matplotlib.pyplot as plt
import numpy as np

# model setup
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
config = T5Config.from_pretrained("google/flan-t5-large")
# model.eval()

def find_difference(t1, t2):
    word_diff = -1
    pointer = 0
    while(pointer < len(t1)):
        if t1[pointer] != t2[pointer]: word_diff = pointer
        pointer+=1
    return word_diff

# input and tokenizing
# "The man gave the woman his jacket. Who owned the jacket, the man or the woman?"
input_text1 = "The man showed the woman his jacket. Who owned the jacket, the man or the woman?"
input_text2 = "The man showed the woman her jacket. Who owned the jacket, the man or the woman?"
inputs = tokenizer([input_text1, input_text2],
                   padding=True,
                   return_tensors="pt",
                   return_attention_mask=True)
inputs_ids = inputs.input_ids
tokens1 = tokenizer.convert_ids_to_tokens(inputs_ids[0])
tokens2 = tokenizer.convert_ids_to_tokens(inputs_ids[1])
print(tokens1)

# REQ: tokens1 and tokens2 differ by one word, and are the same length.
diff_idx = find_difference(tokens1, tokens2)
diff_t1 = tokens1[diff_idx]
diff_t2 = tokens2[diff_idx]
tokens3 = [tokens1[i] if i!=diff_idx else diff_t1 + "/" + diff_t2 for i in range(len(tokens1))]

print(tokens1)
print(tokens2)

# call generate through the model, output_attentions=True
outputs = model.generate(
    input_ids=inputs_ids,
    attention_mask=inputs.attention_mask,
    output_attentions=True,
    return_dict_in_generate=True
)

#set up global variables
cur_layer_idx = 0
cur_head_idx = 0
cur_layer_attentions = outputs.encoder_attentions[cur_layer_idx]
num_heads_per_layer = cur_layer_attentions.shape[1]
total_num_layers = config.num_layers
fig, axs = None, None
cb1, cb2, cb3 = None, None, None
tooltips = {}  #setup tooltip

def text_colorchange(ax):
    y_labels = ax.get_yticklabels()
    if diff_idx < len(y_labels):
        y_labels[diff_idx].set_color('red')
        y_labels[diff_idx].set_fontweight('bold')
    x_labels = ax.get_xticklabels()
    if diff_idx < len(x_labels):
        x_labels[diff_idx].set_color('red')
        x_labels[diff_idx].set_fontweight('bold')
    ax.figure.canvas.draw()


def plot_attention_head(head_idx):
    global fig, axs, cur_layer_attentions, cb1, cb2, cb3, tooltips
    cur_layer_attentions = outputs.encoder_attentions[cur_layer_idx]
    if cb1 is not None: cb1.remove()
    if cb2 is not None: cb2.remove()
    if cb3 is not None: cb3.remove()
    if fig is None or axs is None:
        fig, axs = plt.subplots(1, 3, figsize=(50, 8))
    fig.subplots_adjust(
        left=0.05, right=0.95,
        top=0.9, bottom=0.1,
        wspace=0.3
    )
    fig.suptitle(f"Layer {cur_layer_idx} - Head {head_idx}", fontsize=16)

    # clear each axis
    for ax in axs:
        ax.clear()
    ax1 = axs[0]
    ax2 = axs[1]
    ax3 = axs[2]

    #extract attention for sentence 1
    cur_head1 = cur_layer_attentions[0, cur_head_idx, :, :]
    attention_tokens1 = cur_layer_attentions.shape[2]
    a1 = cur_head1.numpy().reshape(attention_tokens1, attention_tokens1)
    ax1.set_xticks(np.arange(len(tokens1)))
    ax1.set_yticks(np.arange(len(tokens1)))
    ax1.set_xticklabels(tokens1, rotation=90)
    ax1.set_yticklabels(tokens1)
    im1 = ax1.imshow(a1)
    ax1.set_title("Sentence 1 Attention")
    # cb1 = fig.colorbar(im1, ax=ax1, orientation='horizontal', shrink=0.6)
    cb1 = fig.colorbar(im1, ax=ax1, shrink=0.4, pad=0.1)
    # cb1.clim(0.2, 0.8)
    #plt.clim([-0.05,.08])

    #extract attention for sentence 2
    cur_head2 = cur_layer_attentions[1, cur_head_idx, :, :]
    attention_tokens2 = cur_layer_attentions.shape[2]
    a2 = cur_head2.numpy().reshape(attention_tokens2, attention_tokens2)
    im2 = ax2.imshow(a2)
    ax2.set_xticks(np.arange(len(tokens2)))
    ax2.set_yticks(np.arange(len(tokens2)))
    ax2.set_xticklabels(tokens2, rotation=90)
    ax2.set_yticklabels(tokens2)
    ax2.set_title("Sentence 2 Attention")
    # cb2 = fig.colorbar(im2, ax=ax2, orientation='horizontal', location='top', pad=0.1)
    cb2 = fig.colorbar(im2, ax=ax2, shrink=0.4, pad=0.1)

    #compute the difference of these attentions
    diff = a1 - a2
    im_diff = ax3.imshow(diff)
    ax3.set_xticks(np.arange(len(tokens3)))
    ax3.set_yticks(np.arange(len(tokens3)))
    ax3.set_xticklabels(tokens3, rotation=90)
    ax3.set_yticklabels(tokens3)
    ax3.set_title("Difference")
    # cb3 = fig.colorbar(im_diff, ax=ax3, orientation='horizontal', shrink=0.6)
    cb3 = fig.colorbar(im_diff, ax=ax3, shrink=0.4, pad=0.1)

    #initialize tooltip
    for ax in axs:
        annotation = ax.annotate(
            "", xy=(0, 0), xytext=(-60, 10), textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->", color="white"),
            zorder=100
        )
        ax.title.set_zorder(1)
        annotation.set_zorder(100)
        annotation.set_visible(False)
        tooltips[ax] = annotation

    text_colorchange(ax1)
    text_colorchange(ax2)
    text_colorchange(ax3)

    fig.canvas.draw_idle()

def on_hover(event):
    if event.inaxes is None:
        for tooltip in tooltips.values():
            if tooltip.get_visible():
                tooltip.set_visible(False)
        fig.canvas.draw_idle()
        return

    hovered_ax = event.inaxes
    if hovered_ax not in axs: return #if it's not on the plot don't do anything
    x_pos = int(np.floor(event.xdata))
    y_pos = int(np.floor(event.ydata))

    attentions = None
    tokens_x = None
    tokens_y = None
    if(hovered_ax == axs[0]):
        attentions = cur_layer_attentions[0, cur_head_idx, :, :].numpy()
        tokens_x = tokens1
        tokens_y = tokens1
    elif(hovered_ax == axs[1]):
        attentions = cur_layer_attentions[1, cur_head_idx, :, :].numpy()
        tokens_x = tokens2
        tokens_y = tokens2
    else: #must be hovering over the third axis
        attentions = cur_layer_attentions[0, cur_head_idx, :, :].numpy() - cur_layer_attentions[1, cur_head_idx, :, :].numpy()
        tokens_x = tokens3
        tokens_y = tokens3

    tooltip = tooltips[hovered_ax]

    if 0 <= x_pos < attentions.shape[1] and 0 <= y_pos < attentions.shape[0]: #shape[0] is num rows, shape[1] is num cols
        cur_attention = attentions[y_pos, x_pos]
        tooltip.xy = (x_pos, y_pos)
        tooltip.set_text(f"input: {tokens_y[y_pos]}\noutput: {tokens_x[x_pos]}\nactivation: {cur_attention:.3f}")
        tooltip.set_visible(True)
        fig.canvas.draw_idle()
    else:
        tooltip.set_visible(False)

    for ax, other_tooltip in tooltips.items():
        if ax!=hovered_ax and other_tooltip.get_visible():
            other_tooltip.set_visible(False)

    fig.canvas.draw_idle()


def next_attention_head(event):
    global cur_head_idx
    global cur_layer_idx
    if event.key == 'right':
        if cur_layer_idx == total_num_layers-1 and cur_head_idx == num_heads_per_layer-1:
            cur_layer_idx = 0
            cur_head_idx = 0
        elif cur_head_idx == num_heads_per_layer - 1:
            cur_head_idx = 0
            cur_layer_idx += 1
        else:
            cur_head_idx += 1
    elif event.key == 'left':
        if cur_layer_idx == 0 and cur_head_idx == 0:
            cur_layer_idx = total_num_layers - 1
            cur_head_idx = num_heads_per_layer - 1
        elif cur_head_idx == 0:
            cur_head_idx = num_heads_per_layer-1
            cur_layer_idx -= 1
        else:
            cur_head_idx -= 1
    else:
        return
    plot_attention_head(cur_head_idx)

def main():
    # Initial plot
    plot_attention_head(cur_head_idx)
    fig.canvas.mpl_connect('key_press_event', next_attention_head)
    fig.canvas.mpl_connect('motion_notify_event', on_hover)
    plt.show()

if __name__ == '__main__':
   main()