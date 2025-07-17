from transformers import T5ForConditionalGeneration, T5Config, T5Tokenizer
import torch
import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from matplotlib.widgets import TextBox

# model setup
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
config = T5Config.from_pretrained("google/flan-t5-large")

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

# get the output of the model
output1 = tokenizer.decode(outputs[0][0], skip_special_tokens=True)
output2 = tokenizer.decode(outputs[0][1], skip_special_tokens=True)

#set up global variables
cur_layer_idx = 0
cur_head_idx = 0
cur_layer_attentions = outputs.encoder_attentions[cur_layer_idx]
num_heads_per_layer = cur_layer_attentions.shape[1]
total_num_layers = config.num_layers
fig, axs = None, None
cb1, cb2, cb3 = None, None, None
tooltips = {}  #setup tooltip
range_slider_ax, range_slider = None, None
layer_textbox_ax, layer_textbox, head_textbox_ax, head_textbox = None, None, None, None
im1, im2, im_diff = None, None, None
original_range_im1, original_range_im2, original_range_imdiff = None, None, None
all_lines1, all_lines2, all_lines3 = [], [], []
hovered_lines = []
output_text1_ax, output_text2_ax = None, None

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

def compute_new_range(im, percent):
    cur_min, cur_max = None, None
    if(im == im1): cur_min, cur_max = original_range_im1
    elif(im == im2): cur_min, cur_max = original_range_im2
    else: cur_min, cur_max = original_range_imdiff

    new_min, new_max = None, None
    if abs(cur_min) < abs(cur_max): #we want to keep the min the same, shrink the max
        new_total_range = (cur_max - cur_min) * percent
        new_max = cur_min + new_total_range
        new_min = cur_min
    else: #we want to keep the max the same, shrink the min
        new_total_range = (cur_max - cur_min) * percent
        new_min = cur_max - new_total_range
        new_max = cur_max
    im.set_clim(new_min, new_max)

def slider_update(val):
    global im1, im2, im_diff
    new_percent = val
    #compute the new min and max
    compute_new_range(im1, new_percent)
    compute_new_range(im2, new_percent)
    compute_new_range(im_diff, new_percent)
    fig.canvas.draw_idle()

def init_slider(fig):
    global range_slider_ax, range_slider
    range_slider_ax = fig.add_axes([0.1, 0.03, 0.2, 0.03])
    range_slider = Slider(range_slider_ax, 'Adjust Range', 0.0, 1.0, valinit=1.0)
    range_slider.poly.set_alpha(0.0)
    range_slider.on_changed(slider_update)

def submit_layeridx(text):
    global cur_head_idx, cur_layer_idx
    if not text.isdigit():
        print("Error: You entered an invalid layer number. Program exit")
        plt.close()
        exit(1)
    cur_layer_idx = int(text)
    plot_attention_head(cur_head_idx)
    range_slider.reset()
    # print(f"Layer: {cur_layer_idx} - Head: {cur_head_idx}")

def submit_headidx(text):
    global cur_head_idx, cur_layer_idx
    if not text.isdigit():
        print("Error: You entered an invalid layer number. Program exit")
        plt.close()
        exit(1)
    cur_head_idx = int(text)
    plot_attention_head(cur_head_idx)
    range_slider.reset()
    # print(f"Layer: {cur_layer_idx} - Head: {cur_head_idx}")

def init_text_boxes(fig):
    global layer_textbox_ax, layer_textbox, head_textbox_ax, head_textbox
    layer_textbox_ax = fig.add_axes([0.45, 0.925, 0.05, 0.05])
    head_textbox_ax = fig.add_axes([0.55, 0.925, 0.05, 0.05])
    layer_textbox = TextBox(layer_textbox_ax, label='Layer ', initial=str(cur_layer_idx))
    head_textbox = TextBox(head_textbox_ax, label='Head ', initial=str(cur_head_idx))
    layer_textbox.on_submit(submit_layeridx)
    head_textbox.on_submit(submit_headidx)
    layer_textbox.label.set_fontsize(16)
    layer_textbox.text_disp.set_fontsize(16)
    head_textbox.label.set_fontsize(16)
    head_textbox.text_disp.set_fontsize(16)

def init_tooltip(axs, tooltips):
    # initialize tooltip
    for ax in axs.flatten():
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

def draw_line_diff(ax, i, j, attention, spacing):
    # print(f"Attention: {attention}")
    y1 = 1 - (i + 0.6) * spacing
    y2 = 1 - (j + 0.6) * spacing
    cur_attention = attention
    alpha = abs(cur_attention)
    if cur_attention < 0:
        return ax.plot(
            [0, 1],
            [y1, y2],
            transform=ax.transAxes,
            color='red',
            alpha=alpha,
            linewidth=1
        )[0]
    else:
        return ax.plot(
            [0, 1],
            [y1, y2],
            color='green',
            transform=ax.transAxes,
            alpha=alpha,
            linewidth=1
        )[0]

def draw_line_prompts(ax, i, j, attention, spacing, color):
    y1 = 1 - (i + 0.6) * spacing
    y2 = 1 - (j + 0.6) * spacing
    cur_attention = attention
    alpha = cur_attention
    return ax.plot(
                    [0, 1],
                    [y1, y2],
                    color=color,
                    transform=ax.transAxes,
                    alpha=alpha,
                    linewidth=1
                )[0]

def compute_tokenbounds(tokens, spacing):
    token_bounds = []
    for i in range(len(tokens)):
        y_center =  1 - (i + 0.6) * spacing
        height = spacing * 0.9
        ymin = y_center - height / 2
        ymax = y_center + height / 2
        token_bounds.append((ymin, ymax))
    return token_bounds

def init_linevisualizations(a1, a2, ax4, ax5, ax6, diff):
    global all_lines1, all_lines2, all_lines3
    spacing1 = 1 / len(tokens1)
    for i, token in enumerate(tokens1):
        y = 1 - (i + 0.6) * spacing1
        if i != diff_idx:
            ax4.text(0, y, token, ha='right', va='center', fontsize=10, transform=ax4.transAxes)
            ax4.text(1, y, token, ha='left', va='center', fontsize=10, transform=ax4.transAxes)
        else:
            ax4.text(0, y, token, ha='right', va='center', fontsize=10, color='red', weight='bold',
                     transform=ax4.transAxes)
            ax4.text(1, y, token, ha='left', va='center', fontsize=10, color='red', weight='bold',
                     transform=ax4.transAxes)
    for i in range(len(tokens1)):
        for j in range(len(tokens1)):
            cur_attention = a1[i, j]
            if cur_attention > 0.01:
                line = draw_line_prompts(ax4, i, j, cur_attention, spacing1, 'blue')
                all_lines1.append(line)
    ax4.axis("off")

    spacing2 = 1 / len(tokens2)
    for i, token in enumerate(tokens2):
        # y = 1 - i / len(tokens2)
        y = 1 - (i + 0.6) * spacing2
        if i != diff_idx:
            ax5.text(0, y, token, ha='right', va='center', fontsize=10, transform=ax5.transAxes)
            ax5.text(1, y, token, ha='left', va='center', fontsize=10, transform=ax5.transAxes)
        else:
            ax5.text(0, y, token, ha='right', va='center', fontsize=10, color='red', weight='bold',
                     transform=ax5.transAxes)
            ax5.text(1, y, token, ha='left', va='center', fontsize=10, color='red', weight='bold',
                     transform=ax5.transAxes)
    for i in range(len(tokens2)):
        for j in range(len(tokens2)):
            cur_attention = a2[i, j]
            if cur_attention > 0.01:
                line = draw_line_prompts(ax5, i, j, cur_attention, spacing2, 'blue')
                all_lines2.append(line)
    ax5.axis("off")

    spacing3 = 1 / len(tokens3)
    for i, token in enumerate(tokens3):
        y = 1 - (i + 0.6) * spacing3
        if i != diff_idx:
            ax6.text(0, y, token, ha='right', va='center', fontsize=10, transform=ax6.transAxes)
            ax6.text(1, y, token, ha='left', va='center', fontsize=10, transform=ax6.transAxes)
        else:
            ax6.text(0, y, token, ha='right', va='center', fontsize=10, color='red', weight='bold',
                     transform=ax6.transAxes)
            ax6.text(1, y, token, ha='left', va='center', fontsize=10, color='red', weight='bold',
                     transform=ax6.transAxes)
    for i in range(len(tokens3)):
        for j in range(len(tokens3)):
            cur_attention = diff[i, j]
            if abs(cur_attention) > 0.01:
                line = draw_line_diff(ax6, i, j, cur_attention, spacing3)
                all_lines3.append(line)
    ax6.axis("off")

def init_matrixvisualizations(ax1, ax2, ax3, cur_layer_attentions, fig):
    global im1, cb1, original_range_im1, im2, cb2, original_range_im2, im_diff, cb3, original_range_imdiff
    # extract attention for sentence 1
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
    cb1 = fig.colorbar(im1, ax=ax1, shrink=0.8, pad=0.1)
    original_range_im1 = im1.get_clim()
    # cb1.clim(0.2, 0.8)
    # plt.clim([-0.05,.08])
    # extract attention for sentence 2
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
    cb2 = fig.colorbar(im2, ax=ax2, shrink=0.8, pad=0.1)
    original_range_im2 = im2.get_clim()
    # compute the difference of these attentions
    diff = a1 - a2
    im_diff = ax3.imshow(diff)
    ax3.set_xticks(np.arange(len(tokens3)))
    ax3.set_yticks(np.arange(len(tokens3)))
    ax3.set_xticklabels(tokens3, rotation=90)
    ax3.set_yticklabels(tokens3)
    ax3.set_title("Difference")
    # cb3 = fig.colorbar(im_diff, ax=ax3, orientation='horizontal', shrink=0.6)
    cb3 = fig.colorbar(im_diff, ax=ax3, shrink=0.8, pad=0.1)
    original_range_imdiff = im_diff.get_clim()
    return a1, a2, diff

def plot_attention_head(head_idx):
    global fig, axs, cur_layer_attentions, cb1, cb2, cb3, tooltips, im1, im2, im_diff
    global range_slider_ax, range_slider
    global original_range_im1, original_range_im2, original_range_imdiff
    global output_text1_ax, output_text2_ax

    cur_layer_attentions = outputs.encoder_attentions[cur_layer_idx]
    if cb1 is not None: cb1.remove()
    if cb2 is not None: cb2.remove()
    if cb3 is not None: cb3.remove()
    if fig is None or axs is None:
        fig, axs = plt.subplots(2, 3, figsize=(50, 8))
        output_text1_ax = fig.add_axes([0.05, 0.91, 0.4, 0.1])
        output_text1_ax.axis("off")
        output_text1_ax.text(0.0, 0.5, f"Output 1: {output1}", fontsize=12, va="center", ha="left")
        output_text2_ax = fig.add_axes([0.05, 0.89, 0.4, 0.1])
        output_text2_ax.axis("off")
        output_text2_ax.text(0.0, 0.5, f"Output 2: {output2}", fontsize=12, va="center", ha="left")

    fig.subplots_adjust(
        left=0.075, right=0.925,
        top=0.9, bottom=0.1,
        wspace=0.5
    )
    # fig.suptitle(f"Layer {cur_layer_idx} - Head {head_idx}", fontsize=16)

    if range_slider is None:
        init_slider(fig)

    if head_textbox is None and layer_textbox is None:
        init_text_boxes(fig)

    # clear each axis
    for ax in axs.flatten():
        ax.clear()
    ax1 = axs[0, 0] # ax1, 2, 3 are the activations in a "grid" format
    ax2 = axs[0, 1]
    ax3 = axs[0, 2]
    ax4 = axs[1, 0] # ax4, 5, 6 are the activations in a "line" format
    ax5 = axs[1, 1]
    ax6 = axs[1, 2]

    # for matrix visualizations:
    a1, a2, diff = init_matrixvisualizations(ax1, ax2, ax3, cur_layer_attentions, fig)

    # for line visualizations:
    init_linevisualizations(a1, a2, ax4, ax5, ax6, diff)

    init_tooltip(axs, tooltips)

    text_colorchange(ax1)
    text_colorchange(ax2)
    text_colorchange(ax3)

    fig.canvas.draw_idle()

def reset_lines(hovered_lines):
    for line in all_lines1:
        line.set_visible(True)
    for line in all_lines2:
        line.set_visible(True)
    for line in all_lines3:
        line.set_visible(True)
    for line in hovered_lines:
        line.remove()
    hovered_lines.clear()
    fig.canvas.draw_idle()

def on_unhover(event):
    global hovered_lines
    if not hovered_lines == []:
        reset_lines(hovered_lines)

def click_linevisualizations(event):
    global hovered_lines
    hovered_ax = event.inaxes
    if hovered_ax not in axs:
        reset_lines(hovered_lines)
        return
    # check if the hovered axes are correct, handle accordingly
    if hovered_ax == axs[1, 0]:
        attentions = cur_layer_attentions[0, cur_head_idx, :, :].numpy()
        tokens = tokens1
    elif hovered_ax == axs[1, 1]:
        attentions = cur_layer_attentions[1, cur_head_idx, :, :].numpy()
        tokens = tokens2
    elif hovered_ax == axs[1, 2]:
        attentions = cur_layer_attentions[0, cur_head_idx, :, :].numpy() - cur_layer_attentions[1, cur_head_idx, :, :].numpy()
        tokens = tokens3
    else:
        reset_lines(hovered_lines)
        return

    token_bounds = compute_tokenbounds(tokens, spacing=1 / len(tokens))
    x, y = event.x, event.y
    inv = hovered_ax.transAxes.inverted()
    x_axes, y_axes = inv.transform((x, y))

    hovered_token = None
    for i, (ymin, ymax) in enumerate(token_bounds):
        if ymin <= y_axes <= ymax:
            hovered_token = i
            break

    if hovered_token is None: return

    if hovered_ax == axs[1, 0]:
        for line in all_lines1:
            line.set_visible(False)
    elif hovered_ax == axs[1, 1]:
        for line in all_lines2:
            line.set_visible(False)
    elif hovered_ax == axs[1, 2]:
        for line in all_lines3:
            line.set_visible(False)

    if hovered_lines:
        for line in hovered_lines:
            line.remove()
        hovered_lines.clear()

    if x_axes < 0.5:
        for j in range(len(tokens)):
            if abs(attentions[hovered_token, j]) > 0.01:
                if hovered_ax == axs[1, 0]:
                    new_line = draw_line_prompts(hovered_ax, hovered_token, j, attentions[hovered_token, j], spacing=1 / len(tokens), color="blue")
                elif hovered_ax == axs[1, 1]:
                    new_line = draw_line_prompts(hovered_ax, hovered_token, j, attentions[hovered_token, j], spacing=1 / len(tokens), color="blue")
                if hovered_ax == axs[1, 2]:
                    new_line = draw_line_diff(hovered_ax, hovered_token, j, attentions[hovered_token, j], spacing=1 / len(tokens))
                hovered_lines.append(new_line)
    else:
        for i in range(len(tokens)):
            if abs(attentions[i, hovered_token]) > 0.01:
                if hovered_ax == axs[1, 0]:
                    new_line = draw_line_prompts(hovered_ax, i, hovered_token, attentions[i, hovered_token], spacing=1 / len(tokens), color="blue")
                elif hovered_ax == axs[1, 1]:
                    new_line = draw_line_prompts(hovered_ax, i, hovered_token, attentions[i, hovered_token], spacing=1 / len(tokens), color="blue")
                if hovered_ax == axs[1, 2]:
                    new_line = draw_line_diff(hovered_ax, i, hovered_token, attentions[i, hovered_token], spacing=1 / len(tokens))
                hovered_lines.append(new_line)
    fig.canvas.draw_idle()

def on_hover(event):
    if event.inaxes is None:
        for tooltip in tooltips.values():
            if tooltip.get_visible():
                tooltip.set_visible(False)
        fig.canvas.draw_idle()
        return

    hovered_ax = event.inaxes
    if hovered_ax not in axs: return
    x_pos = round(event.xdata) #int(np.floor(event.xdata))
    y_pos = round(event.ydata)
    # print(f"event.xdata: {event.xdata}, event.ydata: {event.ydata}")
    # print(f"x: {x_pos}, y: {y_pos}")

    attentions = None
    tokens_x = None
    tokens_y = None
    if(hovered_ax == axs[0, 0]):
        attentions = cur_layer_attentions[0, cur_head_idx, :, :].numpy()
        tokens_x = tokens1
        tokens_y = tokens1
    elif(hovered_ax == axs[0, 1]):
        attentions = cur_layer_attentions[1, cur_head_idx, :, :].numpy()
        tokens_x = tokens2
        tokens_y = tokens2
    elif (hovered_ax == axs[0, 2]):
        attentions = cur_layer_attentions[0, cur_head_idx, :, :].numpy() - cur_layer_attentions[1, cur_head_idx, :, :].numpy()
        tokens_x = tokens3
        tokens_y = tokens3
    else:
        click_linevisualizations(event)
        return

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
    global range_slider
    global cur_head_idx
    global cur_layer_idx
    global head_textbox, layer_textbox
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
    range_slider.reset()
    layer_textbox.set_val(str(cur_layer_idx))
    head_textbox.set_val(str(cur_head_idx))
    # print(f"Layer: {cur_layer_idx} - Head: {cur_head_idx}")

def parse_args():
    global cur_head_idx, cur_layer_idx
    if len(sys.argv) == 2:
        desired_layer = int(sys.argv[1])
        if desired_layer >= total_num_layers or desired_layer < 0:
            print("You have entered an invalid layer number. Please try again.")
            exit()
        cur_layer_idx = desired_layer
    elif len(sys.argv) == 3:
        desired_layer = int(sys.argv[1])
        desired_head = int(sys.argv[2])
        if desired_layer >= total_num_layers or desired_head >= num_heads_per_layer\
                or desired_layer < 0 or desired_head < 0:
            print("You have entered an invalid layer number or an invalid attention head number. Please try again.")
            exit()
        cur_layer_idx = desired_layer
        cur_head_idx = desired_head

def main():
    # Initial plot
    parse_args()
    plot_attention_head(cur_head_idx)
    fig.canvas.mpl_connect('key_press_event', next_attention_head)
    fig.canvas.mpl_connect('motion_notify_event', on_hover)
    fig.canvas.mpl_connect("axes_leave_event", on_unhover)
    # fig.canvas.mpl_connect("button_press_event", click_linevisualizations)
    plt.show()

if __name__ == '__main__':
   main()