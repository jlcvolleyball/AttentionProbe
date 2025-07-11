"""
Attention visualization module for the AttentionProbe application.
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, TextBox
from typing import List, Tuple, Optional, Dict, Any

from config import UI_CONFIG
from utils import ModelManager, find_difference


class AttentionVisualizer:
    """Handles attention visualization for T5 models."""
    
    def __init__(self, model_manager: ModelManager, prompts: List[str]):
        """
        Initialize the attention visualizer.
        
        Args:
            model_manager: Model manager instance
            prompts: List of prompts to visualize
        """
        self.model_manager = model_manager
        self.prompts = prompts
        self.tokenizer = model_manager.tokenizer
        self.config = model_manager.config
        
        # Process inputs
        self._process_inputs()
        
        # Visualization state
        self.cur_layer_idx = 0
        self.cur_head_idx = 0
        self.cur_overall_idx = 0
        self.fig = None
        self.axs = None
        self.tooltips = {}
        self.highlight_indices = None
        
        # UI elements
        self.range_slider = None
        self.layer_textbox = None
        self.head_textbox = None
        
        # Visualization elements
        self.im1 = None
        self.im2 = None
        self.im_diff = None
        self.original_range_im1 = None
        self.original_range_im2 = None
        self.original_range_imdiff = None
        self.all_lines1 = []
        self.all_lines2 = []
        self.all_lines3 = []
        self.hovered_lines = []
        
    def _process_inputs(self):
        """Process input prompts and get model outputs."""
        # Get model outputs with attention
        self.outputs = self.model_manager.get_attention_outputs(
            self.prompts, 
            max_length=UI_CONFIG['max_generation_length']
        )
        
        # Tokenize inputs
        inputs = self.tokenizer(
            self.prompts,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True
        )
        
        self.inputs_ids = inputs.input_ids
        self.tokens1 = self.tokenizer.convert_ids_to_tokens(self.inputs_ids[0])
        self.tokens2 = self.tokenizer.convert_ids_to_tokens(self.inputs_ids[1])
        
        # Find difference between tokens
        self.diff_idx = find_difference(self.tokens1, self.tokens2)
        self.diff_t1 = self.tokens1[self.diff_idx]
        self.diff_t2 = self.tokens2[self.diff_idx]
        self.tokens3 = [
            self.tokens1[i] if i != self.diff_idx else f"{self.diff_t1}/{self.diff_t2}" 
            for i in range(len(self.tokens1))
        ]
        
        # Get attention data
        self.cur_layer_attentions = self.outputs.encoder_attentions[self.cur_layer_idx]
        self.num_heads_per_layer = self.cur_layer_attentions.shape[1]
        self.total_num_layers = self.config.num_layers
        
        print(f"Tokens 1: {self.tokens1}")
        print(f"Tokens 2: {self.tokens2}")
        
    def _compute_new_range(self, im, percent: float):
        """Compute new range for image normalization."""
        if im == self.im1:
            cur_min, cur_max = self.original_range_im1
        elif im == self.im2:
            cur_min, cur_max = self.original_range_im2
        else:
            cur_min, cur_max = self.original_range_imdiff
            
        if abs(cur_min) < abs(cur_max):
            new_total_range = (cur_max - cur_min) * percent
            new_max = cur_min + new_total_range
            new_min = cur_min
        else:
            new_total_range = (cur_max - cur_min) * percent
            new_min = cur_max - new_total_range
            new_max = cur_max
            
        im.set_clim(new_min, new_max)
        
    def _slider_update(self, val):
        """Update visualization based on slider value."""
        self._compute_new_range(self.im1, val)
        self._compute_new_range(self.im2, val)
        self._compute_new_range(self.im_diff, val)
        self.fig.canvas.draw_idle()
        
    def _init_slider(self):
        """Initialize the range slider."""
        range_slider_ax = self.fig.add_axes([0.1, 0.03, 0.2, 0.03])
        self.range_slider = Slider(
            range_slider_ax, 
            'Adjust Range', 
            UI_CONFIG['slider_range'][0], 
            UI_CONFIG['slider_range'][1], 
            valinit=UI_CONFIG['slider_default']
        )
        self.range_slider.poly.set_alpha(0.0)
        self.range_slider.on_changed(self._slider_update)
        
    def _submit_layeridx(self, text):
        """Handle layer index submission."""
        if not text.isdigit():
            print("Error: You entered an invalid layer number.")
            return
            
        self.cur_layer_idx = int(text)
        self._plot_attention_head(self.cur_head_idx)
        self.range_slider.reset()
        print(f"Layer: {self.cur_layer_idx} - Head: {self.cur_head_idx}")
        
    def _submit_headidx(self, text):
        """Handle head index submission."""
        if not text.isdigit():
            print("Error: You entered an invalid head number.")
            return
            
        self.cur_head_idx = int(text)
        self._plot_attention_head(self.cur_head_idx)
        self.range_slider.reset()
        print(f"Layer: {self.cur_layer_idx} - Head: {self.cur_head_idx}")
        
    def _init_text_boxes(self):
        """Initialize text boxes for layer and head selection."""
        layer_textbox_ax = self.fig.add_axes([0.45, 0.925, 0.05, 0.05])
        head_textbox_ax = self.fig.add_axes([0.55, 0.925, 0.05, 0.05])
        
        self.layer_textbox = TextBox(layer_textbox_ax, label='Layer ', initial="0")
        self.head_textbox = TextBox(head_textbox_ax, label='Head ', initial="0")
        
        self.layer_textbox.on_submit(self._submit_layeridx)
        self.head_textbox.on_submit(self._submit_headidx)
        
        # Set font sizes
        for textbox in [self.layer_textbox, self.head_textbox]:
            textbox.label.set_fontsize(16)
            textbox.text_disp.set_fontsize(16)
            
    def _init_tooltip(self):
        """Initialize tooltips for the visualization."""
        for ax in self.axs.flatten():
            annotation = ax.annotate(
                "", xy=(0, 0), xytext=(-60, 10), textcoords="offset points",
                bbox=dict(boxstyle="round", fc="w"),
                arrowprops=dict(arrowstyle="->", color="white"),
                zorder=100
            )
            ax.title.set_zorder(1)
            annotation.set_zorder(100)
            annotation.set_visible(False)
            self.tooltips[ax] = annotation
            
    def _text_colorchange(self, ax):
        """Change text color to highlight differences."""
        y_labels = ax.get_yticklabels()
        if self.diff_idx < len(y_labels):
            y_labels[self.diff_idx].set_color(UI_CONFIG['highlight_color'])
            y_labels[self.diff_idx].set_fontweight('bold')
            
        x_labels = ax.get_xticklabels()
        if self.diff_idx < len(x_labels):
            x_labels[self.diff_idx].set_color(UI_CONFIG['highlight_color'])
            x_labels[self.diff_idx].set_fontweight('bold')
            
        ax.figure.canvas.draw()
        
    def _draw_line_diff(self, ax, i, j, attention, spacing):
        """Draw attention lines for difference visualization."""
        y1 = 1 - (i + 0.6) * spacing
        y2 = 1 - (j + 0.6) * spacing
        alpha = abs(attention)
        
        if attention < 0:
            return ax.plot(
                [0, 1], [y1, y2],
                transform=ax.transAxes,
                color='red',
                alpha=alpha,
                linewidth=1
            )[0]
        else:
            return ax.plot(
                [0, 1], [y1, y2],
                color='green',
                transform=ax.transAxes,
                alpha=alpha,
                linewidth=1
            )[0]
            
    def _draw_line_prompts(self, ax, i, j, attention, spacing, color):
        """Draw attention lines for prompt visualization."""
        y1 = 1 - (i + 0.6) * spacing
        y2 = 1 - (j + 0.6) * spacing
        alpha = attention
        
        return ax.plot(
            [0, 1], [y1, y2],
            color=color,
            transform=ax.transAxes,
            alpha=alpha,
            linewidth=1
        )[0]
        
    def _compute_tokenbounds(self, tokens, spacing):
        """Compute token boundaries for visualization."""
        token_bounds = []
        for i in range(len(tokens)):
            y_center = 1 - (i + 0.6) * spacing
            height = spacing * 0.9
            ymin = y_center - height / 2
            ymax = y_center + height / 2
            token_bounds.append((ymin, ymax))
        return token_bounds
        
    def _init_line_visualizations(self, ax4, ax5, ax6):
        """Initialize line-based visualizations."""
        spacing1 = 1 / len(self.tokens1)
        
        # Draw tokens and lines for first prompt
        for i, token in enumerate(self.tokens1):
            y = 1 - (i + 0.6) * spacing1
            color = UI_CONFIG['highlight_color'] if i == self.diff_idx else UI_CONFIG['normal_color']
            weight = 'bold' if i == self.diff_idx else 'normal'
            
            ax4.text(0, y, token, ha='right', va='center', fontsize=UI_CONFIG['font_size'], 
                    color=color, weight=weight, transform=ax4.transAxes)
            ax4.text(1, y, token, ha='left', va='center', fontsize=UI_CONFIG['font_size'], 
                    color=color, weight=weight, transform=ax4.transAxes)
            
        # Draw attention lines
        for i in range(len(self.tokens1)):
            for j in range(len(self.tokens1)):
                attention_val = self.cur_layer_attentions[0, self.cur_head_idx, i, j].item()
                line = self._draw_line_prompts(ax4, i, j, attention_val, spacing1, 'blue')
                self.all_lines1.append(line)
                
        # Similar for second prompt
        spacing2 = 1 / len(self.tokens2)
        for i, token in enumerate(self.tokens2):
            y = 1 - (i + 0.6) * spacing2
            color = UI_CONFIG['highlight_color'] if i == self.diff_idx else UI_CONFIG['normal_color']
            weight = 'bold' if i == self.diff_idx else 'normal'
            
            ax5.text(0, y, token, ha='right', va='center', fontsize=UI_CONFIG['font_size'], 
                    color=color, weight=weight, transform=ax5.transAxes)
            ax5.text(1, y, token, ha='left', va='center', fontsize=UI_CONFIG['font_size'], 
                    color=color, weight=weight, transform=ax5.transAxes)
            
        for i in range(len(self.tokens2)):
            for j in range(len(self.tokens2)):
                attention_val = self.cur_layer_attentions[1, self.cur_head_idx, i, j].item()
                line = self._draw_line_prompts(ax5, i, j, attention_val, spacing2, 'blue')
                self.all_lines2.append(line)
                
        # Difference visualization
        spacing3 = 1 / len(self.tokens3)
        for i, token in enumerate(self.tokens3):
            y = 1 - (i + 0.6) * spacing3
            color = UI_CONFIG['highlight_color'] if i == self.diff_idx else UI_CONFIG['normal_color']
            weight = 'bold' if i == self.diff_idx else 'normal'
            
            ax6.text(0, y, token, ha='right', va='center', fontsize=UI_CONFIG['font_size'], 
                    color=color, weight=weight, transform=ax6.transAxes)
            ax6.text(1, y, token, ha='left', va='center', fontsize=UI_CONFIG['font_size'], 
                    color=color, weight=weight, transform=ax6.transAxes)
                    
        for i in range(len(self.tokens3)):
            for j in range(len(self.tokens3)):
                attn1 = self.cur_layer_attentions[0, self.cur_head_idx, i, j].item()
                attn2 = self.cur_layer_attentions[1, self.cur_head_idx, i, j].item()
                diff = attn1 - attn2
                line = self._draw_line_diff(ax6, i, j, diff, spacing3)
                self.all_lines3.append(line)
                
    def _init_matrix_visualizations(self, ax1, ax2, ax3):
        """Initialize matrix-based visualizations."""
        # First prompt attention matrix
        attn1 = self.cur_layer_attentions[0, self.cur_head_idx].detach().numpy()
        self.im1 = ax1.imshow(attn1, cmap='Blues', aspect='auto')
        ax1.set_title(f'Prompt 1 - Layer {self.cur_layer_idx}, Head {self.cur_head_idx}')
        ax1.set_xticks(range(len(self.tokens1)))
        ax1.set_xticklabels(self.tokens1, rotation=45, ha='right')
        ax1.set_yticks(range(len(self.tokens1)))
        ax1.set_yticklabels(self.tokens1)
        self._text_colorchange(ax1)
        plt.colorbar(self.im1, ax=ax1)
        
        # Second prompt attention matrix
        attn2 = self.cur_layer_attentions[1, self.cur_head_idx].detach().numpy()
        self.im2 = ax2.imshow(attn2, cmap='Blues', aspect='auto')
        ax2.set_title(f'Prompt 2 - Layer {self.cur_layer_idx}, Head {self.cur_head_idx}')
        ax2.set_xticks(range(len(self.tokens2)))
        ax2.set_xticklabels(self.tokens2, rotation=45, ha='right')
        ax2.set_yticks(range(len(self.tokens2)))
        ax2.set_yticklabels(self.tokens2)
        self._text_colorchange(ax2)
        plt.colorbar(self.im2, ax=ax2)
        
        # Difference matrix
        diff = attn1 - attn2
        self.im_diff = ax3.imshow(diff, cmap='RdBu_r', aspect='auto', vmin=-np.max(np.abs(diff)), vmax=np.max(np.abs(diff)))
        ax3.set_title(f'Difference - Layer {self.cur_layer_idx}, Head {self.cur_head_idx}')
        ax3.set_xticks(range(len(self.tokens3)))
        ax3.set_xticklabels(self.tokens3, rotation=45, ha='right')
        ax3.set_yticks(range(len(self.tokens3)))
        ax3.set_yticklabels(self.tokens3)
        self._text_colorchange(ax3)
        plt.colorbar(self.im_diff, ax=ax3)
        
        # Store original ranges
        self.original_range_im1 = (self.im1.get_array().min(), self.im1.get_array().max())
        self.original_range_im2 = (self.im2.get_array().min(), self.im2.get_array().max())
        self.original_range_imdiff = (self.im_diff.get_array().min(), self.im_diff.get_array().max())
        
    def _plot_attention_head(self, head_idx):
        """Plot attention for a specific head."""
        self.cur_head_idx = head_idx
        self.cur_layer_attentions = self.outputs.encoder_attentions[self.cur_layer_idx]
        
        # Clear previous visualizations
        for ax in self.axs.flatten():
            ax.clear()
            
        # Reinitialize visualizations
        self._init_matrix_visualizations(self.axs[0, 0], self.axs[0, 1], self.axs[0, 2])
        self._init_line_visualizations(self.axs[1, 0], self.axs[1, 1], self.axs[1, 2])
        
        # Update text boxes
        self.layer_textbox.set_val(str(self.cur_layer_idx))
        self.head_textbox.set_val(str(self.cur_head_idx))
        
        self.fig.canvas.draw()
        
    def _reset_lines(self):
        """Reset line highlighting."""
        for line in self.hovered_lines:
            line.set_alpha(0.3)
        self.hovered_lines.clear()
        
    def _on_hover(self, event):
        """Handle mouse hover events."""
        if event.inaxes not in self.axs.flatten():
            return
            
        ax = event.inaxes
        if ax not in self.tooltips:
            return
            
        # Find closest token
        spacing = 1 / len(self.tokens1)
        y_pos = event.ydata
        token_idx = int((1 - y_pos) / spacing)
        
        if 0 <= token_idx < len(self.tokens1):
            # Highlight lines
            self._reset_lines()
            for line in self.all_lines1:
                if hasattr(line, '_y1') and abs(line._y1 - (1 - (token_idx + 0.6) * spacing)) < 0.01:
                    line.set_alpha(0.8)
                    self.hovered_lines.append(line)
                    
            # Show tooltip
            tooltip = self.tooltips[ax]
            tooltip.xy = (event.xdata, event.ydata)
            tooltip.set_text(f'Token: {self.tokens1[token_idx]}')
            tooltip.set_visible(True)
            self.fig.canvas.draw_idle()
            
    def _on_unhover(self, event):
        """Handle mouse unhover events."""
        for tooltip in self.tooltips.values():
            tooltip.set_visible(False)
        self._reset_lines()
        self.fig.canvas.draw_idle()
        
    def _next_attention_head(self, event):
        """Move to next attention head."""
        if event.key == 'right':
            self.cur_head_idx = (self.cur_head_idx + 1) % self.num_heads_per_layer
        elif event.key == 'left':
            self.cur_head_idx = (self.cur_head_idx - 1) % self.num_heads_per_layer
        elif event.key == 'up':
            self.cur_layer_idx = (self.cur_layer_idx + 1) % self.total_num_layers
        elif event.key == 'down':
            self.cur_layer_idx = (self.cur_layer_idx - 1) % self.total_num_layers
        elif event.key == 'q':
            plt.close(self.fig)
            return
            
        self._plot_attention_head(self.cur_head_idx)
        
    def visualize(self):
        """Create and display the attention visualization."""
        # Create figure and subplots
        self.fig, self.axs = plt.subplots(2, 3, figsize=UI_CONFIG['figure_size'])
        self.fig.suptitle('Attention Visualization', fontsize=16)
        
        # Initialize UI elements
        self._init_slider()
        self._init_text_boxes()
        self._init_tooltip()
        
        # Initial plot
        self._plot_attention_head(self.cur_head_idx)
        
        # Connect events
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_hover)
        self.fig.canvas.mpl_connect('axes_leave_event', self._on_unhover)
        self.fig.canvas.mpl_connect('key_press_event', self._next_attention_head)
        
        plt.tight_layout()
        plt.show()


def main():
    """Main function for attention visualization."""
    if len(sys.argv) != 3:
        print("Usage: python attention_visualizer.py <prompt1> <prompt2>")
        sys.exit(1)
        
    prompt1, prompt2 = sys.argv[1], sys.argv[2]
    
    # Initialize model and visualizer
    model_manager = ModelManager("google/flan-t5-large")
    visualizer = AttentionVisualizer(model_manager, [prompt1, prompt2])
    
    # Display visualization
    visualizer.visualize()


if __name__ == '__main__':
    main() 