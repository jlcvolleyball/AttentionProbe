from transformers import T5ForConditionalGeneration, T5Config, T5Tokenizer
import torch
import sys

# model setup
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
config = T5Config.from_pretrained("google/flan-t5-base")
model.eval()

# input and tokenizing
# "The man gave the woman his jacket. Who owned the jacket, the man or the woman?"
input_text1 = "The man gave the woman his jacket. Who owned the jacket, the man or the woman?"
input_text2 = "The man gave the woman her jacket. Who owned the jacket, the man or the woman?"

my_input1 = tokenizer(input_text1, return_tensors="pt")
my_input2 = tokenizer(input_text2, return_tensors="pt")

input_ids1 = tokenizer(input_text1, return_tensors="pt").input_ids
input_ids2 = tokenizer(input_text2, return_tensors="pt").input_ids

def print_architecture_info():
    print("\n\n")
    print("=================================================================================")
    print("                             FLAN-T5 ARCHITECTURE INFO                           ")
    print("=================================================================================")
    for i, layer in enumerate(model.encoder.block):
        if i == 0:
            print(f"Embedding Tokens: {model.encoder.embed_tokens}")
        print(f"Encoder Layer {i}:")
        print("  Self-Attention:")
        print(f"    q: {layer.layer[0].SelfAttention.q.weight.shape}")
        print(f"    k: {layer.layer[0].SelfAttention.k.weight.shape}")
        print(f"    v: {layer.layer[0].SelfAttention.v.weight.shape}")
        print(f"    o: {layer.layer[0].SelfAttention.o.weight.shape}")
        print("  Feed-Forward:")
        ff_layer = layer.layer[1].DenseReluDense
        print(f"    wi_0: {ff_layer.wi_0.weight.shape}")
        print(f"    wi_1: {ff_layer.wi_1.weight.shape}")

    for i, layer in enumerate(model.decoder.block):
        if i == 0:
            print(f"Embedding Tokens: {model.decoder.embed_tokens}")
        print(f"Decoder Layer {i}:")
        print("  Self-Attention:")
        print(f"    q: {layer.layer[0].SelfAttention.q.weight.shape}")
        print(f"    k: {layer.layer[0].SelfAttention.k.weight.shape}")
        print(f"    v: {layer.layer[0].SelfAttention.v.weight.shape}")
        print(f"    o: {layer.layer[0].SelfAttention.o.weight.shape}")
        print("  Cross-Attention:")
        print(f"    q: {layer.layer[1].EncDecAttention.q.weight.shape}")
        print(f"    k: {layer.layer[1].EncDecAttention.k.weight.shape}")
        print(f"    v: {layer.layer[1].EncDecAttention.v.weight.shape}")
        print(f"    o: {layer.layer[1].EncDecAttention.o.weight.shape}")
        print("  Feed-Forward:")
        ff_layer = layer.layer[2].DenseReluDense
        print(f"    wi_0: {ff_layer.wi_0.weight.shape}")
        print(f"    wi_1: {ff_layer.wi_1.weight.shape}")

    print(f"lm_head: {model.lm_head.weight.shape}")

    print(f"Layers: {config.num_layers}")
    print(f"Attention Heads: {config.num_heads}")
    print(f"Model Dimension (d_model): {config.d_model}")
    print(f"Feed-Forward Dimension (d_ff): {config.d_ff}")

def main():
    # call generate with output_attention=True
    outputs1 = model.generate(input_ids=input_ids1, output_attentions=True, return_dict_in_generate=True)
    outputs2 = model.generate(input_ids=input_ids2, output_attentions=True, return_dict_in_generate=True)

    # Get all self-attention heads from layer 0 of encoder
    him_encoder_layer_0_attentions = outputs1.encoder_attentions[
        0]  # (batch_size, num_heads, generated_length, sequence_length)
    # layer 0 head 2 activation
    him_head_3_activations = him_encoder_layer_0_attentions[:, 7, :, :]
    print("HIM information")
    print(him_encoder_layer_0_attentions)
    print(him_head_3_activations)
    print(him_head_3_activations.shape)
    print("\n\n\n")

    print("HER information")
    her_encoder_layer_0_attentions = outputs2.encoder_attentions[0]
    her_head_3_activations = her_encoder_layer_0_attentions[:, 7, :, :]
    print(her_encoder_layer_0_attentions)
    print(her_head_3_activations)
    print(her_head_3_activations.shape)
    print("\n\n\n")

    generated_ids = outputs1.sequences  # this contains token IDs
    detokenized_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(detokenized_output)

    if len(sys.argv) > 1 and sys.argv[1] == "info": print_architecture_info()

if __name__ == '__main__':
   main()