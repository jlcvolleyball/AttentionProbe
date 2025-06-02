from transformers import T5ForConditionalGeneration, T5Config, T5Tokenizer
import torch
import sys

# setting up the model
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

def main():
    print(input_ids1)
    list_of_ids = input_ids1.numpy()[0]
    decoded = tokenizer.batch_decode(list_of_ids)
    print(decoded)
    zipped_dict = dict(zip(list_of_ids.tolist(), decoded))
    print(zipped_dict)

if __name__ == '__main__':
    main()