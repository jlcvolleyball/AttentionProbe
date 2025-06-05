from transformers import T5Tokenizer, T5ForConditionalGeneration
import sys
import torch
import time

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print("Is CUDA available?", torch.cuda.is_available())

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
if len(sys.argv) > 1 and sys.argv[1] == "gpu":
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large").to(device)
    print(f"Model is on device: {next(model.parameters()).device}")
else:
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
    print(f"Model is on device: {next(model.parameters()).device}")

input_list = [
    "The man showed the woman his jacket. Who owned the jacket?",
    "The man showed the woman her jacket. Who owned the jacket?",
    "The man showed the woman his jacket. Who owned the jacket, and why?",
    "The man showed the woman her jacket. Who owned the jacket, and why?",
    "The man showed the woman his jacket. Who owned the jacket, the man or the woman?",
    "The man showed the woman her jacket. Who owned the jacket, the man or the woman?",
    "John showed Mary his jacket. Who owned the jacket?",
    "John showed Mary her jacket. Who owned the jacket?",
    "John showed Mary his jacket. Who owned the jacket, and why?",
    "John showed Mary her jacket. Who owned the jacket, and why?"
]

def main():
    start_time = 0
    end_time = 0
    print(f'Results for {model.name_or_path}:\n')
    if len(sys.argv) > 1 and sys.argv[1] == "gpu":
        print("I'm here")
        for input_text in input_list:
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
            start_time = time.time()
            outputs = model.generate(input_ids)
            end_time = time.time()
            output_text = tokenizer.decode(outputs[0][1:-1])
            print(f'{input_text}  ->  {output_text}')
    else:
        for input_text in input_list:
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids
            start_time = time.time()
            outputs = model.generate(input_ids)
            end_time = time.time()
            output_text = tokenizer.decode(outputs[0][1:-1])
            print(f'{input_text}  ->  {output_text}')
    print(f"Model runtime: {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    main()
