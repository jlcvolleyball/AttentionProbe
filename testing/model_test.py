from transformers import T5Tokenizer, T5ForConditionalGeneration
import sys
import torch
import time
import gc

cur_mode = "cpu"

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
if len(sys.argv) > 1 and sys.argv[1] == "gpu":
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large").to(device)
    print(f"Model is on device: {next(model.parameters()).device}")
else:
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
    print(f"Model is on device: {next(model.parameters()).device}")

input_list = []

def gen_input_list(filename):
    full_filename = "model_test_inputs/" + filename
    with open(full_filename, "r") as f:
        for line in f:
            input_list.append(line.strip())

def main():
    # initialize time tracking
    start_time = 0
    end_time = 0

    if len(sys.argv) >= 3:
        if cur_mode == "cpu":
            start_idx = 1
        else:
            start_idx = 2
        flag = sys.argv[start_idx]
        if flag == "-f":
            gen_input_list(sys.argv[start_idx + 1])

    print(f'Results for {model.name_or_path}:\n')
    if len(sys.argv) > 1 and sys.argv[1] == "gpu":
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

    gc.collect()

if __name__ == '__main__':
    main()
