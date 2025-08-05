from transformers import T5ForConditionalGeneration, T5Config, T5Tokenizer
import torch
import subprocess

# set up global variables
prompt1, prompt2 = "", ""
default_prompt1 = "A man walked into a room with two cats and a refrigerator. He scratched them. What did the man scratch?"
default_prompt2 = "A man walked into a room with two cats and a refrigerator. He scratched it. What did the man scratch?"

def find_difference(t1, t2):
    word_diff = -1
    pointer = 0
    while(pointer < len(t1)):
        if t1[pointer] != t2[pointer]: word_diff = pointer
        pointer+=1
    return word_diff

def execute_introduction():
    print("Hello! Welcome to Demo 2. In this demonstration, we will ask you to input two of your own prompts.")
    print("We will run your sentences on Google's FLAN-T5 Large model, and will show you interesting attention heads. \n\n")
    print("In this demo, we will focus on attention heads that pay attention to number agreement.")

def transition_description():
    print("Now, we will present some of the notable attention heads. Press q to exit from the demonstration.")

def check_sentence_validity(sentence, order):
    if " it " not in sentence and " her " not in sentence: return False
    if sentence.count(" his ") > 1: return False
    if sentence.count(" her ") > 1: return False
    return True


def run_model():
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
    input_list = [
        prompt1,
        prompt2
    ]

    for input_text in input_list:
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids
        outputs = model.generate(input_ids, max_length=20)
        output_text = tokenizer.decode(outputs[0][1:-1])
        print(f'{input_text}  ->  {output_text}')

def main():
    global prompt1, prompt2
    execute_introduction()

    print("Please input your prompt below. \nREQUIREMENTS: This prompt must contain one of the following "
          "pronouns once: them")
    prompt1 = input("My first prompt: ")
    # while not check_sentence_validity(prompt1, 1):
    #     if prompt1 == "0":
    #         prompt1 = default_prompt1
    #         break
    #     print("Your prompt does not satisfy the requirements. Please reenter a valid prompt below.")
    #     prompt1 = input("My first prompt: ")
    #
    # print("Now, from your inputted prompt, we have generated an identical prompt, but with your instance of the "
    #       "pronoun his or her replaced the other.")
    # gen_prompt2()
    print("Please input your second prompt. ")
    prompt2 = input("My second prompt: ")

    print("\n\Your two prompts are:")
    print(f"Prompt1: {prompt1}")
    print(f"Prompt2: {prompt2}")

    print("Let's run both your prompts through the model. Here is the output of the model below: ")
    run_model()

    transition_description()

    args = ["python", "demo2_attentionvis.py", prompt1, prompt2]
    subprocess.run(args)

if __name__ == '__main__':
    main()
