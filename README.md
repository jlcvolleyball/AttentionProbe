# AttentionProbe

**AttentionProbe** is an interactive tool that provides custom visualizations of attention head activations in Google's FLAN-T5-large LLM. This project 
was developed as part of Carnegie Mellon University's Summer Undergraduate Research Fellowship (SURF) Program in 2025, where the project aimed to identify
meaningful and interpretable attention heads. AttentionProbe has lately been extended to allow users to explore how embeddings change across layers in the 
model.

---

## Features
AttentionProbe encompasses 4 **interactive demonstrations** that feature attention heads performing **pronoun resolution, number agreement, noun phrase identification,
and prepositional phrase attachment**. These demos allow you to input a prompt (or pair of prompts) in order to see the attention head in action. 

Along with the tailored demos, AttentionProbe also allows for independent exploration using its **base visualization** functionality. The base visualization supports 
navigation through all 384 attention heads across the 24 layers of the model, with tools such as a range slider. 

Other attention head visualization features include:
- A **matrix visualization**, which plots attention head activations as a matrix, # rows = # cols = length of tokenized sequence of input prompt.
- A **line visualization**, which plots attention head activations as lines, inspired by the visualizations in [BertViz](https://github.com/jessevig/bertviz)
- Interactive Matplotlib UI with sliders, tooltips, and input boxes

AttentionProbe also features 3 types of **embedding visualizations**, which allow the user to explore how the embeddings change through layers via matrix visualizations
and cosine similarity graphs

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/jlcvolleyball/AttentionProbe.git
cd AttentionProbe
```

### 2. Set up environment
```bash
pip install -r requirements.txt
```

### 3. Run the desired tool
The four attention head demos are contained in ```demo_pronoun_res.py```, ```demo_number_agreement.py```, ```demo_noun_phrase.py```, and ```demo_prep_phrase_attach.py```
To run the demo,
```bash
python demo_pronoun_res.py
```

The base visualization is contained in ```attention_visualizer.py```
For more detailed usage information, run
```bash
python attention_visualizer.py
```

The embedding visualizations are contained in ```extract_embeddings.py```
For more detailed usage information, run
```bash
python extract_embeddings.py
```
