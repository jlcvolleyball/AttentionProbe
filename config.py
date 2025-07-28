"""
Configuration settings for the AttentionProbe application.
"""

# Model Configuration
MODEL_NAME = "google/flan-t5-large"
DEVICE = "cpu"

# Demo Configuration
DEMO_CONFIGS = {
    "pronoun_resolution": {
        "name": "Demo 1",
        "description": "In this demo, we will focus on attention heads that perform pronoun resolution.",
        "keywords": ["his", "her"],
        "max_keyword_count": 1,
        "auto_generate_prompt2": True,
        "need_prompt2": True,
        "default_prompt1": "The man showed the woman his jacket. Who owned the jacket, the man or the woman?",
        "default_prompt2": "The man showed the woman her jacket. Who owned the jacket, the man or the woman?",
        "validation_message": "This prompt must contain one of the following pronouns once: his, her",
        "interesting_heads": [(0, 15), (2, 6), (2, 8), (2, 9), (3, 6), (3, 9)]
    },
    "number_agreement": {
        "name": "Demo 2", 
        "description": "In this demo, we will focus on attention heads that pay attention to number agreement.",
        "keywords": ["them", "it"],
        "max_keyword_count": 1,
        "auto_generate_prompt2": True,
        "need_prompt2": True,
        "default_prompt1": "A man walked into a room with two cats and a refrigerator. He scratched them. What did the man scratch?",
        "default_prompt2": "A man walked into a room with two cats and a refrigerator. He scratched it. What did the man scratch?",
        "validation_message": "This prompt must contain one of the following pronouns once: them, it",
        "interesting_heads": [(3, 9), (6, 14), (10, 9), (11, 15), (16, 10), (22, 14)]
    },
    "noun_phrases": {
        "name": "Demo 3",
        "description": "In this demo, we will focus on attention heads that identify noun phrases, mapping the phrase to the first 'the' in the noun phrase",
        "keywords": ['the'],
        "max_keyword_count": -1, # this represents unlimited occurrences of "the" are allowed.
        "auto_generate_prompt2": False,
        "need_prompt2": False,
        "default_prompt1": "The big white fluffy cat walked down the long black road in the morning.",
        "default_prompt2": "The big white fluffy cat walked down the long black road in the morning.",
        "validation_message": "Your prompt must include at least one noun phrase starting with 'the'. An example of a noun phrase: The big white fluffy cat",
        "interesting_heads": [(7, 14), (8, 3), (8, 12), (8, 14), (9, 14), (10, 3), (10, 12), (11, 3)]
    },
    "prep_phrase_attach": {
        "name": "Demo 4",
        "description": "In this demo, we will focus on attention heads that perform prepositional phrase attachment.",
        "keywords": ["in", "with", "for"],
        "max_keyword_count": 1,
        "auto_generate_prompt2": False,
        "need_prompt2": True,
        "default_prompt1": "They discussed the plan for hours. What was the plan for?",
        "default_prompt2": "They discussed the plan for dinner. What was the plan for?",
        "validation_message": "This prompt must contain one of the following prepositions: in, with, for",
        "interesting_heads": [(12, 8), (14, 8), (16, 10), (18, 12), (21, 10)]
    }
}

# UI Configuration
UI_CONFIG = {
    "max_generation_length": 20,
    "slider_range": (0.0, 1.0),
    "slider_default": 1.0,
    "figure_size": (50, 8),
    "font_size": 10,
    "highlight_color": "red",
    "normal_color": "black"
}

# Validation Rules
VALIDATION_RULES = {
    "max_pronoun_count": 1,
    "required_pronouns": ["his", "her", "them", "it"]
} 