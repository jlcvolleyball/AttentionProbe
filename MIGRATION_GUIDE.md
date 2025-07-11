# Migration Guide: Original to Refactored AttentionProbe

This guide helps you understand the changes made during the refactoring and how to migrate from the original codebase to the new modular architecture.

## 🔄 What Changed

### Before (Original Structure)
```
demo1.py (89 lines)          # Pronoun resolution demo
demo2.py (81 lines)          # Number agreement demo  
demo1_attentionvis.py (501 lines)  # Visualization for demo1
demo2_attentionvis.py (501 lines)  # Visualization for demo2
attention_visualizations.py (554 lines)  # General visualization
```

### After (Refactored Structure)
```
demo1_refactored.py (15 lines)     # Pronoun resolution demo
demo2_refactored.py (15 lines)     # Number agreement demo
demo_base.py (95 lines)            # Shared demo logic
attention_visualizer.py (400 lines) # Unified visualization
config.py (50 lines)               # Configuration
utils.py (120 lines)               # Utilities
```

## 📊 Code Reduction Summary

| Metric | Before | After | Reduction |
|--------|--------|-------|-----------|
| Total Lines | ~1,700 | ~695 | ~60% |
| Duplicate Code | ~400 lines | 0 lines | 100% |
| Demo Files | 2 separate files | 2 files sharing base class | 85% reduction per demo |
| Visualization Files | 3 separate files | 1 unified file | 67% reduction |

## 🚀 Key Improvements

### 1. **Eliminated Code Duplication**
**Before:**
```python
# demo1.py and demo2.py had nearly identical structure
def find_difference(t1, t2):
    word_diff = -1
    pointer = 0
    while(pointer < len(t1)):
        if t1[pointer] != t2[pointer]: word_diff = pointer
        pointer+=1
    return word_diff
```

**After:**
```python
# Single implementation in utils.py
def find_difference(tokens1: List[str], tokens2: List[str]) -> int:
    """Find the index where two token lists differ."""
    if len(tokens1) != len(tokens2):
        raise ValueError("Token lists must have the same length")
    
    for i, (t1, t2) in enumerate(zip(tokens1, tokens2)):
        if t1 != t2:
            return i
    return -1
```

### 2. **Configuration Management**
**Before:**
```python
# Hardcoded values scattered throughout files
MODEL_NAME = "google/flan-t5-large"
default_prompt1 = "The man showed the woman his jacket..."
default_prompt2 = "The man showed the woman her jacket..."
```

**After:**
```python
# Centralized in config.py
DEMO_CONFIGS = {
    "pronoun_resolution": {
        "name": "Demo 1",
        "description": "Focus on attention heads that perform pronoun resolution.",
        "pronouns": ["his", "her"],
        "default_prompt1": "The man showed the woman his jacket...",
        "default_prompt2": "The man showed the woman her jacket...",
        # ... more configuration
    }
}
```

### 3. **Class-Based Architecture**
**Before:**
```python
# Global variables and procedural code
prompt1, prompt2 = "", ""
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")

def run_model():
    # ... implementation
```

**After:**
```python
# Class-based with proper encapsulation
class BaseDemo:
    def __init__(self, demo_type: str):
        self.config = DEMO_CONFIGS[demo_type]
        self.model_manager = ModelManager("google/flan-t5-large")
        self.prompt1 = ""
        self.prompt2 = ""
    
    def run_model_inference(self):
        # ... implementation
```

## 🔧 Migration Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Update Import Statements
If you have code that imports the original modules:

**Before:**
```python
from demo1 import find_difference, run_model
```

**After:**
```python
from utils import find_difference
from demo_base import BaseDemo
```

### Step 3: Update Function Calls
**Before:**
```python
# Direct function calls
find_difference(tokens1, tokens2)
run_model()
```

**After:**
```python
# Class-based approach
demo = BaseDemo("pronoun_resolution")
demo.run()
```

### Step 4: Configuration Changes
**Before:**
```python
# Hardcoded values
if " his " not in sentence and " her " not in sentence: return False
```

**After:**
```python
# Configuration-driven
if validate_sentence(sentence, self.config['pronouns']):
    # proceed
```

## 🧪 Testing Migration

### Run Tests
```bash
python test_refactored.py
```

### Test Functionality
```bash
# Test Demo 1
python demo1_refactored.py

# Test Demo 2  
python demo2_refactored.py

# Test direct visualization
python attention_visualizer.py "The man showed his jacket" "The man showed her jacket"
```

## 🔍 Backward Compatibility

### What's Preserved
- ✅ All original functionality
- ✅ Same user interface and prompts
- ✅ Same visualization capabilities
- ✅ Same model outputs

### What's Changed
- ❌ File structure and organization
- ❌ Import statements
- ❌ Global variable usage
- ❌ Code duplication

### What's Improved
- ✅ Better error handling
- ✅ Type hints for better IDE support
- ✅ Comprehensive documentation
- ✅ Easier testing
- ✅ More maintainable code

## 🚨 Breaking Changes

### 1. **Import Changes**
If you import functions from the original files, you'll need to update imports:

**Before:**
```python
from demo1 import find_difference
```

**After:**
```python
from utils import find_difference
```

### 2. **Global Variables**
Global variables are now encapsulated in classes:

**Before:**
```python
global prompt1, prompt2
prompt1 = input("My first prompt: ")
```

**After:**
```python
self.prompt1 = self.get_user_prompt(1)
```

### 3. **Function Signatures**
Some utility functions now have type hints and better error handling:

**Before:**
```python
def find_difference(t1, t2):
    # No error handling
```

**After:**
```python
def find_difference(tokens1: List[str], tokens2: List[str]) -> int:
    # With error handling and type hints
```

## 🎯 Benefits of Migration

### For Developers
- **Easier Maintenance**: Changes in one place affect all demos
- **Better Testing**: Modular design enables unit testing
- **Type Safety**: Type hints catch errors early
- **Documentation**: Comprehensive docstrings

### For Users
- **Same Experience**: No change in functionality
- **Better Error Messages**: More informative validation
- **Faster Development**: New features can be added more easily

### For the Project
- **Reduced Bugs**: Less duplicate code means fewer bugs
- **Faster Development**: New demos can be created quickly
- **Better Collaboration**: Clearer code structure

## 📞 Support

If you encounter issues during migration:

1. **Check the test file**: `test_refactored.py` shows expected behavior
2. **Review the README**: `README_REFACTORED.md` has detailed documentation
3. **Compare files**: Original and refactored versions are both available
4. **Run tests**: Ensure all functionality works as expected

## 🔮 Future Development

The refactored architecture makes it easy to add new features:

```python
# Adding a new demo type
DEMO_CONFIGS["new_demo"] = {
    "name": "Demo 3",
    "description": "New attention analysis",
    "pronouns": ["new", "words"],
    # ... configuration
}

# The new demo automatically works with all existing infrastructure
demo = BaseDemo("new_demo")
demo.run()
``` 