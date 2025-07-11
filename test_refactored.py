"""
Tests for the refactored AttentionProbe codebase.
"""

import unittest
from unittest.mock import Mock, patch
from config import DEMO_CONFIGS, VALIDATION_RULES
from utils import validate_sentence, generate_contrast_prompt, find_difference, ModelManager
from demo_base import BaseDemo


class TestUtils(unittest.TestCase):
    """Test utility functions."""
    
    def test_find_difference(self):
        """Test finding differences between token lists."""
        tokens1 = ["The", "man", "showed", "his", "jacket"]
        tokens2 = ["The", "man", "showed", "her", "jacket"]
        
        diff_idx = find_difference(tokens1, tokens2)
        self.assertEqual(diff_idx, 3)
        
        # Test identical lists
        diff_idx = find_difference(tokens1, tokens1)
        self.assertEqual(diff_idx, -1)
        
    def test_validate_sentence(self):
        """Test sentence validation."""
        # Valid sentences
        self.assertTrue(validate_sentence("The man showed his jacket", ["his", "her"]))
        self.assertTrue(validate_sentence("The woman showed her jacket", ["his", "her"]))
        
        # Invalid sentences
        self.assertFalse(validate_sentence("The man showed jacket", ["his", "her"]))
        self.assertFalse(validate_sentence("The man showed his his jacket", ["his", "her"]))
        
    def test_generate_contrast_prompt(self):
        """Test contrasting prompt generation."""
        original = "The man showed his jacket"
        result = generate_contrast_prompt(original, ["his", "her"])
        self.assertEqual(result, "The man showed her jacket")
        
        original = "The woman showed her jacket"
        result = generate_contrast_prompt(original, ["his", "her"])
        self.assertEqual(result, "The woman showed his jacket")


class TestModelManager(unittest.TestCase):
    """Test ModelManager class."""
    
    @patch('utils.T5Tokenizer.from_pretrained')
    @patch('utils.T5ForConditionalGeneration.from_pretrained')
    @patch('utils.T5Config.from_pretrained')
    def test_model_loading(self, mock_config, mock_model, mock_tokenizer):
        """Test model loading."""
        manager = ModelManager("test-model")
        manager.load_model()
        
        mock_tokenizer.assert_called_once_with("test-model")
        mock_model.assert_called_once_with("test-model")
        mock_config.assert_called_once_with("test-model")


class TestBaseDemo(unittest.TestCase):
    """Test BaseDemo class."""
    
    def test_demo_initialization(self):
        """Test demo initialization."""
        demo = BaseDemo("pronoun_resolution")
        self.assertEqual(demo.config["name"], "Demo 1")
        self.assertEqual(demo.config["pronouns"], ["his", "her"])
        
    def test_invalid_demo_type(self):
        """Test invalid demo type handling."""
        with self.assertRaises(ValueError):
            BaseDemo("invalid_demo")
            
    def test_config_access(self):
        """Test configuration access."""
        demo = BaseDemo("number_agreement")
        self.assertEqual(demo.config["name"], "Demo 2")
        self.assertEqual(demo.config["pronouns"], ["them", "it"])


class TestConfiguration(unittest.TestCase):
    """Test configuration settings."""
    
    def test_demo_configs_structure(self):
        """Test demo configuration structure."""
        for demo_type, config in DEMO_CONFIGS.items():
            required_keys = ["name", "description", "pronouns", "default_prompt1", 
                           "default_prompt2", "validation_message", "interesting_heads"]
            
            for key in required_keys:
                self.assertIn(key, config, f"Missing key '{key}' in {demo_type}")
                
    def test_validation_rules(self):
        """Test validation rules."""
        self.assertIn("max_pronoun_count", VALIDATION_RULES)
        self.assertIn("required_pronouns", VALIDATION_RULES)
        self.assertIsInstance(VALIDATION_RULES["max_pronoun_count"], int)
        self.assertIsInstance(VALIDATION_RULES["required_pronouns"], list)


if __name__ == '__main__':
    unittest.main() 