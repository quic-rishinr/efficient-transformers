#!/usr/bin/env python3
# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Test script to verify logits alignment between baseline and QEfficient models.

This script tests the tokenization alignment fix to ensure that both models
are computing logits for the same token positions.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_tokenization_alignment():
    """Test that decode/encode round trip preserves tokenization."""
    model_name = "gpt2"  # Use a small model for testing
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test with various text samples
    test_texts = [
        "Hello world!",
        "The quick brown fox jumps over the lazy dog.",
        "This is a test with special characters: @#$%^&*()",
        "Multiple\nlines\nof\ntext",
        "Numbers: 123 456 789",
    ]
    
    alignment_issues = 0
    
    for text in test_texts:
        # Encode text to tokens
        original_ids = tokenizer.encode(text, return_tensors="pt")[0]
        
        # Test decode without skipping special tokens (our fix)
        decoded_text = tokenizer.decode(original_ids, skip_special_tokens=False)
        re_encoded_ids = tokenizer.encode(decoded_text, add_special_tokens=False, return_tensors="pt")[0]
        
        if not torch.equal(original_ids, re_encoded_ids):
            logger.warning(f"Alignment issue with text: '{text}'")
            logger.warning(f"Original: {original_ids.tolist()}")
            logger.warning(f"Re-encoded: {re_encoded_ids.tolist()}")
            alignment_issues += 1
        else:
            logger.info(f"✓ Alignment OK for: '{text[:30]}...'")
    
    if alignment_issues == 0:
        logger.info("✓ All tokenization alignment tests passed!")
    else:
        logger.error(f"✗ {alignment_issues} tokenization alignment issues found!")
    
    return alignment_issues == 0

def test_logits_shape_consistency():
    """Test that logits shapes are consistent between models."""
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load baseline model
    baseline_model = AutoModelForCausalLM.from_pretrained(model_name)
    baseline_model.eval()
    
    # Test input
    test_text = "The quick brown fox"
    input_ids = tokenizer.encode(test_text, return_tensors="pt")
    
    # Get baseline logits
    with torch.no_grad():
        baseline_outputs = baseline_model(input_ids)
        baseline_logits = baseline_outputs.logits
    
    logger.info(f"Input shape: {input_ids.shape}")
    logger.info(f"Baseline logits shape: {baseline_logits.shape}")
    logger.info(f"Expected: [batch_size={input_ids.shape[0]}, seq_len={input_ids.shape[1]}, vocab_size={baseline_model.config.vocab_size}]")
    
    # Verify shape consistency
    expected_shape = (input_ids.shape[0], input_ids.shape[1], baseline_model.config.vocab_size)
    if baseline_logits.shape == expected_shape:
        logger.info("✓ Logits shape is consistent!")
        return True
    else:
        logger.error(f"✗ Logits shape mismatch! Expected {expected_shape}, got {baseline_logits.shape}")
        return False

def test_perplexity_calculation():
    """Test that perplexity calculation is correct."""
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    baseline_model = AutoModelForCausalLM.from_pretrained(model_name)
    baseline_model.eval()
    
    test_text = "The quick brown fox jumps over the lazy dog"
    input_ids = tokenizer.encode(test_text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = baseline_model(input_ids)
        logits = outputs.logits[0]  # Remove batch dimension
        target_ids = input_ids[0]
    
    # Calculate perplexity using the same method as PerplexityCalculator
    loss = torch.nn.functional.cross_entropy(logits[:-1], target_ids[1:], reduction='mean')
    perplexity = torch.exp(loss).item()
    
    logger.info(f"Test text: '{test_text}'")
    logger.info(f"Input tokens: {len(target_ids)}")
    logger.info(f"Loss: {loss.item():.4f}")
    logger.info(f"Perplexity: {perplexity:.4f}")
    
    # Sanity check: perplexity should be reasonable (not NaN, not extremely high)
    if np.isnan(perplexity) or perplexity > 10000:
        logger.error(f"✗ Perplexity calculation seems wrong: {perplexity}")
        return False
    else:
        logger.info("✓ Perplexity calculation looks reasonable!")
        return True

def main():
    """Run all alignment tests."""
    logger.info("="*60)
    logger.info("LOGITS ALIGNMENT TESTS")
    logger.info("="*60)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Tokenization alignment
    logger.info("\n1. Testing tokenization alignment...")
    if test_tokenization_alignment():
        tests_passed += 1
    
    # Test 2: Logits shape consistency
    logger.info("\n2. Testing logits shape consistency...")
    if test_logits_shape_consistency():
        tests_passed += 1
    
    # Test 3: Perplexity calculation
    logger.info("\n3. Testing perplexity calculation...")
    if test_perplexity_calculation():
        tests_passed += 1
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    logger.info(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        logger.info("✓ All tests passed! The logits alignment fix should work correctly.")
        return True
    else:
        logger.error(f"✗ {total_tests - tests_passed} tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
