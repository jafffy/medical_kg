#!/usr/bin/env python3
"""
Test cases for JSON parsing robustness in OpenRouter client.

This module tests various malformed JSON responses that might come from LLMs
and ensures our parsing can handle them gracefully.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from soap_kg.utils.openrouter_client import OpenRouterClient

def test_json_parsing_robustness():
    """Test JSON parsing with various malformed inputs"""
    
    client = OpenRouterClient()
    
    # Test cases with various malformed JSON responses
    test_cases = [
        # Case 1: Perfect JSON - should work
        {
            "name": "Perfect JSON",
            "input": '[{"text": "aspirin", "type": "MEDICATION", "confidence": 0.9}]',
            "expected_type": "list",
            "should_succeed": True
        },
        
        # Case 2: JSON with markdown code blocks
        {
            "name": "JSON with markdown",
            "input": '```json\n[{"text": "aspirin", "type": "MEDICATION", "confidence": 0.9}]\n```',
            "expected_type": "list",
            "should_succeed": True
        },
        
        # Case 3: JSON with trailing comma
        {
            "name": "JSON with trailing comma",
            "input": '[{"text": "aspirin", "type": "MEDICATION", "confidence": 0.9,}]',
            "expected_type": "list",
            "should_succeed": True
        },
        
        # Case 4: JSON with single quotes
        {
            "name": "JSON with single quotes",
            "input": "[{'text': 'aspirin', 'type': 'MEDICATION', 'confidence': 0.9}]",
            "expected_type": "list",
            "should_succeed": True
        },
        
        # Case 5: JSON with explanatory text before
        {
            "name": "JSON with explanatory text",
            "input": 'Here are the extracted entities:\n[{"text": "aspirin", "type": "MEDICATION", "confidence": 0.9}]',
            "expected_type": "list",
            "should_succeed": True
        },
        
        # Case 6: JSON with explanatory text after
        {
            "name": "JSON with text after",
            "input": '[{"text": "aspirin", "type": "MEDICATION", "confidence": 0.9}]\nThese are the medical entities found.',
            "expected_type": "list",
            "should_succeed": True
        },
        
        # Case 7: SOAP object with missing quotes
        {
            "name": "SOAP object missing quotes",
            "input": '{subjective: [], objective: [{"text": "aspirin", "type": "MEDICATION"}], assessment: [], plan: []}',
            "expected_type": "dict",
            "should_succeed": True
        },
        
        # Case 8: JSON with unescaped quotes in values
        {
            "name": "Unescaped quotes in values",
            "input": '[{"text": "aspirin 325mg "daily"", "type": "MEDICATION", "confidence": 0.9}]',
            "expected_type": "list",
            "should_succeed": True
        },
        
        # Case 9: JSON with comments
        {
            "name": "JSON with comments",
            "input": '[\n  // This is aspirin\n  {"text": "aspirin", "type": "MEDICATION", "confidence": 0.9}\n]',
            "expected_type": "list",
            "should_succeed": True
        },
        
        # Case 10: Partial JSON object
        {
            "name": "Partial JSON object",
            "input": '{"text": "aspirin", "type": "MEDICATION"',
            "expected_type": "list",
            "should_succeed": False  # Should gracefully fail
        },
        
        # Case 11: Empty response
        {
            "name": "Empty response",
            "input": '',
            "expected_type": "list",
            "should_succeed": False  # Should return empty list
        },
        
        # Case 12: Non-JSON response
        {
            "name": "Non-JSON response",
            "input": 'I could not extract any entities from this text.',
            "expected_type": "list",
            "should_succeed": False  # Should return empty list
        },
        
        # Case 13: Multiple JSON objects (should take first/longest)
        {
            "name": "Multiple JSON objects",
            "input": '{"invalid": "object"} [{"text": "aspirin", "type": "MEDICATION", "confidence": 0.9}]',
            "expected_type": "list",
            "should_succeed": True
        },
        
        # Case 14: JSON with newlines in values
        {
            "name": "JSON with newlines",
            "input": '[{"text": "aspirin\\n325mg", "type": "MEDICATION", "confidence": 0.9}]',
            "expected_type": "list",
            "should_succeed": True
        },
        
        # Case 15: Malformed SOAP response
        {
            "name": "Malformed SOAP response",
            "input": 'The SOAP categories are:\nsubjective: []\nobjective: [{"text": "aspirin"}]\nassessment: []\nplan: []',
            "expected_type": "dict",
            "should_succeed": False  # Should return default structure
        }
    ]
    
    print("Testing JSON Parsing Robustness")
    print("=" * 50)
    
    passed = 0
    failed = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print(f"Input: {test_case['input'][:100]}{'...' if len(test_case['input']) > 100 else ''}")
        
        try:
            result = client._parse_json_with_fallbacks(
                test_case['input'], 
                expected_type=test_case['expected_type']
            )
            
            if test_case['should_succeed']:
                if result is not None and len(str(result)) > 0:
                    print(f"✅ PASS: Successfully parsed - {type(result).__name__}")
                    if test_case['expected_type'] == 'list' and isinstance(result, list):
                        print(f"   Result: {len(result)} items")
                    elif test_case['expected_type'] == 'dict' and isinstance(result, dict):
                        print(f"   Result: {len(result)} keys")
                    passed += 1
                else:
                    print(f"❌ FAIL: Expected success but got None or empty result")
                    failed += 1
            else:
                if result is None or (isinstance(result, (list, dict)) and len(result) == 0):
                    print(f"✅ PASS: Correctly handled malformed input")
                    passed += 1
                else:
                    print(f"❌ FAIL: Expected failure but got result: {result}")
                    failed += 1
                    
        except Exception as e:
            print(f"❌ FAIL: Unexpected exception: {e}")
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed} passed, {failed} failed")
    print(f"Success rate: {passed/(passed+failed)*100:.1f}%")
    
    return passed, failed

def test_specific_error_cases():
    """Test specific error cases from the logs"""
    
    client = OpenRouterClient()
    
    print("\n" + "="*50)
    print("Testing Specific Error Cases from Logs")
    print("="*50)
    
    # Cases based on the actual error messages from the logs
    error_cases = [
        {
            "name": "Expecting value: line 3 column 9",
            "input": 'Here are the entities:\n\n[{"text": aspirin", "type": "MEDICATION"}]',  # Missing quote
            "expected_type": "list"
        },
        {
            "name": "Expecting property name enclosed in double quotes",
            "input": '{\nsubjective: [],\nobjective: [{"text": "aspirin"}],\n}',  # Missing quotes on keys
            "expected_type": "dict"
        },
        {
            "name": "Expecting ',' delimiter",
            "input": '[{"text": "aspirin" "type": "MEDICATION"}]',  # Missing comma
            "expected_type": "list"
        }
    ]
    
    for i, case in enumerate(error_cases, 1):
        print(f"\nError Case {i}: {case['name']}")
        print(f"Input: {case['input']}")
        
        try:
            result = client._parse_json_with_fallbacks(
                case['input'], 
                expected_type=case['expected_type']
            )
            
            if result is not None:
                print(f"✅ Successfully recovered from error: {result}")
            else:
                print(f"✅ Gracefully handled error (returned None)")
                
        except Exception as e:
            print(f"❌ Failed to handle error case: {e}")

def test_real_llm_responses():
    """Test with realistic LLM response patterns"""
    
    client = OpenRouterClient()
    
    print("\n" + "="*50)
    print("Testing Realistic LLM Response Patterns")
    print("="*50)
    
    realistic_responses = [
        {
            "name": "Claude-style response",
            "input": '''I'll extract the medical entities from the clinical text:

```json
[
  {"text": "Furosemide", "type": "MEDICATION", "confidence": 0.95},
  {"text": "Ipratropium Bromide", "type": "MEDICATION", "confidence": 0.92}
]
```

These are the main medication entities I found in the text.''',
            "expected_type": "list"
        },
        {
            "name": "GPT-style response with explanation",
            "input": '''Based on the clinical text, here are the extracted entities:

[
  {
    "text": "chest pain",
    "type": "SYMPTOM", 
    "confidence": 0.9
  },
  {
    "text": "aspirin",
    "type": "MEDICATION",
    "confidence": 0.95
  }
]

I identified these entities based on the medical terminology present in the text.''',
            "expected_type": "list"
        },
        {
            "name": "Messy SOAP categorization",
            "input": '''Here's the SOAP categorization:

{
  "subjective": [
    {"text": "chest pain", "type": "SYMPTOM"}
  ],
  "objective": [
    {"text": "BP 140/90", "type": "VITAL_SIGN"},
  ],
  "assessment": [],
  "plan": [
    {"text": "aspirin", "type": "MEDICATION"}
  ]
}

The entities have been categorized according to SOAP methodology.''',
            "expected_type": "dict"
        }
    ]
    
    for i, case in enumerate(realistic_responses, 1):
        print(f"\nRealistic Case {i}: {case['name']}")
        
        try:
            result = client._parse_json_with_fallbacks(
                case['input'], 
                expected_type=case['expected_type']
            )
            
            if result is not None:
                print(f"✅ Successfully parsed realistic response")
                print(f"   Type: {type(result).__name__}")
                if isinstance(result, list):
                    print(f"   Items: {len(result)}")
                elif isinstance(result, dict):
                    print(f"   Keys: {list(result.keys())}")
            else:
                print(f"❌ Failed to parse realistic response")
                
        except Exception as e:
            print(f"❌ Error parsing realistic response: {e}")

if __name__ == "__main__":
    print("JSON Parsing Robustness Test Suite")
    print("=" * 50)
    
    # Run all tests
    passed, failed = test_json_parsing_robustness()
    test_specific_error_cases()
    test_real_llm_responses()
    
    print(f"\n{'='*50}")
    print("Test Suite Complete")
    print(f"Main tests: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! JSON parsing is robust.")
    else:
        print("⚠️  Some tests failed. Consider improving the parsing logic.")