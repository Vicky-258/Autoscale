
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from temporal.patterns import Pattern
from temporal.classifier import BurstClassifier

def test_enum_mismatch():
    print("Testing Enum vs String mismatch...")
    classifier = BurstClassifier()
    
    # patterns.py returns Pattern.BURST (Enum)
    # classifier.py checks if pattern == "BURST" (String)
    
    input_pattern = Pattern.BURST
    print(f"Input: {input_pattern} (Type: {type(input_pattern)})")
    
    state, explanation = classifier.update(input_pattern)
    
    print(f"Resulting State: {state}")
    print(f"Explanation: {explanation}")
    
    if state == "BURST":
        print("SUCCESS: Classifier handled Enum correctly (or logic matched).")
    else:
        print("FAILURE: Classifier failed to match Enum to String.")

if __name__ == "__main__":
    test_enum_mismatch()
