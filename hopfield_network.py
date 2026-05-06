# hopfield_network.py
import numpy as np
import streamlit as st
from PIL import Image

class HopfieldNetwork:
    def __init__(self, pattern_size=100):
        self.pattern_size = pattern_size
        self.grid_size = int(np.sqrt(pattern_size))
        self.weights = np.zeros((pattern_size, pattern_size))
        self.patterns = []
        self.pattern_names = []
        self.is_trained = False
    
    def train(self, patterns, pattern_names=None):
        self.patterns = patterns
        self.pattern_names = pattern_names if pattern_names else [f"Pattern_{i}" for i in range(len(patterns))]
        
        self.weights = np.zeros((self.pattern_size, self.pattern_size))
        
        for pattern in patterns:
            pattern_vector = pattern.flatten()
            self.weights += np.outer(pattern_vector, pattern_vector)
        
        np.fill_diagonal(self.weights, 0)
        self.weights /= self.pattern_size
        self.is_trained = True
        
    def recall(self, noisy_pattern, max_iterations=100):
        if not self.is_trained:
            return None
        
        current_state = noisy_pattern.flatten().copy()
        
        for iteration in range(max_iterations):
            indices = np.random.permutation(self.pattern_size)
            for i in indices:
                net_input = np.dot(self.weights[i], current_state)
                current_state[i] = 1 if net_input >= 0 else -1
        
        best_match_idx = -1
        best_similarity = -1
        
        for idx, stored_pattern in enumerate(self.patterns):
            stored_flat = stored_pattern.flatten()
            similarity = np.sum(stored_flat == current_state) / self.pattern_size
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_idx = idx
        
        match_name = self.pattern_names[best_match_idx] if best_match_idx >= 0 else "Unknown"
        
        return {
            'match_name': match_name,
            'confidence': float(best_similarity),
            'iterations': iteration + 1
        }


def create_digit_pattern(digit):
    """Create 10x10 pattern for digits 0-9"""
    pattern = np.ones((10, 10)) * -1
    
    if digit == 0:
        for i in range(10):
            pattern[i, 2] = 1
            pattern[i, 7] = 1
        pattern[0, 3:7] = 1
        pattern[9, 3:7] = 1
            
    elif digit == 1:
        for i in range(10):
            pattern[i, 4:6] = 1
        pattern[0, 4:6] = 1
        pattern[9, 3:7] = 1
        
    elif digit == 2:
        pattern[0, 2:8] = 1
        pattern[4, 2:8] = 1
        pattern[9, 2:8] = 1
        for i in range(1, 4):
            pattern[i, 7] = 1
        for i in range(5, 9):
            pattern[i, 2] = 1
            
    elif digit == 3:
        pattern[0, 2:8] = 1
        pattern[4, 2:8] = 1
        pattern[9, 2:8] = 1
        for i in range(10):
            pattern[i, 7] = 1
            
    elif digit == 4:
        for i in range(10):
            pattern[i, 2] = 1
            pattern[i, 7] = 1
        pattern[4, 3:7] = 1
        pattern[0, 4:6] = 1
        
    elif digit == 5:
        pattern[0, 2:8] = 1
        pattern[4, 2:8] = 1
        pattern[9, 2:8] = 1
        for i in range(1, 4):
            pattern[i, 2] = 1
        for i in range(5, 9):
            pattern[i, 7] = 1
            
    elif digit == 6:
        pattern[0, 2:8] = 1
        pattern[4, 2:8] = 1
        pattern[9, 2:8] = 1
        for i in range(10):
            pattern[i, 2] = 1
        for i in range(5, 9):
            pattern[i, 7] = 1
            
    elif digit == 7:
        pattern[0, 2:8] = 1
        for i in range(10):
            pattern[i, 7] = 1
            
    elif digit == 8:
        pattern[0, 2:8] = 1
        pattern[4, 2:8] = 1
        pattern[9, 2:8] = 1
        for i in range(10):
            pattern[i, 2] = 1
            pattern[i, 7] = 1
            
    elif digit == 9:
        pattern[0, 2:8] = 1
        pattern[4, 2:8] = 1
        pattern[9, 2:8] = 1
        for i in range(10):
            pattern[i, 7] = 1
        for i in range(1, 4):
            pattern[i, 2] = 1
            
    return pattern


def get_digit_patterns():
    """Get all digit patterns 0-9"""
    patterns = []
    names = []
    for digit in range(10):
        patterns.append(create_digit_pattern(digit))
        names.append(str(digit))
    return patterns, names


_trained_network = None

def get_trained_network():
    global _trained_network
    if _trained_network is None:
        patterns, names = get_digit_patterns()
        network = HopfieldNetwork(pattern_size=100)
        network.train(patterns, names)
        _trained_network = network
    return _trained_network