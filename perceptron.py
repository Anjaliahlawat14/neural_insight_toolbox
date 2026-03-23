import numpy as np
import pandas as pd

# ---------------- ACTIVATION ----------------
def activate(value):
    return 1 if value >= 0 else 0

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return max(0, x)

def tanh(x):
    return np.tanh(x)

# ---------------- SINGLE AND TWO INPUT NEURONS ----------------
def single_input(x, w, threshold):
    total = x * w
    output = activate(total - threshold)
    return output, total

def two_input(a, b, w1, w2, threshold):
    total = a * w1 + b * w2
    output = activate(total - threshold)
    return output, total

# ---------------- TRAIN 2-INPUT PERCEPTRON ----------------
def train_perceptron(inputs, targets, lr=0.1, epochs=10, activation='step'):
    weights = np.zeros(inputs.shape[1])
    threshold = 0
    history = []
    
    for ep in range(epochs):
        epoch_errors = []
        for i in range(len(inputs)):
            net = np.dot(inputs[i], weights)
            
            if activation == 'step':
                output = activate(net - threshold)
            elif activation == 'sigmoid':
                output = 1 if sigmoid(net - threshold) >= 0.5 else 0
            elif activation == 'relu':
                output = 1 if relu(net - threshold) >= 0.5 else 0
            else:
                output = activate(net - threshold)
                
            error = targets[i] - output
            epoch_errors.append(abs(error))
            
            # Update weights and threshold
            weights += lr * error * inputs[i]
            threshold -= lr * error
        
        avg_error = np.mean(epoch_errors)
        history.append({
            'epoch': ep + 1,
            'weights': weights.copy(),
            'threshold': threshold,
            'error': avg_error
        })

    return weights, threshold, history

# ---------------- TRAIN 3-INPUT PERCEPTRON ----------------
def train_perceptron_3(inputs, targets, lr=0.1, epochs=5, activation='step'):
    weights = np.zeros(inputs.shape[1])
    threshold = 0
    history = []
    
    for ep in range(epochs):
        epoch_errors = []
        for i in range(len(inputs)):
            net = np.dot(inputs[i], weights)
            
            if activation == 'step':
                output = activate(net - threshold)
            elif activation == 'sigmoid':
                output = 1 if sigmoid(net - threshold) >= 0.5 else 0
            elif activation == 'relu':
                output = 1 if relu(net - threshold) >= 0.5 else 0
            else:
                output = activate(net - threshold)
                
            error = targets[i] - output
            epoch_errors.append(abs(error))
            
            weights += lr * error * inputs[i]
            threshold -= lr * error
        
        avg_error = np.mean(epoch_errors)
        history.append({
            'epoch': ep + 1,
            'weights': weights.copy(),
            'threshold': threshold,
            'error': avg_error
        })

    return weights, threshold, history

# ---------------- LOGIC GATE TRUTH TABLES ----------------
def get_gate_table(gate_type):
    """Return truth table for different logic gates"""
    if gate_type == "AND":
        return {
            'inputs': [(0,0), (0,1), (1,0), (1,1)],
            'outputs': [0, 0, 0, 1],
            'weights': (1, 1),
            'threshold': 1.5,
            'name': 'AND Gate',
            'description': 'Output is 1 only when both inputs are 1'
        }
    elif gate_type == "OR":
        return {
            'inputs': [(0,0), (0,1), (1,0), (1,1)],
            'outputs': [0, 1, 1, 1],
            'weights': (1, 1),
            'threshold': 0.5,
            'name': 'OR Gate',
            'description': 'Output is 1 when at least one input is 1'
        }
    elif gate_type == "NAND":
        return {
            'inputs': [(0,0), (0,1), (1,0), (1,1)],
            'outputs': [1, 1, 1, 0],
            'weights': (-1, -1),
            'threshold': -1.5,
            'name': 'NAND Gate',
            'description': 'Opposite of AND gate'
        }
    elif gate_type == "NOR":
        return {
            'inputs': [(0,0), (0,1), (1,0), (1,1)],
            'outputs': [1, 0, 0, 0],
            'weights': (-1, -1),
            'threshold': -0.5,
            'name': 'NOR Gate',
            'description': 'Opposite of OR gate'
        }
    elif gate_type == "XOR":
        return {
            'inputs': [(0,0), (0,1), (1,0), (1,1)],
            'outputs': [0, 1, 1, 0],
            'weights': None,
            'threshold': None,
            'name': 'XOR Gate',
            'description': 'Output is 1 when inputs are different (requires MLP)'
        }
    elif gate_type == "NOT":
        return {
            'inputs': [(0,), (1,)],
            'outputs': [1, 0],
            'weights': (-1,),
            'threshold': -0.5,
            'name': 'NOT Gate',
            'description': 'Inverts the input'
        }
    return None

# ---------------- VISUALIZATION HELPERS ----------------
def get_perceptron_diagram(weights, threshold, num_inputs=2):
    """Generate ASCII diagram of perceptron"""
    diagram = []
    diagram.append(" " * 10 + "╔══════════════════╗")
    diagram.append(" " * 10 + "║    PERCEPTRON    ║")
    diagram.append(" " * 10 + "╠══════════════════╣")
    
    for i in range(num_inputs):
        arrow = "─" * 8 + f"[w{i+1}={weights[i]:.2f}]" + "─▶"
        diagram.append(f"Input {i+1} {arrow}")
    
    diagram.append(" " * 10 + "│                  │")
    diagram.append(" " * 10 + "│      Σ + φ       │")
    threshold_str = f"θ={threshold:.2f}"
    diagram.append(" " * 10 + f"│   {threshold_str:^14}   │")
    diagram.append(" " * 10 + "│                  │")
    diagram.append(" " * 10 + "│        ↓         │")
    diagram.append(" " * 10 + "╰────────┬─────────╯")
    diagram.append(" " * 21 + "│")
    diagram.append(" " * 21 + "▼")
    diagram.append(" " * 21 + "Output")
    
    return "\n".join(diagram)

# ---------------- TRAINING VISUALIZATION ----------------
def create_training_history_df(history):
    """Convert training history to DataFrame for plotting"""
    df = pd.DataFrame(history)
    return df

def calculate_training_accuracy(weights, threshold, inputs, targets):
    """Calculate accuracy of trained perceptron"""
    correct = 0
    predictions = []
    
    for i in range(len(inputs)):
        net = np.dot(inputs[i], weights)
        output = activate(net - threshold)
        predictions.append(output)
        
        if output == targets[i]:
            correct += 1
    
    accuracy = correct / len(inputs) * 100
    return accuracy, predictions