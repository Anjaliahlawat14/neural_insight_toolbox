# 🧠 NeuraFlow
   NEURAL NETWORK TOOLBOX

A multi-functional AI-powered analytics toolkit that integrates machine learning models, neural networks, computer vision, and sentiment analysis to provide intelligent insights and predictions.

---

## 🌐 LIVE DEMO

**Access the deployed application here:**  
🔗 [**https://neuralflow-p1f3.onrender.com/**](https://neuralflow-p1f3.onrender.com/)

> *Note: It may take 15-20 seconds to wake up after inactivity. Please refresh the page if needed.*

---

## 🚀 FEATURES

### 1. 🧠 Perceptron Logic Gates
- Learn neural network fundamentals with interactive logic gates
- Train perceptrons for AND, OR, NAND, NOR gates in real-time
- Choose from Step, Sigmoid, or ReLU activation functions

### 2. 📉 Gradient Descent Visualizer
- Compare three optimization algorithms:
  - Batch Gradient Descent (BGD)
  - Stochastic Gradient Descent (SGD)
  - Mini-Batch Gradient Descent (MBGD)
- Visualize loss convergence and regression fitting

### 3. 📊 Neural Network Sales Predictor
- Custom implementation of MLP (Multi-Layer Perceptron) from scratch
- Predicts units sold using regression models
- Supports real-world dataset (sports footwear)
- Features: Hidden layer configuration, early stopping, validation split

### 4. 👤 Face Detection System
- Real-time face detection using Haar Cascades
- Live webcam capture with face counting
- Eye and smile detection
- Face count statistics and visualization

### 5. 🎙️ Sentiment Analysis
- Text-based sentiment detection (Positive / Neutral / Negative)
- Voice input support with Speech Recognition
- Smart handling of negations (e.g., "not good" → Negative)
- Personalized suggestions based on detected sentiment

### 6. 🧠 Hopfield Network - Digit Recognition
- Associative memory model for pattern recognition
- Recognizes handwritten digits (0-9)
- 10×10 bipolar pattern conversion
- Confidence scoring for predictions
- Perfect for understanding content-addressable memory

### 7. 📝 LSTM Predictor
- **Next Word Prediction:** Predicts the next word in a sentence
- Trained on 100+ common English phrases
- Context-aware suggestions (bigram, trigram, quadgram)
- **Number Sequence Prediction:** Recognizes patterns in sequences
  - Arithmetic progressions (1,2,3,4,5 → 6)
  - Geometric progressions (2,4,8,16 → 32)
  - Fibonacci sequences (1,1,2,3,5 → 8)
  - Square/Cube numbers

### 8. 🎨 Interactive UI
- Built with Streamlit
- Real-time predictions and visualizations
- Modern gradient design with smooth animations
- Responsive layout for all devices

---

## 🛠️ TECH STACK

| Category | Technologies |
|----------|--------------|
| **Programming** | Python 3.11 |
| **Core Libraries** | NumPy, Pandas, Scikit-learn |
| **Visualization/UI** | Streamlit, Matplotlib, Plotly |
| **Computer Vision** | OpenCV, Haar Cascades |
| **NLP** | TextBlob, NLTK, SpeechRecognition |
| **Neural Networks** | Custom implementation (NumPy) |
| **ML/AI** | LSTM (Custom), Hopfield Network |
| **Image Processing** | Pillow, SciPy |

---

## 📂 PROJECT STRUCTURE
sports/
- app.py # Main Streamlit app
- model.py # ML models & data processing
- perceptron.py # Perceptron implementation
- global_sports_footwear_sales.csv # Dataset
- requirements.txt # Dependencies
- README.md # Project documentation

---

## ▶️ HOW TO RUN THE PROJECT

### Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/neural-insight-nexus.git
cd neural-insight-nexus

pip install -r requirements.txt

streamlit run app.py
```
---

## 📊 DATASET
- Global Sports Footwear Sales Dataset
- Includes features like:
- Brand, Category, Gender
- Price, Discount, Rating
- Sales Channel, Country

---

## 🎯 USE CASES
- Business sales prediction
- Learning machine learning from scratch
- Understanding optimization algorithms
-Real-time sentiment analysis
