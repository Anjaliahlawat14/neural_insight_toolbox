import sys
sys.path = [p for p in sys.path if "nldl" not in p.lower()]

import os
import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import json
import io
import time
import streamlit.components.v1 as components

# Import new features
import cv2
from face_detection import FaceDetector
from sentiment_analysis import SentimentAnalyzer
from lstm_model import LSTMPredictor, COMMON_PHRASES, COMMON_SEQUENCES
from hopfield_network import get_trained_network
import matplotlib.pyplot as plt

# # from hopfield_network import HopfieldNetwork, AlphabetPatterns
# from hopfield_network import get_trained_hopfield_network

# # Use the trained network for recognition
# hopfield = get_trained_hopfield_network()
# result = hopfield.recall(your_drawing_pattern)

# from hopfield_network import *

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("⚠️ Plotly not installed. Using basic visualizations. Install with: `pip install plotly`")

st.set_page_config(
    page_title="Neuraflow",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========== CUSTOM CSS ==========
st.markdown("""
<style>
    /* Modern gradient background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        min-height: 100vh;
    }
    
    /* Main header styling with animation */
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 25px;
        margin-bottom: 2.5rem;
        color: white;
        box-shadow: 0 12px 35px rgba(0,0,0,0.2);
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 1px, transparent 1px);
        background-size: 50px 50px;
        animation: float 20s linear infinite;
    }
    @keyframes float {
        0% { transform: translate(0, 0) rotate(0deg); }
        100% { transform: translate(-50px, -50px) rotate(360deg); }
    }
    .main-title {
        font-size: 3.5rem;
        font-weight: 900;
        margin-bottom: 0.8rem;
        letter-spacing: -0.5px;
        text-shadow: 0 4px 8px rgba(0,0,0,0.2);
        background: linear-gradient(45deg, #fff, #e0e7ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .main-subtitle {
        font-size: 1.4rem;
        opacity: 0.95;
        font-weight: 400;
        letter-spacing: 0.5px;
    }
    
    /* Custom panel styling */
    .custom-panel {
        background: white;
        padding: 2.2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.08);
        border: 1px solid rgba(226, 232, 240, 0.8);
        margin-bottom: 2.5rem;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .custom-panel:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.15);
    }
    
    /* Panel title */
    .panel-title {
        color: #2d3748;
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        position: relative;
        border-bottom: 3px solid #667eea;
    }
    
    /* Option cards */
    .option-card {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border: 2px solid #eef2f7;
        transition: all 0.3s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        cursor: pointer;
        margin-bottom: 1.5rem;
    }
    .option-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.2);
        border-color: #667eea;
    }
    .option-icon {
        font-size: 3.5rem;
        margin-bottom: 1rem;
        color: #667eea;
    }
    .option-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2d3748;
        margin-bottom: 0.8rem;
    }
    .option-description {
        color: #718096;
        font-size: 0.9rem;
        line-height: 1.5;
        margin-bottom: 1rem;
    }
    
    /* Enhanced buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        transition: all 0.3s ease;
        font-size: 1rem;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }
    
    /* Feature highlight */
    .feature-highlight {
        display: flex;
        align-items: center;
        margin: 1rem 0;
        padding: 1rem;
        background: rgba(102, 126, 234, 0.05);
        border-radius: 12px;
        border-left: 4px solid #667eea;
    }
    
    /* Grid layout */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.5rem;
        margin: 1.5rem 0;
    }
    
    /* Gate cards */
    .gate-card {
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: 700;
    }
    
    /* Perceptron info styling */
    .perceptron-info {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
    }
    
    .perceptron-info-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #2d3748;
        margin-bottom: 0.5rem;
    }
    
    .concept-item {
        display: flex;
        align-items: center;
        margin: 0.5rem 0;
    }
    
    .concept-icon {
        background: #667eea;
        color: white;
        padding: 0.3rem 0.6rem;
        border-radius: 8px;
        margin-right: 0.8rem;
        min-width: 36px;
        text-align: center;
    }
    
    .gate-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 0.5rem;
        margin: 1rem 0;
    }
    
    .gate-item {
        padding: 0.8rem;
        border-radius: 10px;
        text-align: center;
        transition: transform 0.2s ease;
    }
    
    .gate-item:hover {
        transform: translateY(-2px);
    }
    
    .gate-name {
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 0.2rem;
    }
    
    .gate-desc {
        font-size: 0.9rem;
        color: #4a5568;
    }
    
    .limitation-box {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.05) 0%, rgba(245, 158, 11, 0.05) 100%);
        padding: 1.2rem;
        border-radius: 15px;
        border-left: 4px solid #EF4444;
        margin-top: 1.5rem;
    }
    
    /* Metrics display */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        text-align: center;
        border: 1px solid #e2e8f0;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.1);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #667eea;
        margin: 0.5rem 0;
    }
    
    .metric-name {
        font-size: 1rem;
        color: #4a5568;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #718096;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Prediction display */
    .prediction-display {
        background: linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%);
        border-radius: 20px;
        padding: 2.5rem;
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 10px 25px rgba(78, 205, 196, 0.3);
    }
    
    .prediction-value {
        font-size: 3.5rem;
        font-weight: 900;
        margin: 1rem 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Summary card */
    .summary-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .summary-key {
        font-weight: 600;
        color: #4a5568;
        font-size: 0.9rem;
    }
    
    .summary-value {
        font-weight: 700;
        color: #2d3748;
        font-size: 1rem;
        margin-top: 0.3rem;
    }
    
    /* Quick stats */
    .quick-stat {
        background: rgba(102, 126, 234, 0.08);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #667eea;
        margin: 0.3rem 0;
    }
    
    .stat-label {
        font-size: 0.8rem;
        color: #4a5568;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Webcam container styling */
    .webcam-container {
        max-width: 480px;
        margin: 0 auto;
    }
    
    /* Hopfield canvas styling */
    .hopfield-grid {
        display: grid;
        grid-template-columns: repeat(10, 1fr);
        gap: 2px;
        background: #f0f0f0;
        padding: 10px;
        border-radius: 10px;
    }
    .hopfield-cell {
        aspect-ratio: 1;
        border: 1px solid #ccc;
        border-radius: 4px;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .hopfield-cell:hover {
        transform: scale(1.05);
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# ========== SESSION STATE ==========
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = 'units_sold'
if 'model_type' not in st.session_state:
    st.session_state.model_type = None
if 'gate_data' not in st.session_state:
    st.session_state.gate_data = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'file_processed' not in st.session_state:
    st.session_state.file_processed = False
if 'train_clicked' not in st.session_state:
    st.session_state.train_clicked = False
if 'selected_tab_index' not in st.session_state:
    st.session_state.selected_tab_index = 0
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Dataset"
if 'training_in_progress' not in st.session_state:
    st.session_state.training_in_progress = False
if 'gd_data' not in st.session_state:
    st.session_state.gd_data = None
if 'gd_type' not in st.session_state:
    st.session_state.gd_type = None
if 'gd_result' not in st.session_state:
    st.session_state.gd_result = None
if 'show_gd_details' not in st.session_state:
    st.session_state.show_gd_details = False

# New session states for face detection and sentiment analysis
if 'face_detector' not in st.session_state:
    st.session_state.face_detector = FaceDetector()
if 'sentiment_analyzer' not in st.session_state:
    st.session_state.sentiment_analyzer = SentimentAnalyzer()
if 'face_detection_result' not in st.session_state:
    st.session_state.face_detection_result = None
if 'sentiment_result' not in st.session_state:
    st.session_state.sentiment_result = None

# LSTM session states
if 'lstm_predictor' not in st.session_state:
    st.session_state.lstm_predictor = LSTMPredictor()
if 'text_model_trained' not in st.session_state:
    st.session_state.text_model_trained = False
if 'sequence_model_trained' not in st.session_state:
    st.session_state.sequence_model_trained = False
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'sequence_result' not in st.session_state:
    st.session_state.sequence_result = None
if 'current_text' not in st.session_state:
    st.session_state.current_text = ""

# Hopfield Network session states
if 'hopfield_canvas' not in st.session_state:
    st.session_state.hopfield_canvas = np.ones((10, 10)) * -1
if 'hopfield_trained' not in st.session_state:
    st.session_state.hopfield_trained = False
if 'hopfield_network' not in st.session_state:
    st.session_state.hopfield_network = None
if 'recognition_result' not in st.session_state:
    st.session_state.recognition_result = None

# ========== HEADER ==========
st.markdown("""
<div class="main-header">
    <div class="main-title">🧠 Neuraflow</div>
    <div class="main-subtitle">Interactive Neural Network Learning Platform</div>
</div>
""", unsafe_allow_html=True)

# ========== HOME PAGE ==========
if st.session_state.current_page == 'home':
    st.markdown('<div class="home-page">', unsafe_allow_html=True)
    
    st.markdown("<div class='custom-panel'>", unsafe_allow_html=True)
    
    st.markdown("""
    # 🎯 Welcome to Neural Network Predictor
    
    **Choose Your Learning Path:**
    """)
    
    # Create four columns for the options
    col1, col2, col3, col4 = st.columns(4, gap="large")
    
    with col1:
        st.markdown("""
        <div class='option-card' id='perceptron-card'>
            <div class='option-icon'>⚡</div>
            <div class='option-title'>Perceptron Logic Gates</div>
            <div class='option-description'>
                Learn neural networks with interactive logic gates. Train perceptrons for AND, OR, NAND, NOR in real-time.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Select Perceptron", key="perceptron_home", use_container_width=True):
            st.session_state.current_page = 'perceptron'
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class='option-card' id='gradient-card'>
            <div class='option-icon'>📉</div>
            <div class='option-title'>Gradient Descent</div>
            <div class='option-description'>
                Visualize and compare different gradient descent algorithms: 
                Batch, Stochastic, and Mini-Batch.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Select Gradient Descent", key="gradient_home", use_container_width=True):
            st.session_state.current_page = 'gradient_descent'
            st.rerun()
    
    with col3:
        st.markdown("""
        <div class='option-card' id='neural-card'>
            <div class='option-icon'>📊</div>
            <div class='option-title'>Sales Predictor</div>
            <div class='option-description'>
                Advanced sales forecasting powered by neural networks. 
                Predict units sold with real-world business insights.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Select Sales Predictor", key="neural_network_home", use_container_width=True):
            st.session_state.current_page = 'neural_network'
            st.rerun()
    
    with col4:
        st.markdown("""
        <div class='option-card' id='face-card'>
            <div class='option-icon'>👤</div>
            <div class='option-title'>Face Detection</div>
            <div class='option-description'>
                Real-time face detection from webcam. Count faces and get statistics.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Select Face Detection", key="face_detection_home", use_container_width=True):
            st.session_state.current_page = 'face_detection'
            st.rerun()
    
    # Second row for additional features
    st.markdown("<br>", unsafe_allow_html=True)
    col5, col6, col7, col8 = st.columns(4, gap="large")
    
    with col5:
        st.markdown("""
        <div class='option-card' id='sentiment-card'>
            <div class='option-icon'>🎙️</div>
            <div class='option-title'>Sentiment Analysis</div>
            <div class='option-description'>
                Analyze sentiment from text using advanced NLP models.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Select Sentiment Analysis", key="sentiment_home", use_container_width=True):
            st.session_state.current_page = 'sentiment_analysis'
            st.rerun()
    
    with col6:
        st.markdown("""
        <div class='option-card' id='hopfield-card'>
            <div class='option-icon'>🧠</div>
            <div class='option-title'>Hopfield Pattern Recognition</div>
            <div class='option-description'>
                Draw letters on a canvas and let the Hopfield Network recognize them!
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Select Hopfield Network", key="hopfield_home", use_container_width=True):
            st.session_state.current_page = 'hopfield'
            st.rerun()
    
    with col7:
        st.markdown("""
        <div class='option-card' id='lstm-card'>
            <div class='option-icon'>📝</div>
            <div class='option-title'>LSTM Predictor</div>
            <div class='option-description'>
                Predict next words in a sentence or next numbers in a sequence using LSTM neural networks.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Select LSTM Predictor", key="lstm_home", use_container_width=True):
            st.session_state.current_page = 'lstm'
            st.rerun()
    
    with col8:
        st.markdown("""
        <div class='option-card' id='coming-soon-card'>
            <div class='option-icon'>🚀</div>
            <div class='option-title'>More Features</div>
            <div class='option-description'>
                More exciting features coming soon!
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='margin-top: 2rem; padding: 1.5rem; background: rgba(102, 126, 234, 0.05); border-radius: 20px;'>
        <h3 style='color: #2d3748; margin-bottom: 1rem;'>✨ Key Features</h3>
        <div class='feature-highlight'>
            <span style='font-size: 1.5rem; margin-right: 1rem;'>🎮</span>
            <div>
                <strong>Interactive Learning</strong>
                <div style='color: #718096;'>Hands-on experience with perceptrons and neural networks</div>
            </div>
        </div>
        <div class='feature-highlight'>
            <span style='font-size: 1.5rem; margin-right: 1rem;'>📈</span>
            <div>
                <strong>Real-world Applications</strong>
                <div style='color: #718096;'>Sales forecasting with practical business insights</div>
            </div>
        </div>
        <div class='feature-highlight'>
            <span style='font-size: 1.5rem; margin-right: 1rem;'>👤</span>
            <div>
                <strong>Face Detection</strong>
                <div style='color: #718096;'>Real-time face detection from webcam</div>
            </div>
        </div>
        <div class='feature-highlight'>
            <span style='font-size: 1.5rem; margin-right: 1rem;'>🎙️</span>
            <div>
                <strong>Sentiment Analysis</strong>
                <div style='color: #718096;'>Text sentiment analysis with smart suggestions</div>
            </div>
        </div>
        <div class='feature-highlight'>
            <span style='font-size: 1.5rem; margin-right: 1rem;'>🧠</span>
            <div>
                <strong>Hopfield Network</strong>
                <div style='color: #718096;'>Pattern recognition with associative memory</div>
            </div>
        </div>
        <div class='feature-highlight'>
            <span style='font-size: 1.5rem; margin-right: 1rem;'>📝</span>
            <div>
                <strong>LSTM Predictor</strong>
                <div style='color: #718096;'>Next word and sequence prediction</div>
            </div>
        </div>
    </div>
    
    
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ========== PERCEPTRON LOGIC GATES PAGE ==========
elif st.session_state.current_page == 'perceptron':
    st.markdown('<div class="perceptron-page">', unsafe_allow_html=True)
    
    if st.button("← Back to Home", key="back_perceptron"):
        st.session_state.current_page = 'home'
        st.rerun()
    
    st.markdown("<div class='panel-title'>⚡ Perceptron Logic Gates</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<div class='custom-panel'>", unsafe_allow_html=True)
        
        gate_type = st.selectbox(
            "Select Logic Gate",
            ["AND", "OR", "NAND", "NOR", "XOR", "NOT"],
            key="gate_select"
        )
        
        try:
            from perceptron import get_gate_table
            gate_data = get_gate_table(gate_type)
        except:
            gate_data = {
                'inputs': [(0,0), (0,1), (1,0), (1,1)],
                'outputs': [0, 0, 0, 1],
                'weights': (1, 1),
                'threshold': 1.5,
                'name': 'AND Gate',
                'description': 'Output is 1 only when both inputs are 1'
            }
        
        if gate_data:
            st.session_state.gate_data = gate_data
            
            st.markdown(f"### {gate_data['name']}")
            st.markdown(f"**Description:** {gate_data['description']}")
            
            if gate_type == "NOT":
                x = st.radio("Input:", [0, 1], horizontal=True, key="not_input")
                try:
                    from perceptron import single_input
                    
                    if gate_data['weights']:
                        y, net = single_input(x, gate_data['weights'][0], gate_data['threshold'])
                        
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Weight", f"{gate_data['weights'][0]:.2f}")
                        with col_b:
                            st.metric("Threshold", f"{gate_data['threshold']:.2f}")
                        with col_c:
                            st.metric("Output", f"{y}")
                        
                        st.markdown(f"**Net Input:** `{net:.2f}`")
                except:
                    st.error("Error loading perceptron functions")
            elif gate_type == "XOR":
                st.warning("⚠️ XOR gate requires a multi-layer perceptron and cannot be implemented with a single perceptron.")
                st.info("XOR is not linearly separable. Try using the Neural Network Sales Predictor for complex patterns.")
            else:
                col_a, col_b = st.columns(2)
                with col_a:
                    x1 = st.radio("Input 1:", [0, 1], horizontal=True, key="input1")
                with col_b:
                    x2 = st.radio("Input 2:", [0, 1], horizontal=True, key="input2")
                
                try:
                    from perceptron import two_input
                    
                    if gate_data['weights']:
                        y, net = two_input(x1, x2, gate_data['weights'][0], gate_data['weights'][1], gate_data['threshold'])
                        
                        col_a, col_b, col_c, col_d = st.columns(4)
                        with col_a:
                            st.metric("Weight 1", f"{gate_data['weights'][0]:.2f}")
                        with col_b:
                            st.metric("Weight 2", f"{gate_data['weights'][1]:.2f}")
                        with col_c:
                            st.metric("Threshold", f"{gate_data['threshold']:.2f}")
                        with col_d:
                            st.metric("Output", f"{y}")
                        
                        st.markdown(f"**Net Input:** `{net:.2f}`")
                except:
                    st.error("Error loading perceptron functions")
            
            st.markdown("#### Truth Table")
            if gate_type == "NOT":
                truth_df = pd.DataFrame({
                    'Input': [i[0] for i in gate_data['inputs']],
                    'Output': gate_data['outputs']
                })
            else:
                truth_df = pd.DataFrame({
                    'Input 1': [i[0] for i in gate_data['inputs']],
                    'Input 2': [i[1] for i in gate_data['inputs']],
                    'Output': gate_data['outputs']
                })
            st.dataframe(truth_df, use_container_width=True)
            
            if gate_type != "XOR":
                st.markdown("---")
                st.markdown("### Train Perceptron")
                
                train_gate = st.selectbox("Gate to Train", ["AND", "OR", "NAND", "NOR"], key="train_gate")
                epochs = st.slider("Training Epochs", 1, 50, 10, key="perceptron_epochs")
                activation = st.selectbox("Activation Function", ["step", "sigmoid", "relu"], key="activation")
                
                if st.button("Train Perceptron", type="primary", use_container_width=True):
                    train_data = get_gate_table(train_gate)
                    X_train = np.array(train_data['inputs'])
                    y_train = np.array(train_data['outputs'])
                    
                    from perceptron import train_perceptron
                    weights, threshold, history = train_perceptron(
                        X_train, y_train, lr=0.1, epochs=epochs, activation=activation
                    )
                    
                    st.success("✅ Training completed!")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.write("**Learned Weights:**")
                        st.write(weights)
                    with col_b:
                        st.write("**Learned Threshold:**")
                        st.write(f"{threshold:.4f}")
                    
                    st.markdown("#### Predictions after training:")
                    results = []
                    for inp in X_train:
                        net = np.dot(inp, weights)
                        if activation == 'step':
                            output = 1 if net >= threshold else 0
                        elif activation == 'sigmoid':
                            from perceptron import sigmoid
                            output = 1 if sigmoid(net - threshold) >= 0.5 else 0
                        elif activation == 'relu':
                            from perceptron import relu
                            output = 1 if relu(net - threshold) >= 0.5 else 0
                        results.append(f"Input {inp} → Output {output}")
                    
                    for result in results:
                        st.write(f"• {result}")
                    
                    if history:
                        st.markdown("#### Training History")
                        history_df = pd.DataFrame(history)
                        st.dataframe(history_df[['epoch', 'error']], use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='custom-panel'>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='perceptron-info'>
            <div class='perceptron-info-title'>🧠 What is a Perceptron?</div>
            <div style='color: #4a5568;'>
                A perceptron is the simplest type of artificial neural network. It's a binary classifier that makes decisions based on weighted inputs.
            </div>
        </div>
        
        <div style='margin-bottom: 1.5rem;'>
            <div style='font-size: 1.2rem; font-weight: 700; color: #2d3748; margin-bottom: 0.8rem;'>⚙️ Key Concepts:</div>
            <div class='concept-item'>
                <span class='concept-icon'>⚖️</span>
                <div>
                    <strong>Weights:</strong> Determine the importance of each input
                </div>
            </div>
            <div class='concept-item'>
                <span class='concept-icon'>🎯</span>
                <div>
                    <strong>Threshold (θ):</strong> Activation threshold for output
                </div>
            </div>
            <div class='concept-item'>
                <span class='concept-icon'>⚡</span>
                <div>
                    <strong>Activation Function:</strong> Determines output (0 or 1)
                </div>
            </div>
        </div>
        
        <div class='limitation-box'>
            <div style='font-size: 1.1rem; font-weight: 700; color: #2d3748; margin-bottom: 0.5rem;'>⚠️ Limitations:</div>
            <div style='color: #4a5568; font-size: 0.95rem;'>
                Single-layer perceptrons can only solve linearly separable problems. XOR gate demonstrates the need for multi-layer networks.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ========== GRADIENT DESCENT PAGE ==========
elif st.session_state.current_page == 'gradient_descent':
    st.markdown('<div class="gradient-page">', unsafe_allow_html=True)
    
    if st.button("← Back to Home", key="back_gradient"):
        st.session_state.current_page = 'home'
        st.rerun()
    
    st.markdown("<div class='panel-title'>📉 Gradient Descent Algorithms</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<div class='custom-panel'>", unsafe_allow_html=True)
        
        try:
            from gradient_descent import GradientDescentDemo, get_sample_data
            import matplotlib.pyplot as plt
            
            st.markdown("### 📊 Dataset Configuration")
            
            col_data1, col_data2 = st.columns(2)
            with col_data1:
                use_sample = st.checkbox("Use Sample Dataset", value=True, key="use_sample_data")
            
            if use_sample:
                sample_x, sample_y = get_sample_data()
                x_data = sample_x
                y_data = sample_y
                st.info(f"**Sample Dataset:** X = {x_data}, Y = {y_data}")
            else:
                col_x, col_y = st.columns(2)
                with col_x:
                    x_input = st.text_area("Enter X values (comma separated):", "1, 21, 15, 19, 12", height=100)
                with col_y:
                    y_input = st.text_area("Enter Y values (comma separated):", "15, 24, 35, 37, 10", height=100)
                
                try:
                    x_data = [float(x.strip()) for x in x_input.split(',')]
                    y_data = [float(y.strip()) for y in y_input.split(',')]
                    
                    if len(x_data) != len(y_data):
                        st.error("X and Y must have the same number of values!")
                        st.stop()
                    
                    if len(x_data) < 2:
                        st.error("At least 2 data points are required!")
                        st.stop()
                    
                    st.success(f"Dataset loaded: {len(x_data)} data points")
                except:
                    st.error("Invalid input format! Please enter numbers separated by commas.")
                    st.stop()
            
            st.markdown("### ⚙️ Algorithm Settings")
            
            gd_type = st.selectbox(
                "Select Gradient Descent Type",
                ["Batch Gradient Descent (BGD)", 
                 "Stochastic Gradient Descent (SGD)", 
                 "Mini-Batch Gradient Descent (MBGD)"],
                key="gd_type_select"
            )
            
            col_params1, col_params2, col_params3 = st.columns(3)
            with col_params1:
                learning_rate = st.number_input(
                    "Learning Rate (α)",
                    min_value=0.0001,
                    max_value=1.0,
                    value=0.001,
                    step=0.0001,
                    format="%.4f",
                    key="lr_gd"
                )
            with col_params2:
                epochs = st.slider(
                    "Number of Epochs",
                    min_value=1,
                    max_value=100,
                    value=10,
                    key="epochs_gd"
                )
            with col_params3:
                if "Mini-Batch" in gd_type:
                    batch_size = st.slider(
                        "Batch Size",
                        min_value=1,
                        max_value=len(x_data),
                        value=min(2, len(x_data)),
                        key="batch_size_gd"
                    )
                else:
                    batch_size = 1
            
            gd = GradientDescentDemo(x_data, y_data)
            
            col_run1, col_run2 = st.columns([3, 1])
            with col_run1:
                if st.button("🚀 Run Gradient Descent", type="primary", use_container_width=True):
                    with st.spinner("Running gradient descent..."):
                        if "Batch" in gd_type:
                            result = gd.batch_gradient_descent(lr=learning_rate, epochs=epochs)
                        elif "Stochastic" in gd_type:
                            result = gd.stochastic_gradient_descent(lr=learning_rate, epochs=epochs)
                        else:
                            result = gd.mini_batch_gradient_descent(
                                lr=learning_rate, 
                                epochs=epochs, 
                                batch_size=batch_size
                            )
                        
                        st.session_state.gd_result = result
                        st.session_state.gd_data = (x_data, y_data)
                        st.session_state.gd_type = gd_type
                        st.success("✅ Gradient descent completed!")
            
            with col_run2:
                st.session_state.show_gd_details = st.checkbox(
                    "Show Details", 
                    value=st.session_state.show_gd_details,
                    key="show_details_gd"
                )
            
            if st.session_state.gd_result:
                result = st.session_state.gd_result
                
                st.markdown("---")
                st.markdown("### 📈 Results")
                
                col_res1, col_res2, col_res3 = st.columns(3)
                with col_res1:
                    st.metric("Final Weight (w)", f"{result['final_weight']:.6f}")
                with col_res2:
                    final_loss = result['history'][-1]['total_loss'] if result['history'] else 0
                    st.metric("Final Loss", f"{final_loss:.4f}")
                with col_res3:
                    st.metric("Algorithm", result['type'])
                
                st.markdown("#### 📊 Visualizations")
                
                try:
                    fig1, fig2 = gd.visualize_results()
                    
                    col_viz1, col_viz2 = st.columns(2)
                    with col_viz1:
                        st.markdown("##### Regression Fit")
                        st.pyplot(fig1)
                    
                    if fig2:
                        with col_viz2:
                            st.markdown("##### Loss Convergence")
                            st.pyplot(fig2)
                
                except Exception as e:
                    st.warning(f"Could not generate visualizations: {e}")
                
                if st.session_state.show_gd_details:
                    st.markdown("#### 🔍 Detailed Steps")
                    
                    for epoch_data in result['history']:
                        epoch_num = epoch_data['epoch']
                        
                        with st.expander(f"Epoch {epoch_num} - Weight: {epoch_data['weight']:.6f}, Loss: {epoch_data['total_loss']:.4f}", expanded=False):
                            if 'details' in epoch_data:
                                step_data = []
                                for i, step in enumerate(epoch_data['details']):
                                    step_data.append({
                                        'Step': i+1,
                                        'x': step['x'],
                                        'y': step['y'],
                                        'ŷ': f"{step['ypred']:.3f}",
                                        'Loss': f"{step['loss']:.3f}",
                                        'Gradient': f"{step['gradient']:.3f}"
                                    })
                                
                                if step_data:
                                    df_steps = pd.DataFrame(step_data)
                                    st.dataframe(df_steps, use_container_width=True)
        
        except ImportError as e:
            st.error(f"❌ Could not import gradient_descent module: {e}")
            st.info("Make sure `gradient_descent.py` is in the same directory as `app.py`")
        except Exception as e:
            st.error(f"❌ Error in gradient descent: {str(e)}")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='custom-panel'>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='perceptron-info'>
            <div class='perceptron-info-title'>📉 What is Gradient Descent?</div>
            <div style='color: #4a5568;'>
                Gradient Descent is an optimization algorithm used to minimize functions, 
                especially in machine learning for training models.
            </div>
        </div>
        
        <div style='margin-bottom: 1.5rem;'>
            <div style='font-size: 1.2rem; font-weight: 700; color: #2d3748; margin-bottom: 0.8rem;'>⚙️ Key Concepts:</div>
            <div class='concept-item'>
                <span class='concept-icon'>🎯</span>
                <div>
                    <strong>Loss Function:</strong> Measures prediction error
                </div>
            </div>
            <div class='concept-item'>
                <span class='concept-icon'>📉</span>
                <div>
                    <strong>Gradient:</strong> Direction of steepest ascent
                </div>
            </div>
            <div class='concept-item'>
                <span class='concept-icon'>⚡</span>
                <div>
                    <strong>Learning Rate:</strong> Step size for updates
                </div>
            </div>
        </div>
        
        <div class='limitation-box'>
            <div style='font-size: 1.1rem; font-weight: 700; color: #2d3748; margin-bottom: 0.5rem;'>💡 Tips:</div>
            <div style='color: #4a5568; font-size: 0.95rem;'>
                1. Small learning rate for stable convergence<br>
                2. BGD: Stable but slow for large datasets<br>
                3. SGD: Fast but noisy updates<br>
                4. MBGD: Balance between speed and stability
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ========== FACE DETECTION PAGE ==========
elif st.session_state.current_page == 'face_detection':
    st.markdown('<div class="face-detection-page">', unsafe_allow_html=True)
    
    if st.button("← Back to Home", key="back_face"):
        st.session_state.current_page = 'home'
        st.rerun()
    
    st.markdown("<div class='panel-title'>👤 Face Detection System</div>", unsafe_allow_html=True)
    
    # Only Live Webcam
    st.markdown("### Live Webcam Face Detection")
    st.markdown("Capture from webcam for real-time face detection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        duration = st.slider(
            "Capture Duration (seconds)",
            min_value=2,
            max_value=10,
            value=5,
            key="webcam_duration"
        )
    
    with col2:
        st.markdown("")
        st.markdown("")
        if st.button("🎥 Start Webcam Capture", type="primary", use_container_width=True):
            try:
                # Create placeholders
                video_placeholder = st.empty()
                status_placeholder = st.empty()
                
                frames_captured = []
                final_stats = None
                
                # Start webcam capture with streaming
                status_placeholder.info("🎬 Starting webcam... Please wait.")
                
                generator = st.session_state.face_detector.live_webcam_detection_with_stream(duration)
                
                for frame, stat in generator:
                    if frame is not None:
                        # Show live video feed with smaller size
                        video_placeholder.image(frame, use_container_width=False, width=480, caption="Live Webcam Feed")
                        
                        # Show current face count
                        if 'current_faces' in stat:
                            status_placeholder.info(f"🎥 Capturing... {stat['elapsed']:.1f}/{duration} seconds | 👤 Faces: {stat['current_faces']}")
                        
                        frames_captured.append(frame)
                    else:
                        # Capture completed
                        final_stats = stat
                
                # Display final results
                if final_stats and final_stats.get('success', False):
                    status_placeholder.success("✅ Capture completed!")
                    
                    st.markdown("---")
                    st.markdown("### 📸 Captured Frames")
                    
                    # Show a few sample frames
                    if frames_captured:
                        sample_indices = [0, len(frames_captured)//2, -1] if len(frames_captured) > 2 else range(len(frames_captured))
                        sample_cols = st.columns(min(3, len(sample_indices)))
                        
                        for idx, col in zip(sample_indices, sample_cols):
                            if idx < len(frames_captured):
                                with col:
                                    st.image(frames_captured[idx], use_container_width=True)
                                    st.caption(f"Frame {idx + 1}")
                    
                    # Display statistics
                    st.markdown("---")
                    st.markdown("### 📊 Webcam Analysis Results")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📸 Frames Captured", final_stats.get('total_frames', 0))
                    with col2:
                        st.metric("👤 Avg Faces/Frame", f"{final_stats.get('avg_faces', 0):.2f}")
                    with col3:
                        st.metric("🔝 Max Faces Detected", final_stats.get('max_faces', 0))
                    
                    if final_stats.get('max_faces', 0) > 0:
                        st.success(f"🎉 Great! Detected up to {final_stats['max_faces']} faces!")
                    else:
                        st.info("ℹ️ No faces detected. Make sure your face is clearly visible and well-lit.")
                        
                    # Create a simple chart of face counts over time
                    if PLOTLY_AVAILABLE and 'face_counts' in final_stats and final_stats['face_counts']:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            y=final_stats['face_counts'],
                            mode='lines+markers',
                            name='Face Count',
                            line=dict(color='#667eea', width=2),
                            marker=dict(size=6)
                        ))
                        fig.update_layout(
                            title='Face Count During Capture',
                            xaxis_title='Frame Number',
                            yaxis_title='Number of Faces',
                            height=300
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    if final_stats and 'error' in final_stats:
                        status_placeholder.error(f"❌ {final_stats['error']}")
                    else:
                        status_placeholder.error("❌ Webcam capture failed. Please try again.")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("""
                **Troubleshooting tips:**
                1. Make sure your webcam is connected and not in use
                2. Check browser permissions for camera access
                3. Try refreshing the page and allowing camera access
                4. If using a laptop, ensure the camera is enabled
                """)
    
    st.markdown("---")
    st.markdown("""
    <div class='perceptron-info'>
        <div class='perceptron-info-title'>📖 About Face Detection</div>
        <div style='color: #4a5568;'>
            Face detection uses computer vision algorithms to identify and locate human faces in real-time from your webcam feed.
        </div>
    </div>
    
    <div class='feature-grid'>
        <div class='gate-item' style='background: rgba(102, 126, 234, 0.1);'>
            <div class='gate-name'>🎯 How it works</div>
            <div class='gate-desc'>Uses Haar Cascades to detect facial features in real-time</div>
        </div>
        <div class='gate-item' style='background: rgba(16, 185, 129, 0.1);'>
            <div class='gate-name'>⚡ Real-time</div>
            <div class='gate-desc'>Processes video frames live from your webcam</div>
        </div>
        <div class='gate-item' style='background: rgba(245, 158, 11, 0.1);'>
            <div class='gate-name'>📊 Statistics</div>
            <div class='gate-desc'>Provides face count and analysis metrics</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ========== SENTIMENT ANALYSIS PAGE ==========
elif st.session_state.current_page == 'sentiment_analysis':
    st.markdown('<div class="sentiment-page">', unsafe_allow_html=True)
    
    if st.button("← Back to Home", key="back_sentiment"):
        st.session_state.current_page = 'home'
        st.rerun()
    
    st.markdown("<div class='panel-title'>🎙️ Sentiment Analysis</div>", unsafe_allow_html=True)
    
    # Two tabs: Text Input and Live Microphone (removed Audio Upload)
    tab1, tab2 = st.tabs(["✍️ Text Input", "🎙️ Live Microphone"])
    
    with tab1:
        st.markdown("### Text Sentiment Analysis")
        st.markdown("Enter text to analyze its sentiment")
        
        text_input = st.text_area(
            "Enter your text here:",
            height=150,
            placeholder="Example: I am not feeling good today",
            key="sentiment_text"
        )
        
        if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
            if text_input.strip():
                with st.spinner("Analyzing sentiment..."):
                    result = st.session_state.sentiment_analyzer.analyze_text(text_input)
                    st.session_state.sentiment_result = result
                    
                    sentiment_color = {
                        'Positive': '#10B981',
                        'Negative': '#EF4444',
                        'Neutral': '#F59E0B'
                    }
                    
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, {sentiment_color[result['sentiment']]}20 0%, white 100%); 
                                padding: 2rem; border-radius: 20px; margin: 1rem 0;'>
                        <div style='text-align: center;'>
                            <div style='font-size: 1.2rem; color: #4a5568;'>Sentiment</div>
                            <div style='font-size: 3rem; font-weight: 800; color: {sentiment_color[result['sentiment']]};'>
                                {result['sentiment']}
                            </div>
                            <div style='font-size: 1rem; color: #718096; margin-top: 0.5rem;'>
                                Confidence: {result['confidence']:.1%}
                            </div>
                            <div style='font-size: 0.8rem; color: #718096; margin-top: 0.3rem;'>
                                Model: {result.get('model_used', 'N/A')}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Polarity Score", f"{result['polarity']:.3f}")
                    with col2:
                        st.metric("Subjectivity", f"{result['subjectivity']:.3f}")
                    with col3:
                        st.metric("Positive Words", result['positive_count'])
                    with col4:
                        st.metric("Negative Words", result['negative_count'])
                    
                    st.markdown("### 💡 Suggestions")
                    for suggestion in result['suggestions']:
                        st.markdown(f"• {suggestion}")
                    
                    with st.expander("📝 View Analyzed Text"):
                        st.write(result['text_analyzed'])
            else:
                st.warning("Please enter some text to analyze.")
    
    with tab2:
        st.markdown("### Live Microphone Sentiment Analysis")
        st.markdown("Speak into your microphone for real-time sentiment analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            duration = st.slider(
                "Recording Duration (seconds)",
                min_value=3,
                max_value=10,
                value=5,
                key="mic_duration"
            )
        
        with col2:
            st.markdown("")
            st.markdown("")
            if st.button("🎤 Start Recording", type="primary", use_container_width=True):
                with st.spinner(f"Recording for {duration} seconds..."):
                    try:
                        result = st.session_state.sentiment_analyzer.live_microphone_analysis(duration)
                        
                        if 'error' in result:
                            st.error(f"Error: {result['error']}")
                        else:
                            sentiment_color = {
                                'Positive': '#10B981',
                                'Negative': '#EF4444',
                                'Neutral': '#F59E0B'
                            }
                            
                            st.success("✅ Recording completed!")
                            
                            st.markdown(f"""
                            <div style='background: linear-gradient(135deg, {sentiment_color[result['sentiment']]}20 0%, white 100%); 
                                        padding: 2rem; border-radius: 20px; margin: 1rem 0;'>
                                <div style='text-align: center;'>
                                    <div style='font-size: 1.2rem; color: #4a5568;'>Detected Sentiment</div>
                                    <div style='font-size: 3rem; font-weight: 800; color: {sentiment_color[result['sentiment']]};'>
                                        {result['sentiment']}
                                    </div>
                                    <div style='font-size: 1rem; color: #718096; margin-top: 0.5rem;'>
                                        Confidence: {result['confidence']:.1%}
                                    </div>
                                    <div style='font-size: 0.8rem; color: #718096; margin-top: 0.3rem;'>
                                        Model: {result.get('model_used', 'N/A')}
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown("### 🎤 Recognized Speech")
                            st.info(result['recognized_text'])
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Polarity Score", f"{result['polarity']:.3f}")
                            with col2:
                                st.metric("Subjectivity", f"{result['subjectivity']:.3f}")
                            
                            st.markdown("### 💡 Suggestions")
                            for suggestion in result['suggestions']:
                                st.markdown(f"• {suggestion}")
                                
                    except Exception as e:
                        st.error(f"Error accessing microphone: {str(e)}")
                        st.info("Please make sure you have a microphone connected and permissions granted.")
    
    st.markdown("---")
    st.markdown("""
    <div class='perceptron-info'>
        <div class='perceptron-info-title'>📖 About Sentiment Analysis</div>
        <div style='color: #4a5568;'>
            Sentiment analysis uses natural language processing to determine the emotional tone behind words.
        </div>
    </div>
    
    <div class='feature-grid'>
        <div class='gate-item' style='background: rgba(102, 126, 234, 0.1);'>
            <div class='gate-name'>🎯 How it works</div>
            <div class='gate-desc'>Uses advanced NLP models for accurate sentiment detection</div>
        </div>
        <div class='gate-item' style='background: rgba(16, 185, 129, 0.1);'>
            <div class='gate-name'>🧠 Smart Analysis</div>
            <div class='gate-desc'>Handles negations like "not good" correctly</div>
        </div>
        <div class='gate-item' style='background: rgba(245, 158, 11, 0.1);'>
            <div class='gate-name'>🎙️ Voice Input</div>
            <div class='gate-desc'>Speak directly into your microphone for analysis</div>
        </div>
        <div class='gate-item' style='background: rgba(102, 126, 234, 0.1);'>
            <div class='gate-name'>💡 Smart Suggestions</div>
            <div class='gate-desc'>Provides personalized suggestions based on your text</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ========== LSTM PREDICTOR PAGE ==========


# ========== LSTM PREDICTOR PAGE ==========
elif st.session_state.current_page == 'lstm':
    st.markdown('<div class="lstm-page">', unsafe_allow_html=True)
    
    if st.button("← Back to Home", key="back_lstm"):
        st.session_state.current_page = 'home'
        st.rerun()
    
    st.markdown("<div class='panel-title'>📝 LSTM Predictor</div>", unsafe_allow_html=True)
    
    # Initialize session states
    if 'lstm_predictor' not in st.session_state:
        from lstm_model import LSTMPredictor
        st.session_state.lstm_predictor = LSTMPredictor()
    if 'text_model_trained' not in st.session_state:
        st.session_state.text_model_trained = False
    if 'sequence_model_trained' not in st.session_state:
        st.session_state.sequence_model_trained = False
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
    if 'sequence_result' not in st.session_state:
        st.session_state.sequence_result = None
    if 'current_text' not in st.session_state:
        st.session_state.current_text = ""
    if 'text_input_value' not in st.session_state:
        st.session_state.text_input_value = ""
    
    # Create tabs for different modes
    tab1, tab2 = st.tabs(["📖 Next Word Prediction", "🔢 Number Sequence Prediction"])
    
    with tab1:
        st.markdown("### 📖 Next Word Prediction")
        st.markdown("Write a sentence and get suggestions for the next word")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Train button
            if not st.session_state.text_model_trained:
                if st.button("🎓 Train Text Model", type="primary", use_container_width=True):
                    with st.spinner("Training text prediction model..."):
                        st.session_state.lstm_predictor.train_text_model()
                        st.session_state.text_model_trained = True
                        st.success("✅ Text model trained successfully on 100+ common phrases!")
                        st.rerun()
            else:
                st.success("✅ Text model is ready!")
            
            st.markdown("---")
            
            # Callback for text change
            def on_text_change():
                st.session_state.current_text = st.session_state.text_input_widget
                st.session_state.prediction_result = None
            
            # Input text - using session state value properly
            user_text = st.text_area(
                "Enter your sentence:",
                height=100,
                placeholder="Example: the cat sat on the",
                key="text_input_widget",
                value=st.session_state.text_input_value,
                on_change=on_text_change
            )
            
            # Number of predictions
            num_predictions = st.slider(
                "Number of suggestions:",
                min_value=1,
                max_value=5,
                value=3,
                key="num_predictions"
            )
            
            # Predict button
            col_predict, col_clear = st.columns(2)
            with col_predict:
                if st.button("🔮 Predict Next Word", type="primary", use_container_width=True):
                    if not st.session_state.text_model_trained:
                        st.warning("⚠️ Please train the text model first!")
                    elif not user_text.strip():
                        st.warning("⚠️ Please enter some text!")
                    else:
                        predictions = st.session_state.lstm_predictor.predict_next_word(
                            user_text, num_predictions
                        )
                        st.session_state.prediction_result = predictions
                        st.session_state.current_text = user_text
            
            with col_clear:
                if st.button("🗑️ Clear", use_container_width=True):
                    st.session_state.prediction_result = None
                    st.session_state.current_text = ""
                    st.session_state.text_input_value = ""
                    st.rerun()
            
            # Show predictions
            if st.session_state.get('prediction_result'):
                st.markdown("### 💡 Suggestions:")
                
                st.markdown(f"**Your input:** _{st.session_state.current_text}_")
                st.markdown("**Click a word to add it to your sentence:**")
                
                # Display predictions as buttons
                pred_cols = st.columns(len(st.session_state.prediction_result))
                for idx, word in enumerate(st.session_state.prediction_result):
                    with pred_cols[idx]:
                        if st.button(f"📝 {word}", key=f"suggest_word_{idx}", use_container_width=True):
                            # Add the word to the text
                            new_text = st.session_state.current_text + " " + word
                            st.session_state.text_input_value = new_text
                            st.session_state.current_text = new_text
                            st.session_state.prediction_result = None
                            st.rerun()
                
                # Show complete sentence examples
                st.markdown("---")
                st.markdown("**Complete sentence examples:**")
                for word in st.session_state.prediction_result[:3]:
                    st.info(f"💬 {st.session_state.current_text} **{word}**")
        
        with col2:
            st.markdown("### ℹ️ How it works")
            st.markdown("""
            **Context-Aware Text Prediction**
            
            The model learns patterns from 100+ common English phrases:
            
            **Examples that work well:**
            - "the cat sat on" → **the**, **a**, **his**
            - "I am at" → **home**, **work**, **school**, **the office**
            - "the weather is" → **nice**, **good**, **bad**, **cold**
            - "thank you for" → **your**, **the**, **helping**
            
            **What the model learned:**
            - "the cat sat on the" → **mat**, **chair**, **floor**, **couch**
            - "the dog ran" → **away**, **fast**, **quickly**
            - "I am going" → **to**, **home**, **there**
            - "she is very" → **happy**, **kind**, **smart**, **beautiful**
            
            **Tips for best results:**
            - Start with common phrases
            - Use 3-4 words for better context
            - Click on suggested words to build sentences
            """)
            
            if st.session_state.text_model_trained:
                with st.expander("📚 Sample training data"):
                    st.markdown("""
                    **The model was trained on phrases like:**
                    - the cat sat on the mat
                    - the dog ran in the park
                    - I am at home right now
                    - I am at work all day
                    - what is your name
                    - how are you today
                    - thank you for your help
                    - the weather is nice today
                    - she is very happy
                    - he is very smart
                    - i am going to the store
                    - i am going to school
                    """)
    
    with tab2:
        st.markdown("### 🔢 Number Sequence Prediction")
        st.markdown("Enter a sequence of numbers and predict the next one")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.info("✅ Sequence model is always ready! No training needed.")
            
            st.markdown("---")
            
            # Input sequence
            st.markdown("#### Enter your sequence:")
            
            col_seq1, col_seq2 = st.columns(2)
            with col_seq1:
                sequence_input = st.text_input(
                    "Numbers (comma-separated):",
                    placeholder="1, 2, 3, 4, 5",
                    key="sequence_input"
                )
            
            with col_seq2:
                st.markdown("#### Quick examples:")
                example_sequences = {
                    "Arithmetic": "1, 2, 3, 4, 5",
                    "Even numbers": "2, 4, 6, 8, 10",
                    "Odd numbers": "1, 3, 5, 7, 9",
                    "Fibonacci": "1, 1, 2, 3, 5",
                    "Squares": "1, 4, 9, 16, 25",
                    "Cubes": "1, 8, 27, 64, 125",
                    "Powers of 2": "2, 4, 8, 16, 32",
                    "Decreasing": "100, 90, 80, 70, 60"
                }
                selected_example = st.selectbox("Select example:", ["Custom"] + list(example_sequences.keys()), key="example_selector")
                
                if selected_example != "Custom" and selected_example:
                    sequence_input = example_sequences[selected_example]
            
            # Process sequence
            numbers = []
            if sequence_input:
                try:
                    numbers = [int(x.strip()) for x in sequence_input.split(',')]
                    st.markdown(f"**Your sequence:** {numbers}")
                    
                    # Show detected pattern
                    if len(numbers) >= 2:
                        diff = numbers[1] - numbers[0]
                        if diff > 0:
                            st.markdown(f"📈 **Detected pattern:** Increasing by {diff}")
                        elif diff < 0:
                            st.markdown(f"📉 **Detected pattern:** Decreasing by {abs(diff)}")
                        else:
                            st.markdown(f"➡️ **Detected pattern:** Constant value")
                except:
                    st.error("Please enter valid numbers separated by commas")
                    numbers = []
            
            # Predict button
            if st.button("🔮 Predict Next Number", type="primary", use_container_width=True):
                if len(numbers) < 2:
                    st.warning("⚠️ Please enter at least 2 numbers!")
                else:
                    prediction = st.session_state.lstm_predictor.predict_next_number(numbers)
                    if prediction is not None:
                        st.session_state.sequence_result = prediction
                        st.success(f"✅ Predicted next number: **{prediction}**")
                    else:
                        st.warning("Could not predict the next number. Try a different sequence pattern!")
            
            # Show result with visualization
            if st.session_state.get('sequence_result'):
                st.markdown("---")
                st.markdown("### 📊 Sequence Visualization")
                
                # Show the sequence with prediction
                extended_sequence = numbers + [st.session_state.sequence_result]
                
                # Create visualization
                import plotly.graph_objects as go
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(len(numbers))),
                    y=numbers,
                    mode='lines+markers',
                    name='Original Sequence',
                    line=dict(color='#667eea', width=3),
                    marker=dict(size=10, color='#667eea')
                ))
                fig.add_trace(go.Scatter(
                    x=[len(numbers)],
                    y=[st.session_state.sequence_result],
                    mode='markers',
                    name='Predicted',
                    marker=dict(color='#10b981', size=20, symbol='star', line=dict(color='white', width=2))
                ))
                
                fig.update_layout(
                    title='Number Sequence with Prediction',
                    xaxis_title='Position in Sequence',
                    yaxis_title='Value',
                    height=350,
                    showlegend=True,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.info(f"🔮 **Complete sequence:** {', '.join(map(str, extended_sequence))}")
                
                # Button to clear result
                if st.button("🗑️ Clear Result", use_container_width=True):
                    st.session_state.sequence_result = None
                    st.rerun()
        
        with col2:
            st.markdown("### ℹ️ How it works")
            st.markdown("""
            **Pattern Recognition for Sequences**
            
            The model automatically detects multiple pattern types:
            
            **1. Arithmetic Sequences:**
            - 1, 2, 3, 4, 5 → **6**
            - 10, 20, 30, 40 → **50**
            
            **2. Geometric Sequences:**
            - 2, 4, 8, 16 → **32**
            - 3, 9, 27, 81 → **243**
            
            **3. Special Sequences:**
            - Fibonacci: 1, 1, 2, 3, 5 → **8**
            - Squares: 1, 4, 9, 16 → **25**
            - Cubes: 1, 8, 27, 64 → **125**
            - Powers of 2: 2, 4, 8, 16 → **32**
            
            **4. Alternating Patterns:**
            - 1, 2, 1, 2, 1 → **2**
            
            **Tips for best results:**
            - Enter at least 3-4 numbers
            - Try different pattern types
            - Use the example dropdown for testing
            """)
            
            with st.expander("📊 Pattern examples to try:"):
                st.markdown("""
                | Pattern Type | Sequence | Next Number |
                |-------------|----------|-------------|
                | Counting | 1, 2, 3, 4, 5 | 6 |
                | Even numbers | 2, 4, 6, 8, 10 | 12 |
                | Odd numbers | 1, 3, 5, 7, 9 | 11 |
                | Multiples of 5 | 5, 10, 15, 20 | 25 |
                | Fibonacci | 1, 1, 2, 3, 5 | 8 |
                | Squares | 1, 4, 9, 16 | 25 |
                | Cubes | 1, 8, 27, 64 | 125 |
                | Powers of 2 | 2, 4, 8, 16 | 32 |
                | Decreasing | 100, 90, 80, 70 | 60 |
                | Alternating | 1, 2, 1, 2, 1 | 2 |
                """)
    
    st.markdown("---")
    
    # Educational content
    with st.expander("📚 Learn About LSTM & Markov Chains", expanded=False):
        col_info1, col_info2 = st.columns(2)
        
        with col_info1:
            st.markdown("""
            ### 🧠 How Text Prediction Works
            
            **Markov Chains** are used for text prediction:
            
            1. **Training**: The model analyzes hundreds of sentences
            2. **Pattern Learning**: It learns which words commonly follow which sequences
            3. **Context Window**: Looks at 1-3 previous words for better accuracy
            4. **Prediction**: Suggests the most likely next words
            
            ### 🔢 How Number Prediction Works
            
            **Multiple Pattern Detectors:**
            
            1. **Arithmetic Progression**: Constant difference
            2. **Geometric Progression**: Constant ratio
            3. **Fibonacci Sequence**: Sum of previous two
            4. **Power Sequences**: Squares, cubes, powers of 2
            5. **Alternating Patterns**: Oscillating values
            """)
        
        with col_info2:
            st.markdown("""
            ### 📈 Real-World Applications
            
            **Text Prediction:**
            - Email auto-completion
            - Search engine suggestions
            - Chatbot responses
            - Code completion
            
            **Sequence Prediction:**
            - Stock market forecasting
            - Weather prediction
            - Sales forecasting
            - Time series analysis
            
            ### 💡 Pro Tips
            
            - Use clear, common phrases for text prediction
            - Enter 3-4 numbers for accurate sequence prediction
            - The more context, the better the prediction
            - Click on suggested words to build sentences
            """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ========== HOPFIELD NETWORK PATTERN RECOGNITION PAGE ==========

# ========== HOPFIELD NETWORK DIGIT RECOGNITION PAGE ==========
# ========== HOPFIELD NETWORK DIGIT RECOGNITION PAGE ==========
elif st.session_state.current_page == 'hopfield':
    st.markdown('<div class="hopfield-page">', unsafe_allow_html=True)
    
    if st.button("← Back to Home", key="back_hopfield"):
        st.session_state.current_page = 'home'
        st.rerun()
    
    st.markdown("<div class='panel-title'>🔢 Hopfield Network - Digit Recognition (0-9)</div>", unsafe_allow_html=True)
    
    from hopfield_network import get_trained_network
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    
    # Initialize session state
    if 'hopfield_prediction' not in st.session_state:
        st.session_state.hopfield_prediction = None
    if 'hopfield_confidence' not in st.session_state:
        st.session_state.hopfield_confidence = 0
    
    col1, col2 = st.columns([1.2, 0.8])
    
    with col1:
        st.markdown("### ✏️ Draw or Upload a Digit")
        st.markdown("Draw a digit from **0 to 9** on paper or in any image editor")
        
        st.info("💡 **Tip:** Draw a clear digit on white paper with black ink, take a photo, then upload here.")
        
        uploaded_file = st.file_uploader("Upload a digit image (JPG/PNG)", type=["jpg", "jpeg", "png"], key="digit_uploader")
        
        if uploaded_file is not None:
            # Load and process image
            img = Image.open(uploaded_file)
            
            # Convert to grayscale and resize
            img_gray = img.convert('L')
            img_resized = img_gray.resize((10, 10), Image.Resampling.LANCZOS)
            img_array = np.array(img_resized)
            
            # Display the processed image
            col_display1, col_display2, col_display3 = st.columns([1, 2, 1])
            with col_display2:
                st.image(img_resized.resize((200, 200)), caption="Processed Image (10x10 pixels)", use_container_width=True)
            
            if st.button("🔍 Recognize Digit", type="primary", use_container_width=True):
                # Convert to bipolar pattern (-1 for white, 1 for black)
                pattern = np.where(img_array < 128, 1, -1)
                
                # Get prediction
                network = get_trained_network()
                result = network.recall(pattern)
                
                if result:
                    st.session_state.hopfield_prediction = result['match_name']
                    st.session_state.hopfield_confidence = result['confidence']
                    st.rerun()
                else:
                    st.error("Recognition failed!")
        
        st.markdown("<p style='text-align: center; color: #666; font-size: 12px; margin-top: 10px;'>📤 Upload a clear image of a digit (0-9) for recognition</p>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🎯 Recognition Result")
        
        # Network status
        network = get_trained_network()
        st.success(f"✅ Hopfield Network ready with {len(network.patterns)} patterns (digits 0-9)")
        
        st.markdown("---")
        
        # Display prediction
        if st.session_state.get('hopfield_prediction'):
            predicted = st.session_state.hopfield_prediction
            confidence = st.session_state.hopfield_confidence
            
            # Color coding
            if confidence > 0.7:
                bg_color = "#10B981"
                emoji = "🎉"
            elif confidence > 0.5:
                bg_color = "#F59E0B"
                emoji = "👍"
            else:
                bg_color = "#EF4444"
                emoji = "🤔"
            
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, {bg_color}15 0%, {bg_color}05 100%); 
                        padding: 2rem; border-radius: 20px; text-align: center; margin: 1rem 0;
                        border: 2px solid {bg_color}40;'>
                <div style='font-size: 1rem; color: #4a5568;'>{emoji} Hopfield Network Recognized</div>
                <div style='font-size: 5rem; font-weight: 800; color: {bg_color}; margin: 0.5rem 0;'>
                    {predicted}
                </div>
                <div style='font-size: 1rem; color: #718096;'>
                    Confidence: {confidence:.1%}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence bar
            st.markdown(f"""
            <div style='margin-top: 1rem;'>
                <div style='font-size: 0.9rem; color: #4a5568; margin-bottom: 0.5rem;'>
                    Recognition Confidence
                </div>
                <div style='height: 12px; background: #e2e8f0; border-radius: 6px; overflow: hidden;'>
                    <div style='height: 100%; width: {confidence*100}%; background: linear-gradient(90deg, {bg_color}, {bg_color}80); border-radius: 6px;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Feedback
            if confidence > 0.7:
                st.success("✅ High confidence! The network is very sure about this digit.")
            elif confidence > 0.5:
                st.info("📝 Moderate confidence. Try a clearer image for better results.")
            else:
                st.warning("⚠️ Low confidence. The network is unsure. Try a different image.")
            
            # Clear button
            if st.button("Clear Result", use_container_width=True):
                st.session_state.hopfield_prediction = None
                st.session_state.hopfield_confidence = 0
                st.rerun()
        else:
            st.info("👈 Upload a digit image and click 'Recognize Digit'!")
            
            st.markdown("""
            <div style='background: #f0fdf4; padding: 1rem; border-radius: 12px; margin-top: 1rem;'>
                <strong>💡 Instructions:</strong><br>
                1. Draw a digit (0-9) on white paper with dark ink<br>
                2. Take a clear photo or scan<br>
                3. Upload using the file uploader<br>
                4. Click "Recognize Digit"<br><br>
                <strong>Tips for best results:</strong><br>
                • Use black on white background<br>
                • Center the digit in the image<br>
                • Make the digit large and clear<br>
                • Avoid shadows, glare, and background clutter<br>
                • Use high contrast images
            </div>
            """, unsafe_allow_html=True)
    
    # Show stored patterns
    with st.expander("🔍 View Stored Digit Patterns", expanded=False):
        st.markdown("These are the patterns the Hopfield network learned for digits 0-9:")
        
        from hopfield_network import get_digit_patterns
        patterns, names = get_digit_patterns()
        
        # Display patterns in 2 rows
        for row in range(2):
            cols = st.columns(5)
            for col_idx in range(5):
                idx = row * 5 + col_idx
                if idx < len(patterns):
                    with cols[col_idx]:
                        pattern = patterns[idx]
                        name = names[idx]
                        
                        # Create visual representation
                        fig, ax = plt.subplots(figsize=(2, 2))
                        ax.imshow(pattern, cmap='coolwarm', vmin=-1, vmax=1, interpolation='nearest')
                        ax.set_title(f"Digit {name}", fontsize=10, fontweight='bold')
                        ax.axis('off')
                        st.pyplot(fig)
                        plt.close(fig)
    
    # Educational content
    st.markdown("---")
    st.markdown("### 🧠 How the Hopfield Network Recognizes Digits")
    
    col_info1, col_info2, col_info3 = st.columns(3)
    
    with col_info1:
        st.markdown("""
        <div style='text-align: center; padding: 1rem; background: rgba(102, 126, 234, 0.05); border-radius: 12px;'>
            <div style='font-size: 2rem;'>📚</div>
            <strong>Training Phase</strong><br>
            <span style='font-size: 0.85rem; color: #666;'>Network learns patterns for digits 0-9 using Hebbian learning</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col_info2:
        st.markdown("""
        <div style='text-align: center; padding: 1rem; background: rgba(102, 126, 234, 0.05); border-radius: 12px;'>
            <div style='font-size: 2rem;'>⚡</div>
            <strong>Recognition Phase</strong><br>
            <span style='font-size: 0.85rem; color: #666;'>Your digit image is converted to a pattern and compared to stored patterns</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col_info3:
        st.markdown("""
        <div style='text-align: center; padding: 1rem; background: rgba(102, 126, 234, 0.05); border-radius: 12px;'>
            <div style='font-size: 2rem;'>🎯</div>
            <strong>Energy Minimization</strong><br>
            <span style='font-size: 0.85rem; color: #666;'>Network finds the closest matching digit by minimizing energy</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ========== NEURAL NETWORK SALES PREDICTOR PAGE ==========
elif st.session_state.current_page == 'neural_network':
    # [Keep your existing neural_network code here unchanged]
    st.markdown('<div class="neural-network-page">', unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        st.markdown("---")
        
        if st.button("🏠 Back to Home", use_container_width=True):
            st.session_state.current_page = 'home'
            st.session_state.uploaded_file = None
            st.session_state.file_processed = False
            st.rerun()
        
        st.markdown("---")
        
        col_switch1, col_switch2 = st.columns(2)
        with col_switch1:
            if st.button("⚡ Perceptron", use_container_width=True):
                st.session_state.current_page = 'perceptron'
                st.rerun()
        
        with col_switch2:
            if st.button("📉 Gradient", use_container_width=True):
                st.session_state.current_page = 'gradient_descent'
                st.rerun()
        
        st.markdown("---")
        st.markdown("### 📁 File Upload")
        
        uploaded_file_sidebar = st.file_uploader(
            "Upload CSV File",
            type=["csv"],
            help="Upload your dataset for sales prediction",
            key="sidebar_uploader"
        )
        
        if uploaded_file_sidebar:
            if 'uploaded_file' not in st.session_state or st.session_state.uploaded_file != uploaded_file_sidebar:
                st.session_state.uploaded_file = uploaded_file_sidebar
                st.session_state.file_processed = False
        
        if st.session_state.get('uploaded_file'):
            st.markdown("---")
            st.markdown("### 📊 Current File")
            st.info(f"📄 **{st.session_state.uploaded_file.name}**")
            
            if st.button("🗑️ Remove File", use_container_width=True):
                st.session_state.uploaded_file = None
                st.session_state.file_processed = False
                st.rerun()
    
    if st.button("← Back to Home", key="back_neural"):
        st.session_state.current_page = 'home'
        st.session_state.uploaded_file = None
        st.session_state.file_processed = False
        st.rerun()
    
    st.markdown("<div class='panel-title'>📊 Neural Network Sales Predictor</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='custom-panel'>", unsafe_allow_html=True)
    
    uploaded_file = st.session_state.get('uploaded_file', None)
    
    if not uploaded_file:
        st.markdown("""
        <div style='border: 3px dashed #667eea; border-radius: 20px; padding: 3rem; text-align: center; background: rgba(102, 126, 234, 0.05); margin: 2rem 0;'>
            <div style='font-size: 3rem; margin-bottom: 1.5rem;'>📁</div>
            <div style='font-size: 1.5rem; font-weight: 700; color: #2d3748; margin-bottom: 1rem;'>
                Upload Your Dataset
            </div>
            <div style='color: #4a5568; margin-bottom: 2rem;'>
                Please upload your CSV file using the sidebar uploader on the left
            </div>
            <div style='color: #667eea; font-weight: 700;'>
                → Use the sidebar to upload your CSV file
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        with open("data.csv", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            from model import load_footwear_data, NeuralNetworkScratch, regression_metrics
            
            with st.spinner("🔄 Loading and preprocessing data..."):
                data = load_footwear_data("data.csv", target='units_sold')
                
                if data is None:
                    st.error("❌ Failed to load data. Please check your CSV file format.")
                    st.stop()
                
                st.session_state.data = data
                st.session_state.file_processed = True
            
            df = data["df"]
            feature_names = data["feature_names"]
            X_train = data["X_train"]
            y_train = data["y_train"]
            X_test = data["X_test"]
            y_test = data["y_test"]
            
            categorical_cols = data.get("categorical_cols", [])
            categorical_values = {}
            
            for col in categorical_cols:
                if col in df.columns:
                    original_df = pd.read_csv("data.csv")
                    if col in original_df.columns:
                        categorical_values[col] = sorted(original_df[col].astype(str).unique().tolist())
                    else:
                        categorical_values[col] = ["Unknown"]
                else:
                    categorical_values[col] = ["Unknown"]
            
            tab_names = ["Dataset", "Train Model", "Visualization", "Predict Units Sold"]
            
            if 'selected_tab_index' not in st.session_state:
                st.session_state.selected_tab_index = 0
            
            tab_cols = st.columns(len(tab_names))
            selected_tab_index = st.session_state.selected_tab_index
            
            for i, tab_name in enumerate(tab_names):
                with tab_cols[i]:
                    if st.button(
                        tab_name,
                        key=f"tab_{i}",
                        use_container_width=True,
                        type="primary" if i == selected_tab_index else "secondary"
                    ):
                        st.session_state.selected_tab_index = i
                        st.rerun()
            
            st.markdown("---")
            
            if selected_tab_index == 0:
                st.markdown("### Dataset Overview")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f"""
                    <div class='quick-stat'>
                        <div class='stat-label'>Total Records</div>
                        <div class='stat-value'>{len(df):,}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class='quick-stat'>
                        <div class='stat-label'>Features</div>
                        <div class='stat-value'>{len(feature_names)}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    target_col = data['target_column']
                    st.markdown(f"""
                    <div class='quick-stat'>
                        <div class='stat-label'>Target Column</div>
                        <div class='stat-value'>{target_col.replace('_', ' ').title()}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col4:
                    if target_col in df.columns:
                        target_mean = df[target_col].mean()
                        st.markdown(f"""
                        <div class='quick-stat'>
                            <div class='stat-label'>Avg Units Sold</div>
                            <div class='stat-value'>{target_mean:,.0f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("#### 📋 Data Preview")
                st.dataframe(df.head(10), use_container_width=True, height=350)
                
                st.markdown("#### 📈 Statistical Summary")
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                if len(numeric_cols) > 0:
                    stats_df = pd.DataFrame({
                        'Feature': numeric_cols,
                        'Mean': [df[col].mean() for col in numeric_cols],
                        'Std Dev': [df[col].std() for col in numeric_cols],
                        'Min': [df[col].min() for col in numeric_cols],
                        'Max': [df[col].max() for col in numeric_cols]
                    })
                    
                    for col in ['Mean', 'Std Dev']:
                        stats_df[col] = stats_df[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
                    
                    st.dataframe(stats_df, use_container_width=True)
            
            elif selected_tab_index == 1:
                st.markdown("### Model Training")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🏗️ Architecture")
                    hidden_size = st.slider(
                        "Number of neurons in hidden layer",
                        8, 256, 64, step=8,
                        key="hidden_size"
                    )
                    
                    activation = st.selectbox(
                        "Select activation function",
                        ["ReLU", "Sigmoid", "Tanh"],
                        index=0,
                        key="activation"
                    )
                
                with col2:
                    st.markdown("#### ⚙️ Training Parameters")
                    learning_rate = st.select_slider(
                        "Select learning rate",
                        options=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
                        value=0.005,
                        key="learning_rate"
                    )
                    
                    epochs = st.slider(
                        "Number of training iterations",
                        100, 5000, 1000, step=100,
                        key="epochs"
                    )
                
                adv_col1, adv_col2 = st.columns(2)
                
                with adv_col1:
                    batch_size = st.selectbox(
                        "Samples per training batch",
                        [16, 32, 64, 128, 256],
                        index=2,
                        key="batch_size"
                    )
                    
                    validation_split = st.slider(
                        "Percentage of data for validation",
                        0.0, 0.5, 0.2, step=0.05,
                        key="validation_split"
                    )
                
                with adv_col2:
                    early_stopping = st.checkbox(
                        "Enable early stopping to prevent overfitting",
                        value=True,
                        key="early_stopping"
                    )
                    
                    if early_stopping:
                        patience = st.slider(
                            "Patience (epochs without improvement)",
                            10, 100, 50, step=5,
                            key="patience"
                        )
                
                st.markdown("---")
                st.markdown("#### 📋 Current Configuration")
                
                config_col1, config_col2, config_col3 = st.columns(3)
                
                with config_col1:
                    st.markdown(f"""
                    <div class='summary-card'>
                        <div class='summary-key'>Architecture</div>
                        <div class='summary-value'>Input: {X_train.shape[1]} → Hidden: {hidden_size} → Output: 1</div>
                    </div>
                    <div class='summary-card'>
                        <div class='summary-key'>Activation</div>
                        <div class='summary-value'>{activation}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with config_col2:
                    st.markdown(f"""
                    <div class='summary-card'>
                        <div class='summary-key'>Learning Rate</div>
                        <div class='summary-value'>{learning_rate}</div>
                    </div>
                    <div class='summary-card'>
                        <div class='summary-key'>Batch Size</div>
                        <div class='summary-value'>{batch_size}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with config_col3:
                    st.markdown(f"""
                    <div class='summary-card'>
                        <div class='summary-key'>Training Samples</div>
                        <div class='summary-value'>{X_train.shape[0]:,}</div>
                    </div>
                    <div class='summary-card'>
                        <div class='summary-key'>Features</div>
                        <div class='summary-value'>{len(feature_names)}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                train_button_key = f"train_button_{hidden_size}_{learning_rate}_{epochs}"
                
                if st.button("🚀 Start Training", type="primary", use_container_width=True, key=train_button_key):
                    st.session_state.training_in_progress = True
                    training_output = st.empty()
                    
                    with training_output.container():
                        st.markdown("### 🚀 Training in Progress...")
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        try:
                            model = NeuralNetworkScratch(
                                input_size=X_train.shape[1],
                                hidden_size=hidden_size,
                                lr=learning_rate,
                                epochs=epochs,
                                batch_size=batch_size
                            )
                            
                            train_start_time = time.time()
                            
                            if validation_split > 0:
                                from sklearn.model_selection import train_test_split
                                X_train_split, X_val, y_train_split, y_val = train_test_split(
                                    X_train, y_train, test_size=validation_split, random_state=42
                                )
                                model.fit(X_train_split, y_train_split, X_val, y_val, verbose=True)
                            else:
                                model.fit(X_train, y_train, verbose=True)
                            
                            train_end_time = time.time()
                            model.training_time = train_end_time - train_start_time
                            
                            preds = model.predict(X_test)
                            
                            if 'scaler_y' in data:
                                preds_original = data['scaler_y'].inverse_transform(preds)
                                y_test_original = data['scaler_y'].inverse_transform(y_test)
                            else:
                                preds_original = preds
                                y_test_original = y_test
                            
                            metrics = regression_metrics(y_test_original, preds_original)
                            
                            st.session_state.model = model
                            st.session_state.model_trained = True
                            st.session_state.predictions = {
                                'actual': y_test_original,
                                'predicted': preds_original,
                                'scaled_actual': y_test,
                                'scaled_predicted': preds
                            }
                            st.session_state.metrics = metrics
                            
                            progress_bar.empty()
                            status_text.empty()
                            
                            st.success("✅ Model trained successfully!")
                            st.session_state.training_in_progress = False
                            
                            st.session_state.selected_tab_index = 1
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Training failed: {str(e)}")
                            import traceback
                            st.error(traceback.format_exc())
                            st.session_state.training_in_progress = False
                
                if st.session_state.model_trained and not st.session_state.training_in_progress:
                    st.markdown("---")
                    st.markdown("### 📊 Training Results")
                    
                    metrics = st.session_state.metrics
                    
                    st.markdown(f"""
                    <div class='metrics-grid'>
                        <div class='metric-card'>
                            <div class='metric-label'>MAE</div>
                            <div class='metric-value'>{metrics['MAE']:.2f}</div>
                            <div class='metric-name'>Mean Absolute Error</div>
                        </div>
                        <div class='metric-card'>
                            <div class='metric-label'>RMSE</div>
                            <div class='metric-value'>{metrics['RMSE']:.2f}</div>
                            <div class='metric-name'>Root Mean Square Error</div>
                        </div>
                        <div class='metric-card'>
                            <div class='metric-label'>R² Score</div>
                            <div class='metric-value'>{metrics['R2']:.4f}</div>
                            <div class='metric-name'>Coefficient of Determination</div>
                        </div>
                        <div class='metric-card'>
                            <div class='metric-label'>MAPE</div>
                            <div class='metric-value'>{metrics['MAPE']:.2f}%</div>
                            <div class='metric-name'>Mean Absolute Percentage Error</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    st.markdown("#### 📈 Additional Metrics")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Explained Variance", f"{metrics.get('Explained Variance', 0):.4f}")
                    with col2:
                        st.metric("MSE", f"{metrics.get('MSE', 0):.4f}")
                    with col3:
                        st.metric("Max Error", f"{metrics.get('Max Error', 0):.2f}")
            
            elif selected_tab_index == 2:
                st.markdown("### 📊 Visualization")
                
                if not st.session_state.model_trained:
                    st.warning("⚠️ Please train a model first in the Training tab.")
                else:
                    if PLOTLY_AVAILABLE and hasattr(st.session_state.model, 'losses') and st.session_state.model.losses:
                        loss_df = pd.DataFrame({
                            'Epoch': range(1, len(st.session_state.model.losses) + 1),
                            'Loss': st.session_state.model.losses
                        })
                        
                        fig = px.line(
                            loss_df, 
                            x='Epoch', 
                            y='Loss',
                            title='📉 Training Loss',
                            line_shape='spline'
                        )
                        
                        if hasattr(st.session_state.model, 'val_losses') and st.session_state.model.val_losses:
                            loss_df['Validation Loss'] = st.session_state.model.val_losses
                            fig.add_scatter(x=loss_df['Epoch'], y=loss_df['Validation Loss'], 
                                           mode='lines', name='Validation Loss', line=dict(color='red', dash='dash'))
                        
                        fig.update_layout(
                            height=400,
                            xaxis_title="Epoch",
                            yaxis_title="Loss",
                            hovermode="x unified"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    if st.session_state.predictions:
                        actual = st.session_state.predictions['actual'].flatten()
                        predicted = st.session_state.predictions['predicted'].flatten()
                        
                        if PLOTLY_AVAILABLE and len(actual) > 0:
                            fig = go.Figure()
                            
                            fig.add_trace(go.Scatter(
                                x=actual,
                                y=predicted,
                                mode='markers',
                                name='Predictions',
                                marker=dict(
                                    color='#667eea',
                                    size=8,
                                    opacity=0.7,
                                    line=dict(width=1, color='white')
                                )
                            ))
                            
                            min_val = min(actual.min(), predicted.min())
                            max_val = max(actual.max(), predicted.max())
                            fig.add_trace(go.Scatter(
                                x=[min_val, max_val],
                                y=[min_val, max_val],
                                mode='lines',
                                line=dict(color='red', dash='dash', width=2),
                                name='Perfect Fit'
                            ))
                            
                            r2 = st.session_state.metrics['R2']
                            fig.add_annotation(
                                x=0.05,
                                y=0.95,
                                xref="paper",
                                yref="paper",
                                text=f"R² = {r2:.4f}",
                                showarrow=False,
                                font=dict(size=14, color="black"),
                                bgcolor="white",
                                bordercolor="black",
                                borderwidth=1,
                                borderpad=4,
                                opacity=0.8
                            )
                            
                            fig.update_layout(
                                title='📈 Actual vs Predicted Units Sold',
                                xaxis_title='Actual Units Sold',
                                yaxis_title='Predicted Units Sold',
                                height=500,
                                showlegend=True,
                                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    if st.session_state.predictions:
                        errors = st.session_state.predictions['actual'].flatten() - st.session_state.predictions['predicted'].flatten()
                        
                        if PLOTLY_AVAILABLE and len(errors) > 0:
                            fig = px.histogram(
                                x=errors,
                                title='📊 Distribution of Prediction Errors',
                                nbins=30,
                                color_discrete_sequence=['#667eea']
                            )
                            
                            fig.update_layout(
                                height=400,
                                xaxis_title="Prediction Error (Actual - Predicted)",
                                yaxis_title="Frequency",
                                bargap=0.1
                            )
                            st.plotly_chart(fig, use_container_width=True)
            
            elif selected_tab_index == 3:
                st.markdown("### 🔮 Make Predictions")
                
                if not st.session_state.model_trained:
                    st.warning("⚠️ Please train a model first in the Training tab.")
                else:
                    st.markdown("""
                    <div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%); padding: 1.5rem; border-radius: 15px; margin-bottom: 1.5rem;'>
                        <div style='font-size: 1.2rem; font-weight: 700; color: #2d3748; margin-bottom: 1rem;'>
                            <span>📝</span> Enter Product Details
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 🏷️ Product Information")
                        input_values = {}
                        
                        for cat_col in categorical_cols:
                            if cat_col in categorical_values:
                                options = categorical_values[cat_col]
                                if options:
                                    default_idx = min(0, len(options) - 1)
                                    label = cat_col.replace('_', ' ').title()
                                    try:
                                        input_values[cat_col] = st.selectbox(
                                            label,
                                            options,
                                            index=default_idx,
                                            key=f"cat_{cat_col}"
                                        )
                                    except:
                                        input_values[cat_col] = options[0] if options else "Unknown"
                    
                    with col2:
                        st.markdown("#### 💰 Pricing & Features")
                        
                        target_mean = df[data['target_column']].mean() if data['target_column'] in df.columns else 0
                        
                        if 'base_price' in df.columns:
                            min_price = float(df['base_price'].min()) if len(df) > 0 else 0
                            max_price = float(df['base_price'].max()) if len(df) > 0 else 500
                            avg_price = float(df['base_price'].mean()) if len(df) > 0 else 100
                            
                            input_values['base_price'] = st.number_input(
                                "Base Price ($)", 
                                min_value=min_price, 
                                max_value=max_price, 
                                value=float(avg_price),
                                step=1.0,
                                key="base_price_input"
                            )
                        
                        if 'discount_percent' in df.columns or 'discount' in df.columns:
                            discount_col = 'discount_percent' if 'discount_percent' in df.columns else 'discount'
                            avg_discount = float(df[discount_col].mean()) if len(df) > 0 else 10
                            
                            input_values['discount_percent'] = st.slider(
                                "Discount %", 
                                min_value=0.0, 
                                max_value=100.0, 
                                value=float(avg_discount),
                                step=0.5,
                                key="discount_input"
                            )
                        
                        if 'customer_rating' in df.columns or 'rating' in df.columns:
                            rating_col = 'customer_rating' if 'customer_rating' in df.columns else 'rating'
                            avg_rating = float(df[rating_col].mean()) if len(df) > 0 else 4.0
                            
                            input_values['customer_rating'] = st.slider(
                                "Customer Rating", 
                                min_value=1.0, 
                                max_value=5.0, 
                                value=float(avg_rating),
                                step=0.1,
                                key="rating_input"
                            )
                        
                        if 'size' in df.columns:
                            avg_size = float(df['size'].mean()) if len(df) > 0 else 9.0
                            
                            input_values['size'] = st.slider(
                                "Size", 
                                min_value=5.0, 
                                max_value=15.0, 
                                value=float(avg_size),
                                step=0.5,
                                key="size_input"
                            )
                    
                    st.markdown("---")
                    st.markdown("#### 📋 Input Summary")
                    
                    summary_cols = st.columns(3)
                    input_keys = list(input_values.keys())
                    
                    chunk_size = (len(input_keys) + 2) // 3
                    for i in range(3):
                        with summary_cols[i]:
                            start_idx = i * chunk_size
                            end_idx = min((i + 1) * chunk_size, len(input_keys))
                            for key in input_keys[start_idx:end_idx]:
                                st.markdown(f"""
                                <div class='summary-card'>
                                    <div class='summary-key'>{key.replace('_', ' ').title()}</div>
                                    <div class='summary-value'>{input_values[key]}</div>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    predict_button_key = "predict_button_" + "_".join([f"{k}_{v}" for k, v in input_values.items()])
                    
                    if st.button("🔮 Predict Units Sold", type="primary", use_container_width=True, key=predict_button_key):
                        try:
                            from model import prepare_prediction_input
                            
                            X_pred = prepare_prediction_input(
                                input_values,
                                feature_names,
                                categorical_cols,
                                data['original_df'],
                                data['scaler_X']
                            )
                            
                            pred_scaled = st.session_state.model.predict(X_pred)
                            
                            if 'scaler_y' in data:
                                prediction = data['scaler_y'].inverse_transform(pred_scaled)[0][0]
                            else:
                                prediction = pred_scaled[0][0]
                            
                            prediction = max(0, float(prediction))
                            prediction = int(round(prediction))
                            
                            st.markdown(f"""
                            <div class='prediction-display'>
                                <div class='prediction-label'>📊 Predicted Units Sold</div>
                                <div class='prediction-value'>
                                    {prediction:,} units
                                </div>
                                <div class='prediction-subtitle'>
                                    Based on current input parameters
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            confidence_score = min(95, max(60, 100 - (abs(prediction - target_mean) / target_mean * 20) if target_mean > 0 else 85))
                            
                            st.markdown(f"""
                            <div style='margin-top: 2rem; text-align: center;'>
                                <div style='font-size: 0.9rem; color: #4a5568; margin-bottom: 0.5rem;'>
                                    ⚡ Confidence Score: <span style='font-weight: 700; color: #10B981;'>{confidence_score:.0f}%</span>
                                </div>
                                <div style='height: 8px; background: #e2e8f0; border-radius: 4px; overflow: hidden;'>
                                    <div style='height: 100%; width: {confidence_score}%; background: linear-gradient(90deg, #10B981, #34D399); border-radius: 4px;'></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.error(f"❌ Prediction failed: {str(e)}")
                            st.info("Please check if all required features are provided in your input.")
        
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            st.info("""
            **Common Issues:**
            1. Make sure your CSV file has the correct format
            2. Ensure the 'units_sold' column exists
            3. Check that all columns have proper data types
            4. Try using the sample format shown on the home page
            """)
    
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ========== FOOTER ==========
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #718096; padding: 2rem 0;'>
    <div>🧠 <strong>Neural Network Predictor</strong> | Academic Edition</div>
    <div style='font-size: 0.8rem; margin-top: 0.5rem;'>Face Detection | Sentiment Analysis | Hopfield Network | LSTM Predictor | Neural Networks | Gradient Descent</div>
</div>
""", unsafe_allow_html=True)