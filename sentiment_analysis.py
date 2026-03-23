# sentiment_analysis.py
import re
import numpy as np
from textblob import TextBlob

# Try to use a lighter transformer model if available
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class SentimentAnalyzer:
    def __init__(self):
        self.classifier = None
        
        # Try to load a lightweight transformer model
        if TRANSFORMERS_AVAILABLE:
            try:
                print("🔄 Loading lightweight sentiment model...")
                self.classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
                self.model_used = "Transformer (DistilBERT)"
                print("✅ Transformer model loaded successfully")
            except Exception as e:
                print(f"⚠️ Could not load transformer model: {e}")
                self.classifier = None
        
        # Fallback to enhanced keyword matching with negation handling
        if self.classifier is None:
            self.model_used = "Enhanced Keyword + TextBlob"
            self.init_keyword_analyzer()
    
    def init_keyword_analyzer(self):
        """Initialize keyword-based sentiment analyzer"""
        self.positive_words = {
            'good': 1.5, 'great': 2.0, 'excellent': 2.5, 'amazing': 2.5, 
            'wonderful': 2.5, 'fantastic': 2.5, 'happy': 2.0, 'love': 2.0, 
            'like': 1.0, 'best': 2.0, 'awesome': 2.0, 'brilliant': 2.0,
            'perfect': 2.0, 'beautiful': 1.5, 'nice': 1.0, 'pleasant': 1.5,
            'enjoy': 1.5, 'pleased': 1.5, 'satisfied': 1.5, 'outstanding': 2.5,
            'superb': 2.5, 'exceptional': 2.5, 'glad': 1.5, 'joy': 2.0,
            'excited': 2.0, 'grateful': 1.5, 'thankful': 1.5, 'blessed': 1.5
        }
        
        self.negative_words = {
            'bad': -1.5, 'terrible': -2.5, 'awful': -2.5, 'horrible': -2.5, 
            'poor': -1.5, 'worst': -2.5, 'hate': -2.0, 'dislike': -1.5, 
            'angry': -2.0, 'sad': -2.0, 'disappointed': -2.0, 'frustrated': -2.0,
            'annoying': -1.5, 'useless': -2.0, 'waste': -1.5, 'disgusting': -2.5,
            'unhappy': -2.0, 'upset': -1.5, 'furious': -2.5, 'pathetic': -2.0,
            'mediocre': -1.0, 'disappointing': -2.0, 'regret': -1.5, 'sorry': -1.0,
            'down': -2.0, 'depressed': -2.5, 'anxious': -2.0, 'stressed': -1.5,
            'tired': -1.0, 'exhausted': -1.5, 'unwell': -1.5
        }
        
        # Negation words
        self.negation_words = {
            'not', 'no', 'never', 'neither', 'nor', "n't", 'cannot', "can't", 
            "don't", "doesn't", "didn't", "won't", "wouldn't", "shouldn't",
            "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't"
        }
        
        # Special negation phrases
        self.negation_phrases = [
            (r'\bnot\s+feeling\s+good\b', 'bad'),
            (r'\bnot\s+feeling\s+well\b', 'unwell'),
            (r'\bnot\s+good\b', 'bad'),
            (r'\bnot\s+great\b', 'bad'),
            (r'\bnot\s+happy\b', 'sad'),
            (r'\bnot\s+sad\b', 'happy'),
            (r'\bnot\s+down\b', 'happy'),
            (r'\bnot\s+bad\b', 'good'),
        ]
    
    def analyze_text(self, text):
        """Analyze sentiment using transformer or keyword method"""
        text = text.strip()
        
        if not text:
            return {
                'sentiment': 'Neutral',
                'confidence': 0,
                'score': 0,
                'polarity': 0,
                'subjectivity': 0,
                'positive_count': 0,
                'negative_count': 0,
                'suggestions': ['Please enter some text to analyze.'],
                'text_analyzed': text,
                'model_used': 'No input'
            }
        
        # Use transformer if available
        if self.classifier is not None:
            try:
                result = self.classifier(text)[0]
                label = result['label']
                score = result['score']
                
                if label == 'POSITIVE':
                    sentiment = 'Positive'
                    confidence = score
                else:
                    sentiment = 'Negative'
                    confidence = score
                
                # Add polarity and subjectivity
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
                
                # Count keywords for display
                positive_count, negative_count = self.count_keywords(text.lower())
                
                # Generate suggestions
                suggestions = self.generate_suggestions(sentiment, text, score)
                
                return {
                    'sentiment': sentiment,
                    'confidence': float(confidence),
                    'score': float(score if sentiment == 'Positive' else -score),
                    'polarity': float(polarity),
                    'subjectivity': float(subjectivity),
                    'positive_count': positive_count,
                    'negative_count': negative_count,
                    'suggestions': suggestions,
                    'text_analyzed': text,
                    'model_used': 'Transformer (DistilBERT)'
                }
            except Exception as e:
                print(f"Transformer error: {e}")
                # Fall through to keyword method
        
        # Fallback to keyword method
        return self.keyword_analysis(text)
    
    def keyword_analysis(self, text):
        """Fallback keyword-based sentiment analysis"""
        processed_text = self.preprocess_text(text)
        final_score, polarity, keyword_score, positive_count, negative_count = self.calculate_sentiment_score(processed_text)
        
        blob = TextBlob(text)
        subjectivity = blob.sentiment.subjectivity
        
        if final_score > 0.15:
            sentiment = 'Positive'
            confidence = min(0.95, 0.6 + (final_score * 0.25))
        elif final_score < -0.15:
            sentiment = 'Negative'
            confidence = min(0.95, 0.6 + (abs(final_score) * 0.25))
        else:
            sentiment = 'Neutral'
            confidence = 0.7
        
        suggestions = self.generate_suggestions(sentiment, text, final_score)
        
        return {
            'sentiment': sentiment,
            'confidence': float(confidence),
            'score': float(final_score),
            'polarity': float(polarity),
            'subjectivity': float(subjectivity),
            'positive_count': positive_count,
            'negative_count': negative_count,
            'suggestions': suggestions,
            'text_analyzed': text,
            'model_used': 'Enhanced Keyword + TextBlob'
        }
    
    def preprocess_text(self, text):
        """Preprocess text for keyword analysis"""
        text = text.lower().strip()
        return text
    
    def handle_special_phrases(self, text):
        """Handle special negation phrases"""
        for pattern, replacement in self.negation_phrases:
            if re.search(pattern, text):
                text = re.sub(pattern, replacement, text)
        return text
    
    def handle_negations(self, text):
        """Handle individual word negations"""
        words = text.split()
        negated_indices = set()
        
        for i, word in enumerate(words):
            if word in self.negation_words:
                for j in range(i + 1, min(i + 4, len(words))):
                    negated_indices.add(j)
        
        return words, negated_indices
    
    def calculate_sentiment_score(self, text):
        """Calculate weighted sentiment score with negation handling"""
        text = self.handle_special_phrases(text)
        words, negated_indices = self.handle_negations(text)
        
        total_score = 0
        word_count = 0
        positive_count = 0
        negative_count = 0
        
        for i, word in enumerate(words):
            is_negated = i in negated_indices
            
            if word in self.positive_words:
                score = self.positive_words[word]
                if is_negated:
                    score = -score
                total_score += score
                word_count += 1
                if score > 0:
                    positive_count += 1
                else:
                    negative_count += 1
                    
            elif word in self.negative_words:
                score = self.negative_words[word]
                if is_negated:
                    score = -score
                total_score += score
                word_count += 1
                if score > 0:
                    positive_count += 1
                else:
                    negative_count += 1
        
        keyword_score = total_score / max(word_count, 1)
        
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if any(neg in text for neg in self.negation_words):
            polarity = -polarity
        
        final_score = (keyword_score * 0.7) + (polarity * 0.3)
        
        return final_score, polarity, keyword_score, positive_count, negative_count
    
    def count_keywords(self, text):
        """Count positive and negative keywords"""
        positive_keywords = list(self.positive_words.keys())
        negative_keywords = list(self.negative_words.keys())
        
        words = text.split()
        positive_count = sum(1 for word in words if word in positive_keywords)
        negative_count = sum(1 for word in words if word in negative_keywords)
        
        return positive_count, negative_count
    
    def generate_suggestions(self, sentiment, text, score):
        """Generate context-aware suggestions"""
        suggestions = []
        text_lower = text.lower()
        
        if sentiment == 'Positive':
            if 'not feeling down' in text_lower or 'not sad' in text_lower:
                suggestions.append("😊 That's wonderful to hear! Recognizing when you're feeling better is a great step.")
                suggestions.append("💪 Keep this positive momentum going!")
                suggestions.append("🌟 Remember this feeling - it shows that things can and do get better.")
            elif 'happy' in text_lower or 'great' in text_lower or 'good' in text_lower:
                suggestions.append("🎉 Great to see you're feeling positive! Share this good energy with someone today.")
                suggestions.append("💫 Take a moment to appreciate what's making you feel this way.")
                suggestions.append("📝 Consider journaling about this positive moment.")
            else:
                suggestions.append("🎉 Great to see positive sentiment! Enjoy this moment.")
                suggestions.append("📝 Consider noting what made you feel this way for future reference.")
        
        elif sentiment == 'Negative':
            if 'not feeling good' in text_lower or 'not good' in text_lower:
                suggestions.append("🌧️ I understand you're not feeling good today. That's completely okay and valid.")
                suggestions.append("💝 Sometimes acknowledging how you feel is the first step to feeling better.")
                suggestions.append("🧘 Take a moment for self-care - a short walk, deep breathing, or talking to someone.")
                suggestions.append("🌟 Remember: This feeling is temporary. Things can and do get better.")
            elif 'sad' in text_lower or 'down' in text_lower:
                suggestions.append("🌧️ It's okay to feel down sometimes. These feelings are valid and temporary.")
                suggestions.append("💝 Talking to someone you trust can help lighten the load.")
                suggestions.append("🧘 Try some deep breathing: breathe in for 4, hold for 4, exhale for 4.")
                suggestions.append("🎵 Listening to uplifting music might help lift your mood.")
            elif 'tired' in text_lower or 'exhausted' in text_lower:
                suggestions.append("😴 Rest is important! Your body and mind need time to recharge.")
                suggestions.append("🛌 Try to prioritize sleep tonight. Even a short break can help.")
                suggestions.append("💧 Stay hydrated and have a healthy snack - sometimes fatigue is physical.")
            elif 'angry' in text_lower or 'frustrated' in text_lower:
                suggestions.append("😤 Frustration is understandable. Take a few deep breaths.")
                suggestions.append("🧘 Try this: breathe in for 4 counts, hold for 4, exhale for 4.")
                suggestions.append("✍️ Writing down what's bothering you can help process these feelings.")
            elif 'bad' in text_lower or 'terrible' in text_lower:
                suggestions.append("💭 It sounds like you're going through a difficult time. What's one small thing that could help?")
                suggestions.append("🌱 Remember that this feeling is temporary. You've gotten through tough moments before.")
            else:
                suggestions.append("💭 I hear that you're having a difficult time. Be kind to yourself.")
                suggestions.append("🌱 This feeling is temporary. Things will get better.")
        
        else:  # Neutral
            if len(text.split()) < 5:
                suggestions.append("📝 Your text is brief. Adding more detail could help better understand your sentiment.")
                suggestions.append("💭 How are you really feeling? Sometimes it helps to elaborate on your thoughts.")
            elif 'okay' in text_lower or 'fine' in text_lower:
                suggestions.append("🤔 'Okay' can sometimes mask deeper feelings. Is there anything specific on your mind?")
                suggestions.append("💫 Even if things feel neutral, that's okay. What could make today a little better?")
            else:
                suggestions.append("📊 Your sentiment appears neutral. Are you feeling balanced, or is there more you'd like to share?")
                suggestions.append("🎯 Setting a small, achievable goal for today might add some positive direction.")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_suggestions = []
        for suggestion in suggestions:
            if suggestion not in seen:
                seen.add(suggestion)
                unique_suggestions.append(suggestion)
        
        return unique_suggestions[:6]


if __name__ == "__main__":
    print("Testing Sentiment Analyzer...")
    analyzer = SentimentAnalyzer()
    
    test_texts = [
        "I am not feeling good today",
        "I am feeling good today",
        "This is terrible, I'm so frustrated",
        "I love this amazing product!"
    ]
    
    for text in test_texts:
        print(f"\n{'='*60}")
        print(f"Text: {text}")
        result = analyzer.analyze_text(text)
        print(f"Sentiment: {result['sentiment']} (Confidence: {result['confidence']:.1%})")
        print(f"Model: {result['model_used']}")
        print("Suggestions:")
        for suggestion in result['suggestions']:
            print(f"  • {suggestion}")