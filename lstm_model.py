# lstm_model.py
import numpy as np
import random
import re
from collections import defaultdict, Counter

class ImprovedTextPredictor:
    """Fixed text predictor with proper context learning"""
    
    def __init__(self):
        self.bigram = defaultdict(Counter)  # 2-word sequences
        self.trigram = defaultdict(Counter)  # 3-word sequences
        self.quadgram = defaultdict(Counter)  # 4-word sequences
        self.word_freq = Counter()
        self.is_trained = False
        
    def clean_text(self, text):
        """Clean and tokenize text"""
        text = text.lower().strip()
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        words = text.split()
        return words
    
    def train(self, texts):
        """Train the model on text data"""
        print("Training text prediction model...")
        
        for text in texts:
            words = self.clean_text(text)
            
            if len(words) < 2:
                continue
                
            # Update word frequencies
            self.word_freq.update(words)
            
            # Build n-grams
            for i in range(len(words) - 1):
                # Bigrams (2 words)
                self.bigram[words[i]][words[i + 1]] += 1
                
                # Trigrams (3 words)
                if i >= 1:
                    context = (words[i - 1], words[i])
                    self.trigram[context][words[i + 1]] += 1
                
                # Quadgrams (4 words)
                if i >= 2:
                    context = (words[i - 2], words[i - 1], words[i])
                    self.quadgram[context][words[i + 1]] += 1
        
        self.is_trained = True
        print(f"Trained: {len(self.bigram)} bigrams, {len(self.trigram)} trigrams, {len(self.quadgram)} quadgrams")
        return True
    
    def get_common_words(self, n=5, exclude=None):
        """Get most common words"""
        common = self.word_freq.most_common(20)
        words = [word for word, count in common if word != exclude]
        return words[:n]
    
    def predict_next_words(self, text, num_predictions=3):
        """Predict next words based on input"""
        if not self.is_trained:
            return ["the", "and", "to", "of", "a"][:num_predictions]
        
        if not text or not text.strip():
            return self.get_common_words(num_predictions)
        
        words = self.clean_text(text)
        
        if not words:
            return self.get_common_words(num_predictions)
        
        # Try different context lengths (most specific first)
        
        # 1. Try quadgram (3 words context)
        if len(words) >= 3:
            context = (words[-3], words[-2], words[-1])
            if context in self.quadgram:
                next_words = self.quadgram[context]
                valid_words = [(w, c) for w, c in next_words.items() if w not in ['<START>', '<END>', '']]
                if valid_words:
                    valid_words.sort(key=lambda x: x[1], reverse=True)
                    return [word for word, count in valid_words[:num_predictions]]
        
        # 2. Try trigram (2 words context)
        if len(words) >= 2:
            context = (words[-2], words[-1])
            if context in self.trigram:
                next_words = self.trigram[context]
                valid_words = [(w, c) for w, c in next_words.items() if w not in ['<START>', '<END>', '']]
                if valid_words:
                    valid_words.sort(key=lambda x: x[1], reverse=True)
                    return [word for word, count in valid_words[:num_predictions]]
        
        # 3. Try bigram (1 word context)
        if len(words) >= 1:
            context = words[-1]
            if context in self.bigram:
                next_words = self.bigram[context]
                valid_words = [(w, c) for w, c in next_words.items() if w not in ['<START>', '<END>', '']]
                if valid_words:
                    valid_words.sort(key=lambda x: x[1], reverse=True)
                    return [word for word, count in valid_words[:num_predictions]]
        
        # Fallback: Return common words
        return self.get_common_words(num_predictions)
    
    def generate_completion(self, text, max_words=5):
        """Generate text completion"""
        if not self.is_trained:
            return "Please train the model first"
        
        words = self.clean_text(text)
        result = words.copy()
        
        for _ in range(max_words):
            next_words = self.predict_next_words(' '.join(result), 3)
            if not next_words:
                break
            # Pick the most confident prediction
            result.append(next_words[0])
        
        return ' '.join(result)


class LSTMPredictor:
    """Main predictor class"""
    
    def __init__(self):
        self.text_model = ImprovedTextPredictor()
        self.sequence_model = None
        self.text_trained = False
        
    def train_text_model(self):
        """Train with comprehensive, realistic text corpus"""
        
        # Extensive training data with common patterns
        training_texts = [
            # "going to" patterns
            "i am going to the store",
            "i am going to school",
            "i am going to work",
            "i am going to home",
            "i am going to the park",
            "i am going to the movies",
            "i am going to the beach",
            "i am going to the gym",
            "i am going to the office",
            "i am going to the hospital",
            "i am going to the restaurant",
            "i am going to the library",
            "we are going to the party",
            "they are going to the game",
            "she is going to the mall",
            "he is going to the airport",
            
            # "the cat" patterns
            "the cat sat on the mat",
            "the cat is sleeping",
            "the cat is eating",
            "the cat is playing",
            "the cat ran away",
            "the cat climbed the tree",
            "the cat caught the mouse",
            "the cat likes milk",
            "the cat has fur",
            
            # "sat on" patterns
            "sat on the chair",
            "sat on the couch",
            "sat on the floor",
            "sat on the bed",
            "sat on the table",
            "sat on the bench",
            "sat on the ground",
            "sat on the roof",
            
            # "I am" patterns
            "i am happy today",
            "i am tired of waiting",
            "i am feeling good",
            "i am at home",
            "i am at work",
            "i am at school",
            "i am at the office",
            "i am very excited",
            "i am so happy",
            "i am not sure",
            
            # "the weather" patterns
            "the weather is nice today",
            "the weather is bad outside",
            "the weather is cold",
            "the weather is hot",
            "the weather is sunny",
            "the weather is rainy",
            "the weather is cloudy",
            
            # Common phrases
            "thank you for your help",
            "thank you very much",
            "thank you so much",
            "how are you doing",
            "how was your day",
            "what is your name",
            "where are you going",
            "when will you arrive",
            "why are you late",
            "can you help me",
            "please let me know",
            "sorry for the trouble",
            "nice to meet you",
            "good luck with that",
            "have a nice day",
            "take care of yourself",
            
            # Action sequences
            "he walked to the door",
            "she opened the window",
            "they closed the gate",
            "we locked the car",
            "i turned on the light",
            "you turned off the tv",
            
            # Descriptive
            "the old man walked slowly",
            "the young girl laughed loudly",
            "the big dog barked fiercely",
            "the small cat purred softly",
            "the fast car zoomed past",
            "the slow turtle crawled along",
            
            # More common patterns
            "i like to eat pizza",
            "i want to go home",
            "i need to sleep",
            "i love to read books",
            "i hate to wake up early",
            "she is very kind",
            "he is very smart",
            "they are very friendly",
            "we are very happy",
            
            # Time expressions
            "i will see you tomorrow",
            "we met yesterday evening",
            "she called me last week",
            "he arrived this morning",
            "they are coming next month"
        ]
        
        self.text_model.train(training_texts)
        self.text_trained = True
        return True
    
    def predict_next_word(self, text, num_predictions=3):
        """Predict next word"""
        if not self.text_trained:
            return ["Train", "model", "first"]
        return self.text_model.predict_next_words(text, num_predictions)
    
    def predict_next_number(self, sequence):
        """Predict next number in sequence"""
        if len(sequence) < 2:
            return None
        
        # Check for arithmetic progression
        diff = sequence[1] - sequence[0]
        is_arithmetic = all(sequence[i+1] - sequence[i] == diff for i in range(len(sequence)-1))
        if is_arithmetic:
            return sequence[-1] + diff
        
        # Check for geometric progression
        if sequence[0] != 0:
            ratio = sequence[1] / sequence[0]
            is_geometric = all(abs(sequence[i+1] / sequence[i] - ratio) < 0.0001 for i in range(len(sequence)-1) if sequence[i] != 0)
            if is_geometric:
                return sequence[-1] * ratio
        
        # Check for Fibonacci
        if len(sequence) >= 3:
            if sequence[2] == sequence[0] + sequence[1]:
                is_fibonacci = all(sequence[i+2] == sequence[i+1] + sequence[i] for i in range(len(sequence)-2))
                if is_fibonacci:
                    return sequence[-1] + sequence[-2]
        
        # Check for squares
        if all(abs(x**0.5 - round(x**0.5)) < 0.0001 for x in sequence):
            next_sqrt = round(sequence[-1]**0.5) + 1
            return next_sqrt ** 2
        
        return None
    
    def generate_completion(self, text, max_words=5):
        """Generate text completion"""
        if not self.text_trained:
            return "Please train the model first"
        return self.text_model.generate_completion(text, max_words)


# For display purposes
COMMON_PHRASES = [
    "the cat sat on the mat",
    "i am going to the store",
    "what is your name",
    "how are you today"
]

COMMON_SEQUENCES = [
    [1, 2, 3, 4, 5],
    [2, 4, 6, 8, 10],
    [1, 1, 2, 3, 5, 8],
    [1, 4, 9, 16, 25]
]