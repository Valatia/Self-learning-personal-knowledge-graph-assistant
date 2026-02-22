"""
Text processing utilities for REXI.
"""

import re
import string
from typing import List, Dict, Optional
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import spacy

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class TextProcessor:
    """Text processing utilities."""
    
    def __init__(self, language: str = "english"):
        """Initialize text processor."""
        self.language = language
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words(language))
        
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.nlp = None
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/]', ' ', text)
        
        # Remove extra spaces again
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def tokenize_sentences(self, text: str) -> List[str]:
        """Tokenize text into sentences."""
        if not text:
            return []
        
        # Use NLTK for sentence tokenization
        sentences = sent_tokenize(text)
        
        # Clean each sentence
        cleaned_sentences = []
        for sentence in sentences:
            cleaned = self.clean_text(sentence)
            if cleaned and len(cleaned.split()) > 2:  # Keep meaningful sentences
                cleaned_sentences.append(cleaned)
        
        return cleaned_sentences
    
    def tokenize_words(self, text: str) -> List[str]:
        """Tokenize text into words."""
        if not text:
            return []
        
        # Use spaCy if available
        if self.nlp:
            doc = self.nlp(text)
            tokens = [token.text for token in doc if not token.is_space]
        else:
            # Fallback to NLTK
            tokens = word_tokenize(text)
        
        # Clean and normalize tokens
        cleaned_tokens = []
        for token in tokens:
            token = token.lower().strip()
            token = token.strip(string.punctuation)
            
            if token and token not in self.stop_words and len(token) > 2:
                cleaned_tokens.append(token)
        
        return cleaned_tokens
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords from text."""
        if not text:
            return []
        
        # Tokenize and count word frequencies
        words = self.tokenize_words(text)
        word_freq = {}
        
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Return top keywords
        keywords = [word for word, freq in sorted_words[:max_keywords]]
        return keywords
    
    def extract_named_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities using spaCy."""
        if not text or not self.nlp:
            return {}
        
        doc = self.nlp(text)
        entities = {
            "persons": [],
            "organizations": [],
            "locations": [],
            "concepts": [],
            "dates": []
        }
        
        for ent in doc.ents:
            entity_text = ent.text.strip()
            if ent.label_ == "PERSON":
                entities["persons"].append(entity_text)
            elif ent.label_ == "ORG":
                entities["organizations"].append(entity_text)
            elif ent.label_ in ["GPE", "LOC"]:
                entities["locations"].append(entity_text)
            elif ent.label_ == "DATE":
                entities["dates"].append(entity_text)
            elif ent.label_ in ["PRODUCT", "EVENT", "WORK_OF_ART"]:
                entities["concepts"].append(entity_text)
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Chunk text into overlapping segments."""
        if not text:
            return []
        
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            if chunk_words:
                chunk = " ".join(chunk_words)
                chunks.append(chunk)
        
        return chunks
    
    def calculate_readability(self, text: str) -> Dict[str, float]:
        """Calculate basic readability metrics."""
        if not text:
            return {"flesch_score": 0.0, "avg_sentence_length": 0.0}
        
        sentences = self.tokenize_sentences(text)
        words = self.tokenize_words(text)
        
        if not sentences or not words:
            return {"flesch_score": 0.0, "avg_sentence_length": 0.0}
        
        # Average sentence length
        avg_sentence_length = len(words) / len(sentences)
        
        # Simple readability score (lower is easier to read)
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * (len(words) / len(sentences)))
        
        return {
            "flesch_score": max(0, flesch_score),
            "avg_sentence_length": avg_sentence_length
        }
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity using Jaccard similarity."""
        if not text1 or not text2:
            return 0.0
        
        # Get word sets
        words1 = set(self.tokenize_words(self.normalize_text(text1)))
        words2 = set(self.tokenize_words(self.normalize_text(text2)))
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def extract_key_phrases(self, text: str, min_phrase_length: int = 2) -> List[str]:
        """Extract key phrases from text."""
        if not text:
            return []
        
        # Get keywords
        keywords = self.extract_keywords(text, max_keywords=50)
        
        # Find phrases (consecutive keywords)
        sentences = self.tokenize_sentences(text)
        phrases = set()
        
        for sentence in sentences:
            words = self.tokenize_words(sentence)
            
            # Find consecutive keyword sequences
            for i in range(len(words)):
                phrase = []
                for j in range(i, min(len(words), i + min_phrase_length)):
                    if words[j] in keywords:
                        phrase.append(words[j])
                    else:
                        break
                
                if len(phrase) >= min_phrase_length:
                    phrases.add(" ".join(phrase))
        
        return list(phrases)
    
    def summarize_text(self, text: str, max_sentences: int = 3) -> str:
        """Simple extractive summarization."""
        if not text:
            return ""
        
        sentences = self.tokenize_sentences(text)
        
        if len(sentences) <= max_sentences:
            return text
        
        # Score sentences based on word frequency
        word_freq = {}
        for word in self.tokenize_words(text):
            word_freq[word] = word_freq.get(word, 0) + 1
        
        sentence_scores = []
        for sentence in sentences:
            words = self.tokenize_words(sentence)
            score = sum(word_freq.get(word, 0) for word in words)
            sentence_scores.append((score, sentence))
        
        # Sort by score and return top sentences
        sentence_scores.sort(key=lambda x: x[0], reverse=True)
        top_sentences = [sentence for score, sentence in sentence_scores[:max_sentences]]
        
        return " ".join(top_sentences)
