"""
LLM service for language model operations.
"""

from typing import List, Dict, Optional, Any
import openai
import tiktoken
import logging

from rexi.config.settings import get_settings

logger = logging.getLogger(__name__)

class LLMService:
    """Service for Large Language Model operations."""
    
    def __init__(self):
        """Initialize LLM service."""
        self.settings = get_settings()
        self.client: Optional[openai.OpenAI] = None
        self.tokenizer: Optional[tiktoken.Encoding] = None
        self._initialize()
    
    def _initialize(self):
        """Initialize OpenAI client and tokenizer."""
        try:
            if self.settings.openai_api_key:
                self.client = openai.OpenAI(api_key=self.settings.openai_api_key)
                logger.info("OpenAI client initialized")
            else:
                logger.warning("OpenAI API key not provided")
            
            # Initialize tokenizer for the model
            try:
                self.tokenizer = tiktoken.encoding_for_model(self.settings.llm_model)
            except KeyError:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
                logger.warning(f"Using default tokenizer for model {self.settings.llm_model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM service: {e}")
            raise
    
    def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> str:
        """Generate chat completion."""
        if not self.client:
            raise ValueError("LLM client not initialized")
        
        try:
            response = self.client.chat.completions.create(
                model=self.settings.llm_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
            
            if stream:
                return response  # Return stream object for streaming
            else:
                return response.choices[0].message.content
                
        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            raise
    
    def structured_extraction(
        self, 
        text: str, 
        schema: Dict[str, Any],
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """Extract structured information from text."""
        if not self.client:
            raise ValueError("LLM client not initialized")
        
        try:
            # Create prompt for structured extraction
            system_prompt = f"""
            You are a structured information extraction system.
            Extract information from the following text according to this schema:
            {schema}
            
            Return the extracted information as a valid JSON object.
            Only include information that is explicitly mentioned in the text.
            If a field is not found, use null or omit it.
            """
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ]
            
            response = self.client.chat.completions.create(
                model=self.settings.llm_model,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"}
            )
            
            import json
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Structured extraction failed: {e}")
            return {}
    
    def generate_entities_and_relations(self, text: str) -> Dict[str, Any]:
        """Extract entities and relations from text."""
        schema = {
            "entities": [
                {
                    "name": "string",
                    "type": "string (person|concept|skill|topic|project|tool|paper|book|event|organization|idea|task|goal|emotion|habit)",
                    "description": "string",
                    "confidence": "number (0-1)"
                }
            ],
            "relations": [
                {
                    "source": "string",
                    "target": "string", 
                    "type": "string (enables|causes|improves|depends_on|contradicts|learned_from|used_in|part_of|related_to|precedes|follows|inspired_by|supports|applied_to)",
                    "confidence": "number (0-1)"
                }
            ]
        }
        
        return self.structured_extraction(text, schema)
    
    def summarize_text(self, text: str, max_length: int = 200) -> str:
        """Summarize text to specified length."""
        if not self.client:
            raise ValueError("LLM client not initialized")
        
        try:
            messages = [
                {
                    "role": "system", 
                    "content": f"You are a helpful assistant. Summarize the following text in no more than {max_length} words. Focus on the most important information."
                },
                {"role": "user", "content": text}
            ]
            
            response = self.client.chat.completions.create(
                model=self.settings.llm_model,
                messages=messages,
                temperature=0.3,
                max_tokens=max_length * 2  # Approximate token limit
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return ""
    
    def answer_question(
        self, 
        question: str, 
        context: str = "",
        temperature: float = 0.3
    ) -> str:
        """Answer a question based on provided context."""
        if not self.client:
            raise ValueError("LLM client not initialized")
        
        try:
            system_prompt = """
            You are a helpful AI assistant. Answer the user's question based on the provided context.
            If the context doesn't contain enough information to answer the question, say so clearly.
            Provide accurate, concise answers with citations to the context when possible.
            """
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
            ]
            
            response = self.client.chat.completions.create(
                model=self.settings.llm_model,
                messages=messages,
                temperature=temperature
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Question answering failed: {e}")
            return ""
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if not self.tokenizer:
            self._initialize()
        
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            logger.error(f"Token counting failed: {e}")
            return 0
    
    def truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to maximum token count."""
        if not self.tokenizer:
            self._initialize()
        
        try:
            tokens = self.tokenizer.encode(text)
            if len(tokens) <= max_tokens:
                return text
            
            truncated_tokens = tokens[:max_tokens]
            return self.tokenizer.decode(truncated_tokens)
        except Exception as e:
            logger.error(f"Text truncation failed: {e}")
            return text[:max_tokens * 4]  # Rough fallback
    
    def is_available(self) -> bool:
        """Check if LLM service is available."""
        return self.client is not None
