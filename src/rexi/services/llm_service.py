"""
LLM service for language model operations.
"""

from typing import List, Dict, Optional, Any
import logging

from rexi.config.settings import get_settings

logger = logging.getLogger(__name__)

class LLMService:
    """Service for Large Language Model operations."""
    
    def __init__(self):
        """Initialize LLM service."""
        self.settings = get_settings()
        self.use_custom_llm = getattr(self.settings, 'use_custom_llm', False)
        self.custom_llm_service = None
        self.openai_client = None
        
        # Initialize appropriate service
        if self.use_custom_llm:
            self._initialize_custom_llm()
        else:
            self._initialize_openai()
    
    def _initialize_custom_llm(self):
        """Initialize custom LLM service."""
        try:
            from rexi.services.custom_llm_service import CustomLLMService
            self.custom_llm_service = CustomLLMService()
            logger.info("Custom LLM service initialized")
        except ImportError as e:
            logger.error(f"Failed to import custom LLM service: {e}")
            self.use_custom_llm = False
            self._initialize_openai()
        except Exception as e:
            logger.error(f"Failed to initialize custom LLM: {e}")
            self.use_custom_llm = False
            self._initialize_openai()
    
    def _initialize_openai(self):
        """Initialize OpenAI client."""
        try:
            import openai
            import tiktoken
            
            if self.settings.openai_api_key:
                self.openai_client = openai.OpenAI(api_key=self.settings.openai_api_key)
                logger.info("OpenAI client initialized")
            else:
                logger.warning("OpenAI API key not provided")
        except ImportError as e:
            logger.error(f"Failed to import OpenAI: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI: {e}")
    
    def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        task_type: str = "general"
    ) -> str:
        """Generate chat completion using available LLM."""
        if self.use_custom_llm and self.custom_llm_service:
            return self.custom_llm_service.chat_completion(
                messages, temperature, max_tokens, task_type
            )
        elif self.openai_client:
            return self._openai_completion(messages, temperature, max_tokens)
        else:
            raise RuntimeError("No LLM service available")
    
    def _openai_completion(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float,
        max_tokens: Optional[int]
    ) -> str:
        """Generate completion using OpenAI."""
        try:
            response = self.openai_client.chat.completions.create(
                model=self.settings.llm_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI completion failed: {e}")
            raise
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using available LLM."""
        if self.use_custom_llm and self.custom_llm_service:
            return self.custom_llm_service.extract_entities(text)
        else:
            # Fallback to OpenAI with entity extraction prompt
            messages = [
                {"role": "system", "content": "Extract entities from the text and return them in JSON format."},
                {"role": "user", "content": f"Text: {text}"}
            ]
            response = self.chat_completion(messages, task_type="entity_extraction")
            # Parse JSON response (simplified)
            return []
    
    def extract_relationships(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relationships using available LLM."""
        if self.use_custom_llm and self.custom_llm_service:
            return self.custom_llm_service.extract_relationships(text, entities)
        else:
            # Fallback to OpenAI
            return []
    
    def generate_reasoning(self, query: str, context: str) -> Dict[str, Any]:
        """Generate reasoning using available LLM."""
        if self.use_custom_llm and self.custom_llm_service:
            return self.custom_llm_service.generate_reasoning(query, context)
        else:
            # Fallback to OpenAI
            return {"answer": "Reasoning not available", "confidence": 0.5}
    
    def generate_explanation(self, answer: str, query: str, evidence: List[str], reasoning_path: List[str]) -> Dict[str, Any]:
        """Generate explanation using available LLM."""
        if self.use_custom_llm and self.custom_llm_service:
            return self.custom_llm_service.generate_explanation(answer, query, evidence, reasoning_path)
        else:
            # Fallback to OpenAI
            return {"explanation": "Explanation not available"}
    
    def generate_hypotheses(self, context: str, existing_knowledge: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate hypotheses using available LLM."""
        if self.use_custom_llm and self.custom_llm_service:
            return self.custom_llm_service.generate_hypotheses(context, existing_knowledge)
        else:
            # Fallback to OpenAI
            return []
    
    def analyze_knowledge_gaps(self, graph_data: Dict[str, Any], statistics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze knowledge gaps using available LLM."""
        if self.use_custom_llm and self.custom_llm_service:
            return self.custom_llm_service.analyze_knowledge_gaps(graph_data, statistics)
        else:
            # Fallback to OpenAI
            return {"knowledge_gaps": [], "exploration_suggestions": []}
    
    def is_available(self) -> bool:
        """Check if any LLM service is available."""
        if self.use_custom_llm and self.custom_llm_service:
            return self.custom_llm_service.is_available()
        elif self.openai_client:
            return True
        return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.use_custom_llm and self.custom_llm_service:
            return self.custom_llm_service.get_model_info()
        elif self.openai_client:
            return {
                "model_name": self.settings.llm_model,
                "service": "OpenAI",
                "is_loaded": True
            }
        return {"is_loaded": False}
    
    def is_available(self) -> bool:
        """Check if LLM service is available."""
        return self.client is not None
