"""
Custom LLM service for REXI - local inference with fine-tuned models.
"""

import logging
from typing import List, Dict, Optional, Any, Union
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    pipeline
)
from accelerate import Accelerator
import time
import json

from rexi.config.settings import get_settings

logger = logging.getLogger(__name__)

class CustomLLMService:
    """Service for custom LLM operations with local inference."""
    
    def __init__(self):
        """Initialize custom LLM service."""
        self.settings = get_settings()
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.pipeline: Optional[Any] = None
        self.accelerator: Optional[Accelerator] = None
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Model configuration
        self.model_name = getattr(self.settings, 'custom_llm_model', 'meta-llama/Llama-3.1-8B-Instruct')
        self.quantization_enabled = getattr(self.settings, 'llm_quantization', True)
        self.max_tokens = getattr(self.settings, 'llm_max_tokens', 2048)
        self.temperature = getattr(self.settings, 'llm_temperature', 0.7)
        
        # REXI-specific prompt templates
        self.prompt_templates = {
            "entity_extraction": self._get_entity_extraction_template(),
            "relationship_extraction": self._get_relationship_extraction_template(),
            "reasoning": self._get_reasoning_template(),
            "explanation": self._get_explanation_template(),
            "hypothesis_generation": self._get_hypothesis_template(),
            "knowledge_gap_analysis": self._get_knowledge_gap_template()
        }
        
        self._initialize()
    
    def _initialize(self):
        """Initialize the custom LLM model and tokenizer."""
        try:
            logger.info(f"Initializing custom LLM: {self.model_name}")
            
            # Configure quantization if enabled
            quantization_config = None
            if self.quantization_enabled and self.device == "cuda":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                logger.info("4-bit quantization enabled")
            
            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Initialize model
            model_kwargs = {
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "device_map": "auto" if self.device == "cuda" else None,
                "trust_remote_code": True
            }
            
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Initialize text generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                return_full_text=False
            )
            
            logger.info(f"Custom LLM initialized successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize custom LLM: {e}")
            raise
    
    def _get_entity_extraction_template(self) -> str:
        """Get template for entity extraction."""
        return """You are an expert entity extraction specialist for REXI knowledge graphs.

Extract entities from the following text and classify them into these types:
- PERSON: People, names, roles
- CONCEPT: Ideas, theories, principles
- SKILL: Abilities, competencies, expertise
- TOPIC: Subjects, domains, fields
- PROJECT: Initiatives, endeavors, work
- TOOL: Software, instruments, resources
- PAPER: Research papers, articles
- BOOK: Books, publications
- EVENT: Meetings, conferences, occurrences
- ORGANIZATION: Companies, institutions, groups

Text: {text}

Provide entities in JSON format:
{{
    "entities": [
        {{"name": "entity_name", "type": "ENTITY_TYPE", "confidence": 0.9, "description": "brief description"}}
    ]
}}

Entities:"""
    
    def _get_relationship_extraction_template(self) -> str:
        """Get template for relationship extraction."""
        return """You are an expert relationship extraction specialist for REXI knowledge graphs.

Extract relationships between entities from the following text using these types:
- ENABLES: A enables B
- CAUSES: A causes B
- IMPROVES: A improves B
- DEPENDS_ON: A depends on B
- CONTRADICTS: A contradicts B
- LEARNED_FROM: A learned from B
- USED_IN: A used in B
- PART_OF: A is part of B
- RELATED_TO: A related to B
- PRECEDES: A precedes B
- FOLLOWS: A follows B
- INSPIRED_BY: A inspired by B
- SUPPORTS: A supports B
- APPLIED_TO: A applied to B

Text: {text}
Entities: {entities}

Provide relationships in JSON format:
{{
    "relationships": [
        {{"source": "entity1", "target": "entity2", "type": "RELATIONSHIP_TYPE", "confidence": 0.8, "description": "brief description"}}
    ]
}}

Relationships:"""
    
    def _get_reasoning_template(self) -> str:
        """Get template for reasoning tasks."""
        return """You are an expert reasoning specialist for REXI knowledge graphs.

Answer the following query using multi-hop reasoning over the provided knowledge graph information.

Query: {query}
Knowledge Graph Context: {context}

Instructions:
1. Analyze the relationships in the knowledge graph
2. Perform multi-hop reasoning if needed
3. Consider temporal aspects if relevant
4. Provide evidence-based answers
5. Include confidence scores

Provide answer in JSON format:
{{
    "answer": "detailed answer",
    "reasoning_path": ["step1", "step2", "step3"],
    "confidence": 0.85,
    "evidence": ["source1", "source2"],
    "entities_mentioned": ["entity1", "entity2"]
}}

Answer:"""
    
    def _get_explanation_template(self) -> str:
        """Get template for explanation generation."""
        return """You are an expert explanation specialist for REXI knowledge graphs.

Explain the reasoning path for the following answer using the provided knowledge graph evidence.

Answer: {answer}
Query: {query}
Evidence: {evidence}
Reasoning Path: {reasoning_path}

Instructions:
1. Explain each step in the reasoning chain
2. Reference specific evidence from the knowledge graph
3. Highlight key relationships and entities
4. Explain confidence levels
5. Provide alternative perspectives if applicable

Provide explanation in JSON format:
{{
    "explanation": "detailed step-by-step explanation",
    "key_insights": ["insight1", "insight2"],
    "confidence_breakdown": {{"step1": 0.9, "step2": 0.8}},
    "alternative_paths": ["alternative1", "alternative2"],
    "evidence_strength": "strong/moderate/weak"
}}

Explanation:"""
    
    def _get_hypothesis_template(self) -> str:
        """Get template for hypothesis generation."""
        return """You are an expert hypothesis generation specialist for REXI knowledge graphs.

Generate testable hypotheses based on the provided knowledge graph context.

Context: {context}
Existing Knowledge: {existing_knowledge}

Instructions:
1. Identify gaps in current knowledge
2. Generate plausible hypotheses
3. Suggest testing methods
4. Estimate confidence levels
5. Consider potential implications

Provide hypotheses in JSON format:
{{
    "hypotheses": [
        {{
            "statement": "testable hypothesis statement",
            "test_method": "how to test this hypothesis",
            "expected_outcome": "expected result if true",
            "confidence": 0.7,
            "novelty_score": 0.8,
            "test_complexity": "low/medium/high"
        }}
    ]
}}

Hypotheses:"""
    
    def _get_knowledge_gap_template(self) -> str:
        """Get template for knowledge gap analysis."""
        return """You are an expert knowledge gap analysis specialist for REXI knowledge graphs.

Analyze the provided knowledge graph to identify gaps and suggest improvements.

Knowledge Graph Data: {graph_data}
Current Statistics: {statistics}

Instructions:
1. Identify missing entity types
2. Find isolated entities
3. Detect missing relationship types
4. Suggest exploration areas
5. Prioritize by importance and feasibility

Provide analysis in JSON format:
{{
    "knowledge_gaps": [
        {{
            "type": "entity_scarsity/isolated_entities/missing_relationships",
            "description": "description of the gap",
            "severity": "low/medium/high",
            "suggestion": "how to address this gap",
            "priority": 1-10
        }}
    ],
    "exploration_suggestions": [
        {{
            "target": "what to explore",
            "method": "how to explore",
            "expected_value": "potential benefit"
        }}
    ]
}}

Analysis:"""
    
    def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        task_type: str = "general"
    ) -> str:
        """Generate chat completion using custom LLM."""
        try:
            if not self.pipeline:
                raise RuntimeError("LLM pipeline not initialized")
            
            # Format messages into prompt
            if task_type in self.prompt_templates:
                # Use specialized template
                prompt = self._format_prompt_with_template(messages, task_type)
            else:
                # Use general chat format
                prompt = self._format_chat_prompt(messages)
            
            # Generate response
            params = {
                "max_new_tokens": max_tokens or self.max_tokens,
                "temperature": temperature or self.temperature,
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id,
                "return_full_text": False
            }
            
            start_time = time.time()
            result = self.pipeline(prompt, **params)
            generation_time = time.time() - start_time
            
            response_text = result[0]["generated_text"] if result else ""
            
            logger.info(f"Generated response in {generation_time:.2f}s for task: {task_type}")
            return response_text.strip()
            
        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            raise
    
    def _format_prompt_with_template(self, messages: List[Dict[str, str]], task_type: str) -> str:
        """Format prompt using specialized template."""
        template = self.prompt_templates.get(task_type, "{text}")
        
        # Extract relevant information from messages
        context = {}
        for message in messages:
            if message.get("role") == "user":
                content = message.get("content", "")
                # Parse structured content if JSON
                try:
                    if content.startswith("{") and content.endswith("}"):
                        context.update(json.loads(content))
                    else:
                        context["text"] = content
                except json.JSONDecodeError:
                    context["text"] = content
        
        # Format template with context
        try:
            return template.format(**context)
        except KeyError as e:
            logger.warning(f"Missing context key {e} for template {task_type}")
            return template.format(text=context.get("text", ""))
    
    def _format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Format messages into chat prompt."""
        prompt_parts = []
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant: ")
        return "\n".join(prompt_parts)
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using custom LLM."""
        try:
            messages = [
                {"role": "user", "content": json.dumps({"text": text})}
            ]
            
            response = self.chat_completion(messages, task_type="entity_extraction")
            
            # Parse JSON response
            try:
                result = json.loads(response)
                return result.get("entities", [])
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse entity extraction response: {response}")
                return []
                
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return []
    
    def extract_relationships(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relationships using custom LLM."""
        try:
            messages = [
                {"role": "user", "content": json.dumps({
                    "text": text,
                    "entities": entities
                })}
            ]
            
            response = self.chat_completion(messages, task_type="relationship_extraction")
            
            # Parse JSON response
            try:
                result = json.loads(response)
                return result.get("relationships", [])
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse relationship extraction response: {response}")
                return []
                
        except Exception as e:
            logger.error(f"Relationship extraction failed: {e}")
            return []
    
    def generate_reasoning(self, query: str, context: str) -> Dict[str, Any]:
        """Generate reasoning using custom LLM."""
        try:
            messages = [
                {"role": "user", "content": json.dumps({
                    "query": query,
                    "context": context
                })}
            ]
            
            response = self.chat_completion(messages, task_type="reasoning")
            
            # Parse JSON response
            try:
                result = json.loads(response)
                return result
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse reasoning response: {response}")
                return {"answer": response, "confidence": 0.5}
                
        except Exception as e:
            logger.error(f"Reasoning generation failed: {e}")
            return {"answer": "Reasoning failed", "confidence": 0.0}
    
    def generate_explanation(self, answer: str, query: str, evidence: List[str], reasoning_path: List[str]) -> Dict[str, Any]:
        """Generate explanation using custom LLM."""
        try:
            messages = [
                {"role": "user", "content": json.dumps({
                    "answer": answer,
                    "query": query,
                    "evidence": evidence,
                    "reasoning_path": reasoning_path
                })}
            ]
            
            response = self.chat_completion(messages, task_type="explanation")
            
            # Parse JSON response
            try:
                result = json.loads(response)
                return result
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse explanation response: {response}")
                return {"explanation": response}
                
        except Exception as e:
            logger.error(f"Explanation generation failed: {e}")
            return {"explanation": "Explanation generation failed"}
    
    def generate_hypotheses(self, context: str, existing_knowledge: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate hypotheses using custom LLM."""
        try:
            messages = [
                {"role": "user", "content": json.dumps({
                    "context": context,
                    "existing_knowledge": existing_knowledge
                })}
            ]
            
            response = self.chat_completion(messages, task_type="hypothesis_generation")
            
            # Parse JSON response
            try:
                result = json.loads(response)
                return result.get("hypotheses", [])
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse hypothesis response: {response}")
                return []
                
        except Exception as e:
            logger.error(f"Hypothesis generation failed: {e}")
            return []
    
    def analyze_knowledge_gaps(self, graph_data: Dict[str, Any], statistics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze knowledge gaps using custom LLM."""
        try:
            messages = [
                {"role": "user", "content": json.dumps({
                    "graph_data": graph_data,
                    "statistics": statistics
                })}
            ]
            
            response = self.chat_completion(messages, task_type="knowledge_gap_analysis")
            
            # Parse JSON response
            try:
                result = json.loads(response)
                return result
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse knowledge gap response: {response}")
                return {"knowledge_gaps": [], "exploration_suggestions": []}
                
        except Exception as e:
            logger.error(f"Knowledge gap analysis failed: {e}")
            return {"knowledge_gaps": [], "exploration_suggestions": []}
    
    def is_available(self) -> bool:
        """Check if the custom LLM service is available."""
        return self.pipeline is not None and self.model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "quantization_enabled": self.quantization_enabled,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "is_loaded": self.is_available()
        }
