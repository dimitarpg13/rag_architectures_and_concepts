"""
GraphRAG Improved Token Counter and Cost Estimator

This module provides accurate token counting and cost estimation for Microsoft GraphRAG operations.

Key features:
- Accurate prompt template sizes based on actual GraphRAG prompts
- Empirical calibration factors for better accuracy
- Post-indexing estimation using actual entity/community counts
- Validation functions to compare estimates vs actual costs

Models supported:
- Chat Model: gpt-4o-mini ($0.15/1M input, $0.60/1M output)
- Embedding Model: text-embedding-3-small ($0.02/1M tokens)

Usage:
    from graphrag_improved_cost_estimator import ImprovedGraphRAGCostEstimator

    estimator = ImprovedGraphRAGCostEstimator()
    
    # Estimate indexing cost
    indexing_estimate = estimator.estimate_indexing_cost("./input")
    print(indexing_estimate)
    
    # Estimate query cost
    query_estimate = estimator.estimate_query_cost("What is X?", method="local")
    print(query_estimate)
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime

import tiktoken

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


# =============================================================================
# Model Pricing Configuration (as of January 2026)
# Prices are per 1M tokens
# =============================================================================

@dataclass
class ModelPricing:
    """Pricing configuration for OpenAI models (per 1M tokens).
    
    Attributes:
        GPT4O_MINI_INPUT: Cost per 1M input tokens for gpt-4o-mini
        GPT4O_MINI_OUTPUT: Cost per 1M output tokens for gpt-4o-mini
        EMBEDDING_3_SMALL: Cost per 1M tokens for text-embedding-3-small
        GPT4O_INPUT: Cost per 1M input tokens for gpt-4o
        GPT4O_OUTPUT: Cost per 1M output tokens for gpt-4o
        EMBEDDING_3_LARGE: Cost per 1M tokens for text-embedding-3-large
    """
    GPT4O_MINI_INPUT: float = 0.15
    GPT4O_MINI_OUTPUT: float = 0.60
    EMBEDDING_3_SMALL: float = 0.02
    GPT4O_INPUT: float = 2.50
    GPT4O_OUTPUT: float = 10.00
    EMBEDDING_3_LARGE: float = 0.13


# =============================================================================
# GraphRAG Prompt Template Sizes (Empirically Measured)
# =============================================================================

@dataclass
class PromptTemplateSizes:
    """Estimated token sizes for GraphRAG prompt templates.
    
    These values are based on analysis of actual GraphRAG prompt templates
    from the graphrag package source code.
    
    Attributes:
        ENTITY_EXTRACTION_SYSTEM: System prompt for entity extraction (~1800 tokens)
        ENTITY_EXTRACTION_EXAMPLES: Few-shot examples for entity extraction (~500 tokens)
        ENTITY_SUMMARIZATION: Prompt for entity summarization (~400 tokens)
        RELATIONSHIP_EXTRACTION: Prompt for relationship extraction (~300 tokens)
        CLAIM_EXTRACTION_SYSTEM: System prompt for claim extraction (~1200 tokens)
        CLAIM_EXTRACTION_EXAMPLES: Few-shot examples for claim extraction (~400 tokens)
        COMMUNITY_REPORT_SYSTEM: System prompt for community report generation (~1500 tokens)
        COMMUNITY_REPORT_EXAMPLES: Few-shot examples for community reports (~600 tokens)
        LOCAL_SEARCH_SYSTEM: System prompt for local search queries (~800 tokens)
        GLOBAL_SEARCH_MAP: Prompt for global search map phase (~600 tokens)
        GLOBAL_SEARCH_REDUCE: Prompt for global search reduce phase (~500 tokens)
    """
    # Entity extraction prompt (~2000 tokens for system + examples)
    ENTITY_EXTRACTION_SYSTEM: int = 1800
    ENTITY_EXTRACTION_EXAMPLES: int = 500
    
    # Entity summarization prompt
    ENTITY_SUMMARIZATION: int = 400
    
    # Relationship extraction (often combined with entity extraction)
    RELATIONSHIP_EXTRACTION: int = 300
    
    # Claim extraction prompt
    CLAIM_EXTRACTION_SYSTEM: int = 1200
    CLAIM_EXTRACTION_EXAMPLES: int = 400
    
    # Community report generation prompt
    COMMUNITY_REPORT_SYSTEM: int = 1500
    COMMUNITY_REPORT_EXAMPLES: int = 600
    
    # Query prompts
    LOCAL_SEARCH_SYSTEM: int = 800
    GLOBAL_SEARCH_MAP: int = 600
    GLOBAL_SEARCH_REDUCE: int = 500


@dataclass
class GraphRAGConfig:
    """Configuration matching GraphRAG settings.
    
    Attributes:
        chat_model: Name of the chat model (default: gpt-4o-mini)
        embedding_model: Name of the embedding model (default: text-embedding-3-small)
        chunk_size: Size of text chunks in tokens (default: 1200)
        chunk_overlap: Overlap between chunks in tokens (default: 100)
        max_gleanings: Number of additional extraction passes (default: 1)
        community_report_max_length: Maximum length of community reports (default: 2000)
        max_tokens: Maximum tokens for LLM responses (default: 4000)
        claim_extraction_enabled: Whether claim extraction is enabled (default: True)
    """
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 1200  # tokens
    chunk_overlap: int = 100  # tokens
    max_gleanings: int = 1
    community_report_max_length: int = 2000
    max_tokens: int = 4000
    claim_extraction_enabled: bool = True


@dataclass
class CalibrationFactors:
    """Calibration factors to adjust estimates based on observed accuracy.
    
    These factors are multipliers applied to base estimates.
    Values > 1.0 increase estimates, < 1.0 decrease them.
    Adjust based on your observed actual vs estimated ratios.
    
    Attributes:
        entity_count_multiplier: Multiplier for entity count estimation (default: 1.2)
        output_token_multiplier: Multiplier for output token estimation (default: 1.3)
        embedding_multiplier: Multiplier for embedding token estimation (default: 1.1)
        local_context_multiplier: Multiplier for local search context (default: 1.15)
        global_context_multiplier: Multiplier for global search context (default: 1.2)
    """
    # Indexing calibration
    entity_count_multiplier: float = 1.2  # Entities are often underestimated
    output_token_multiplier: float = 1.3  # LLM outputs tend to be longer than expected
    embedding_multiplier: float = 1.1
    
    # Query calibration
    local_context_multiplier: float = 1.15
    global_context_multiplier: float = 1.2


@dataclass
class TokenCount:
    """Container for token count results.
    
    Attributes:
        total_tokens: Total tokens (input + output + embedding)
        input_tokens: LLM input tokens
        output_tokens: LLM output tokens
        embedding_tokens: Embedding tokens
    """
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    embedding_tokens: int = 0

    def __add__(self, other: 'TokenCount') -> 'TokenCount':
        """Add two TokenCount objects together."""
        return TokenCount(
            total_tokens=self.total_tokens + other.total_tokens,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            embedding_tokens=self.embedding_tokens + other.embedding_tokens
        )


@dataclass
class CostEstimate:
    """Container for cost estimation results.
    
    Attributes:
        token_counts: TokenCount object with token breakdowns
        llm_input_cost: Cost for LLM input tokens in USD
        llm_output_cost: Cost for LLM output tokens in USD
        embedding_cost: Cost for embedding tokens in USD
        total_cost: Total cost in USD
        operation: Description of the operation estimated
        model_chat: Chat model used
        model_embedding: Embedding model used
        timestamp: ISO timestamp of when estimate was created
        details: Dictionary with additional details
        confidence: Confidence level (low, medium, high)
    """
    token_counts: TokenCount = field(default_factory=TokenCount)
    llm_input_cost: float = 0.0
    llm_output_cost: float = 0.0
    embedding_cost: float = 0.0
    total_cost: float = 0.0
    operation: str = ""
    model_chat: str = ""
    model_embedding: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    details: Dict = field(default_factory=dict)
    confidence: str = "medium"  # low, medium, high

    def __str__(self) -> str:
        """Return a formatted string representation of the cost estimate."""
        return f"""
{'='*60}
📊 GraphRAG Cost Estimate - {self.operation}
{'='*60}
⏰ Timestamp: {self.timestamp}
📈 Confidence: {self.confidence.upper()}

🤖 Models:
   Chat Model: {self.model_chat}
   Embedding Model: {self.model_embedding}

🔢 Token Counts:
   LLM Input Tokens:    {self.token_counts.input_tokens:,}
   LLM Output Tokens:   {self.token_counts.output_tokens:,}
   Embedding Tokens:    {self.token_counts.embedding_tokens:,}
   Total Tokens:        {self.token_counts.total_tokens:,}

💰 Cost Breakdown (USD):
   LLM Input Cost:      ${self.llm_input_cost:.6f}
   LLM Output Cost:     ${self.llm_output_cost:.6f}
   Embedding Cost:      ${self.embedding_cost:.6f}
{'-'*40}
   💵 TOTAL COST:       ${self.total_cost:.6f}
{'='*60}
"""

    def to_dict(self) -> Dict:
        """Convert cost estimate to dictionary."""
        return {
            "operation": self.operation,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "models": {
                "chat": self.model_chat,
                "embedding": self.model_embedding
            },
            "tokens": {
                "input": self.token_counts.input_tokens,
                "output": self.token_counts.output_tokens,
                "embedding": self.token_counts.embedding_tokens,
                "total": self.token_counts.total_tokens
            },
            "costs": {
                "llm_input": self.llm_input_cost,
                "llm_output": self.llm_output_cost,
                "embedding": self.embedding_cost,
                "total": self.total_cost
            },
            "details": self.details
        }


class ImprovedGraphRAGCostEstimator:
    """Improved token counter and cost estimator for GraphRAG operations.
    
    Key improvements over basic estimator:
    1. Uses actual GraphRAG prompt template sizes
    2. Applies empirical calibration factors
    3. Can update estimates using actual indexed data
    4. Provides validation functions to compare estimates vs actuals
    
    Example:
        >>> estimator = ImprovedGraphRAGCostEstimator()
        >>> estimate = estimator.estimate_indexing_cost("./input")
        >>> print(estimate)
        >>> print(f"Total cost: ${estimate.total_cost:.4f}")
    """

    def __init__(
        self,
        config: Optional[GraphRAGConfig] = None,
        pricing: Optional[ModelPricing] = None,
        prompts: Optional[PromptTemplateSizes] = None,
        calibration: Optional[CalibrationFactors] = None
    ):
        """Initialize the cost estimator.
        
        Args:
            config: GraphRAG configuration settings
            pricing: Model pricing configuration
            prompts: Prompt template size estimates
            calibration: Calibration factors for adjusting estimates
        """
        self.config = config or GraphRAGConfig()
        self.pricing = pricing or ModelPricing()
        self.prompts = prompts or PromptTemplateSizes()
        self.calibration = calibration or CalibrationFactors()
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Store actual indexed data for improved query estimation
        self._indexed_stats: Optional[Dict] = None

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string using cl100k_base encoding.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            Number of tokens in the text
        """
        return len(self.tokenizer.encode(text))

    def count_tokens_in_file(self, file_path: Union[str, Path]) -> int:
        """Count tokens in a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Number of tokens in the file
            
        Raises:
            FileNotFoundError: If the file does not exist
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        text = path.read_text(encoding="utf-8")
        return self.count_tokens(text)

    def count_tokens_in_directory(
        self,
        dir_path: Union[str, Path],
        pattern: str = "*.txt"
    ) -> Dict[str, int]:
        """Count tokens in all matching files in a directory.
        
        Args:
            dir_path: Path to the directory
            pattern: Glob pattern for matching files (default: "*.txt")
            
        Returns:
            Dictionary mapping filename to token count
            
        Raises:
            FileNotFoundError: If the directory does not exist
        """
        path = Path(dir_path)
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")
        results = {}
        for file_path in path.glob(pattern):
            results[file_path.name] = self.count_tokens_in_file(file_path)
        return results

    def _estimate_chunks(self, total_tokens: int) -> int:
        """Estimate number of text chunks.
        
        Args:
            total_tokens: Total tokens in the input documents
            
        Returns:
            Estimated number of chunks
        """
        effective_chunk_size = self.config.chunk_size - self.config.chunk_overlap
        if effective_chunk_size <= 0:
            effective_chunk_size = self.config.chunk_size
        return max(1, (total_tokens + effective_chunk_size - 1) // effective_chunk_size)

    def _estimate_entities(self, total_tokens: int, num_chunks: int) -> int:
        """Improved entity estimation using content density heuristics.
        
        Entities are estimated based on:
        - Token density (entities per 100 tokens)
        - Chunk count (minimum entities per chunk)
        - Calibration factor
        
        Args:
            total_tokens: Total tokens in the input documents
            num_chunks: Estimated number of chunks
            
        Returns:
            Estimated number of entities
        """
        # Base estimate: ~3-5 entities per 100 tokens for typical business documents
        entities_from_density = (total_tokens / 100) * 4
        
        # Minimum: at least 5 entities per chunk
        entities_from_chunks = num_chunks * 5
        
        # Take the higher estimate and apply calibration
        base_estimate = max(entities_from_density, entities_from_chunks)
        calibrated = int(base_estimate * self.calibration.entity_count_multiplier)
        
        return max(10, calibrated)  # Minimum 10 entities

    def _estimate_communities(self, num_entities: int) -> int:
        """Estimate communities using Leiden algorithm heuristics.
        
        Leiden algorithm typically creates communities with 5-15 entities each.
        
        Args:
            num_entities: Estimated number of entities
            
        Returns:
            Estimated number of communities
        """
        # Average community size of ~8 entities
        avg_community_size = 8
        return max(1, num_entities // avg_community_size)

    def _calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        embedding_tokens: int
    ) -> Tuple[float, float, float, float]:
        """Calculate costs for given token counts.
        
        Args:
            input_tokens: Number of LLM input tokens
            output_tokens: Number of LLM output tokens
            embedding_tokens: Number of embedding tokens
            
        Returns:
            Tuple of (llm_input_cost, llm_output_cost, embedding_cost, total_cost)
        """
        llm_input_cost = (input_tokens / 1_000_000) * self.pricing.GPT4O_MINI_INPUT
        llm_output_cost = (output_tokens / 1_000_000) * self.pricing.GPT4O_MINI_OUTPUT
        embedding_cost = (embedding_tokens / 1_000_000) * self.pricing.EMBEDDING_3_SMALL
        total_cost = llm_input_cost + llm_output_cost + embedding_cost
        return llm_input_cost, llm_output_cost, embedding_cost, total_cost

    def estimate_indexing_cost(
        self,
        input_path: Union[str, Path],
        file_pattern: str = "*.txt"
    ) -> CostEstimate:
        """Estimate the cost of indexing documents with improved accuracy.
        
        Accounts for all GraphRAG indexing workflows:
        1. Entity extraction (with gleanings)
        2. Entity summarization
        3. Relationship extraction
        4. Claim extraction (if enabled)
        5. Community report generation
        6. Embeddings generation
        
        Args:
            input_path: Path to input file or directory
            file_pattern: Glob pattern for matching files (default: "*.txt")
            
        Returns:
            CostEstimate object with token counts and cost breakdown
        """
        path = Path(input_path)

        # Count document tokens
        if path.is_file():
            doc_tokens = {path.name: self.count_tokens_in_file(path)}
        else:
            doc_tokens = self.count_tokens_in_directory(path, file_pattern)

        total_input_tokens = sum(doc_tokens.values())
        num_chunks = self._estimate_chunks(total_input_tokens)
        num_entities = self._estimate_entities(total_input_tokens, num_chunks)
        num_communities = self._estimate_communities(num_entities)
        
        # Calculate extraction passes
        extraction_passes = 1 + self.config.max_gleanings

        # ===== ENTITY EXTRACTION =====
        # Each chunk goes through entity extraction with full prompt
        entity_extraction_input_per_chunk = (
            self.prompts.ENTITY_EXTRACTION_SYSTEM +
            self.prompts.ENTITY_EXTRACTION_EXAMPLES +
            self.config.chunk_size  # The actual text chunk
        )
        entity_extraction_input = entity_extraction_input_per_chunk * extraction_passes * num_chunks
        
        # Output: entities and relationships in JSON format (~800 tokens per chunk)
        entity_extraction_output = 800 * extraction_passes * num_chunks

        # ===== ENTITY SUMMARIZATION =====
        # Each unique entity gets summarized
        entity_summarization_input = (
            self.prompts.ENTITY_SUMMARIZATION + 200  # Entity descriptions
        ) * num_entities
        entity_summarization_output = 150 * num_entities  # Summary per entity

        # ===== CLAIM EXTRACTION (if enabled) =====
        claim_input = 0
        claim_output = 0
        if self.config.claim_extraction_enabled:
            claim_input = (
                self.prompts.CLAIM_EXTRACTION_SYSTEM +
                self.prompts.CLAIM_EXTRACTION_EXAMPLES +
                self.config.chunk_size
            ) * num_chunks
            claim_output = 400 * num_chunks  # Claims per chunk

        # ===== COMMUNITY REPORT GENERATION =====
        # Each community gets a report
        community_input_per = (
            self.prompts.COMMUNITY_REPORT_SYSTEM +
            self.prompts.COMMUNITY_REPORT_EXAMPLES +
            1000  # Entity/relationship context for the community
        )
        community_input = community_input_per * num_communities
        community_output = self.config.community_report_max_length * num_communities

        # ===== TOTAL LLM TOKENS =====
        total_llm_input = int((
            entity_extraction_input +
            entity_summarization_input +
            claim_input +
            community_input
        ) * self.calibration.output_token_multiplier)
        
        total_llm_output = int((
            entity_extraction_output +
            entity_summarization_output +
            claim_output +
            community_output
        ) * self.calibration.output_token_multiplier)

        # ===== EMBEDDINGS =====
        # Embeddings for: entities, text units, community reports
        entity_embedding_tokens = num_entities * 100  # Avg entity description length
        text_unit_embedding_tokens = total_input_tokens
        community_embedding_tokens = num_communities * self.config.community_report_max_length
        
        total_embedding_tokens = int((
            entity_embedding_tokens +
            text_unit_embedding_tokens +
            community_embedding_tokens
        ) * self.calibration.embedding_multiplier)

        # Calculate costs
        llm_input_cost, llm_output_cost, embedding_cost, total_cost = self._calculate_cost(
            total_llm_input, total_llm_output, total_embedding_tokens
        )

        return CostEstimate(
            token_counts=TokenCount(
                total_tokens=total_llm_input + total_llm_output + total_embedding_tokens,
                input_tokens=total_llm_input,
                output_tokens=total_llm_output,
                embedding_tokens=total_embedding_tokens
            ),
            llm_input_cost=llm_input_cost,
            llm_output_cost=llm_output_cost,
            embedding_cost=embedding_cost,
            total_cost=total_cost,
            operation="Indexing",
            model_chat=self.config.chat_model,
            model_embedding=self.config.embedding_model,
            confidence="medium",
            details={
                "input_documents": len(doc_tokens),
                "document_tokens": total_input_tokens,
                "estimated_chunks": num_chunks,
                "estimated_entities": num_entities,
                "estimated_communities": num_communities,
                "extraction_passes": extraction_passes,
                "claim_extraction": self.config.claim_extraction_enabled,
                "breakdown": {
                    "entity_extraction_input": entity_extraction_input,
                    "entity_summarization_input": entity_summarization_input,
                    "claim_extraction_input": claim_input,
                    "community_report_input": community_input,
                }
            }
        )

    def load_indexed_stats(self, output_path: Union[str, Path]) -> Dict:
        """Load actual statistics from indexed output for improved query estimation.
        
        This method reads the parquet files generated by GraphRAG indexing
        to get actual entity and community counts.
        
        Args:
            output_path: Path to the GraphRAG output directory
            
        Returns:
            Dictionary with entity, relationship, community, and text_unit counts
        """
        if not PANDAS_AVAILABLE:
            print("⚠️ Warning: pandas not available, cannot load indexed stats")
            return {"loaded": False}
        
        output_dir = Path(output_path)
        stats = {
            "entities": 0,
            "relationships": 0,
            "communities": 0,
            "text_units": 0,
            "loaded": False
        }
        
        try:
            # Load entities
            entities_file = output_dir / "entities.parquet"
            if entities_file.exists():
                entities_df = pd.read_parquet(entities_file)
                stats["entities"] = len(entities_df)
            
            # Load relationships
            rels_file = output_dir / "relationships.parquet"
            if rels_file.exists():
                rels_df = pd.read_parquet(rels_file)
                stats["relationships"] = len(rels_df)
            
            # Load communities
            communities_file = output_dir / "communities.parquet"
            if communities_file.exists():
                communities_df = pd.read_parquet(communities_file)
                stats["communities"] = len(communities_df)
            
            # Load text units
            text_units_file = output_dir / "text_units.parquet"
            if text_units_file.exists():
                text_units_df = pd.read_parquet(text_units_file)
                stats["text_units"] = len(text_units_df)
            
            stats["loaded"] = True
            self._indexed_stats = stats
            
        except Exception as e:
            print(f"⚠️ Warning: Could not load indexed stats: {e}")
        
        return stats

    def estimate_query_cost(
        self,
        query: str,
        method: str = "local",
        num_queries: int = 1,
        use_indexed_stats: bool = True
    ) -> CostEstimate:
        """Estimate the cost of running queries with improved accuracy.
        
        If indexed stats are available, uses actual entity/community counts.
        Otherwise falls back to estimates.
        
        Args:
            query: The query text
            method: Query method ("local" or "global")
            num_queries: Number of times this query will be run (default: 1)
            use_indexed_stats: Whether to use actual indexed stats if available
            
        Returns:
            CostEstimate object with token counts and cost breakdown
        """
        query_tokens = self.count_tokens(query)
        
        # Get entity/community counts
        if use_indexed_stats and self._indexed_stats and self._indexed_stats.get("loaded"):
            num_entities = self._indexed_stats["entities"]
            num_communities = self._indexed_stats["communities"]
            num_relationships = self._indexed_stats["relationships"]
            confidence = "high"
        else:
            # Fallback to estimates
            num_entities = 50  # Default estimate
            num_communities = 5
            num_relationships = 100
            confidence = "medium"
        
        if method.lower() == "local":
            # Local search retrieves relevant entities and builds context
            # Context size depends on number of retrieved entities (typically top-k)
            top_k_entities = min(20, num_entities)
            entity_context = top_k_entities * 150  # Avg entity description
            relationship_context = min(50, num_relationships) * 50  # Related relationships
            
            context_tokens = int((
                entity_context + relationship_context
            ) * self.calibration.local_context_multiplier)
            
            prompt_template = self.prompts.LOCAL_SEARCH_SYSTEM
            input_tokens = (query_tokens + context_tokens + prompt_template) * num_queries
            output_tokens = 800 * num_queries  # Response tokens
            embedding_tokens = query_tokens * num_queries  # Query embedding
            
        else:  # global search
            # Global search uses map-reduce over community reports
            communities_to_process = num_communities
            
            # Map phase: query each community report
            map_input_per_community = (
                self.prompts.GLOBAL_SEARCH_MAP +
                query_tokens +
                self.config.community_report_max_length
            )
            map_input = int(
                map_input_per_community * communities_to_process * 
                self.calibration.global_context_multiplier
            )
            map_output = 500 * communities_to_process  # Intermediate answers
            
            # Reduce phase: combine all intermediate answers
            reduce_input = (
                self.prompts.GLOBAL_SEARCH_REDUCE +
                query_tokens +
                map_output  # All intermediate answers
            )
            reduce_output = 1200  # Final comprehensive answer
            
            input_tokens = (map_input + reduce_input) * num_queries
            output_tokens = (map_output + reduce_output) * num_queries
            embedding_tokens = 0  # Global search doesn't use embeddings
        
        llm_input_cost, llm_output_cost, embedding_cost, total_cost = self._calculate_cost(
            input_tokens, output_tokens, embedding_tokens
        )
        
        return CostEstimate(
            token_counts=TokenCount(
                total_tokens=input_tokens + output_tokens + embedding_tokens,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                embedding_tokens=embedding_tokens
            ),
            llm_input_cost=llm_input_cost,
            llm_output_cost=llm_output_cost,
            embedding_cost=embedding_cost,
            total_cost=total_cost,
            operation=f"Query ({method.upper()})",
            model_chat=self.config.chat_model,
            model_embedding=self.config.embedding_model,
            confidence=confidence,
            details={
                "query_tokens": query_tokens,
                "method": method,
                "num_queries": num_queries,
                "entities_available": num_entities,
                "communities_available": num_communities,
                "using_indexed_stats": use_indexed_stats and self._indexed_stats is not None
            }
        )

    def estimate_total_session_cost(
        self,
        input_path: Union[str, Path],
        queries: List[Dict[str, str]],
        file_pattern: str = "*.txt"
    ) -> CostEstimate:
        """Estimate total cost for a complete GraphRAG session.
        
        Args:
            input_path: Path to input file or directory
            queries: List of query dictionaries with "query" and "method" keys
            file_pattern: Glob pattern for matching files (default: "*.txt")
            
        Returns:
            CostEstimate object with combined token counts and costs
        """
        indexing_estimate = self.estimate_indexing_cost(input_path, file_pattern)
        
        total_query_tokens = TokenCount()
        for q in queries:
            query_estimate = self.estimate_query_cost(
                q.get("query", ""),
                q.get("method", "local"),
                use_indexed_stats=False  # Use estimates for pre-indexing
            )
            total_query_tokens = total_query_tokens + query_estimate.token_counts
        
        combined_tokens = indexing_estimate.token_counts + total_query_tokens
        llm_input_cost, llm_output_cost, embedding_cost, total_cost = self._calculate_cost(
            combined_tokens.input_tokens,
            combined_tokens.output_tokens,
            combined_tokens.embedding_tokens
        )
        
        return CostEstimate(
            token_counts=combined_tokens,
            llm_input_cost=llm_input_cost,
            llm_output_cost=llm_output_cost,
            embedding_cost=embedding_cost,
            total_cost=total_cost,
            operation="Full Session (Indexing + Queries)",
            model_chat=self.config.chat_model,
            model_embedding=self.config.embedding_model,
            confidence="medium",
            details={
                "indexing_cost": indexing_estimate.total_cost,
                "num_queries": len(queries),
                "queries_cost": total_cost - indexing_estimate.total_cost,
                **indexing_estimate.details
            }
        )

    def compare_estimate_vs_actual(
        self,
        estimate: CostEstimate,
        output_path: Union[str, Path]
    ) -> Dict:
        """Compare estimated costs with actual indexed output.
        
        Loads the actual indexed data and compares estimated entity/community
        counts with actual counts to assess estimation accuracy.
        
        Args:
            estimate: A CostEstimate from estimate_indexing_cost
            output_path: Path to the GraphRAG output directory
            
        Returns:
            Dictionary with comparison data including accuracy percentages
        """
        actual_stats = self.load_indexed_stats(output_path)
        
        if not actual_stats.get("loaded"):
            return {"error": "Could not load actual statistics"}
        
        estimated = estimate.details
        
        comparison = {
            "entities": {
                "estimated": estimated.get("estimated_entities", 0),
                "actual": actual_stats["entities"],
                "accuracy": 0.0
            },
            "communities": {
                "estimated": estimated.get("estimated_communities", 0),
                "actual": actual_stats["communities"],
                "accuracy": 0.0
            },
            "relationships": {
                "estimated": estimated.get("estimated_entities", 0) * 2,  # Rough estimate
                "actual": actual_stats["relationships"],
                "accuracy": 0.0
            },
            "text_units": {
                "estimated": estimated.get("estimated_chunks", 0),
                "actual": actual_stats["text_units"],
                "accuracy": 0.0
            }
        }
        
        # Calculate accuracy percentages
        for key in comparison:
            est = comparison[key]["estimated"]
            act = comparison[key]["actual"]
            if act > 0:
                # Accuracy as percentage (100% = perfect match)
                comparison[key]["accuracy"] = min(est, act) / max(est, act) * 100
        
        return comparison


# =============================================================================
# Utility Functions
# =============================================================================

def print_comparison_report(comparison: Dict) -> None:
    """Print a formatted comparison report.
    
    Args:
        comparison: Dictionary from compare_estimate_vs_actual
    """
    print("=" * 60)
    print("📊 ESTIMATE vs ACTUAL COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<15} {'Estimated':>12} {'Actual':>12} {'Accuracy':>12}")
    print("-" * 60)
    
    for metric, values in comparison.items():
        if isinstance(values, dict) and "estimated" in values:
            print(
                f"{metric:<15} {values['estimated']:>12,} "
                f"{values['actual']:>12,} {values['accuracy']:>11.1f}%"
            )
    
    print("=" * 60)
    
    # Overall assessment
    accuracies = [
        v["accuracy"] for v in comparison.values() 
        if isinstance(v, dict) and "accuracy" in v
    ]
    avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
    
    if avg_accuracy >= 80:
        assessment = "✅ EXCELLENT - Estimates are highly accurate"
    elif avg_accuracy >= 60:
        assessment = "✅ GOOD - Estimates are reasonably accurate"
    elif avg_accuracy >= 40:
        assessment = "⚠️ FAIR - Consider adjusting calibration factors"
    else:
        assessment = "❌ POOR - Calibration factors need significant adjustment"
    
    print(f"\n📈 Overall Accuracy: {avg_accuracy:.1f}%")
    print(f"   {assessment}")


def suggest_calibration_adjustments(comparison: Dict) -> Dict[str, float]:
    """Suggest calibration factor adjustments based on comparison results.
    
    Args:
        comparison: Dictionary from compare_estimate_vs_actual
        
    Returns:
        Dictionary with suggested calibration factor values
    """
    suggestions = {}
    
    if "entities" in comparison:
        est = comparison["entities"]["estimated"]
        act = comparison["entities"]["actual"]
        if est > 0 and act > 0:
            suggestions["entity_count_multiplier"] = act / est
    
    if "communities" in comparison:
        est = comparison["communities"]["estimated"]
        act = comparison["communities"]["actual"]
        if est > 0 and act > 0:
            suggestions["community_adjustment"] = act / est
    
    return suggestions


def print_pricing_info(pricing: Optional[ModelPricing] = None) -> None:
    """Print current model pricing information.
    
    Args:
        pricing: ModelPricing object (uses defaults if not provided)
    """
    pricing = pricing or ModelPricing()
    print("=" * 50)
    print("💰 OpenAI Model Pricing (per 1M tokens)")
    print("=" * 50)
    print("\n🤖 Chat Models:")
    print(f"   gpt-4o-mini (input):  ${pricing.GPT4O_MINI_INPUT:.2f}")
    print(f"   gpt-4o-mini (output): ${pricing.GPT4O_MINI_OUTPUT:.2f}")
    print(f"   gpt-4o (input):       ${pricing.GPT4O_INPUT:.2f}")
    print(f"   gpt-4o (output):      ${pricing.GPT4O_OUTPUT:.2f}")
    print("\n📊 Embedding Models:")
    print(f"   text-embedding-3-small: ${pricing.EMBEDDING_3_SMALL:.2f}")
    print(f"   text-embedding-3-large: ${pricing.EMBEDDING_3_LARGE:.2f}")
    print("=" * 50)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Data classes
    "ModelPricing",
    "PromptTemplateSizes",
    "GraphRAGConfig",
    "CalibrationFactors",
    "TokenCount",
    "CostEstimate",
    # Main class
    "ImprovedGraphRAGCostEstimator",
    # Utility functions
    "print_comparison_report",
    "suggest_calibration_adjustments",
    "print_pricing_info",
]


# =============================================================================
# CLI Entry Point (when run as script)
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Estimate GraphRAG indexing and query costs"
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default="./input",
        help="Path to input directory or file (default: ./input)"
    )
    parser.add_argument(
        "--pattern",
        default="*.txt",
        help="File pattern to match (default: *.txt)"
    )
    parser.add_argument(
        "--query",
        help="Estimate cost for a specific query"
    )
    parser.add_argument(
        "--method",
        choices=["local", "global"],
        default="local",
        help="Query method (default: local)"
    )
    parser.add_argument(
        "--output-dir",
        help="Path to GraphRAG output directory for validation"
    )
    parser.add_argument(
        "--pricing",
        action="store_true",
        help="Show model pricing information"
    )
    
    args = parser.parse_args()
    
    if args.pricing:
        print_pricing_info()
        exit(0)
    
    estimator = ImprovedGraphRAGCostEstimator()
    
    # Estimate indexing cost
    print("\n📊 INDEXING COST ESTIMATE")
    print("=" * 60)
    try:
        indexing_estimate = estimator.estimate_indexing_cost(
            args.input_path, args.pattern
        )
        print(indexing_estimate)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        exit(1)
    
    # If query provided, estimate query cost
    if args.query:
        print("\n📊 QUERY COST ESTIMATE")
        print("=" * 60)
        query_estimate = estimator.estimate_query_cost(
            args.query, args.method, use_indexed_stats=False
        )
        print(query_estimate)
    
    # If output directory provided, compare estimates with actuals
    if args.output_dir:
        print("\n📊 ESTIMATE VS ACTUAL COMPARISON")
        print("=" * 60)
        comparison = estimator.compare_estimate_vs_actual(
            indexing_estimate, args.output_dir
        )
        if "error" not in comparison:
            print_comparison_report(comparison)
            suggestions = suggest_calibration_adjustments(comparison)
            if suggestions:
                print("\n💡 Suggested Calibration Adjustments:")
                for key, value in suggestions.items():
                    print(f"   {key}: {value:.2f}")
        else:
            print(f"❌ Error: {comparison['error']}")
