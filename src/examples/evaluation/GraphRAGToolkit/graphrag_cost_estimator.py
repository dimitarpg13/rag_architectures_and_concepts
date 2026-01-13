"""
GraphRAG Token Counter and Cost Estimator

This module provides utilities for counting tokens and estimating costs
for GraphRAG operations including indexing and querying.

Models supported (as configured in graphrag_demo.ipynb):
- Chat Model: gpt-4o-mini
- Embedding Model: text-embedding-3-small

Usage:
    from graphrag_cost_estimator import GraphRAGCostEstimator
    
    estimator = GraphRAGCostEstimator()
    
    # Estimate indexing cost for a document
    estimate = estimator.estimate_indexing_cost("path/to/document.txt")
    print(estimate)
    
    # Estimate query cost
    query_cost = estimator.estimate_query_cost("What is the main topic?", method="local")
    print(query_cost)
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime

try:
    import tiktoken
except ImportError:
    raise ImportError("tiktoken is required. Install with: pip install tiktoken")


# =============================================================================
# Model Pricing Configuration (as of January 2026)
# Prices are per 1M tokens unless otherwise noted
# =============================================================================

@dataclass
class ModelPricing:
    """Pricing configuration for OpenAI models."""
    
    # GPT-4o-mini pricing (per 1M tokens)
    GPT4O_MINI_INPUT: float = 0.15      # $0.15 per 1M input tokens
    GPT4O_MINI_OUTPUT: float = 0.60     # $0.60 per 1M output tokens
    
    # Text-embedding-3-small pricing (per 1M tokens)
    EMBEDDING_3_SMALL: float = 0.02     # $0.02 per 1M tokens
    
    # Alternative models (for reference)
    GPT4O_INPUT: float = 2.50           # $2.50 per 1M input tokens
    GPT4O_OUTPUT: float = 10.00         # $10.00 per 1M output tokens
    EMBEDDING_3_LARGE: float = 0.13     # $0.13 per 1M tokens


@dataclass
class GraphRAGConfig:
    """Configuration matching graphrag_demo.ipynb settings."""
    
    # Models (hardcoded from notebook)
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    
    # Chunking settings
    chunk_size: int = 1200          # tokens per chunk
    chunk_overlap: int = 100        # overlap between chunks
    
    # Extraction settings
    max_gleanings: int = 1          # number of extraction passes
    
    # Community reports
    community_report_max_length: int = 2000
    
    # Max tokens for LLM calls
    max_tokens: int = 4000


@dataclass 
class TokenCount:
    """Container for token count results."""
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    embedding_tokens: int = 0
    
    def __add__(self, other: 'TokenCount') -> 'TokenCount':
        return TokenCount(
            total_tokens=self.total_tokens + other.total_tokens,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            embedding_tokens=self.embedding_tokens + other.embedding_tokens
        )


@dataclass
class CostEstimate:
    """Container for cost estimation results."""
    
    # Token counts
    token_counts: TokenCount = field(default_factory=TokenCount)
    
    # Cost breakdown (in USD)
    llm_input_cost: float = 0.0
    llm_output_cost: float = 0.0
    embedding_cost: float = 0.0
    total_cost: float = 0.0
    
    # Metadata
    operation: str = ""
    model_chat: str = ""
    model_embedding: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Details
    details: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "operation": self.operation,
            "timestamp": self.timestamp,
            "models": {
                "chat": self.model_chat,
                "embedding": self.model_embedding
            },
            "tokens": {
                "total": self.token_counts.total_tokens,
                "llm_input": self.token_counts.input_tokens,
                "llm_output": self.token_counts.output_tokens,
                "embedding": self.token_counts.embedding_tokens
            },
            "costs_usd": {
                "llm_input": round(self.llm_input_cost, 6),
                "llm_output": round(self.llm_output_cost, 6),
                "embedding": round(self.embedding_cost, 6),
                "total": round(self.total_cost, 6)
            },
            "details": self.details
        }
    
    def __str__(self) -> str:
        """Pretty print the estimate."""
        lines = [
            "=" * 60,
            f"📊 GraphRAG Cost Estimate - {self.operation}",
            "=" * 60,
            f"⏰ Timestamp: {self.timestamp}",
            "",
            "🤖 Models:",
            f"   Chat Model: {self.model_chat}",
            f"   Embedding Model: {self.model_embedding}",
            "",
            "🔢 Token Counts:",
            f"   LLM Input Tokens:    {self.token_counts.input_tokens:,}",
            f"   LLM Output Tokens:   {self.token_counts.output_tokens:,}",
            f"   Embedding Tokens:    {self.token_counts.embedding_tokens:,}",
            f"   Total Tokens:        {self.token_counts.total_tokens:,}",
            "",
            "💰 Cost Breakdown (USD):",
            f"   LLM Input Cost:      ${self.llm_input_cost:.6f}",
            f"   LLM Output Cost:     ${self.llm_output_cost:.6f}",
            f"   Embedding Cost:      ${self.embedding_cost:.6f}",
            "-" * 40,
            f"   💵 TOTAL COST:       ${self.total_cost:.6f}",
            "=" * 60
        ]
        
        if self.details:
            lines.append("")
            lines.append("📋 Details:")
            for key, value in self.details.items():
                lines.append(f"   {key}: {value}")
        
        return "\n".join(lines)


class GraphRAGCostEstimator:
    """
    Token counter and cost estimator for GraphRAG operations.
    
    This estimator uses the models configured in graphrag_demo.ipynb:
    - Chat Model: gpt-4o-mini
    - Embedding Model: text-embedding-3-small
    """
    
    def __init__(
        self,
        config: Optional[GraphRAGConfig] = None,
        pricing: Optional[ModelPricing] = None
    ):
        """
        Initialize the cost estimator.
        
        Args:
            config: GraphRAG configuration (uses defaults from notebook if None)
            pricing: Model pricing configuration (uses current pricing if None)
        """
        self.config = config or GraphRAGConfig()
        self.pricing = pricing or ModelPricing()
        
        # Initialize tokenizer for the chat model (cl100k_base for GPT-4 family)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in a text string.
        
        Args:
            text: Input text to count tokens for
            
        Returns:
            Number of tokens
        """
        return len(self.tokenizer.encode(text))
    
    def count_tokens_in_file(self, file_path: Union[str, Path]) -> int:
        """
        Count tokens in a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Number of tokens in the file
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
        """
        Count tokens in all matching files in a directory.
        
        Args:
            dir_path: Path to the directory
            pattern: Glob pattern for files to include
            
        Returns:
            Dictionary mapping file names to token counts
        """
        path = Path(dir_path)
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")
        
        results = {}
        for file_path in path.glob(pattern):
            results[file_path.name] = self.count_tokens_in_file(file_path)
        
        return results
    
    def estimate_chunks(self, total_tokens: int) -> int:
        """
        Estimate the number of chunks for a given token count.
        
        Args:
            total_tokens: Total number of tokens in the document
            
        Returns:
            Estimated number of chunks
        """
        effective_chunk_size = self.config.chunk_size - self.config.chunk_overlap
        if effective_chunk_size <= 0:
            effective_chunk_size = self.config.chunk_size
        
        return max(1, (total_tokens + effective_chunk_size - 1) // effective_chunk_size)
    
    def _calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        embedding_tokens: int
    ) -> tuple:
        """
        Calculate costs for given token counts.
        
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
        """
        Estimate the cost of indexing documents.
        
        GraphRAG indexing involves:
        1. Chunking documents
        2. Entity extraction (LLM calls per chunk)
        3. Relationship extraction (LLM calls per chunk)
        4. Community detection and summarization
        5. Embedding generation for entities and text units
        
        Args:
            input_path: Path to input file or directory
            file_pattern: Glob pattern for files (if directory)
            
        Returns:
            CostEstimate with detailed breakdown
        """
        path = Path(input_path)
        
        # Count tokens in input
        if path.is_file():
            doc_tokens = {path.name: self.count_tokens_in_file(path)}
        else:
            doc_tokens = self.count_tokens_in_directory(path, file_pattern)
        
        total_input_tokens = sum(doc_tokens.values())
        num_chunks = self.estimate_chunks(total_input_tokens)
        
        # Estimate LLM usage for indexing
        # Each chunk requires multiple LLM calls:
        # 1. Entity extraction prompt (~500 tokens) + chunk + response (~500 tokens)
        # 2. With max_gleanings passes
        extraction_passes = 1 + self.config.max_gleanings
        
        # Prompts overhead (entity extraction, relationship extraction, etc.)
        prompt_overhead = 800  # Average prompt template size
        
        # Input tokens: chunk + prompt for each extraction
        llm_input_per_chunk = (self.config.chunk_size + prompt_overhead) * extraction_passes
        total_llm_input = llm_input_per_chunk * num_chunks
        
        # Output tokens: estimated response per chunk
        avg_output_per_chunk = 600  # Average extraction output
        total_llm_output = avg_output_per_chunk * extraction_passes * num_chunks
        
        # Community reports: estimate ~10% of chunks form communities
        num_communities = max(1, num_chunks // 10)
        community_input = num_communities * 2000  # Community context
        community_output = num_communities * self.config.community_report_max_length
        
        total_llm_input += community_input
        total_llm_output += community_output
        
        # Embedding tokens: entities + text units + community reports
        # Estimate: ~30 entities per 1000 tokens, each ~50 tokens description
        estimated_entities = max(10, (total_input_tokens // 1000) * 30)
        entity_embedding_tokens = estimated_entities * 50
        
        # Text unit embeddings (each chunk)
        text_unit_embedding_tokens = total_input_tokens
        
        # Community embeddings
        community_embedding_tokens = num_communities * self.config.community_report_max_length
        
        total_embedding_tokens = (
            entity_embedding_tokens + 
            text_unit_embedding_tokens + 
            community_embedding_tokens
        )
        
        # Calculate costs
        llm_input_cost, llm_output_cost, embedding_cost, total_cost = self._calculate_cost(
            total_llm_input, total_llm_output, total_embedding_tokens
        )
        
        # Build estimate
        token_counts = TokenCount(
            total_tokens=total_llm_input + total_llm_output + total_embedding_tokens,
            input_tokens=total_llm_input,
            output_tokens=total_llm_output,
            embedding_tokens=total_embedding_tokens
        )
        
        estimate = CostEstimate(
            token_counts=token_counts,
            llm_input_cost=llm_input_cost,
            llm_output_cost=llm_output_cost,
            embedding_cost=embedding_cost,
            total_cost=total_cost,
            operation="Indexing",
            model_chat=self.config.chat_model,
            model_embedding=self.config.embedding_model,
            details={
                "input_documents": len(doc_tokens),
                "document_tokens": total_input_tokens,
                "estimated_chunks": num_chunks,
                "estimated_entities": estimated_entities,
                "estimated_communities": num_communities,
                "extraction_passes": extraction_passes
            }
        )
        
        return estimate
    
    def estimate_query_cost(
        self,
        query: str,
        method: str = "local",
        num_queries: int = 1
    ) -> CostEstimate:
        """
        Estimate the cost of running queries.
        
        Args:
            query: The query text
            method: Query method ("local" or "global")
            num_queries: Number of similar queries to estimate for
            
        Returns:
            CostEstimate with detailed breakdown
        """
        query_tokens = self.count_tokens(query)
        
        if method.lower() == "local":
            # Local search: retrieves relevant entities and text units
            # Context: ~5000 tokens of retrieved content
            # Prompt template: ~500 tokens
            context_tokens = 5000
            prompt_template = 500
            
            input_tokens = (query_tokens + context_tokens + prompt_template) * num_queries
            output_tokens = 800 * num_queries  # Average response length
            
            # Embedding for query
            embedding_tokens = query_tokens * num_queries
            
        else:  # global
            # Global search: map-reduce over community reports
            # Map phase: query each community report
            estimated_communities = 5  # Typical number of relevant communities
            map_context = 2000  # Community report size
            map_prompt = 300
            
            map_input = (query_tokens + map_context + map_prompt) * estimated_communities
            map_output = 500 * estimated_communities  # Intermediate responses
            
            # Reduce phase: combine intermediate responses
            reduce_input = query_tokens + (map_output)  # All map outputs
            reduce_output = 1000  # Final synthesized response
            
            input_tokens = (map_input + reduce_input) * num_queries
            output_tokens = (map_output + reduce_output) * num_queries
            
            # No embedding needed for global search
            embedding_tokens = 0
        
        # Calculate costs
        llm_input_cost, llm_output_cost, embedding_cost, total_cost = self._calculate_cost(
            input_tokens, output_tokens, embedding_tokens
        )
        
        token_counts = TokenCount(
            total_tokens=input_tokens + output_tokens + embedding_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            embedding_tokens=embedding_tokens
        )
        
        estimate = CostEstimate(
            token_counts=token_counts,
            llm_input_cost=llm_input_cost,
            llm_output_cost=llm_output_cost,
            embedding_cost=embedding_cost,
            total_cost=total_cost,
            operation=f"Query ({method.upper()})",
            model_chat=self.config.chat_model,
            model_embedding=self.config.embedding_model,
            details={
                "query_tokens": query_tokens,
                "method": method,
                "num_queries": num_queries
            }
        )
        
        return estimate
    
    def estimate_total_session_cost(
        self,
        input_path: Union[str, Path],
        queries: List[Dict[str, str]],
        file_pattern: str = "*.txt"
    ) -> CostEstimate:
        """
        Estimate total cost for a complete GraphRAG session.
        
        Args:
            input_path: Path to input documents
            queries: List of dicts with 'query' and 'method' keys
            file_pattern: Glob pattern for input files
            
        Returns:
            Combined CostEstimate for entire session
        """
        # Indexing cost
        indexing_estimate = self.estimate_indexing_cost(input_path, file_pattern)
        
        # Query costs
        total_query_tokens = TokenCount()
        query_details = []
        
        for i, q in enumerate(queries):
            query_text = q.get("query", "")
            method = q.get("method", "local")
            
            query_estimate = self.estimate_query_cost(query_text, method)
            total_query_tokens = total_query_tokens + query_estimate.token_counts
            query_details.append({
                f"query_{i+1}": {
                    "method": method,
                    "tokens": query_estimate.token_counts.total_tokens,
                    "cost": query_estimate.total_cost
                }
            })
        
        # Combine totals
        combined_tokens = indexing_estimate.token_counts + total_query_tokens
        
        llm_input_cost, llm_output_cost, embedding_cost, total_cost = self._calculate_cost(
            combined_tokens.input_tokens,
            combined_tokens.output_tokens,
            combined_tokens.embedding_tokens
        )
        
        estimate = CostEstimate(
            token_counts=combined_tokens,
            llm_input_cost=llm_input_cost,
            llm_output_cost=llm_output_cost,
            embedding_cost=embedding_cost,
            total_cost=total_cost,
            operation="Full Session (Indexing + Queries)",
            model_chat=self.config.chat_model,
            model_embedding=self.config.embedding_model,
            details={
                "indexing_cost": indexing_estimate.total_cost,
                "num_queries": len(queries),
                "queries_cost": total_cost - indexing_estimate.total_cost,
                **indexing_estimate.details
            }
        )
        
        return estimate


# =============================================================================
# Convenience Functions
# =============================================================================

def count_tokens(text: str) -> int:
    """Quick token count for a string."""
    estimator = GraphRAGCostEstimator()
    return estimator.count_tokens(text)


def estimate_indexing(input_path: Union[str, Path]) -> CostEstimate:
    """Quick indexing cost estimate."""
    estimator = GraphRAGCostEstimator()
    return estimator.estimate_indexing_cost(input_path)


def estimate_query(query: str, method: str = "local") -> CostEstimate:
    """Quick query cost estimate."""
    estimator = GraphRAGCostEstimator()
    return estimator.estimate_query_cost(query, method)


def print_pricing_info():
    """Print current model pricing information."""
    pricing = ModelPricing()
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
# CLI Interface
# =============================================================================

def main():
    """Command-line interface for the cost estimator."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="GraphRAG Token Counter and Cost Estimator"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Count tokens command
    count_parser = subparsers.add_parser("count", help="Count tokens in a file or text")
    count_parser.add_argument("input", help="File path or text string")
    count_parser.add_argument("--text", "-t", action="store_true", 
                              help="Treat input as text instead of file path")
    
    # Estimate indexing command
    index_parser = subparsers.add_parser("index", help="Estimate indexing cost")
    index_parser.add_argument("input_path", help="Path to input file or directory")
    index_parser.add_argument("--pattern", "-p", default="*.txt",
                              help="File pattern (default: *.txt)")
    
    # Estimate query command
    query_parser = subparsers.add_parser("query", help="Estimate query cost")
    query_parser.add_argument("query_text", help="Query text")
    query_parser.add_argument("--method", "-m", choices=["local", "global"],
                              default="local", help="Query method (default: local)")
    query_parser.add_argument("--num", "-n", type=int, default=1,
                              help="Number of queries (default: 1)")
    
    # Pricing info command
    subparsers.add_parser("pricing", help="Show model pricing information")
    
    args = parser.parse_args()
    
    if args.command == "count":
        estimator = GraphRAGCostEstimator()
        if args.text:
            tokens = estimator.count_tokens(args.input)
            print(f"Token count: {tokens:,}")
        else:
            tokens = estimator.count_tokens_in_file(args.input)
            print(f"Token count for '{args.input}': {tokens:,}")
    
    elif args.command == "index":
        estimate = estimate_indexing(args.input_path)
        print(estimate)
    
    elif args.command == "query":
        estimator = GraphRAGCostEstimator()
        estimate = estimator.estimate_query_cost(
            args.query_text, args.method, args.num
        )
        print(estimate)
    
    elif args.command == "pricing":
        print_pricing_info()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

