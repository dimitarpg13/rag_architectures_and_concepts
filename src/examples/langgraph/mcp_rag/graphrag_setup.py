"""One-time setup script for GraphRAG indexing.

Creates the project directory structure, writes input text about LLM agents,
generates settings.yaml, and runs `graphrag index` to build the knowledge graph.

Usage:
    python graphrag_setup.py

Requires OPENAI_API_KEY (or GRAPHRAG_API_KEY) to be set in the environment or .env file.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

GRAPHRAG_DIR = Path(__file__).parent / "graphrag_data"

INPUT_TEXT = """\
# LLM-Powered Autonomous Agents

## Overview

LLM-powered autonomous agents use large language models as their core controller.
The agent system has three key components: planning, memory, and tool use. The
planning component allows the agent to break down complex tasks into smaller
subgoals and self-reflect to refine its approach.

## Planning

### Task Decomposition

Task decomposition breaks a complicated task into smaller, manageable steps. Chain
of Thought (CoT) prompting is a standard technique that instructs the model to
"think step by step". This transforms big tasks into multiple manageable ones and
sheds light on the model's thinking process.

Tree of Thoughts (ToT) extends Chain of Thought by exploring multiple reasoning
possibilities at each step. It generates multiple thoughts per step, creates a tree
structure of reasoning paths, and uses search algorithms like BFS or DFS to find
the optimal solution path.

### Self-Reflection

Self-reflection allows autonomous agents to iteratively improve by refining past
action decisions and correcting previous mistakes. ReAct integrates reasoning and
acting within LLMs by extending the action space to include both task-specific
actions and language generation for reasoning traces.

Reflexion is a framework that equips agents with dynamic memory and self-reflection
capabilities to improve reasoning skills. It uses a heuristic function to determine
when the trajectory is inefficient and stores reflective text in a sliding window
for future reference.

## Memory

### Short-Term Memory

Short-term memory in LLM agents maps to the in-context learning window. It is
finite and bounded by the transformer's context length. The agent uses this memory
to maintain information about the current task session including recent observations
and intermediate reasoning steps.

### Long-Term Memory

Long-term memory provides the agent with the capability to retain and recall
information over extended periods. This is usually implemented via an external
vector store that the agent can query. Maximum Inner Product Search (MIPS) is
commonly used to find the most relevant past experiences.

### Sensory Memory

Sensory memory in cognitive science corresponds to the raw input embedding
representations in LLM agents. This includes the text embeddings, image encodings,
and other modality representations that serve as the initial perception layer before
higher-level processing occurs.

## Tool Use

Tool use is a remarkable capability of LLMs that allows them to interact with
external APIs and systems.

MRKL (Modular Reasoning, Knowledge, and Language) is a neuro-symbolic architecture
that combines LLMs with external knowledge sources and symbolic reasoning modules
for complex tasks.

TALM (Tool Augmented Language Models) fine-tunes language models to learn to use
tools via text generation. Toolformer is trained to decide which APIs to call, when
to call them, what arguments to pass, and how to best incorporate the results into
future token prediction.

ChatGPT Plugins and OpenAI API function calling are practical implementations of
tool use in LLMs. They allow models to access up-to-date information, run
computations, and use third-party services. This represents a significant step
toward LLM agents operating in the real world.

HuggingGPT is a framework that uses ChatGPT as a task planner to select models
available on the HuggingFace platform according to model descriptions. It can then
summarize the response based on the execution results, demonstrating multi-model
orchestration capabilities.

API-Bank is a benchmark for evaluating tool-augmented LLMs. It consists of 73 API
tools, a complete tool-augmented LLM workflow, and 264 annotated dialogues
involving 568 API calls.

## Case Studies

### Generative Agents

Generative agents simulate human behavior using LLM-powered agents. Each agent has
a memory stream that records experiences in natural language. A retrieval model
surfaces relevant memories based on recency, importance, and relevance to the
current situation.

### AutoGPT

AutoGPT is a notable proof-of-concept of an autonomous agent powered by GPT-4. It
chains together LLM thoughts to autonomously achieve user-defined goals. It has
internet access, long and short-term memory management, GPT-4 for text generation,
and file storage capabilities.

### GPT-Engineer

GPT-Engineer aims to generate an entire codebase from a single prompt. It asks
clarifying questions, generates technical specifications, and writes all necessary
code. This demonstrates the potential for LLM agents in software engineering
automation.

## Challenges

The challenges of building LLM-powered agents include finite context length that
restricts the inclusion of historical information and detailed instructions.
Difficulties in long-term planning and task decomposition over lengthy histories
remain significant hurdles.

The reliability of natural language interfaces is a key challenge. LLMs may make
formatting errors and occasionally exhibit rebellious behavior. Agent systems must
implement robust error handling and recovery mechanisms to deal with these inherent
limitations.
"""

SETTINGS = {
    "models": {
        "default_chat_model": {
            "api_key": "${GRAPHRAG_API_KEY}",
            "type": "openai_chat",
            "model": "gpt-4o-mini",
            "model_supports_json": True,
            "max_tokens": 4000,
            "temperature": 0,
        },
        "default_embedding_model": {
            "api_key": "${GRAPHRAG_API_KEY}",
            "type": "openai_embedding",
            "model": "text-embedding-3-small",
        },
    },
    "input": {
        "type": "file",
        "file_type": "text",
        "base_dir": "input",
        "file_encoding": "utf-8",
        "file_pattern": ".*\\.txt",
    },
    "storage": {"type": "file", "base_dir": "output"},
    "cache": {"type": "file", "base_dir": "cache"},
    "reporting": {"type": "file", "base_dir": "logs"},
    "chunks": {"size": 1200, "overlap": 100},
    "entity_extraction": {"max_gleanings": 1},
    "claim_extraction": {"enabled": True},
    "community_reports": {"max_length": 2000},
    "cluster_graph": {"max_cluster_size": 10},
}


def setup_project_dirs() -> None:
    for subdir in ("input", "output", "cache", "logs"):
        (GRAPHRAG_DIR / subdir).mkdir(parents=True, exist_ok=True)
    print(f"Project directory: {GRAPHRAG_DIR}")


def write_input_text() -> None:
    input_file = GRAPHRAG_DIR / "input" / "llm_agents.txt"
    input_file.write_text(INPUT_TEXT, encoding="utf-8")
    print(f"Input text written: {input_file} ({len(INPUT_TEXT)} chars)")


def write_settings() -> None:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY must be set in the environment")

    os.environ.setdefault("GRAPHRAG_API_KEY", api_key)

    settings_file = GRAPHRAG_DIR / "settings.yaml"
    with open(settings_file, "w") as f:
        yaml.dump(SETTINGS, f, default_flow_style=False, sort_keys=False)
    print(f"Settings written: {settings_file}")


def run_indexing() -> None:
    print("\nStarting GraphRAG indexing (this may take a few minutes)...\n")
    result = subprocess.run(
        [sys.executable, "-m", "graphrag", "index", "--root", str(GRAPHRAG_DIR)],
        capture_output=True,
        text=True,
        cwd=str(GRAPHRAG_DIR),
    )

    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        print("Indexing failed:")
        print(result.stderr)
        raise RuntimeError("GraphRAG indexing failed")

    print("Indexing completed successfully.")


def verify_outputs() -> None:
    output_dir = GRAPHRAG_DIR / "output"
    parquet_files = list(output_dir.glob("*.parquet"))
    print(f"\nOutput files ({len(parquet_files)}):")
    for f in sorted(parquet_files):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name} ({size_kb:.1f} KB)")


def main() -> None:
    print("=== GraphRAG Setup ===\n")
    setup_project_dirs()
    write_input_text()
    write_settings()
    run_indexing()
    verify_outputs()
    print("\nSetup complete. The knowledge graph MCP server can now use GraphRAG.")


if __name__ == "__main__":
    main()
