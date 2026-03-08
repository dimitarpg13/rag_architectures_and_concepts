"""Sample data about LLM-powered autonomous agents for the distributed RAG demo.

Three data sources cover the same knowledge domain from different angles:
  - Vector store: text chunks for semantic search
  - Knowledge graph: entity-relationship triples
  - Structured DB: structured metadata about techniques and papers
"""

VECTOR_STORE_CHUNKS: list[dict] = [
    {
        "id": "chunk_01",
        "text": (
            "LLM-powered autonomous agents use large language models as their core "
            "controller. The agent system has three key components: planning, memory, "
            "and tool use. The planning component allows the agent to break down complex "
            "tasks into smaller subgoals and self-reflect to refine its approach."
        ),
        "metadata": {"source": "lilian_weng_blog", "section": "overview"},
    },
    {
        "id": "chunk_02",
        "text": (
            "Task decomposition breaks a complicated task into smaller, manageable steps. "
            "Chain of Thought (CoT) prompting is a standard technique that instructs the "
            "model to 'think step by step'. This transforms big tasks into multiple "
            "manageable ones and sheds light on the model's thinking process."
        ),
        "metadata": {"source": "lilian_weng_blog", "section": "planning"},
    },
    {
        "id": "chunk_03",
        "text": (
            "Tree of Thoughts (ToT) extends Chain of Thought by exploring multiple "
            "reasoning possibilities at each step. It generates multiple thoughts per "
            "step, creates a tree structure of reasoning paths, and uses search "
            "algorithms like BFS or DFS to find the optimal solution path."
        ),
        "metadata": {"source": "lilian_weng_blog", "section": "planning"},
    },
    {
        "id": "chunk_04",
        "text": (
            "Self-reflection allows autonomous agents to iteratively improve by refining "
            "past action decisions and correcting previous mistakes. ReAct integrates "
            "reasoning and acting within LLMs by extending the action space to include "
            "both task-specific actions and language generation for reasoning traces."
        ),
        "metadata": {"source": "lilian_weng_blog", "section": "planning"},
    },
    {
        "id": "chunk_05",
        "text": (
            "Reflexion is a framework that equips agents with dynamic memory and "
            "self-reflection capabilities to improve reasoning skills. It uses a "
            "heuristic function to determine when the trajectory is inefficient and "
            "stores reflective text in a sliding window for future reference."
        ),
        "metadata": {"source": "lilian_weng_blog", "section": "planning"},
    },
    {
        "id": "chunk_06",
        "text": (
            "Short-term memory in LLM agents maps to the in-context learning window. "
            "It is finite and bounded by the transformer's context length. The agent "
            "uses this memory to maintain information about the current task session "
            "including recent observations and intermediate reasoning steps."
        ),
        "metadata": {"source": "lilian_weng_blog", "section": "memory"},
    },
    {
        "id": "chunk_07",
        "text": (
            "Long-term memory provides the agent with the capability to retain and "
            "recall information over extended periods. This is usually implemented via "
            "an external vector store that the agent can query. Maximum Inner Product "
            "Search (MIPS) is commonly used to find the most relevant past experiences."
        ),
        "metadata": {"source": "lilian_weng_blog", "section": "memory"},
    },
    {
        "id": "chunk_08",
        "text": (
            "Sensory memory in cognitive science corresponds to the raw input embedding "
            "representations in LLM agents. This includes the text embeddings, image "
            "encodings, and other modality representations that serve as the initial "
            "perception layer before higher-level processing occurs."
        ),
        "metadata": {"source": "lilian_weng_blog", "section": "memory"},
    },
    {
        "id": "chunk_09",
        "text": (
            "Tool use is a remarkable capability of LLMs that allows them to interact "
            "with external APIs and systems. MRKL (Modular Reasoning, Knowledge, and "
            "Language) is a neuro-symbolic architecture that combines LLMs with external "
            "knowledge sources and symbolic reasoning modules for complex tasks."
        ),
        "metadata": {"source": "lilian_weng_blog", "section": "tools"},
    },
    {
        "id": "chunk_10",
        "text": (
            "TALM (Tool Augmented Language Models) fine-tunes language models to learn "
            "to use tools via text generation. Toolformer is trained to decide which "
            "APIs to call, when to call them, what arguments to pass, and how to best "
            "incorporate the results into future token prediction."
        ),
        "metadata": {"source": "lilian_weng_blog", "section": "tools"},
    },
    {
        "id": "chunk_11",
        "text": (
            "ChatGPT Plugins and OpenAI API function calling are practical "
            "implementations of tool use in LLMs. They allow models to access up-to-date "
            "information, run computations, and use third-party services. This represents "
            "a significant step toward LLM agents operating in the real world."
        ),
        "metadata": {"source": "lilian_weng_blog", "section": "tools"},
    },
    {
        "id": "chunk_12",
        "text": (
            "HuggingGPT is a framework that uses ChatGPT as a task planner to select "
            "models available on the HuggingFace platform according to model descriptions. "
            "It can then summarize the response based on the execution results, "
            "demonstrating multi-model orchestration capabilities."
        ),
        "metadata": {"source": "lilian_weng_blog", "section": "tools"},
    },
    {
        "id": "chunk_13",
        "text": (
            "API-Bank is a benchmark for evaluating tool-augmented LLMs. It consists of "
            "73 API tools, a complete tool-augmented LLM workflow, and 264 annotated "
            "dialogues involving 568 API calls. It evaluates three levels: call, "
            "retrieve-and-call, and plan-retrieve-and-call."
        ),
        "metadata": {"source": "lilian_weng_blog", "section": "tools"},
    },
    {
        "id": "chunk_14",
        "text": (
            "The challenges of building LLM-powered agents include finite context length "
            "that restricts the inclusion of historical information and detailed "
            "instructions. Difficulties in long-term planning and task decomposition "
            "over lengthy histories remain significant hurdles."
        ),
        "metadata": {"source": "lilian_weng_blog", "section": "challenges"},
    },
    {
        "id": "chunk_15",
        "text": (
            "The reliability of natural language interfaces is a key challenge. LLMs may "
            "make formatting errors and occasionally exhibit rebellious behavior. Agent "
            "systems must implement robust error handling and recovery mechanisms to "
            "deal with these inherent limitations."
        ),
        "metadata": {"source": "lilian_weng_blog", "section": "challenges"},
    },
    {
        "id": "chunk_16",
        "text": (
            "Generative agents simulate human behavior using LLM-powered agents. Each "
            "agent has a memory stream that records experiences in natural language. "
            "A retrieval model surfaces relevant memories based on recency, importance, "
            "and relevance to the current situation."
        ),
        "metadata": {"source": "lilian_weng_blog", "section": "case_studies"},
    },
    {
        "id": "chunk_17",
        "text": (
            "AutoGPT is a notable proof-of-concept of an autonomous agent powered by "
            "GPT-4. It chains together LLM thoughts to autonomously achieve user-defined "
            "goals. It has internet access, long and short-term memory management, "
            "GPT-4 for text generation, and file storage capabilities."
        ),
        "metadata": {"source": "lilian_weng_blog", "section": "case_studies"},
    },
    {
        "id": "chunk_18",
        "text": (
            "GPT-Engineer aims to generate an entire codebase from a single prompt. "
            "It asks clarifying questions, generates technical specifications, and writes "
            "all necessary code. This demonstrates the potential for LLM agents in "
            "software engineering automation."
        ),
        "metadata": {"source": "lilian_weng_blog", "section": "case_studies"},
    },
]


KNOWLEDGE_GRAPH_TRIPLES: list[dict] = [
    {"subject": "LLM Agent", "predicate": "has_component", "object": "Planning"},
    {"subject": "LLM Agent", "predicate": "has_component", "object": "Memory"},
    {"subject": "LLM Agent", "predicate": "has_component", "object": "Tool Use"},
    {"subject": "Planning", "predicate": "includes_technique", "object": "Task Decomposition"},
    {"subject": "Planning", "predicate": "includes_technique", "object": "Self-Reflection"},
    {"subject": "Task Decomposition", "predicate": "uses_method", "object": "Chain of Thought"},
    {"subject": "Task Decomposition", "predicate": "uses_method", "object": "Tree of Thoughts"},
    {"subject": "Chain of Thought", "predicate": "extended_by", "object": "Tree of Thoughts"},
    {"subject": "Tree of Thoughts", "predicate": "uses_algorithm", "object": "BFS"},
    {"subject": "Tree of Thoughts", "predicate": "uses_algorithm", "object": "DFS"},
    {"subject": "Self-Reflection", "predicate": "implemented_by", "object": "ReAct"},
    {"subject": "Self-Reflection", "predicate": "implemented_by", "object": "Reflexion"},
    {"subject": "ReAct", "predicate": "is_a", "object": "Agent Framework"},
    {"subject": "ReAct", "predicate": "combines", "object": "Reasoning"},
    {"subject": "ReAct", "predicate": "combines", "object": "Acting"},
    {"subject": "Reflexion", "predicate": "is_a", "object": "Agent Framework"},
    {"subject": "Reflexion", "predicate": "uses", "object": "Dynamic Memory"},
    {"subject": "Memory", "predicate": "has_type", "object": "Short-Term Memory"},
    {"subject": "Memory", "predicate": "has_type", "object": "Long-Term Memory"},
    {"subject": "Memory", "predicate": "has_type", "object": "Sensory Memory"},
    {"subject": "Short-Term Memory", "predicate": "implemented_as", "object": "In-Context Learning"},
    {"subject": "Long-Term Memory", "predicate": "implemented_via", "object": "Vector Store"},
    {"subject": "Long-Term Memory", "predicate": "uses_algorithm", "object": "MIPS"},
    {"subject": "Tool Use", "predicate": "implemented_by", "object": "MRKL"},
    {"subject": "Tool Use", "predicate": "implemented_by", "object": "TALM"},
    {"subject": "Tool Use", "predicate": "implemented_by", "object": "Toolformer"},
    {"subject": "Tool Use", "predicate": "implemented_by", "object": "ChatGPT Plugins"},
    {"subject": "MRKL", "predicate": "is_a", "object": "Neuro-Symbolic Architecture"},
    {"subject": "TALM", "predicate": "uses_technique", "object": "Fine-Tuning"},
    {"subject": "Toolformer", "predicate": "uses_technique", "object": "Self-Supervised Learning"},
    {"subject": "HuggingGPT", "predicate": "uses", "object": "ChatGPT"},
    {"subject": "HuggingGPT", "predicate": "orchestrates", "object": "HuggingFace Models"},
    {"subject": "AutoGPT", "predicate": "powered_by", "object": "GPT-4"},
    {"subject": "AutoGPT", "predicate": "is_a", "object": "Autonomous Agent"},
    {"subject": "GPT-Engineer", "predicate": "is_a", "object": "Code Generation Agent"},
    {"subject": "Generative Agents", "predicate": "simulate", "object": "Human Behavior"},
    {"subject": "Generative Agents", "predicate": "uses", "object": "Memory Stream"},
]


STRUCTURED_DB_RECORDS: list[dict] = [
    {
        "name": "Chain of Thought",
        "category": "planning",
        "year": 2022,
        "authors": "Jason Wei et al.",
        "paper": "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
        "description": "Prompting technique that instructs models to produce intermediate reasoning steps before the final answer.",
    },
    {
        "name": "Tree of Thoughts",
        "category": "planning",
        "year": 2023,
        "authors": "Shunyu Yao et al.",
        "paper": "Tree of Thoughts: Deliberate Problem Solving with Large Language Models",
        "description": "Extends CoT by exploring multiple reasoning paths using tree search algorithms.",
    },
    {
        "name": "ReAct",
        "category": "planning",
        "year": 2022,
        "authors": "Shunyu Yao et al.",
        "paper": "ReAct: Synergizing Reasoning and Acting in Language Models",
        "description": "Framework combining reasoning traces and task-specific actions in an interleaved manner.",
    },
    {
        "name": "Reflexion",
        "category": "planning",
        "year": 2023,
        "authors": "Noah Shinn et al.",
        "paper": "Reflexion: Language Agents with Verbal Reinforcement Learning",
        "description": "Agents with dynamic memory and self-reflection for iterative improvement.",
    },
    {
        "name": "MRKL",
        "category": "tool_use",
        "year": 2022,
        "authors": "Ehud Karpas et al.",
        "paper": "MRKL Systems: A modular, neuro-symbolic architecture",
        "description": "Neuro-symbolic architecture combining LLMs with expert modules and knowledge sources.",
    },
    {
        "name": "Toolformer",
        "category": "tool_use",
        "year": 2023,
        "authors": "Timo Schick et al.",
        "paper": "Toolformer: Language Models Can Teach Themselves to Use Tools",
        "description": "Self-supervised approach for LLMs to learn API usage via text generation.",
    },
    {
        "name": "HuggingGPT",
        "category": "tool_use",
        "year": 2023,
        "authors": "Yongliang Shen et al.",
        "paper": "HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face",
        "description": "Framework using ChatGPT as a planner to orchestrate HuggingFace models.",
    },
    {
        "name": "Generative Agents",
        "category": "memory",
        "year": 2023,
        "authors": "Joon Sung Park et al.",
        "paper": "Generative Agents: Interactive Simulacra of Human Behavior",
        "description": "Simulated agents with memory streams that exhibit believable human-like behavior.",
    },
    {
        "name": "TALM",
        "category": "tool_use",
        "year": 2022,
        "authors": "Aaron Parisi et al.",
        "paper": "TALM: Tool Augmented Language Models",
        "description": "Fine-tuning language models to learn tool usage via iterative text generation.",
    },
    {
        "name": "API-Bank",
        "category": "benchmark",
        "year": 2023,
        "authors": "Minghao Li et al.",
        "paper": "API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs",
        "description": "Benchmark with 73 APIs and 264 dialogues for evaluating tool-augmented LLMs.",
    },
    {
        "name": "AutoGPT",
        "category": "autonomous_agent",
        "year": 2023,
        "authors": "Toran Bruce Richards",
        "paper": "AutoGPT (open-source project)",
        "description": "Autonomous GPT-4 agent that chains LLM thoughts to achieve user-defined goals.",
    },
    {
        "name": "GPT-Engineer",
        "category": "autonomous_agent",
        "year": 2023,
        "authors": "Anton Osika",
        "paper": "GPT-Engineer (open-source project)",
        "description": "Agent that generates entire codebases from a single natural language prompt.",
    },
]
