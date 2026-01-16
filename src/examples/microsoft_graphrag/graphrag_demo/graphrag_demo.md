# GraphRAG Demo - UML Diagrams

This document contains UML diagrams describing the architecture and interactions of the GraphRAG toolkit demo.

---

## 1. Static Class Diagram - Core Components

```mermaid
classDiagram
    class GraphRAGSystem {
        +PROJECT_DIR: Path
        +INPUT_DIR: Path
        +OUTPUT_DIR: Path
        +run_index()
        +run_query(query, method)
    }

    class Configuration {
        +llm: LLMConfig
        +embeddings: EmbeddingsConfig
        +input: InputConfig
        +storage: StorageConfig
        +cache: CacheConfig
        +chunks: ChunksConfig
        +entity_extraction: EntityExtractionConfig
        +community_reports: CommunityReportsConfig
        +save_to_yaml(path)
    }

    class LLMConfig {
        +api_key: str
        +type: str
        +model: str
        +max_tokens: int
        +temperature: float
        +model_supports_json: bool
    }

    class EmbeddingsConfig {
        +api_key: str
        +type: str
        +model: str
    }

    class InputConfig {
        +type: str
        +file_type: str
        +base_dir: str
        +file_encoding: str
        +file_pattern: str
    }

    class Document {
        +file_path: Path
        +content: str
        +encoding: str
        +read_text()
        +write_text(content)
    }

    class KnowledgeGraph {
        +entities: List~Entity~
        +relationships: List~Relationship~
        +communities: List~Community~
        +add_node(entity)
        +add_edge(relationship)
    }

    class Entity {
        +name: str
        +type: str
        +description: str
    }

    class Relationship {
        +source: str
        +target: str
        +description: str
        +weight: float
    }

    class Community {
        +id: str
        +members: List~Entity~
        +summary: str
        +level: int
    }

    class QueryEngine {
        +method: str
        +local_search(query)
        +global_search(query)
    }

    class OutputArtifacts {
        +entities_parquet: Path
        +relationships_parquet: Path
        +communities_parquet: Path
        +embeddings_parquet: Path
        +load_entities()
        +load_relationships()
    }

    GraphRAGSystem --> Configuration : uses
    GraphRAGSystem --> Document : processes
    GraphRAGSystem --> KnowledgeGraph : builds
    GraphRAGSystem --> QueryEngine : uses
    GraphRAGSystem --> OutputArtifacts : generates

    Configuration --> LLMConfig : contains
    Configuration --> EmbeddingsConfig : contains
    Configuration --> InputConfig : contains

    KnowledgeGraph --> Entity : contains
    KnowledgeGraph --> Relationship : contains
    KnowledgeGraph --> Community : contains

    QueryEngine --> KnowledgeGraph : queries
    OutputArtifacts --> Entity : stores
    OutputArtifacts --> Relationship : stores
```

---

## 2. Static Class Diagram - Visualization Components

```mermaid
classDiagram
    class GraphVisualizer {
        +OUTPUT_DIR: Path
        +visualize_knowledge_graph()
        +save_visualization(path)
    }

    class NetworkXGraph {
        +nodes: List
        +edges: List
        +add_node(name, attributes)
        +add_edge(source, target)
        +number_of_nodes()
        +number_of_edges()
    }

    class MatplotlibRenderer {
        +figure: Figure
        +draw(graph, pos, options)
        +add_legend(elements)
        +save(path, dpi)
        +show()
    }

    class DataExplorer {
        +explore_graphrag_outputs()
        +list_parquet_files()
        +display_entities(limit)
        +display_relationships(limit)
    }

    class PandasDataFrame {
        +columns: List
        +read_parquet(path)
        +head(n)
        +to_string()
    }

    GraphVisualizer --> NetworkXGraph : creates
    GraphVisualizer --> MatplotlibRenderer : uses
    GraphVisualizer --> PandasDataFrame : loads data from

    DataExplorer --> PandasDataFrame : uses
    DataExplorer --> OutputArtifacts : reads

    class OutputArtifacts {
        +artifacts_dir: Path
        +entities_file: Path
        +relationships_file: Path
    }
```

---

## 3. Sequence Diagram - Setup and Configuration Flow

```mermaid
sequenceDiagram
    participant User
    participant Notebook
    participant Environment
    participant FileSystem
    participant Configuration

    User->>Notebook: Run installation cell
    Notebook->>Environment: pip install graphrag
    Environment-->>Notebook: Installation complete

    User->>Notebook: Run environment setup
    Notebook->>Environment: load_dotenv()
    Environment-->>Notebook: Load .env variables
    Notebook->>Environment: Check GRAPHRAG_API_KEY
    alt API key exists
        Environment-->>Notebook: ✅ API key configured
    else API key missing
        Environment-->>Notebook: ⚠️ Warning: No API key
    end

    User->>Notebook: Run project setup
    Notebook->>FileSystem: Create INPUT_DIR
    Notebook->>FileSystem: Create OUTPUT_DIR
    FileSystem-->>Notebook: Directories created

    User->>Notebook: Run sample data preparation
    Notebook->>FileSystem: Write techcorp_report.txt
    FileSystem-->>Notebook: File saved

    User->>Notebook: Run configuration cell
    Notebook->>Configuration: Create settings dict
    Configuration->>FileSystem: Save settings.yaml
    FileSystem-->>Notebook: Configuration saved
```

---

## 4. Sequence Diagram - Indexing Process

```mermaid
sequenceDiagram
    participant User
    participant Notebook
    participant GraphRAG
    participant LLM as LLM (GPT-4o-mini)
    participant EmbeddingModel as Embedding Model
    participant FileSystem

    User->>Notebook: run_graphrag_index()
    Notebook->>GraphRAG: graphrag index --root .

    Note over GraphRAG: Phase 1: Document Chunking
    GraphRAG->>FileSystem: Read input documents
    FileSystem-->>GraphRAG: Document content
    GraphRAG->>GraphRAG: Split into chunks (1200 chars, 100 overlap)

    Note over GraphRAG: Phase 2: Entity Extraction
    loop For each chunk
        GraphRAG->>LLM: Extract entities and relationships
        LLM-->>GraphRAG: Entities, relationships, claims
    end

    Note over GraphRAG: Phase 3: Graph Construction
    GraphRAG->>GraphRAG: Build knowledge graph
    GraphRAG->>GraphRAG: Merge duplicate entities
    GraphRAG->>GraphRAG: Calculate relationship weights

    Note over GraphRAG: Phase 4: Community Detection
    GraphRAG->>GraphRAG: Run Leiden algorithm
    GraphRAG->>GraphRAG: Create hierarchical communities
    
    loop For each community
        GraphRAG->>LLM: Generate community summary
        LLM-->>GraphRAG: Community report
    end

    Note over GraphRAG: Phase 5: Embedding Generation
    GraphRAG->>EmbeddingModel: Generate entity embeddings
    EmbeddingModel-->>GraphRAG: Vector embeddings
    GraphRAG->>EmbeddingModel: Generate text unit embeddings
    EmbeddingModel-->>GraphRAG: Vector embeddings

    Note over GraphRAG: Phase 6: Save Artifacts
    GraphRAG->>FileSystem: Save entities.parquet
    GraphRAG->>FileSystem: Save relationships.parquet
    GraphRAG->>FileSystem: Save communities.parquet
    GraphRAG->>FileSystem: Save embeddings.parquet

    GraphRAG-->>Notebook: ✅ Indexing complete
    Notebook-->>User: Display results
```

---

## 5. Sequence Diagram - Query Flow (Local Search)

```mermaid
sequenceDiagram
    participant User
    participant Notebook
    participant QueryEngine
    participant VectorStore
    participant KnowledgeGraph
    participant LLM as LLM (GPT-4o-mini)

    User->>Notebook: run_graphrag_query(query, "local")
    Notebook->>QueryEngine: graphrag query --method local

    Note over QueryEngine: Local Search Strategy
    QueryEngine->>VectorStore: Embed query
    VectorStore-->>QueryEngine: Query embedding

    QueryEngine->>VectorStore: Find similar entities
    VectorStore-->>QueryEngine: Top-k relevant entities

    QueryEngine->>KnowledgeGraph: Get entity context
    KnowledgeGraph-->>QueryEngine: Entity descriptions
    
    QueryEngine->>KnowledgeGraph: Get connected relationships
    KnowledgeGraph-->>QueryEngine: Related edges

    QueryEngine->>KnowledgeGraph: Get source text units
    KnowledgeGraph-->>QueryEngine: Original text chunks

    Note over QueryEngine: Context Assembly
    QueryEngine->>QueryEngine: Rank and prioritize context
    QueryEngine->>QueryEngine: Build context window

    QueryEngine->>LLM: Generate answer with context
    LLM-->>QueryEngine: Response

    QueryEngine-->>Notebook: Formatted response
    Notebook-->>User: Display answer
```

---

## 6. Sequence Diagram - Query Flow (Global Search)

```mermaid
sequenceDiagram
    participant User
    participant Notebook
    participant QueryEngine
    participant CommunityReports
    participant LLM as LLM (GPT-4o-mini)

    User->>Notebook: run_graphrag_query(query, "global")
    Notebook->>QueryEngine: graphrag query --method global

    Note over QueryEngine: Global Search Strategy
    QueryEngine->>CommunityReports: Load community summaries
    CommunityReports-->>QueryEngine: All community reports

    QueryEngine->>QueryEngine: Select relevant communities
    
    Note over QueryEngine: Map-Reduce Pattern
    loop For each community batch
        QueryEngine->>LLM: Generate partial answer
        LLM-->>QueryEngine: Intermediate response
    end

    QueryEngine->>LLM: Reduce/combine partial answers
    LLM-->>QueryEngine: Final synthesized response

    QueryEngine-->>Notebook: Formatted response
    Notebook-->>User: Display answer
```

---

## 7. Sequence Diagram - Exploration and Visualization

```mermaid
sequenceDiagram
    participant User
    participant Notebook
    participant DataExplorer
    participant FileSystem
    participant Pandas
    participant NetworkX
    participant Matplotlib

    User->>Notebook: explore_graphrag_outputs()
    Notebook->>DataExplorer: Initialize explorer
    
    DataExplorer->>FileSystem: List output/artifacts/*.parquet
    FileSystem-->>DataExplorer: File list
    DataExplorer-->>User: Display available files

    DataExplorer->>Pandas: read_parquet(entities.parquet)
    Pandas-->>DataExplorer: Entities DataFrame
    DataExplorer-->>User: Display entities table

    DataExplorer->>Pandas: read_parquet(relationships.parquet)
    Pandas-->>DataExplorer: Relationships DataFrame
    DataExplorer-->>User: Display relationships table

    User->>Notebook: visualize_knowledge_graph()
    
    Notebook->>Pandas: Load entities
    Pandas-->>Notebook: Entities data
    Notebook->>Pandas: Load relationships
    Pandas-->>Notebook: Relationships data

    Notebook->>NetworkX: Create Graph()
    loop For each entity
        Notebook->>NetworkX: add_node(name, type)
    end
    loop For each relationship
        Notebook->>NetworkX: add_edge(source, target)
    end

    Notebook->>NetworkX: spring_layout(G)
    NetworkX-->>Notebook: Node positions

    Notebook->>Matplotlib: Create figure(15, 10)
    Notebook->>Matplotlib: draw(G, pos, colors)
    Notebook->>Matplotlib: Add title and legend
    Notebook->>Matplotlib: savefig(knowledge_graph.png)
    Notebook->>Matplotlib: show()
    
    Matplotlib-->>User: Display visualization
```

---

## 8. Component Diagram - Overall Architecture

```mermaid
flowchart TB
    subgraph Input["📥 Input Layer"]
        DOC[("📄 Documents\n(input/*.txt)")]
        ENV["🔑 Environment\n(.env)"]
        CONFIG["⚙️ Configuration\n(settings.yaml)"]
    end

    subgraph Processing["⚡ Processing Layer"]
        CHUNKER["📦 Chunker"]
        EXTRACTOR["🔍 Entity Extractor"]
        GRAPH_BUILDER["🕸️ Graph Builder"]
        COMMUNITY["👥 Community Detector"]
        EMBEDDER["📊 Embedder"]
    end

    subgraph External["☁️ External Services"]
        LLM["🤖 LLM API\n(GPT-4o-mini)"]
        EMBED_API["📈 Embedding API\n(text-embedding-3-small)"]
    end

    subgraph Storage["💾 Storage Layer"]
        ENTITIES[("entities.parquet")]
        RELS[("relationships.parquet")]
        COMMUNITIES[("communities.parquet")]
        EMBEDDINGS[("embeddings.parquet")]
    end

    subgraph Query["🔎 Query Layer"]
        LOCAL["🎯 Local Search"]
        GLOBAL["🌐 Global Search"]
    end

    subgraph Output["📤 Output Layer"]
        RESPONSE["💬 Query Response"]
        VIZ["📊 Visualization"]
    end

    DOC --> CHUNKER
    ENV --> CONFIG
    CONFIG --> CHUNKER
    CONFIG --> EXTRACTOR
    
    CHUNKER --> EXTRACTOR
    EXTRACTOR <--> LLM
    EXTRACTOR --> GRAPH_BUILDER
    GRAPH_BUILDER --> COMMUNITY
    COMMUNITY <--> LLM
    COMMUNITY --> EMBEDDER
    EMBEDDER <--> EMBED_API

    GRAPH_BUILDER --> ENTITIES
    GRAPH_BUILDER --> RELS
    COMMUNITY --> COMMUNITIES
    EMBEDDER --> EMBEDDINGS

    ENTITIES --> LOCAL
    RELS --> LOCAL
    EMBEDDINGS --> LOCAL
    COMMUNITIES --> GLOBAL
    
    LOCAL <--> LLM
    GLOBAL <--> LLM
    
    LOCAL --> RESPONSE
    GLOBAL --> RESPONSE
    
    ENTITIES --> VIZ
    RELS --> VIZ
```

---

## 9. State Diagram - GraphRAG Pipeline States

```mermaid
stateDiagram-v2
    [*] --> Uninitialized

    Uninitialized --> Configured: Create settings.yaml
    Configured --> DataPrepared: Add input documents
    
    DataPrepared --> Indexing: Run graphrag index
    
    state Indexing {
        [*] --> Chunking
        Chunking --> ExtractingEntities
        ExtractingEntities --> BuildingGraph
        BuildingGraph --> DetectingCommunities
        DetectingCommunities --> GeneratingEmbeddings
        GeneratingEmbeddings --> SavingArtifacts
        SavingArtifacts --> [*]
    }
    
    Indexing --> Indexed: Indexing complete
    Indexing --> Error: Indexing failed
    
    Error --> DataPrepared: Fix issues
    
    Indexed --> Querying: Run query
    
    state Querying {
        [*] --> SelectMethod
        SelectMethod --> LocalSearch: method=local
        SelectMethod --> GlobalSearch: method=global
        
        LocalSearch --> GeneratingResponse
        GlobalSearch --> GeneratingResponse
        GeneratingResponse --> [*]
    }
    
    Querying --> Indexed: Query complete
    
    Indexed --> Exploring: Explore outputs
    Exploring --> Indexed: Done exploring
    
    Indexed --> Visualizing: Visualize graph
    Visualizing --> Indexed: Done visualizing
    
    Indexed --> Cleanup: Run cleanup
    Cleanup --> Uninitialized: Files removed
```

---

## 10. Entity Relationship Diagram - Data Model

```mermaid
erDiagram
    DOCUMENT ||--o{ TEXT_UNIT : "split into"
    TEXT_UNIT ||--o{ ENTITY : "contains"
    TEXT_UNIT ||--o{ RELATIONSHIP : "contains"
    TEXT_UNIT ||--o{ CLAIM : "contains"
    
    ENTITY ||--o{ RELATIONSHIP : "source"
    ENTITY ||--o{ RELATIONSHIP : "target"
    ENTITY }o--o{ COMMUNITY : "belongs to"
    
    COMMUNITY ||--o{ COMMUNITY_REPORT : "has"
    
    ENTITY {
        string id PK
        string name
        string type
        string description
        vector embedding
    }
    
    RELATIONSHIP {
        string id PK
        string source FK
        string target FK
        string description
        float weight
    }
    
    COMMUNITY {
        string id PK
        int level
        string title
        list member_ids
    }
    
    COMMUNITY_REPORT {
        string id PK
        string community_id FK
        string summary
        float rank
    }
    
    TEXT_UNIT {
        string id PK
        string document_id FK
        string text
        int chunk_index
        vector embedding
    }
    
    DOCUMENT {
        string id PK
        string file_path
        string title
        string raw_content
    }
    
    CLAIM {
        string id PK
        string subject_id FK
        string object_id FK
        string claim_type
        string description
    }
```

---

## Summary

These diagrams illustrate:

1. **Class Diagrams**: The main components and their relationships in the GraphRAG system
2. **Sequence Diagrams**: The flow of operations for setup, indexing, and querying
3. **Component Diagram**: The overall system architecture
4. **State Diagram**: The lifecycle states of a GraphRAG project
5. **ER Diagram**: The data model and relationships between entities

The GraphRAG toolkit transforms unstructured documents into a queryable knowledge graph through:
- **Entity extraction** using LLMs
- **Graph construction** with relationships
- **Community detection** for hierarchical understanding
- **Dual query modes** (local for specific facts, global for synthesis)

