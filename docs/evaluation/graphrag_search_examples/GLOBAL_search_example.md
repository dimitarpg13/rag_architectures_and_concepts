## Example of GLOBAL search for evaluation of GraphRAG designs

### Dataset (Mini Knowledge Graph)

#### Entities & Relationships:

```
Project_Alpha -> USES_TECHNOLOGY -> Python
Project_Alpha -> HAS_CHALLENGE -> Scalability
Project_Alpha -> TEAM_SIZE -> 5_people
Project_Alpha -> STATUS -> Active
Project_Beta -> USES_TECHNOLOGY -> Java
Project_Beta -> HAS_CHALLENGE -> Legacy_Integration
Project_Beta -> TEAM_SIZE -> 8_people
Project_Beta -> STATUS -> Active
Project_Gamma -> USES_TECHNOLOGY -> Python
Project_Gamma -> HAS_CHALLENGE -> Scalability
Project_Gamma -> TEAM_SIZE -> 3_people
Project_Gamma -> STATUS -> Completed
Project_Delta -> USES_TECHNOLOGY -> JavaScript
Project_Delta -> HAS_CHALLENGE -> Performance
Project_Delta -> TEAM_SIZE -> 6_people
Project_Delta -> STATUS -> Active
```

#### Document Chunks:

* "Project Alpha is building a Python-based data pipeline but facing scalability issues with 5 team members."
  
* "Project Beta team of 8 is integrating Java microservices with legacy systems."
  
* "Project Gamma successfully completed their Python analytics platform despite scalability constraints with just 3 developers."
  
* "Project Delta's 6-person team is optimizing JavaScript performance for their web application."

### GLOBAL Question:

"What are the most common technical challenges across all our active projects?"

### Expected Answer:

Across the 3 active projects (Alpha, Beta, Delta):

* Scalability - affects 1 project (Project Alpha)

* Legacy Integration - affects 1 project (Project Beta)
  
* Performance - affects 1 project (Project Delta)
  
* Summary: Each active project faces distinct technical challenges. There's no dominant pattern, but infrastructure concerns (scalability and performance) affect 2 out of 3 active projects (67%).

### Why This Tests GraphRAG's GLOBAL Capability:

**Requires aggregation**: Must analyze ALL projects, not trace specific relationships.

**Pattern detection**: Needs to identify themes across the entire corpus, not answer about specific entities.

**Filtering + summarization**: Must filter to "active" projects, then aggregate "challenges" across them.

**Community detection**: A good GraphRAG system would identify challenge clusters and summarize patterns.

**Traditional RAG struggles**: Vector search might retrieve individual project challenges but won't aggregate/summarize patterns across all documents.

**Graph advantage**: Can query all `PROJECT -> HAS_CHALLENGE` relationships where `STATUS = Active`, then generate summary statistics and insights about the portfolio.

This demonstrates GraphRAG's ability to answer thematic, analytical questions about the entire dataset rather than navigating specific entity relationships.
