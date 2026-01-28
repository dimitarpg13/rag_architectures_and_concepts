## Example of DRIFT search for evaluation of GraphRAG designs

### Dataset (Mini Knowledge Graph)

#### Entities & Relationships:

```
Alice -> WORKS_AT -> TechCorp
Alice -> COLLABORATES_WITH -> Bob
Bob -> WORKS_AT -> DataLabs
Bob -> MENTORS -> Carol
Carol -> WORKS_AT -> DataLabs
Carol -> SPECIALIZES_IN -> Machine Learning
TechCorp -> PARTNERS_WITH -> DataLabs
TechCorp -> LOCATED_IN -> San Francisco
DataLabs -> LOCATED_IN -> Boston
Machine Learning -> SUBFIELD_OF -> Artificial Intelligence
```
#### Document Chunks:

* "Alice is a senior engineer at TechCorp working on cloud infrastructure."

* "Bob works at DataLabs and collaborates remotely with Alice on distributed systems."

* "Carol is Bob's mentee at DataLabs, specializing in machine learning applications."

* "TechCorp and DataLabs have a strategic partnership on AI initiatives."

* "TechCorp is headquartered in San Francisco, while DataLabs operates from Boston."

### DRIFT Question

"Which city should we visit to meet Alice's collaborator's mentee who specializes in ML?"

### Expected Answer

Boston - Because:

1. Alice collaborates with Bob (Alice -> Bob)
2. Bob mentors Carol (Bob -> Carol)
3. Carol specializes in ML (Carol -> ML)
4. Carol works at DataLabs (Carol -> DataLabs)
5. DataLabs is in Boston (DataLabs -> Boston)

### Why This Tests GraphRAG

* Multi-hop reasoning: Requires traversing 5+ relationship hops across the graph.

* Entity disambiguation: "Alice's collaborator's mentee" requires resolving pronouns through relationship chains.

* Traditional RAG would fail: Vector similarity alone wouldn't connect "Alice" to "Boston" - they rarely co-occur in text chunks.

* Graph traversal wins: GraphRAG can follow: Alice → COLLABORATES_WITH → Bob → MENTORS → Carol → SPECIALIZES_IN → ML and Carol → WORKS_AT → DataLabs → LOCATED_IN → Boston

This simple example demonstrates GraphRAG's ability to perform complex relational reasoning that would be nearly impossible with pure vector similarity search.
