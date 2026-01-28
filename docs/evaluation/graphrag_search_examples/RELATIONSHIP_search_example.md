## Example of RELATIONSHIP search for evaluation of GraphRAG designs

### Dataset (Mini Knowledge Graph)

#### Entities & Relationships:

Professional Relationships
```
Alice -> REPORTS_TO -> Eve
Alice -> MENTORS -> Bob
Alice -> COLLABORATES_WITH -> Carol
Alice -> MANAGES -> Project_Alpha
Bob -> REPORTS_TO -> Alice
Bob -> COLLABORATES_WITH -> David
Bob -> CONTRIBUTES_TO -> Project_Alpha
Carol -> REPORTS_TO -> Frank
Carol -> MENTORS -> David
Carol -> COLLABORATES_WITH -> Alice
Carol -> MANAGES -> Project_Beta
David -> REPORTS_TO -> Carol
David -> CONTRIBUTES_TO -> Project_Beta
David -> COLLABORATES_WITH -> Bob
Eve -> MANAGES -> Alice
Eve -> MANAGES -> Frank
Frank -> REPORTS_TO -> Eve
Frank -> MANAGES -> Carol
```

#### Document Chunks:

* "Alice reports to Eve and mentors Bob while collaborating with Carol on cross-functional initiatives."

* "Bob is Alice's direct report and works closely with David on technical implementations for Project Alpha."

* "Carol manages David and reports to Frank, while also collaborating with Alice across departments."

* "David contributes to Project Beta under Carol's guidance and partners with Bob on shared components."

* "Eve is the executive overseeing both Alice and Frank, who manage their respective teams."

* "Frank manages Carol's team and reports directly to Eve at the executive level."

### RELATIONSHIP Question

 "Who are all the people that Alice has direct or indirect influence over?"

### Expected Answer

Alice has influence over 3 people through direct and indirect relationships:

**Direct influence (1 person)**:

* **Bob** - Alice mentors Bob and Bob reports to Alice

**Indirect influence (2 people)**:

* **David** - Bob collaborates with David, extending Alice's reach through Bob

* **Carol** - Alice collaborates with Carol (peer relationship, but represents influence through partnership)

**Alternative interpretation (management chain only)**: If we strictly consider reporting/management hierarchy:

* Bob reports to Alice (direct)
  
* That's it - no one reports to Bob, so Alice's hierarchical influence ends there

**Most comprehensive answer**: Alice influences 2 people directly through her organizational position:

**1. Bob** (direct report + mentee)

**2. Carol** (collaboration partner - peer influence)
   
And through Bob's network: **3. David** (Bob's collaborator)

### Why This Tests GraphRAG's RELATIONSHIP Capability

**Multi-hop traversal**: Must follow relationship chains from Alice outward.

**Relationship type awareness**: Different relationship types (REPORTS_TO, MENTORS, COLLABORATES_WITH) have different influence semantics.

**Bidirectional reasoning**: Must understand that "Alice mentors Bob" implies influence, even without explicit influence edges.

**Network effect**: Direct vs indirect influence requires understanding relationship transitivity.

**Traditional RAG fails**: Vector similarity might find mentions of people connected to Alice but can't trace the relationship structure or distinguish direct from indirect connections.

**Graph advantage**: Can execute graph traversal queries like:
```cypher
MATCH (alice)-[r:REPORTS_TO|MENTORS|MANAGES*1..2]-(person) WHERE alice.name = 'Alice' RETURN person
```

### Bonus Relationship Questions for Same Dataset

**Q2**: **"Who is Bob's grand-manager?"** (2-hop management chain)

* **Answer**: Eve

* **Reasoning**: `Bob → REPORTS_TO → Alice → REPORTS_TO → Eve`
  
* **Tests**: Specific relationship path traversal

**Q3**: **"How many people does Eve manage directly or indirectly?"**

* **Answer**: 5 people

  * Direct: Alice, Frank (2)

  * Through Alice: Bob (1)

  * Through Frank: Carol (1)

  * Through Carol: David (1)

* **Tests**: Counting all nodes in subtree

**Q4**: **"What is the shortest path between Bob and Carol?"**

* **Answer**: `Bob → COLLABORATES_WITH → Alice → COLLABORATES_WITH → Carol` (2 hops)

* **Tests**: Shortest path algorithms

**Q5**: **"Who are David's peers?"** (people at same level in hierarchy)

* **Answer**: Bob (both are individual contributors who report to someone)

* **Tests**: Understanding organizational structure equivalence

**Q6**: **"Which projects have overlapping team members?"**

* **Answer**: Projects Alpha and Beta share Bob-David collaboration
 
* **Reasoning**:
  
  * Bob contributes to Alpha, collaborates with David

  * David contributes to Beta, collaborates with Bob

* **Tests**: Finding connections through shared relationships

**Q7**: **"Who has the most collaboration relationships?"**

* **Answer**: Alice and Carol (both have 1 collaboration each: `Alice↔Carol`)

* **Tests**: Degree centrality, counting relationship instances

**Q8**: **"What is the management depth of the organization?"**

* **Answer**: 3 levels

  * Level 1: Eve (top)

  * Level 2: Alice, Frank

  * Level 3: Bob, Carol

  * Level 4: David

* **Actually 4 levels!**

* **Tests**: Tree depth calculation

**Q9**: **"Does anyone have both a mentor and a manager relationship with the same person?"**

* **Answer**: Yes, Bob has both REPORTS_TO and is MENTORED_BY Alice

* **Tests**: Multi-edge detection between same nodes

**Q10**: **"Who are the boundary spanners?"** (people connecting different parts of the organization)

* **Answer**: Alice and Carol (they collaborate across reporting chains)

* **Reasoning**: Alice (under Eve→Alice) collaborates with Carol (under Eve→Frank→Carol)

* **Tests**: Bridge detection in social networks

This demonstrates GraphRAG's ability to perform relationship analysis, network traversal, and structural queries that are fundamental to understanding connected data.

