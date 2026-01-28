### Distinction between RELATIONSHIP search and DRIFT search for evaluation of GraphRAG designs

#### Key Differences

**DRIFT Questions**

* Navigate from entity A to a **SPECIFIC** entity B through a defined relationship path

* Typically have a **singular** answer (one person, one place, one thing)

* Focus on **directed traversal** with a clear endpoint

* Example: "Which city for Alice's collaborator's mentee?" → Answer: **Boston** (one specific location)

**RELATIONSHIP Questions**

* Find **ALL entities** connected to entity A by certain relationship types

* Typically have **multiple answers** (set of entities)

* Focus on **network analysis** and connection patterns

* Example: "Who does Alice influence?" → Answer: **Bob, Carol, David** (set of people)

The Question posed in [RELATIONSHIP_search_example.md](https://github.com/nike-edaaml/genai_tutorials/blob/main/eval_frameworks/docs/graphrag_eval_questions/RELATIONSHIP_search_example.md) **IS Partially DRIFT**

The question **"Who are all the people that Alice has direct or indirect influence over?"** does involve:

* ✅ Multi-hop traversal (Alice → Bob → David)
  
* ✅ Following relationship chains

* ✅ Inferential reasoning about connections

**But it differs from pure DRIFT because**:

* ❌ No specific target entity (wants ALL influenced people, not "Alice's mentee's collaborator")
  
* ❌ Asks for a set/collection rather than singular answer
  
* ❌ Requires network aggregation rather than path finding

#### Better Pure RELATIONSHIP Questions

Questions that are more clearly RELATIONSHIP-focused (less DRIFT-like):

**Q1**: **"What types of relationships does Alice have?"**

* Answer: REPORTS_TO, MENTORS, COLLABORATES_WITH, MANAGES

* Tests: Relationship type enumeration (no traversal needed)

**Q2**: **"Who has bidirectional relationships?"**

* Answer: Alice ↔ Carol (both collaborate with each other)

* Tests: Symmetric relationship detection

**Q3**: **"What is the density of collaboration relationships in the organization?"**

* Answer: 2 collaboration edges / 6 people = network density metric

* Tests: Graph topology analysis

#### More Pure DRIFT Examples

**Q1**: **"What project does the person who Bob collaborates with report to their manager about?"**

* Bob collaborates with David

* David reports to Carol

* Carol manages Project_Beta

* Answer: **Project_Beta**

**Q2**: **"Who is the grand-manager of Alice's mentee?"**

* Alice mentors Bob

* Bob reports to Alice

* Alice reports to Eve

* Answer: Eve

#### The Spectrum

```
Pure LOCAL ← → Pure RELATIONSHIP ← → Pure DRIFT ← →    Pure GLOBAL
"What is       "Who does Alice      "What city is      "What's the most
Bob's role?"   collaborate with?"    Alice's mentee's  common skill?"
               (immediate            colleague in?"    (aggregate all)
                connections)         (specific path)
```

The question posed in [RELATIONSHIP_search_example.md](https://github.com/nike-edaaml/genai_tutorials/blob/main/eval_frameworks/docs/graphrag_eval_questions/RELATIONSHIP_search_example.md) sits between **RELATIONSHIP** and **DRIFT** - it requires **DRIFT**-style traversal but asks for **RELATIONSHIP**-style network analysis.

