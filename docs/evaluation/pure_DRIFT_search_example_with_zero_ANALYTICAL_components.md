## Example of pure DRIFT search with zero ANALYTICAL components for evaluation of GraphRAG designs

### Dataset (Mini Knowledge Graph)

#### Entities & Relationships:

Simple organization structure
```
Alice -> MANAGES -> Bob
Bob -> MENTORS -> Carol
Carol -> WORKS_ON -> Project_Phoenix
Project_Phoenix -> LOCATED_IN -> Building_C
Building_C -> HAS_ADDRESS -> "123 Tech Street, Austin"
David -> MANAGES -> Eve
Eve -> MENTORS -> Frank
Frank -> WORKS_ON -> Project_Atlas
Project_Atlas -> LOCATED_IN -> Building_A
Building_A -> HAS_ADDRESS -> "456 Innovation Ave, Boston"
Carol -> REPORTS_TO -> Bob
Frank -> REPORTS_TO -> Eve
```

#### Document Chunks:

* "Alice manages Bob, who mentors Carol at the company."

* "Carol works on Project Phoenix, which is based in Building C."

* "Building C is located at 123 Tech Street in Austin."

* "David oversees Eve, who mentors Frank on his projects."

* "Frank contributes to Project Atlas in Building A at 456 Innovation Ave in Boston."

### Pure DRIFT Question

  "What is the address of the building where Alice's direct report's mentee works?"

### Expected Answer

  "123 Tech Street, Austin"

### Reasoning Path (Pure Traversal, No Analytics)

```
Step 1: Alice's direct report
        Alice -> MANAGES -> Bob

Step 2: Bob's mentee
        Bob -> MENTORS -> Carol

Step 3: Carol works on
        Carol -> WORKS_ON -> Project_Phoenix

Step 4: Project located in
        Project_Phoenix -> LOCATED_IN -> Building_C

Step 5: Building address
        Building_C -> HAS_ADDRESS -> "123 Tech Street, Austin"
```

### Why This Is Pure DRIFT (Zero Analytics)

* ✅ Single directed path: One clear route from start (Alice) to end (address)

* ✅ No aggregation: Not counting anything, not averaging, not summing

* ✅ No computation: No math, no calculations, no ratios

* ✅ No comparison: Not comparing multiple entities or values

* ✅ No filtering: Not selecting from multiple options based on criteria

* ✅ No pattern detection: Not looking for themes or trends

* ✅ Specific answer: One concrete value that exists in the graph

* ✅ 5-hop traversal: Pure relationship following across 5 edges

### More Pure DRIFT Examples (Same Dataset)

**Q2**: **"Where does the person mentored by David's direct report work?"**

**Path**:

```
David -> MANAGES -> Eve Eve -> MENTORS -> Frank Frank -> WORKS_ON -> Project_Atlas
```
**Answer**: Project_Atlas (specific project, no analysis)
---
**Q3**: **"What building houses the project that Bob's mentee works on?"**

**Path**:

```
Bob -> MENTORS -> Carol Carol -> WORKS_ON -> Project_Phoenix Project_Phoenix -> LOCATED_IN -> Building_C
```

**Answer**: **Building_C** (specific building, no analysis)
---
**Q4**: "Who manages the person who mentors the person working on Project Atlas?"

**Path** (reverse lookup):

```Project_Atlas <- WORKS_ON <- Frank Frank <- MENTORS <- Eve Eve <- MANAGES <- David```

**Answer**: **David** (specific person, no analysis)
---
**What Makes These Pure DRIFT**

All these questions:

* Start at a specific entity (Alice, David, Bob, Project)

* Follow a predetermined path through relationships

* End at a specific entity or value

* Require zero computation or aggregation

* Have one correct answer that exists in the graph

**Contrast: NOT Pure DRIFT**

Here's what would make it analytical (and thus NOT pure DRIFT):

❌ "How many people does Alice indirectly influence?" (requires counting)

❌ "Which manager has more mentees?" (requires comparison)

❌ "What's the average number of hops from managers to projects?" (requires aggregation)

❌ "Do all mentees work on projects?" (requires pattern checking across multiple paths)

### Summary
Pure DRIFT is like following GPS directions: "Turn left at Alice, go straight through Bob, turn right at Carol, your destination Building C is on the right." No decisions, no calculations, no analysis - just follow the path!

