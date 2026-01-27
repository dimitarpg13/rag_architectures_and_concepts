# GraphRAG Test Dataset & Questions

## Dataset Overview
A simple organizational dataset with people, projects, skills, and locations to test various GraphRAG query capabilities.

---

## Knowledge Graph Schema

### Entities & Relationships

```
# People
Alice -> WORKS_AT -> TechCorp
Alice -> HAS_SKILL -> Python
Alice -> HAS_SKILL -> Machine_Learning
Alice -> LOCATED_IN -> San_Francisco
Alice -> MANAGES -> Project_Phoenix

Bob -> WORKS_AT -> TechCorp
Bob -> HAS_SKILL -> Java
Bob -> HAS_SKILL -> Databases
Bob -> LOCATED_IN -> New_York
Bob -> COLLABORATES_WITH -> Alice
Bob -> CONTRIBUTES_TO -> Project_Phoenix

Carol -> WORKS_AT -> DataLabs
Carol -> HAS_SKILL -> Python
Carol -> HAS_SKILL -> Machine_Learning
Carol -> HAS_SKILL -> Statistics
Carol -> LOCATED_IN -> Boston
Carol -> MENTORED_BY -> Alice
Carol -> MANAGES -> Project_Atlas

David -> WORKS_AT -> DataLabs
David -> HAS_SKILL -> JavaScript
David -> HAS_SKILL -> Frontend
David -> LOCATED_IN -> Austin
David -> REPORTS_TO -> Carol
David -> CONTRIBUTES_TO -> Project_Atlas

Eve -> WORKS_AT -> StartupXYZ
Eve -> HAS_SKILL -> Python
Eve -> HAS_SKILL -> DevOps
Eve -> LOCATED_IN -> San_Francisco
Eve -> FOUNDER_OF -> StartupXYZ

# Organizations
TechCorp -> PARTNERS_WITH -> DataLabs
TechCorp -> LOCATED_IN -> San_Francisco
TechCorp -> INDUSTRY -> Technology
TechCorp -> EMPLOYEES_COUNT -> 50

DataLabs -> LOCATED_IN -> Boston
DataLabs -> INDUSTRY -> Data_Science
DataLabs -> EMPLOYEES_COUNT -> 30

StartupXYZ -> LOCATED_IN -> San_Francisco
StartupXYZ -> INDUSTRY -> AI
StartupXYZ -> EMPLOYEES_COUNT -> 5

# Projects
Project_Phoenix -> STATUS -> Active
Project_Phoenix -> USES_TECH -> Python
Project_Phoenix -> USES_TECH -> AWS
Project_Phoenix -> BUDGET -> 500K
Project_Phoenix -> START_DATE -> 2024-01-15
Project_Phoenix -> CHALLENGE -> Scalability

Project_Atlas -> STATUS -> Active
Project_Atlas -> USES_TECH -> Python
Project_Atlas -> USES_TECH -> TensorFlow
Project_Atlas -> BUDGET -> 300K
Project_Atlas -> START_DATE -> 2024-03-01
Project_Atlas -> CHALLENGE -> Data_Quality

Project_Legacy -> STATUS -> Completed
Project_Legacy -> USES_TECH -> Java
Project_Legacy -> BUDGET -> 200K
Project_Legacy -> START_DATE -> 2023-06-01
Project_Legacy -> END_DATE -> 2024-01-01
```

### Document Chunks

**Doc 1:** "Alice is a senior engineer at TechCorp based in San Francisco. She specializes in Python and machine learning, currently managing Project Phoenix which focuses on building scalable ML infrastructure."

**Doc 2:** "Bob works at TechCorp in New York and collaborates remotely with Alice on Project Phoenix. He brings strong Java and database expertise to the distributed systems architecture."

**Doc 3:** "Carol joined DataLabs in Boston after being mentored by Alice during her PhD. She now leads Project Atlas, applying machine learning and statistics to improve data quality for enterprise clients."

**Doc 4:** "David reports to Carol at DataLabs and handles the frontend development for Project Atlas using JavaScript and modern web frameworks from the Austin office."

**Doc 5:** "Eve founded StartupXYZ in San Francisco, focusing on AI-powered DevOps tools. She has a Python background and is building a small but ambitious team."

**Doc 6:** "TechCorp and DataLabs announced a strategic partnership to combine TechCorp's infrastructure expertise with DataLabs' data science capabilities."

**Doc 7:** "Project Phoenix at TechCorp faces scalability challenges as the team works to handle increasing ML workloads on AWS infrastructure with a $500K budget."

**Doc 8:** "Project Atlas launched in March 2024 with $300K funding, addressing data quality issues using TensorFlow-based models."

**Doc 9:** "Project Legacy was a Java-based initiative completed in early 2024 after 7 months of development with a $200K budget."

---

## Test Questions & Expected Answers

### 1. DRIFT Questions (Multi-hop Traversal)

#### Q1.1: "Which city should I visit to meet Alice's mentee's team member who works on frontend?"

**Expected Answer:** Austin

**Reasoning Path:**
1. Alice mentors Carol (Alice -> MENTORED_BY -> Carol)
2. Carol manages David (David -> REPORTS_TO -> Carol)
3. David works on frontend (David -> HAS_SKILL -> Frontend)
4. David is located in Austin (David -> LOCATED_IN -> Austin)

**What it tests:** 4-hop relationship traversal, entity resolution through possessives

---

#### Q1.2: "What programming language is used by the founder of the company in the same city where Alice works?"

**Expected Answer:** Python

**Reasoning Path:**
1. Alice works in San Francisco (Alice -> LOCATED_IN -> San_Francisco)
2. StartupXYZ is in San Francisco (StartupXYZ -> LOCATED_IN -> San_Francisco)
3. Eve founded StartupXYZ (Eve -> FOUNDER_OF -> StartupXYZ)
4. Eve has Python skill (Eve -> HAS_SKILL -> Python)

**What it tests:** Complex spatial + organizational relationships, entity co-location reasoning

---

#### Q1.3: "What is the budget of the project managed by Bob's collaborator?"

**Expected Answer:** $500K

**Reasoning Path:**
1. Bob collaborates with Alice (Bob -> COLLABORATES_WITH -> Alice)
2. Alice manages Project Phoenix (Alice -> MANAGES -> Project_Phoenix)
3. Project Phoenix has $500K budget (Project_Phoenix -> BUDGET -> 500K)

**What it tests:** Traversal through collaboration relationships to project attributes

---

### 2. GLOBAL Questions (Aggregation & Summarization)

#### Q2.1: "What are the most common technical skills across all employees?"

**Expected Answer:**
- **Python**: 4 people (Alice, Carol, Eve, and used in Projects)
- **Machine Learning**: 2 people (Alice, Carol)
- **Java**: 1 person (Bob)
- **JavaScript**: 1 person (David)
- **Databases**: 1 person (Bob)
- **DevOps**: 1 person (Eve)
- **Statistics**: 1 person (Carol)
- **Frontend**: 1 person (David)

**Most common:** Python is the dominant skill (4/5 people = 80%)

**What it tests:** Aggregation across all entities, pattern detection, frequency analysis

---

#### Q2.2: "What challenges are active projects currently facing?"

**Expected Answer:**
- **Project Phoenix**: Scalability challenges
- **Project Atlas**: Data quality issues

**Summary:** Infrastructure scalability and data quality are the primary challenges across active projects.

**What it tests:** Filtering by status, thematic summarization, challenge identification

---

#### Q2.3: "Which cities have the most technology professionals?"

**Expected Answer:**
- **San Francisco**: 2 people (Alice, Eve)
- **New York**: 1 person (Bob)
- **Boston**: 1 person (Carol)
- **Austin**: 1 person (David)

**What it tests:** Geographic distribution analysis, entity counting by location

---

#### Q2.4: "What is the total budget allocated across all active projects?"

**Expected Answer:** $800K

**Calculation:**
- Project Phoenix: $500K (Active)
- Project Atlas: $300K (Active)
- Project Legacy: $200K (Completed - not counted)

**What it tests:** Numerical aggregation with filtering, financial analysis

---

### 3. LOCAL Questions (Entity-specific Details)

#### Q3.1: "What skills does Carol have?"

**Expected Answer:** Python, Machine Learning, Statistics

**What it tests:** Direct entity attribute retrieval, simple 1-hop queries

---

#### Q3.2: "Which company does Bob work for?"

**Expected Answer:** TechCorp

**What it tests:** Basic entity relationship lookup

---

#### Q3.3: "What technologies does Project Phoenix use?"

**Expected Answer:** Python, AWS

**What it tests:** Project-specific attribute retrieval

---

### 4. COMPARATIVE Questions

#### Q4.1: "Which organization has more employees: TechCorp or DataLabs?"

**Expected Answer:** TechCorp has more employees (50 vs 30)

**What it tests:** Numerical comparison between entities

---

#### Q4.2: "Which active project has a larger budget?"

**Expected Answer:** Project Phoenix ($500K vs Project Atlas $300K)

**What it tests:** Filtering + comparison, project analysis

---

#### Q4.3: "Who has more skills: Alice or Carol?"

**Expected Answer:** Carol has 3 skills (Python, Machine Learning, Statistics), Alice has 2 skills (Python, Machine Learning). Carol has more skills.

**What it tests:** Relationship counting and comparison

---

### 5. TEMPORAL Questions

#### Q5.1: "Which project started first in 2024?"

**Expected Answer:** Project Phoenix (started January 15, 2024, before Project Atlas on March 1, 2024)

**What it tests:** Date comparison, temporal reasoning

---

#### Q5.2: "Are there any completed projects?"

**Expected Answer:** Yes, Project Legacy (completed January 1, 2024)

**What it tests:** Status filtering, temporal state identification

---

### 6. RELATIONSHIP Questions

#### Q6.1: "Who collaborates with Alice?"

**Expected Answer:** Bob

**What it tests:** Bidirectional relationship queries

---

#### Q6.2: "Which organizations are partners?"

**Expected Answer:** TechCorp and DataLabs

**What it tests:** Organization-level relationship retrieval

---

#### Q6.3: "Who reports to Carol?"

**Expected Answer:** David

**What it tests:** Hierarchical relationship navigation

---

### 7. COMPLEX ANALYTICAL Questions

#### Q7.1: "Which skills are required for working on active projects at DataLabs?"

**Expected Answer:**
- Project Atlas (active at DataLabs): Python, TensorFlow, Machine Learning, Statistics (from Carol), JavaScript, Frontend (from David)

**What it tests:** Multi-entity aggregation, organizational + project filtering

---

#### Q7.2: "Are there any people who share skills with project managers?"

**Expected Answer:**
- Alice (manages Phoenix): has Python, ML
- Carol (manages Atlas): has Python, ML, Statistics
- People with shared skills: Bob has Python (from Phoenix), Eve has Python, Carol shares with Alice

**What it tests:** Complex pattern matching, role-based filtering, skill overlap analysis

---

#### Q7.3: "Which city has employees from partnered organizations?"

**Expected Answer:**
- **San Francisco**: Alice from TechCorp
- **Boston**: Carol from DataLabs
- (TechCorp and DataLabs are partners)

**What it tests:** Multi-hop reasoning combining partnerships + locations

---

## Evaluation Criteria

### Success Metrics for GraphRAG:

1. **Accuracy**: Correct answer provided
2. **Completeness**: All relevant information included
3. **Reasoning Path**: Can trace the graph traversal logic
4. **Efficiency**: Number of hops/queries required
5. **Handling Ambiguity**: Resolves pronouns and references correctly

### Expected GraphRAG Advantages:

- **DRIFT questions**: Should handle all multi-hop queries correctly
- **GLOBAL questions**: Should aggregate across entire dataset
- **COMPARATIVE questions**: Should retrieve and compare attributes
- **TEMPORAL questions**: Should filter by dates and status
- **Complex queries**: Should combine multiple reasoning types

### Where Traditional RAG Fails:

- **Q1.1, Q1.2, Q1.3**: Multi-hop reasoning not captured by vector similarity
- **Q2.1, Q2.4**: Aggregation requires structural understanding, not semantic search
- **Q4.1, Q4.2**: Comparisons need explicit attribute retrieval
- **Q7.3**: Combining partnerships + geography is relationship-dependent

---

## Usage Instructions

1. **Build the Knowledge Graph**: Create nodes for all entities and edges for all relationships
2. **Index Document Chunks**: Use for text-based retrieval when needed
3. **Run Each Question**: Compare GraphRAG vs Traditional RAG performance
4. **Evaluate**: Use success metrics to assess quality
5. **Iterate**: Identify failure modes and improve the system

---

## Extension Ideas

To increase complexity:
- Add more temporal relationships (project timelines, employee tenure)
- Include hierarchical structures (company departments, skill levels)
- Add quantitative attributes (project success metrics, employee ratings)
- Introduce conflicting information to test reasoning
- Add more document chunks with ambiguous references

