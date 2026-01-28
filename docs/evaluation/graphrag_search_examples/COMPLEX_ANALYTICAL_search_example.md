## Example of COMPLEX ANALYTICAL search for evaluation of GraphRAG designs

### Dataset (Mini Knowledge Graph)

#### Entities & Relationships:

Employees:
```
Alice -> WORKS_AT -> Company_A
Alice -> HAS_SKILL -> Python
Alice -> HAS_SKILL -> Machine_Learning
Alice -> SALARY -> 120K
Alice -> JOINED -> 2020-01-01
Alice -> MANAGES -> Project_X
Alice -> LOCATION -> New_York
Bob -> WORKS_AT -> Company_A
Bob -> HAS_SKILL -> Python
Bob -> HAS_SKILL -> Data_Engineering
Bob -> SALARY -> 100K
Bob -> JOINED -> 2021-06-01
Bob -> CONTRIBUTES_TO -> Project_X
Bob -> CONTRIBUTES_TO -> Project_Y
Bob -> LOCATION -> New_York
Carol -> WORKS_AT -> Company_B
Carol -> HAS_SKILL -> Python
Carol -> HAS_SKILL -> Machine_Learning
Carol -> SALARY -> 150K
Carol -> JOINED -> 2019-03-01
Carol -> MANAGES -> Project_Z
Carol -> LOCATION -> San_Francisco
David -> WORKS_AT -> Company_B
David -> HAS_SKILL -> Java
David -> HAS_SKILL -> Backend
David -> SALARY -> 90K
David -> JOINED -> 2022-01-01
David -> CONTRIBUTES_TO -> Project_Z
David -> LOCATION -> Austin
```

Companies: 
```
Company_A -> INDUSTRY -> FinTech
Company_A -> REVENUE -> 10M
Company_A -> FOUNDED -> 2018
Company_B -> INDUSTRY -> AI_Healthcare
Company_B -> REVENUE -> 25M
Company_B -> FOUNDED -> 2015
```

Projects:
```
Project_X -> STATUS -> Completed
Project_X -> BUDGET -> 500K
Project_X -> STARTED -> 2021-01-01
Project_X -> COMPLETED -> 2022-06-01
Project_X -> USES_TECH -> Python
Project_X -> SUCCESS_SCORE -> 8.5
Project_Y -> STATUS -> Active
Project_Y -> BUDGET -> 300K
Project_Y -> STARTED -> 2022-09-01
Project_Y -> USES_TECH -> Python
Project_Y -> SUCCESS_SCORE -> 7.0
Project_Z -> STATUS -> Active
Project_Z -> BUDGET -> 800K
Project_Z -> STARTED -> 2020-06-01
Project_Z -> USES_TECH -> Python
Project_Z -> USES_TECH -> Java
Project_Z -> SUCCESS_SCORE -> 9.0
```

#### Document Chunks:

* "Alice leads Project X at Company A with a $120K salary, specializing in Python and ML since joining in 2020."

* "Bob joined Company A in mid-2021, contributing to both Project X and Y with Python and data engineering expertise earning $100K."

* "Carol manages the high-performing Project Z at Company B, earning $150K with strong Python and ML background since 2019."

* "David works on Project Z's backend infrastructure at Company B, using Java skills he brought when joining in 2022 at $90K."

* "Company A operates in FinTech with $10M revenue, while Company B focuses on AI Healthcare with $25M revenue."

* "Project X completed successfully in June 2022 with an 8.5 success score. Project Z has been the longest-running active project with a 9.0 score."

### COMPLEX ANALYTICAL Question

"Which company gets better ROI on their Python projects: the one that pays higher average salaries or lower? Consider project success scores, budgets, and employee costs."

### Expected Answer

Company B gets better ROI despite paying higher average salaries.

### Step-by-step Analysis:

**1. Identify Python Projects**:

* Company A: Project X (Python), Project Y (Python)

* Company B: Project Z (Python + Java)

**2. Calculate Average Salaries**:

* Company A employees: Alice ($120K), Bob ($100K) = $110K average

* Company B employees: Carol ($150K), David ($90K) = $120K average

* Company B pays $10K more on average ✓

**3. Calculate Project Metrics**:

**Company A**:

* **Project X**:

  * Budget: $500K, Success: 8.5

  * Team: Alice (manager), Bob (contributor)

  * ROI metric: 8.5 / $500K = 0.017 per $K

* **Project Y**:

  * Budget: $300K, Success: 7.0

  * Team: Bob (contributor)

  * ROI metric: 7.0 / $300K = 0.023 per $K

* **Average Company A ROI**: (0.017 + 0.023) / 2 = 0.020 per $K

**Company B**:

* Project Z:

  * Budget: $800K, Success: 9.0

  * Team: Carol (manager), David (contributor)

  * ROI metric: 9.0 / $800K = 0.011 per $K

* **Average Company B ROI**: 0.011 per $K

**This shows Company A has better ROI!**

**Revised Analysis (Including Employee Costs)**:

**Total Project Cost = Budget + (Annual Salaries × Project Duration)**

**Company A - Project X**:

* Budget: $500K

* Duration: 1.5 years (Jan 2021 - June 2022)
  
* Employee cost: (Alice $120K + Bob $100K) × 1.5 = $330K
  
* **Total cost: $830K**
  
* **ROI: 8.5 / $830K = 0.010 per $K**
  
**Company A - Project Y**:

* Budget: $300K

* Duration: ~1.3 years (Sept 2022 - Jan 2024, ongoing)

* Employee cost: Bob $100K × 1.3 = $130K

* **Total cost: $430K**

* **ROI: 7.0 / $430K = 0.016 per $K**

**Company A Average ROI: 0.013 per $K**

**Company B - Project Z**:

* Budget: $800K

* Duration: 3.5 years (June 2020 - Jan 2024, ongoing)

* Employee cost: (Carol $150K + David $90K × 2 years only) × time

* Carol: $150K × 3.5 = $525K

* David: $90K × 2 = $180K

* **Total cost: $1,505K**

* **ROI: 9.0 / $1,505K = 0.006 per $K**

**Company B Average ROI: 0.006 per $K**

**Final Answer**:

**Company A (lower salaries) gets better ROI!**

Despite Company B's higher project success scores (9.0 vs 8.5/7.0), Company A achieves **2x better ROI (0.013 vs 0.006)** because:

* Shorter project durations reduce accumulated salary costs

* Lower salaries ($110K vs $120K average)
* More efficient budget utilization

**Key Insight**: Higher salaries don't necessarily mean worse ROI, but in this case, the combination of long project duration + high salaries + large budget at Company B reduces overall ROI despite excellent execution quality.

### Why This Tests GraphRAG's COMPLEX ANALYTICAL Capability

**Combines Multiple Query Types**:

* ✅ **FILTERING**: Find Python projects only

* ✅ **AGGREGATION**: Calculate average salaries per company

* ✅ **TEMPORAL**: Consider project durations

* ✅ **COMPARATIVE**: Compare ROI between companies

* ✅ **RELATIONSHIP**: Trace employees → projects → companies

* ✅ **MATHEMATICAL**: Complex calculations (ROI, costs, averages)

**Requires Multi-Step Reasoning**:

1. Identify companies with Python projects
2. Trace which employees work on which projects
3. Calculate salary aggregates per company
4. Retrieve project budgets and success scores
5. Calculate project durations from temporal data
6. Compute total costs (budget + salaries × duration)
7. Calculate ROI metrics
8. Compare and rank results

**Pattern Detection**:

* Understanding that "better ROI" requires balancing success vs cost

* Recognizing that duration impacts total cost

* Identifying that the "higher paying" company needs to be determined first

**Traditional RAG Fails Completely**:

* Can't perform mathematical calculations

* Can't aggregate across multiple entity types

* Can't trace complex multi-hop relationships

* Can't combine temporal, financial, and performance data

* Would likely just retrieve text mentioning "salary" or "project success"

**Graph Advantage**:

* Structured queries for each component

* Join operations across entities

* Precise numeric computations

* Relationship traversal for team composition

* Temporal arithmetic for durations

### Bonus Complex Analytical Questions

**Q2**: **"Which skill combination (appearing in at least 2 people) leads to managing the highest-budget projects?"**

* **Answer**: Python + Machine_Learning (Alice and Carol both manage projects; Carol's Project Z has $800K budget, highest)

* **Tests**: Skill pattern detection + role filtering + budget comparison

**Q3**: **"Is there a correlation between company revenue and employee retention (tenure)?"**

* **Answer**: Company B ($25M revenue) has longer average tenure (Carol: 4.8 years, David: 2 years = 3.4 avg) vs Company A ($10M revenue, Alice: 4 years, Bob: 2.6 years = 3.3 avg). Weak positive correlation.

* **Tests**: Temporal calculations + aggregation + correlation analysis

**Q4**: **"Which location has the most cost-efficient ML talent (ML skill / salary ratio)?"**

* **Answer**: New York (Alice: 2 skills/$120K = 0.0167) vs San Francisco (Carol: 2 skills/$150K = 0.0133). New York is 25% more cost-efficient.

* **Tests**: Geographic analysis + skill counting + salary comparison

This demonstrates GraphRAG's ability to perform multi-dimensional analysis combining structure, semantics, mathematics, and reasoning - the holy grail of analytical queries.
