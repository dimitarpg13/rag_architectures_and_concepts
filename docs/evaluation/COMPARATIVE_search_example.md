## Example of COMPARATIVE search for evaluation of GraphRAG designs

### Dataset (Mini Knowledge Graph)

#### Entities & Relationships:

```
Company_A -> REVENUE -> 10M
Company_A -> EMPLOYEES -> 50
Company_A -> FOUNDED -> 2015
Company_A -> LOCATION -> New_York
Company_A -> INDUSTRY -> FinTech
Company_B -> REVENUE -> 25M
Company_B -> EMPLOYEES -> 120
Company_B -> FOUNDED -> 2018
Company_B -> LOCATION -> San_Francisco
Company_B -> INDUSTRY -> FinTech
Company_C -> REVENUE -> 8M
Company_C -> EMPLOYEES -> 35
Company_C -> FOUNDED -> 2012
Company_C -> LOCATION -> Boston
Company_C -> INDUSTRY -> FinTech
Company_D -> REVENUE -> 15M
Company_D -> EMPLOYEES -> 80
Company_D -> FOUNDED -> 2016
Company_D -> LOCATION -> Austin
Company_D -> INDUSTRY -> HealthTech
```
#### Document Chunks:

* "Company A, founded in 2015, operates in New York's FinTech sector with 50 employees and $10M annual revenue."
  
* "Company B is the largest in our portfolio, based in San Francisco with 120 employees and $25M revenue since 2018."
  
* "Company C started in Boston in 2012 as a boutique FinTech firm, now at 35 employees with $8M revenue."
  
* "Company D pivoted to HealthTech in Austin in 2016, growing to 80 employees and $15M revenue."

### COMPARATIVE Question:

"Which FinTech company has the best revenue per employee ratio?"

### Expected Answer:

Company A has the best revenue per employee ratio in FinTech:

Company A: $10M / 50 = $200K per employee

Company B: $25M / 120 = $208K per employee ✓ BEST

Company C: $8M / 35 = $229K per employee ✓✓ ACTUALLY BEST!

Wait, let me recalculate:

Company A: 10,000,000 / 50 = 200,000

Company B: 25,000,000 / 120 = 208,333

Company C: 8,000,000 / 35 = 228,571

**Correct Answer**: **Company C** with $229K revenue per employee

_(Note: Company D is excluded because it's HealthTech, not FinTech)_

### Why This Tests GraphRAG's COMPARATIVE Capability:

**Filtering required**: Must identify only FinTech companies (exclude Company D).

**Multi-attribute retrieval**: Need both REVENUE and EMPLOYEES for each entity.

**Mathematical computation**: Calculate derived metric (revenue/employees) not stored in graph.

**Ranking/comparison**: Compare computed values across entities to find "best".

**Traditional RAG fails**: Vector search can't retrieve structured data for calculations, and can't perform mathematical comparisons across entities.

**Graph advantage**: Can query all entities where INDUSTRY = FinTech, retrieve their REVENUE and EMPLOYEES attributes, compute ratios, and rank results.

### Bonus Comparative Questions for Same Dataset

**Q2**: "Which company is the oldest?"

* **Answer**: Company C (founded 2012)
* **Tests**: Temporal comparison, finding MIN value

**Q3**: "Do larger companies (by employees) generate proportionally more revenue?"

* **Answer**: Not necessarily - Company C has fewest employees but highest efficiency
* **Tests**: Correlation analysis, pattern detection across multiple attributes

**Q4**: "Which companies have more than 50 employees but less than $20M revenue?"

* **Answer**: Company A (50 employees, $10M) and Company D (80 employees, $15M)
* **Tests**: Multi-criteria filtering with range conditions

This demonstrates GraphRAG's ability to perform structured comparisons and computations that require precise attribute retrieval and mathematical reasoning.
