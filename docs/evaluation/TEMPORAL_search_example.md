## Example of TEMPORAL search for evaluation of GraphRAG designs

### Dataset (Mini Knowledge Graph)

#### Entities & Relationships:


Employee Career Events 
```
Alice -> JOINED -> TechCorp -> DATE: 2020-01-15
Alice -> PROMOTED_TO -> Senior_Engineer -> DATE: 2022-06-01
Alice -> STARTED_PROJECT -> Project_X -> DATE: 2023-03-10
Project_X -> COMPLETED -> DATE: 2023-11-20

Bob -> JOINED -> TechCorp -> DATE: 2021-07-01
Bob -> PROMOTED_TO -> Team_Lead -> DATE: 2023-01-15
Bob -> STARTED_PROJECT -> Project_Y -> DATE: 2023-05-20
Project_Y -> STATUS -> In_Progress

Carol -> JOINED -> TechCorp -> DATE: 2019-03-20
Carol -> LEFT -> TechCorp -> DATE: 2022-12-31
Carol -> STARTED_PROJECT -> Project_Z -> DATE: 2021-08-15
Project_Z -> COMPLETED -> DATE: 2022-10-05

David -> JOINED -> TechCorp -> DATE: 2022-09-01
David -> STARTED_PROJECT -> Project_X -> DATE: 2023-03-10
David -> COLLABORATES_WITH -> Alice
```

#### Document Chunks:

* "Alice joined TechCorp in January 2020 and was promoted to Senior Engineer in June 2022."
  
* "Alice started leading Project X in March 2023, which was successfully completed by November 2023."
  
* "Bob came aboard in July 2021 and became Team Lead in January 2023, currently working on Project Y."
  
* "Carol was an early employee joining in March 2019, but left at the end of 2022 after completing Project Z."
  
* "David joined in September 2022 and joined Alice on Project X when it kicked off in March 2023."

### TEMPORAL Question

"Who was at the company when Project Z was completed?"

### Expected Answer

Carol, Alice, and Bob were at TechCorp when Project Z was completed (October 5, 2022).

### Reasoning:

* Project Z completed: 2022-10-05
  
* Carol: Joined 2019-03-20, Left 2022-12-31 ✓ (still there in Oct 2022)

* Alice: Joined 2020-01-15, still employed ✓ (there in Oct 2022)

* Bob: Joined 2021-07-01, still employed ✓ (there in Oct 2022)

* David: Joined 2022-09-01, still employed ✓ (joined 1 month before!)

**Corrected Answer**: Carol, Alice, Bob, AND David (all 4 were employed on 2022-10-05)

### Why This Tests GraphRAG's TEMPORAL Capability

**Time-point reasoning**: Must determine who was employed at a specific date `(2022-10-05)`.

**Range checking**: For each employee, verify: `JOIN_DATE <= 2022-10-05 <= LEAVE_DATE` (or no leave date).

**Event sequencing**: Understanding that Carol left `AFTER` Project Z completed, so she was there.

**Historical state query**: Reconstructing the organization's state at a past moment in time.

**Traditional RAG fails**: Vector similarity can't perform date comparisons or temporal logic. Might retrieve "Carol left in 2022" and incorrectly exclude her.

**Graph advantage**: Can query all employees where `JOINED <= 2022-10-05 AND (LEFT is null OR LEFT >= 2022-10-05)`, providing precise temporal reasoning.


### Bonus Temporal Questions for Same Dataset

**Q2**: "Who got promoted first: Alice or Bob?"
Answer: Alice (June 2022 vs January 2023)
Tests: Direct temporal comparison between events

**Q3**: "How long did it take to complete Project X?"
Answer: 8 months, 10 days (March 10 to November 20, 2023)
Tests: Duration calculation between start and end dates

**Q4**: "Which projects were active in summer 2023 (June-August)?"
Answer: Project X (started March 10, completed Nov 20) and Project Y (started May 20, still ongoing)
Tests: Date range overlap detection

**Q5**: "Who has been at the company the longest?"
Answer: Alice (joined January 2020, currently 4+ years of tenure)
Tests: Duration calculation from past to present, MAX tenure

**Q6**: "Were there any employees who worked on projects before getting promoted?"
Answer: No. Alice was promoted (June 2022) before starting Project X (March 2023). Bob was promoted (Jan 2023) before Project Y (May 2023).
Tests: Event sequencing across different relationship types

**Q7**: "What was the average time between joining and first promotion?"
* **Answer**:

  * Alice: 2 years, 5 months (Jan 2020 → June 2022)

  * Bob: 1 year, 6 months (July 2021 → Jan 2023)

  * Average: ~2 years

* **Tests**: Multiple date calculations and aggregation

This demonstrates GraphRAG's ability to perform precise temporal reasoning, date arithmetic, and historical state queries that require structured time-based logic
