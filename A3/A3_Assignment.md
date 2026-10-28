# DATA3010 · Assignment 3
## Patterns and groups

**Out:** Wednesday, October 21 · **Due:** Wednesday, October 28, 11:59 PM (confirm the date in the LMS)
**Weight:** see the syllabus · **Marks:** 100 · **Work alone**

Submit one file: `A3_Starter.ipynb`, renamed `A3_LastName.ipynb`, with all outputs saved.

---

## The problem

The campus convenience store wants two answers before it rearranges its shelves and plans next month's app offers:

1. **What sells together?** You have 1,500 receipts from last month (Part A).
2. **Who shops here?** You have 400 app members, each described by how often they come, how much they spend, when they come and what they buy (Part B).

The data cells in the starter notebook are fixed. Do not change them, so everyone's numbers match.

**Two warnings that are part of the assignment.**
- A rule can have a high confidence and still mean nothing. Finding one is your job, in Task 3.
- Not every column that holds numbers is a quantity. One member column is a code. Finding it is your job, in Task 4.

---

## What to hand in

| Task | What you do | Marks |
|---|---|---|
| **Part A** | **What sells together?** | |
| 1 | Build the basket table and describe it | 10 |
| 2 | Find the frequent items and frequent pairs, using the Apriori idea | 20 |
| 3 | Write the rules with support, confidence and lift; find a misleading rule; check a rare pair | 20 |
| **Part B** | **Who shops here?** | |
| 4 | Choose the clustering features, justify what you leave out, and scale | 10 |
| 5 | Choose k with inertia and silhouette, profile and name the groups, and show what scaling changed | 20 |
| 6 | Try Ward and DBSCAN, compare them with k-means, and choose a method | 10 |
| 7 | Give the store two recommendations, and answer an AI-written analysis | 10 |
| | **Total** | **100** |

---

## How each task is marked

**Task 1 (10).** Basket table with one True/False row per receipt (5). Number of receipts, number of items, average basket size and the 5 most common items (5).

**Task 2 (20).** Support of every item and the list of frequent items (6). Candidate pairs built only from frequent items, with the number of candidates and the number of all possible pairs (6). Frequent pairs shown and sorted (8).

**Task 3 (20).** Both rules for every frequent pair, with support, confidence and lift, sorted by lift (8). A high-confidence, low-lift rule found and explained in plain words (6). The rare pair checked, and what the 5% cut-off hid (6).

**Task 4 (10).** Features chosen (3). Each excluded column justified; the code column must be argued as a code, not as "not useful" (4). Spreads shown and all members scaled (3).

**Task 5 (20).** Table of k, inertia and silhouette, and a choice of k that uses both columns (7). Profile in original units with sizes, and a name for each group backed by its numbers (9). Unscaled k-means compared, with the reason (4).

**Task 6 (10).** Ward compared with k-means in a crosstab (3). DBSCAN with at least four `eps` values, giving clusters and noise for each (4). A method chosen and defended with numbers (3).

**Task 7 (10).** Two concrete actions, one from each part, using at least three numbers (5). At least four real problems found in the AI paragraph, each with evidence and a fix (5).

**Marks are lost for:**
- judging a rule by confidence alone
- clustering on ID or code columns
- clustering unscaled features in different units
- choosing k by the lowest inertia
- describing clusters in scaled units
- splitting into train and test for clustering
- a notebook that does not run top to bottom

---

## Rules on AI use

You may use an AI tool to explain a concept, to debug an error, or to review code you wrote. You may not use it to produce your analysis or your written answers.

Say where you used it, in the statement at the end of the notebook. An honest line costs you nothing. A submission whose reasoning does not match its code will be treated as an academic integrity case.

The point of Task 7b is that AI-written analysis often runs, sounds confident, and is wrong. You are being marked on catching that.

---

## Practical notes

Everything you need was done in class:
- **Oct 7:** baskets, support, Apriori, confidence, lift
- **Oct 14:** k-means, scaling, inertia, silhouette, profiling and naming groups
- **Oct 16:** hierarchical clustering, DBSCAN, choosing eps

Some more tips:
- Use `RANDOM_STATE` everywhere a random seed is asked for, and `MIN_SUPPORT` for the support cut-off, so your numbers can be checked.
- Cluster numbers can come out in a different order from a classmate's. Describe groups by their numbers, not by their cluster label.
- If a result surprises you, say so and show the number. An unexplained result costs fewer marks than a confident wrong claim.
- Ask questions in office hours (Williston 110) before the last day.
