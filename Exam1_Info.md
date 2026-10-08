# DATA3010 · Exam 1 information

## When and where

**Friday, October 9, 2026**, in our usual room, ANXCN-101, during class time.

| Time | What happens |
| --- | --- |
| 3:00 | Arrive, open your laptop, download the exam notebook from Brightspace |
| 3:05 | Start |
| 4:35 | Stop writing |
| 4:40 | Last moment to upload to Brightspace |

## Format

- One Jupyter notebook, like our lab notebooks. You write the code yourself and answer short questions in plain language.
- The data is created inside the notebook. No files to download, no internet needed for the data.
- **100 points**, worth **15%** of your course grade.

| Part | Topic | Points |
| --- | --- | --- |
| A | Preprocessing | 30 |
| B | Classification | 55 |
| C | Decisions and an AI-written analysis to check | 15 |

## What it covers: weeks 1–4

You should be able to do each of these on your own:

**Preprocessing**

- Inspect a table: data types, missing values, duplicate rows, class balance
- Turn text into numbers (for example a price written as `"$1,299.00"`)
- Standardize category labels that mean the same thing (`"Shoes"`, `" SHOES"`, `"shoes"`)
- Find impossible values, blank them, and keep a missing-value flag
- Decide which columns cannot be model inputs: IDs, constants, and anything only known **after** the outcome (leakage)
- One-hot encode a category column

**Classification**

- Split with `train_test_split` using `stratify`, and report the baseline
- Fill missing values **after** the split, using the training rows only
- KNN inside a pipeline with `StandardScaler`
- Decision trees: choose `max_depth` with cross-validation on the training rows, not with the test set
- Confusion matrix, accuracy, precision, recall, and what each kind of mistake means in practice
- Logistic regression probabilities with `predict_proba`, and choosing a threshold using costs
- Recommend a model and defend it with numbers

**Not on Exam 1:** frequent itemset mining (this week's topic). It will be on Exam 2.

## Allowed

- Your own laptop
- Your own course notebooks and notes, and the course materials in Brightspace
- The official pandas and scikit-learn documentation

## Not allowed

- Any AI tool: ChatGPT, Claude, Gemini, Copilot, or AI features built into your editor or into Colab. **Turn them off before 3:05.**
- Messaging apps, email, or help from anyone in or outside the room
- Searching the web for solutions

Using any of these is an academic integrity violation.

## Do this before Friday (it takes two minutes)

Open a notebook on **the laptop you will bring** and run this cell:

```python
import numpy, pandas, sklearn
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
print("pandas", pandas.__version__, "| scikit-learn", sklearn.__version__)
print("Ready for Exam 1")
```

If it prints `Ready for Exam 1`, you are set. If it fails, email me (<othmans1@wit.edu>) by **Thursday evening** so we can fix it before the exam. Exam time will not be extended for setup problems.

Bring your charger.

## How to submit

1. **Kernel → Restart and Run All**, and check every cell ran.
2. Save the notebook.
3. Upload the `.ipynb` file to the Exam 1 folder in Brightspace by **4:40 PM**.

If you use Colab, download the notebook as `.ipynb` first (File → Download → .ipynb), then upload that file.

## How it is graded

- Each task is graded on the method, not only the final number. If something fails, leave your code in and write one line saying what you were trying to do: that earns partial credit.
- Written answers must use numbers from **your own** results.
- Use the variable names given in each task, and `random_state=RANDOM_STATE` wherever a random seed is used.

## How to prepare

1. Redo the exercises in the Classification I, II and III notebooks **without** opening your old answers.
2. Go back over Assignment 2, especially the leakage question and the threshold table.
3. For each of these, be able to say *why*, not only *how*:
   - Why do we split before filling missing values?
   - Why do we choose a setting with cross-validation instead of the test set?
   - Why can a model with high accuracy be the worst choice?
   - Why is 0.5 not automatically the right threshold?

## Accommodations

If you have testing accommodations, email me by **Thursday, October 8** so we can arrange them.

Questions: <othmans1@wit.edu>
