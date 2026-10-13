# DATA3010 · Exam 1 Cheat Sheet

> Built from your Lecture 1–6 notebooks (Preprocessing Lab, Lecture 2 apartments, Lecture 3 churn, C1 gym KNN, C2 trees/errors, C3 probabilities).
> **Swap the column names** (`"price"`, `"category"`, `"target"`…) for the ones in the exam. **Use the variable names the task gives you** and `random_state=RANDOM_STATE` everywhere.

---

## Contents
0. [Exam-day checklist](#0-exam-day-checklist)
1. [Setup cell (paste first)](#1-setup-cell)
2. **Part A · Preprocessing**
   - [2.1 Inspect the table](#21-inspect-the-table)
   - [2.2 Text → numbers](#22-text--numbers)
   - [2.3 Standardize category labels](#23-standardize-category-labels)
   - [2.4 Impossible values: flag, then blank](#24-impossible-values-flag-then-blank)
   - [2.5 Duplicates](#25-duplicates)
   - [2.6 Which columns can't be inputs (IDs, constants, leakage)](#26-which-columns-cannot-be-model-inputs)
   - [2.7 One-hot encoding](#27-one-hot-encoding)
3. **Part B · Classification**
   - [3.1 X, y and the baseline](#31-x-y-and-the-baseline)
   - [3.2 train_test_split with stratify](#32-train_test_split-with-stratify)
   - [3.3 Fill missing values AFTER the split](#33-fill-missing-values-after-the-split)
   - [3.4 KNN in a pipeline + choosing k with CV](#34-knn-in-a-pipeline--choosing-k-with-cv)
   - [3.5 Decision tree + choosing max_depth with CV](#35-decision-tree--choosing-max_depth-with-cv)
   - [3.6 Confusion matrix, accuracy, precision, recall, F1](#36-confusion-matrix-accuracy-precision-recall-f1)
   - [3.7 Logistic regression, predict_proba, threshold by cost](#37-logistic-regression-predict_proba-threshold-by-cost)
   - [3.8 Comparison table → recommend a model](#38-comparison-table--recommend-a-model)
   - [3.9 Rare class extras (Dummy, class_weight)](#39-rare-class-extras)
4. **Part C · Decisions + auditing AI code**
5. [The four "WHY" questions (ready answers)](#5-the-four-why-questions)
6. [Common errors and fixes](#6-common-errors-and-fixes)
7. [Written-answer templates](#7-written-answer-templates)
8. [Full end-to-end template](#8-full-end-to-end-template)

---

## 0. Exam-day checklist

- [ ] **AI features OFF** in VS Code / Colab before 3:05 (Copilot, Claude extension, etc.).
- [ ] Download notebook from Brightspace, open it, **run the setup cell first**.
- [ ] Time: 90 min for 100 pts ≈ **~0.9 min per point**. Part A 30 → ~25 min, Part B 55 → ~50 min, Part C 15 → ~12 min. Leave **5 min** at the end.
- [ ] Stuck? **Leave the code in + one comment line saying what you were trying to do** → partial credit. Move on.
- [ ] Written answers **must quote numbers from your own output** ("accuracy 0.88 vs baseline 0.62…").
- [ ] At the end: **Kernel → Restart and Run All** → check every cell ran → Save → upload `.ipynb` by **4:40**.

**The golden order:**
```
inspect → fix text/labels → flag + blank impossible values → drop IDs/constants/leakage
→ X, y → SPLIT (stratify) → baseline → fill missing using TRAIN only
→ model (pipeline) → tune with CV on TRAIN → test ONCE → confusion matrix / costs → recommend
```

---

## 1. Setup cell

```python
import numpy as np
import pandas as pd
from IPython.display import display

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 140)
RANDOM_STATE = 42   # the exam probably defines this already - don't redefine it differently
```

---

# PART A · PREPROCESSING (30 pts)

## 2.1 Inspect the table

```python
df.shape                       # (rows, columns)
df.head()                      # first 5 rows
df.dtypes                      # object / str = text, needs converting before math
df.info()                      # dtypes + non-null counts in one go
df.describe()                  # min / max -> spot impossible values (negative, 0, 999...)
df.nunique()                   # 1 unique value = constant column; = len(df) = probably an ID

# Missing values
df.isna().sum()                         # count per column
df.isna().mean().round(3)               # share per column
df[df["col"].isna()]                    # rows where col is blank

# Duplicates
df.duplicated().sum()                               # exact duplicate rows (every column identical)
df[df.duplicated(keep=False)]                       # show ALL copies
df[df.duplicated(subset="id_col", keep=False)]      # same ID appears more than once
df["id_col"].duplicated().sum()

# Categories / labels
df["category"].value_counts(dropna=False)           # dropna=False shows NaN too
df["category"].unique()

# Class balance of the target
df["target"].value_counts()
df["target"].value_counts(normalize=True).round(3)  # shares -> baseline = the biggest one
```

**Inspect everything in one cell:**
```python
print("Shape:", df.shape)
print("\nDtypes:\n", df.dtypes)
print("\nMissing:\n", df.isna().sum())
print("\nExact duplicate rows:", df.duplicated().sum())
print("Repeated IDs:", df["id_col"].duplicated().sum())
print("\nUnique values per column:\n", df.nunique())
print("\nClass balance:\n", df["target"].value_counts(normalize=True).round(3))
display(df.describe())
```

**Writing evidence:** give *column + ID + value*. e.g. "`price`, row P07, `"$0"` → impossible price."

---

## 2.2 Text → numbers

### Price like `"$1,299.00"` (also `"1,299"`, `" $85 "`)
```python
df["price"] = pd.to_numeric(
    df["price_text"]
      .str.replace("$", "", regex=False)
      .str.replace(",", "", regex=False)
      .str.strip()
)
```
If some values are junk like `"N/A"`, `"call"`, `"unknown"`, use `errors="coerce"` (turns them into NaN):
```python
df["price"] = pd.to_numeric(df["price_text"].str.replace("$", "", regex=False)
                                            .str.replace(",", "", regex=False)
                                            .str.strip(), errors="coerce")
```

### Other common text → number patterns
```python
# Percent "45%"  ->  45.0   (or /100 for 0.45)
df["rate"] = pd.to_numeric(df["rate_text"].str.replace("%", "", regex=False).str.strip())

# Units "12 km", "3.5 kg", "30 min"  ->  12.0
df["dist"] = pd.to_numeric(df["dist_text"].str.replace("km", "", regex=False).str.strip())
# any letters/units at all: keep only digits, dot and minus
df["dist"] = pd.to_numeric(df["dist_text"].str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")

# Words that mean a number ("Studio" = 0 bedrooms)
df["bedrooms"] = df["bedrooms"].str.strip().str.lower().replace({"studio": "0"}).astype(int)

# Yes / No style -> 1 / 0 (clean first, then map)
yes_no = {"yes": 1, "y": 1, "true": 1, "1": 1, "no": 0, "n": 0, "false": 0, "0": 0}
df["auto_pay"] = df["auto_pay"].astype(str).str.strip().str.lower().map(yes_no)

# Already a number but stored as text ("42")
df["age"] = pd.to_numeric(df["age"], errors="coerce")

# Dates
df["posted"] = pd.to_datetime(df["posted"])
df["days_ago"] = (pd.Timestamp("2026-10-09") - df["posted"]).dt.days
```

**Check it worked:**
```python
print(df["price"].dtype)                                  # should be float64 / int64
display(df[["price_text", "price"]].head(10))             # before vs after side by side
print("Blank after conversion:", df["price"].isna().sum())
```

---

## 2.3 Standardize category labels

`"Shoes"`, `" SHOES"`, `"shoes "` → all the same thing.

### Easy case (only case/space differences)
```python
df["category"] = df["category"].str.strip().str.lower()     # -> "shoes"
# or Title Case for nice labels:
df["category"] = df["category"].str.strip().str.title()     # -> "Shoes"
```

### Abbreviations / synonyms → use a map (strip + lower FIRST, so the map keys are all lowercase)
```python
cat_map = {
    "shoes": "Shoes", "shoe": "Shoes", "sneakers": "Shoes",
    "electronics": "Electronics", "elec": "Electronics",
    "jp": "Jamaica Plain", "jamaica plain": "Jamaica Plain",
}
df["category"] = df["category"].str.strip().str.lower().map(cat_map)
```

**Always check nothing fell through the map** (`.map` turns unknown labels into NaN!):
```python
print(df["category"].value_counts(dropna=False))
before = raw["category"]                                         # the original column
print("Unmapped labels:", before[df["category"].isna() & before.notna()].unique())
```
Not sure what labels exist? Run `df["category"].str.strip().str.lower().value_counts()` first, then write the map.

---

## 2.4 Impossible values: flag, then blank

Rule: **state the rule → make the flag → blank the value → check**.
Missing ≠ invalid. NaN is *unknown*; age 150 is *wrong*.

```python
# 1. Rule as a True/False mask. Notna() so existing blanks aren't called "invalid".
bad_age   = ~df["age"].between(0, 120) & df["age"].notna()   # outside 0..120
bad_price = df["price"] <= 0                                  # 0 or negative

# 2. Keep a flag column BEFORE blanking (so it can be traced later)
df["age_was_invalid"]   = bad_age
df["price_was_invalid"] = bad_price

# 3. Blank them
df.loc[bad_age,   "age"]   = np.nan
df.loc[bad_price, "price"] = np.nan

# 4. Check
print(df[["age_was_invalid", "price_was_invalid"]].sum())
print("No impossible ages left:", not ((df["age"] < 0) | (df["age"] > 120)).any())
```

**Missing-value flag** (do this before any filling, often right before the split):
```python
df["age_was_missing"] = df["age"].isna()      # True where blank (includes the ones you just blanked)
```

Show the rows that break any rule:
```python
invalid_any = bad_age | bad_price
display(df[invalid_any])
```

Typical rules: age 0–120, price/fee/rent > 0, size > 0, counts ≥ 0, percentages 0–100, dates not in the future.
**Don't** blank a value just because it's big (e.g. $9,500 rent): it's an *outlier*, not *impossible*. Mention it, keep it.

> Don't fill the **target** (`y`). Rows with a missing target are dropped from modelling: `df = df[df["target"].notna()]`.

---

## 2.5 Duplicates

```python
# Exact copies (every column identical) -> safe to remove
df = df.drop_duplicates().reset_index(drop=True)

# Same ID, different values (conflicting) -> need a RULE, e.g. keep the most recent
df = (df.sort_values("posted")
        .drop_duplicates(subset="id_col", keep="last")
        .reset_index(drop=True))
```
Why not just `keep="first"` blindly? It might keep the stale/wrong row. Use a reason (latest date, most complete row).

---

## 2.6 Which columns cannot be model inputs

| Type | How to spot it | Example | Why drop |
|---|---|---|---|
| **ID** | `nunique() == len(df)`, name ends in `_id` | `member_id`, `order_no` | Just a label; no meaning, model memorizes it |
| **Constant** | `nunique() == 1` | `city = "Boston"` everywhere | Tells the model nothing |
| **Leakage** | Only known **after** the outcome | `cancellation_reason`, `cancel_form_submitted`, `refund_amount`, `return_date`, `days_late` | Not available at prediction time → fake-perfect accuracy |
| **Duplicate info** | Same thing in other units | `size_m2` = `sqft` × 0.0929 | Redundant (high correlation alone isn't enough) |
| **Raw text replaced** | You made a clean version | `rent_text` → `rent` | Already converted |
| **The target** | — | `renewed`, `churned` | It's the answer! |

```python
# Find constant and ID-like columns automatically
print("Constant:", [c for c in df.columns if df[c].nunique(dropna=False) == 1])
print("ID-like :", [c for c in df.columns if df[c].nunique() == len(df)])

# Leakage check: does a column almost perfectly match the target?
display(df.groupby("target")["suspect_col"].mean())   # e.g. 0.95 vs 0.00 -> leakage!
display(pd.crosstab(df["suspect_col"], df["target"]))

drop_cols = ["id_col", "constant_col", "leaky_col", "target"]
X = df.drop(columns=drop_cols)
y = df["target"]
print(list(X.columns))
```

**Leakage test question to ask yourself:** *"At the moment we'd make the prediction, would we already know this value?"* If no → leakage → exclude.

---

## 2.7 One-hot encoding

Unordered categories (neighborhood, color, plan type) → one 0/1 column each.
Ordered numbers that really are numbers (bedrooms 0,1,2,3) can stay as one column.

### Simple way: `pd.get_dummies`
```python
# OPTION 1 - encode one or more columns inside the table (originals are replaced)
df = pd.get_dummies(df, columns=["category"], dtype=int)
```
```python
# OPTION 2 - just look at / build the dummies for one column (df unchanged)
dummies = pd.get_dummies(df["category"], prefix="category", dtype=int)
df_encoded = pd.concat([df.drop(columns=["category"]), dummies], axis=1)
```
`dtype=int` gives 0/1 instead of True/False.

### Train-only way: `OneHotEncoder` (use when encoding after the split)
```python
enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)   # older sklearn: sparse=False
enc.fit(X_train[["category"]])                                      # learn categories from TRAIN only

train_dummies = pd.DataFrame(enc.transform(X_train[["category"]]),
                             columns=enc.get_feature_names_out(), index=X_train.index)
test_dummies  = pd.DataFrame(enc.transform(X_test[["category"]]),
                             columns=enc.get_feature_names_out(), index=X_test.index)

X_train = pd.concat([X_train.drop(columns=["category"]), train_dummies], axis=1)
X_test  = pd.concat([X_test.drop(columns=["category"]),  test_dummies],  axis=1)
```
`handle_unknown="ignore"` → a category never seen in training becomes all zeros instead of crashing.
`get_dummies` on the test rows alone can produce **different columns** than training; the encoder can't.

> On the exam, `get_dummies` on the whole table **before** splitting is usually acceptable for one-hot (it's a fixed rule, not a learned number), unless the task says to fit an encoder on training rows. Fitting means/medians/scalers before the split is NOT OK.

**Why not codes 1, 2, 3?** It invents an order and distances (Fenway = 2 × Allston?). Models like KNN/logistic regression would treat 3 as "more" than 1.

---

# PART B · CLASSIFICATION (55 pts)

## 3.1 X, y and the baseline

```python
feature_cols = ["f1", "f2", "f3"]         # NOT the ID, NOT the target, NOT leaky columns
X = df[feature_cols]
y = df["target"]

class_counts = y.value_counts()
baseline_accuracy = class_counts.max() / len(y)     # = always guessing the most common class
print(class_counts)
print("Baseline:", round(baseline_accuracy, 3))
```
**Baseline** = accuracy of always predicting the majority class. A model that can't beat it learned nothing.

---

## 3.2 train_test_split with stratify

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)

print("Train:", X_train.shape, "| Test:", X_test.shape)
test_baseline = y_test.value_counts().max() / len(y_test)
print("Test baseline:", round(test_baseline, 3))
print("Positive share train/test:", round(y_train.mean(), 3), round(y_test.mean(), 3))  # ~equal
```
- `stratify=y` keeps the same class share in train and test (so the baseline is comparable, and a rare class definitely appears in the test set).
- Use whatever `test_size` the exam says (0.25, 0.3, 0.2…).
- Order of outputs is always **X_train, X_test, y_train, y_test**.

---

## 3.3 Fill missing values AFTER the split

Medians are **learned** numbers → learn from **training rows only**, then apply to both.

### Numeric: fill with the training median
```python
num_cols = ["age", "income"]

# Flag first (if the task wants flags and you haven't made them yet)
for c in num_cols:
    X_train[c + "_was_missing"] = X_train[c].isna()
    X_test[c + "_was_missing"]  = X_test[c].isna()

train_medians = X_train[num_cols].median()          # learned from TRAIN only
X_train[num_cols] = X_train[num_cols].fillna(train_medians)
X_test[num_cols]  = X_test[num_cols].fillna(train_medians)   # same TRAIN numbers on test

print(X_train.isna().sum().sum(), X_test.isna().sum().sum())  # both should be 0
```
> If you get a `SettingWithCopyWarning`, add `.copy()` right after the split:
> `X_train, X_test = X_train.copy(), X_test.copy()`

### Categorical: fill with the training mode
```python
train_mode = X_train["category"].mode()[0]
X_train["category"] = X_train["category"].fillna(train_mode)
X_test["category"]  = X_test["category"].fillna(train_mode)
```

### Group median (e.g. sqft by bedroom count), training rows only
```python
train_medians = X_train.groupby("bedrooms")["sqft"].median()
X_train["sqft"] = X_train["sqft"].fillna(X_train["bedrooms"].map(train_medians))
X_test["sqft"]  = X_test["sqft"].fillna(X_test["bedrooms"].map(train_medians))
```

### sklearn way: `SimpleImputer`
```python
imp = SimpleImputer(strategy="median")              # or "mean", "most_frequent"
X_train[num_cols] = imp.fit_transform(X_train[num_cols])   # fit on TRAIN
X_test[num_cols]  = imp.transform(X_test[num_cols])        # only transform TEST
print(imp.statistics_)                                     # the medians it learned
```
Or put it inside the pipeline (then it re-learns properly inside every CV fold too):
```python
model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                      KNeighborsClassifier(n_neighbors=5)).fit(X_train, y_train)
```

**Fit vs transform:** `fit_transform` on **train**, `transform` on **test**. Never fit on test.

---

## 3.4 KNN in a pipeline + choosing k with CV

```python
knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
knn.fit(X_train, y_train)
print("Train acc:", round(knn.score(X_train, y_train), 3))
print("Test acc :", round(knn.score(X_test, y_test), 3))
```
**Why scale for KNN?** KNN uses distance; a feature with big numbers (minutes 15–120) drowns out a small one (visits 0–25). `StandardScaler` → each feature mean 0, std 1. The pipeline fits the scaler on training rows only and reuses those numbers on the test rows.

### k table with cross-validation (train rows only)
```python
k_values = [1, 3, 5, 7, 9, 11, 15, 21]
rows = []
for k in k_values:
    pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    cv = cross_val_score(pipe, X_train, y_train, cv=5).mean()     # TRAIN only
    pipe.fit(X_train, y_train)
    rows.append({"k": k,
                 "train_accuracy": round(pipe.score(X_train, y_train), 3),
                 "cv_accuracy": round(cv, 3),
                 "test_accuracy": round(pipe.score(X_test, y_test), 3)})
k_table = pd.DataFrame(rows)
display(k_table)

best_k = int(k_table.loc[k_table["cv_accuracy"].idxmax(), "k"])   # by CV, NOT test
print("Best k:", best_k)

best_knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=best_k)).fit(X_train, y_train)
print("Final test accuracy:", round(best_knn.score(X_test, y_test), 3))
```
- **k = 1** → train accuracy 1.00 (memorizes) → **overfitting**, jagged boundary.
- **Large k** → too smooth → **underfitting**.
- Big gap train ≫ test/CV = overfitting.

---

## 3.5 Decision tree + choosing max_depth with CV

```python
tree = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE).fit(X_train, y_train)
print(export_text(tree, feature_names=list(X_train.columns)))   # read the rules out loud
print("Leaves:", tree.get_n_leaves(), "| depth:", tree.get_depth())
```
Trees **don't need scaling** (one feature at a time, "is visits ≤ 6.5?").

### depth table with cross-validation
```python
depths = [1, 2, 3, 4, 5, 7, None]       # None = unlimited
rows = []
for d in depths:
    t = DecisionTreeClassifier(max_depth=d, random_state=RANDOM_STATE)
    cv = cross_val_score(t, X_train, y_train, cv=5).mean()       # TRAIN only, fresh tree
    t.fit(X_train, y_train)
    rows.append({"max_depth": str(d),
                 "train_accuracy": round(t.score(X_train, y_train), 3),
                 "cv_accuracy": round(cv, 3),
                 "test_accuracy": round(t.score(X_test, y_test), 3),
                 "leaves": t.get_n_leaves()})
depth_table = pd.DataFrame(rows)
display(depth_table)

best_depth = depths[depth_table["cv_accuracy"].idxmax()]   # use the LIST so None stays None
print("Best depth:", best_depth)

best_tree = DecisionTreeClassifier(max_depth=best_depth, random_state=RANDOM_STATE).fit(X_train, y_train)
print("Final test accuracy:", round(best_tree.score(X_test, y_test), 3))
```
> Gotcha: putting `None` in a pandas column turns it into `NaN`. That's why `best_depth` is taken from the `depths` **list** with the index. (If the task wants a numeric column, drop the `str(...)`.)

- Unlimited depth → train 1.00 → memorized → overfit.
- If two depths tie (or nearly tie) on CV, pick the **simpler (shallower)** one: easier to explain, less overfit.

---

## 3.6 Confusion matrix, accuracy, precision, recall, F1

```python
pred = best_tree.predict(X_test)

cm = confusion_matrix(y_test, pred)
print(cm)                                   # layout: [[TN, FP],
                                            #          [FN, TP]]
tn, fp, fn, tp = cm.ravel()
print(f"TN={tn} FP={fp} FN={fn} TP={tp}")

print("Accuracy :", round(accuracy_score(y_test, pred), 3))
print("Precision:", round(precision_score(y_test, pred, zero_division=0), 3))
print("Recall   :", round(recall_score(y_test, pred), 3))
print("F1       :", round(f1_score(y_test, pred), 3))
```
Nicer labelled matrix:
```python
display(pd.DataFrame(cm, index=["actual 0", "actual 1"], columns=["pred 0", "pred 1"]))
```

### Scores table for several models
```python
preds = {"tree": best_tree.predict(X_test), "knn": best_knn.predict(X_test)}
scores = pd.DataFrame({
    name: {"accuracy":  accuracy_score(y_test, p),
           "precision": precision_score(y_test, p, zero_division=0),
           "recall":    recall_score(y_test, p),
           "f1":        f1_score(y_test, p)}
    for name, p in preds.items()}).T.round(3)
display(scores)
```

### What they mean (class 1 = the "positive" thing, e.g. renewed / fraud / refund)

| | Formula | Plain question |
|---|---|---|
| **Accuracy** | (TP+TN) / all | How often are we right overall? (misleading with rare classes) |
| **Precision** | TP / (TP+FP) | When we say "yes", how often are we right? |
| **Recall** | TP / (TP+FN) | Of the real "yes" cases, how many did we catch? |
| **F1** | 2·P·R / (P+R) | Balance of precision and recall |

| Mistake | Meaning | Hurts when… |
|---|---|---|
| **FP** (false alarm) | Predicted 1, actually 0 | Acting is expensive: spam filter hides a real email, wrongly accusing a customer of fraud, unnecessary surgery |
| **FN** (missed case) | Predicted 0, actually 1 | Missing is expensive: missed fraud / disease / refund request ($40 vs a $2 call) |

**Always translate into the story:** "FP = we called a member who was never going to ask for a refund ($2 of staff time). FN = we missed a refund request ($40 + unhappy member)."

**Precision vs recall tradeoff:** lower threshold → flag more → recall ↑, precision ↓.

---

## 3.7 Logistic regression, predict_proba, threshold by cost

```python
logreg = make_pipeline(StandardScaler(),
                       LogisticRegression(random_state=RANDOM_STATE)).fit(X_train, y_train)

proba = logreg.predict_proba(X_test)[:, 1]      # column 1 = probability of class 1
print("min / max probability:", proba.min().round(3), proba.max().round(3))

# What the model learned (positive weight -> pushes toward class 1)
weights = pd.Series(logreg.named_steps["logisticregression"].coef_[0],
                    index=X_train.columns).round(2)
display(weights.sort_values())
```
`predict()` = cut at 0.50. `predict_proba()` = the risk score; **you** choose the cut.

### Threshold table with costs
```python
COST_FN = 40     # cost of a missed positive  (use the exam's numbers / names)
COST_FP = 2      # cost of a false alarm

thresholds = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
rows = []
for t in thresholds:
    pred_t = (proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred_t, labels=[0, 1]).ravel()
    rows.append({"threshold": t,
                 "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                 "precision": round(precision_score(y_test, pred_t, zero_division=0), 3),
                 "recall": round(recall_score(y_test, pred_t, zero_division=0), 3),
                 "accuracy": round(accuracy_score(y_test, pred_t), 3),
                 "cost": fn * COST_FN + fp * COST_FP})
threshold_table = pd.DataFrame(rows)
display(threshold_table)

best_threshold = threshold_table.loc[threshold_table["cost"].idxmin(), "threshold"]
print("Cheapest threshold:", best_threshold)
```
⚠️ **Read the cost wording carefully!** Variants:
- "Each **call/flag/action** costs $2" (paid for TP **and** FP) → `cost = fn*COST_MISSED + (tp+fp)*COST_CALL`  ← this is how the gym refund notebook did it
- "Each **false alarm** costs $X" → `cost = fn*COST_FN + fp*COST_FP`
- A TP can have a benefit/saving → subtract it: `cost = fn*A + fp*B - tp*SAVING`

> Strictly, picking the threshold should also be done on training rows (CV) rather than the test rows. If the exam's task says to build the table on the test set, do what it says, but you can mention this in a written answer.

Train-side version (choose threshold without touching the test set):
```python
from sklearn.model_selection import cross_val_predict
proba_train_cv = cross_val_predict(logreg, X_train, y_train, cv=5, method="predict_proba")[:, 1]
# ...then run the same threshold loop with y_train and proba_train_cv
```

**Why 0.5 isn't automatically right:** 0.5 assumes both mistakes cost the same and classes are balanced. If a miss costs $40 and a false alarm $2, flagging someone with only 10% risk is still worth it (0.10 × $40 = $4 > $2). Break-even rule of thumb: flag when `p ≥ COST_FP / (COST_FP + COST_FN)` = 2/42 ≈ 0.05.

---

## 3.8 Comparison table → recommend a model

```python
COST_FN, COST_FP = 40, 2
models = [
    ("knn",                 best_knn,  None),
    ("tree",                best_tree, None),
    ("logistic regression", logreg,    best_threshold),   # uses chosen threshold
]
rows = []
for name, model, thr in models:
    if thr is None:
        p = model.predict(X_test)
    else:
        p = (model.predict_proba(X_test)[:, 1] >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, p, labels=[0, 1]).ravel()
    rows.append({"model": name,
                 "accuracy": round(accuracy_score(y_test, p), 3),
                 "precision": round(precision_score(y_test, p, zero_division=0), 3),
                 "recall": round(recall_score(y_test, p, zero_division=0), 3),
                 "f1": round(f1_score(y_test, p, zero_division=0), 3),
                 "FP": fp, "FN": fn,
                 "cost": fn * COST_FN + fp * COST_FP})
comparison = pd.DataFrame(rows).set_index("model")
display(comparison)
print("Cheapest:", comparison["cost"].idxmin(), "| Most accurate:", comparison["accuracy"].idxmax())
```

### Fair comparison on TRAIN rows with CV (any metric)
```python
cv_scores = {name: round(cross_val_score(model, X_train, y_train, cv=5, scoring="recall").mean(), 3)
             for name, model, _ in models}
cv_scores = pd.Series(cv_scores)
display(cv_scores)
```
`scoring=` options: `"accuracy"` (default), `"recall"`, `"precision"`, `"f1"`, `"roc_auc"`.

**Recommending:** pick by the **business goal** (usually lowest cost, or the metric the task says), not by accuracy alone. Use ≥ 2 numbers. Name a risk/what would change your mind.

---

## 3.9 Rare class extras

```python
# "Always predict the majority class" model = the baseline as a model
dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
print("Dummy accuracy:", dummy.score(X_test, y_test))   # high accuracy, recall 0!

# Make a tree take the rare class seriously
tree_bal = DecisionTreeClassifier(max_depth=3, class_weight="balanced",
                                  random_state=RANDOM_STATE).fit(X_train, y_train)
# also works: LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE)
```
92% accuracy with 8% positives can mean **recall = 0**: the model catches nobody.

---

# PART C · DECISIONS + AUDITING AI CODE (15 pts)

## 4.1 Checklist for auditing AI-written analysis/code

Go through every line and ask: **what did it learn, and from which rows?**

| Look for | Problem | Fix |
|---|---|---|
| `StandardScaler().fit(X)` / `fit_transform(X)` **before** `train_test_split` | Test rows leak into the means/stds | Split first; put the scaler in a `make_pipeline` |
| `fillna(df.median())` / `SimpleImputer` on all rows before split | Test rows leak into the fill values | Split first, compute medians on `X_train` |
| A column only known after the outcome (`cancel_form`, `refund_amount`, `cancellation_reason`) | **Target leakage** → suspiciously perfect score (0.99/1.00) | Drop it from X |
| ID column left in X | Meaningless feature | Drop it |
| No `stratify=y` | Rare class may be under/over-represented in test | `stratify=y` |
| No `random_state` | Not reproducible | `random_state=RANDOM_STATE` |
| Choosing k / depth / model / threshold by **test** accuracy | Test set got "fitted"; final number is too optimistic | Use `cross_val_score` on train |
| Only reports accuracy | Hides FN/FP; useless with imbalance | Confusion matrix, precision, recall, cost |
| Ignores baseline | 0.92 might = baseline | Compare to majority-class share |
| `df.dropna()` on everything | Throws away lots of rows, may bias data | Flag + fill (train-only) |
| `drop_duplicates(subset="id", keep="first")` blindly | May keep stale/wrong row | Exact dupes first; rule for conflicting IDs |
| KNN without scaling | Big-unit feature dominates distance | `make_pipeline(StandardScaler(), KNN)` |
| `pd.get_dummies` separately on train and test | Columns may not match | Encode once, or fit `OneHotEncoder` on train |
| Default threshold 0.5 "because it's standard" | Ignores costs | Threshold table with costs |
| Claims numbers that don't match your output | Made-up / wrong | Quote your own numbers |
| Labels categories as 1,2,3 | Invents an order | One-hot |

**Red flag:** accuracy of 1.00 or close → **check for leakage first**.

Template answer: *"The code runs, but it's not trustworthy: (1) [problem] — [why it matters] — [fix]. (2)… The honest accuracy after fixing it is [my number], vs the AI's [their number], and the baseline is [x]."*

---

## 5. The four "WHY" questions

**Why split before filling missing values?**
Filling with a median/mean *learns* a number from the data. If you compute it on all rows, the test rows influence the training data, so information from the test set leaks into the model and the test score is no longer an honest estimate of performance on new, unseen data. Compute the median from training rows only and apply that same number to the test rows (just like a real future customer would get).

**Why choose a setting (k, max_depth, threshold) with cross-validation instead of the test set?**
If you try many settings and pick the one that scores best on the test set, you've tuned the model to those specific test rows, so the test score becomes optimistically biased; it's no longer "unseen". Cross-validation picks the setting using only the training rows (train on 4 folds, score on the 5th, 5 times, average), which is also more stable than one split. The test set is used **once**, at the end, for an honest final number.

**Why can a model with high accuracy be the worst choice?**
With an imbalanced target (e.g. 8% refunds), predicting "no" for everyone gives 92% accuracy but catches **zero** positives (recall 0). Accuracy treats all mistakes as equal, but in practice FN and FP have very different costs ($40 missed refund vs $2 call). The higher-accuracy model can have the highest total cost. Compare against the baseline and look at recall/precision/cost.

**Why is 0.5 not automatically the right threshold?**
0.5 only makes sense if both errors cost the same. The threshold is a business decision: when a missed positive costs much more than a false alarm, a lower threshold (e.g. 0.05–0.15) flags more people, increases recall, and lowers total cost. Also, with a rare class, few people ever get a probability above 0.5, so the default cut catches almost nobody. Choose the threshold that minimizes cost.

### Other likely short questions
- **Overfitting:** train ≫ test/CV (e.g. k=1 or unlimited tree with train 1.00). Fix: larger k / smaller max_depth.
- **Underfitting:** both train and test low. Fix: smaller k / deeper tree.
- **Why scale for KNN but not trees?** KNN adds up distances across features (units compete). A tree asks one feature at a time.
- **Why stratify?** Keeps class proportions the same in train and test → comparable baseline, rare class present in test.
- **Why keep a missing/invalid flag?** The filled value is a guess, not a measurement; the flag lets a reader (and the model) know which values were made up, and "was missing" can itself carry information.
- **Why not fill with 0?** 0 is a real value ("free", "age 0"); it invents meaning and distorts the distribution.
- **Why one-hot instead of 1, 2, 3?** Numbers invent an order and distances between unordered categories.
- **Why is an ID a bad feature?** Unique per row, no real meaning; the model can only memorize it; won't generalize to new IDs.
- **Missing vs invalid:** NaN = unknown. Age 150 = known to be wrong → blank it + flag it.
- **Classification vs regression:** label (renew yes/no) vs number (rent).
- **Train 0.64 vs baseline 0.62?** Barely better than guessing; could be luck → not useful.
- **Unseen category at prediction time?** `OneHotEncoder(handle_unknown="ignore")` → all zeros. Don't refit on new data.

---

## 6. Common errors and fixes

| Error / symptom | Cause | Fix |
|---|---|---|
| `ValueError: could not convert string to float: 'Shoes'` | Text column in X | One-hot it, or drop it |
| `ValueError: Input X contains NaN` | Missing values reached KNN/LogReg | Fill (train medians) or add `SimpleImputer` to the pipeline |
| `ValueError: Input y contains NaN` | Missing target | `df = df[df["target"].notna()]` before X/y |
| `ValueError: The least populated class in y has only 1 member` | stratify with a tiny class | Check `y.value_counts()`; maybe y isn't cleaned (e.g. "Yes"/"yes") |
| Column is all NaN after `.map(...)` | Map keys didn't match (case/spaces) | `.str.strip().str.lower()` **before** `.map`, lowercase keys |
| Column becomes `None`/empty after `fillna(..., inplace=True)` | `inplace=True` returns `None` | Never combine `=` with `inplace=True`: `df["c"] = df["c"].fillna(v)` |
| `fillna(inplace=True)` on `df["c"]` does nothing | pandas 3 copy-on-write | Same fix: assign back |
| `SettingWithCopyWarning` | Editing a slice | `.copy()` after slicing/splitting; use `df.loc[mask, "col"] = ...` |
| `.str` accessor error: `Can only use .str accessor with string values` | Column is already numeric (or mixed) | Use `.astype(str)` first, or skip the str cleaning |
| `TypeError: __init__() got an unexpected keyword argument 'sparse_output'` | Old sklearn | Use `sparse=False` |
| `KeyError: 'col'` | Typo, or column already dropped/encoded | `print(df.columns.tolist())` |
| `best_depth` is `nan` | `None` in a DataFrame became NaN | `best_depth = depths[table["cv_accuracy"].idxmax()]` |
| Precision warning "ill-defined" | Model never predicted 1 | `precision_score(..., zero_division=0)` |
| `confusion_matrix(...).ravel()` gives only 1 value | Only one class in y/pred | `confusion_matrix(y_test, pred, labels=[0, 1])` |
| Accuracy 1.00 | Leakage or target in X | Check `X.columns`; check `groupby(target)` of suspects |
| Target is "Yes"/"No" text | Metrics need 0/1 for precision/recall | `y = df["target"].str.strip().str.lower().map({"yes": 1, "no": 0})` |
| Models fit on whole df | `fit(X, y)` instead of `fit(X_train, y_train)` | Always fit on train only |

**Quick debug cell:**
```python
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print(X_train.dtypes)                   # all numeric?
print(X_train.isna().sum().sum())       # 0?
print(y_train.value_counts())           # 0/1 with both classes?
```

---

## 7. Written-answer templates

Fill in the brackets with **your own numbers**.

**Baseline / is the model good?**
> The baseline (always predicting [majority class]) is [0.62]. The model's test accuracy is [0.88], [26] points above it, so it learned something real. (If it's only [0.64]: barely above baseline, could be luck, not useful.)

**Train vs test gap**
> Train accuracy [1.00] vs test [0.80]: the model memorized the training rows (overfitting). A larger k / smaller max_depth generalizes better: CV [0.87].

**Choosing k / depth**
> I picked [max_depth = 3] because it had the highest 5-fold cross-validation accuracy on the training rows ([0.89]), not by looking at the test set. It's also simple (only [8] leaves), which makes it easy to explain. Its test accuracy is [0.90] vs baseline [0.62].

**Confusion matrix interpretation**
> Of [50] test rows, the model got [TP] true positives and [TN] true negatives. It made [FP] false positives ([what that means: called a member who wasn't going to ask, $2 each]) and [FN] false negatives ([missed a refund request, $40 each]). Precision [0.xx] means when it says "yes" it's right [xx]% of the time; recall [0.xx] means it catches [xx]% of the real positives.

**Threshold choice**
> At the default 0.50 the model catches only [1] of [16] positives (cost $[x]). At threshold [0.10] it catches [13] of [16] with [y] false alarms, total cost $[z], the lowest in the table. Because a miss ($40) costs 20× more than a false alarm ($2), a low threshold is worth it. Not in the cost column: [customers annoyed by calls / staff capacity / costs might change].

**Recommend a model**
> I recommend [model] (with threshold [t]). It has the lowest cost ($[x] vs $[y] for [other]) and the highest recall ([0.94]), meaning it catches [15] of [16] [positives]. [Other model] has higher accuracy ([0.93]) but misses [12] positives, which costs much more. One concern: [small test set / only [16] positives so one row changes recall by 6 points / costs are estimates / need more data]. I'd change my mind if [false alarms became expensive / costs changed].

**AI claim "Model A has higher accuracy so it's better"**
> Accuracy alone is misleading here: [positive class] is only [8]% of the rows, so predicting "no" for everyone already scores [0.92]. Model A's recall is [0.00]: it catches none of the [16] cases. Using the costs, Model A costs $[640] and Model B costs $[150]. So Model B is the better choice despite lower accuracy.

**AI claim "use the default 0.5 threshold"**
> 0.5 assumes both mistakes cost the same. Here a miss costs $[40] and a call $[2]. In my table 0.5 catches [1]/[16] and costs $[x], while [0.10] costs $[y]. The threshold should be chosen from the costs.

**AI claim "try several models on the test set and pick the best accuracy"**
> That uses the test set to make a choice, so its score is no longer an honest estimate on unseen data. Compare with cross-validation on the training rows (e.g. CV recall: tree [0.81], knn [0.10]) and use the test set once at the end. And choose by cost/recall, not accuracy, because of the class imbalance.

---

## 8. Full end-to-end template

Copy, then edit the CAPS parts. Runs top to bottom on a typical exam table.

```python
# ======================= PART A: PREPROCESSING =======================
df = RAW_TABLE.copy()

# --- inspect ---
print(df.shape); print(df.dtypes); print(df.isna().sum())
print("dup rows:", df.duplicated().sum(), "| repeated IDs:", df["ID_COL"].duplicated().sum())
print(df["TARGET"].value_counts(normalize=True))

# --- duplicates ---
df = df.drop_duplicates().reset_index(drop=True)

# --- text -> numbers ---
df["PRICE"] = pd.to_numeric(df["PRICE"].astype(str).str.replace("$", "", regex=False)
                                       .str.replace(",", "", regex=False).str.strip(),
                            errors="coerce")

# --- standardize labels ---
df["CAT"] = df["CAT"].str.strip().str.title()
print(df["CAT"].value_counts(dropna=False))

# --- impossible values: flag then blank ---
bad = df["PRICE"] <= 0
df["PRICE_was_invalid"] = bad
df.loc[bad, "PRICE"] = np.nan

# --- missing flag (before filling) ---
df["PRICE_was_missing"] = df["PRICE"].isna()

# --- one-hot ---
df = pd.get_dummies(df, columns=["CAT"], dtype=int)

# --- drop non-inputs ---
drop_cols = ["ID_COL", "CONSTANT_COL", "LEAKY_COL", "TARGET"]
X = df.drop(columns=drop_cols)
y = df["TARGET"]
print("Features:", list(X.columns))

# ======================= PART B: CLASSIFICATION =======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)
X_train, X_test = X_train.copy(), X_test.copy()

baseline = y_test.value_counts().max() / len(y_test)
print("Baseline:", round(baseline, 3))

# --- fill missing with TRAIN medians ---
num_cols = X_train.columns[X_train.isna().any()].tolist()
train_medians = X_train[num_cols].median()
X_train[num_cols] = X_train[num_cols].fillna(train_medians)
X_test[num_cols]  = X_test[num_cols].fillna(train_medians)

# --- KNN: pick k by CV ---
k_rows = []
for k in [1, 3, 5, 7, 9, 11, 15, 21]:
    pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    k_rows.append({"k": k, "cv_accuracy": cross_val_score(pipe, X_train, y_train, cv=5).mean()})
k_table = pd.DataFrame(k_rows); display(k_table.round(3))
best_k = int(k_table.loc[k_table["cv_accuracy"].idxmax(), "k"])
knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=best_k)).fit(X_train, y_train)

# --- Tree: pick depth by CV ---
depths = [1, 2, 3, 4, 5, None]
d_cv = [cross_val_score(DecisionTreeClassifier(max_depth=d, random_state=RANDOM_STATE),
                        X_train, y_train, cv=5).mean() for d in depths]
depth_table = pd.DataFrame({"max_depth": [str(d) for d in depths], "cv_accuracy": d_cv})
display(depth_table.round(3))
best_depth = depths[int(np.argmax(d_cv))]
tree = DecisionTreeClassifier(max_depth=best_depth, random_state=RANDOM_STATE).fit(X_train, y_train)
print(export_text(tree, feature_names=list(X_train.columns)))

# --- Logistic regression + cost threshold ---
logreg = make_pipeline(StandardScaler(), LogisticRegression(random_state=RANDOM_STATE)).fit(X_train, y_train)
proba = logreg.predict_proba(X_test)[:, 1]
COST_FN, COST_FP = 40, 2
t_rows = []
for t in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
    p = (proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, p, labels=[0, 1]).ravel()
    t_rows.append({"threshold": t, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                   "precision": precision_score(y_test, p, zero_division=0),
                   "recall": recall_score(y_test, p, zero_division=0),
                   "cost": fn * COST_FN + fp * COST_FP})
threshold_table = pd.DataFrame(t_rows); display(threshold_table.round(3))
best_threshold = threshold_table.loc[threshold_table["cost"].idxmin(), "threshold"]

# --- Compare all ---
c_rows = []
for name, model, thr in [("knn", knn, None), ("tree", tree, None), ("logreg", logreg, best_threshold)]:
    p = model.predict(X_test) if thr is None else (model.predict_proba(X_test)[:, 1] >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, p, labels=[0, 1]).ravel()
    c_rows.append({"model": name, "accuracy": accuracy_score(y_test, p),
                   "precision": precision_score(y_test, p, zero_division=0),
                   "recall": recall_score(y_test, p, zero_division=0),
                   "f1": f1_score(y_test, p, zero_division=0),
                   "TN": tn, "FP": fp, "FN": fn, "TP": tp,
                   "cost": fn * COST_FN + fp * COST_FP})
comparison = pd.DataFrame(c_rows).set_index("model").round(3)
display(comparison)
```

---

### Tiny reference: things you'll type a lot

```python
df.loc[mask, "col"] = value            # set values where mask is True
df[df["col"] > 5]                      # filter rows
df.drop(columns=["a", "b"])            # drop columns
df["col"].round(2)                     # round
df.groupby("g")["x"].mean()            # average per group
df.groupby("g")["x"].transform("median")   # group value on every row (same length as df)
s.idxmax() / s.idxmin()                # index label of biggest / smallest
df.sort_values("col", ascending=False)
pd.crosstab(df["a"], df["b"])          # count table
X_train.mean()                         # check scaler numbers: pipe.steps[0][1].mean_
```

**Good luck! Breathe. Read each task twice, use the given variable names, and write *why* with your numbers.**
