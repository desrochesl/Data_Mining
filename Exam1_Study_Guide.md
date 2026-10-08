# DATA3010 Exam 1 Study Guide

This study guide covers all the required topics for Exam 1, including code examples and answers to conceptual questions.

## Part 1: Preprocessing

### 1. Inspecting a Table

```python
import pandas as pd

# Load data
df = pd.read_csv('data.csv')

# Data types and non-null counts
df.info()

# Count missing values per column
df.isna().sum()

# Check for duplicate rows
df.duplicated().sum()

# Class balance of the target variable
df['target_column'].value_counts(normalize=True)
```

### 2. Turning Text into Numbers

When numerical columns are stored as text (e.g., prices with currency symbols).

```python
# Remove '$' and ',' then convert to float
df['price'] = df['price'].astype(str).str.replace('$', '').str.replace(',', '').astype(float)
```

### 3. Standardizing Category Labels

Cleaning up inconsistent text formatting in categorical columns.

```python
# Convert everything to lowercase and strip leading/trailing whitespace
df['category'] = df['category'].str.lower().str.strip()
```

### 4. Impossible Values and Missing Value Flags

Handling values that are clearly errors (e.g., negative ages).

```python
import numpy as np

# Find impossible values and replace them with NaN
df.loc[df['age'] < 0, 'age'] = np.nan

# Keep a missing-value flag (1 if missing, 0 if not)
df['age_missing'] = df['age'].isna().astype(int)
```

### 5. Dropping Invalid Model Inputs (Data Leakage)

You must remove columns that are IDs, constants, or represent **Data Leakage** (information only known *after* the outcome occurs).

```python
# Drop ID and leakage columns
df = df.drop(columns=['customer_id', 'post_purchase_feedback'])
```

### 6. One-Hot Encoding

Converting categorical variables into dummy/indicator variables.

```python
# One-hot encode the 'color' column
df = pd.get_dummies(df, columns=['color'], drop_first=True)
```

---

## Part 2: Classification

### 1. Splitting and Baseline

Always split the data before applying imputations or transformations.

```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=['target'])
y = df['target']

# Split with stratification to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Baseline accuracy (predicting the majority class)
baseline_acc = y_train.value_counts(normalize=True).max()
print(f"Baseline accuracy: {baseline_acc}")
```

### 2. Filling Missing Values

Fill missing values **after** splitting, using statistics from the **training set only**.

```python
# Example: Fill missing ages with the median of the training set
age_median = X_train['age'].median()

X_train['age'] = X_train['age'].fillna(age_median)
X_test['age'] = X_test['age'].fillna(age_median) 
```

### 3. KNN with a Pipeline

KNN requires feature scaling (Standardization).

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

knn_pipe = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5)
)

knn_pipe.fit(X_train, y_train)
```

### 4. Decision Trees & Cross-Validation

Choose hyperparameters (like `max_depth`) using cross-validation on the training set.

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

for depth in [2, 3, 4, 5, 6]:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    scores = cross_val_score(dt, X_train, y_train, cv=5)
    print(f"Depth: {depth}, Mean CV Accuracy: {scores.mean():.4f}")
```

### 5. Evaluating the Model

Understanding confusion matrices, precision, and recall.

```python
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

y_pred = knn_pipe.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
```

* **Precision:** Out of all instances the model predicted as positive, how many were actually positive? (Cost of False Positives).
* **Recall:** Out of all actual positive instances, how many did the model find? (Cost of False Negatives).

### 6. Logistic Regression and Thresholds

Using probabilities to make decisions based on business costs.

```python
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Get probabilities for the positive class (class 1)
probs = log_reg.predict_proba(X_test)[:, 1]

# Apply a custom threshold (e.g., 0.3)
custom_threshold = 0.3
y_pred_custom = (probs >= custom_threshold).astype(int)
```

---

## Part 3: Essential "Why" Questions

**Why do we split before filling missing values?**
To prevent **data leakage**. If we fill missing values using the mean/median of the entire dataset, information from the test set "leaks" into the training set. The model must learn only from the training data.

**Why do we choose a setting with cross-validation instead of the test set?**
If we tune hyperparameters (like `max_depth`) based on the test set's performance, we overfit to the test set. The test set must remain entirely unseen until the final model evaluation to provide an honest estimate of real-world performance.

**Why can a model with high accuracy be the worst choice?**
If the dataset is highly imbalanced (e.g., 99% healthy, 1% sick), a "dumb" model that always predicts "healthy" will have 99% accuracy. However, it fails to find the sick patients, which is the actual goal of the model.

**Why is 0.5 not automatically the right threshold?**
The default 0.5 threshold assumes False Positives and False Negatives have the same cost. In the real world, costs are often asymmetric. For example, in fraud detection, missing a fraudulent transaction (False Negative) is much more costly than temporarily flagging a legitimate one (False Positive), so we lower the threshold to catch more fraud.
