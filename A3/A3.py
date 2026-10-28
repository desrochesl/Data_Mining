# %% [markdown]
# # Assignment 3 · Patterns and groups
# ## The campus convenience store: what sells together, and who shops here?
# **DATA3010 · put your name here:** Luna DesRoches
#
# Read `A3_Assignment.md` for the full instructions, the marks, and the rules on AI use.
#
# The two data cells below are fixed. Do not change them: everyone's numbers should match.
#
# Write your code under each task. Write your answers in the answer cells. Submit this notebook with its outputs.

# %%
import numpy as np
import pandas as pd
from itertools import combinations
from IPython.display import display

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score

pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 120)

RANDOM_STATE = 42
MIN_SUPPORT = 0.05  # an itemset is "frequent" if it is in at least 5% of receipts

print("Setup complete.")

# %% [markdown]
# # Part A · What sells together?
# The store keeps one row per item on each receipt. 1,500 receipts from the last month.
#
# | Column | Meaning |
# |---|---|
# | `receipt_id` | Receipt number |
# | `item` | One item on that receipt |
# | `price_usd` | Price of that item |

# %%
# Fixed data cell, part A. Do not change it.
rng = np.random.default_rng(1021)
n_receipts = 1500

# Each visit has a reason; the reason decides what usually goes in the basket.
visit_types = ["morning", "lunch", "late_night", "healthy", "study", "quick"]
type_share = [0.27, 0.22, 0.15, 0.13, 0.05, 0.18]
usual_items = {
    "morning": {"coffee": 0.9, "muffin": 0.45, "banana": 0.15},
    "lunch": {"sandwich": 0.85, "chips": 0.6, "soda": 0.45, "coffee": 0.3},
    "late_night": {
        "instant_noodles": 0.75,
        "energy_drink": 0.65,
        "candy": 0.5,
        "chips": 0.35,
    },
    "healthy": {"salad": 0.8, "water": 0.7, "banana": 0.45, "protein_bar": 0.35},
    "study": {"notebook": 0.85, "pens": 0.75, "coffee": 0.4},
    "quick": {"water": 0.45, "gum": 0.35, "candy": 0.25, "soda": 0.3},
}
# Anyone may also grab one of these, whatever the reason for the visit.
add_ons = {
    "coffee": 0.22,
    "water": 0.18,
    "gum": 0.06,
    "candy": 0.06,
    "chips": 0.07,
    "protein_bar": 0.04,
}

lines = []
for r in range(1, n_receipts + 1):
    kind = rng.choice(visit_types, p=type_share)
    items = {item for item, p in usual_items[kind].items() if rng.random() < p}
    items |= {item for item, p in add_ons.items() if rng.random() < p}
    if not items:
        items = {rng.choice(list(usual_items[kind]))}
    for item in sorted(items):
        lines.append({"receipt_id": f"R{r:04d}", "item": item, "price_usd": None})

prices = {
    "coffee": 2.5,
    "muffin": 2.75,
    "banana": 0.75,
    "sandwich": 6.5,
    "chips": 1.75,
    "soda": 2.0,
    "instant_noodles": 1.5,
    "energy_drink": 3.25,
    "candy": 1.5,
    "salad": 7.0,
    "water": 1.25,
    "protein_bar": 2.5,
    "notebook": 4.0,
    "pens": 3.0,
    "gum": 1.0,
}
order_lines = pd.DataFrame(lines)
order_lines["price_usd"] = order_lines["item"].map(prices)

print("Order lines:", len(order_lines))
display(order_lines.head(8))

# %% [markdown]
# ## Task 1 · Build the baskets (10 marks)
# Turn the order lines into a **basket table**: one row per receipt, one column per item, `True` if the receipt has that item.
#
# **Report:** the number of receipts, the number of different items, the average number of items per receipt, and the 5 most common items.

# %%
# basket = ...      one row per receipt, one True/False column per item
#                   (hint: pd.crosstab(order_lines["receipt_id"], order_lines["item"]) > 0)
basket = pd.crosstab(order_lines["receipt_id"], order_lines["item"]) > 0
receipts, items = basket.shape
print(f"Number of receipts: {receipts}\nNumber of items: {items}")
display(basket.describe())
display(basket.sum().sort_values(ascending=False))
avg_unique_items = basket.sum(axis=1).mean()
print(f"Average distinct items per receipt: {avg_unique_items:.2f}")

# %% [markdown]
# **Your answer:**
#
# *There are 1500 receipts, 15 different items, there are 2.4 items bought per receipt. The 5 most common items are: coffee, water, chips, sandwich, and candy.*
#

# %% [markdown]
# ## Task 2 · Frequent items and frequent pairs (20 marks)
# **a)** Compute the **support** of every item (the share of receipts that contain it). List the items whose support is at least `MIN_SUPPORT`.
#
# **b)** Use the Apriori idea: a pair can only be frequent if **both** of its items are frequent. Build candidate pairs only from the frequent items. Report how many candidate pairs you check, and how many pairs there would be if you checked every pair of the 15 items.
#
# **c)** Keep the **frequent pairs**: candidates with support at least `MIN_SUPPORT`. Show them, sorted by support.


# %%
# a) item_support = ...   frequent_items = [...]
# b) candidate pairs from frequent_items only   (hint: itertools.combinations)
# c) frequent_pairs = ...  a DataFrame with item_a, item_b, support
def support(items):
    return basket[items].all(axis=1).mean()


item_support = basket.mean().sort_values(ascending=False)
display(item_support)
frequent_items = sorted(item_support[item_support >= MIN_SUPPORT].index)
display(frequent_items)

candidate_pairs = list(combinations(frequent_items, 2))

frequent_pairs = (
    pd.DataFrame(
        {"item_a": a, "item_b": b, "support": support([a, b])}
        for a, b in candidate_pairs
    )
    .sort_values("support", ascending=False)
    .reset_index(drop=True)
)

print(f"Total pairs (all items): {len(list(combinations(item_support.index, 2)))}")
print(f"Candidate pairs checked: {len(candidate_pairs)}")
print(f"Frequent pairs: {len(frequent_pairs)}")
display(frequent_pairs)

# %% [markdown]
# **Your answer:**
#
# **Total pairs (all items): 105 Candidate pairs checked: 78 Frequent pairs: 78**

# %% [markdown]
# ## Task 3 · Rules: confidence and lift (20 marks)
# **a)** For every frequent pair, write both rules ("if A then B" and "if B then A") with their **support, confidence and lift**. Show the table sorted by lift, highest first.
#
# **b)** Find a rule with **confidence of at least 0.45 but lift between 0.9 and 1.2**. Explain in plain words why it is not a useful rule even though its confidence looks high.
#
# **c)** Lower the minimum support to **0.02** and check the pair `notebook` & `pens`. Report its support, confidence and lift. What did the 5% cut-off hide, and is this pair worth acting on?

# %%
# a) rules = ...  columns: if_item, then_item, support, confidence, lift   (sorted by lift)
#    confidence(A -> B) = support(A and B) / support(A)
#    lift(A -> B)       = confidence(A -> B) / support(B)
# b) filter rules for confidence >= 0.45 and lift between 0.9 and 1.2
# c) notebook & pens
rules = (
    pd.DataFrame(
        [
            {
                "if_item": a,
                "then_item": b,
                "support": round(sup, 3),
                "confidence": round(sup / support([a]), 3),
                "lift": round((sup / support([a])) / support([b]), 3),
            }
            for x, y in zip(frequent_pairs["item_a"], frequent_pairs["item_b"])
            for a, b in [(x, y), (y, x)]
            for sup in [support([a, b])]
        ]
    )
    .sort_values("lift", ascending=False)
    .reset_index(drop=True)
)

# display(rules)
filtered_rules = rules[
    (rules["confidence"] >= 0.45) & (rules["lift"] > 0.9) & (rules["lift"] < 1.2)
]
display(rules, "confidence < 0.45, 0.9 < lift < 1.2:", filtered_rules)

# %% [markdown]
# **Your answer:**
#
# *Notebook and pens had a support of 0.030, confidence of 0.714, and a lift of 21.429. Lowering the min support revealed that the lift was exceptionally high, while the confidence was just 2 hundreths off of the minimum.*

# %% [markdown]
# # Part B · Who shops here?
# The store's app has 400 members. For each one, the store averaged their last three months.
#
# | Column | Meaning |
# |---|---|
# | `member_id` | App account number |
# | `visits_per_month` | Average visits per month |
# | `avg_basket_usd` | Average spend per visit, in dollars |
# | `late_night_share` | Share of visits after 10 PM (0 to 1) |
# | `healthy_share` | Share of spending on salad, fruit, water and protein bars (0 to 1) |
# | `home_zip` | Home ZIP code, stored as a number (for example 2115 means 02115) |
#
# There is no answer column. You will find groups of members and describe them.

# %%
# Fixed data cell, part B. Do not change it.
rng = np.random.default_rng(2110)


def group(n, visits, basket, late, healthy):
    return pd.DataFrame(
        {
            "visits_per_month": np.clip(
                rng.normal(visits[0], visits[1], n), 1, 40
            ).round(0),
            "avg_basket_usd": np.clip(
                rng.normal(basket[0], basket[1], n), 1.5, 40
            ).round(2),
            "late_night_share": np.clip(rng.normal(late[0], late[1], n), 0, 1).round(2),
            "healthy_share": np.clip(rng.normal(healthy[0], healthy[1], n), 0, 1).round(
                2
            ),
        }
    )


members = (
    pd.concat(
        [
            group(130, (22, 3.5), (5.0, 1.2), (0.06, 0.04), (0.20, 0.07)),
            group(105, (12, 3.0), (11.5, 2.0), (0.10, 0.06), (0.30, 0.08)),
            group(80, (11, 3.0), (8.5, 1.8), (0.72, 0.09), (0.10, 0.06)),
            group(75, (5, 1.5), (21.0, 3.5), (0.20, 0.08), (0.62, 0.10)),
            pd.DataFrame(
                {
                    "visits_per_month": rng.integers(1, 35, 10).astype(float),
                    "avg_basket_usd": rng.uniform(2, 35, 10).round(2),
                    "late_night_share": rng.uniform(0, 1, 10).round(2),
                    "healthy_share": rng.uniform(0, 1, 10).round(2),
                }
            ),
        ],
        ignore_index=True,
    )
    .sample(frac=1, random_state=5)
    .reset_index(drop=True)
)

members.insert(0, "member_id", [f"A{i:04d}" for i in range(1, len(members) + 1)])
members["home_zip"] = rng.choice(
    [2115, 2120, 2130, 2139, 2215, 2446, 1801], len(members)
)

print("Members:", len(members))
display(members.head())

# %% [markdown]
# ## Task 4 · Choose and prepare the features (10 marks)
# Decide which columns can be clustering inputs. Explain in one sentence each why you leave any column out.
#
# Then show the standard deviation of each feature you keep, and scale them with `StandardScaler`. Call the result `X_scaled`.

# %%
# feature_cols = [...]
# X = members[feature_cols]
# X_scaled = ...
feature_cols = [
    "visits_per_month",
    "avg_basket_usd",
    "late_night_share",
    "healthy_share",
]
X = members[feature_cols]
X_scaled = StandardScaler().fit_transform(X)

print(X.std())
print(X_scaled)

# %% [markdown]
# **Your answer:**
#
# *I chose the columns `["visits_per_month", "avg_basket_usd", "late_night_share", "healthy_share"]`. I chose these features because they each are integers that have a meaning to their values, and they are not personally identifible columns.*

# %% [markdown]
# ## Task 5 · k-means: choose k and describe the groups (20 marks)
# **a)** For k from 2 to 8, fit `KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)` on `X_scaled` and report inertia and silhouette in one table. Choose k and say why, using both columns.
#
# **b)** Fit your chosen k. Show each cluster's size and the average of each feature **in the original units**. Give each cluster a short name backed by its numbers.
#
# **c)** Fit the same k-means on the **unscaled** `X`. In two or three sentences, say what changes and why.

# %%
# a) k_table = ...   (k, inertia, silhouette)
# b) final_model = ...   profile in original units, sizes, names
# c) the same on unscaled X
rows = []
k_values = range(2, 8)
for k in k_values:
    model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X_scaled)
    inertia = model.inertia_
    silhouette = silhouette_score(X_scaled, model.labels_)
    rows.append(
        {
            "k": k,
            "inertia": inertia,
            "silhouette": silhouette,
        }
    )

k_table = pd.DataFrame(rows)
best_k = int(k_table.loc[k_table["silhouette"].idxmax(), "k"])
print(k_table)
print(f"Best k: {best_k}")

final_model = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE).fit(
    X_scaled
)

segments = members.copy()
segments["cluster"] = final_model.labels_

display(
    segments.groupby("cluster")[feature_cols]
    .mean()
    .assign(size=segments["cluster"].value_counts())
    .round(2)
)
print("Overall average:\n", members[feature_cols].mean().round(2))


# %% [markdown]
# **Your answer:**
#
# I choose k=4. This value is the peak for sihlouette and the intertia drops a small amount after this k.
#
#
# Clusters:
#
# 0. Frequent quick shoppers, `n=133, visits/month=22.86, avg_basket_usd=5.06, late_night_share=0.08, healthy_share=0.21`. These numbers tell us that the highest shopper visits 22.86 times per month and spends at least $5.06 per basket. They also rarely shop at night.
# 1. Night-owls, `n=93, visits/month=8.76, avg_basket_size=10.65, late_night_share=0.4, healthy_share=0.24`. These buyers have the highest late night activity; while having a moderate amount of visits.
# 2. Health-enthusiests, `n=74, visits/month=5.69, avg_basket_size=21.74, late_night_share=0.22, healthy_share=0.63`. These shoppers tend to visit less frequently, but spend more per visit and are significantly more health conscious than most spenders.
# 3. Balanced Regulars, `n=111, visits/month=13.94, avg_basket_size=9.84, late_night_share=0.31, healthy_share=0.23`. These buyers are average in all aspects; they visit and spend a fair amount. They represent the common man who may visit after work.
#
# Without scaling, features with larger magnitudes and variances (like visits/month and avg_basket) completely dominate the calculations over ratio features bounded between 0 and 1 (late_night_share and healthy_share). This causes the clusters to be split based on visits and basket size, ignoring shopping patterns.

# %% [markdown]
# ## Task 6 · Try a second method (10 marks)
# **a)** Fit `AgglomerativeClustering` with Ward linkage and your k on `X_scaled`. Compare it with your k-means clusters using `pd.crosstab`.
#
# **b)** Run `DBSCAN(eps=..., min_samples=5)` on `X_scaled` for at least four `eps` values. For each, report the number of clusters (not counting -1) and the number of noise points.
#
# **c)** Which method would you hand to the store for these members, and why? Use your numbers.

# %%
# a) ward_labels = ...    pd.crosstab(final_model.labels_, ward_labels)
# b) eps_table = ...      (eps, clusters, noise)
# c) your choice in the answer cell
ward_labels = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit_predict(
    X_scaled
)
display(
    pd.crosstab(
        final_model.labels_, ward_labels, rownames=["KMeans"], colnames=["Ward"]
    )
)

eps_values = [0.2, 0.3, 0.5, 0.7]

rows = []
for eps in eps_values:
    e = DBSCAN(eps=eps, min_samples=5).fit_predict(X_scaled)
    rows.append(
        {
            "eps": eps,
            "clusters": len(set(e) - {-1}),
            "noise": int((e == -1).sum()),
        }
    )
eps_table = pd.DataFrame(rows)

display(eps_table)

# %% [markdown]
# **Your answer:**
#
# I would had the k-means model to the store. The comparison shows k-means and Ward agree on clustering.
#
# KMeans:
# Ward       0    1   2    3
# KMeans
# 0          0  128   0    0   -> 100% of KMeans 0 are Ward 1
# 1          2    4   0  104   -> 104 of 110 in KMeans 1 are Ward 3
# 2          1    0  80    0   -> 80 of 81 in KMeans 2 are Ward
#
# Each cluster shows a near perfect agreement between the models. Each cluster represents robust, compact groups.

# %% [markdown]
# ## Task 7 · Recommend, and answer the AI (10 marks)
# **a) Your recommendation.** In about 200 words, give the store **two** concrete actions: one from Part A (what to place or promote together) and one from Part B (which group to target, and how). Use at least three numbers from your results.
#
# **b) The AI analysis.** The store manager asks an AI assistant to do the same job. It replies:
#
# > "For the baskets, the strongest rule is sandwich → coffee: about half of the customers who buy a sandwich also buy coffee (confidence 0.50), so put a coffee machine next to the sandwich fridge. I used a minimum support of 5%, so there are no other patterns worth reporting. For the members, I clustered all six columns without scaling, because they are all numbers already. I chose k = 8 because it had the lowest inertia. Cluster 7 has the highest number, so it is your most valuable segment."
#
# Find **at least four** things wrong with that reply. For each one: what is wrong, the evidence from your notebook, and what you would do instead.

# %%


# %% [markdown]
# **Your answer:**
#
# **a)**
# Part A:
#
# Instead of relying on coffee and sandwiches, which have a lift near 1.08. I recommend placing energy drinks next to instant noodles and offer a combo deal. In the data, energy_drink -> instant noodles boasts a support of 8%, confidence of 71.9%, and lift of 5.764. This indicates customers buying energy drinks are 6x more likely to buy noodles compared to the baseline.
#
# Part B:
#
# Cluster 1, the "Night-Owl" cluster; contains 93 members who has a 40% share of late night shoppers. This cluster is well above the average of night shoppers (24%), we should provide quick discounts in the evening to drive up visit frequency and basket size.
#
#
# **b)**
# 1. Sandwich -> coffee may not be the strongest rule. Judging solely by confidence ignores the effects of lift and support. I would instead evalutate with lift and support alongside confidence to ensure items genuinely drive one another.
# 2. It is a bad idea to use every column, the dataset contains columns like `member_id` and `home_zip`; which are arbitrary identifiers. Instead I would remove those features and any other that may be an arbitrary number.
# 3. Leaving them unscaled results in having features with high variance to overpower ratio features which contain values between 0 and 1. To fix this, I would scale the dataset to ensure each feature is weighed accordingly.
# 4. The AI recommended them to chose Cluster 7 because it has the highets number. The cluster index numbers are arbitrary labels that have no rank or value. To fix this, I would look at the inertia and sihlouette scores. I would pick the cluster with the peak of the sihlouette score and the elbow of the inertia score.

# %% [markdown]
# ## Before you submit
# - Restart the kernel and run all cells, top to bottom, with no errors.
# - Every task has code, output, and a written answer.
# - If you used an AI tool, say where and for what, in one or two lines below.
#
# **AI use statement:**
#
# In this assignment; I used AI to explain and remind me on concepts, debug, and review my work to ensure I meet the requirements for this assignment. I used it to double check my work on the coding blocks for Tasks 1-6. No code was written by any AI.
