import pandas as pd, numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
df = pd.read_csv("D:/FlyRank_Intern/Resource/flyrank-ml-internship-starter/data/raw/content_refresh_anonymized.csv")

# The label: a page is 'declining' when its recent trend is down. Simple, honest starter label.
df["is_declining_label"] = df["trend_direction"].str.lower().eq("down").astype(int)
# print(df.shape[0], "pages |  declining rate:", round(df["is_declining_label"].mean(), 3))

stale   = (df["days_since_last_update"] >= 180).astype(int)
visible = (df["impressions_90d"] >= 500).astype(int)
df["hand_rule_score"] = stale * visible * df["impressions_90d"]

top10 = df.sort_values("hand_rule_score", ascending=False).head(10)
top10[["impressions_90d", "days_since_last_update", "avg_position", "ctr", "trend_direction"]]

def precision_at_k(scores, labels, k):
    order = np.argsort(-np.asarray(scores))
    topk = np.asarray(labels)[order[:k]]
    return topk.mean()

y = df["is_declining_label"].values
features = ["content_age_days", "days_since_last_update", "impressions_90d",
            "avg_position", "ctr", "word_count"]
X = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
tree = DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=42)
# tree.fit(X, y)
tree.fit(X_train, y_train)
pred = tree.predict(X_test)
# print(export_text(tree, feature_names=features))
print("--------------------------------")

tree_score = tree.predict_proba(X_test)[:, 1]
for k in (20, 50):
    hand_score_test = df.loc[X_test.index, "hand_rule_score"]
    # hr = precision_at_k(df["hand_rule_score"], y_test, k)
    hr = precision_at_k(hand_score_test, y_test, k)
    tr = precision_at_k(tree_score, y_test, k)
    print(f"Precision@{k}:  hand rule {hr:.3f}   vs   tree {tr:.3f}")
print("--------------------------------")

# X_leaky = df[features + ["trend_pct"]].replace([np.inf, -np.inf], np.nan).fillna(0)
# leaky = DecisionTreeClassifier(max_depth=2, class_weight="balanced", random_state=42).fit(X_leaky, y)
# print(f"'Leaky' tree Precision@50: {precision_at_k(leaky.predict_proba(X_leaky)[:,1], y, 50):.3f}  <- looks amazing")
# print(export_text(leaky, feature_names=features + ["trend_pct"]))

