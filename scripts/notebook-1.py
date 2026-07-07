import json
import pandas as pd, numpy as np

res = json.load(open("D:/FlyRank_Intern/Resource/flyrank-ml-internship-starter/outputs/model_results.json"))

base = res["baseline"]["baseline_precision_at_50"]
rf   = res["models"]["random_forest"]["precision_at_50"]

print(f"Hand-written rule  Precision@50: {base:.3f}   (~{round(base*50)} of the top 50 right)")
print(f"Random forest      Precision@50: {rf:.3f}   (~{round(rf*50)} of the top 50 right)")
print(f"\nThe learned model roughly {rf/base:.1f}x the rule on this metric.")
print("Validation split used:", res["split_strategy"], "(pages from a client are never in both train and test)")
print("--------------------------------")


df = pd.read_csv("D:/FlyRank_Intern/Resource/flyrank-ml-internship-starter/data/raw/content_refresh_anonymized.csv")
print(df.shape[0], "rows,", df.shape[1], "columns")
df.head(3)
print("--------------------------------")


corr = df["search_volume"].corr(df["impressions_90d"])
print(f"Correlation between search_volume and impressions_90d: {corr:.3f}")
print("Near zero -> keyword search volume barely predicts the traffic a page actually gets.")
print("--------------------------------")

visible = df[df["impressions_90d"] >= 100]
ctr_by_pos = visible.groupby(["position_tier","content_type"])["ctr"].mean().sort_values(ascending=False)
print(ctr_by_pos.round(4).to_string())
ctr_by_pos.plot(kind="bar", title="Mean CTR by position tier and content type (impressions >= 100)", ylabel="CTR");
print("--------------------------------")

wc = df.groupby("trend_direction")["word_count"].median()
print(wc.round(0).to_string())
print("\n'down' vs 'up' pages have almost the same median word count -> length is not the lever.")
print("--------------------------------")