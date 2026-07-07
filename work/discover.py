import json
import pandas as pd, numpy as np


x = "competition"
df = pd.read_csv("D:/FlyRank_Intern/Resource/flyrank-ml-internship-starter/data/raw/content_refresh_anonymized.csv")
filtered_df = df[df["impressions_90d"] > 0]
corr = filtered_df[x].corr(filtered_df["impressions_90d"])
print(f"Correlation between {x} and impressions_90d: {corr:.3f}")
print(f"Near zero -> {x} barely predicts the traffic a page actually gets.")
print(corr)