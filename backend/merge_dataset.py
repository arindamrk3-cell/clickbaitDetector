import pandas as pd

# Load dataset 1
df1 = pd.read_csv("../dataset/train1.csv")

# Rename columns
df1 = df1.rename(columns={
    "headline": "text",
    "clickbait": "label"
})

# Keep only needed columns
df1 = df1[["text", "label"]]

# Load dataset 2
df2 = pd.read_csv("../dataset/train2.csv")

# Rename columns
df2 = df2.rename(columns={
    "title": "text",
    "label": "label"
})

# Convert labels
df2["label"] = df2["label"].map({
    "clickbait": 1,
    "news": 0
})

# Keep only needed columns
df2 = df2[["text", "label"]]

# Combine both
final_df = pd.concat([df1, df2], ignore_index=True)

# Remove null values
final_df = final_df.dropna()

# Shuffle dataset
final_df = final_df.sample(frac=1).reset_index(drop=True)

# Save final dataset
final_df.to_csv("../dataset/final_dataset.csv", index=False)

print("Dataset merged successfully!")
print("Total samples:", len(final_df))