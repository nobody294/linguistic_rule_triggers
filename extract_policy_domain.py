import pandas as pd

input_file = "annotations_spiderweb_gold.csv"
output_file = "id_policy_domain.csv"

df = pd.read_csv(input_file)

# Keep only the first and third columns by column name
result = df[["ID", "policy_domain"]].copy()

# Keep only the first two parts of the ID
# Example: "ch_16_1000001_de" -> "ch_16"
result["ID"] = result["ID"].astype(str).apply(
    lambda x: "_".join(x.split("_")[:2])
)

# Count how many unique labels exist in the third column
unique_label_count = result["policy_domain"].nunique()

label_counts = result["policy_domain"].value_counts()



# Save the processed data to a new CSV file
result.to_csv(output_file, index=False, encoding="utf-8")

# Print the number of unique labels in policy_domain
print(f"Number of unique labels in policy_domain: {unique_label_count}")
print("Counts for each label: ")
for label, count in label_counts.items():
    print(f"{label}: {count}")
