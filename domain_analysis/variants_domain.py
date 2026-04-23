import pandas as pd
import os

label_file = "id_policy_domain.csv"
matched_files = [
    "data/CI/original_CI_4B.csv",
    "data/CI/negation_CI_4B.csv",
    "data/CI/opposite_CI_4B.csv",
    "data/CI/active_passive_CI_4B.csv",
    "data/CI/it-clefts_CI_4B.csv",
    "data/CI/SVC_CI_4B.csv",
    "data/CI/wh-clefts_CI_4B.csv",
]

df_label = pd.read_csv(label_file)
label_id = df_label[["ID", "policy_domain"]].copy()
label_id["ID"] = label_id["ID"].astype(str)

for one_file in matched_files:
    df_one_file = pd.read_csv(one_file)
    one_file_id = df_one_file["ID"].copy()
    one_file_id = one_file_id.astype(str).apply(
        lambda x: "_".join(x.split("_")[:2])
    )

    df_result = pd.DataFrame({"ID": one_file_id})
    result = df_result.merge(label_id, on="ID", how="left")

    label_counts = result["policy_domain"].value_counts()
    label_counts_df = label_counts.reset_index()
    label_counts_df.columns = ["policy_domain", "count"]

    base_name = os.path.splitext(os.path.basename(one_file))[0]
    output_file = f"data/domain/{base_name}_policy_domain.csv"

    label_counts_df.to_csv(output_file, index=False, encoding="utf-8")

    print(f"Saved: {output_file}")
