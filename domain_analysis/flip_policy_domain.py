import pandas as pd
import os

label_file = "id_policy_domain.csv"
matched_files = [
    "data/flip rate/active_passive_flip_4B.csv",
    "data/flip rate/active_passive_flip_12B.csv",
    "data/flip rate/it-clefts_flip_4B.csv",
    "data/flip rate/it-clefts_flip_12B.csv",
    "data/flip rate/negation_flip_4B_qwen.csv",
    "data/flip rate/negation_flip_4B.csv",
    "data/flip rate/negation_flip_12B.csv",
    "data/flip rate/negation_flip_14B.csv",
    "data/flip rate/opposite_flip_4B.csv",
    "data/flip rate/opposite_flip_12B.csv",
    "data/flip rate/SVC_flip_4B.csv",
    "data/flip rate/SVC_flip_12B.csv",
    "data/flip rate/wh-clefts_flip_4B.csv",
    "data/flip rate/wh-clefts_flip_12B.csv",
    "data/flip rate/active_passive_flip_4B_qwen.csv",
    "data/flip rate/active_passive_flip_14B.csv",
    "data/flip rate/it-clefts_flip_4B_qwen.csv",
    "data/flip rate/it-clefts_flip_14B.csv",
    "data/flip rate/opposite_flip_4B_qwen.csv",
    "data/flip rate/opposite_flip_14B.csv",
    "data/flip rate/SVC_flip_4B_qwen.csv",
    "data/flip rate/SVC_flip_14B.csv",
    "data/flip rate/wh-clefts_flip_4B_qwen.csv",
    "data/flip rate/wh-clefts_flip_14B.csv"
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
    print(f"Result was saved into {output_file}")
