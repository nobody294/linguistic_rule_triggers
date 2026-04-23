import pandas as pd
import numpy as np
import json

def bootstrap_means(arr, m: int = 1000, seed: int = 123, decimals: int = 2):
    arr = np.asarray(arr, dtype=float)
    n = len(arr)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(m, n))
    means_un = arr[idx].mean(axis=1)

    lower, upper = np.percentile(means_un, [2.5, 97.5])
    means = np.round(means_un, decimals)
    lower = round(float(lower), decimals)
    upper = round(float(upper), decimals)

    return means, (lower, upper)

def run(input_csv: str, output_csv: str, decimals: int = 2) -> None:
    df = pd.read_csv(input_csv)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"]).copy()
    df["score_rescaled"] = ((df["score"] - 4.0) / 3.0).round(decimals)

    grouped = (
        df[["ID", "score_rescaled"]]
        .groupby("ID", sort=False)["score_rescaled"]
        .apply(list)
        .reset_index()
    )

    rows = []
    for _, r in grouped.iterrows():
        means, (low, high) = bootstrap_means(arr=r["score_rescaled"])
        rows.append({
            "ID": r["ID"],
            "score_list": json.dumps([float(x) for x in means]),
            "CI": json.dumps([float(low), float(high)])
        })
    
    out_df = pd.DataFrame(rows, columns=["ID", "score_list", "CI"])
    out_df.to_csv(output_csv, index=False)
    print(f"Done. Wrote {len(out_df)} rows to: {output_csv}")

if __name__ == "__main__":
    # run(input_csv="data/responses/original_responses_4B_qwen.csv", output_csv="data/CI/original_CI_4B_qwen.csv")
    # run(input_csv="data/responses/original_responses_4B.csv", output_csv="data/CI/original_CI_4B.csv")
    # run(input_csv="data/responses/original_responses_12B.csv", output_csv="data/CI/original_CI_12B.csv")
    # run(input_csv="data/responses/original_responses_14B.csv", output_csv="data/CI/original_CI_14B.csv")

    # run(input_csv="data/responses/negation_responses_4B_qwen.csv", output_csv="data/CI/negation_CI_4B_qwen.csv")
    # run(input_csv="data/responses/negation_responses_4B.csv", output_csv="data/CI/negation_CI_4B.csv")
    # run(input_csv="data/responses/negation_responses_12B.csv", output_csv="data/CI/negation_CI_12B.csv")
    # run(input_csv="data/responses/negation_responses_14B.csv", output_csv="data/CI/negation_CI_14B.csv")

    # run(input_csv="data/responses/opposite_responses_4B.csv", output_csv="data/CI/opposite_CI_4B.csv")
    # run(input_csv="data/responses/opposite_responses_4B_qwen.csv", output_csv="data/CI/opposite_CI_4B_qwen.csv")
    # run(input_csv="data/responses/opposite_responses_12B.csv", output_csv="data/CI/opposite_CI_12B.csv")
    # run(input_csv="data/responses/opposite_responses_14B.csv", output_csv="data/CI/opposite_CI_14B.csv")

    # run(input_csv="data/responses/active_passive_responses_4B.csv", output_csv="data/CI/active_passive_CI_4B.csv")
    # run(input_csv="data/responses/active_passive_responses_4B_qwen.csv", output_csv="data/CI/active_passive_CI_4B_qwen.csv")
    # run(input_csv="data/responses/active_passive_responses_12B.csv", output_csv="data/CI/active_passive_CI_12B.csv")
    # run(input_csv="data/responses/active_passive_responses_14B.csv", output_csv="data/CI/active_passive_CI_14B.csv")

    # run(input_csv="data/responses/it-clefts_responses_4B.csv", output_csv="data/CI/it-clefts_CI_4B.csv")
    # run(input_csv="data/responses/it-clefts_responses_4B_qwen.csv", output_csv="data/CI/it-clefts_CI_4B_qwen.csv")
    # run(input_csv="data/responses/it-clefts_responses_12B.csv", output_csv="data/CI/it-clefts_CI_12B.csv")
    # run(input_csv="data/responses/it-clefts_responses_14B.csv", output_csv="data/CI/it-clefts_CI_14B.csv")

    # run(input_csv="data/responses/wh-clefts_responses_4B.csv", output_csv="data/CI/wh-clefts_CI_4B.csv")
    # run(input_csv="data/responses/wh-clefts_responses_4B_qwen.csv", output_csv="data/CI/wh-clefts_CI_4B_qwen.csv")
    # run(input_csv="data/responses/wh-clefts_responses_12B.csv", output_csv="data/CI/wh-clefts_CI_12B.csv")
    # run(input_csv="data/responses/wh-clefts_responses_14B.csv", output_csv="data/CI/wh-clefts_CI_14B.csv")

    # run(input_csv="data/responses/SVC_responses_4B.csv", output_csv="data/CI/SVC_CI_4B.csv")
    # run(input_csv="data/responses/SVC_responses_4B_qwen.csv", output_csv="data/CI/SVC_CI_4B_qwen.csv")
    # run(input_csv="data/responses/SVC_responses_12B.csv", output_csv="data/CI/SVC_CI_12B.csv")
    # run(input_csv="data/responses/SVC_responses_14B.csv", output_csv="data/CI/SVC_CI_14B.csv")



    # run(input_csv="data/responses/combine_negation_opposite_responses_4B.csv", output_csv="data/CI/combine_negation_opposite_CI_4B.csv")
    # run(input_csv="data/responses/combine_negation_opposite_responses_4B_qwen.csv", output_csv="data/CI/combine_negation_opposite_CI_4B_qwen.csv")
    # run(input_csv="data/responses/combine_negation_opposite_responses_12B.csv", output_csv="data/CI/combine_negation_opposite_CI_12B.csv")
    # run(input_csv="data/responses/combine_negation_opposite_responses_14B.csv", output_csv="data/CI/combine_negation_opposite_CI_14B.csv")

    # run(input_csv="data/responses/combine_negation_active_passive_responses_4B.csv", output_csv="data/CI/combine_negation_active_passive_CI_4B.csv")
    # run(input_csv="data/responses/combine_negation_active_passive_responses_12B.csv", output_csv="data/CI/combine_negation_active_passive_CI_12B.csv")
    run(input_csv="data/responses/combine_negation_active_passive_responses_4B_qwen.csv", output_csv="data/CI/combine_negation_active_passive_CI_4B_qwen.csv")
    run(input_csv="data/responses/combine_negation_active_passive_responses_14B.csv", output_csv="data/CI/combine_negation_active_passive_CI_14B.csv")

    # run(input_csv="data/responses/combine_negation_it-clefts_responses_4B.csv", output_csv="data/CI/combine_negation_it-clefts_CI_4B.csv")
    # run(input_csv="data/responses/combine_negation_it-clefts_responses_12B.csv", output_csv="data/CI/combine_negation_it-clefts_CI_12B.csv")
    run(input_csv="data/responses/combine_negation_it-clefts_responses_4B_qwen.csv", output_csv="data/CI/combine_negation_it-clefts_CI_4B_qwen.csv")
    run(input_csv="data/responses/combine_negation_it-clefts_responses_14B.csv", output_csv="data/CI/combine_negation_it-clefts_CI_14B.csv")

    # run(input_csv="data/responses/combine_negation_wh-clefts_responses_4B.csv", output_csv="data/CI/combine_negation_wh-clefts_CI_4B.csv")
    # run(input_csv="data/responses/combine_negation_wh-clefts_responses_12B.csv", output_csv="data/CI/combine_negation_wh-clefts_CI_12B.csv")
    run(input_csv="data/responses/combine_negation_wh-clefts_responses_4B_qwen.csv", output_csv="data/CI/combine_negation_wh-clefts_CI_4B_qwen.csv")
    run(input_csv="data/responses/combine_negation_wh-clefts_responses_14B.csv", output_csv="data/CI/combine_negation_wh-clefts_CI_14B.csv")

    # run(input_csv="data/responses/combine_opposite_active_passive_responses_4B.csv", output_csv="data/CI/combine_opposite_active_passive_CI_4B.csv")
    # run(input_csv="data/responses/combine_opposite_active_passive_responses_12B.csv", output_csv="data/CI/combine_opposite_active_passive_CI_12B.csv")
    run(input_csv="data/responses/combine_opposite_active_passive_responses_4B_qwen.csv", output_csv="data/CI/combine_opposite_active_passive_CI_4B_qwen.csv")
    run(input_csv="data/responses/combine_opposite_active_passive_responses_14B.csv", output_csv="data/CI/combine_opposite_active_passive_CI_14B.csv")

    # run(input_csv="data/responses/combine_opposite_it-clefts_responses_4B.csv", output_csv="data/CI/combine_opposite_it-clefts_CI_4B.csv")
    # run(input_csv="data/responses/combine_opposite_it-clefts_responses_12B.csv", output_csv="data/CI/combine_opposite_it-clefts_CI_12B.csv")
    run(input_csv="data/responses/combine_opposite_it-clefts_responses_4B_qwen.csv", output_csv="data/CI/combine_opposite_it-clefts_CI_4B_qwen.csv")
    run(input_csv="data/responses/combine_opposite_it-clefts_responses_14B.csv", output_csv="data/CI/combine_opposite_it-clefts_CI_14B.csv")

    # run(input_csv="data/responses/combine_opposite_wh-clefts_responses_4B.csv", output_csv="data/CI/combine_opposite_wh-clefts_CI_4B.csv")
    # run(input_csv="data/responses/combine_opposite_wh-clefts_responses_12B.csv", output_csv="data/CI/combine_opposite_wh-clefts_CI_12B.csv")
    run(input_csv="data/responses/combine_opposite_wh-clefts_responses_4B_qwen.csv", output_csv="data/CI/combine_opposite_wh-clefts_CI_4B_qwen.csv")
    run(input_csv="data/responses/combine_opposite_wh-clefts_responses_14B.csv", output_csv="data/CI/combine_opposite_wh-clefts_CI_14B.csv")

    # run(input_csv="data/responses/combine_it-clefts_active_passive_responses_4B.csv", output_csv="data/CI/combine_it-clefts_active_passive_CI_4B.csv")
    # run(input_csv="data/responses/combine_it-clefts_active_passive_responses_12B.csv", output_csv="data/CI/combine_it-clefts_active_passive_CI_12B.csv")
    run(input_csv="data/responses/combine_it-clefts_active_passive_responses_4B_qwen.csv", output_csv="data/CI/combine_it-clefts_active_passive_CI_4B_qwen.csv")
    run(input_csv="data/responses/combine_it-clefts_active_passive_responses_14B.csv", output_csv="data/CI/combine_it-clefts_active_passive_CI_14B.csv")

    # run(input_csv="data/responses/combine_wh-clefts_active_passive_responses_4B.csv", output_csv="data/CI/combine_wh-clefts_active_passive_CI_4B.csv")
    # run(input_csv="data/responses/combine_wh-clefts_active_passive_responses_12B.csv", output_csv="data/CI/combine_wh-clefts_active_passive_CI_12B.csv")
    run(input_csv="data/responses/combine_wh-clefts_active_passive_responses_4B_qwen.csv", output_csv="data/CI/combine_wh-clefts_active_passive_CI_4B_qwen.csv")
    run(input_csv="data/responses/combine_wh-clefts_active_passive_responses_14B.csv", output_csv="data/CI/combine_wh-clefts_active_passive_CI_14B.csv")

    # run(input_csv="data/responses/combine_negation_SVC_responses_4B.csv", output_csv="data/CI/combine_negation_SVC_CI_4B.csv")
    # run(input_csv="data/responses/combine_negation_SVC_responses_12B.csv", output_csv="data/CI/combine_negation_SVC_CI_12B.csv")
    run(input_csv="data/responses/combine_negation_SVC_responses_4B_qwen.csv", output_csv="data/CI/combine_negation_SVC_CI_4B_qwen.csv")
    run(input_csv="data/responses/combine_negation_SVC_responses_14B.csv", output_csv="data/CI/combine_negation_SVC_CI_14B.csv")

    # run(input_csv="data/responses/combine_opposite_SVC_responses_4B.csv", output_csv="data/CI/combine_opposite_SVC_CI_4B.csv")
    # run(input_csv="data/responses/combine_opposite_SVC_responses_12B.csv", output_csv="data/CI/combine_opposite_SVC_CI_12B.csv")
    run(input_csv="data/responses/combine_opposite_SVC_responses_4B_qwen.csv", output_csv="data/CI/combine_opposite_SVC_CI_4B_qwen.csv")
    run(input_csv="data/responses/combine_opposite_SVC_responses_14B.csv", output_csv="data/CI/combine_opposite_SVC_CI_14B.csv")

    # run(input_csv="data/responses/combine_active_passive_SVC_responses_4B.csv", output_csv="data/CI/combine_active_passive_SVC_CI_4B.csv")
    # run(input_csv="data/responses/combine_active_passive_SVC_responses_12B.csv", output_csv="data/CI/combine_active_passive_SVC_CI_12B.csv")
    run(input_csv="data/responses/combine_active_passive_SVC_responses_4B_qwen.csv", output_csv="data/CI/combine_active_passive_SVC_CI_4B_qwen.csv")
    run(input_csv="data/responses/combine_active_passive_SVC_responses_14B.csv", output_csv="data/CI/combine_active_passive_SVC_CI_14B.csv")

    # run(input_csv="data/responses/combine_it-clefts_SVC_responses_4B.csv", output_csv="data/CI/combine_it-clefts_SVC_CI_4B.csv")
    # run(input_csv="data/responses/combine_it-clefts_SVC_responses_12B.csv", output_csv="data/CI/combine_it-clefts_SVC_CI_12B.csv")
    run(input_csv="data/responses/combine_it-clefts_SVC_responses_4B_qwen.csv", output_csv="data/CI/combine_it-clefts_SVC_CI_4B_qwen.csv")
    run(input_csv="data/responses/combine_it-clefts_SVC_responses_14B.csv", output_csv="data/CI/combine_it-clefts_SVC_CI_14B.csv")

    # run(input_csv="data/responses/combine_wh-clefts_SVC_responses_4B.csv", output_csv="data/CI/combine_wh-clefts_SVC_CI_4B.csv")
    # run(input_csv="data/responses/combine_wh-clefts_SVC_responses_12B.csv", output_csv="data/CI/combine_wh-clefts_SVC_CI_12B.csv")
    run(input_csv="data/responses/combine_wh-clefts_SVC_responses_4B_qwen.csv", output_csv="data/CI/combine_wh-clefts_SVC_CI_4B_qwen.csv")
    run(input_csv="data/responses/combine_wh-clefts_SVC_responses_14B.csv", output_csv="data/CI/combine_wh-clefts_SVC_CI_14B.csv")