import re
import os
import pandas as pd
from typing import Dict, Set

files = {
    "original": "data/responses/original_responses_12B.csv",
    "active_passive": "data/responses/active_passive_responses_12B.csv",
    "it_cleft": "data/responses/it-clefts_responses_12B.csv",
    "wh_cleft": "data/responses/wh-clefts_responses_12B.csv",
    "SVC": "data/responses/SVC_responses_12B.csv",
    "negation": "data/responses/negation_responses_12B.csv",
    "opposite": "data/responses/opposite_responses_12B.csv"
}

domain_map_path = "id_policy_domain.csv"

REVERSING_VARIANTS: Set[str] = {"negation", "opposite"}

LIKERT_MIN = 1
LIKERT_MAX = 7

BASE_ID_RE = re.compile(r"^[a-z]{2}_[0-9]{1,2}")

def extract_base_id(id_: str) -> str:
    id_ = str(id_).strip()
    m = BASE_ID_RE.match(id_)
    if m:
        return m.group()
    else:
        print(f"[warn] {id_} caused an error when extracting base id")
        return None

def reverse_likert(x: float, min_val: int = 1, max_val: int = 7) -> float:
    # 1-7 时就是 8 - x
    return (min_val + max_val) - x

def load_one(variant: str, path: str, reversing_variants: Set[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[["ID", "score"]].copy()
    df["ID"] = df["ID"].astype(str).str.strip()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")

    if variant in reversing_variants:
        mask = df["score"].notna()
        df.loc[mask, "score"] = df.loc[mask, "score"].apply(
            lambda x: reverse_likert(x, LIKERT_MIN, LIKERT_MAX)
        )

    df["base_id"] = df["ID"].map(extract_base_id)
    df["variant"] = variant

    df["rule_type"] = "reversing" if variant in reversing_variants else "preserving_or_base"

    df = df.dropna(subset=["score", "base_id", "ID"])
    return df[["ID", "base_id", "variant", "rule_type", "score"]]

def load_domain_map(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[["ID", "policy_domain"]].copy()
    df["ID"] = df["ID"].astype(str).str.strip()
    df["policy_domain"] = df["policy_domain"].astype(str).str.strip()

    df = df.rename(columns={"ID": "base_id"})
    df = df.dropna(subset=["base_id", "policy_domain"]).drop_duplicates(subset=["base_id"])
    return df

def load_all(
    files: Dict[str, str],
    domain_map_path: str,
    reversing_variants: Set[str]
) -> pd.DataFrame:
    frames = []
    for variant, path in files.items():
        if not os.path.exists(path):
            print(f"[warn] {path} does not exist")
            continue
        frames.append(load_one(variant, path, reversing_variants))

    if not frames:
        raise RuntimeError("No input files were loaded. Please check paths.")

    df_all = pd.concat(frames, ignore_index=True)

    domain_map = load_domain_map(domain_map_path)
    df_all = df_all.merge(domain_map, on="base_id", how="left", validate="m:1")

    missing = df_all["policy_domain"].isna().sum()
    if missing > 0:
        print(f"[warn] {missing} rows could not be matched to a policy_domain and will be dropped")

    df_all = df_all.dropna(subset=["policy_domain"]).copy()

    return df_all[["ID", "base_id", "variant", "rule_type", "policy_domain", "score"]]


def compute_mu_by_domain_direct(
    df_all: pd.DataFrame,
    include_original: bool = True
) -> pd.DataFrame:
    df = df_all.copy()
    if not include_original:
        df = df[df["variant"] != "original"].copy()

    mu_prompt = (
        df.groupby(["policy_domain", "variant", "ID"], sort=False)["score"]
        .agg(MU_prompt=lambda x: float(x.var(ddof=1)) if x.size > 1 else 0.0)
        .reset_index()
    )

    mu_domain = (
        mu_prompt.groupby("policy_domain", sort=False)
        .agg(
            MU=("MU_prompt", "mean"),
            n_prompt_cells=("MU_prompt", "size"),
            n_variants=("variant", "nunique")
        )
        .reset_index()
    )

    return mu_domain


def compute_as_ps_by_domain_direct(
    df_all: pd.DataFrame,
    base_variant: str = "original",
    complete_only: bool = False
) -> pd.DataFrame:
    df = df_all.copy()

    non_base_variants = sorted([v for v in df["variant"].dropna().unique() if v != base_variant])
    expected_n_rules = len(non_base_variants)

    g_sv = (
        df.groupby(["policy_domain", "base_id", "variant"], sort=False)["score"]
        .mean()
        .rename("ybar_sv")
        .reset_index()
    )

    original_df = (
        g_sv[g_sv["variant"] == base_variant][["policy_domain", "base_id", "ybar_sv"]]
        .rename(columns={"ybar_sv": "ybar_original"})
    )

    reworded_df = (
        g_sv[g_sv["variant"] != base_variant]
        .groupby(["policy_domain", "base_id"], sort=False)["ybar_sv"]
        .agg(
            ybar_reworded="mean",
            n_rules_used="size"
        )
        .reset_index()
    )

    pair = original_df.merge(
        reworded_df,
        on=["policy_domain", "base_id"],
        how="inner",
        validate="1:1"
    )

    if complete_only:
        pair = pair[pair["n_rules_used"] == expected_n_rules].copy()

    rows = []
    for domain, g in pair.groupby("policy_domain", sort=False):
        n_statements = int(g.shape[0])

        if n_statements == 0:
            rows.append({
                "policy_domain": domain,
                "AS": float("nan"),
                "PS": float("nan"),
                "n_statements": 0,
                "avg_n_rules_used": float("nan")
            })
            continue

        diff = g["ybar_original"] - g["ybar_reworded"]
        as_domain = float(((diff * diff) / 2.0).mean())

        ybar_s = (g["ybar_original"] + g["ybar_reworded"]) / 2.0
        ps_domain = float(ybar_s.var(ddof=1)) if n_statements > 1 else 0.0

        rows.append({
            "policy_domain": domain,
            "AS": as_domain,
            "PS": ps_domain,
            "n_statements": n_statements,
            "avg_n_rules_used": float(g["n_rules_used"].mean())
        })

    return pd.DataFrame(rows)

def compute_mu_per_variant_domain(df_all: pd.DataFrame) -> pd.DataFrame:
    mu_prompt = (
        df_all.groupby(["variant", "policy_domain", "ID"], sort=False)["score"]
        .agg(MU_prompt=lambda x: float(x.var(ddof=1)) if x.size > 1 else 0.0)
        .reset_index()
    )

    mu_variant_domain = (
        mu_prompt.groupby(["variant", "policy_domain"], sort=False)["MU_prompt"]
        .agg(MU="mean", n_prompts="size")
        .reset_index()
    )
    return mu_variant_domain


if __name__ == "__main__":
    df_all = load_all(files, domain_map_path, REVERSING_VARIANTS)

    print("=== Loaded variants ===")
    print(sorted(df_all["variant"].unique().tolist()))
    print()

    mu_variant_domain = compute_mu_per_variant_domain(df_all)
    print("=== MU per variant × policy_domain ===")
    for _, r in mu_variant_domain.iterrows():
        print(
            f"variant={r['variant']:<15}  "
            f"domain={r['policy_domain']:<25}  "
            f"MU={float(r['MU']):.6f}  "
            f"(n_prompts={int(r['n_prompts'])})"
        )

    print()

    mu_domain = compute_mu_by_domain_direct(df_all, include_original=True)
    asps_domain = compute_as_ps_by_domain_direct(
        df_all,
        base_variant="original",
        complete_only=False
    )

    print("=== Direct pooled MU by policy_domain ===")
    for _, r in mu_domain.iterrows():
        print(
            f"domain={r['policy_domain']:<25}  "
            f"MU={float(r['MU']):.6f}  "
            f"(n_prompt_cells={int(r['n_prompt_cells'])}, "
            f"n_variants={int(r['n_variants'])})"
        )

    print()
    print("=== Direct pooled AS / PS by policy_domain ===")
    for _, r in asps_domain.iterrows():
        print(
            f"domain={r['policy_domain']:<25}  "
            f"AS={float(r['AS']):.6f}  "
            f"PS={float(r['PS']):.6f}  "
            f"(n_statements={int(r['n_statements'])}, "
            f"avg_n_rules_used={float(r['avg_n_rules_used']):.2f})"
        )
