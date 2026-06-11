import argparse
import os

import numpy as np
import pandas as pd
import scipy.stats as stats

# Groupes résolus par préfixe (colonnes one-hot commençant par PREFIX_)
PREFIX_GROUPS = ["ROLE", "TARGET", "HATE", "INTENTION", "VERBAL_ABUSE", "CONTEXT"]

# Groupes résolus par liste fixe
FIXED_GROUPS = {
    "EMOTIONS": [
        "Admiration", "Autre", "Colere", "Culpabilite", "Degout",
        "Embarras", "Fierte", "Jalousie", "Joie", "Peur",
        "Surprise", "Tristesse",
    ],
    "MODES": ["Suggeree", "Montree", "Comportementale", "Designee"],
}

ALL_GROUP_NAMES = PREFIX_GROUPS + list(FIXED_GROUPS.keys())


# ---------------------------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------------------------

def resolve_group(group_name, df_columns):
    """Résout un nom de groupe en liste de colonnes du DataFrame."""
    name = group_name.upper()
    if name in FIXED_GROUPS:
        cols = [c for c in FIXED_GROUPS[name] if c in df_columns]
        if not cols:
            raise ValueError(
                f"Aucune colonne du groupe '{name}' trouvée dans le dataset."
            )
        return cols

    if name in PREFIX_GROUPS:
        prefix = f"{name}_"
        cols = [c for c in df_columns if c.startswith(prefix)]
        if not cols:
            raise ValueError(
                f"Aucune colonne avec le préfixe '{prefix}' trouvée dans le dataset."
            )
        return sorted(cols)

    raise ValueError(
        f"Groupe inconnu : '{group_name}'. "
        f"Groupes valides : {', '.join(ALL_GROUP_NAMES)}"
    )


def is_prefix_group(group_name):
    """Vérifie si un groupe est un groupe à préfixe (variable catégorielle)."""
    return group_name.upper() in PREFIX_GROUPS


def format_p_values_for_csv(df):
    """Formate uniquement les p-values pour l'export CSV."""
    df_export = df.copy()
    if "p_value" in df_export.columns:
        df_export["p_value"] = df_export["p_value"].map(
            lambda value: "" if pd.isna(value) else f"{value:.4f}"
        )
    return df_export


# ---------------------------------------------------------------------------
# Mode pairwise
# ---------------------------------------------------------------------------

def run_pairwise(df, cols_a, cols_b, group_a, group_b, output_dir, save_csv):
    """Corrélation pairwise entre colonnes binaires."""
    rows = []
    for col_a in cols_a:
        for col_b in cols_b:
            mask = df[col_a].notna() & df[col_b].notna()
            sub = df.loc[mask, [col_a, col_b]]
            n = len(sub)
            n_a = int((sub[col_a] == 1).sum())
            n_b = int((sub[col_b] == 1).sum())
            n_intersect = int(((sub[col_a] == 1) & (sub[col_b] == 1)).sum())

            if n < 2 or sub[col_a].std() == 0 or sub[col_b].std() == 0:
                rows.append({
                    "col_a": col_a, "col_b": col_b,
                    "phi": np.nan, "chi2": np.nan,
                    "p_value": np.nan, "n": n,
                    "n_a": n_a, "n_b": n_b, "n_intersect": n_intersect,
                })
                continue

            # Phi (= Pearson r pour binaires)
            phi, _ = stats.pearsonr(sub[col_a], sub[col_b])

            # Chi-square 2×2
            contingency = pd.crosstab(sub[col_a], sub[col_b])
            if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                chi2_val, p_val = np.nan, np.nan
            else:
                chi2_val, p_val, dof, _ = stats.chi2_contingency(contingency)

            rows.append({
                "col_a": col_a, "col_b": col_b,
                "phi": phi, "chi2": chi2_val,
                "p_value": p_val, "n": n,
                "n_a": n_a, "n_b": n_b, "n_intersect": n_intersect,
            })

    df_result = pd.DataFrame(rows)

    # Distribution
    dist_rows = []
    for col_a in cols_a:
        mask_a = df[col_a] == 1
        row = {"variable": col_a, "n": int(mask_a.sum())}
        for col_b in cols_b:
            row[col_b] = df.loc[mask_a, col_b].mean() if mask_a.sum() > 0 else np.nan
        dist_rows.append(row)

    df_dist = pd.DataFrame(dist_rows).set_index("variable")

    # Sauvegarde conditionnelle
    saved_files = []
    if save_csv:
        ga = group_a.lower()
        gb = group_b.lower()

        pairwise_path = os.path.join(output_dir, f"pairwise_{ga}_{gb}.csv")
        dist_path = os.path.join(output_dir, f"distribution_{ga}_{gb}.csv")

        format_p_values_for_csv(df_result).to_csv(
            pairwise_path, index=False, float_format="%.2f"
        )
        df_dist.to_csv(dist_path, float_format="%.2f")
        saved_files = [pairwise_path, dist_path]

    return df_result, saved_files


# ---------------------------------------------------------------------------
# Mode global
# ---------------------------------------------------------------------------

def run_global(df, group_a, group_b, cols_b, output_dir, save_csv):
    """Chi-square global (catégorielle vs binaire)."""
    # Déterminer quel groupe est catégoriel
    if is_prefix_group(group_a):
        cat_col = group_a.upper()
        binary_cols = cols_b
    elif is_prefix_group(group_b):
        cat_col = group_b.upper()
        binary_cols = resolve_group(group_a, df.columns)
    else:
        return None, []

    if cat_col not in df.columns:
        print(f"[!] Colonne catégorielle '{cat_col}' absente du dataset.")
        return None, []

    rows = []
    df_valid = df.dropna(subset=[cat_col]).copy()
    df_valid[cat_col] = df_valid[cat_col].astype(str).str.strip()
    df_valid = df_valid[df_valid[cat_col] != ""]

    for bin_col in binary_cols:
        contingency = pd.crosstab(df_valid[cat_col], df_valid[bin_col])
        n = contingency.to_numpy().sum()

        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            chi2_val, p_val, dof = np.nan, np.nan, np.nan
        else:
            try:
                chi2_val, p_val, dof, _ = stats.chi2_contingency(contingency)
            except Exception:
                chi2_val, p_val, dof = np.nan, np.nan, np.nan

        rows.append({
            "categorical_var": cat_col,
            "binary_var": bin_col,
            "chi2": chi2_val,
            "p_value": p_val,
            "df": dof,
            "n": n,
        })

    df_result = pd.DataFrame(rows)

    saved_files = []
    if save_csv:
        ga = group_a.lower()
        gb = group_b.lower()
        global_path = os.path.join(output_dir, f"global_{ga}_{gb}.csv")
        format_p_values_for_csv(df_result).to_csv(
            global_path, index=False, float_format="%.2f"
        )
        saved_files = [global_path]

    return df_result, saved_files


# Affichage console

def print_group_sizes(df, group_name, cols):
    """Affiche la taille d'échantillon pour chaque colonne d'un groupe."""
    print(f"  Effectifs {group_name} :")
    for col in cols:
        n_valid = int(df[col].notna().sum())
        n_pos = int((df[col] == 1).sum())
        print(f"    {col:30s}  n=1: {n_pos:4d} / {n_valid}")


def print_summary(df, group_a, group_b, cols_a, cols_b, mode,
                  df_pairwise, df_global, saved_files):
    print(f"\n Corrélation {group_a} × {group_b} (mode: {mode})")
    print()

    # Effectifs par groupe
    print_group_sizes(df, group_a, cols_a)
    print()
    print_group_sizes(df, group_b, cols_b)
    print()

    if df_pairwise is not None and len(df_pairwise) > 0:
        n_total = len(df_pairwise)
        n_sig = df_pairwise["p_value"].dropna().lt(0.05).sum()

        # Top 5
        df_sorted = df_pairwise.dropna(subset=["phi"]).copy()
        df_sorted["abs_phi"] = df_sorted["phi"].abs()
        top5 = df_sorted.nlargest(5, "abs_phi")
        if len(top5) > 0:
            print(f"Top 5 des corrélations parmi {n_total} paires testées :")
            for i, (_, row) in enumerate(top5.iterrows(), 1):
                p_str = f"{row['p_value']:.2e}" if row["p_value"] < 0.01 else f"{row['p_value']:.2f}"
                print(f"  {i}. {row['col_a']} × {row['col_b']}"
                      f"          Phi={row['phi']:.2f}  p={p_str}"
                      f"  n_a={int(row['n_a'])}  n_b={int(row['n_b'])}"
                      f"  n(A∩B)={int(row['n_intersect'])}")

    if df_global is not None and len(df_global) > 0:
        n_total = len(df_global)
        n_sig = df_global["p_value"].dropna().lt(0.05).sum()
        print(f"\nGlobal : {n_total} tests effectués, {n_sig} significatifs (p < 0.05)")

        df_sorted = df_global.dropna(subset=["chi2"]).copy()
        top5 = df_sorted.nlargest(5, "chi2")
        if len(top5) > 0:
            print("Top 5 associations (par chi2) :")
            for i, (_, row) in enumerate(top5.iterrows(), 1):
                p_str = f"{row['p_value']:.2e}" if row["p_value"] < 0.01 else f"{row['p_value']:.2f}"
                print(f"  {i}. {row['categorical_var']} × {row['binary_var']}"
                      f"    chi2={row['chi2']:.2f}  p={p_str}")

    if saved_files:
        print(f"\nFichiers sauvegardés :")
        for f in saved_files:
            print(f"- {f}")
    print()


# Point d'entrée
def parse_args():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_output_dir = os.path.join(base_dir, "results", "correlation")

    parser = argparse.ArgumentParser(
        description=(
            f"Script de corrélation groupes valides : {', '.join(ALL_GROUP_NAMES)}"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "group_a",
        help="Premier groupe de colonnes (ex: ROLE, HATE, EMOTIONS).",
    )
    parser.add_argument(
        "group_b",
        help="Second groupe de colonnes (ex: EMOTIONS, MODES).",
    )
    parser.add_argument(
        "--mode",
        choices=["pairwise", "global", "all"],
        default="all",
    )
    parser.add_argument(
        "--output-dir", default=default_output_dir,
    )
    parser.add_argument(
        "--save-csv", action="store_true",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(
        base_dir, "data", "processed", "correlation_binary_dataset.xlsx"
    )
    df = pd.read_excel(input_path)

    # Résolution des groupes
    group_a = args.group_a.upper()
    group_b = args.group_b.upper()
    cols_a = resolve_group(group_a, df.columns)
    cols_b = resolve_group(group_b, df.columns)

    print(f"{group_a} → {cols_a}")
    print(f"{group_b} → {cols_b}")

    mode = args.mode
    df_pairwise = None
    df_global = None
    saved_files = []

    # Exécution
    if mode in ("pairwise", "all"):
        df_pairwise, files = run_pairwise(
            df, cols_a, cols_b, group_a, group_b, args.output_dir, args.save_csv,
        )
        saved_files.extend(files)

    if mode in ("global", "all"):
        df_global, files = run_global(
            df, group_a, group_b, cols_b, args.output_dir, args.save_csv,
        )
        saved_files.extend(files)

    # Résumé
    print_summary(df, group_a, group_b, cols_a, cols_b, mode,
                  df_pairwise, df_global, saved_files)

if __name__ == "__main__":
    main()
