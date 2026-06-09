"""Script de corrélation entre features

Teste les corrélations entre deux groupes de colonnes binaires (mode pairwise)
et/ou l'association globale entre une variable catégorielle et des colonnes
binaires (mode global).
"""

import argparse
import os

import numpy as np
import pandas as pd
import scipy.stats as stats

# Groupes résolus par préfixe (colonnes one-hot commençant par PREFIX_)
PREFIX_GROUPS = ["ROLE", "HATE", "INTENTION", "VERBAL_ABUSE"]

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

def cramers_v(chi2, n, table_shape):
    """Calcule le V de Cramér (formule correcte)."""
    rows, cols = table_shape
    denominator = n * min(rows - 1, cols - 1)
    if denominator == 0:
        return np.nan
    return np.sqrt(chi2 / denominator)
    # Formule simplifiee (correcte uniquement pour tables 2x2) :
    # return np.sqrt(chi2 / n)


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

def run_pairwise(df, cols_a, cols_b, group_a, group_b, output_dir):
    """Corrélation pairwise entre colonnes binaires."""
    rows = []
    for col_a in cols_a:
        for col_b in cols_b:
            mask = df[col_a].notna() & df[col_b].notna()
            sub = df.loc[mask, [col_a, col_b]]
            n = len(sub)

            if n < 2 or sub[col_a].std() == 0 or sub[col_b].std() == 0:
                rows.append({
                    "col_a": col_a, "col_b": col_b,
                    "phi": np.nan, "chi2": np.nan,
                    "p_value": np.nan, "cramers_v": np.nan, "n": n,
                })
                continue

            # Phi (= Pearson r pour binaires)
            phi, _ = stats.pearsonr(sub[col_a], sub[col_b])

            # Chi-square 2×2
            contingency = pd.crosstab(sub[col_a], sub[col_b])
            if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                chi2_val, p_val, cv = np.nan, np.nan, np.nan
            else:
                chi2_val, p_val, dof, _ = stats.chi2_contingency(contingency)
                cv = cramers_v(chi2_val, n, contingency.shape)

            rows.append({
                "col_a": col_a, "col_b": col_b,
                "phi": phi, "chi2": chi2_val,
                "p_value": p_val, "cramers_v": cv, "n": n,
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

    # Sauvegarde
    ga = group_a.lower()
    gb = group_b.lower()

    pairwise_path = os.path.join(output_dir, f"pairwise_{ga}_{gb}.csv")
    dist_path = os.path.join(output_dir, f"distribution_{ga}_{gb}.csv")

    format_p_values_for_csv(df_result).to_csv(
        pairwise_path, index=False, float_format="%.2f"
    )
    df_dist.to_csv(dist_path, float_format="%.2f")

    return df_result, [pairwise_path, dist_path]


# ---------------------------------------------------------------------------
# Mode global
# ---------------------------------------------------------------------------

def run_global(df, group_a, group_b, cols_b, output_dir):
    """Chi-square global et V de Cramér (catégorielle vs binaire)."""
    # Déterminer quel groupe est catégoriel
    if is_prefix_group(group_a):
        cat_col = group_a.upper()
        binary_cols = cols_b
    elif is_prefix_group(group_b):
        cat_col = group_b.upper()
        binary_cols = resolve_group(group_a, df.columns)
    else:
        print(f"[!] Mode global ignoré : aucun des deux groupes "
              f"({group_a}, {group_b}) n'est une variable catégorielle.")
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
            chi2_val, p_val, dof, cv = np.nan, np.nan, np.nan, np.nan
        else:
            try:
                chi2_val, p_val, dof, _ = stats.chi2_contingency(contingency)
                cv = cramers_v(chi2_val, n, contingency.shape)
            except Exception:
                chi2_val, p_val, dof, cv = np.nan, np.nan, np.nan, np.nan

        rows.append({
            "categorical_var": cat_col,
            "binary_var": bin_col,
            "chi2": chi2_val,
            "p_value": p_val,
            "cramers_v": cv,
            "df": dof,
            "n": n,
        })

    df_result = pd.DataFrame(rows)

    ga = group_a.lower()
    gb = group_b.lower()
    global_path = os.path.join(output_dir, f"global_{ga}_{gb}.csv")
    format_p_values_for_csv(df_result).to_csv(
        global_path, index=False, float_format="%.2f"
    )

    return df_result, [global_path]


# ---------------------------------------------------------------------------
# Affichage console
# ---------------------------------------------------------------------------

def print_summary(group_a, group_b, mode, df_pairwise, df_global, saved_files):
    """Affiche un résumé concis des résultats dans le terminal."""
    print("")
    print(f" Corrélation {group_a} × {group_b} (mode: {mode})")
    print("")

    if df_pairwise is not None and len(df_pairwise) > 0:
        n_total = len(df_pairwise)
        n_sig = df_pairwise["p_value"].dropna().lt(0.05).sum()
        print(f"\nPairwise : {n_total} paires testées")

        # Top 5 par |Phi|
        df_sorted = df_pairwise.dropna(subset=["phi"]).copy()
        df_sorted["abs_phi"] = df_sorted["phi"].abs()
        top5 = df_sorted.nlargest(5, "abs_phi")
        if len(top5) > 0:
            print("Top 5 des corrélations :")
            for i, (_, row) in enumerate(top5.iterrows(), 1):
                p_str = f"{row['p_value']:.2e}" if row["p_value"] < 0.01 else f"{row['p_value']:.2f}"
                print(f"  {i}. {row['col_a']} × {row['col_b']}"
                      f"Corrélation Phi={row['phi']:.2f}  p={p_str}")

    if df_global is not None and len(df_global) > 0:
        n_total = len(df_global)
        n_sig = df_global["p_value"].dropna().lt(0.05).sum()
        print(f"\nGlobal : {n_total} tests effectués, {n_sig} significatifs (p < 0.05)")

        # ceci n'est pas très informatif, je trouve
        df_sorted = df_global.dropna(subset=["cramers_v"]).copy()
        top5 = df_sorted.nlargest(5, "cramers_v")
        if len(top5) > 0:
            print("Top 5 associations (par V de Cramér) :")
            for i, (_, row) in enumerate(top5.iterrows(), 1):
                p_str = f"{row['p_value']:.2e}" if row["p_value"] < 0.01 else f"{row['p_value']:.2f}"
                print(f"  {i}. {row['categorical_var']} × {row['binary_var']}"
                      f"    V={row['cramers_v']:.2f}  p={p_str}")

    if saved_files:
        print(f"\nFichiers sauvegardés :")
        for f in saved_files:
            print(f"  → {f}")
    print()


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def parse_args():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_input = os.path.join(
        base_dir, "data", "processed",
        "correlation_binary_dataset.xlsx",
    )
    default_output_dir = os.path.join(base_dir, "results", "correlation")

    parser = argparse.ArgumentParser(
        description=(
            "Script unifié de corrélation. Compare deux groupes de colonnes.\n"
            f"Groupes valides : {', '.join(ALL_GROUP_NAMES)}"
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
        help="Mode d'analyse (défaut: all).",
    )
    parser.add_argument(
        "--input", default=default_input,
        help="Fichier Excel préparé (défaut: data/processed/correlation_binary_dataset.xlsx).",
    )
    parser.add_argument(
        "--output-dir", default=default_output_dir,
        help="Dossier de sortie des CSV (défaut: results/correlation/).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    # Lecture du dataset préparé
    df = pd.read_excel(args.input)

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
            df, cols_a, cols_b, group_a, group_b, args.output_dir,
        )
        saved_files.extend(files)

    if mode in ("global", "all"):
        df_global, files = run_global(
            df, group_a, group_b, cols_b, args.output_dir,
        )
        saved_files.extend(files)

    # Résumé
    print_summary(group_a, group_b, mode, df_pairwise, df_global, saved_files)

if __name__ == "__main__":
    main()
