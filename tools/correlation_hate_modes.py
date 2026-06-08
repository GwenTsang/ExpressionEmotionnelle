import os
import pandas as pd
import numpy as np
import scipy.stats as stats

def main():
    # Define paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(BASE_DIR, 'data', 'raw', 'xlsx', 'CyberAdoAgg_gold_global_total_latest.xlsx')
    results_dir = os.path.join(BASE_DIR, 'results', 'correlation')
    os.makedirs(results_dir, exist_ok=True)
    
    # Read the Excel file
    print(f"[+] Lecture du fichier : {file_path}")
    df = pd.read_excel(file_path)
    
    col_hate = 'HATE'
    cols_modes = ["Suggeree", "Montree", "Comportementale", "Designee"]
    
    # Check that the columns exist in the DataFrame
    missing_cols = [col for col in cols_modes if col not in df.columns]
    if missing_cols:
        print(f"Warning: Les colonnes de modes suivantes sont absentes : {missing_cols}")
        cols_modes = [col for col in cols_modes if col in df.columns]
        
    if col_hate not in df.columns:
        print(f"Erreur: La colonne '{col_hate}' est absente du fichier.")
        return
        
    # Clean the HATE column
    df_clean = df.dropna(subset=[col_hate]).copy()
    df_clean[col_hate] = df_clean[col_hate].astype(str).str.strip()
    df_clean = df_clean[df_clean[col_hate] != '']
    
    # Convert modes to binary integers
    for mode in cols_modes:
        df_clean[mode] = pd.to_numeric(df_clean[mode], errors='coerce').fillna(0).astype(int)
        
    # Configure pandas display
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    hate_categories = sorted(df_clean[col_hate].unique())
    n_total = len(df_clean)
    
    print(f"[+] Nombre total de lignes valides : {n_total}")
    print(f"[+] Catégories HATE détectées : {hate_categories}\n")
    
    # --- 1. Distribution matrix (Proportions of presence) ---
    mean_matrix = []
    for cat in hate_categories:
        cat_df = df_clean[df_clean[col_hate] == cat]
        mean_row = {"HATE": cat}
        for mode in cols_modes:
            mean_row[mode] = cat_df[mode].mean()
        mean_matrix.append(mean_row)
        
    df_means = pd.DataFrame(mean_matrix).set_index("HATE")
    means_csv_path = os.path.join(results_dir, "distribution_modes_par_hate.csv")
    df_means.to_csv(means_csv_path, float_format='%.4f')
    
    print("=" * 100)
    print(" 1. DISTRIBUTION DES MODES PAR CATEGORIE DE HATE (Proportion de présence)")
    print("=" * 100)
    print((df_means * 100).round(2).astype(str) + "%")
    print()
    
    # --- 2. Pearson Correlation & P-values ---
    corr_matrix = []
    pval_matrix = []
    
    for cat in hate_categories:
        cat_dummy = (df_clean[col_hate] == cat).astype(int)
        corr_row = {"HATE": cat}
        pval_row = {"HATE": cat}
        
        for mode in cols_modes:
            if cat_dummy.std() == 0 or df_clean[mode].std() == 0:
                r_val, p_val = np.nan, np.nan
            else:
                r_val, p_val = stats.pearsonr(cat_dummy, df_clean[mode])
            corr_row[mode] = r_val
            pval_row[mode] = p_val
            
        corr_matrix.append(corr_row)
        pval_matrix.append(pval_row)
        
    df_corrs = pd.DataFrame(corr_matrix).set_index("HATE").round(4)
    df_pvals = pd.DataFrame(pval_matrix).set_index("HATE")
    
    corrs_csv_path = os.path.join(results_dir, "correlation_pearson_hate_modes.csv")
    pvals_csv_path = os.path.join(results_dir, "correlation_pvalue_hate_modes.csv")
    
    df_corrs.to_csv(corrs_csv_path, float_format='%.4f')
    df_pvals.to_csv(pvals_csv_path, float_format='%.4f')
    
    print("=" * 100)
    print(" 2. COEFFICIENTS DE CORRELATION (Pearson r / Coefficient Phi) PAR PAIR HATE-MODE")
    print("=" * 100)
    print(df_corrs.round(2))
    print()
    
    print("=" * 100)
    print(" 3. SIGNIFICATIVITE (p-values) DES CORRELATIONS")
    print("=" * 100)
    print(df_pvals.round(4))
    print()
    
    # --- 3. Chi-Square & Cramér's V (Overall Association) ---
    chi2_results = []
    for mode in cols_modes:
        contingency_table = pd.crosstab(df_clean[col_hate], df_clean[mode])
        n = contingency_table.sum().sum()
        
        if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
            chi2, p_val, dof, cramers_v = np.nan, np.nan, np.nan, np.nan
        else:
            try:
                chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)
                # Cramer's V for C=2, R=3 is sqrt(chi2 / (n * min(R-1, C-1))) = sqrt(chi2 / n)
                cramers_v = np.sqrt(chi2 / n)
            except Exception:
                chi2, p_val, dof, cramers_v = np.nan, np.nan, np.nan, np.nan
                
        chi2_results.append({
            "Mode": mode,
            "Chi2": chi2,
            "p-value": p_val,
            "dof": dof,
            "Cramers_V": cramers_v,
            "Significative": "Oui (p < 0.05)" if (pd.notna(p_val) and p_val < 0.05) else "Non"
        })
        
    df_chi2 = pd.DataFrame(chi2_results).set_index("Mode")
    chi2_csv_path = os.path.join(results_dir, "association_chisquare_cramersv_hate_modes.csv")
    df_chi2.to_csv(chi2_csv_path, float_format='%.4f')
    
    print("=" * 100)
    print(" 4. TEST DE CHI-DEUX ET V DE CRAMER (Association Globale HATE - Mode)")
    print("=" * 100)
    print(df_chi2.round(4))
    print()
    
    print("[+] Fichiers générés avec succès :")
    print(f"    - Distribution: {means_csv_path}")
    print(f"    - Corrélation Pearson (r / Phi): {corrs_csv_path}")
    print(f"    - P-values Pearson: {pvals_csv_path}")
    print(f"    - Chi-deux & Cramér's V: {chi2_csv_path}")

if __name__ == "__main__":
    main()
