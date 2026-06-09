import os
import pandas as pd
import numpy as np
import scipy.stats as stats

def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(BASE_DIR, 'data', 'raw', 'xlsx', 'CyberAdoAgg_gold_global_total_latest.xlsx')
    results_dir = os.path.join(BASE_DIR, 'results', 'correlation')
    os.makedirs(results_dir, exist_ok=True)
    df = pd.read_excel(file_path)
    
    col_role = 'ROLE'
    EMOTION_LABELS = [
        "Admiration", "Autre", "Colere", "Culpabilite", "Degout",
        "Embarras", "Fierte", "Jalousie", "Joie", "Peur",
        "Surprise", "Tristesse", "Autre"
    ]
    
    # Dédoublonner sans préserver l'ordre
    unique_emotions = list(set(EMOTION_LABELS))
            
    # Vérification des colonnes dans le fichier Excel
    missing_cols = [col for col in unique_emotions if col not in df.columns]
    if missing_cols:
        print(f"Attention : Les colonnes suivantes sont absentes du fichier Excel : {missing_cols}")
        unique_emotions = [col for col in unique_emotions if col in df.columns]
        
    if col_role not in df.columns:
        print(f"Erreur : La colonne '{col_role}' est absente du fichier.")
        return
        
    # Nettoyage de la colonne ROLE (suppression des valeurs manquantes, suppression des espaces inutiles)
    df_clean = df.dropna(subset=[col_role]).copy()
    df_clean[col_role] = df_clean[col_role].astype(str).str.strip()
    
    # S'assurer que les rôles vides après strip sont exclus
    df_clean = df_clean[df_clean[col_role] != '']
    
    # Conversion et nettoyage des colonnes d'émotions en entiers binaires (0 ou 1)
    for emo in unique_emotions:
        df_clean[emo] = pd.to_numeric(df_clean[emo], errors='coerce').fillna(0).astype(int)
        
    # Configurer l'affichage de pandas pour ne pas tronquer les colonnes
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    roles = sorted(df_clean[col_role].unique())
    print(f"Nombre total de lignes {len(df_clean)}")
    
    # 1. Distribution des émotions par rôle (proportions/pourcentages de présence)
    mean_matrix = []
    for role in roles:
        role_df = df_clean[df_clean[col_role] == role]
        mean_row = {"Role": role}
        for emo in unique_emotions:
            mean_row[emo] = role_df[emo].mean()
        mean_matrix.append(mean_row)
        
    df_means = pd.DataFrame(mean_matrix).set_index("Role")
    means_csv_path = os.path.join(results_dir, "distribution_emotions_par_role.csv")
    df_means.to_csv(means_csv_path, float_format='%.2f')
    
    print(" 1. DISTRIBUTION DES EMOTIONS PAR ROLE (Proportions de présence)")
    print("")
    print((df_means * 100).round(2).astype(str) + "%")
    
    # 2. Coefficients de corrélation (Pearson r / coefficient Phi) & p-values
    corr_matrix = []
    pval_matrix = []
    
    for role in roles:
        role_dummy = (df_clean[col_role] == role).astype(int)
        corr_row = {"Role": role}
        pval_row = {"Role": role}
        
        for emo in unique_emotions:
            # Si une colonne d'émotions ou le dummy de rôle est constant, le coefficient n'est pas défini.
            if role_dummy.std() == 0 or df_clean[emo].std() == 0:
                r_val, p_val = np.nan, np.nan
            else:
                r_val, p_val = stats.pearsonr(role_dummy, df_clean[emo])
            corr_row[emo] = r_val
            pval_row[emo] = p_val
            
        corr_matrix.append(corr_row)
        pval_matrix.append(pval_row)
        
    # Arrondi des coefficients de corrélation à 2 décimales pour l'affichage et l'export
    df_corrs = pd.DataFrame(corr_matrix).set_index("Role").round(2)
    df_pvals = pd.DataFrame(pval_matrix).set_index("Role")
    
    corrs_csv_path = os.path.join(results_dir, "correlation_pearson_role_emotions.csv")
    pvals_csv_path = os.path.join(results_dir, "correlation_pvalue_role_emotions.csv")
    
    df_corrs.to_csv(corrs_csv_path, float_format='%.2f')
    df_pvals.to_csv(pvals_csv_path, float_format='%.2f')
    print("")
    print(" 2. COEFFICIENTS DE CORRELATION (Pearson r / Coefficient Phi) PAR PAIR ROLE-EMOTION")
    print("")
    print(df_corrs) # arrondi à 2 décimales
    print("")
    print(" 3. VALEURS DE SIGNIFICATIVITE (p-values) PAR PAIR ROLE-EMOTION")
    print(df_pvals.round(4))
    
    print(f"\n Résultats sauvegardés dans :")
    print(f"    - {means_csv_path}")
    print(f"    - {corrs_csv_path}")
    print(f"    - {pvals_csv_path}")

if __name__ == "__main__":
    main()