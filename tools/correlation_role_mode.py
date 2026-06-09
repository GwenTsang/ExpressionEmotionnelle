import pandas as pd
import scipy.stats as stats
import os

def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(BASE_DIR, 'data', 'raw', 'xlsx', 'CyberAdoAgg_gold_global_total_latest.xlsx')
    results_dir = os.path.join(BASE_DIR, 'results', 'correlation')
    os.makedirs(results_dir, exist_ok=True)
    df = pd.read_excel(file_path)

    # Colonnes
    col_role = 'ROLE'
    cols_modes = ["Suggeree", "Montree", "Comportementale", "Designee"]

    # Nettoyage
    df_clean = df.dropna(subset=[col_role] + cols_modes).copy()
    df_clean[col_role] = df_clean[col_role].astype(str).str.strip()
    
    print("Corrélation entre le rôle et le mode d'expression dans CyberAggAdo ?")

    # Liste pour stocker les résultats du test du Chi-deux
    chi2_results_summary = []
    saved_files = []

    for mode in cols_modes:
        print(f"\n Corrélation entre '{col_role}' et '{mode}'")
        
        # Création du tableau de contingence (cross-tabulation)
        contingency_table = pd.crosstab(df_clean[col_role], df_clean[mode])
        print("\nTableau de contingence :")
        print(contingency_table)

        # 2. Sauvegarde du tableau de contingence en CSV
        ct_csv_path = os.path.join(results_dir, f"contingence_role_{mode.lower()}.csv")
        contingency_table.to_csv(ct_csv_path)
        saved_files.append(ct_csv_path)

        # Test du Chi-deux
        try:
            chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
            
            print(f"\nRésultats du test du Chi-deux :")
            print(f"Statistique Chi-deux (χ²) = {chi2:.4f}")
            print(f"P-value = {p:.4e}")
            print(f"Degrés de liberté = {dof}")

            # Interprétation
            is_significant = p < 0.05
            if is_significant:
                print(f"Il y a une corrélation/association significative entre '{col_role}' et '{mode}'.")
            else:
                print(f"Il n'y a PAS de corrélation significative entre '{col_role}' et '{mode}'.")
                
            # Ajout des résultats à la liste pour le CSV global
            chi2_results_summary.append({
                "Mode_Expression": mode,
                "Chi2_Statistique": round(chi2, 4),
                "P_value": p,
                "Degres_Liberte": dof,
                "Significatif": "Oui" if is_significant else "Non"
            })
            
        except Exception as e:
            print(f"Erreur lors du calcul du Chi-deux pour {mode}: {e}")

    # 3. Sauvegarde du résumé global des tests du Chi-deux en CSV
    if chi2_results_summary:
        df_chi2_summary = pd.DataFrame(chi2_results_summary)
        summary_csv_path = os.path.join(results_dir, "correlation_chi2_role_modes.csv")
        df_chi2_summary.to_csv(summary_csv_path, index=False)
        saved_files.append(summary_csv_path)

    # Affichage des fichiers sauvegardés à la fin du script
    print(f"\n Résultats sauvegardés dans :")
    for file in saved_files:
        print(f"    - {file}")

if __name__ == "__main__":
    main()