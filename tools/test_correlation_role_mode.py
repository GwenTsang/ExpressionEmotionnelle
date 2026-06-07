import pandas as pd
import scipy.stats as stats
import os

def main():
    file_path = '/workspaces/codespaces-blank/ExpressionEmotionnelle/data/raw/xlsx/CyberAdoAgg_gold_global_total_latest.xlsx'

    print(f"Chargement du fichier: {file_path}")
    df = pd.read_excel(file_path)

    # Colonnes
    col_role = 'ROLE'
    cols_modes = ["Suggeree", "Montree", "Comportementale", "Designee"]

    # Nettoyage
    df_clean = df.dropna(subset=[col_role] + cols_modes).copy()
    df_clean[col_role] = df_clean[col_role].astype(str).str.strip()
    
    print("Corrélation entre le rôle et le mode d'expression dans CyberAggAdo ?")

    for mode in cols_modes:
        print(f"\n Corrélation entre '{col_role}' et '{mode}'")
        
        # Création du tableau de contingence (cross-tabulation)
        contingency_table = pd.crosstab(df_clean[col_role], df_clean[mode])
        print("\nTableau de contingence :")
        print(contingency_table)

        # Test du Chi-deux
        try:
            chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
            
            print(f"\nRésultats du test du Chi-deux :")
            print(f"Statistique Chi-deux (χ²) = {chi2:.4f}")
            print(f"P-value = {p:.4e}")
            print(f"Degrés de liberté = {dof}")

            # Interprétation
            if p < 0.05:
                print(f"Il y a une corrélation/association significative entre '{col_role}' et '{mode}'.")
            else:
                print(f"Il n'y a PAS de corrélation significative entre '{col_role}' et '{mode}'.")
        except Exception as e:
            print(f"Erreur lors du calcul du Chi-deux")

if __name__ == "__main__":
    main()
