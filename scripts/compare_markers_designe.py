import pandas as pd
import os

# File paths
XLSX_STANZA_PATH = r"C:\Users\Philo\ExpressionEmotionnelle\results\xlsx\specificity_results_stanza\markers_stanza.csv"
XLSX_SPACY_PATH = r"C:\Users\Philo\ExpressionEmotionnelle\results\xlsx\specificity_results_spacy\markers_spacy.csv"

GLOZZ_STANZA_PATH = r"C:\Users\Philo\ExpressionEmotionnelle\results\glozz\specificity_results_stanza\markers.csv"
GLOZZ_SPACY_PATH = r"C:\Users\Philo\ExpressionEmotionnelle\results\glozz\specificity_results_spacy\markers.csv"

def load_and_filter_xlsx(file_path):
    """Load XLSX markers and filter for 'Désignée' mode (value 1.0)."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return set()
    
    # Try reading with low_memory=False to avoid DtypeWarnings
    df = pd.read_csv(file_path, low_memory=False)
    
    if 'Désignée' in df.columns:
        filtered = df[df['Désignée'] == 1.0]
        markers = set(m for m in filtered['marker_value'].dropna().astype(str).str.lower().str.strip() if len(m) > 2)
        return markers
    else:
        print(f"'Désignée' column missing in {file_path}")
        return set()

def load_and_filter_glozz(file_path):
    """Load Glozz markers and filter for 'Designee' mode."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return set()
        
    df = pd.read_csv(file_path, low_memory=False)
    
    if 'mode' in df.columns:
        # Glozz uses 'Designee' or potentially other variations
        filtered = df[df['mode'].str.lower().str.strip() == 'designee']
        markers = set(m for m in filtered['marker_value'].dropna().astype(str).str.lower().str.strip() if len(m) > 2)
        return markers
    else:
        print(f"'mode' column missing in {file_path}")
        return set()

def compare_sets(set1, name1, set2, name2, context_name):
    """Compare two sets of markers and print the results."""
    print("=" * 60)
    print(f"{context_name}")
    print("=" * 60)
    print(f"Marqueurs uniques dans {name1} : {len(set1)}")
    print(f"Marqueurs uniques dans {name2} : {len(set2)}")
    
    common = set1.intersection(set2)
    only_in_1 = set1 - set2
    only_in_2 = set2 - set1
    
    print(f"Marqueurs communs (recoupements) : {len(common)}")
    print(f"Marqueurs exclusifs à {name1} : {len(only_in_1)}")
    print(f"Marqueurs exclusifs à {name2} : {len(only_in_2)}")
    
    if len(common) > 0:
        print(f"\n[=] Exemples de recoupements (communs aux deux corpus) (max 30):")
        print(", ".join(list(common)[:30]))
        
    if len(only_in_1) > 0:
        print(f"\n[+] Exemples de marqueurs UNIQUEMENT dans {name1} (max 30):")
        print(", ".join(list(only_in_1)[:30]))
        
    if len(only_in_2) > 0:
        print(f"\n[+] Exemples de marqueurs UNIQUEMENT dans {name2} (max 30):")
        print(", ".join(list(only_in_2)[:30]))
    print("\n")

def main():
    print("Chargement et filtrage des données XLSX...")
    xlsx_stanza = load_and_filter_xlsx(XLSX_STANZA_PATH)
    xlsx_spacy = load_and_filter_xlsx(XLSX_SPACY_PATH)
    
    print("Chargement et filtrage des données Glozz...")
    glozz_stanza = load_and_filter_glozz(GLOZZ_STANZA_PATH)
    glozz_spacy = load_and_filter_glozz(GLOZZ_SPACY_PATH)
    
    # Union des marqueurs Stanza et SpaCy par corpus
    xlsx_markers = xlsx_stanza.union(xlsx_spacy)
    glozz_markers = glozz_stanza.union(glozz_spacy)
    
    # Comparaison entre le corpus XLSX et le corpus Glozz
    compare_sets(xlsx_markers, "Corpus XLSX", glozz_markers, "Corpus Glozz", "Comparaison XLSX vs Glozz (Mode: Désigné)")

if __name__ == "__main__":
    main()
