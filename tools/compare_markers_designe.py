import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MARKERS_PATH = os.path.join(BASE_DIR, "results", "simplesitemo", "markers.csv")

def load_and_filter_markers(file_path):
    """Load unified markers and split by corpus for 'Désignée' mode."""
    cyber_markers = set()
    ttk_markers = set()

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return cyber_markers, ttk_markers
        
    df = pd.read_csv(file_path, low_memory=False)
    
    if 'mode' in df.columns and 'source_file' in df.columns:
        filtered = df[df['mode'].str.lower().str.strip() == 'désignée']
        
        for _, row in filtered.iterrows():
            marker = str(row.get('marker_value', '')).strip().lower()
            if not marker or marker == 'nan' or len(marker) <= 2:
                continue
                
            source = str(row.get('source_file', '')).strip()
            if source == 'CyberAggAdo':
                cyber_markers.add(marker)
            else:
                ttk_markers.add(marker)
                
    else:
        print(f"Required columns ('mode' or 'source_file') missing in {file_path}")
        
    return cyber_markers, ttk_markers

def compare_sets(set1, name1, set2, name2, context_name):
    """Compare two sets of markers and print the results."""
    print("=" * 60)
    print(f"{context_name}")
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
    print("Chargement et filtrage des données unifiées...")
    cyber_markers, ttk_markers = load_and_filter_markers(MARKERS_PATH)
    
    # Comparaison entre le corpus CyberAggAdoLarge et le corpus TextToKids
    compare_sets(cyber_markers, "Corpus CyberAggAdoLarge", ttk_markers, "Corpus TextToKids", "Comparaison CyberAggAdoLarge vs TextToKids (Mode: Désigné)")

if __name__ == "__main__":
    main()
