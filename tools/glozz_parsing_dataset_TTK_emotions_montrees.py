import os
import re
import xml.etree.ElementTree as ET
import pandas as pd

# Liste des natures
NATURES = [
    "SAdj", "SAdv", "SN", "SPrep", "Proposition", "Conj. de coordination",
    "Conj. de subordination", "Dislocation droite", "Dislocation gauche",
    "Enonce averbal", "Enonce clive", "Enonce elliptique", "Enonce exclamatif",
    "Interjection", "Point d'exclamation", "Points de suspension",
    "Accumulation", "Autre"
]

def process_glozz_sentences_montree(base_dir, output_file="emotions_montrees.xlsx"):
    sentence_end_pattern = re.compile(r'(?:(?<!\.)\.(?!\.)|[!?]+|\n+)(?=\s|$)')

    sentences_with_montree = []
    
    # Compteurs pour le diagnostic
    total_montree_found = 0
    undefined_nature_count = 0
    undefined_nature_details = {} 

    if not os.path.exists(base_dir):
        print(f"Erreur : Le dossier '{base_dir}' est introuvable.")
        return

    # Parcours du corpus
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            aa_dir = os.path.join(item_path, "aa")
            ac_dir = os.path.join(item_path, "ac")

            if os.path.isdir(aa_dir) and os.path.isdir(ac_dir):
                for filename in os.listdir(aa_dir):
                    if filename.endswith(".aa"):
                        base_name = filename[:-3]
                        aa_path = os.path.join(aa_dir, filename)
                        ac_path = os.path.join(ac_dir, base_name + ".ac")

                        if not os.path.exists(ac_path):
                            continue

                        with open(ac_path, 'r', encoding='utf-8') as f:
                            raw_text = f.read()

                        # 3. Découpage du texte en phrases
                        sentences = []
                        start_idx = 0
                        for match in sentence_end_pattern.finditer(raw_text):
                            end_idx = match.end()
                            sent_text = raw_text[start_idx:end_idx].strip()
                            if sent_text:
                                sentences.append({
                                    'text': sent_text.replace('\n', ' '),
                                    'start': start_idx,
                                    'end': end_idx,
                                    'emotions': [] # dictionnaires {'text': ..., 'nature': ...}
                                })
                            start_idx = end_idx

                        if start_idx < len(raw_text):
                            sent_text = raw_text[start_idx:].strip()
                            if sent_text:
                                sentences.append({'text': sent_text.replace('\n', ' '), 'start': start_idx, 'end': len(raw_text), 'emotions': []})

                        # Parsing XML
                        try:
                            tree = ET.parse(aa_path)
                            root = tree.getroot()
                            for unit in root.findall('.//unit'):
                                type_elem = unit.find('.//type')
                                if type_elem is not None and type_elem.text == 'SitEmo':
                                    mode_val = None
                                    nature_val = "Nature NON RENSEIGNÉE"
                                    
                                    for feature in unit.findall('.//feature'):
                                        if feature.get('name') == 'Mode':
                                            mode_val = feature.text.strip() if feature.text else ""
                                        elif feature.get('name') == 'Nature':
                                            nature_val = feature.text.strip() if feature.text else "Nature VIDE"

                                    if mode_val and mode_val.lower() in ['montree', 'montrée']:
                                        total_montree_found += 1
                                        
                                        if nature_val not in NATURES:
                                            undefined_nature_count += 1
                                            undefined_nature_details[nature_val] = undefined_nature_details.get(nature_val, 0) + 1

                                        start_node = unit.find('.//positioning/start/singlePosition')
                                        end_node = unit.find('.//positioning/end/singlePosition')
                                        
                                        if start_node is not None and end_node is not None:
                                            emo_start = int(start_node.get('index'))
                                            emo_end = int(end_node.get('index'))
                                            emo_text = raw_text[emo_start:emo_end].strip().replace('\n', ' ')
                                            
                                            # On stocke le texte et la nature séparément au lieu de les fusionner
                                            for s in sentences:
                                                if s['start'] <= emo_start <= s['end']:
                                                    s['emotions'].append({'text': emo_text, 'nature': nature_val})
                                                    break

                            for s in sentences:
                                if len(s['emotions']) > 0:
                                    sentences_with_montree.append(s)
                        except ET.ParseError as e:
                            print(f"Erreur XML dans le fichier {aa_path} : {e}")

    total_sentences = len(sentences_with_montree)
    print(f"Total phrases contenant au moins une émotion montrée : {total_sentences}")

    # Exportation vers Excel avec Pandas
    if total_sentences > 0:
        # Trouver le maximum de SitEmo dans une seule phrase
        max_emotions = max(len(s['emotions']) for s in sentences_with_montree)
        
        # Création dynamique des noms de colonnes
        columns = ["Phrase Complète"]
        for i in range(1, max_emotions + 1):
            columns.append(f"Émotion Montrée {i}")
        for i in range(1, max_emotions + 1):
            columns.append(f"nature_linguistique_segment_{i}")
            
        data_rows = []
        
        # Remplissage des lignes
        for s in sentences_with_montree:
            row = [s['text']]
            
            # Extraction des textes des émotions
            emo_texts = [emo['text'] for emo in s['emotions']]
            # Compléter avec des vides si moins d'émotions que le max
            emo_texts += [""] * (max_emotions - len(emo_texts))
            
            # Extraction des natures linguistiques
            emo_natures = [emo['nature'] for emo in s['emotions']]
            # Compléter avec des vides si moins d'émotions que le max
            emo_natures += [""] * (max_emotions - len(emo_natures))
            
            # Ajout au tableau (d'abord tous les textes, puis toutes les natures)
            row.extend(emo_texts)
            row.extend(emo_natures)
            
            data_rows.append(row)
            
        # Création du DataFrame et sauvegarde en Excel
        df = pd.DataFrame(data_rows, columns=columns)
        df.to_excel(output_file, index=False)
        print(f"\n=> Fichier Excel : {output_file}")

    # Affichage du diagnostic sur les Natures
    print("DIAGNOSTIC DES NATURES LINGUISTIQUES (pour le mode 'montree')")
    print(f"Total des SitEmo 'montree' trouvées : {total_montree_found}")
    print(f"Total dont la nature est non définie/indéterminée/hors liste : {undefined_nature_count}")
    
    if undefined_nature_count > 0:
        print("valeurs hors nomenclature :")
        for bad_nature, count in sorted(undefined_nature_details.items(), key=lambda x: x[1], reverse=True):
            print(f"  - '{bad_nature}' : {count} occurrence(s)")

if __name__ == "__main__":
    BASE_DIR = "./data/raw/glozz"
    OUTPUT_FILE = "dataset_emotions_montrees.xlsx"
    process_glozz_sentences_montree(BASE_DIR, OUTPUT_FILE)
