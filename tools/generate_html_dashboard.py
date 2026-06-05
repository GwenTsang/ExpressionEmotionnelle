import pandas as pd
import json
import os

# Chemins des fichiers
XLSX_STANZA_PATH = r"C:\Users\Philo\ExpressionEmotionnelle\results\xlsx\specificity_results_stanza\markers_stanza.csv"
XLSX_SPACY_PATH = r"C:\Users\Philo\ExpressionEmotionnelle\results\xlsx\specificity_results_spacy\markers_spacy.csv"

GLOZZ_STANZA_PATH = r"C:\Users\Philo\ExpressionEmotionnelle\results\glozz\specificity_results_stanza\markers.csv"
GLOZZ_SPACY_PATH = r"C:\Users\Philo\ExpressionEmotionnelle\results\glozz\specificity_results_spacy\markers.csv"

EMOTIONS_12 = ['Colère', 'Dégoût', 'Joie', 'Peur', 'Surprise', 'Tristesse', 
               'Admiration', 'Culpabilité', 'Embarras', 'Fierté', 'Jalousie', 'Autre']

def normalize_emotion(emo):
    """Normalise le nom de l'émotion pour correspondre aux 12 émotions de base."""
    if not isinstance(emo, str):
        return None
    emo = emo.strip().lower()
    mapping = {
        'colere': 'Colère',
        'colère': 'Colère',
        'degout': 'Dégoût',
        'dégoût': 'Dégoût',
        'joie': 'Joie',
        'peur': 'Peur',
        'surprise': 'Surprise',
        'tristesse': 'Tristesse',
        'admiration': 'Admiration',
        'culpabilite': 'Culpabilité',
        'culpabilité': 'Culpabilité',
        'embarras': 'Embarras',
        'fierte': 'Fierté',
        'fierté': 'Fierté',
        'jalousie': 'Jalousie',
        'autre': 'Autre',
        'aucun': None,
        'aucune': None,
        '': None
    }
    return mapping.get(emo, emo.capitalize())

def get_xlsx_data(file_path):
    if not os.path.exists(file_path):
        return []
    df = pd.read_csv(file_path, low_memory=False)
    if 'Désignée' not in df.columns:
        return []
    
    filtered = df[df['Désignée'] == 1.0]
    
    data = []
    for _, row in filtered.iterrows():
        marker = str(row.get('marker_value', '')).strip().lower()
        if not marker or marker == 'nan' or len(marker) <= 2:
            continue
            
        emotions = []
        for emo in EMOTIONS_12:
            if emo in row and row[emo] == 1:
                emotions.append(emo)
                
        if not emotions:
            emotions.append('Autre')
            
        for emo in emotions:
            data.append({'marker': marker, 'corpus': 'XLSX', 'emotion': emo})
    return data

def get_glozz_data(file_path):
    if not os.path.exists(file_path):
        return []
    df = pd.read_csv(file_path, low_memory=False)
    if 'mode' not in df.columns:
        return []
        
    filtered = df[df['mode'].astype(str).str.lower().str.strip() == 'designee']
    
    data = []
    for _, row in filtered.iterrows():
        marker = str(row.get('marker_value', '')).strip().lower()
        if not marker or marker == 'nan' or len(marker) <= 2:
            continue
            
        emotions = set()
        c1 = normalize_emotion(row.get('categorie1'))
        c2 = normalize_emotion(row.get('categorie2'))
        
        if c1: emotions.add(c1)
        if c2: emotions.add(c2)
        
        if not emotions:
            emotions.add('Autre')
            
        for emo in emotions:
            if emo in EMOTIONS_12:
                data.append({'marker': marker, 'corpus': 'Glozz', 'emotion': emo})
            else:
                data.append({'marker': marker, 'corpus': 'Glozz', 'emotion': 'Autre'})
                
    return data

def generate_html(data):
    # Regrouper par marqueur
    grouped = {}
    for item in data:
        m = item['marker']
        c = item['corpus']
        e = item['emotion']
        
        if m not in grouped:
            grouped[m] = {'marker': m, 'corpus': set(), 'emotions': set()}
            
        grouped[m]['corpus'].add(c)
        grouped[m]['emotions'].add(e)
        
    final_data = []
    for m, vals in grouped.items():
        final_data.append({
            'marker': m,
            'corpus': list(vals['corpus']),
            'emotions': list(vals['emotions'])
        })
        
    json_data = json.dumps(final_data)
    
    # Construction du HTML et CSS
    html_content = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Visualisation des Marqueurs - Mode Désigné</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
            color: #333;
        }}
        h1 {{
            text-align: center;
            color: #2c3e50;
            margin-bottom: 5px;
        }}
        p.subtitle {{
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 30px;
        }}
        .controls {{
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-bottom: 30px;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            flex-wrap: wrap;
        }}
        .control-group {{
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}
        select {{
            padding: 10px;
            font-size: 16px;
            border: 1px solid #ced4da;
            border-radius: 4px;
            min-width: 200px;
            background-color: #fff;
        }}
        .stats {{
            text-align: center;
            margin-bottom: 20px;
            font-size: 1.1em;
            color: #6c757d;
            font-weight: bold;
        }}
        .markers-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            justify-content: center;
            padding: 10px;
        }}
        .marker-chip {{
            background: white;
            padding: 8px 16px;
            border-radius: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border: 1px solid #e9ecef;
            font-size: 14px;
            transition: transform 0.2s;
            cursor: pointer;
        }}
        .marker-chip:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        /* Couleurs pour les corpus */
        .corpus-xlsx {{
            border-left: 5px solid #3498db;
        }}
        .corpus-glozz {{
            border-left: 5px solid #e74c3c;
        }}
        .corpus-both {{
            border-left: 5px solid #9b59b6;
        }}
        
        .tooltip {{
            position: relative;
            display: inline-block;
        }}
        .tooltip .tooltiptext {{
            visibility: hidden;
            width: max-content;
            background-color: #2c3e50;
            color: #fff;
            text-align: left;
            border-radius: 6px;
            padding: 10px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            transform: translateX(-50%);
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 13px;
            line-height: 1.4;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }}
        .tooltip:hover .tooltiptext {{
            visibility: visible;
            opacity: 1;
        }}
        .legend {{
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-bottom: 20px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 14px;
            color: #555;
        }}
        .color-box {{
            width: 16px;
            height: 16px;
            border-radius: 3px;
        }}
    </style>
</head>
<body>

    <h1>Explorateur de Marqueurs Linguistiques</h1>
    <p class="subtitle">Mode d'expression : <strong>Désigné</strong></p>
    
    <div class="legend">
        <div class="legend-item"><div class="color-box" style="background: #3498db;"></div> Uniquement dans XLSX</div>
        <div class="legend-item"><div class="color-box" style="background: #e74c3c;"></div> Uniquement dans Glozz</div>
        <div class="legend-item"><div class="color-box" style="background: #9b59b6;"></div> Commun aux deux corpus</div>
    </div>

    <div class="controls">
        <div class="control-group">
            <label for="corpusFilter">Filtrer par Corpus :</label>
            <select id="corpusFilter">
                <option value="all">Tous les corpus</option>
                <option value="XLSX">Uniquement XLSX (exclusif)</option>
                <option value="Glozz">Uniquement Glozz (exclusif)</option>
                <option value="both">Communs aux deux corpus</option>
                <option value="contains_xlsx">Contient dans XLSX</option>
                <option value="contains_glozz">Contient dans Glozz</option>
            </select>
        </div>
        
        <div class="control-group">
            <label for="emotionFilter">Filtrer par Émotion :</label>
            <select id="emotionFilter">
                <option value="all">Toutes les émotions</option>
                {''.join(f'<option value="{e}">{e}</option>' for e in EMOTIONS_12)}
            </select>
        </div>
    </div>

    <div class="stats" id="stats">Chargement...</div>

    <div class="markers-container" id="markersContainer"></div>

    <script>
        const markersData = {json_data};
        
        // Tri alphabétique par défaut
        markersData.sort((a, b) => a.marker.localeCompare(b.marker));
        
        const corpusFilter = document.getElementById('corpusFilter');
        const emotionFilter = document.getElementById('emotionFilter');
        const container = document.getElementById('markersContainer');
        const stats = document.getElementById('stats');

        function renderMarkers() {{
            const corpusVal = corpusFilter.value;
            const emoVal = emotionFilter.value;
            
            container.innerHTML = '';
            
            let count = 0;
            
            markersData.forEach(item => {{
                // Filtrage du corpus
                let showCorpus = false;
                if (corpusVal === 'all') {{
                    showCorpus = true;
                }} else if (corpusVal === 'both' && item.corpus.length === 2) {{
                    showCorpus = true;
                }} else if (corpusVal === 'XLSX' && item.corpus.length === 1 && item.corpus[0] === 'XLSX') {{
                    showCorpus = true;
                }} else if (corpusVal === 'Glozz' && item.corpus.length === 1 && item.corpus[0] === 'Glozz') {{
                    showCorpus = true;
                }} else if (corpusVal === 'contains_xlsx' && item.corpus.includes('XLSX')) {{
                    showCorpus = true;
                }} else if (corpusVal === 'contains_glozz' && item.corpus.includes('Glozz')) {{
                    showCorpus = true;
                }}
                
                // Filtrage de l'émotion
                let showEmo = false;
                if (emoVal === 'all') {{
                    showEmo = true;
                }} else if (item.emotions.includes(emoVal)) {{
                    showEmo = true;
                }}
                
                if (showCorpus && showEmo) {{
                    count++;
                    const div = document.createElement('div');
                    
                    let corpusClass = 'corpus-both';
                    if (item.corpus.length === 1) {{
                        corpusClass = item.corpus[0] === 'XLSX' ? 'corpus-xlsx' : 'corpus-glozz';
                    }}
                    
                    div.className = `marker-chip tooltip ${{corpusClass}}`;
                    div.textContent = item.marker;
                    
                    const tooltip = document.createElement('span');
                    tooltip.className = 'tooltiptext';
                    tooltip.innerHTML = `<strong>Marqueur :</strong> ${{item.marker}}<br>
                                         <strong>Corpus :</strong> ${{item.corpus.join(', ')}}<br>
                                         <strong>Émotions :</strong> ${{item.emotions.join(', ')}}`;
                    
                    div.appendChild(tooltip);
                    container.appendChild(div);
                }}
            }});
            
            stats.textContent = `Affichage de ${{count}} marqueur(s) (sur ${{markersData.length}} au total)`;
        }}

        corpusFilter.addEventListener('change', renderMarkers);
        emotionFilter.addEventListener('change', renderMarkers);

        // Rendu initial
        renderMarkers();
    </script>
</body>
</html>
"""
    output_path = r"C:\Users\Philo\ExpressionEmotionnelle\results\dashboard_marqueurs_designe.html"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"HTML généré avec succès: {output_path}")

def main():
    print("Lecture des données XLSX...")
    data_xlsx_stanza = get_xlsx_data(XLSX_STANZA_PATH)
    data_xlsx_spacy = get_xlsx_data(XLSX_SPACY_PATH)
    
    print("Lecture des données Glozz...")
    data_glozz_stanza = get_glozz_data(GLOZZ_STANZA_PATH)
    data_glozz_spacy = get_glozz_data(GLOZZ_SPACY_PATH)
    
    all_data = data_xlsx_stanza + data_xlsx_spacy + data_glozz_stanza + data_glozz_spacy
    
    print(f"Génération du HTML avec {len(all_data)} correspondances extraites...")
    generate_html(all_data)

if __name__ == "__main__":
    main()
