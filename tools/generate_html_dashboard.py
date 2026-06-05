import pandas as pd
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MARKERS_PATH = os.path.join(BASE_DIR, "results", "simplesitemo", "markers.csv")

EMOTIONS_12 = ['Colère', 'Dégoût', 'Joie', 'Peur', 'Surprise', 'Tristesse', 
               'Admiration', 'Culpabilité', 'Embarras', 'Fierté', 'Jalousie', 'Autre']

def get_markers_data(file_path):
    if not os.path.exists(file_path):
        return []
    df = pd.read_csv(file_path, low_memory=False)
    if 'mode' not in df.columns or 'source_file' not in df.columns:
        return []
        
    filtered = df[df['mode'].astype(str).str.lower().str.strip() == 'désignée']
    
    data = []
    for _, row in filtered.iterrows():
        marker = str(row.get('marker_value', '')).strip().lower()
        if not marker or marker == 'nan' or len(marker) <= 2:
            continue
            
        emotion = str(row.get('emotion', '')).strip()
        if not emotion or emotion not in EMOTIONS_12:
            emotion = 'Autre'
            
        source = str(row.get('source_file', '')).strip()
        corpus = 'CyberAggAdoLarge' if source == 'CyberAggAdo' else 'TextToKids'
            
        data.append({'marker': marker, 'corpus': corpus, 'emotion': emotion})
                
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
        .corpus-cyber {{
            border-left: 5px solid #3498db;
        }}
        .corpus-ttk {{
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
        <div class="legend-item"><div class="color-box" style="background: #3498db;"></div> Uniquement dans CyberAggAdoLarge</div>
        <div class="legend-item"><div class="color-box" style="background: #e74c3c;"></div> Uniquement dans TextToKids</div>
        <div class="legend-item"><div class="color-box" style="background: #9b59b6;"></div> Commun aux deux corpus</div>
    </div>

    <div class="controls">
        <div class="control-group">
            <label for="corpusFilter">Filtrer par Corpus :</label>
            <select id="corpusFilter">
                <option value="all">Tous les corpus</option>
                <option value="CyberAggAdoLarge">Uniquement CyberAggAdoLarge (exclusif)</option>
                <option value="TextToKids">Uniquement TextToKids (exclusif)</option>
                <option value="both">Communs aux deux corpus</option>
                <option value="contains_cyber">Contient dans CyberAggAdoLarge</option>
                <option value="contains_ttk">Contient dans TextToKids</option>
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
                }} else if (corpusVal === 'CyberAggAdoLarge' && item.corpus.length === 1 && item.corpus[0] === 'CyberAggAdoLarge') {{
                    showCorpus = true;
                }} else if (corpusVal === 'TextToKids' && item.corpus.length === 1 && item.corpus[0] === 'TextToKids') {{
                    showCorpus = true;
                }} else if (corpusVal === 'contains_cyber' && item.corpus.includes('CyberAggAdoLarge')) {{
                    showCorpus = true;
                }} else if (corpusVal === 'contains_ttk' && item.corpus.includes('TextToKids')) {{
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
                        corpusClass = item.corpus[0] === 'CyberAggAdoLarge' ? 'corpus-cyber' : 'corpus-ttk';
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
    output_path = os.path.join(BASE_DIR, "results", "dashboard_marqueurs_designe.html")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"HTML généré avec succès: {output_path}")

def main():
    print("Lecture des données unifiées...")
    all_data = get_markers_data(MARKERS_PATH)
    
    print(f"Génération du HTML avec {len(all_data)} correspondances extraites...")
    generate_html(all_data)

if __name__ == "__main__":
    main()
