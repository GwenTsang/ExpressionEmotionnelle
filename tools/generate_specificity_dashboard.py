import pandas as pd
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPECIFICITY_PATH = os.path.join(BASE_DIR, "results", "simplesitemo", "specificity_results", "entropy_per_marker_emotion.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "results", "dashboard_specificite.html")

EMOTIONS = ['Colère', 'Dégoût', 'Joie', 'Peur', 'Surprise', 'Tristesse', 
            'Admiration', 'Culpabilité', 'Embarras', 'Fierté', 'Jalousie']

def generate_dashboard():
    if not os.path.exists(SPECIFICITY_PATH):
        print(f"Fichier introuvable: {SPECIFICITY_PATH}")
        return

    df = pd.read_csv(SPECIFICITY_PATH)
    
    # Process data to send to JS
    data = []
    for _, row in df.iterrows():
        # Find dominant emotion
        max_p = 0
        dominant = "Aucune"
        for emo in EMOTIONS:
            p_col = f"P({emo})"
            if p_col in row and row[p_col] > max_p:
                max_p = row[p_col]
                dominant = emo
                
        data.append({
            'marker': str(row['marker_value']),
            'type': str(row['marker_type']),
            'count': int(row['total_count']),
            'entropy': round(float(row['entropy']), 3),
            'norm_entropy': round(float(row['normalized_entropy']), 3),
            'dominant_emotion': dominant
        })
        
    json_data = json.dumps(data)
    
    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Spécificité des Marqueurs Émotionnels</title>
    <style>
        :root {{
            --text-main: #111827;
            --text-muted: #6b7280;
            --border-color: #e5e7eb;
            --bg-main: #ffffff;
            --bg-alt: #f9fafb;
            --hover-bg: #f3f4f6;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            color: var(--text-main);
            background-color: var(--bg-main);
            margin: 0;
            padding: 40px 20px;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
        }}
        h1 {{
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
            letter-spacing: -0.025em;
        }}
        p.desc {{
            color: var(--text-muted);
            margin-bottom: 2rem;
            font-size: 0.95rem;
        }}
        .controls {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border-color);
        }}
        input[type="text"] {{
            padding: 8px 12px;
            border: 1px solid var(--border-color);
            border-radius: 4px;
            font-size: 0.9rem;
            width: 300px;
            outline: none;
            transition: border-color 0.2s;
        }}
        input[type="text"]:focus {{
            border-color: #9ca3af;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
            text-align: left;
        }}
        th, td {{
            padding: 12px 16px;
            border-bottom: 1px solid var(--border-color);
        }}
        th {{
            color: var(--text-muted);
            font-weight: 500;
            cursor: pointer;
            user-select: none;
        }}
        th:hover {{
            color: var(--text-main);
        }}
        tr:hover td {{
            background-color: var(--hover-bg);
        }}
        .entropy-low {{
            font-weight: 600;
            color: #111827;
        }}
        .type-badge {{
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Spécificité des Marqueurs Émotionnels</h1>
        <p class="desc">Une entropie basse indique qu'un marqueur est hautement spécifique à une émotion donnée.</p>
        
        <div class="controls">
            <input type="text" id="searchInput" placeholder="Rechercher un marqueur ou une émotion...">
            <div id="rowCount" style="color: var(--text-muted); font-size: 0.9rem;"></div>
        </div>

        <table id="dataTable">
            <thead>
                <tr>
                    <th onclick="sortTable('marker')">Marqueur ↕</th>
                    <th onclick="sortTable('type')">Type ↕</th>
                    <th onclick="sortTable('dominant_emotion')">Émotion Principale ↕</th>
                    <th onclick="sortTable('count')" style="text-align: right;">Occurrences ↕</th>
                    <th onclick="sortTable('entropy')" style="text-align: right;">Entropie ↕</th>
                </tr>
            </thead>
            <tbody id="tableBody"></tbody>
        </table>
    </div>

    <script>
        const rawData = {json_data};
        let currentData = [...rawData];
        let sortCol = 'entropy';
        let sortAsc = true;

        const tableBody = document.getElementById('tableBody');
        const searchInput = document.getElementById('searchInput');
        const rowCount = document.getElementById('rowCount');

        function renderTable() {{
            tableBody.innerHTML = '';
            
            // Ne render que les 1000 premiers pour les performances s'il y en a beaucoup
            const dataToRender = currentData.slice(0, 1000);
            
            dataToRender.forEach(row => {{
                const tr = document.createElement('tr');
                
                const entropyClass = row.entropy < 1.0 ? 'entropy-low' : '';
                
                tr.innerHTML = `
                    <td>${{row.marker}}</td>
                    <td><span class="type-badge">${{row.type}}</span></td>
                    <td>${{row.dominant_emotion}}</td>
                    <td style="text-align: right;">${{row.count}}</td>
                    <td style="text-align: right;" class="${{entropyClass}}">${{row.entropy.toFixed(3)}}</td>
                `;
                tableBody.appendChild(tr);
            }});
            
            let countText = `${{currentData.length}} marqueurs trouvés`;
            if (currentData.length > 1000) {{
                countText += ' (1000 premiers affichés)';
            }}
            rowCount.textContent = countText;
        }}

        function sortTable(column) {{
            if (sortCol === column) {{
                sortAsc = !sortAsc;
            }} else {{
                sortCol = column;
                sortAsc = true;
            }}
            
            currentData.sort((a, b) => {{
                let valA = a[column];
                let valB = b[column];
                
                if (typeof valA === 'string') valA = valA.toLowerCase();
                if (typeof valB === 'string') valB = valB.toLowerCase();
                
                if (valA < valB) return sortAsc ? -1 : 1;
                if (valA > valB) return sortAsc ? 1 : -1;
                return 0;
            }});
            
            renderTable();
        }}

        searchInput.addEventListener('input', (e) => {{
            const term = e.target.value.toLowerCase();
            currentData = rawData.filter(row => 
                row.marker.toLowerCase().includes(term) ||
                row.dominant_emotion.toLowerCase().includes(term)
            );
            
            const currentAsc = sortAsc;
            sortAsc = !currentAsc; 
            sortTable(sortCol);
        }});

        sortAsc = false; 
        sortTable('entropy');
    </script>
</body>
</html>"""
    
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write(html)
        
    print(f"Nouveau dashboard généré : {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_dashboard()
