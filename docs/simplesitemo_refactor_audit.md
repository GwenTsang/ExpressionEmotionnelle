# SimpleSitEmo : audit et cible minimale de refactoring

Ce document remplace la premiere version trop large de la proposition
`SimpleSitEmo`. L'objectif est maintenant plus incremental : definir le plus
petit format commun utile pour unifier les pipelines Glozz et XLSX, puis ajouter
des colonnes ou des variantes d'analyse seulement si une tache concrete les
exige.

## Objectif

Construire un format commun entre :

- les unites `SitEmo` du corpus Glozz ;
- les spans `span1_text`, `span2_text`, etc. du fichier XLSX
  `data/raw/xlsx/CyberAdoAgg_gold_global_total_latest.xlsx`.

La motivation principale est d'eviter deux pipelines conceptuellement
differentes :

- Glozz : extraction de marqueurs depuis des segments annotes intra-phrastiques
  (`text_span`) ;
- XLSX actuel : extraction de marqueurs depuis toute la ligne `TEXT`, avec
  propagation de labels globaux de ligne.

Le pipeline cible doit extraire les marqueurs depuis des unites textuelles
annotees comparables dans les deux sources.

## Constats techniques utiles

### XLSX actuel

Apres restauration des labels de spans depuis la version GitHub anterieure, le
fichier actuel verifie les invariants suivants :

```text
Emo=1 : 398 lignes
Emo=1 avec au moins un span*_text : 398
Emo=1 sans span*_text : 0
Emo=1 avec spans_json rempli : 398
span*_text sans span*_cat/mode : 0
n_spans incoherents : 0
spans_json parse errors : 0
spans_json count mismatches : 0
```

Les lignes `Montree=1` n'ont plus un encodage emotionnel different : elles ont
maintenant aussi `span*_text`, `span*_cat`, `span*_mode` et `spans_json`.

La difference restante est que les lignes `Montree=1` possedent aussi des
colonnes de nature linguistique :

```text
nature_linguistique_span_1
nature_linguistique_span_2
nature_linguistique_span_3
nature_linguistique_span_4
```

Ces colonnes existent actuellement pour les emotions montrees. Les valeurs
manquantes pour les autres modes doivent etre traitees comme un manque
annotationnel, pas comme une difference de schema.

### Quatre doublons exacts de spans

Quatre lignes contiennent deux spans avec exactement le meme contenu textuel dans
la meme ligne. Les deux spans ont le meme mode `Montrée`, mais des emotions
differentes.

```text
idx=105, ID=111
"gros tat"
span1 = Montrée / Colère + Dégoût
span2 = Montrée / Autre

idx=253, ID=260
"tema sa gueule"
span1 = Montrée / Colère + Dégoût
span2 = Montrée / Autre

idx=328, ID=335
"gros lard"
span1 = Montrée / Colère + Dégoût
span2 = Montrée / Autre

idx=339, ID=346
"sale merde"
span1 = Montrée / Autre + Colère
span2 = Montrée / Dégoût
```

Pour `SimpleSitEmo`, ces doublons doivent etre fusionnes en une seule unite avec
trois emotions :

```text
"gros tat"       -> Colère, Dégoût, Autre
"tema sa gueule" -> Colère, Dégoût, Autre
"gros lard"      -> Colère, Dégoût, Autre
"sale merde"     -> Autre, Colère, Dégoût
```

Avant cette fusion, aucun span XLSX n'a plus de deux emotions dans `span*_cat` :

```text
spans avec 1 emotion : 418
spans avec 2 emotions : 77
spans avec 3+ emotions : 0
```

Apres fusion, le format cible doit donc autoriser jusqu'a trois emotions.

## Schema minimal propose

Le schema `SimpleSitEmo` doit rester volontairement petit :

```text
source_file
text_span
mode
emotion1
emotion2
emotion3
nature_linguistique
```

Raison d'etre des champs :

- `source_file` : rattachement minimal a la source.
- `text_span` : unite textuelle depuis laquelle extraire les marqueurs.
- `mode` : un seul mode d'expression.
- `emotion1`, `emotion2`, `emotion3` : une a trois emotions annotees.
- `nature_linguistique` : nature annotee quand disponible.

Champs volontairement exclus pour l'instant :

```text
source_format
source_row
source_idx
source_id
source_unit_id
origin_text
start_idx
end_idx
segments_json
is_discontinuous
raw_emotions
analysis_emotions
n_emotions
has_other_emotion
out_of_schema_affect
out_of_schema_source
notes
```

Ces informations sont soit recalculables, soit utiles seulement pour des audits
specifiques, soit trop lourdes pour le format analytique minimal. Les offsets
posent aussi un probleme de schema avec les unites discontinues Glozz : soit il
faudrait ajouter `segments_json`, soit accepter une approximation. Pour garder
`SimpleSitEmo` simple, on ne stocke aucun offset. Le principe est de conserver
les scripts de generation dans le repository afin de pouvoir remonter aux
fichiers sources en cas d'audit.

## Choix de normalisation des labels

Pour simplifier les scripts, il faut choisir une seule convention et l'appliquer
des la creation des fichiers `SimpleSitEmo`.

Decision recommandee : utiliser uniquement les versions accentuees dans les
fichiers normalises.

Emotions :

```text
Colère
Dégoût
Joie
Peur
Surprise
Tristesse
Admiration
Culpabilité
Embarras
Fierté
Jalousie
Autre
```

Modes :

```text
Comportementale
Désignée
Montrée
Suggérée
```

Les scripts `build_simplesitemo_xlsx.py` et `build_simplesitemo_glozz.py`
devront faire la conversion une seule fois a l'entree. Les scripts en aval ne
devraient pas tester les deux variantes accentuees/non accentuees.

## Conversion XLSX vers SimpleSitEmo

Source de verite :

```text
spanN_text
spanN_cat
spanN_mode
nature_linguistique_span_N
```

Les colonnes globales comme `Colere`, `Degout`, `Montree`, `Emo`, etc. doivent
servir a verifier la coherence du fichier, pas a produire les labels des unites.
La colonne `TEXT` reste utile dans le script de build pour verifier les spans ou
pour d'eventuels audits, mais elle n'est pas stockee dans `SimpleSitEmo`.

Algorithme minimal :

```text
pour chaque ligne XLSX:
    si Emo != 1:
        ignorer

    pour N dans 1..4:
        si spanN_text est vide:
            continuer

        text_span = spanN_text
        mode = normaliser spanN_mode en label accentue
        emotions = parser spanN_cat, separateur " + ", labels accentues
        nature_linguistique = nature_linguistique_span_N

        creer une unite SimpleSitEmo candidate

    fusionner les candidates de la meme ligne qui ont:
        meme text_span exact
        meme mode

    pour chaque unite fusionnee:
        dedupliquer les emotions en conservant l'ordre
        remplir emotion1, emotion2, emotion3
```

Comme aucun offset n'est stocke, il n'est pas necessaire de resoudre les cas ou
un meme `spanN_text` apparait plusieurs fois dans `TEXT` pour la premiere
version analytique. Ces cas pourront etre audites en relancant le script de
generation sur les sources.

## Conversion Glozz vers SimpleSitEmo

Source de verite :

```text
unit_id
type == SitEmo
text_span
Mode
Categorie
Categorie2
Nature
```

Le parseur Glozz actuel extrait deja la plupart de ces champs, mais il faudra
ajouter `Nature` si elle n'est pas encore exposee dans la table de sortie.

Algorithme minimal :

```text
pour chaque unite Glozz:
    si type != SitEmo:
        ignorer pour le premier build SimpleSitEmo

    text_span = text_span Glozz
    mode = normaliser Mode en label accentue
    emotions = Categorie + Categorie2 non vides, labels accentues
    nature_linguistique = Nature

    remplir emotion1, emotion2, emotion3
```

Pour les unites discontinues, ne pas ajouter `segments_json` dans cette premiere
version. On conserve seulement le `text_span` concatene deja produit par le
parseur. C'est une perte d'information acceptable pour le format simplifie. Les
fichiers historiques Glozz restent disponibles pour les audits plus fins.

## Traitement de Autre

`Autre` reste une emotion annotee dans `emotion1`, `emotion2` ou `emotion3`.
Elle ne doit pas etre supprimee du fichier `SimpleSitEmo`.

En revanche, les analyses principales pourront simplement l'ignorer dans les
scripts :

```text
emotions_for_analysis = [emotion1, emotion2, emotion3] sans Autre
```

Il n'est pas necessaire de stocker une colonne `analysis_emotions` ou
`has_other_emotion` : ces informations sont triviales a recalculer.

La question specifique `Autre` / `mépris-haine` est documentee separement dans :

```text
ToDo/08_analyse_exploratoire_mepris_haine.md
```

## Pipeline incremental recommande

Commencer par de petites taches verifiables :

1. Definir le schema minimal `SimpleSitEmo`.
2. Ecrire `build_simplesitemo_xlsx.py`.
3. Ecrire `build_simplesitemo_glozz.py`.
4. Ecrire `extract_markers_simplesitemo.py`.
5. Adapter les scripts de specificite pour lire les marqueurs issus de
   `SimpleSitEmo`.
6. Ajouter ensuite seulement les variantes d'analyse : exclusion de `Autre`,
   analyse exploratoire `mépris-haine`, etc.

Principe : ne pas ajouter de colonnes ou de branches de code pour des besoins
hypothetiques. Ajouter une complexite seulement lorsqu'une analyse concrete la
requiert.

## Taches liees

- `ToDo/01_definir_schema_simplesitemo.md`
- `ToDo/02_build_simplesitemo_xlsx.md`
- `ToDo/03_build_simplesitemo_glozz.md`
- `ToDo/04_fusionner_simplesitemo_parquet.md`
- `ToDo/05_extract_markers_simplesitemo.md`
- `ToDo/06_adapter_specificite_simplesitemo.md`
- `ToDo/07_exclure_autre_analyses_principales.md`
- `ToDo/08_analyse_exploratoire_mepris_haine.md`
