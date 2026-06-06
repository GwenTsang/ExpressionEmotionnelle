

Dans la grande majorité des cas, dans le Glozz, l'expérienceur est annoté sur un segment textuel (un pronom, un nom propre, un groupe nominal, etc.). Cependant, il existe une exception majeure détaillée à la Page 50 (Section 3.4.4).
Lorsque l'émotion est ressentie par la Doxa (l'opinion publique), le Scripteur ou le Narrateur, il arrive fréquemment que ces entités ne soient mentionnées par aucun mot dans la phrase. Le guide prévoit alors la création d'une Unité « fantôme ».

Puisque le logiciel Glozz oblige l'annotateur à sélectionner quelque chose pour créer une unité, le guide donne la consigne suivante :

"Pour délimiter une Unité fantôme, l’annotateur sélectionne le premier caractère qui jouxte à droite l’Unité SitEmo [...] et correspond généralement à un espace ou un signe de ponctuation [...] pour traduire le fait que l’unité n’existe pas matériellement dans le texte."

Il est possible d'identifier informatiquement ces unités « fantômes » (sans réalité lexicale) grâce à la combinaison de deux facteurs :

1. Le texte capturé (`text_span`) ne sera qu'un espace `" "` ou une ponctuation (`"."`, `","`, `"!"`).
2. Le trait (feature) `Nature` de cette unité aura obligatoirement pour valeur **`N/A`** (comme défini à la page 50 et dans le schéma XML en annexe). Le trait `Entite` contiendra alors "Doxa", "Scripteur" ou "Narrateur".
