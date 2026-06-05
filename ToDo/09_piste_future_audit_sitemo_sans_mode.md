# Piste future 09 : auditer les unites `SitEmo` sans mode

- Objectif : exporter les 164 unites `SitEmo` sans mode dans un fichier dedie pour audit manuel.
- Formats possibles : `CSV`, `XLSX` ou format equivalent pertinent.
- Statut : piste future, hors premiere implementation de `SimpleSitEmo.parquet`.

Hypothese a tester :

- certaines unites sans mode pourraient ne pas se distinguer nettement des autres unites `SitEmo` ;
- elles pourraient se rapprocher davantage des modes `Montrée` et `Suggérée` que de `Comportementale` ou `Désignée`.

Procedure exploratoire possible :

- constituer un jeu d'entrainement avec les `SitEmo` modales ;
- extraire des features textuelles et contextuelles ;
- utiliser une validation croisee ;
- calibrer les probabilites ;
- proposer une annotation candidate uniquement lorsque la confiance est suffisante.
