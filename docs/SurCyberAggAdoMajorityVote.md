Les fichiers dans `data/CyberAgression-Large` contiennent les étiquettes majoritaires parmi les trois annotateurs.

Le repo original `https://github.com/aollagnier/CyberAgression-Large` contient un autre dossier, nommé Cyber_Aggression_Batch2, qui contient 36 fichiers XLSX ayant chacun trois onglets. Chaque onglet correspond aux étiquettes proposées par un annotateur.

La méthode pour produire les XLSX dans `data/CyberAgression-Large` consiste à garder une étiquette si elle a été proposée par deux annotateurs, et à remplir "File: ... NULL." sinon.

Après avoir inspecté manuellement les données j'ai constaté qu'il était intéressant d'enrichir la colonne 'TARGET' car il y a de très nombreux cas où un seul annotateur a proposé une annotation et où les deux autres ont rempli "NULL". L'enrichissement consiste à conserver l'étiquette non NULL.