# Piste future 10 : simplifier les interfaces CLI

- Objectif : simplifier drastiquement les `argparse` / `parse_args`.
- Moment pertinent : apres l'unification de la pipeline d'analyse autour de `SimpleSitEmo.parquet`.
- Statut : piste future.

Principe :

- les scripts de construction des donnees doivent avoir une interface courte et specialisee ;
- la pipeline d'analyse doit recevoir un fichier Parquet normalise en entree ;
- les options historiques liees au parsing Glozz/XLSX brut doivent disparaitre de la pipeline aval.
