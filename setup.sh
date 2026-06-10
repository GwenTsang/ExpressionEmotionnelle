#!/usr/bin/env bash
# Installation des dépendances du pipeline SimpleSitEmo.
set -e

pip install -r requirements.txt

python -m spacy download fr_core_news_sm

echo "Installation terminée."
echo "  Pour lancer l'analyse     : python -m pipeline.run_analysis --step all"
