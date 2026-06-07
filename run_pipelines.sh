#!/bin/bash
set -e

mkdir -p data/pipeline_1
mkdir -p results/pipeline_1

echo "1. Extraction Glozz (Pipeline 1)"
python -m pipeline_1.build_simplesitemo_glozz -o data/pipeline_1/SimpleSitEmo_glozz.parquet

echo "2. Extraction XLSX (Pipeline 1)"
python -m pipeline_1.build_simplesitemo_xlsx -o data/pipeline_1/SimpleSitEmo_xlsx.parquet

echo "3. Merge (Pipeline 1)"
python -m pipeline_1.merge_simplesitemo --xlsx data/pipeline_1/SimpleSitEmo_xlsx.parquet --glozz data/pipeline_1/SimpleSitEmo_glozz.parquet -o data/pipeline_1/SimpleSitEmo.parquet

echo "4. Analysis (Pipeline 1)"
python -m pipeline_1.run_analysis --input data/pipeline_1/SimpleSitEmo.parquet --output-dir results/pipeline_1 --lemmatizer spacy --min-freq 3

mkdir -p data/pipeline_2
mkdir -p results/pipeline_2

echo "1. Extraction Glozz (Pipeline 2)"
python -m pipeline_2.build_simplesitemo_glozz -o data/pipeline_2/SimpleSitEmo_glozz.parquet

echo "2. Extraction XLSX (Pipeline 2)"
python -m pipeline_2.build_simplesitemo_xlsx -o data/pipeline_2/SimpleSitEmo_xlsx.parquet

echo "3. Merge (Pipeline 2)"
python -m pipeline_2.merge_simplesitemo --xlsx data/pipeline_2/SimpleSitEmo_xlsx.parquet --glozz data/pipeline_2/SimpleSitEmo_glozz.parquet -o data/pipeline_2/SimpleSitEmo.parquet

echo "4. Analysis (Pipeline 2)"
python -m pipeline_2.run_analysis --input data/pipeline_2/SimpleSitEmo.parquet --output-dir results/pipeline_2 --lemmatizer spacy --min-freq 3
