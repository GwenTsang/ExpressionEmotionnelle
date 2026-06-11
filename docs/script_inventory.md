## Active pipeline

| Script | Status | Notes |
|:---|:---|:---|
| `setup.sh` | Active pipeline | Installs Python dependencies and the spaCy French model. |
| `glozz/__init__.py` | Active pipeline | Public API for the Glozz parser package. |
| `glozz/glozz_parser.py` | Active pipeline | Parses raw Glozz `.aa` / `.ac` annotations. CLI supports `--help`. |
| `pipeline/__init__.py` | Active pipeline | Package marker for the SimpleSitEmo pipeline. |
| `pipeline/build_simplesitemo_xlsx.py` | Active pipeline | Builds the XLSX SimpleSitEmo parquet. CLI supports `--help`. |
| `pipeline/build_simplesitemo_glozz.py` | Active pipeline | Builds the Glozz SimpleSitEmo parquet. CLI supports `--help`. |
| `pipeline/merge_simplesitemo.py` | Active pipeline | Merges source parquets into `data/SimpleSitEmo.parquet`. CLI supports `--help`. |
| `pipeline/run_analysis.py` | Active pipeline | Orchestrates marker extraction and specificity. CLI supports `--help`. |
| `pipeline/extract_markers.py` | Active pipeline | Extracts marker rows from `SimpleSitEmo.parquet`. CLI supports `--help`. |
| `pipeline/marker_specificity.py` | Active pipeline | Computes entropy and hypothesis reports. CLI supports `--help`. |
| `pipeline/nlp_utils.py` | Active pipeline | Shared NLP backends and token helpers. |
| `pipeline/emotion_taxonomy.py` | Active pipeline | Canonical labels and normalization helpers. |
| `pipeline/marker_contract.py` | Active pipeline | Marker table schema validation. |

## Analysis tools

| Script | Status | Notes |
|:---|:---|:---|
| `pipeline/build_token_lemma_table.py` | Analysis tool | Builds token-level form/lemma tables for granularity analysis. CLI supports `--help`. |
| `pipeline/compute_granularity.py` | Analysis tool | Computes surface-vs-lemma dilution reports. CLI supports `--help`. |
| `pipeline/viz_flexional_families.py` | Analysis tool | Builds flexional-family HTML reports. CLI supports `--help`. |
| `tools/match_marker_values_to_lexicon.py` | Analysis tool | Matches current `results/simplesitemo/markers.csv` values against the emotion lexicon. CLI supports `--help`. |
| `tools/top_markers.py` | Analysis tool | Prints/export top markers from specificity outputs. CLI supports `--help`. |
| `tools/prepare_correlation_dataset.py` | Analysis tool | Builds the CyberAdoAgg binary correlation dataset from the raw XLSX. |
| `tools/correlation.py` | Analysis tool | Runs pairwise/global correlation analyses. CLI supports `--help`. |
| `tools/run_correlations.py` | Analysis tool | Batch-runs `tools/correlation.py`. CLI supports `--help`. |
| `tools/generate_cyberado_role_target_visualization.py` | Analysis tool | Generates the ROLE -> TARGET flow HTML. CLI supports `--help`. |

## Legacy/archive candidates

| Script | Status | Notes |
|:---|:---|:---|
| `tools/compare_markers_designe.py` | Legacy/archive candidate | Hard-coded exploratory comparison for designated-mode markers. |
| `tools/generate_html_dashboard.py` | Legacy/archive candidate | Hard-coded HTML dashboard for designated markers. |
| `tools/generate_specificity_dashboard.py` | Legacy/archive candidate | Hard-coded HTML dashboard for specificity output. |

## Migration-only

No retained Python or shell script is migration-only. The migration-only
material is documentation, mainly
`docs/analyse_entropie_declencheurs_xlsx_glozz.md`, which records the old variant comparison and is not an active entrypoint.