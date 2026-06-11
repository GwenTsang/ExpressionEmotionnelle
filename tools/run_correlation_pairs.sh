#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

python_bin="${PYTHON:-python}"

binary_groups=(
  EMOTIONS
  MODES
)

categorical_groups=(
  HATE
  VERBAL_ABUSE
  INTENTION
  TARGET
  ROLE
  CONTEXT
)

for binary_group in "${binary_groups[@]}"; do
  for categorical_group in "${categorical_groups[@]}"; do
    printf '==> %s x %s\n' "${binary_group}" "${categorical_group}"
    "${python_bin}" tools/correlation.py \
      "${binary_group}" \
      "${categorical_group}" \
      "$@"
  done
done

"${python_bin}" tools/correlation.py EMOTIONS MODES "$@"
