"""Lance les analyses de correlation pour toutes les paires de groupes."""

import argparse
import shlex
import subprocess
import sys
from itertools import combinations
from pathlib import Path


GROUPS = [
    "ROLE",
    "HATE",
    "INTENTION",
    "VERBAL_ABUSE",
    "EMOTIONS",
    "MODES",
]

FIXED_ONLY_PAIR = ("EMOTIONS", "MODES")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Lance tools/correlation.py sur toutes les combinaisons de groupes."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["all", "global", "pairwise"],
        default="all",
        help=(
            "Mode transmis a tools/correlation.py. En mode all, la paire "
            "EMOTIONS/MODES est forcee en pairwise. En mode global, elle est "
            "ignoree car aucun groupe n'est categoriel."
        ),
    )
    parser.add_argument(
        "--input",
        help="Fichier Excel a transmettre a tools/correlation.py.",
    )
    parser.add_argument(
        "--output-dir",
        help="Dossier de sortie a transmettre a tools/correlation.py.",
    )
    return parser.parse_args()


def build_command(group_a, group_b, args):
    command = [sys.executable, "tools/correlation.py", group_a, group_b]

    if args.mode == "pairwise":
        command.extend(["--mode", "pairwise"])
    elif (group_a, group_b) == FIXED_ONLY_PAIR:
        command.extend(["--mode", "pairwise"])
    elif args.mode == "global":
        command.extend(["--mode", "global"])

    if args.input:
        command.extend(["--input", args.input])
    if args.output_dir:
        command.extend(["--output-dir", args.output_dir])

    return command


def format_command(command):
    display = ["python" if i == 0 else part for i, part in enumerate(command)]
    return shlex.join(display)


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    failures = []
    skipped = []

    for group_a, group_b in combinations(GROUPS, 2):
        if args.mode == "global" and (group_a, group_b) == FIXED_ONLY_PAIR:
            skipped.append((group_a, group_b))
            print(f"[skip] {group_a} {group_b} : mode global impossible")
            continue

        command = build_command(group_a, group_b, args)
        print(f"\n[run] {format_command(command)}")

        result = subprocess.run(command, cwd=repo_root)
        if result.returncode != 0:
            failures.append((group_a, group_b, result.returncode))

    if skipped:
        print("\nPaires ignorees :")
        for group_a, group_b in skipped:
            print(f"  - {group_a} x {group_b}")

    if failures:
        print("\nCommandes en echec :")
        for group_a, group_b, returncode in failures:
            print(f"  - {group_a} x {group_b} (code {returncode})")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
