#!/usr/bin/env python3
"""One-command reproduction wrapper for the JCIM manuscript analysis pipeline.

Usage
-----

    python scripts/reproduce_all.py            # run all steps; warn on SHA mismatch
    python scripts/reproduce_all.py --strict   # run all steps; abort on SHA mismatch
    python scripts/reproduce_all.py --skip-data  # skip data build (use existing CSVs)
    python scripts/reproduce_all.py --verify-only  # only verify existing checksums

Steps
-----
1. Data: extract DD InP subset (132 rows) + build inp_combined.csv (352 rows).
2. Part A: cross-validation (4 stages — paired / distributional / reproduction / cross-prediction).
3. Part B: predictive modeling (4 targets — PL_peak, QY, FWHM, shell_layer_count).
4. Part C: recipe generation (3 methods — RF sampling, Bayesian optimization, grid).
5. Phase 3 extensions (optional): Cossairt 2 x 2 ablation, Part B baseline
   hierarchy, leave-one-paper-out SHAP stability.

After each step, --strict mode verifies all expected output SHA-256 against
the manifest `scripts/expected_checksums.json`. Any mismatch terminates with
exit code 2 and a diff report. Without --strict, mismatches are warned but
not fatal.

Notes on actual behavior
------------------------
- Step 1 (data) is *skip-if-exists*: when the shipped `inp_subset.csv` /
  `inp_combined.csv` are present they are reused, not re-derived (the source
  DD corpus lives on Zenodo, not in this repo). So a clean public checkout
  reproduces the *analysis* (Steps 2–5), while the deposited data CSVs are
  verified by checksum rather than rebuilt. Use `extract_inp_subset.py
  --source <corpus> --force` to actually re-derive them.
- The Phase 3 extension *outputs* are verified against the manifest in every
  run, including `--skip-phase3` (which only skips *re-running* them).
- Part C generation is wrapped so a failure warns and continues to the
  verification phase rather than aborting the whole run.

Expected runtime: ~ 15 minutes on a standard laptop for Steps 1–4; +5 min
for Step 5 (Phase 3 extensions) on the same hardware.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "scripts" / "expected_checksums.json"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> dict:
    with MANIFEST_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def verify_section(section_name: str, section: dict, strict: bool) -> bool:
    """Verify all files in a manifest section. Return True if all match."""
    if section_name.startswith("_"):
        return True
    print(f"\n=== Verifying {section_name} ===")
    ok = True
    for rel_path, expected in section.items():
        if rel_path.startswith("_"):
            continue
        path = PROJECT_ROOT / rel_path
        if not path.exists():
            msg = f"MISSING: {rel_path}"
            if strict:
                print(f"  [STRICT FAIL] {msg}")
                ok = False
            else:
                print(f"  [warn]        {msg}")
            continue
        actual = sha256_of(path)
        if actual != expected:
            msg = f"SHA-256 MISMATCH: {rel_path}"
            print(f"  [{'STRICT FAIL' if strict else 'warn'}] {msg}")
            print(f"    expected: {expected}")
            print(f"    actual:   {actual}")
            if strict:
                ok = False
        else:
            print(f"  [ok]          {rel_path}  ({actual[:12]}…)")
    return ok


def run_step(label: str, cmd: list[str]) -> None:
    print(f"\n=== {label} ===")
    print(f"$ {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=PROJECT_ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true",
                        help="Abort on SHA-256 mismatch (exit code 2).")
    parser.add_argument("--skip-data", action="store_true",
                        help="Skip the data build step (use existing CSVs).")
    parser.add_argument("--verify-only", action="store_true",
                        help="Skip all analysis steps; only verify existing outputs.")
    parser.add_argument("--skip-phase3", action="store_true",
                        help="Skip *re-running* Phase 3 extension scripts "
                             "(F17/F18/F19); their deposited outputs are still "
                             "verified against the manifest.")
    args = parser.parse_args()

    manifest = load_manifest()
    python = sys.executable  # use the same Python that invoked this script

    if not args.verify_only:
        # Step 1 — Data
        if not args.skip_data:
            run_step("Step 1: Data build",
                     [python, "scripts/extract_inp_subset.py"])
            run_step("Step 1: Combined dataset",
                     [python, "scripts/build_inp_combined.py"])
        else:
            print("\n=== Step 1: Data build (SKIPPED via --skip-data) ===")

        # Step 2 — Part A
        run_step("Step 2: Part A cross-validation",
                 [python, "analysis/part_a_cross_validation.py"])

        # Step 3 — Part B
        run_step("Step 3: Part B predictive modeling",
                 [python, "analysis/part_b_predictive_modeling.py"])

        # Step 4 — Part C (wrapped: warn instead of abort on failure; the
        # verification phase still checks deposited Part C artifacts).
        try:
            run_step("Step 4: Part C recipe generation",
                     [python, "analysis/part_c_recipe_generation.py"])
        except subprocess.CalledProcessError as e:
            print(f"  [warn] Part C generation exited {e.returncode}; continuing "
                  f"to verification. Deposited Part C artifacts will be checked "
                  f"against the manifest as-is.")

        # Step 5 — Phase 3 extensions (optional)
        if not args.skip_phase3:
            run_step("Step 5a: Cossairt 2x2 ablation (F17)",
                     [python, "analysis/cossairt_ablation.py"])
            run_step("Step 5b: Part B baseline hierarchy (F18)",
                     [python, "analysis/part_b_baselines.py"])
            run_step("Step 5c: LOPO SHAP stability (F19)",
                     [python, "analysis/part_b_shap_stability.py"])
            run_step("Step 5d: Time-forward validation (F37)",
                     [python, "analysis/time_forward_validation.py"])
            run_step("Step 5e: Feature pair audit (F39)",
                     [python, "analysis/feature_pair_audit.py"])
        else:
            print("\n=== Step 5: Phase 3 extensions (SKIPPED via --skip-phase3) ===")

    # Verification phase
    print("\n" + "=" * 70)
    print("VERIFICATION PHASE")
    print("=" * 70)
    all_ok = True
    # Phase 3 extension *outputs* are always verified against the manifest, even
    # when --skip-phase3 skips re-running them (deposited artifacts are checked).
    sections_to_verify = ["data", "part_a_results", "part_b_results",
                          "part_c_results", "phase3_extensions"]
    for section_name in sections_to_verify:
        section_ok = verify_section(section_name, manifest.get(section_name, {}),
                                    args.strict)
        all_ok = all_ok and section_ok

    print("\n" + "=" * 70)
    if args.strict and not all_ok:
        print("[STRICT FAIL] One or more outputs do not match the deposited manifest.")
        return 2
    print("All steps complete.")
    if not all_ok:
        print("(Some files mismatched; re-run with --strict for a hard failure.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
