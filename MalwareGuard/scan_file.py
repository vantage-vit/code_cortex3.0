#!/usr/bin/env python3
r"""
================================================================================
  CODE CORTEX 3.0  -  LOCAL PE SCANNER CLI (scan_file.py)
================================================================================
Usage:
    python scan_file.py <path_to_file_or_folder> [--threshold <float>]

Examples:
    python scan_file.py "C:\Users\lavan\Downloads\malicious_files"
    python scan_file.py "C:\Users\lavan\Downloads\benign_test_1.exe"
    python scan_file.py "C:\Windows\System32\cmd.exe"
    python scan_file.py "sample.exe" --threshold 0.49
================================================================================
"""

import sys
import os
import argparse
import glob

# Ensure ML_ENGINE is importable from script's dir or parent dirs
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(CURRENT_DIR, "ML_ENGINE")):
    sys.path.insert(0, CURRENT_DIR)
elif os.path.exists(os.path.join(CURRENT_DIR, "..", "ML_ENGINE")):
    sys.path.insert(0, os.path.abspath(os.path.join(CURRENT_DIR, "..")))

from ML_ENGINE.pe_extractor import extract_features_for_scaler
from ML_ENGINE.predict import MalwarePredictor


def scan_single(file_path: str, predictor: MalwarePredictor, custom_threshold: float = None, verbose: bool = True):
    file_size = os.path.getsize(file_path)
    file_name = os.path.basename(file_path)

    try:
        features = extract_features_for_scaler(file_path)
    except Exception as e:
        if verbose:
            print(f"  [!] Extraction failed for {file_name}: {e}")
        return {
            "filename": file_name,
            "path": file_path,
            "file_size": file_size,
            "error": str(e)
        }

    result = predictor.predict(features, explain=True, threshold=custom_threshold)

    verdict = result["verdict"]
    is_malware = result["is_malware"]
    prob_pct = result["malware_probability"] * 100
    raw_pct = result["raw_probability"] * 100
    risk = result["risk_level"]
    thresh_pct = result["threshold_applied"] * 100

    if verbose:
        print("=" * 74)
        print(f"  MALWAREGUARD (CODE CORTEX 3.0)  -  PE ANALYSIS")
        print("=" * 74)
        print(f"  Target File : {file_name}")
        print(f"  Full Path   : {os.path.abspath(file_path)}")
        print(f"  File Size   : {file_size:,} bytes ({file_size / 1024:.2f} KB)")
        print("-" * 74)
        verdict_badge = "[ MALWARE DETECTED ]" if is_malware else "[ BENIGN / CLEAN ]"
        print(f"  Verdict             : {verdict_badge} ({verdict})")
        print(f"  Risk Level          : {risk}")
        print(f"  Malware Probability : {prob_pct:.2f}%  (Calibrated)")
        print(f"  Raw Model Score     : {raw_pct:.2f}%")
        print(f"  Threshold Applied   : {thresh_pct:.2f}%")
        print("-" * 74)
        print("  Key PE Metrics:")
        print(f"    - Sections Count          : {int(features.get('SectionsNb', 0))}")
        print(f"    - Section Max Entropy     : {features.get('SectionsMaxEntropy', 0):.4f} / 8.0")
        print(f"    - DLL Imports Count       : {int(features.get('ImportsNbDLL', 0))}")
        print(f"    - Total Imported Functions: {int(features.get('ImportsNb', 0))}")
        print(f"    - Resources Count         : {int(features.get('ResourcesNb', 0))}")
        print(f"    - Version Info Entries    : {int(features.get('VersionInformationSize', 0))}")

        if "top_risk_factors" in result and result["top_risk_factors"]:
            print("-" * 74)
            print("  Top Contributing Factors (SHAP Feature Attribution):")
            for i, factor in enumerate(result["top_risk_factors"][:5], 1):
                direction = "[MALWARE SIGNAL]" if factor["indicates"] == "MALWARE" else "[BENIGN SIGNAL]"
                print(f"    {i}. {factor['feature']:30s} = {factor['value']:<10.2f} {direction} (impact: {factor['impact']:+.4f})")
        print("=" * 74)

    return {
        "filename": file_name,
        "path": file_path,
        "file_size": file_size,
        "verdict": verdict,
        "is_malware": is_malware,
        "malware_probability": prob_pct,
        "raw_probability": raw_pct,
        "risk_level": risk,
        "threshold_applied": thresh_pct,
        "sections": int(features.get("SectionsNb", 0)),
        "entropy": features.get("SectionsMaxEntropy", 0.0),
        "imports": int(features.get("ImportsNb", 0)),
        "error": None
    }


def scan(target_path: str, custom_threshold: float = None):
    if not os.path.exists(target_path):
        print(f"[ERROR] Target path not found: {target_path}")
        sys.exit(1)

    predictor = MalwarePredictor()

    if os.path.isfile(target_path):
        scan_single(target_path, predictor, custom_threshold=custom_threshold, verbose=True)
        return

    # Directory scanning
    all_files = []
    for entry in os.listdir(target_path):
        full = os.path.join(target_path, entry)
        if os.path.isfile(full):
            all_files.append(full)

    if not all_files:
        print(f"[!] No files found in directory: {target_path}")
        return

    print("=" * 86)
    print(f"  MALWAREGUARD (CODE CORTEX 3.0)  -  BATCH FOLDER SCAN")
    print("=" * 86)
    print(f"  Directory    : {os.path.abspath(target_path)}")
    print(f"  Files Found  : {len(all_files)}")
    print("-" * 86)

    results = []
    for f in all_files:
        res = scan_single(f, predictor, custom_threshold=custom_threshold, verbose=False)
        results.append(res)

    # Print summary table
    print(f"{'Filename':38s} | {'Size':>9s} | {'Calibrated':>10s} | {'Raw':>8s} | {'Risk Level':>14s} | {'Verdict':>8s}")
    print("-" * 96)

    malware_count = 0
    clean_count = 0
    error_count = 0

    for r in results:
        fname = r["filename"]
        if len(fname) > 36:
            fname = fname[:33] + "..."
        if r.get("error"):
            print(f"{fname:38s} | {'ERROR':>9s} | {'N/A':>10s} | {'N/A':>8s} | {'ERROR':>14s} | {'FAILED':>8s}")
            error_count += 1
        else:
            sz_str = f"{r['file_size']/1024:.1f} KB"
            prob_str = f"{r['malware_probability']:.2f}%"
            raw_str = f"{r['raw_probability']:.2f}%"
            risk_str = r['risk_level']
            verdict_str = r['verdict']
            if r['is_malware']:
                malware_count += 1
            else:
                clean_count += 1
            print(f"{fname:38s} | {sz_str:>9s} | {prob_str:>10s} | {raw_str:>8s} | {risk_str:>14s} | {verdict_str:>8s}")

    print("=" * 96)
    print("  BATCH SUMMARY:")
    print(f"    - Total Files Analyzed : {len(results)}")
    print(f"    - Threats Detected     : {malware_count} (MALWARE)")
    print(f"    - Clean Files Allowed  : {clean_count} (BENIGN)")
    if error_count > 0:
        print(f"    - Parsing Errors       : {error_count}")
    threat_rate = (malware_count / len(results)) * 100 if results else 0
    print(f"    - Threat Detection Rate: {threat_rate:.1f}%")
    print("=" * 96)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan a PE executable or directory locally using MalwareGuard ML_ENGINE.")
    parser.add_argument("target", help="Path to executable file or folder of files")
    parser.add_argument("--threshold", type=float, default=None, help="Custom probability threshold (e.g. 0.49 for balanced profile)")

    args = parser.parse_args()
    scan(args.target, custom_threshold=args.threshold)
