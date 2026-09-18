"""
================================================================================
  CODE CORTEX 3.0  -  RISK ENGINE (risk_engine.py)
================================================================================

  Single source of truth for threat assessment, risk scoring, and heuristic
  threat analysis.

  Calibrated for the locked 47-feature PE feature contract.

  Risk Taxonomy:
    - TRUSTED_VENDOR  : Cryptographically verified signature from trusted vendor
    - CLEAN           : Calibrated malware probability < 0.05
    - LOW_RISK        : 0.05 <= Probability < Operating Threshold (0.10)
    - SUSPICIOUS      : Operating Threshold <= Probability < 0.50
    - HIGH_THREAT     : 0.50 <= Probability < 0.85
    - CRITICAL_MALWARE: Probability >= 0.85
================================================================================
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(CURRENT_DIR, "artifacts")


class MalwareRiskEngine:
    """
    Unified Risk Engine for Code Cortex 3.0 & MalwareGuard.
    """

    def __init__(self, artifacts_dir: Optional[str] = None):
        self.artifacts_dir = artifacts_dir or ARTIFACTS_DIR
        self.operating_threshold = self._load_threshold()

        # Heuristic thresholds tuned for the 47 PE feature contract
        self.heuristic_thresholds = {
            "SectionsMaxEntropy": 7.2,
            "SectionsMeanEntropy": 6.8,
            "entropy_range": 4.5,
            "ImportsNbDLL": 15,
            "ImportsNb": 200,
            "ResourcesMaxEntropy": 7.0,
            "code_to_image_ratio": 0.85,
        }

    def _load_threshold(self) -> float:
        thresh_path = os.path.join(self.artifacts_dir, "threshold_config.json")
        if os.path.exists(thresh_path):
            try:
                with open(thresh_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    return float(cfg.get("operating_threshold", 0.10))
            except Exception:
                pass
        return 0.10

    def compute_risk_level(self, malware_probability: float, is_trusted_vendor: bool = False) -> str:
        """Assign 6-tier risk level based on probability and authenticode status."""
        if is_trusted_vendor:
            return "TRUSTED_VENDOR"
        if malware_probability < 0.05:
            return "CLEAN"
        elif malware_probability < self.operating_threshold:
            return "LOW_RISK"
        elif malware_probability < 0.50:
            return "SUSPICIOUS"
        elif malware_probability < 0.85:
            return "HIGH_THREAT"
        else:
            return "CRITICAL_MALWARE"

    def compute_heuristic_score(self, features: Dict[str, Union[int, float]]) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Evaluate heuristic risk factors based on PE features.
        Returns a normalized score (0.0 - 1.0) and list of threat factors.
        """
        factors: List[Dict[str, Any]] = []
        score = 0.0
        max_score = 0.0

        # 1. Section Max Entropy (packing/encryption indicator)
        max_ent = float(features.get("SectionsMaxEntropy", 0.0))
        if max_ent > 0.0:
            weight = 0.35
            max_score += weight
            if max_ent > self.heuristic_thresholds["SectionsMaxEntropy"]:
                norm = min((max_ent - 7.0) / 1.0, 1.0)
                score += norm * weight
                factors.append({
                    "factor": "High Section Entropy",
                    "value": round(max_ent, 4),
                    "description": f"Section max entropy {max_ent:.2f}/8.0 indicates possible packing or encryption",
                    "severity": "HIGH" if max_ent > 7.5 else "MEDIUM"
                })

        # 2. Entropy Range
        ent_range = float(features.get("entropy_range", 0.0))
        if ent_range > self.heuristic_thresholds["entropy_range"]:
            weight = 0.15
            max_score += weight
            score += min((ent_range - 4.0) / 3.0, 1.0) * weight
            factors.append({
                "factor": "Wide Section Entropy Spread",
                "value": round(ent_range, 4),
                "description": f"Wide variation in section entropy ({ent_range:.2f}) suggests mixed packed/code sections",
                "severity": "MEDIUM"
            })

        # 3. DLL Import Count
        dll_count = float(features.get("ImportsNbDLL", 0.0))
        if dll_count > self.heuristic_thresholds["ImportsNbDLL"]:
            weight = 0.20
            max_score += weight
            norm = min((dll_count - 15) / 35, 1.0)
            score += norm * weight
            factors.append({
                "factor": "Elevated DLL Import Count",
                "value": int(dll_count),
                "description": f"Executable imports {int(dll_count)} DLLs, indicating complex runtime dependencies",
                "severity": "MEDIUM" if dll_count < 30 else "HIGH"
            })

        # 4. Resource Entropy
        res_ent = float(features.get("ResourcesMaxEntropy", 0.0))
        if res_ent > self.heuristic_thresholds["ResourcesMaxEntropy"]:
            weight = 0.15
            max_score += weight
            norm = min((res_ent - 6.5) / 1.5, 1.0)
            score += norm * weight
            factors.append({
                "factor": "High Resource Section Entropy",
                "value": round(res_ent, 4),
                "description": f"Embedded resources exhibit high entropy ({res_ent:.2f}), possibly containing encrypted payload",
                "severity": "HIGH"
            })

        # 5. Code to Image Ratio
        code_ratio = float(features.get("code_to_image_ratio", 0.0))
        if code_ratio > self.heuristic_thresholds["code_to_image_ratio"]:
            weight = 0.15
            max_score += weight
            score += weight * 0.8
            factors.append({
                "factor": "Disproportionate Code-to-Image Ratio",
                "value": round(code_ratio, 4),
                "description": f"Code section represents {code_ratio * 100:.1f}% of image size",
                "severity": "LOW"
            })

        final_score = score / max_score if max_score > 0 else 0.0
        return round(final_score, 4), factors

    def assess_threat(
        self,
        prediction: Dict[str, Any],
        features: Dict[str, Union[int, float]],
        authenticode: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Combine ML prediction with heuristic evaluation.
        """
        is_trusted = False
        if authenticode and authenticode.get("valid") and authenticode.get("trusted"):
            is_trusted = True

        p_malware = float(prediction.get("malware_probability", 0.0))
        risk_level = self.compute_risk_level(p_malware, is_trusted_vendor=is_trusted)
        heuristic_score, threat_factors = self.compute_heuristic_score(features)

        if is_trusted:
            vendor = authenticode.get("vendor", "Trusted Entity")
            combined_score = 0
            summary = f"Valid Authenticode digital signature verified from trusted vendor ({vendor}). Marked BENIGN."
        else:
            # Combined score out of 100
            combined_score = int(round((0.75 * p_malware + 0.25 * heuristic_score) * 100))
            is_mal = p_malware >= self.operating_threshold
            summary = (
                f"Prediction: {'MALWARE' if is_mal else 'BENIGN'} "
                f"(Calibrated Malware Probability: {p_malware * 100:.2f}%, Risk: {risk_level}). "
                f"{len(threat_factors)} heuristic threat factor(s) identified."
            )

        # UI friendly risk level mapping
        ui_risk_map = {
            "TRUSTED_VENDOR": "LOW",
            "CLEAN": "LOW",
            "LOW_RISK": "LOW",
            "SUSPICIOUS": "MEDIUM",
            "HIGH_THREAT": "HIGH",
            "CRITICAL_MALWARE": "CRITICAL"
        }

        return {
            "is_malicious": (p_malware >= self.operating_threshold) and not is_trusted,
            "malware_probability": p_malware,
            "risk_level": risk_level,
            "ui_risk": ui_risk_map.get(risk_level, "MEDIUM"),
            "heuristic_score": heuristic_score,
            "combined_risk_score": combined_score,
            "threat_factors": threat_factors,
            "assessment_summary": summary,
            "operating_threshold": self.operating_threshold
        }
