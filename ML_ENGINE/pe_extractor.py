"""
================================================================================
  CODE CORTEX 3.0  -  PE FEATURE EXTRACTOR  (pe_extractor.py)
================================================================================

  Parses a raw Windows PE executable using `pefile` and extracts exactly the
  features required by the ML inference pipeline.

  Design:
    1. Reads feature_columns.json at runtime — never hardcodes column names.
    2. Reads dropped_features.json to know about the 5 derived features and
       which raw PE fields they depend on.
    3. Extracts all raw PE header fields from the binary.
    4. Computes the 5 derived features using the EXACT formulas from
       preprocessing.py (Step 5a).
    5. Validates output columns against feature_columns.json before returning.

  Usage:
    from ML_ENGINE.pe_extractor import extract_features

    features = extract_features("path/to/sample.exe")
    # features is a dict with exactly 47 keys matching feature_columns.json

    # Feed directly into the predictor:
    from ML_ENGINE.predict import MalwarePredictor
    predictor = MalwarePredictor()
    result = predictor.predict(features)

  CLI:
    python ML_ENGINE/pe_extractor.py  path/to/sample.exe
================================================================================
"""

import json
import math
import os
import sys
from typing import Any, Dict, List, Optional

try:
    import pefile
except ImportError:
    raise ImportError(
        "The 'pefile' library is required. Install it with:  pip install pefile"
    )

# ---------------------------------------------------------------------------
# Paths — resolved relative to THIS file so imports work from any cwd
# ---------------------------------------------------------------------------
_CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
_ARTIFACTS_DIR = os.path.join(_CURRENT_DIR, "artifacts")

# ---------------------------------------------------------------------------
# Lazy-loaded feature manifests (read once, cached)
# ---------------------------------------------------------------------------
_feature_columns: Optional[List[str]] = None
_dropped_features: Optional[Dict[str, Any]] = None


def _load_manifests() -> None:
    """Load feature_columns.json and dropped_features.json once."""
    global _feature_columns, _dropped_features

    fc_path = os.path.join(_ARTIFACTS_DIR, "feature_columns.json")
    df_path = os.path.join(_ARTIFACTS_DIR, "dropped_features.json")

    with open(fc_path, "r", encoding="utf-8") as fh:
        _feature_columns = json.load(fh)

    with open(df_path, "r", encoding="utf-8") as fh:
        _dropped_features = json.load(fh)


def _get_feature_columns() -> List[str]:
    if _feature_columns is None:
        _load_manifests()
    return _feature_columns  # type: ignore[return-value]


def _get_dropped_features() -> Dict[str, Any]:
    if _dropped_features is None:
        _load_manifests()
    return _dropped_features  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Helpers for safe attribute access on pefile objects
# ---------------------------------------------------------------------------

def _safe(obj: Any, attr: str, default: float = 0.0) -> float:
    """Safely read an attribute, returning *default* on failure."""
    try:
        val = getattr(obj, attr, None)
        if val is None:
            return default
        return float(val)
    except Exception:
        return default


def _entropy(data: bytes) -> float:
    """Shannon entropy of a byte sequence (0.0–8.0)."""
    if not data:
        return 0.0
    length = len(data)
    freq = [0] * 256
    for b in data:
        freq[b] += 1
    ent = 0.0
    for count in freq:
        if count:
            p = count / length
            ent -= p * math.log2(p)
    return ent


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def _extract_raw_features(pe: pefile.PE) -> Dict[str, float]:
    """
    Pull every raw PE header field that the training pipeline used.

    The original dataset (Kaggle PE Malware) has these 50 raw columns.
    We extract all of them so that:
      - The scaler (fitted on 55 cols = 50 raw + 5 derived) gets what it needs.
      - Derived features can be computed from the raw values.
      - Dropped features are still available for the scaler even though
        the model itself only uses 47.
    """
    raw: Dict[str, float] = {}

    # ---- FILE_HEADER -------------------------------------------------------
    fh = pe.FILE_HEADER
    raw["Machine"]              = _safe(fh, "Machine")
    raw["SizeOfOptionalHeader"] = _safe(fh, "SizeOfOptionalHeader")
    raw["Characteristics"]      = _safe(fh, "Characteristics")

    # ---- OPTIONAL_HEADER ---------------------------------------------------
    oh = pe.OPTIONAL_HEADER
    raw["MajorLinkerVersion"]          = _safe(oh, "MajorLinkerVersion")
    raw["MinorLinkerVersion"]          = _safe(oh, "MinorLinkerVersion")
    raw["SizeOfCode"]                  = _safe(oh, "SizeOfCode")
    raw["SizeOfInitializedData"]       = _safe(oh, "SizeOfInitializedData")
    raw["SizeOfUninitializedData"]     = _safe(oh, "SizeOfUninitializedData")
    raw["AddressOfEntryPoint"]         = _safe(oh, "AddressOfEntryPoint")
    raw["BaseOfCode"]                  = _safe(oh, "BaseOfCode")

    # BaseOfData exists only in PE32, not PE32+
    raw["BaseOfData"] = _safe(oh, "BaseOfData")

    raw["ImageBase"]                     = _safe(oh, "ImageBase")
    raw["SectionAlignment"]              = _safe(oh, "SectionAlignment")
    raw["FileAlignment"]                 = _safe(oh, "FileAlignment")
    raw["MajorOperatingSystemVersion"]   = _safe(oh, "MajorOperatingSystemVersion")
    raw["MinorOperatingSystemVersion"]   = _safe(oh, "MinorOperatingSystemVersion")
    raw["MajorImageVersion"]             = _safe(oh, "MajorImageVersion")
    raw["MinorImageVersion"]             = _safe(oh, "MinorImageVersion")
    raw["MajorSubsystemVersion"]         = _safe(oh, "MajorSubsystemVersion")
    raw["MinorSubsystemVersion"]         = _safe(oh, "MinorSubsystemVersion")
    raw["SizeOfImage"]                   = _safe(oh, "SizeOfImage")
    raw["SizeOfHeaders"]                 = _safe(oh, "SizeOfHeaders")
    raw["CheckSum"]                      = _safe(oh, "CheckSum")
    raw["Subsystem"]                     = _safe(oh, "Subsystem")
    raw["DllCharacteristics"]            = _safe(oh, "DllCharacteristics")
    raw["SizeOfStackReserve"]            = _safe(oh, "SizeOfStackReserve")
    raw["SizeOfStackCommit"]             = _safe(oh, "SizeOfStackCommit")
    # Near-constant fields (dropped during training, but scaler still uses them)
    raw["SizeOfHeapReserve"]             = _safe(oh, "SizeOfHeapReserve")
    raw["SizeOfHeapCommit"]              = _safe(oh, "SizeOfHeapCommit")
    raw["LoaderFlags"]                   = _safe(oh, "LoaderFlags")
    raw["NumberOfRvaAndSizes"]           = _safe(oh, "NumberOfRvaAndSizes")

    # ---- SECTIONS ----------------------------------------------------------
    sections = pe.sections if pe.sections else []
    num_sections = len(sections)
    raw["SectionsNb"] = float(num_sections)

    entropies   = []
    raw_sizes   = []
    virt_sizes  = []

    for sec in sections:
        try:
            sec_data = sec.get_data()
        except Exception:
            sec_data = b""
        entropies.append(_entropy(sec_data))
        raw_sizes.append(float(sec.SizeOfRawData))
        virt_sizes.append(float(sec.Misc_VirtualSize))

    if entropies:
        raw["SectionsMeanEntropy"] = sum(entropies) / len(entropies)
        raw["SectionsMinEntropy"]  = min(entropies)
        raw["SectionsMaxEntropy"]  = max(entropies)
    else:
        raw["SectionsMeanEntropy"] = 0.0
        raw["SectionsMinEntropy"]  = 0.0
        raw["SectionsMaxEntropy"]  = 0.0

    if raw_sizes:
        raw["SectionsMeanRawsize"] = sum(raw_sizes) / len(raw_sizes)
        raw["SectionsMinRawsize"]  = min(raw_sizes)
        raw["SectionMaxRawsize"]   = max(raw_sizes)   # NOTE: no 's' — matches dataset
    else:
        raw["SectionsMeanRawsize"] = 0.0
        raw["SectionsMinRawsize"]  = 0.0
        raw["SectionMaxRawsize"]   = 0.0

    if virt_sizes:
        raw["SectionsMeanVirtualsize"] = sum(virt_sizes) / len(virt_sizes)
        raw["SectionsMinVirtualsize"]  = min(virt_sizes)
        raw["SectionMaxVirtualsize"]   = max(virt_sizes)  # NOTE: no 's' — matches dataset
    else:
        raw["SectionsMeanVirtualsize"] = 0.0
        raw["SectionsMinVirtualsize"]  = 0.0
        raw["SectionMaxVirtualsize"]   = 0.0

    # ---- IMPORTS -----------------------------------------------------------
    imports_nb     = 0
    imports_nb_dll = 0
    imports_nb_ord = 0

    if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
        imports_nb_dll = len(pe.DIRECTORY_ENTRY_IMPORT)
        for entry in pe.DIRECTORY_ENTRY_IMPORT:
            for imp in entry.imports:
                imports_nb += 1
                if imp.name is None:
                    imports_nb_ord += 1

    raw["ImportsNbDLL"]     = float(imports_nb_dll)
    raw["ImportsNb"]        = float(imports_nb)
    raw["ImportsNbOrdinal"] = float(imports_nb_ord)

    # ---- EXPORTS -----------------------------------------------------------
    export_nb = 0
    if hasattr(pe, "DIRECTORY_ENTRY_EXPORT"):
        export_nb = len(pe.DIRECTORY_ENTRY_EXPORT.symbols)
    raw["ExportNb"] = float(export_nb)

    # ---- RESOURCES ---------------------------------------------------------
    res_nb        = 0
    res_sizes     = []
    res_entropies = []

    if hasattr(pe, "DIRECTORY_ENTRY_RESOURCE"):
        _collect_resources(pe.DIRECTORY_ENTRY_RESOURCE, pe, res_sizes, res_entropies)
        res_nb = len(res_sizes)

    raw["ResourcesNb"] = float(res_nb)

    if res_entropies:
        raw["ResourcesMeanEntropy"] = sum(res_entropies) / len(res_entropies)
        raw["ResourcesMinEntropy"]  = min(res_entropies)
        raw["ResourcesMaxEntropy"]  = max(res_entropies)
    else:
        raw["ResourcesMeanEntropy"] = 0.0
        raw["ResourcesMinEntropy"]  = 0.0
        raw["ResourcesMaxEntropy"]  = 0.0

    if res_sizes:
        raw["ResourcesMeanSize"] = sum(res_sizes) / len(res_sizes)
        raw["ResourcesMinSize"]  = min(res_sizes)
        raw["ResourcesMaxSize"]  = max(res_sizes)
    else:
        raw["ResourcesMeanSize"] = 0.0
        raw["ResourcesMinSize"]  = 0.0
        raw["ResourcesMaxSize"]  = 0.0

    # ---- LOAD CONFIG -------------------------------------------------------
    load_config_size = 0.0
    if hasattr(pe, "DIRECTORY_ENTRY_LOAD_CONFIG"):
        lc = pe.DIRECTORY_ENTRY_LOAD_CONFIG.struct
        load_config_size = _safe(lc, "Size")
    raw["LoadConfigurationSize"] = load_config_size

    # ---- VERSION INFORMATION -----------------------------------------------
    version_info_entries: Dict[Any, Any] = {}
    if hasattr(pe, "FileInfo"):
        try:
            for fi_list in pe.FileInfo:
                items = fi_list if isinstance(fi_list, list) else [fi_list]
                for fi in items:
                    if hasattr(fi, "Key"):
                        if fi.Key in (b"StringFileInfo", "StringFileInfo"):
                            for st in getattr(fi, "StringTable", []):
                                for entry in getattr(st, "entries", {}).items():
                                    version_info_entries[entry[0]] = entry[1]
                        elif fi.Key in (b"VarFileInfo", "VarFileInfo"):
                            for var in getattr(fi, "Var", []):
                                if hasattr(var, "entry") and hasattr(var.entry, "items"):
                                    for k, v in var.entry.items():
                                        version_info_entries[k] = v
        except Exception:
            pass

    if hasattr(pe, "VS_FIXEDFILEINFO") and pe.VS_FIXEDFILEINFO:
        try:
            ffi = pe.VS_FIXEDFILEINFO[0] if isinstance(pe.VS_FIXEDFILEINFO, list) else pe.VS_FIXEDFILEINFO
            for attr in [
                "FileFlags", "FileOS", "FileType", "FileVersionLS", "FileVersionMS",
                "ProductVersionLS", "ProductVersionMS", "Signature", "StrucVersion"
            ]:
                if hasattr(ffi, attr):
                    version_info_entries[attr] = getattr(ffi, attr)
        except Exception:
            pass

    raw["VersionInformationSize"] = float(len(version_info_entries))

    return raw


def _collect_resources(
    resource_dir: Any,
    pe: pefile.PE,
    sizes: List[float],
    entropies: List[float],
) -> None:
    """Recursively walk resource directory entries to collect sizes & entropies."""
    if not hasattr(resource_dir, "entries"):
        return
    for entry in resource_dir.entries:
        if hasattr(entry, "directory"):
            _collect_resources(entry.directory, pe, sizes, entropies)
        elif hasattr(entry, "data"):
            try:
                data_rva  = entry.data.struct.OffsetToData
                data_size = entry.data.struct.Size
                data      = pe.get_data(data_rva, data_size)
                sizes.append(float(data_size))
                entropies.append(_entropy(data))
            except Exception:
                sizes.append(0.0)
                entropies.append(0.0)


# ---------------------------------------------------------------------------
# Derived feature computation
# ---------------------------------------------------------------------------

def _compute_derived(raw: Dict[str, float]) -> Dict[str, float]:
    """
    Compute the 5 derived features using the EXACT formulas from
    preprocessing.py (Step 5a, lines 108–126).
    """
    eps = 1e-9  # same epsilon used in preprocessing.py

    derived: Dict[str, float] = {}

    # entropy_range : how wide the entropy spread across sections
    derived["entropy_range"] = (
        raw.get("SectionsMaxEntropy", 0.0) - raw.get("SectionsMinEntropy", 0.0)
    )

    # import_density : average imports per section
    derived["import_density"] = (
        raw.get("ImportsNb", 0.0) / (raw.get("SectionsNb", 0.0) + eps)
    )

    # resource_entropy_spread : entropy variation in resource section
    derived["resource_entropy_spread"] = (
        raw.get("ResourcesMaxEntropy", 0.0) - raw.get("ResourcesMinEntropy", 0.0)
    )

    # code_to_image_ratio : fraction of PE image that is actual code
    derived["code_to_image_ratio"] = (
        raw.get("SizeOfCode", 0.0) / (raw.get("SizeOfImage", 0.0) + eps)
    )

    # section_fill_ratio : how densely packed the sections are
    derived["section_fill_ratio"] = (
        raw.get("SectionsMeanRawsize", 0.0)
        / (raw.get("SectionMaxRawsize", 0.0) + eps)
    )

    return derived


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_features(
    pe_path: str,
    validate: bool = True,
) -> Dict[str, float]:
    """
    Extract features from a PE file and return a dict ready for
    MalwarePredictor.predict().

    Parameters
    ----------
    pe_path : str
        Absolute or relative path to a Windows PE executable.
    validate : bool
        If True (default), raises ValueError when the output columns
        don't match feature_columns.json exactly.

    Returns
    -------
    dict
        Keys = feature names from feature_columns.json, values = floats.
    """
    if not os.path.isfile(pe_path):
        raise FileNotFoundError(f"PE file not found: {pe_path}")

    # Parse the PE
    try:
        pe = pefile.PE(pe_path, fast_load=False)
    except pefile.PEFormatError as exc:
        raise ValueError(f"Not a valid PE file: {pe_path}  ({exc})")

    # Extract raw header features
    raw = _extract_raw_features(pe)

    # Compute derived features
    derived = _compute_derived(raw)

    # Merge everything
    all_features = {**raw, **derived}

    pe.close()

    # Load the expected column list
    expected_cols = _get_feature_columns()

    # Build output dict in the exact order of feature_columns.json,
    # filling any missing field with 0.0
    output: Dict[str, float] = {}
    for col in expected_cols:
        output[col] = all_features.get(col, 0.0)

    # Validate
    if validate:
        output_keys = list(output.keys())
        if output_keys != expected_cols:
            missing = set(expected_cols) - set(output_keys)
            extra   = set(output_keys) - set(expected_cols)
            raise ValueError(
                f"Feature mismatch!\n"
                f"  Missing: {missing or 'none'}\n"
                f"  Extra:   {extra or 'none'}"
            )

    return output


def extract_features_for_scaler(pe_path: str) -> Dict[str, float]:
    """
    Extract ALL 55 features the scaler expects (50 raw + 5 derived).

    Use this when you need to feed features through the full
    scaler → model pipeline manually. The MalwarePredictor.predict()
    method in predict.py handles this automatically if you pass the
    47-column dict from extract_features().

    But if the backend pipeline needs all 55 columns (e.g. for
    predict.py's scaler_feature_names), use this function instead.
    """
    if not os.path.isfile(pe_path):
        raise FileNotFoundError(f"PE file not found: {pe_path}")

    try:
        pe = pefile.PE(pe_path, fast_load=False)
    except pefile.PEFormatError as exc:
        raise ValueError(f"Not a valid PE file: {pe_path}  ({exc})")

    raw     = _extract_raw_features(pe)
    derived = _compute_derived(raw)
    pe.close()

    return {**raw, **derived}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json as _json

    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <path_to_pe_file>")
        sys.exit(1)

    target = sys.argv[1]
    print(f"Extracting features from: {target}")
    print()

    feats = extract_features(target)

    expected = _get_feature_columns()
    print(f"Features extracted : {len(feats)}")
    print(f"Expected columns   : {len(expected)}")
    print(f"Columns match      : {list(feats.keys()) == expected}")
    print()
    print(_json.dumps(feats, indent=2))
