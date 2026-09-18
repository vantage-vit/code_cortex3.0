#!/usr/bin/env python3
"""Debug script to check feature matching."""

import pandas as pd

# Load a sample from the dataset to test with
df = pd.read_csv("dataset/malware.csv", sep='|')
print("Dataset shape:", df.shape)

# Get first row and remove non-feature columns
sample_row = df.iloc[0]
features = sample_row.drop(['legitimate', 'Name', 'md5']).to_dict()

# Convert numpy types to Python types for JSON serialization
features = {k: (int(v) if hasattr(v, 'item') and 'int' in str(type(v))
               else float(v) if hasattr(v, 'item') and 'float' in str(type(v))
               else v) for k, v in features.items()}

print("Number of features:", len(features))
print("Feature names:", sorted(features.keys()))

# Check for problematic fields
problem_fields = ['SectionsMaxVirtualsize', 'SectionsMaxRawsize']
for field in problem_fields:
    if field in features:
        print(f"✓ {field}: {features[field]}")
    else:
        print(f"✗ {field}: MISSING")
        # Check for similar names
        similar = [k for k in features.keys() if field in k or k in field]
        if similar:
            print(f"  Similar fields: {similar}")

# Check what the API expects
import joblib
preprocessor_data = joblib.load('ml/models/preprocessor.pkl')
expected_features = preprocessor_data['feature_columns']
print("\nExpected features count:", len(expected_features))

missing = set(expected_features) - set(features.keys())
extra = set(features.keys()) - set(expected_features)

print("Missing features:", missing)
print("Extra features:", extra)

if missing:
    print("\nFirst 5 missing features:")
    for i, f in enumerate(list(missing)[:5]):
        print(f"  {f}")