#!/usr/bin/env python3
import sys
sys.path.insert(0, 'D:\\nft2\\NFT')

from ai_engine import api

print("[*] Importing api module...")
print("[*] api.model is None?", api.model is None)
print("[*] api.scaler is None?", api.scaler is None)

if api.model is not None:
    print("[OK] Model is loaded in module!")
    # Try a prediction
    import numpy as np
    features = np.array([[5000.0, 30, 0.0, 0]])
    if api.scaler is not None:
        features_scaled = api.scaler.transform(features)
        pred = api.model.predict(features_scaled)[0]
        score = api.model.decision_function(features_scaled)[0]
    else:
        pred = api.model.predict(features)[0]
        score = api.model.decision_function(features)[0]
    print("[*] Prediction:", pred, "Score:", score)
else:
    print("[!] Model is NOT loaded!")
