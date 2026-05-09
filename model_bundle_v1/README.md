# Model Bundle — Versioned Deployment Unit
# ==========================================
# This directory is an IMMUTABLE deployment bundle.
# It contains everything needed to serve a single model version:
#
#   model_bundle_v<N>/
#     ├── model.onnx           ← model artefact (or .pt / .tflite / saved_model/)
#     ├── spec.json            ← ModelSpec definition (name, version, shapes, dtypes)
#     ├── model_hashes.sha256  ← SHA256 of model.onnx (or directory)
#     ├── fingerprint.json     ← environment that produced this bundle
#     └── README.md            ← this file
#
# BUNDLE CONTRACT:
#   1. The spec.json MUST match the ModelSpec registered in ModelRegistry.
#   2. The model_hashes.sha256 MUST be verified before loading.
#   3. The fingerprint.json records the python/numpy/ort/torch/tf versions
#      used to generate and validate this bundle.
#   4. Any change to the model file requires:
#        a. Re-generating model_hashes.sha256
#        b. Re-running golden tests
#        c. Explicit version bump (bundle_v<N+1>)
#
# LOAD PROCEDURE (in production):
#
#   import json
#   from src.models import ModelRegistry, ModelSpec
#   from scripts.verify_hashes import sha256_file
#
#   bundle_dir = Path("model_bundle_v1")
#
#   # 1. Verify hash
#   manifest = (bundle_dir / "model_hashes.sha256").read_text().split()
#   expected_hash = manifest[0]
#   actual_hash   = sha256_file(bundle_dir / "model.onnx")
#   assert actual_hash == expected_hash, "Artefact tampered!"
#
#   # 2. Load spec
#   spec_data = json.loads((bundle_dir / "spec.json").read_text())
#   spec = ModelSpec(**spec_data)
#
#   # 3. Register and resolve
#   ModelRegistry.register("my_model_v1", str(bundle_dir / "model.onnx"), spec)
#   model = ModelRegistry.resolve("my_model_v1")
#
# UPGRADE POLICY:
#   Never modify an existing bundle directory.
#   Create a new bundle_v<N+1> with the new model and regenerated hashes.
