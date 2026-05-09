# model_bundle_v2

This directory holds the trained v2 model artifacts. It is **empty until you train**.

## Contents (after training)

```
model_bundle_v2/
├── decision_net.pt        ← PyTorch model weights + learned temperature T
├── ood_detector.json      ← EnergyOODDetector threshold (fitted on val split)
└── training_meta.json     ← Accuracy metrics, hyperparameters, training date
```

## How to train

See `src/migration/MIGRATION.md` for the full guide. Quick reference:

```bash
# Step 1 — Collect benchmark data
python -m src.training.data_collector

# Step 2 — Train the model
python -m src.training.train \
    --parquet data/benchmark_results.parquet \
    --model-features data/model_features.pkl \
    --output model_bundle_v2/decision_net.pt \
    --epochs 50

# Step 3 — Verify
make test-v2-smoke
```

## Using the trained model

```python
from pathlib import Path
from src.inference.pipeline import InferencePipeline

pipeline = InferencePipeline.from_bundle_dir(Path("model_bundle_v2/"))
result = pipeline.predict(model_facts=..., deployment_profile=...)
print(result.best_runtime, result.confidence)
```
