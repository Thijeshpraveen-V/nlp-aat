| Setting | Val micro F1 | Test micro F1 | Test macro F1 |
|---|---|---|---|
| Final fine-tuned (lr=3e-05, bs=8, epochs=10, sliding window) | 42.0 | 44.8 | 44.5 |
| + regex rule layer (EMAIL, GRADUATION_YEAR) | — | 45.6 | 45.1 |
| Epochs = 5 (instead of 10) | 31.0 | 31.1 | 29.8 |
| + class-weighted loss | 35.1 | 33.9 | 35.8 |
| + oversampling rare-entity resumes | 44.8 | 43.2 | 45.2 |
| Plain truncation (512) instead of sliding window | 35.8 | 38.5 | 34.4 |
| No text normalisation | 42.3 | 44.7 | 44.6 |

| Grid run | Val micro F1 | Best epoch | Test micro F1 |
|---|---|---|---|
| lr=2e-05, bs=16 | 27.8 | 10 | 33.3 |
| lr=2e-05, bs=8 | 39.3 | 8 | 42.0 |
| lr=3e-05, bs=16 | 36.0 | 9 | 38.9 |
| lr=3e-05, bs=8 | 42.4 | 10 | 44.8 |
| lr=5e-05, bs=16 | 40.3 | 9 | 43.8 |
| lr=5e-05, bs=8 | 42.2 | 6 | 45.9 |
