# GBDA: Granular-Ball-Based Domain Adaptation for EEG Emotion Recognition

Code for the paper **"GBDA: A Granular-Ball-Based Domain Adaptation Method for Efficient Cross-Subject and Cross-Session EEG Emotion Recognition"**.

GBDA compresses large-scale source-domain EEG samples into granular balls to reduce training cost, and reweights each granular ball by its transferability to the target domain to suppress negative transfer.

## Directory Structure

```
GBDA/
├── SEED/            # SEED dataset, cross-subject protocol (session 1-3)
├── SEED-IV/         # SEED-IV dataset, cross-subject protocol (session 1-3)
└── 跨session/
    ├── SEED/        # SEED dataset, cross-session protocol (session 1-3)
    └── SEEDIV/      # SEED-IV dataset, cross-session protocol (session 1-3)
```

Each session directory contains the following modules:

- `data_gen_seed.py` — dataset loading and trial generation
- `data_input.py` — DE feature input pipeline
- `grain_ball.py` — granular-ball generation (density splitting, oversized-radius splitting, quantity adjustment)
- `Main_model.py` — feature extraction network
- `Main_train.py` — training / evaluation entry
- `pre_model.py` / `pre_train.py` — pre-training scripts
- `metrics.py` — accuracy evaluation
- `utils.py` — shared utilities

## Requirements

- Python 3.9+
- PyTorch 2.3.1
- NumPy

## Usage

```bash
cd SEED/session1
python Main_train.py
```

## Citation

If you find this code useful, please cite our paper:

```
TODO: citation
```

## License

This code is released for research purposes only. Please contact the authors for commercial use.
