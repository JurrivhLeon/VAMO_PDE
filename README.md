# VAMO: Variational Autoencoding Markov Operator

Official implementation of **VAMO (Variational Autoencoding Markov Operator)** for long-horizon prediction of time-dependent partial differential equations.

VAMO models PDE dynamics in a variational latent space and combines spatially resolved latent representations, structured Gaussian perturbations, and a neural-operator transition model for autoregressive rollout.

This repository accompanies the manuscript:

**Stable by Construction: Variational Latent Markov Operators for Long-Horizon PDE Prediction**

Paper: [arXiv:2609.16621](https://arxiv.org/abs/2609.16621)

## Repository Structure

```text
.
├── cfd2d/                    # 2D compressible fluid-dynamics benchmark
│   └── utils/                # Plotting and analysis utilities
├── euler1d/                  # 1D compressible Euler benchmark
│   └── utils/                # Plotting and analysis utilities
├── ns2d/                     # Periodic 2D incompressible Navier-Stokes benchmark
│   └── utils/                # Plotting and analysis utilities
├── dataset_io.py             # Dataset serialization helpers
├── fno.py                    # Fourier neural operator baseline modules
├── fno_trainer.py            # FNO training utilities
├── latent_flow_VAE.py        # VAMO variational latent-flow modules
├── latent_markov.py          # Latent Markov autoencoder modules
├── latent_markov_trainer.py  # Latent Markov training utilities
├── utils.py                  # Shared numerical and plotting helpers
└── README.md
```

## Installation

We recommend creating a separate Python environment.

```bash
conda create -n vamo python=3.10
conda activate vamo
pip install torch numpy scipy matplotlib tqdm
```

If a pinned `requirements.txt` file is added later, install it with:

```bash
pip install -r requirements.txt
```

## Data

The current repository focuses on three PDE benchmarks:

- `cfd2d`: two-dimensional compressible fluid dynamics;
- `euler1d`: one-dimensional compressible Euler equations;
- `ns2d`: periodic two-dimensional incompressible Navier-Stokes equations.

Dataset-generation scripts are provided inside each PDE directory. Quick entry points are:

```bash
python -m cfd2d.cfd_data --help
python -m euler1d.euler_data --help
python -m ns2d.ns2d_data --help
```

Generated datasets and local experiment settings are intentionally left out of version control.

## Training

Each PDE benchmark contains experiment-specific training scripts. For NS2D, the main entry points are:

```bash
python -m ns2d.train_vamo --dataset-path path/to/dataset.pt
python -m ns2d.train_ae --dataset-path path/to/dataset.pt
python -m ns2d.train_fno --dataset-path path/to/dataset.pt
```

Analogous VAMO, autoencoder, and FNO scripts are provided for the other retained PDE benchmarks where applicable. Script arguments define the model architecture, optimization settings, latent-noise parameters, rollout horizon, and dataset paths.

## Evaluation

Long-horizon rollout evaluation can be performed with the corresponding evaluation scripts. For NS2D:

```bash
python -m ns2d.eval_vamo --dataset-path path/to/dataset.pt --checkpoint path/to/checkpoint.pt
python -m ns2d.eval_ae --dataset-path path/to/dataset.pt --checkpoint-path path/to/checkpoint.pt
python -m ns2d.eval_fno --dataset-path path/to/dataset.pt --checkpoint-path path/to/checkpoint.pt
```

The evaluation code reports rollout prediction errors over horizons extending beyond those observed during training. Plotting and diagnostic scripts are organized under each PDE-specific `utils/` directory.

## Reproducing the Paper Experiments

The repository contains code paths for the main experiments in the paper, including:

- comparisons with deterministic neural-operator baselines;
- long-horizon PDE rollout experiments;
- variational ablation studies;
- comparisons with autoregressive curriculum training;
- supplementary visualization and diagnostic experiments.

Detailed commands for reproducing individual tables and figures will be added here.

## Citation

If you find this work useful, please cite:

```bibtex
@article{liao2026vamo,
  title   = {Stable by Construction: Variational Latent Markov Operators for Long-Horizon PDE Prediction},
  author  = {Liao, Junyi and Guilleminot, Johann and Tarokh, Vahid},
  journal = {arXiv preprint arXiv:2609.16621},
  year    = {2026}
}
```

## Notes

Generated datasets, checkpoints, outputs, plots, logs, and cache files are intentionally ignored by git. Please avoid committing large generated artifacts directly to the repository.
