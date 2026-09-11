# VAMO_PDE

VAMO PDE experiments for the retained PDE suites:

- `cfd2d`: 2D compressible fluid dynamics
- `euler1d`: 1D compressible Euler
- `ns2d`: periodic 2D Navier-Stokes

Shared model and training utilities live at the package root, including `dataset_io.py`, `fno.py`, `fno_trainer.py`, `latent_markov.py`, `latent_markov_trainer.py`, and `latent_flow_VAE.py`.

## Smoke Checks

```bash
python -m cfd2d.cfd_data --help
python -m euler1d.euler_data --help
python -m ns2d.ns2d_data --help
```

## Notes

Generated datasets, checkpoints, outputs, plots, logs, and cache files are intentionally ignored by git.
