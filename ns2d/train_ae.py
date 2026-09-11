"""
Training entrypoint for the multi-channel hidden-space 2D Navier-Stokes
latent Markov autoencoder on the periodic dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from dataset_io import load_dataset_splits
    from latent_markov import (
        FNOProximalStepSimulator2D,
        LatentMarkovModel2D,
        ProximalStepSimulator2D,
        StateDecoder2D,
        StateEncoder2D,
    )
    from latent_markov_trainer import LatentMarkovTrainer2D
except ImportError:
    from dataset_io import load_dataset_splits
    from latent_markov import (
        FNOProximalStepSimulator2D,
        LatentMarkovModel2D,
        ProximalStepSimulator2D,
        StateDecoder2D,
        StateEncoder2D,
    )
    from latent_markov_trainer import LatentMarkovTrainer2D


class PeriodicTrajectoryTensorDataset(Dataset):
    def __init__(self, f_data: torch.Tensor, u0_data: torch.Tensor, u_traj_data: torch.Tensor):
        if f_data.dim() not in (3, 4):
            raise ValueError("f_data must have shape (N,X,Y) or (N,Cf,X,Y)")
        if u0_data.dim() not in (3, 4):
            raise ValueError("u0_data must have shape (N,X,Y) or (N,C,X,Y)")
        if u_traj_data.dim() not in (4, 5):
            raise ValueError("u_traj_data must have shape (N,T,X,Y) or (N,T,C,X,Y)")
        if int(u0_data.shape[0]) != int(u_traj_data.shape[0]) or int(f_data.shape[0]) != int(u_traj_data.shape[0]):
            raise ValueError("f_data, u0_data, and u_traj_data must have the same sample count")
        self.f_data = f_data
        self.u0_data = u0_data
        self.u_traj_data = u_traj_data

    def __len__(self) -> int:
        return int(self.u0_data.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"f": self.f_data[idx], "u0": self.u0_data[idx], "u_traj": self.u_traj_data[idx]}


class PeriodicStepDataset(Dataset):
    def __init__(self, f_data: torch.Tensor, u_traj_data: torch.Tensor):
        if f_data.dim() not in (3, 4):
            raise ValueError("f_data must have shape (N,X,Y) or (N,Cf,X,Y)")
        if u_traj_data.dim() not in (4, 5):
            raise ValueError("u_traj_data must have shape (N,T,X,Y) or (N,T,C,X,Y)")
        if int(f_data.shape[0]) != int(u_traj_data.shape[0]):
            raise ValueError("f_data and u_traj_data must have the same sample count")
        self.f_data = f_data
        self.u_traj_data = u_traj_data
        self.n_samples = int(u_traj_data.shape[0])
        self.n_steps = int(u_traj_data.shape[1] - 1)

    def __len__(self) -> int:
        return self.n_samples * self.n_steps

    def __getitem__(self, idx: int):
        i = idx // self.n_steps
        k = idx % self.n_steps
        return self.u_traj_data[i, k], self.u_traj_data[i, k + 1], self.f_data[i]


def build_step_dataset(split: dict) -> PeriodicStepDataset:
    return PeriodicStepDataset(split["f"], split["u_traj"])


def build_traj_dataset(split: dict) -> PeriodicTrajectoryTensorDataset:
    return PeriodicTrajectoryTensorDataset(split["f"], split["u0"], split["u_traj"])


def set_seed(seed: int, seed_cuda: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if seed_cuda:
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multi-channel hidden-space AE on periodic 2D Navier-Stokes data")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to cached periodic dataset (.pt)")
    parser.add_argument("--n-train", type=int, default=4000)
    parser.add_argument("--n-val", type=int, default=500)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-t-start", type=float, default=None)
    parser.add_argument("--train-t-end", type=float, default=None)

    parser.add_argument("--state-channels", type=int, default=None, help="Override state channel count; inferred from data by default.")
    parser.add_argument("--channel-weights", type=float, nargs="*", default=None, help="Optional per-state-channel loss weights.")
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--enc-blocks", type=int, default=4)
    parser.add_argument("--dec-blocks", type=int, default=4)
    parser.add_argument("--prox-blocks", type=int, default=6)
    parser.add_argument("--prox-simulator-type", type=str, default="cnn", choices=["cnn", "fno"])
    parser.add_argument("--fno-modes-x", type=int, default=16)
    parser.add_argument("--fno-modes-y", type=int, default=16)
    parser.add_argument("--disable-fno-grid", action="store_true")
    parser.add_argument("--use-dt-channel", action="store_true")
    parser.add_argument("--disable-forcing-channel", action="store_true")
    parser.add_argument("--disable-u-grad-feature", action="store_true")

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-step-size", type=int, default=100)
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--rollout-delta-clip", type=float, default=10.0)
    parser.add_argument("--lambda-recon", type=float, default=1.0)
    parser.add_argument("--lambda-spec", type=float, default=1.0)
    parser.add_argument("--spectral-s", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-epoch-pbar", action="store_true")
    parser.add_argument("--output-dir", type=str, default="ns2d/outputs_ae")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _infer_state_channels(split: dict) -> int:
    u0 = split["u0"]
    return 1 if u0.dim() == 3 else int(u0.shape[1])


def _spatial_shape(split: dict) -> tuple[int, int]:
    u0 = split["u0"]
    return (int(u0.shape[-2]), int(u0.shape[-1]))


def _channel_weights_from_split(train_split: dict, state_channels: int, provided) -> Optional[torch.Tensor]:
    if provided is not None and len(provided) > 0:
        weights = torch.tensor(provided, dtype=torch.float32)
        if int(weights.numel()) != int(state_channels):
            raise ValueError(f"--channel-weights must have {state_channels} entries")
        return weights / weights.mean()
    if state_channels == 1:
        return None
    u = train_split["u_traj"]
    if u.dim() != 5:
        return None
    flat = u.permute(2, 0, 1, 3, 4).reshape(state_channels, -1)
    weights = 1.0 / flat.var(dim=1).clamp(min=1e-8)
    return weights / weights.mean()


def _slice_time_window(split: dict, time_values: np.ndarray, t_start: float | None, t_end: float | None, split_name: str):
    if t_start is None and t_end is None:
        return split, time_values, (0, int(time_values.shape[0] - 1))
    start = float(time_values[0]) if t_start is None else float(t_start)
    end = float(time_values[-1]) if t_end is None else float(t_end)
    tol = 1e-8 + 1e-6 * max(1.0, abs(float(time_values[-1] - time_values[0])))
    if start < float(time_values[0]) - tol or end > float(time_values[-1]) + tol:
        raise ValueError(f"Requested time window [{start},{end}] is outside stored range [{float(time_values[0])},{float(time_values[-1])}]")
    if end <= start:
        raise ValueError(f"--train-t-end must be greater than --train-t-start, got [{start},{end}]")
    i0 = int(np.searchsorted(time_values, start - tol, side="left"))
    i1 = int(np.searchsorted(time_values, end + tol, side="right") - 1)
    i0 = max(0, min(i0, int(time_values.shape[0] - 1)))
    i1 = max(0, min(i1, int(time_values.shape[0] - 1)))
    if i1 <= i0:
        raise ValueError(f"Time window [{start},{end}] for {split_name} keeps fewer than two snapshots; nearest indices are {i0}:{i1}")
    u_traj = split["u_traj"][:, i0 : i1 + 1]
    sliced = dict(split)
    sliced["u_traj"] = u_traj
    sliced["u0"] = u_traj[:, 0].clone()
    return sliced, time_values[i0 : i1 + 1], (i0, i1)


def _build_model(n_x: int, n_y: int, h_x: float, h_y: float, dt: float, state_channels: int, args: argparse.Namespace) -> LatentMarkovModel2D:
    boundary_condition = "periodic"
    use_forcing_channel = not args.disable_forcing_channel
    encoder = StateEncoder2D(
        n_x=n_x,
        n_y=n_y,
        latent_channels=args.latent_channels,
        hidden_channels=args.hidden_channels,
        n_blocks=args.enc_blocks,
        use_grad_features=not args.disable_u_grad_feature,
        boundary_condition=boundary_condition,
        state_channels=state_channels,
    )
    decoder = StateDecoder2D(
        n_x=n_x,
        n_y=n_y,
        latent_channels=args.latent_channels,
        hidden_channels=args.hidden_channels,
        n_blocks=args.dec_blocks,
        boundary_condition=boundary_condition,
        state_channels=state_channels,
    )
    if args.prox_simulator_type == "cnn":
        prox_step = ProximalStepSimulator2D(
            n_x=n_x,
            n_y=n_y,
            latent_channels=args.latent_channels,
            hidden_channels=args.hidden_channels,
            n_blocks=args.prox_blocks,
            use_forcing_channel=use_forcing_channel,
            use_dt_channel=args.use_dt_channel,
            default_dt=dt,
            boundary_condition=boundary_condition,
        )
    elif args.prox_simulator_type == "fno":
        prox_step = FNOProximalStepSimulator2D(
            n_x=n_x,
            n_y=n_y,
            latent_channels=args.latent_channels,
            width=args.hidden_channels,
            n_layers=args.prox_blocks,
            modes_x=args.fno_modes_x,
            modes_y=args.fno_modes_y,
            use_forcing_channel=use_forcing_channel,
            use_dt_channel=args.use_dt_channel,
            use_grid_features=not args.disable_fno_grid,
            default_dt=dt,
            boundary_condition=boundary_condition,
        )
    else:
        raise ValueError(f"Unsupported prox-simulator-type: {args.prox_simulator_type}")
    return LatentMarkovModel2D(encoder=encoder, decoder=decoder, transition=prox_step)


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed, seed_cuda=not args.cpu)
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")

    splits = load_dataset_splits(args.dataset_path, map_location="cpu")
    train_split = splits["train"]
    val_split = splits["val"]
    test_split = splits["test"]

    sizes = (int(train_split["u0"].shape[0]), int(val_split["u0"].shape[0]), int(test_split["u0"].shape[0]))
    if sizes != (args.n_train, args.n_val, args.n_test):
        raise ValueError(f"Dataset split sizes do not match CLI arguments: dataset={sizes} vs args={(args.n_train, args.n_val, args.n_test)}")

    meta = splits.get("meta", {})
    n_x, n_y = _spatial_shape(train_split)
    state_channels = _infer_state_channels(train_split) if args.state_channels is None else int(args.state_channels)
    n_steps = int(train_split["u_traj"].shape[1] - 1)
    t_final = float(meta.get("t_final", 1.0))
    t_start = float(meta.get("stored_t_start", meta.get("warmup_time", 0.0)))
    stored_horizon = float(meta.get("stored_time_horizon", t_final - t_start))
    h_x = 1.0 / float(n_x)
    h_y = 1.0 / float(n_y)
    dt = float(meta.get("record_dt", stored_horizon / float(n_steps)))
    time_values_full = t_start + np.arange(n_steps + 1, dtype=np.float64) * dt

    train_split, time_values, window_idx = _slice_time_window(train_split, time_values_full, args.train_t_start, args.train_t_end, "train")
    val_split, _, _ = _slice_time_window(val_split, time_values_full, args.train_t_start, args.train_t_end, "val")
    test_split, _, _ = _slice_time_window(test_split, time_values_full, args.train_t_start, args.train_t_end, "test")
    n_steps_full = n_steps
    n_steps = int(train_split["u_traj"].shape[1] - 1)
    t_window_start = float(time_values[0])
    t_window_end = float(time_values[-1])
    channel_weights = _channel_weights_from_split(train_split, state_channels, args.channel_weights)

    print(f"Device: {device}")
    print(f"Loaded dataset: {args.dataset_path}")
    print(
        f"Grid from data: n_x={n_x}, n_y={n_y}, state_channels={state_channels}, n_steps={n_steps}, "
        f"stored_time=[{t_start:.6f},{t_start + dt * n_steps_full:.6f}], dt={dt:.6f}, "
        f"train_time=[{t_window_start:.6f},{t_window_end:.6f}] "
        f"(indices {window_idx[0]}:{window_idx[1]} of {n_steps_full})"
    )
    if channel_weights is not None:
        print(f"Channel weights: {channel_weights.tolist()}")

    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    train_step_loader = DataLoader(build_step_dataset(train_split), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=seed_worker, generator=loader_generator)
    val_step_loader = DataLoader(build_step_dataset(val_split), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, worker_init_fn=seed_worker)
    test_step_loader = DataLoader(build_step_dataset(test_split), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, worker_init_fn=seed_worker)
    val_traj_loader = DataLoader(build_traj_dataset(val_split), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, worker_init_fn=seed_worker)
    test_traj_loader = DataLoader(build_traj_dataset(test_split), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, worker_init_fn=seed_worker)

    model = _build_model(n_x=n_x, n_y=n_y, h_x=h_x, h_y=h_y, dt=dt, state_channels=state_channels, args=args)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    args_dict = vars(args).copy()
    args_dict.update({
        "state_channels_used": int(state_channels),
        "channel_weights_used": None if channel_weights is None else channel_weights.tolist(),
        "train_time_start_used": t_window_start,
        "train_time_end_used": t_window_end,
        "train_time_index_start": int(window_idx[0]),
        "train_time_index_end": int(window_idx[1]),
        "train_steps_used": int(n_steps),
    })
    with open(os.path.join(run_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(args_dict, f, indent=2)

    trainer = LatentMarkovTrainer2D(
        model=model,
        dt=dt,
        h_x=h_x,
        h_y=h_y,
        lambda_recon=args.lambda_recon,
        lambda_spec=args.lambda_spec,
        spectral_s=args.spectral_s,
        channel_weights=channel_weights,
        lr=args.lr,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        rollout_delta_clip=args.rollout_delta_clip,
        device=device,
        output_dir=run_dir,
        show_epoch_pbar=not args.no_epoch_pbar,
    )

    if args.dry_run:
        print("Dry run val metrics:", trainer.validate(val_step_loader, traj_loader=val_traj_loader))
        print("Dry run test metrics:", trainer.validate(test_step_loader, traj_loader=test_traj_loader))
        return

    print(
        f"Training config: epochs={args.epochs}, lr={args.lr}, lr_step_size={args.lr_step_size}, "
        f"lr_gamma={args.lr_gamma}, lambda_recon={args.lambda_recon}, lambda_spec={args.lambda_spec}, "
        f"spectral_s={args.spectral_s}, rollout_delta_clip={args.rollout_delta_clip}, "
        f"prox_type={args.prox_simulator_type}, fno_modes=({args.fno_modes_x},{args.fno_modes_y}), "
        f"epoch_pbar={not args.no_epoch_pbar}, output={run_dir}"
    )
    history = trainer.fit(
        train_step_loader=train_step_loader,
        val_step_loader=val_step_loader,
        val_traj_loader=val_traj_loader,
        epochs=args.epochs,
        eval_interval=args.eval_interval,
        checkpoint_interval=args.checkpoint_interval,
    )
    print("Training complete.")
    print("Last train metrics:", history["train"][-1])
    if history["val"]:
        print("Last val metrics:", history["val"][-1])
    print("Test metrics:", trainer.validate(test_step_loader, traj_loader=test_traj_loader))
    print(f"Saved training artifacts to: {run_dir}")


if __name__ == "__main__":
    main(parse_args())
