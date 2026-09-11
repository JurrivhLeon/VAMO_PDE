"""
Evaluation and sample-visualization script for multi-channel periodic 2D
Navier-Stokes latent Markov autoencoder models.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

try:
    from dataset_io import load_dataset_splits
    from latent_markov import (
        FNOProximalStepSimulator2D,
        LatentMarkovModel2D,
        ProximalStepSimulator2D,
        StateDecoder2D,
        StateEncoder2D,
    )
    from latent_markov_trainer import rollout_latent_markov_2d
except ImportError:
    from dataset_io import load_dataset_splits
    from latent_markov import (
        FNOProximalStepSimulator2D,
        LatentMarkovModel2D,
        ProximalStepSimulator2D,
        StateDecoder2D,
        StateEncoder2D,
    )
    from latent_markov_trainer import rollout_latent_markov_2d


def set_seed(seed: int, seed_cuda: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if seed_cuda:
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate multi-channel periodic 2D Navier-Stokes hidden-space AE")
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--n-plot-samples", type=int, default=6)
    parser.add_argument("--plot-channel", type=int, default=0)
    parser.add_argument("--snapshot-times", type=str, default="0,2,4,6,8,10")
    parser.add_argument("--delta-clip", type=float, default=10.0)
    parser.add_argument("--output-dir", type=str, default="ns2d/outputs_ae/eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)

    parser.add_argument("--state-channels", type=int, default=None)
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
    return parser.parse_args()


def _time_metadata(meta: Dict, n_steps: int) -> tuple[float, float, float, np.ndarray]:
    t_start = float(meta.get("stored_t_start", meta.get("warmup_time", 0.0)))
    if "record_dt" in meta:
        dt = float(meta["record_dt"])
    elif "stored_time_horizon" in meta:
        dt = float(meta["stored_time_horizon"]) / float(n_steps)
    else:
        dt = float(meta.get("t_final", 1.0)) / float(n_steps)
    if dt <= 0.0:
        raise ValueError(f"Dataset record_dt/dt must be positive, got {dt}")
    t_end = float(meta.get("stored_t_final", t_start + dt * float(n_steps)))
    time_values = t_start + np.arange(n_steps + 1, dtype=np.float64) * dt
    if abs(time_values[-1] - t_end) > max(1e-8, 1e-6 * max(1.0, abs(t_end))):
        t_end = float(time_values[-1])
    return dt, t_start, t_end, time_values


def _parse_snapshot_times(raw: str, t_start: float, t_end: float) -> List[float]:
    vals = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        v = float(tok)
        if v < float(t_start) or v > float(t_end):
            raise ValueError(f"Snapshot time must be in [{t_start},{t_end}], got {v}")
        vals.append(v)
    if not vals:
        horizon = float(t_end) - float(t_start)
        vals = [t_start + frac * horizon for frac in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)]
    return vals


def _load_checkpoint(checkpoint_path: str, map_location: str | torch.device) -> Dict[str, object]:
    try:
        return torch.load(checkpoint_path, map_location=map_location)
    except RuntimeError as exc:
        msg = str(exc)
        if "weights_only=True" not in msg or "legacy .tar format" not in msg:
            raise
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)


def _load_train_args(checkpoint_path: str) -> SimpleNamespace:
    args_path = os.path.join(os.path.dirname(checkpoint_path), "args.json")
    if not os.path.exists(args_path):
        return SimpleNamespace()
    with open(args_path, "r", encoding="utf-8") as f:
        return SimpleNamespace(**json.load(f))


def _arg(args: argparse.Namespace, train_args: SimpleNamespace, name: str):
    return getattr(train_args, name, getattr(args, name))


def _infer_state_channels(split: dict) -> int:
    u0 = split["u0"]
    return 1 if u0.dim() == 3 else int(u0.shape[1])


def _spatial_shape(split: dict) -> tuple[int, int]:
    u0 = split["u0"]
    return int(u0.shape[-2]), int(u0.shape[-1])


def _build_model(n_x: int, n_y: int, dt: float, state_channels: int, args: argparse.Namespace, train_args: SimpleNamespace) -> LatentMarkovModel2D:
    boundary_condition = "periodic"
    hidden_channels = int(_arg(args, train_args, "hidden_channels"))
    latent_channels = int(_arg(args, train_args, "latent_channels"))
    enc_blocks = int(_arg(args, train_args, "enc_blocks"))
    dec_blocks = int(_arg(args, train_args, "dec_blocks"))
    prox_blocks = int(_arg(args, train_args, "prox_blocks"))
    prox_type = str(_arg(args, train_args, "prox_simulator_type"))
    use_dt_channel = bool(_arg(args, train_args, "use_dt_channel"))
    disable_forcing_channel = bool(_arg(args, train_args, "disable_forcing_channel"))
    disable_u_grad_feature = bool(_arg(args, train_args, "disable_u_grad_feature"))

    encoder = StateEncoder2D(
        n_x=n_x,
        n_y=n_y,
        latent_channels=latent_channels,
        hidden_channels=hidden_channels,
        n_blocks=enc_blocks,
        use_grad_features=not disable_u_grad_feature,
        boundary_condition=boundary_condition,
        state_channels=state_channels,
    )
    decoder = StateDecoder2D(
        n_x=n_x,
        n_y=n_y,
        latent_channels=latent_channels,
        hidden_channels=hidden_channels,
        n_blocks=dec_blocks,
        boundary_condition=boundary_condition,
        state_channels=state_channels,
    )
    if prox_type == "cnn":
        transition = ProximalStepSimulator2D(
            n_x=n_x,
            n_y=n_y,
            latent_channels=latent_channels,
            hidden_channels=hidden_channels,
            n_blocks=prox_blocks,
            use_forcing_channel=not disable_forcing_channel,
            use_dt_channel=use_dt_channel,
            default_dt=dt,
            boundary_condition=boundary_condition,
        )
    elif prox_type == "fno":
        transition = FNOProximalStepSimulator2D(
            n_x=n_x,
            n_y=n_y,
            latent_channels=latent_channels,
            width=hidden_channels,
            n_layers=prox_blocks,
            modes_x=int(_arg(args, train_args, "fno_modes_x")),
            modes_y=int(_arg(args, train_args, "fno_modes_y")),
            use_forcing_channel=not disable_forcing_channel,
            use_dt_channel=use_dt_channel,
            use_grid_features=not bool(_arg(args, train_args, "disable_fno_grid")),
            default_dt=dt,
            boundary_condition=boundary_condition,
        )
    else:
        raise ValueError(f"Unsupported prox-simulator-type: {prox_type}")
    return LatentMarkovModel2D(encoder=encoder, decoder=decoder, transition=transition)


def _as_channel_state(u: torch.Tensor) -> torch.Tensor:
    if u.dim() == 3:
        return u.unsqueeze(1)
    return u


def _squeeze_if_scalar(u: torch.Tensor, state_channels: int) -> torch.Tensor:
    if state_channels == 1 and u.dim() == 4:
        return u[:, 0]
    return u


@torch.no_grad()
def _rollout(model: LatentMarkovModel2D, u0: torch.Tensor, f: torch.Tensor, n_steps: int, dt: float, delta_clip: Optional[float]) -> torch.Tensor:
    return rollout_latent_markov_2d(model, u0=u0, f=f, n_steps=n_steps, dt=dt, delta_clip=delta_clip)


@torch.no_grad()
def _evaluate_one_step_mse(model: LatentMarkovModel2D, split: Dict[str, torch.Tensor], device: str, dt: float) -> float:
    u_traj = split["u_traj"].to(device)
    f = split["f"].to(device)
    total_sq = 0.0
    n_elem = 0
    for k in range(int(u_traj.shape[1] - 1)):
        u_pred = model.predict_step(u_traj[:, k], f, dt=dt)
        u_ref = u_traj[:, k + 1]
        total_sq += F.mse_loss(u_pred, u_ref, reduction="sum").item()
        n_elem += int(u_ref.numel())
    return total_sq / max(1, n_elem)


def _spectral_h1_norm_2d(u: torch.Tensor) -> torch.Tensor:
    u_ch = _as_channel_state(u)
    n_x = int(u_ch.shape[-2])
    n_y = int(u_ch.shape[-1])
    u_hat = torch.fft.fft2(u_ch, dim=(-2, -1), norm="ortho")
    real_dtype = u_ch.real.dtype
    kx = 2.0 * torch.pi * torch.fft.fftfreq(n_x, d=1.0 / float(n_x), device=u_ch.device).to(dtype=real_dtype)
    ky = 2.0 * torch.pi * torch.fft.fftfreq(n_y, d=1.0 / float(n_y), device=u_ch.device).to(dtype=real_dtype)
    kx_grid, ky_grid = torch.meshgrid(kx, ky, indexing="ij")
    weight = 1.0 + kx_grid.square() + ky_grid.square()
    power = u_hat.real.square() + u_hat.imag.square()
    return torch.sqrt(torch.sum(power * weight.view(1, 1, n_x, n_y), dim=(-3, -2, -1)))


@torch.no_grad()
def _evaluate_rollout_curves(model: LatentMarkovModel2D, split: Dict[str, torch.Tensor], device: str, dt: float, area: float, delta_clip: Optional[float]) -> Dict[str, np.ndarray | float]:
    u0 = split["u0"].to(device)
    f = split["f"].to(device)
    u_ref = split["u_traj"].to(device)
    n_steps = int(u_ref.shape[1] - 1)
    u_pred = _rollout(model, u0=u0, f=f, n_steps=n_steps, dt=dt, delta_clip=delta_clip)
    diff = u_pred - u_ref

    diff_ch = _as_channel_state(diff)
    ref_ch = _as_channel_state(u_ref)
    num = torch.sqrt(area * torch.sum(diff_ch * diff_ch, dim=(-3, -2, -1)))
    den = torch.sqrt(area * torch.sum(ref_ch * ref_ch, dim=(-3, -2, -1)))
    rel = num[:, 1:] / (den[:, 1:] + 1e-8)

    h1_num = _spectral_h1_norm_2d(diff).detach().cpu()[:, 1:]
    h1_den = _spectral_h1_norm_2d(u_ref).detach().cpu()[:, 1:]
    rel_h1 = h1_num / (h1_den + 1e-12)

    rel_cpu = rel.detach().cpu()
    overall_l2 = (torch.sqrt(torch.sum(num[:, 1:].detach().cpu().square(), dim=1)) / (torch.sqrt(torch.sum(den[:, 1:].detach().cpu().square(), dim=1)) + 1e-8)).numpy().astype(np.float64)
    overall_h1 = (torch.sqrt(torch.sum(h1_num.square(), dim=1)) / (torch.sqrt(torch.sum(h1_den.square(), dim=1)) + 1e-12)).numpy().astype(np.float64)

    return {
        "rel_curve_mean": torch.nanmean(rel_cpu, dim=0).numpy().astype(np.float64),
        "rel_curve_median": np.nanmedian(rel_cpu.numpy(), axis=0).astype(np.float64),
        "rel_h1_curve_mean": torch.nanmean(rel_h1, dim=0).numpy().astype(np.float64),
        "rel_h1_curve_median": np.nanmedian(rel_h1.numpy(), axis=0).astype(np.float64),
        "rollout_rel_mean": float(np.nanmean(overall_l2)),
        "rollout_rel_median": float(np.nanmedian(overall_l2)),
        "rollout_rel_std": float(np.nanstd(overall_l2)),
        "rollout_rel_max": float(np.nanmax(torch.nanmean(rel_cpu, dim=0).numpy())),
        "rollout_rel_h1": float(np.nanmean(overall_h1)),
        "rollout_rel_h1_median": float(np.nanmedian(overall_h1)),
        "rollout_rel_h1_std": float(np.nanstd(overall_h1)),
        "rollout_rel_h1_max": float(np.nanmax(torch.nanmean(rel_h1, dim=0).numpy())),
        "rel_samples": rel_cpu.numpy().astype(np.float64),
        "rel_h1_samples": rel_h1.numpy().astype(np.float64),
        "overall_rel_l2_samples": overall_l2,
        "overall_rel_h1_samples": overall_h1,
        "overall_rel_l2": float(np.nanmean(overall_l2)),
        "overall_rel_h1": float(np.nanmean(overall_h1)),
    }


def _save_rollout_curve_csv(curves: Dict[str, np.ndarray], time_values: np.ndarray, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "time", "rel_l2_mean", "rel_l2_median", "rel_h1_mean", "rel_h1_median"])
        for k in range(len(curves["rel_curve_mean"])):
            writer.writerow([
                k + 1,
                f"{float(time_values[k + 1]):.8f}",
                f"{float(curves['rel_curve_mean'][k]):.12e}",
                f"{float(curves['rel_curve_median'][k]):.12e}",
                f"{float(curves['rel_h1_curve_mean'][k]):.12e}",
                f"{float(curves['rel_h1_curve_median'][k]):.12e}",
            ])
    print(f"Saved rollout curve csv: {out_path}")


def _save_per_sample_errors_json(curves: Dict[str, np.ndarray], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    items = []
    rel_l2 = curves["rel_samples"]
    rel_h1 = curves["rel_h1_samples"]
    for sample_idx in range(int(rel_l2.shape[0])):
        items.append({
            "sample_index": sample_idx,
            "rel_l2": {"all_channels": rel_l2[sample_idx].tolist()},
            "rel_h1": {"all_channels": rel_h1[sample_idx].tolist()},
            "overall_rel_l2": {"all_channels": float(curves["overall_rel_l2_samples"][sample_idx])},
            "overall_rel_h1": {"all_channels": float(curves["overall_rel_h1_samples"][sample_idx])},
        })
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)


def _plot_rollout_curves(curves: Dict[str, np.ndarray], time_values: np.ndarray, out_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping curve plotting because matplotlib is unavailable: {exc}")
        return
    t = time_values[1 : 1 + curves["rel_curve_mean"].shape[0]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), squeeze=False)
    axes[0, 0].plot(t, curves["rel_curve_mean"], linewidth=2)
    axes[0, 0].set_title(f"Relative L2\nagg={curves['rollout_rel_mean']:.4e}")
    axes[0, 0].set_xlabel("time")
    axes[0, 0].set_ylabel("relative L2")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 1].plot(t, curves["rel_h1_curve_mean"], linewidth=2, color="tab:red")
    axes[0, 1].set_title(f"Relative H1\nagg={curves['rollout_rel_h1']:.4e}")
    axes[0, 1].set_xlabel("time")
    axes[0, 1].set_ylabel("relative H1")
    axes[0, 1].grid(alpha=0.3)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved rollout curve plot: {out_path}")


def _select_channel(field: torch.Tensor, channel: int) -> torch.Tensor:
    if field.dim() == 2:
        return field
    if field.dim() == 3:
        c = max(0, min(int(channel), int(field.shape[0] - 1)))
        return field[c]
    raise ValueError(f"Expected field with shape (X,Y) or (C,X,Y), got {tuple(field.shape)}")


@torch.no_grad()
def _plot_test_samples(model: LatentMarkovModel2D, split: Dict[str, torch.Tensor], device: str, dt: float, time_values: np.ndarray, snapshot_times: List[float], n_plot_samples: int, out_dir: str, delta_clip: Optional[float], plot_channel: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping sample plotting because matplotlib is unavailable: {exc}")
        return
    os.makedirs(out_dir, exist_ok=True)
    u_traj = split["u_traj"]
    u0 = split["u0"]
    f = split["f"]
    total = int(u_traj.shape[0])
    n_plot = min(max(1, int(n_plot_samples)), total)
    sample_ids = torch.linspace(0, total - 1, n_plot).long().tolist()
    n_steps = int(u_traj.shape[1] - 1)
    n_cols = len(snapshot_times)

    for sample_id in sample_ids:
        u_pred_i = _rollout(model, u0[sample_id : sample_id + 1].to(device), f[sample_id : sample_id + 1].to(device), n_steps, dt, delta_clip)[0].cpu()
        u_ref_i = u_traj[sample_id]
        state_scale = max(float(torch.max(torch.abs(u_ref_i)).item()), float(torch.max(torch.abs(u_pred_i)).item()), 1e-8)
        fig, axes = plt.subplots(3, n_cols, figsize=(3.1 * n_cols, 8.0), squeeze=False, constrained_layout=True)
        im_last = None
        err_last = None
        for j, t_snap in enumerate(snapshot_times):
            k = int(np.argmin(np.abs(time_values - float(t_snap))))
            k = max(0, min(n_steps, k))
            ref_k = _select_channel(u_ref_i[k], plot_channel)
            pred_k = _select_channel(u_pred_i[k], plot_channel)
            err_k = torch.abs(pred_k - ref_k)
            im_last = axes[0, j].imshow(ref_k.numpy(), origin="lower", cmap="coolwarm", vmin=-state_scale, vmax=state_scale)
            axes[0, j].set_title(f"ref t={float(time_values[k]):g}")
            axes[1, j].imshow(pred_k.numpy(), origin="lower", cmap="coolwarm", vmin=-state_scale, vmax=state_scale)
            axes[1, j].set_title(f"pred t={float(time_values[k]):g}")
            err_last = axes[2, j].imshow(err_k.numpy(), origin="lower", cmap="magma")
            axes[2, j].set_title("abs error")
            for row in range(3):
                axes[row, j].set_xticks([])
                axes[row, j].set_yticks([])
        if im_last is not None:
            fig.colorbar(im_last, ax=axes[0:2, :], fraction=0.015, pad=0.01)
        if err_last is not None:
            fig.colorbar(err_last, ax=axes[2, :], fraction=0.015, pad=0.01)
        fig.suptitle(f"Sample {sample_id}, channel {plot_channel}", fontsize=13)
        out_path = os.path.join(out_dir, f"sample_{sample_id:04d}_channel_{plot_channel}_comparison.png")
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
    print(f"Saved sample plots: {out_dir}")


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed, seed_cuda=(device == "cuda"))
    print(f"Device: {device}")

    splits = load_dataset_splits(args.dataset_path, map_location="cpu")
    split = dict(splits[args.split])
    meta = splits.get("meta", {})
    n_x, n_y = _spatial_shape(split)
    state_channels = int(args.state_channels or _infer_state_channels(split))
    n_steps = int(split["u_traj"].shape[1] - 1)
    h_x = 1.0 / float(n_x)
    h_y = 1.0 / float(n_y)
    area = h_x * h_y
    dt, t_start, t_final, time_values = _time_metadata(meta, n_steps=n_steps)
    if args.max_steps is not None:
        n_steps = min(int(args.max_steps), n_steps)
        split["u_traj"] = split["u_traj"][:, : n_steps + 1]
        split["u0"] = split["u_traj"][:, 0].clone()
        time_values = time_values[: n_steps + 1]
        t_final = float(time_values[-1])
    snapshot_times = _parse_snapshot_times(args.snapshot_times, t_start=t_start, t_end=t_final)

    train_args = _load_train_args(args.checkpoint_path)
    state_channels = int(getattr(train_args, "state_channels_used", None) or getattr(train_args, "state_channels", None) or state_channels)
    print(f"Loaded split={args.split} from {args.dataset_path}")
    print(f"Grid: n_x={n_x}, n_y={n_y}, state_channels={state_channels}, n_steps={n_steps}, stored_time=[{t_start:.6f},{t_final:.6f}], dt={dt:.6f}")
    print(f"Delta clip (L-inf): {args.delta_clip:.6f}")

    model = _build_model(n_x=n_x, n_y=n_y, dt=dt, state_channels=state_channels, args=args, train_args=train_args).to(device)
    ckpt = _load_checkpoint(args.checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint_path}")

    step_mse = _evaluate_one_step_mse(model, split, device=device, dt=dt)
    curves = _evaluate_rollout_curves(model, split, device=device, dt=dt, area=area, delta_clip=args.delta_clip)
    print(f"Split one-step MSE: {step_mse:.8e}")
    print(f"Split rollout mean relative L2: {curves['rollout_rel_mean']:.8e}")
    print(f"Split rollout median relative L2: {curves['rollout_rel_median']:.8e}")
    print(f"Split rollout mean relative H1: {curves['rollout_rel_h1']:.8e}")
    print(f"Split rollout median relative H1: {curves['rollout_rel_h1_median']:.8e}")

    for k in range(len(curves["rel_curve_mean"])):
        print(
            f"  {k + 1:03d}  {float(time_values[k + 1]):8.4f}  "
            f"{curves['rel_curve_mean'][k]:.8e}  {curves['rel_curve_median'][k]:.8e}  "
            f"{curves['rel_h1_curve_mean'][k]:.8e}  {curves['rel_h1_curve_median'][k]:.8e}"
        )

    curve_csv = os.path.join(args.output_dir, f"{args.split}_rollout_error_curve.csv")
    curve_png = os.path.join(args.output_dir, f"{args.split}_rollout_error_curve.png")
    _save_rollout_curve_csv(curves, time_values=time_values, out_path=curve_csv)
    _plot_rollout_curves(curves, time_values=time_values, out_path=curve_png)
    per_sample_path = os.path.join(args.output_dir, f"{args.split}_per_sample_errors.json")
    _save_per_sample_errors_json(curves, out_path=per_sample_path)
    print(f"Saved per-sample errors: {per_sample_path}")

    _plot_test_samples(
        model=model,
        split=split,
        device=device,
        dt=dt,
        time_values=time_values,
        snapshot_times=snapshot_times,
        n_plot_samples=args.n_plot_samples,
        out_dir=os.path.join(args.output_dir, f"{args.split}_sample_comparisons"),
        delta_clip=args.delta_clip,
        plot_channel=args.plot_channel,
    )

    summary = {
        "dataset_path": args.dataset_path,
        "checkpoint_path": args.checkpoint_path,
        "split": args.split,
        "n_x": n_x,
        "n_y": n_y,
        "state_channels": state_channels,
        "n_steps": n_steps,
        "dt": dt,
        "t_start": t_start,
        "t_final": t_final,
        "time_values": time_values.tolist(),
        "error_time_values": time_values[1 : 1 + len(curves["rel_curve_mean"])].tolist(),
        "delta_clip": args.delta_clip,
        "step_mse": step_mse,
        "rollout_rel_l2": curves["rollout_rel_mean"],
        "rollout_rel_l2_median": curves["rollout_rel_median"],
        "rollout_rel_l2_std": curves["rollout_rel_std"],
        "rollout_rel_l2_max": curves["rollout_rel_max"],
        "overall_rel_l2": curves["overall_rel_l2"],
        "rollout_rel_h1": curves["rollout_rel_h1"],
        "rollout_rel_h1_median": curves["rollout_rel_h1_median"],
        "rollout_rel_h1_std": curves["rollout_rel_h1_std"],
        "rollout_rel_h1_max": curves["rollout_rel_h1_max"],
        "overall_rel_h1": curves["overall_rel_h1"],
        "rel_curve_mean": curves["rel_curve_mean"].tolist(),
        "rel_curve_median": curves["rel_curve_median"].tolist(),
        "rel_h1_curve_mean": curves["rel_h1_curve_mean"].tolist(),
        "rel_h1_curve_median": curves["rel_h1_curve_median"].tolist(),
        "snapshot_times": snapshot_times,
        "plot_channel": int(args.plot_channel),
        "max_steps": args.max_steps,
        "seed": int(args.seed),
        "meta": meta,
    }
    with open(os.path.join(args.output_dir, f"{args.split}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved evaluation summary to: {args.output_dir}")


if __name__ == "__main__":
    main(parse_args())
