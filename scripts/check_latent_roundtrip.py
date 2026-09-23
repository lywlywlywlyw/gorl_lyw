"""Measure actor/decoder round trips on dataset states (no environment rollout).

Run with JAX_PLATFORMS=cpu to avoid occupying a training GPU.
All errors are per-coordinate; p95/p99 refer to per-sample RMSE.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from flow_policy import networks
from flow_policy.decoder_1step_fm_residualMLP import Decoder1StepFMState


def errors(delta):
    delta = np.asarray(delta)
    sample_rmse = np.sqrt(np.mean(delta**2, axis=-1))
    return {
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "mae": float(np.mean(np.abs(delta))),
        "sample_rmse_p95": float(np.quantile(sample_rmse, .95)),
        "sample_rmse_p99": float(np.quantile(sample_rmse, .99)),
        "max_abs": float(np.max(np.abs(delta))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.checkpoint.open("rb") as f:
        ck = pickle.load(f)
    with args.dataset.open("rb") as f:
        data = pickle.load(f)
    ids = np.random.default_rng(args.seed).choice(
        len(data["observations"]), args.samples, replace=False
    )
    obs = jnp.asarray(data["observations"][ids])
    actions = jnp.asarray(data["actions"][ids])
    state = Decoder1StepFMState.init(
        jax.random.PRNGKey(0), ck["obs_dim"], ck["action_dim"], ck["config"]
    )
    state = jdc.replace(state, params=ck["params"], obs_stats=ck["obs_stats"])
    normalized = state._normalize_obs(obs)
    forward = jax.jit(lambda z: state._decode_normalized(normalized, z))
    inverse = jax.jit(lambda a: state._inverse_fm_batch_normalized(normalized, a))
    stats = ck["iql_z_obs_stats"]
    encoder_obs = (obs - stats.mean) / (stats.std + 1e-8)
    dist = networks.gaussian_policy_fwd(ck["iql_z_actor_params"], encoder_obs)
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset": str(args.dataset.resolve()),
        "samples": args.samples,
        "seed": args.seed,
        "scope": "random dataset states; not held-out or on-policy rollout",
        "actor_std_mean": float(jnp.mean(dist.scale)),
        "modes": {},
    }
    modes = {
        "actor_mean": dist.loc,
        "actor_sample": dist.sample(jax.random.PRNGKey(args.seed)),
        "normal_prior": jax.random.normal(jax.random.PRNGKey(args.seed + 1), actions.shape),
        "data_inverse": inverse(actions),
    }
    for name, z in modes.items():
        raw = forward(z)
        executed = jnp.clip(raw, -1, 1)
        recovered = inverse(executed)
        repeated = jnp.clip(forward(recovered), -1, 1)
        sat = np.asarray(jnp.any(jnp.abs(raw) > 1, axis=-1))
        q = lambda latent: jnp.minimum(
            networks.q_mlp_fwd(ck["q1_params"], encoder_obs, latent),
            networks.q_mlp_fwd(ck["q2_params"], encoder_obs, latent),
        )
        q_original, q_recovered = q(z), q(recovered)
        result = {
            "latent_norm_mean": float(jnp.mean(jnp.linalg.norm(z, axis=-1))),
            "latent_max_abs": float(jnp.max(jnp.abs(z))),
            "saturated_components": float(jnp.mean(jnp.abs(raw) > 1)),
            "saturated_samples": float(sat.mean()),
            "latent_raw_roundtrip": errors(inverse(raw) - z),
            "latent_executed_roundtrip": errors(recovered - z),
            "action_executed_roundtrip": errors(repeated - executed),
            "action_vs_dataset": errors(executed - actions),
            "q_original_mean": float(jnp.mean(q_original)),
            "q_recovered_mean": float(jnp.mean(q_recovered)),
            "q_roundtrip_abs_gap": float(jnp.mean(jnp.abs(q_original - q_recovered))),
        }
        for label, mask in (("saturated", sat), ("unsaturated", ~sat)):
            if mask.any():
                result["latent_" + label] = errors(np.asarray(recovered - z)[mask])
        report["modes"][name] = result
        print(name, json.dumps(result), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("Report:", args.output, flush=True)


if __name__ == "__main__":
    main()
