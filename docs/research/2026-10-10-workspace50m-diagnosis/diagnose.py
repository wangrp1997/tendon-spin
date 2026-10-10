# Sources: TendonSpin archived workspace10M/20M/30M/40M/50M evaluations;
# HaozhiQi/hora v0.0.1 ActorCritic, RunningMeanStd and compute_hand_reward (MIT),
# https://github.com/HaozhiQi/hora (Qi et al., CoRL2022, arXiv:2210.04887).
# Reuse original pure functions by AST extraction; reuse original actor and
# normalizer through tendonspin.baselines.reference_models. No simulator imports,
# physics, training updates, replacement policy execution or acceptance-rule edits.
"""One bounded CPU analysis of existing teacher trajectories and summaries."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tendon-spin-diagnosis-mpl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from tendonspin.baselines.hora_training import load_reward
from tendonspin.baselines.reference_models import build_model, reference_module
from tendonspin.interfaces import ACTION_NAMES, finger_action_contract

torch.set_num_threads(4)
DT = .0005
CONTROL_STEPS = 100
RATE_STEP = .35 * .05
SOURCE_KEYS = ["elapsed_s", "valid", "net_angle_deg", "object_state", "joint_pos",
               "joint_pos_before", "commands", "action", "motor_effort_requested",
               "finger_contact_count", "support_groups", "palm_contact", "max_normal",
               "drift_mm", "tilt_deg", "normal_force_matrix_w"]
RUNS = {10: ROOT / "outputs/boya_hora1024_workspace10m_v3"}
RUNS.update({n: ROOT / f"outputs/boya_hora1024_workspace50m_v1/actions_{n*1000000:09d}"
             for n in (20, 30, 40, 50)})


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    path.write_text(json.dumps(plain(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def pure_function(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def describe(x):
    x = np.asarray(x)
    if not x.size:
        return None
    return dict(mean=float(x.mean()), median=float(np.median(x)),
                p90=float(np.quantile(x, .9)), min=float(x.min()), max=float(x.max()))


def longest_true(mask):
    mask = np.asarray(mask, bool)
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    if not len(edges):
        return dict(seconds=0., start_s=None, end_s=None)
    starts, ends = edges[::2], edges[1::2]
    i = np.argmax(ends-starts)
    return dict(seconds=float((ends[i]-starts[i])*DT),
                start_s=float(starts[i]*DT), end_s=float(ends[i]*DT))


def endpoint(d, seconds):
    i = np.searchsorted(d["elapsed_s"], seconds+1e-10, side="right") - 1
    return 0. if i < 0 else float(d["net_angle_deg"][i])


def window(d, start, end):
    mask = (d["elapsed_s"] > start+1e-10) & (d["elapsed_s"] <= end+1e-10) & d["valid"]
    if not mask.any():
        return None
    last = float(d["elapsed_s"][np.flatnonzero(mask)[-1]])
    return dict(start_s=float(start), end_s=last,
                net_deg=endpoint(d, last)-endpoint(d, start),
                sensor_zero_hand_contact_fraction=float((d["support_groups"][mask] == 0).mean()),
                finger_contacts_mean=float(d["finger_contact_count"][mask].mean()),
                max_normal_N=float(d["max_normal"][mask].max()),
                max_tilt_deg=float(d["tilt_deg"][mask].max()),
                object_angular_speed_rad_s=describe(np.linalg.norm(d["object_state"][mask, 10:13], axis=1)))


def clipped_normal_mean(mu, sigma, lower, upper):
    """E[clip(X, lower, upper)] for Gaussian X; no action sampling or physics."""
    lo = (lower-mu)/sigma
    hi = (upper-mu)/sigma
    phi = lambda z: torch.exp(-z*z/2) / np.sqrt(2*np.pi)
    cdf = lambda z: .5*(1+torch.erf(z/np.sqrt(2)))
    return lower*cdf(lo) + mu*(cdf(hi)-cdf(lo)) + sigma*(phi(lo)-phi(hi)) + upper*(1-cdf(hi))


def load_training(run, manifest):
    path = run / "training/result.json"
    record = json.loads(path.read_text())
    manifest.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path)))
    result = dict(actions=record["actions_executed"], updates=record["completed_updates"],
                  stop_reason=record["stop_reason"], resume_from=record.get("resume_from"),
                  environment_resume=record.get("environment_resume"))
    for size in (10, 100):
        updates = record["updates"][-size:]
        summaries = [u["training_episode_summary"] for u in updates
                     if u.get("training_episode_summary", {}).get("count", 0)]
        count = sum(s["count"] for s in summaries)
        entry = dict(updates=len(updates), episode_count=count,
                     mean_recent_episode_reward=float(np.mean([u["mean_recent_episode_reward"] for u in updates])),
                     entropy_last=float(updates[-1]["entropy"]))
        for key in ("net_deg", "peak_deg", "backward_deg", "valid_prefix_s"):
            entry[key] = sum(s[key]["mean"]*s["count"] for s in summaries)/count
        entry["stop_counts"] = {reason: sum(s["stop_counts"].get(reason, 0) for s in summaries)
                                for reason in set().union(*(s["stop_counts"] for s in summaries))}
        result[f"last_{size}_updates"] = entry
    # Actual training scalar logs. Full per-step TRAINING components do not exist.
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    tbpath = run / "training/training/stage1_tb"
    events = EventAccumulator(str(tbpath), size_guidance={"scalars": 0}).Reload()
    result["tensorboard"] = {}
    tags = events.Tags()["scalars"]
    for tag in ("rotation_reward", "object_linvel_penalty", "episode_rewards/step", "episode_lengths/step"):
        if tag in tags:
            values = events.Scalars(tag)[-100:]
            result["tensorboard"][tag] = dict(count=len(values), start_step=values[0].step,
                end_step=values[-1].step, mean=float(np.mean([v.value for v in values])),
                last=float(values[-1].value))
    for path in sorted(tbpath.glob("events*")):
        manifest.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path)))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT/"outputs/boya_workspace50m_diagnosis_v1")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    result = dict(schema=1, date="2026-10-10", new_physics_steps=0, new_training_actions=0,
                  offline_inference_only=True, device="cpu", threads=4,
                  configuration="Original workspace teacher, no engine/task/reward/action/observation edits",
                  context="40x32mm/50g cylinder, original grasp44, 16 finger actions, held wrists, four passive mimics; Isaac Sim6.1.0.0/Lab3.0.0rc1, v3 external clipped PD, 28-cache nominal privileged teacher",
                  diagnostic_definitions=dict(common_prefix_s=5, phase_windows_s=[[0, 5], [5, 20], [20, 30]],
                      terminal_windows_s=[1., .2], sensor_contact_threshold_N=1e-6,
                      numerical_target_bound_tolerance_rad=1e-6, training_summary_updates=[10, 100],
                      uncertainty="One fixed original-state rollout per checkpoint; no confidence interval or causal experiment",
                      reward="Offline original Hora function on frozen evaluation control endpoints; canonical original-grasp pose reference; NOT recorded training rewards",
                      plots="Every20th physics sample (100Hz), always retain terminal sample; control/reward curves at20Hz. Pose-rate diagnostic is the signed quaternion increment summed over each complete50ms control interval divided by .05s; no other smoothing."),
                  episodes={}, training={}, inference={}, source_manifest=[])
    manifest = result["source_manifest"]
    namespace = {"torch": torch}
    axis_z = pure_function(ROOT/"tendonspin/physics/isaac_parallel.py", "axis_z", namespace)
    spin = pure_function(ROOT/"tendonspin/rl/isaac_hora.py", "spin_increment", namespace)
    reward_fn = load_reward()
    contract = finger_action_contract(json.loads((ROOT/"docs/data/boya_native_contract.json").read_text()))
    motor_names = [a["name"] for a in contract["actuators"]]
    motor_ids = [motor_names.index(n) for n in ACTION_NAMES]
    lower = np.asarray([contract["actuators"][i]["control_range"][0] for i in motor_ids], np.float32)
    upper = np.asarray([contract["actuators"][i]["control_range"][1] for i in motor_ids], np.float32)
    source_paths = ["tendonspin/rl/isaac_hora.py", "tendonspin/physics/isaac_parallel.py",
                    "tendonspin/physics/isaac_boya.py", "tendonspin/physics/isaac_boya_sharpa.py",
                    "tendonspin/baselines/hora_training.py", "tendonspin/baselines/reference_models.py",
                    "third_party/hora/hora/tasks/allegro_hand_hora.py",
                    "third_party/hora/hora/algo/models/models.py",
                    "third_party/hora/hora/algo/models/running_mean_std.py", "scripts/evaluate_boya_hora.py",
                    "docs/data/boya_native_contract.json", "tendonspin/interfaces.py", "third_party/hora/hora/algo/ppo/ppo.py"]
    for path in source_paths:
        manifest.append(dict(path=path, sha256=sha(ROOT/path)))
    models, norms, controls, plot_data = {}, {}, {}, {}
    initial_anchor = None
    for milestone, run in RUNS.items():
        evaluation = run / "evaluation"
        record = json.loads((evaluation/"result.json").read_text())
        assert record["episode_resets"] == record["controller_switches"] == record["training_actions"] == 0
        checkpoint_path = run/"training/teacher_final.pth"
        assert sha(checkpoint_path) == record["checkpoint_sha256"]
        expected = {x["path"]: x["sha256"] for x in record["sources"]}
        for rel in source_paths:
            if rel in expected:
                assert sha(ROOT/rel) == expected[rel], (milestone, rel)
        for path in [evaluation/"result.json", evaluation/"initial_state.npz", evaluation/"control_inputs.npz", checkpoint_path]:
            manifest.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path)))
        with np.load(evaluation/"initial_state.npz") as f:
            initial = {k: f[k].copy() for k in f.files}
        if initial_anchor is None:
            initial_anchor = initial
        state_error = {k: float(np.max(np.abs(initial[k].astype(float)-initial_anchor[k].astype(float))))
                       for k in ("joint_pos", "joint_vel", "object_state", "commands")}
        assert max(state_error.values()) == 0, state_error
        blocks = {key: [] for key in SOURCE_KEYS}
        for path in sorted(evaluation.glob("physics_*.npz")):
            manifest.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path)))
            with np.load(path) as f:
                for key in SOURCE_KEYS:
                    blocks[key].append(f[key])
        d = {key: np.concatenate(value) for key, value in blocks.items()}
        del blocks
        assert len(d["valid"]) == record["physics_steps"]
        assert int(d["valid"].sum()) == record["valid_steps"]
        assert np.allclose(d["elapsed_s"], np.arange(1, len(d["valid"])+1)*DT, rtol=0, atol=1e-10)
        assert abs(d["net_angle_deg"][-1]-record["net_deg"]) < 1e-9
        with np.load(evaluation/"control_inputs.npz") as f:
            ctl = {k: f[k].copy() for k in f.files}
        controls[milestone] = ctl
        assert np.array_equal(ctl["step"], np.arange(len(ctl["step"]))*CONTROL_STEPS)
        active = [record["joint_names"].index(n) for n in ACTION_NAMES]
        n = len(d["valid"])
        quat = torch.from_numpy(d["object_state"][:, 3:7])
        previous = torch.cat((torch.from_numpy(initial["object_state"][:, 3:7]), quat[:-1]))
        increments = torch.rad2deg(spin(previous, quat)).numpy()
        archived_inc = np.diff(np.r_[0., d["net_angle_deg"]])
        valid = d["valid"]
        integration_error = np.abs(increments[valid].astype(float)-archived_inc[valid])
        axes = axis_z(quat).numpy()
        rotation_axis = -axis_z(torch.from_numpy(initial["object_state"][:, 3:7])).numpy()[0]
        omega = d["object_state"][:, 10:13]
        fixed_rate = omega @ rotation_axis
        moving_rate = -(omega*axes).sum(-1)
        geometric_rate = increments * (np.pi/180) / DT
        end_indices = np.arange(CONTROL_STEPS-1, n, CONTROL_STEPS)
        end_times = d["elapsed_s"][end_indices]
        geometric_control_rate = increments[:len(end_indices)*CONTROL_STEPS].reshape(-1, CONTROL_STEPS).sum(1) * (np.pi/180) / .05
        q = torch.from_numpy(d["joint_pos"][end_indices][:, active])
        q_before = torch.from_numpy(d["joint_pos_before"][end_indices][:, active])
        tau = torch.from_numpy(d["motor_effort_requested"][end_indices][:, active])
        velocity = (q-q_before)/DT
        pose = ((q-torch.from_numpy(initial["joint_pos"][:, active]))**2).sum(-1)
        torque = (tau*tau).sum(-1)
        work = ((tau*velocity).sum(-1))**2
        obj = torch.from_numpy(d["object_state"][end_indices])
        reward, rot, lin = reward_fn(obj[:, 7:10], -.3, obj[:, 10:13],
            torch.from_numpy(rotation_axis[None]).expand(len(q), -1), 1., .5, -.5,
            pose, -.3, torque, -.1, work, -2.)
        components = dict(rotation=rot.numpy(), linear_cost=-.3*lin.numpy(),
                          pose_cost=-.3*pose.numpy(), torque_cost=-.1*torque.numpy(), work_cost=-2*work.numpy(),
                          total=reward.numpy())
        assert np.max(np.abs(sum(components[k] for k in components if k != "total")-components["total"])) < 1e-4
        old_commands = np.vstack((initial["commands"][0, motor_ids],
                                  d["commands"][end_indices[:-1]][:, motor_ids]))
        current_commands = d["commands"][end_indices][:, motor_ids]
        request = ctl["action"][:len(end_indices)]*RATE_STEP
        expected_commands = np.clip(old_commands+request, lower, upper)
        command_error = float(np.max(np.abs(expected_commands-current_commands)))
        assert command_error < 1e-5, command_error
        delta = current_commands-old_commands
        outward = ((old_commands >= upper-1e-6) & (request > 0)) | ((old_commands <= lower+1e-6) & (request < 0))
        limits = (current_commands >= upper-1e-6) | (current_commands <= lower+1e-6)
        command_windows = {}
        reward_windows = {}
        for label, start, end in [("first5s", 0, 5), ("5to20s", 5, 20), ("20to30s", 20, 30),
                                  ("after30s", 30, record["valid_s"]),
                                  ("last1s", max(0, record["valid_s"]-1), record["valid_s"])]:
            m = (end_times > start) & (end_times <= end) & d["valid"][end_indices]
            if not m.any():
                continue
            duration = float(m.sum()*.05)
            command_windows[label] = dict(control_count=int(m.sum()), duration_s=duration,
                outward_blocked_motor_fraction=float(outward[m].mean()), at_target_bound_motor_fraction=float(limits[m].mean()),
                command_travel_rad_per_motor=np.abs(delta[m]).sum(0).tolist(),
                actual_travel_rad_per_motor=np.abs(np.diff(np.vstack((initial["joint_pos"][0, active],
                    d["joint_pos"][end_indices][:, active])), axis=0)[m]).sum(0).tolist(),
                joints_at_bound_over90pct=[ACTION_NAMES[i] for i in np.flatnonzero(limits[m].mean(0) > .9)],
                joints_outward_blocked_over90pct=[ACTION_NAMES[i] for i in np.flatnonzero(outward[m].mean(0) > .9)])
            reward_windows[label] = dict(control_count=int(m.sum()), duration_s=duration,
                components_mean={k: float(v[m].mean()) for k, v in components.items()},
                components_sum={k: float(v[m].sum()) for k, v in components.items()},
                fixed_axis_omega_rad_s=describe(fixed_rate[end_indices][m]),
                moving_axis_omega_rad_s=describe(moving_rate[end_indices][m]),
                rotation_upper_clip_fraction=float((fixed_rate[end_indices][m] >= .5).mean()),
                rotation_lower_clip_fraction=float((fixed_rate[end_indices][m] <= -.5).mean()))
        rate_agreement = {}
        for label, start, end in [("first5s", 0, 5), ("5to20s", 5, 20), ("20to30s", 20, 30)]:
            m = (d["elapsed_s"] > start) & (d["elapsed_s"] <= end) & valid
            c = (end_times > start) & (end_times <= end) & valid[end_indices]
            if not m.any() or not c.any():
                continue
            rate_agreement[label] = dict(
                physics_samples=int(m.sum()), control_samples=int(c.sum()),
                fixed_axis_omega_physics_mean_rad_s=float(fixed_rate[m].mean()),
                moving_axis_omega_physics_mean_rad_s=float(moving_rate[m].mean()),
                pose_spin_physics_mean_rad_s=float(geometric_rate[m].mean()),
                pose_spin_control_interval_mean_rad_s=float(geometric_control_rate[c].mean()),
                fixed_axis_omega_endpoint_mean_rad_s=float(fixed_rate[end_indices][c].mean()),
                moving_axis_omega_endpoint_mean_rad_s=float(moving_rate[end_indices][c].mean()),
                pose_vs_moving_omega_physics_rmse_rad_s=float(np.sqrt(np.mean((geometric_rate[m]-moving_rate[m])**2))),
                clipped_fixed_omega_physics_mean=float(np.clip(fixed_rate[m], -.5, .5).mean()),
                interpretation="Reported engine angular velocity versus actual quaternion motion. Full-step comparison separates a persistent velocity/pose mismatch from endpoint-only sampling. No engine cause established.")
        windows = {f"{start:g}to{end:g}s": window(d, start, end) for start, end in [(0, 5), (5, 20), (20, 30)]}
        for seconds in (1., .2):
            windows[f"last{seconds:g}s"] = window(d, max(0, record["valid_s"]-seconds), record["valid_s"])
        contact_ids = np.flatnonzero((d["support_groups"] > 0) & valid)
        last_contact = float(d["elapsed_s"][contact_ids[-1]]) if len(contact_ids) else 0.
        sample_times = [1., 5., 10., 20., 30., record["valid_s"]-.2, record["valid_s"]]
        points = []
        for t in sample_times:
            if t < 0 or t > record["valid_s"]:
                continue
            i = np.searchsorted(d["elapsed_s"], t+1e-10, side="right")-1
            if i < 0:
                continue
            points.append(dict(t=float(d["elapsed_s"][i]), net_deg=float(d["net_angle_deg"][i]),
                height_m=float(d["object_state"][i, 2]), finger_contacts=int(d["finger_contact_count"][i]),
                support_groups=int(d["support_groups"][i]), tilt_deg=float(d["tilt_deg"][i]),
                normal_N=float(d["max_normal"][i]), fixed_axis_omega_rad_s=float(fixed_rate[i]),
                moving_axis_omega_rad_s=float(moving_rate[i])))
        ep = dict(record={key: record[key] for key in ("training_actions_in_checkpoint", "actual_s", "valid_s", "net_deg", "stop_reason", "checkpoint_sha256")},
                  published_metrics=record["metrics"], initial_state_max_errors=state_error,
                  physics_samples=n, integration_max_increment_error_deg=float(integration_error.max()),
                  integration_endpoint_error_deg=float(np.sum(increments[valid], dtype=np.float64)-record["net_deg"]),
                  command_reconstruction_max_error_rad=command_error,
                  common5s_net_deg=endpoint(d, 5.), windows=windows, sample_points=points,
                  commands=command_windows, reconstructed_reward=reward_windows,
                  angular_velocity_pose_agreement=rate_agreement,
                  last_visible_hand_contact_s=last_contact,
                  valid_net_after_last_visible_contact_deg=record["net_deg"]-endpoint(d, last_contact),
                  longest_sensor_zero_hand_contact=longest_true((d["support_groups"] == 0) & valid),
                  max_object_angular_speed_rad_s=float(np.linalg.norm(omega[valid], axis=1).max()))
        result["episodes"][milestone] = ep
        # Only compact plotting views are held across episodes; full originals stay unchanged.
        view = np.unique(np.r_[np.arange(0, n, 20), n-1])
        plot_data[milestone] = dict(t=d["elapsed_s"][view], net=d["net_angle_deg"][view],
            height=d["object_state"][view, 2], contacts=d["support_groups"][view],
            tilt=d["tilt_deg"][view], fixed_rate=fixed_rate[view], moving_rate=moving_rate[view],
            control_t=end_times, bound_count=limits.sum(1), blocked_count=outward.sum(1),
            control_fixed_rate=fixed_rate[end_indices], control_moving_rate=moving_rate[end_indices],
            control_pose_rate=geometric_control_rate,
            **{f"reward_{k}": v for k, v in components.items()})
        np.savez_compressed(out/f"{milestone}m_diagnostic.npz", **plot_data[milestone])
        result["training"][milestone] = load_training(run, manifest)
        weights = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = build_model().cpu().eval()
        model.load_state_dict(weights["model"])
        norm = reference_module("hora_normalizer").RunningMeanStd((96,)).cpu().eval()
        norm.load_state_dict(weights["running_mean_std"])
        models[milestone], norms[milestone] = model, norm
        with torch.no_grad():
            obs = torch.from_numpy(ctl["obs"])
            priv = torch.from_numpy(ctl["priv_info"])
            mu, logstd, *_ = model._actor_critic(dict(obs=norm(obs), priv_info=priv))
            sigma = logstd.exp()
            replay_error = float((mu.clamp(-1, 1)-torch.from_numpy(ctl["action"])).abs().max())
            assert replay_error < 1e-4, replay_error
            old = torch.from_numpy(old_commands)
            m = len(old)
            action_lo = torch.maximum(torch.full_like(old, -1), (torch.from_numpy(lower)-old)/RATE_STEP)
            action_hi = torch.minimum(torch.full_like(old, 1), (torch.from_numpy(upper)-old)/RATE_STEP)
            stoch_delta = clipped_normal_mean(mu[:m], sigma[:m], action_lo, action_hi)*RATE_STEP
            det_delta = mu[:m].clamp(action_lo, action_hi)*RATE_STEP
            action_info = dict(stored_action_replay_max_abs_error=replay_error,
                raw_mean_abs=float(mu.abs().mean()), bounded_mean_abs=float(mu.clamp(-1, 1).abs().mean()),
                raw_mean_outside_action_bounds_fraction=float((mu.abs()>1).float().mean()),
                sigma_per_joint=sigma[0].tolist(), sigma_mean=float(sigma[0].mean()),
                random_sign_opposite_mean_probability=float((.5*(1-torch.erf(mu.abs()/sigma/np.sqrt(2)))).mean()),
                deterministic_requested_target_travel_mean_rad=float(det_delta.abs().mean()),
                stochastic_expected_target_drift_abs_mean_rad=float(stoch_delta.abs().mean()),
                deterministic_vs_expected_stochastic_target_delta_abs_mean_rad=float((stoch_delta-det_delta).abs().mean()),
                interpretation="Fixed-state Gaussian moments only; no stochastic rollout or rotation-performance claim")
        result["inference"][milestone] = action_info
        print(json.dumps(dict(milestone=milestone, common5s=ep["common5s_net_deg"],
            last_contact=last_contact, post_contact_angle=ep["valid_net_after_last_visible_contact_deg"],
            commands=command_windows.get("20to30s", command_windows.get("first5s")),
            reward=reward_windows.get("20to30s", reward_windows.get("first5s")),
            replay_error=replay_error)), flush=True)
        del d, quat, previous, weights
    # Compare policies on the SAME archived common-prefix observations. This is
    # behavioral drift evidence, not a counterfactual physical evaluation.
    cross = []
    with torch.no_grad():
        for source, ctl in controls.items():
            idx = ctl["step"]*DT < 5.
            obs = torch.from_numpy(ctl["obs"][idx])
            priv = torch.from_numpy(ctl["priv_info"][idx])
            anchor_mu, anchor_logstd, *_ = models[10]._actor_critic(dict(obs=norms[10](obs), priv_info=priv))
            for target in (20, 30, 40, 50):
                mu, logstd, *_ = models[target]._actor_critic(dict(obs=norms[target](obs), priv_info=priv))
                kl = (logstd-anchor_logstd + (anchor_logstd.exp().square()+(anchor_mu-mu).square())/(2*logstd.exp().square())-.5).sum(-1)
                weight_only = models[target].act_inference(dict(obs=norms[10](obs), priv_info=priv)).clamp(-1, 1)
                norm_only = models[10].act_inference(dict(obs=norms[target](obs), priv_info=priv)).clamp(-1, 1)
                anchor = anchor_mu.clamp(-1, 1)
                cross.append(dict(observations_from=source, compared_to10M=target, observations=int(idx.sum()),
                    bounded_action_rmse=float((mu.clamp(-1, 1)-anchor).square().mean().sqrt()),
                    old_to_current_gaussian_KL_mean=float(kl.mean()),
                    weights_changed_old_normalizer_action_rmse=float((weight_only-anchor).square().mean().sqrt()),
                    old_weights_changed_normalizer_action_rmse=float((norm_only-anchor).square().mean().sqrt())))
    result["same_observation_behavior_drift"] = cross
    result["action_names"] = list(ACTION_NAMES)
    result["versions"] = dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__)
    result["wall_s"] = time.monotonic()-started
    result["analysis_script_sha256"] = sha(Path(__file__))
    write_json(out/"diagnosis.json", result)
    # CSV uses observed controls only; it never extends a terminated/censored curve.
    with open(out/"control_curves.csv", "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["milestone_M", "t_s", "bound_target_count", "outward_blocked_count", "fixed_axis_omega_rad_s", "moving_axis_omega_rad_s", "pose_spin_interval_rad_s", *components.keys()])
        for milestone, d in plot_data.items():
            for i, t in enumerate(d["control_t"]):
                writer.writerow([milestone, t, d["bound_count"][i], d["blocked_count"][i],
                                 d["control_fixed_rate"][i], d["control_moving_rate"][i], d["control_pose_rate"][i],
                                 *[float(d[f"reward_{k}"][i]) for k in components]])
    plot(out, plot_data, result)
    print(json.dumps(dict(status="completed", wall_s=time.monotonic()-started, output=str(out)), ensure_ascii=False), flush=True)


def plot(out, data, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#000000"]
    styles = ["-", "--", "-.", ":", "-"]
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "savefig.facecolor": "white", "svg.fonttype": "none"}):
        fig, axs = plt.subplots(3, 2, figsize=(13, 10), layout="constrained")
        for (milestone, d), color, style in zip(data.items(), colors, styles):
            axs[0, 0].plot(d["t"], d["net"], color=color, ls=style, lw=1.7, label=f"{milestone}M")
            axs[0, 0].plot(d["t"][-1], d["net"][-1], marker="x", color=color)
            m = d["t"] <= 5
            axs[0, 1].plot(d["t"][m], d["net"][m], color=color, ls=style, lw=1.7, label=f"{milestone}M")
        axs[0, 0].set(title="A  Original-state episodes (x = observed endpoint)", xlabel="Simulation time (s)", ylabel="Signed valid-prefix angle (deg)")
        axs[0, 0].legend(ncol=3)
        axs[0, 1].set(title="B  Common first 5 s", xlabel="Simulation time (s)", ylabel="Signed valid-prefix angle (deg)")
        d = data[50]
        terminal = d["t"] >= 34.8
        axs[1, 0].plot(d["t"][terminal], d["net"][terminal], color="black", label="50M net angle")
        contact_end = result["episodes"][50]["last_visible_hand_contact_s"]
        axs[1, 0].axvline(contact_end, color="#D55E00", ls="--", label="Last sensor-visible hand contact")
        axs[1, 0].legend(fontsize=8)
        axs[1, 0].set(title="C  50M terminal angle", xlabel="Simulation time (s)", ylabel="Signed angle (deg)")
        axs[1, 1].plot(d["t"][terminal], d["height"][terminal]*1000, color="#0072B2", label="Object center")
        axs[1, 1].axhline(66.801253194, color="black", ls="--", label="Existing workspace lower plane")
        axs[1, 1].axvline(contact_end, color="#D55E00", ls="--")
        axs[1, 1].set(title="D  50M terminal height", xlabel="Simulation time (s)", ylabel="Height (mm)")
        axs[1, 1].legend(fontsize=8)
        axs[2, 0].step(d["control_t"], d["bound_count"], color="#0072B2", label="Targets at software bounds")
        axs[2, 0].step(d["control_t"], d["blocked_count"], color="#D55E00", ls="--", label="Outward commands blocked")
        axs[2, 0].set(title="E  50M motor target limits", xlabel="Simulation time (s)", ylabel="Motor count (of16)", ylim=(-.5, 16.5))
        axs[2, 0].legend(fontsize=8)
        d = data[20]
        m = (d["control_t"] > 20) & (d["control_t"] <= 30)
        for key, color, style, label in zip(["control_fixed_rate", "control_moving_rate", "control_pose_rate"],
                ["#0072B2", "#D55E00", "#000000"], ["-", "--", "-."],
                ["Reported omega: fixed axis", "Reported omega: moving axis", "Actual pose spin: 50ms intervals"]):
            axs[2, 1].plot(d["control_t"][m], d[key][m], color=color, ls=style, lw=1, label=label)
        axs[2, 1].set(title="F  20M velocity / pose mismatch", xlabel="Simulation time (s)", ylabel="Signed angular rate (rad/s)")
        axs[2, 1].legend(fontsize=8)
        for ax in axs.flat:
            ax.grid(alpha=.2)
        fig.suptitle("Offline diagnosis: 5 archived episodes, 0 new physics / learning steps", fontsize=13)
        fig.savefig(out/"diagnosis.png", dpi=160)
        fig.savefig(out/"diagnosis.svg")
        plt.close(fig)
        result["figure_provenance"] = dict(size_inches=[13, 10], dpi=160, pixels=[2080, 1600],
            matplotlib=matplotlib.__version__, formats=["png", "svg"],
            transformations=result["diagnostic_definitions"]["plots"],
            uncertainty="None: five different checkpoints, one trajectory each, no replicate estimate",
            description="A/B compare full observed angle histories and common first5s. C/D align terminal50M angle and object height on separate axes; vertical line marks last sensor-visible hand contact. E shows software target saturation. F contrasts reported20M angular velocities at control endpoints with actual quaternion motion over complete50ms control intervals. Crosses mark actual observed endpoints, including wall-budget censoring.")
        write_json(out/"diagnosis.json", result)


if __name__ == "__main__":
    main()
