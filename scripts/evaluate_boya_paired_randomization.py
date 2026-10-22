"""One frozen C checkpoint on an immutable paired randomized case manifest.

Uses existing PhysX, cache states, termination, native C reward, and teacher.
This separate entry point never learns or resets a case after its first failure.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

from tendonspin.evaluation.paired_randomization import sha256, write_json, summarize

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=30.)
    parser.add_argument('--case-indices', help='Validation subset only; full evaluation uses all cases')
    parser.add_argument('--wall-s', type=float, default=7200.)
    args = parser.parse_args()
    directory = args.suite.resolve()
    suite = json.loads((directory/'suite.json').read_text())
    for source in suite['sources']:
        if sha256(ROOT/source['path']) != source['sha256']:
            raise RuntimeError('Frozen evaluation source changed: ' + source['path'])
    for name, digest in suite['fixture_files'].items():
        if sha256(directory/name) != digest:
            raise RuntimeError('Paired case manifest changed: ' + name)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != suite['gpu_uuid']:
        raise RuntimeError('Require the explicit physical GPU 0 UUID binding')
    if sha256(ROOT/suite['cache']) != suite['cache_sha256']:
        raise RuntimeError('Cache identity changed')
    model_spec = suite['models'][args.model]
    for key in ('checkpoint', 'policy', 'source_record'):
        if sha256(ROOT/model_spec[key]) != model_spec[key+'_sha256']:
            raise RuntimeError('Checkpoint identity changed: ' + key)
    training = json.loads((ROOT/model_spec['source_record']).read_text())
    cases = json.loads((directory/'cases.json').read_text())
    selection = list(range(len(cases))) if not args.case_indices else [int(i) for i in args.case_indices.split(',')]
    if not selection or len(set(selection)) != len(selection) or min(selection) < 0 or max(selection) >= len(cases):
        raise ValueError('Invalid case subset')
    cases = [cases[i] for i in selection]
    if not 0 < args.seconds <= suite['seconds']:
        raise ValueError('Validation horizon must be inside the frozen fixture horizon')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    record = dict(schema='tendonspin-paired-dr-evaluation-v1', pid=os.getpid(), model=args.model,
        gpu_uuid=suite['gpu_uuid'], cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
        num_envs=len(cases), case_indices=selection, requested_s=args.seconds,
        training_actions=0, episode_resets=0, controller_switches=0, automatic_retry=False,
        checkpoint=model_spec, manifest_sha256=suite['fixture_files'], arm='C',
        training_actions_in_checkpoint=training['actions_executed'],
        domain_randomization=True, status='launching', simulation_s=0., completed_cases=0,
        sources=suite['sources'], source_suite=str(directory),
        post_failure='Inert result mask; isolated bodies may continue simulating for other active cases. No resets.',
        reward_boundary='Native C full control including first failing control; inactive cases excluded thereafter')

    def save(status):
        record.update(status=status, wall_s=time.monotonic()-started, updated_unix_s=time.time())
        write_json(out/'result.json', record)
        print('PAIRED_EVAL '+json.dumps({k: record.get(k) for k in (
            'status', 'model', 'num_envs', 'simulation_s', 'completed_cases', 'wall_s', 'estimated_remaining_s')}), flush=True)

    sync_counts = {}
    def waited(name):
        sync_counts[name] = sync_counts.get(name, 0)+1
    from tendonspin.physics.physx_property_sync import install
    record['property_sync'] = install(waited)
    save('launching')
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=False, device='cuda:0').app
    failed = False
    trace = None
    try:
        import numpy as np
        import torch
        from unittest.mock import patch
        from pxr import UsdGeom
        from tendonspin.rl.isaac_spin_progress import SpinProgressBoyaEnv
        from tendonspin.rl.isaac_hora import spin_increment
        from tendonspin.physics.isaac_parallel import tensor
        from tendonspin.baselines.reference_models import build_model, reference_module
        from tendonspin.rl.rotation_reward import require_checkpoint_configuration
        from tendonspin.evaluation.rotation import score_prefix
        torch.set_num_threads(4)
        torch.manual_seed(43)
        np.random.seed(43)
        samples = dict(scale_id=np.array([r['scale_id'] for r in cases]),
            scale=np.array([r['scale'] for r in cases]),
            friction=np.array([r['friction'] for r in cases]),
            com=np.array([r['com_m'] for r in cases]))
        # Supply fixed evaluation inputs through the existing sampler boundary.
        # The original size authoring, material/COM setters and readback remain unchanged.
        with patch('tendonspin.rl.isaac_hora_randomized.static_samples', return_value=samples):
            env = SpinProgressBoyaEnv(ROOT, out, ROOT/suite['cache'], num_envs=len(cases), trace_mode='summary',
                termination_profile=training['termination']['profile'], engine_profile=training['engine_profile'],
                reward_profile=training['reward_profile'], scene_configuration=training['scene_configuration'],
                randomization_configuration=training['domain_randomization_configuration'], training_randomization=True)
        p, dr = env.physics, env.dr
        for key, actual in [('reward_configuration', env.reward_configuration),
                            ('engine_configuration', p.engine_configuration),
                            ('termination', env.termination.spec.to_dict())]:
            if training[key] != actual:
                raise ValueError('Evaluation changed the training ' + key)
        device = p.device
        weights = torch.load(ROOT/model_spec['checkpoint'], map_location=device, weights_only=False)
        require_checkpoint_configuration(weights['contract'], env.reward_configuration)
        for key, value in [('domain_randomization', training['domain_randomization_configuration']),
                           ('scene_configuration', training['scene_configuration']),
                           ('engine_configuration', p.engine_configuration),
                           ('termination', env.termination.spec.to_dict())]:
            if weights['contract'][key] != value:
                raise ValueError('Checkpoint contract mismatch: '+key)
        model = build_model().to(device)
        model.load_state_dict(weights['model']); model.eval(); model.requires_grad_(False)
        norm = reference_module('hora_normalizer').RunningMeanStd((96,)).to(device)
        norm.load_state_dict(weights['running_mean_std']); norm.eval(); norm.requires_grad_(False)
        frozen_hashes = {key: hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes()
            for v in state.values())).hexdigest() for key, state in
            [('model', model.state_dict()), ('running_mean_std', norm.state_dict())]}
        ids = torch.arange(p.num_envs, device=device)
        # Same setter and gain formulas as HoraDomainRandomization.reset; inputs are fixed cases.
        dr.mass.copy_(torch.tensor([r['mass_kg'] for r in cases], device=device))
        dr.p_ratio.copy_(torch.tensor([r['p_ratio'] for r in cases], device=device))
        dr.d_ratio.copy_(torch.tensor([r['d_ratio'] for r in cases], device=device))
        p.object.set_masses_index(masses=dr.mass[:, None], env_ids=ids)
        active_joints = p.adapter.active
        p.adapter.kp[:, active_joints] = dr.base_kp[active_joints][None] * dr.p_ratio
        p.adapter.motor_damping[:, active_joints] = dr.base_damping[active_joints][None] * dr.d_ratio
        dr.force.zero_()
        record['property_readback'] = dr.verify_properties()
        for key, expected in [('kp', dr.base_kp[active_joints][None]*dr.p_ratio),
                              ('motor_damping', dr.base_damping[active_joints][None]*dr.d_ratio)]:
            if not torch.equal(getattr(p.adapter, key)[:, active_joints], expected):
                raise ValueError('PD multiplier application differs')
        geometry = training['scene_configuration']['geometry']
        geo_error = 0.
        for i, row in enumerate(cases):
            cylinder = UsdGeom.Cylinder(p.sim.stage.GetPrimAtPath(f'/World/envs/env_{i}/Cylinder/geometry/mesh'))
            ratio = row['scale']/training['domain_randomization_configuration']['logical_nominal_scale']
            geo_error = max(geo_error, abs(cylinder.GetRadiusAttr().Get()-geometry['diameter_m']*ratio/2),
                abs(cylinder.GetHeightAttr().Get()-geometry['height_m']*ratio))
        if geo_error > 1e-9:
            raise ValueError('Actual USD size differs from paired case')
        record['geometry_max_error_m'] = geo_error
        cache_ids = torch.tensor([r['cache_index'] for r in cases], device=device)
        q = env.cache['q'][cache_ids].clone()
        obj = env.cache['object_state'][cache_ids].clone()
        obj[:, :3] += p.origins; obj[:, 7:] = 0.
        commands = env.cache['commands'][cache_ids].clone()
        p._write_state(q, torch.zeros_like(q), obj, commands)
        env.init_q.copy_(q[:, p.active_joint_ids]); env.cache_ids.copy_(cache_ids)
        env.termination.reset()
        env.history[:] = env._frame(noise=False)[:, None, :]
        initial = p.measure()
        errors = dict(joint_pos=float((initial['joint_pos']-q).abs().max()),
            joint_vel=float(initial['joint_vel'].abs().max()),
            object_state=float((initial['object_state']-obj).abs().max()),
            commands=float((p.adapter.commands-commands).abs().max()))
        if max(errors.values()) > 1e-6:
            raise ValueError('Cache state readback mismatch: '+str(errors))
        record['initial_state_max_errors'] = errors
        initial_obj = initial['object_state'].clone(); initial_obj[:, :3] -= p.origins
        np.savez_compressed(out/'initial_state.npz', joint_pos=initial['joint_pos'].cpu().numpy(),
            joint_vel=initial['joint_vel'].cpu().numpy(), object_state=initial_obj.cpu().numpy(),
            commands=p.adapter.commands.cpu().numpy(), history=env.history.cpu().numpy(),
            cache_index=cache_ids.cpu().numpy())
        actual_mass = tensor(p.object.root_view.get_masses()).reshape(p.num_envs)
        actual_com = tensor(p.object.root_view.get_coms()).reshape(p.num_envs, 1, 7)[..., :3].squeeze(1)
        np.savez_compressed(out/'properties.npz', mass_kg=actual_mass.cpu().numpy(), com_m=actual_com.cpu().numpy(),
            hand_material=tensor(p.hand.root_view.get_material_properties()).cpu().numpy(),
            object_material=tensor(p.object.root_view.get_material_properties()).cpu().numpy(),
            scale=dr.scale.cpu().numpy(), p_ratio=dr.p_ratio.cpu().numpy(), d_ratio=dr.d_ratio.cpu().numpy(),
            kp=p.adapter.kp.cpu().numpy(), damping=p.adapter.motor_damping.cpu().numpy(),
            nominal_inertia=dr.nominal_inertia.cpu().numpy())
        with np.load(directory/'stimuli.npz') as data:
            forces = torch.tensor(data['force_N'][:, selection], device=device)
            noises = torch.tensor(data['noise_rad'][:, selection], device=device)
        controls = round(args.seconds/(p.dt*p.adapter.steps_per_control))
        if abs(controls*p.dt*p.adapter.steps_per_control-args.seconds) > 1e-9:
            raise ValueError('Horizon must contain full controls')
        steps = controls*p.adapter.steps_per_control
        trace = np.lib.format.open_memmap(out/'angles_deg.npy', mode='w+', dtype=np.float64, shape=(steps, p.num_envs))
        net = torch.zeros(p.num_envs, dtype=torch.float64, device=device)
        valid_steps = torch.zeros(p.num_envs, dtype=torch.long, device=device)
        first_failure = torch.zeros_like(valid_steps)
        stopped_at = torch.zeros_like(valid_steps)
        live = torch.ones(p.num_envs, dtype=torch.bool, device=device)
        trace_block = torch.empty((p.adapter.steps_per_control, p.num_envs), dtype=torch.float64, device=device)
        maximum = {key: torch.zeros(p.num_envs, device=device) for key in ('drift_mm', 'tilt_deg')}
        reward_sums = {key: torch.zeros(p.num_envs, dtype=torch.float64, device=device)
            for key in ('reward', 'rotation', 'terminal_failure_cost')}
        control_trace = {key: [] for key in ('obs', 'priv_info', 'action', 'reward', 'rotation', 'terminal_failure_cost', 'first_failure')}
        previous = initial['object_state'][:, 3:7].clone()
        executed = 0
        evaluation_started = time.monotonic()
        record.update(startup_wall_s=evaluation_started-started, engine_configuration=p.engine_configuration,
            reward_configuration=env.reward_configuration, termination=env.termination.spec.to_dict(),
            device_name=torch.cuda.get_device_name(0), normalizer_updates=0)
        save('evaluating')
        with torch.inference_mode():
            for control in range(controls):
                if args.wall_s and time.monotonic()-started >= args.wall_s:
                    record['wall_budget_exhausted'] = True
                    break
                active_at_start = live.clone()
                obs = env.observe()
                action = model.act_inference(dict(obs=norm(obs['obs']), priv_info=obs['priv_info'])).clamp(-1, 1)
                if not torch.isfinite(action).all():
                    raise RuntimeError('Nonfinite frozen policy action')
                p.adapter.set_action(action)
                dr.force.copy_(forces[control])
                env.rotation_reward_signal.begin(previous)
                control_codes = torch.zeros_like(first_failure)
                for substep in range(p.adapter.steps_per_control):
                    before_q, before_v, effort = p.advance()
                    m = p.measure()
                    if not m['finite'].all():
                        raise RuntimeError('Nonfinite physics; stop without retry')
                    env.rotation_reward_signal.advance(m['object_state'][:, 3:7])
                    codes = env.termination.failure_codes(m, p.origins,
                        control_boundary=substep == p.adapter.steps_per_control-1)
                    control_codes = torch.where(control_codes == 0, codes, control_codes)
                    newly_failed = live & (codes != 0)
                    first_failure[newly_failed] = codes[newly_failed]
                    stopped_at[newly_failed] = executed+substep+1
                    live &= codes == 0
                    increment = spin_increment(previous, m['object_state'][:, 3:7])
                    previous = m['object_state'][:, 3:7].clone()
                    net += torch.where(live, torch.rad2deg(increment).double(), 0.)
                    valid_steps += live
                    trace_block[substep] = net
                    for key in maximum:
                        maximum[key] = torch.where(live, torch.maximum(maximum[key], m[key]), maximum[key])
                trace[executed:executed+p.adapter.steps_per_control] = trace_block.cpu().numpy()
                executed += p.adapter.steps_per_control
                tau = effort[:, p.active_joint_ids]
                velocity = (m['joint_pos'][:, p.active_joint_ids]-before_q[:, p.active_joint_ids])/p.dt
                pose = ((m['joint_pos'][:, p.active_joint_ids]-env.init_q)**2).sum(-1)
                components = env.control_reward(m['object_state'][:, 7:10], m['object_state'][:, 10:13],
                    pose, tau.square().sum(-1), (tau*velocity).sum(-1).square(), control_codes)
                if not torch.isfinite(components['reward']).all():
                    raise RuntimeError('Nonfinite native C reward')
                for key, values in reward_sums.items():
                    values += torch.where(active_at_start, components[key].double(), 0.)
                for key in ('obs', 'priv_info'):
                    control_trace[key].append(obs[key].cpu().numpy().copy())
                control_trace['action'].append(action.cpu().numpy().copy())
                for key in ('reward', 'rotation', 'terminal_failure_cost'):
                    control_trace[key].append(torch.where(active_at_start, components[key], 0.).cpu().numpy().copy())
                control_trace['first_failure'].append(first_failure.cpu().numpy().copy())
                # Same raw-radian noise and normalization as HoraBoyaEnv._frame.
                measured = tensor(p.hand.data.joint_pos)[:, p.active_joint_ids]+noises[control]
                frame = torch.cat((2*(measured-env.lower)/(env.upper-env.lower)-1,
                    p.adapter.commands[:, p.adapter.active]), dim=-1)
                env.history = torch.roll(env.history, -1, dims=1); env.history[:, -1] = frame
                if (control+1) % 20 == 0 or control == 0 or not live.any():
                    elapsed = time.monotonic()-evaluation_started
                    remaining = controls-control-1
                    record.update(simulation_s=executed*p.dt, completed_cases=int((~live).sum()),
                        executed_physics_steps=executed,
                        environment_actions_per_s=(control+1)*p.num_envs/elapsed,
                        estimated_remaining_s=remaining*elapsed/(control+1),
                        live_cases=int(live.sum()), synchronized_copies=sync_counts,
                        max_torch_memory_mib=torch.cuda.max_memory_allocated()/1024**2)
                    save('evaluating')
                if not live.any():
                    break
        trace.flush()
        record.update(simulation_s=executed*p.dt, executed_physics_steps=executed,
            completed_cases=int((~live).sum()), live_cases=int(live.sum()), synchronized_copies=sync_counts,
            first_force_readback=dr.first_force_readback)
        vsteps = valid_steps.cpu().numpy()
        codes_np = first_failure.cpu().numpy()
        stopped_np = stopped_at.cpu().numpy()
        maximum_np = {key: value.cpu().numpy() for key, value in maximum.items()}
        rewards_np = {key: value.cpu().numpy() for key, value in reward_sums.items()}
        rows = []
        for i, case in enumerate(cases):
            score = score_prefix(trace[:executed, i], p.dt, int(vsteps[i]), window_s=args.seconds, tail_s=10.)
            reason = env.reasons[int(codes_np[i])] if codes_np[i] else (
                'time limit' if score['complete_window_observed'] else 'wall budget')
            rows.append(dict(case, valid_s=score['valid_seconds'], net_deg=score['net_deg'],
                net_turns=score['net_deg']/360., peak_deg=score['peak_deg'], backward_deg=score['backward_deg'],
                last10s_net_deg=score['full_window_final_tail_net_deg'],
                complete_window=score['complete_window_observed'], stop_reason=reason,
                first_failure_code=int(codes_np[i]), stop_observed_s=float(stopped_np[i]*p.dt) if codes_np[i] else executed*p.dt,
                max_drift_mm=float(maximum_np['drift_mm'][i]), max_tilt_deg=float(maximum_np['tilt_deg'][i]),
                reward_sum=float(rewards_np['reward'][i]), rotation_reward_sum=float(rewards_np['rotation'][i]),
                terminal_failure_cost_sum=float(rewards_np['terminal_failure_cost'][i])))
        write_json(out/'cases_result.json', rows)
        write_json(out/'summary.json', summarize(rows))
        np.savez_compressed(out/'controls.npz', **{k: np.asarray(v) for k, v in control_trace.items()})
        for key, state in [('model', model.state_dict()), ('running_mean_std', norm.state_dict())]:
            digest = hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in state.values())).hexdigest()
            if digest != frozen_hashes[key]:
                raise RuntimeError('Frozen inference state changed: '+key)
        record['inference_state_unchanged'] = frozen_hashes
        record['summary'] = summarize(rows)
        record['wall_budget_exhausted'] = bool(record.get('wall_budget_exhausted', False))
        if record['wall_budget_exhausted']:
            raise RuntimeError('Evaluation incomplete due to wall budget; preserve partial results')
        for source in suite['sources']:
            if sha256(ROOT/source['path']) != source['sha256']:
                raise RuntimeError('Frozen evaluation source changed during evaluation: '+source['path'])
        save('completed')
    except BaseException as error:
        failed = True
        record['error'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        save('error')
    finally:
        if trace is not None:
            trace.flush()
        app.close(exit_code=1 if failed else 0)


if __name__ == '__main__':
    main()
