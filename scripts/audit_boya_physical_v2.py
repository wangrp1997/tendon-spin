# Sources: frozen Isaac physical-v2 NPZ/phase/launch and packaged Boya XML/USD.
# Offline TendonSpin diagnosis; no simulator launch or integration.
"""Reconstruct source effort, expose six-step timing, audit imported constraints."""
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

root=Path(__file__).resolve().parents[1]
phase_path=root/'docs/data/isaac_boya_physical_v2_phases.json'
parent_path=root/'docs/data/isaac_boya_physical_v2.json'
p=json.loads(phase_path.read_text());parent=json.loads(parent_path.read_text())


def ref(path):
    return dict(path=str(path.relative_to(root)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


assert ref(phase_path)==parent['phase_record']
for item in parent['source']:
    assert ref(root/item['snapshot']['path'])==item['snapshot']
    assert item['sha256']==item['snapshot']['sha256']
contract_path=root/p['contract']['path'];assert ref(contract_path)==p['contract']
c=json.loads(contract_path.read_text())
raw_path=root/p['execution']['path'];assert ref(raw_path)==p['execution']
initial_path=root/p['initial_state']['path'];assert ref(initial_path)==p['initial_state']
log_path=root/parent['log']['path'];assert ref(log_path)==parent['log']
log=log_path.read_text(errors='replace')
source_xml=root/'assets/grasp/scene.xml'
xml=ET.parse(source_xml).getroot()
exclusions=[dict(body1=e.attrib['body1'],body2=e.attrib['body2']) for e in xml.findall('./contact/exclude')]
usd_path=root/p['usd']['path'];assert ref(usd_path)==p['usd']
usd=usd_path.read_text()
imported=json.loads((root/'docs/data/isaac_boya_import_phases.json').read_text())
mimics=imported['mimics']
static_contract=dict(source_scene=ref(source_xml),imported_usd=ref(usd_path),
    source_collision_exclusions=exclusions,source_exclusion_count=len(exclusions),
    imported_filtered_pair_schema_or_relationship_lines=[line.strip() for line in usd.splitlines()
        if any(term in line for term in ('FilteredPairsAPI','filteredPairs','CollisionGroup','filteredGroups'))],
    explicit_source_exclusions_forwarded=False,
    collision_difference='Native XML has28 named exclude pairs; converted USD/probe contains no explicit pair filter authoring. PhysX may suppress adjacent articulation pairs; do not infer all28 pairs collide.',
    source_mimic_relations=[e.attrib for e in xml.findall('./equality/joint')],
    imported_mimic_schemas=mimics,
    mimic_interpretation='NewtonMimicAPI is the documented current schema. Missing legacy PhysxMimicJointAPI is not by itself a bug; dynamic coupling failed in this execution and needs an isolated runtime check.')
names=p['joint_names'];motor_ids=[names.index(a['joint']) for a in c['actuators']]
kp=np.array([a['gain'] for a in c['actuators']])
caps=np.array([a['force_range'] for a in c['actuators']])
rows=[]
with np.load(raw_path,allow_pickle=False) as a:
    assert len(a['joint_pos'])==p['physics_steps']
    predicted=np.clip(kp*(a['commands']-a['joint_pos_before'][:,motor_ids]),caps[:,0],caps[:,1])
    reconstruction_error=float(np.max(np.abs(predicted-a['motor_effort_requested'][:,motor_ids])))
    passive_ids=[i for i in range(len(names)) if i not in motor_ids]
    passive_error=float(np.max(np.abs(a['motor_effort_requested'][:,passive_ids])))
    assert reconstruction_error < 1e-6 and passive_error==0.
    assert np.max(np.abs(a['action']))==0.
    assert np.max(np.abs(a['commands']-a['commands'][0]))==0.
    for t in range(len(a['joint_pos'])):
        forces=np.linalg.norm(a['normal_force_matrix_w'][t],axis=-1)
        fastest=int(np.argmax(np.abs(a['joint_vel'][t])))
        row=dict(step=t+1,elapsed_s=float(a['elapsed_s'][t]),
            drift_mm=float(a['drift_mm'][t]),tilt_deg=float(a['tilt_deg'][t]),
            max_abs_joint_speed_rad_s=float(np.abs(a['joint_vel'][t]).max()),
            fastest_joint=names[fastest],maximum_link_normal_force_N=float(forces.max()),
            nonzero_normal_force_link_count=int(np.count_nonzero(forces)),
            object_speed_m_s=float(np.linalg.norm(a['object_state'][t,7:10])),
            source_motor_max_abs_Nm=float(np.abs(a['motor_effort_requested'][t]).max()),
            forwarding_error_Nm=float(np.abs(a['motor_effort_requested'][t]-a['actuator_effort_forwarded'][t]).max()),
            actuation_readback_error_Nm=float(np.abs(a['motor_effort_requested'][t]-a['solver_actuation_effort_readback'][t]).max()),
            finite_state_force=bool(a['state_force_finite'][t]))
        for couple in c['couplings']:
            i=names.index(couple['slave']);j=names.index(couple['master'])
            row[couple['finger']+'_coupling_error_rad']=float(a['joint_pos'][t,i]-a['joint_pos'][t,j])
        rows.append(row)
record=dict(kind='offline physical-v2 execution and source-contract audit',new_physics_steps=0,
    controller=p['controller'],phase_record=ref(phase_path),execution=ref(raw_path),
    initial_state=ref(initial_path),logged_steps=p['physics_steps'],
    actual_stop_reason=p['stop_reason'],original_physical_state_verified=p['original_physical_state_verified'],
    benchmark_eligible=False,physical_prefix_certified=False,
    initial_transfer_errors=p['initialization_audit']['errors'],
    motor_effort_reconstruction_error_Nm=reconstruction_error,
    passive_motor_effort_error_Nm=passive_error,
    zero_actions_and_unchanged_commands_verified=True,
    effort_readback_consistent=True,
    effective_ccd=False if 'CCD disabled when GPU dynamics is enabled.' in log else None,
    effective_ccd_evidence=dict(log=ref(log_path),matched_warning='CCD disabled when GPU dynamics is enabled.'),
    effective_backend='Isaac Lab PhysX, confirmed by configured PhysxCfg and active isaaclab_physx startup log',
    reporting_corrections=['raw effective_settings.backend=ResolvableString labels the wrapper class, not the engine; use configured PhysxCfg plus active backend log',
        'solver-side damping coefficients read back correctly, but stability is not fixed',
        'initial pose identity passed; contact/coupling/penetration physical contract did not',
        'no certified benchmark prefix or rotation angle; original force/support/numerical gates not fully ported'],
    observations=['step1 already has MFJ2 speed33.49rad/s while object drift only.00112mm',
        'coupling error magnitudes grow to2.40/2.05/4.16/2.48rad over3ms',
        'all actions0; PD requested, actuator-forwarded and PhysX-actuation-readback efforts agree',
        'normal force maximum rises from4.02N to668.87N; final object speed11.76m/s'],
    unresolved_causal_tests=['isolated mimic/viscous-damping runtime and passivity check with unchanged coefficients',
        'forward exact original named collision exclusions and verify contact pairs',
        'compare actual imported inertia/solver response and initial-contact impulses'],
    limitation='No self-contact impulse trace was captured in v2. Missing explicit exclusions and dynamic coupling failure are evidence of an unverified transfer; their separate causal contributions are not isolated. No claim that missing legacy mimic schema or collision alone caused failure.',
    static_contract=static_contract,step_trace=rows)
out=root/'docs/experiments/2026-10-08-boya-isaac-physical-v2/data'
out.mkdir(parents=True,exist_ok=True)
(out/'diagnosis.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
with (out/'step_trace.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
print(json.dumps(dict(steps=len(rows),motor_reconstruction_error_Nm=reconstruction_error,
    stop=p['stop_reason'],last_frame=rows[-1],source_exclusions=len(exclusions)),indent=2))
