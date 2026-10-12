# Sharpa cylinder: simulation configuration versus hardware recommendation

User-requested source comparison, 2026-10-12. No training, policy rollout,
physics integration, rendering or hardware commands were executed.

| Source | Diameter × axial height |
| --- | --- |
| Current Boya A/B teacher and B40M replay |40 ×32 mm, 50 g |
| Official Sharpa cylinder USD, before task scaling |80 ×64 mm |
| Official Sharpa simulation configuration, scale 0.5 |40 ×32 mm |
| Official Sharpa README hardware recommendation |48 ×60 mm (radius 24 mm) |

The 0.5 value scales a simulated USD asset. It is not an instruction to halve
the dimensions of the recommended physical cylinder. Public simulation settings
and the hardware recommendation differ; the checked sources do not explain why.
The recommendation is not a measurement of the object in every demonstration.

Current Boya dimensions come from the inherited original Boya grasp model,
not a change made for this A/B continuation. Their equality with the declared
Sharpa simulation dimensions does not establish matching hardware conditions.
See [current object contract](../data/boya_native_contract.json) and
[B40M results](../experiments/2026-10-10-boya-pose-ab-continue50m/RESULTS.md).

## Exact code evidence

Official repository snapshot: 95ccda3d948801bb5da4cb7ffea766e03067a63b.

- [Hardware README](https://github.com/sharpa-robotics/sharpa-rl-lab/blob/95ccda3d948801bb5da4cb7ffea766e03067a63b/README.md):
  “A cylinder with radius of 24mm and height of 60mm via 3D priting is recommended under default configuration.”
- [Simulation configuration](https://github.com/sharpa-robotics/sharpa-rl-lab/blob/95ccda3d948801bb5da4cb7ffea766e03067a63b/rl_isaaclab/tasks/inhand_rotate/sharpa_wave_env_cfg.py):
  cylinder.usd, spawn scale(1,1,1), prestartup scale_range=[0.5,0.5,1].
- [Scale function](https://github.com/sharpa-robotics/sharpa-rl-lab/blob/95ccda3d948801bb5da4cb7ffea766e03067a63b/rl_isaaclab/utils/modified_events.py):
  writes sampled uniform XYZ scale to the object root xformOp:scale.
- [Cylinder USD](https://github.com/sharpa-robotics/sharpa-rl-lab/blob/95ccda3d948801bb5da4cb7ffea766e03067a63b/assets/cylinder/cylinder.usd):
  OpenUSD composed mesh bounds are approximately(0.08,0.08,0.064)m;
  object root scale is (1,1,1). Its SHA256 is
  00d9a0120b7512f5cc69d4a32b57805304daa8ccf0d2649749d26d9dee7ca36f,
  exactly equal to the read-only local Sharpa fork asset. The official
  default.urdf also declares radius 0.04m and length 0.064m.
- [Hardware configuration](https://github.com/sharpa-robotics/sharpa-rl-lab/blob/95ccda3d948801bb5da4cb7ffea766e03067a63b/rl_isaaclab/tasks/inhand_rotate/sharpa_wave_deploy_env_cfg.py)
  does not contain an object-geometry 0.5 multiplier.

These are source-derived dimensions, not a newly executed upstream baseline.
Raw official downloads, hashes and lookup response stay in ignored
outputs/references/sharpa_size_2026-10-12/.

## Paper scope

A bounded OpenAlex search for SharpaWave returned 9 records, all 9 retrieved
on 2026-10-12: https://api.openalex.org/works?search=SharpaWave&per_page=10
with select=id,title,doi,publication_date,primary_location.
The two relevant rotation papers were read from the previously archived full
texts, with exact versions and text hashes in
[paper_manifest.json](2026-10-08-rotation-metrics/data/paper_manifest.json).

- [Tacmap,2602.21625v2](https://arxiv.org/abs/2602.21625v2), Sim-to-Real
  Application: the reported rotation hardware test uses a spherical object.
  Its cylinder contact-characterization experiment is a different task.
- [TacBPM,2609.18174v1](https://arxiv.org/abs/2609.18174v1), Experimental
  Setup: eight teacher sphere scales0.3–1.0 of nominal8cm diameter; hardware
  rotation evaluation includes tennis balls and other shapes.

Neither passage establishes that this repository's cylindrical demo used an
identical40×32mm physical object or explains its48×60mm README recommendation.
The repository README does not identify either paper as the cylinder demo's
reproduction protocol. Do not merge their geometry or result claims.
