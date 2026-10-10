# Frozen20M velocity/pose diagnostic: two original-state episodes

User explicitly approved the proposed two30s physics diagnostics on2026-10-10.
Read workspace50M protocol/results and the offline/source-chain diagnosis before
this experiment. This is diagnostic evaluation, not continued training.

## Frozen task and policy

Use EXACT20M final teacher:
outputs/boya_hora1024_workspace50m_v1/actions_020000000/training/teacher_final.pth,
SHA2569781442fbfe0865946b7d3268a1ad379f796ca967aa8997f6dc427b8263f226e.
Ownmodel/observationnormalizer, deterministicboundedmean, seed43, nolearning.
Each episode independently starts originalgrasp44, NOT a cached or mid-episode
state, and runs uninterrupted until30s or originalphysical/numericalstop.
Noresets, policyswitches, actionreplay, scriptedcontroller or rewardchange.

Same40mm×32mm/50g cylinder/fullgravity/referenceorientation; originalcenter
[-.365438990352,.018340188315,.106061099740]m, WXYZquaternion
[.175062180804,-.901031779718,.346105303444,-.194180544127].
16fingeractions(TH4+FF3+MF3+RF3+LF3), heldwrists, fourpassivemimics,
96propriohistory+9privileged/8latent, noDR/student/tactileactor.
Existing28cache only constructs the environment; evaluation never cache-resets.

IsaacSim6.1.0.0/Lab3.0.0rc1/PhysX/TGS, v3externalclippedPD, prototype
armature4*D_effective*dt, originalCADconvexselfcollision/28excludes, .5friction,
CCDoff, dt.0005s/control20Hz/targetrate.35rad/s. OriginalHoraformula is reused
only to reconstruct per-control rewards; rewards do not affect frozen actions.
boya_workspace: centerz>=.066801253194m and XY bounds
[-.459866091313,-.069149973487]to[-.259054954510,.076535408821]m,
failure aftertwo20Hzoutside samples. Lowcontactdiagnosticonly.
Finite/jointspeed100rad/s/mimic.05 checks retained; oldstrictshadow retained.
Training20stimeout is not applied to these frozen evaluations.

## Only changed experimental factor

A: originalPhysxCfg(min_position_iteration_count=16,min_velocity_iteration_count=4).
B: sameconfiguration EXCEPT min_velocity_iteration_count=16 (4->16).
Maxiterationbounds and authored actor/articulation iterationrequests stayoriginal.
Lab documents islanditerationcounts as maxactorrequest clamped byscenemin/max.
Verify actualUSD sceneattribute before stepping; record actor requests and
allscenephysicsattributes. Treat B as a separately evaluated engineconfiguration.
The override is a local experiment factory wrapper; originalhashedcore files and
installedruntime files are untouched. No solver/backend or contactpropertychange.

## Budget and recording

Exactlyoneepisode perconfiguration, sequentiallyAthenB, maximum60000steps each,
120000totalphysicssteps/0trainingactions.30srequested/1500swallbudget each.
Stopchain onexecutionerror/manualstop, noautomaticretry oradditionalparametersearch.
Normalphysicalearlyfailure is its observed result, not a reason to hide or extend it.
Headless/camerasoff/noimages, existingenvironment, process-onlyEULA,4threads/nice10.
Noresourcewatchdog,cgroup,memorycutoff orglobalenvironment edits.

Reuse existingBoyaParallel.advance/measure, originalHora ActorCritic/normalizer,
termination andscore_prefix. Derived evaluator adds rawgetter instrumentation:
afterp.advance andbeforeLabstateaccess, copy PhysXget_transforms/get_velocities;
thenp.measure; repeatrawgettersafterLabaccess. Preserveallstandardperphysicsstep
state/action/contact/gates plus rawpose/velocitybefore/after, controls/observations,
originalstate, checkpointsourcehashes, actualconfiguration andruntimehashes.
CopyrawGPUbuffers immediately to prevent viewaliasing. No additionalphysicsstep
orRNGdraw forinstrumentation; controlhistory/noise schedule remains original.

## Questions, metrics and interpretation

Compare initialq/qdot/object/commands to archived20M andeachother; require exact
identity for matched original-state reporting. Runtime-sourceandcheckpointidentity
mustpass before execution. Directraw-versusLab pose/velocity andpre/postread
agreement tolerance1e-6 only diagnosesreading, not taskacceptance.

IndependentlyderivefullWORLDrotationvectors fromadjacentXYZWquaternions withSciPy
float64. Compare withrawomega andLabomega onvalidphysicssteps, preservingcontact
labels. Report fullvectorRMSE, meanfixed-original-axisomega, fixed-axisposerate,
moving-axisposerate, reconstructedrotation/penalty/reward means andcontactfraction.
Predeclaredwindows:0-5s,5-20s,20-30s andoverallcommonobservedduration; missingwindows
remainmissing. Report eachownwindow pluscommonprefix; do not extendfailedepisodes.
Primaryepisode evidence remains signedvalid-prefix net/peak/backward/validseconds,
actualstopreason, drift/tilt/forcemaxima. Do not filterangles afterlowcontact.

Raw/Labagreement withpersistentrawvelocity/posedifference localizesbelowtheLab
readingadapter. A reduction with16iterations supportsiteration-sensitivity; no
reduction is a validnegativefinding. No inventedrotationtarget or requiredimprovement
threshold. Two closed-loop trajectories may differ incontactstates, so a result
does not isolate every solvermechanism or establishphysicalfidelity. Onepair is
notstatisticalbaselinevalidation, SOTA, indefinite rotation orhardware readiness.
No follow-on training, algorithmroutechange orpermanentenginefix is authorized
bythispair. Preserveoldrun/checkpoint/provenance andpresentactualfindings.
