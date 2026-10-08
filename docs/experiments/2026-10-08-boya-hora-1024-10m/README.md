# Superseded continuation: actual resume test, not the formal fresh run

This run initially launched as a10Mcontinuation from the65536-action preview.
The user immediately corrected formal initialization to FROM SCRATCH,then authorized
using this already-running short continuation to verify resume before fresh training.
It was explicitly stopped;no samples/weights are used in the separate formal run.
The original protocol/source snapshot/launch command remain historical evidence.

Resume verification passed:loaded model,obs/value normalizers exactly match source;
first resumed update has Adamstep640→720,agentactions65536→73728,epoch8→9,
same lineage and changed weights;all recorded losses finite. No extra simulator
launch was needed. [Verification](../2026-10-08-boya-hora-1024-fresh10m/resume_verification.json).

Actual stop was SIGTERM/exit143 after Isaac replaced the early Python signal handler.
The result.json is a LAST-UPDATE snapshot,not a normal completion record:21complete
updates/172032cumulative/106496addedactions recorded;last periodic safe checkpoint
is update16/131072cumulative. Incomplete work after the last snapshot is not counted.
No final checkpoint or evaluation was produced. [Stage](stopped_stage.json),
[last update](last_reported_training_result.json),[saved checkpoint](latest_safe_checkpoint.json).
The continuing test confirms resume behavior but does NOT validate graceful shutdown.
Training entry now reinstalls save-on-stop handlers AFTER Isaac/scene initialization;
periodic checkpoints remain the completed-state fallback. No extra GPU shutdown test
added;new handler identity will be recorded at formal launch. TensorBoard label now
resume_test. The independent formal run starts zero with fresh seed43 learner.
