#64-environment cache28 stage interrupted for user resize

The user requested increased parallelism with desktop headroom. SentSIGINT to
this supervisor/child group; both exited. New128-env stage starts independently.
Last completed-update record:12 PPO updates,6144 actions; no frozen evaluation.
The cached result.json still says update completed/update budget because shutdown
export failed before writing a final result. This is NOT normal budget completion.

The signal interrupted control-step collection. Interrupted-rollout export then
raised KeyError:reward because its final control record was incomplete. Complete
rollout_001..012 and source snapshots remain; the incomplete13th rollout was not
successfully saved. Total additional partially sampled actions are unverified and
not counted as completed PPO data. This is an interruption-export limitation,
not evidence of physics/optimizer failure. Retain the original log and stale result
with this explicit correction. The128-env runner now handles stop requests at
update boundaries to avoid interrupting control-record construction.

interrupted_result.json is the LAST COMPLETED UPDATE snapshot,not a final result.
Actual stop cause and process evidence:outputs/boya_hora_cache28_stage1/
resize_interruption.json,stage.json,training.log. No pooled sample or rotation count.
