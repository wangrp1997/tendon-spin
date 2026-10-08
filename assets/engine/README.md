# Native runtime identity

The current host has a self-contained copied cp311 MuJoCo3.13 runtime at
`runtime/mujoco-patched/`, including Python bindings and C headers. It is ignored
in Git, and no execution imports the original repository. The required native
library SHA256 is `c0b309bba9a0913718f86fa375ed5003be09ca2131165cee51f9130fbafbf088`.
`build_provenance.json` retains the original binding build and source revision.
The included collision patch applies to MuJoCo source commit
`123347c0eeab7e13c8da0828ab593bbd95bcf335`. Rebuilding produces a separately
identified binary; confirm model and replay comparisons before changing the
strict hash check. Do not silently substitute a pip MuJoCo or Isaac backend.

A fresh clone needs this ignored runtime artifact and the explicitly recorded
warm-start checkpoint to reproduce the declared teacher-v2 pilot. Both are
already copied locally. The original pilot weights are not a trained student.
Original and patched-engine historical results stay separate.
