# Native runtime identity

Native MuJoCo tools expect a separate cp311 MuJoCo3.13 runtime at
`runtime/mujoco-patched/`, including Python bindings and C headers. Runtime
binaries are not distributed in Git. The required native
library SHA256 is `c0b309bba9a0913718f86fa375ed5003be09ca2131165cee51f9130fbafbf088`.
`build_provenance.json` retains the original binding build and source revision.
The included collision patch applies to MuJoCo source commit
`123347c0eeab7e13c8da0828ab593bbd95bcf335`. Rebuilding produces a separately
identified binary; confirm model and replay comparisons before changing the
strict hash check. Do not silently substitute a pip MuJoCo or Isaac backend.

Isaac learning uses its existing Isaac Sim/Lab environment independently of this
native runtime. Engine configurations and their policy artifacts remain distinct.
