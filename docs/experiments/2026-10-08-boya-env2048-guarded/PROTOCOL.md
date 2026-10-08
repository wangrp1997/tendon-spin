# User-authorized2048 throughput trial with desktop resource protection

Read experiment_state and1024 README/comparison. User explicitly authorizes2048
and asks for a mechanism to avoid hanging the desktop. New factors:2048envs and
an OS-isolated training process with independent watchdog.2updates×8×2048=32768
sampled actions,matching512/1024 total action budgets. Seed43 fresh model,28cached
resets,minibatch512/5epochs;compare throughput only,not learned policy scores.

Physics/task/reward/observations remain previous nominal configuration:40×32mm50g,
original reference orientation/fullgravity,16finger inputs,heldwrists,4passive
mimics,v3 nominal armature/clipped externalPD,TGS16/4,28collision exclusions,
selfcollisions,dt.0005s/control20Hz,.35rad/s,privileged teacher,96+9obs,±.02qnoise.
Per-step5mm/15deg/12N/100rad/s/.05radmimic/finite checks and full logging retained.
No property/size/PD randomization/student/full baseline claim. No four-finger gate.

This machine has32logicalCPUs/about62.5GiB RAM/16GiB GPU. Use a named transient
USER systemd service; no global OS/GPU settings. Before releasing a startup gate,
read back cgroup cpu.max,memory.high,memory.max,memory.swap.max and process Nice.
EnforceCPUQuota400%(4cores),MemoryHigh12GiB,MemoryMax16GiB,MemorySwapMax0,
OOMPolicy=stop,KillMode=control-group,Nice10,IOWeight10,IOSchedulingClass=idle,
RuntimeMaxSec300/TimeoutStopSec8. Watchdog remains normal priority outside this
resource group. Torch/OMP/MKL threads4;headless/no cameras. No competing simulator.

Watchdog samples1Hz:whole-deviceGPUmemory/util/temp,systemMemAvailable,memoryPSI,
childRSS,free disk,training heartbeat. Graceful stop ifGPU>10GiB,MemAvailable<12GiB,
free disk<20GiB,memory full-stall avg10>5%,3consecutiveGPUtelemetry failures,heartbeat
older90s,orGPUutil>95%for15consecutive samples. It writes a stop file,not an OS
SIGSTOP (which would retain allocations). Trainer checks before every control action,
saves the current policy plus optimizer/RNG,complete/partial trace with validity masks,
and exits. Grace8s then SIGTERM;5s later SIGKILL the ENTIRE named service cgroup if
needed. EmergencyGPU>12GiB orMemAvailable<8GiB skips to kill for quick reclamation.
Hard service RuntimeMaxSec also works if the watcher dies. Preserve all prior
checkpoints; no claim that a forcibly killed last in-memory update was saved.

Update/phase heartbeat writes atomic files; outside .step,normal loop also checks
stop requests. Memory hard limit includes process descendants and page cache;GPU
memory cannot be hard-capped by Linux cgroups here. User desktop/GPU driver latency
is not measured; controls reduce exhaustion/stall risk,not an absolute freeze guarantee.

First verify actual service limits and stop escalation using two tiny dummy children
with explicitly injected telemetry (no GPU/large-RAM allocation):cooperative stop and
unresponsive child that must be killed. These are required mechanism tests,not robot
experiments or reported GPU events. If resource-group limits cannot be enforced,
do not launch dependent2048 test. No reinstall/root/global settings changes.

Training collection cap240s/externalservice cap300s. Save every completed update
(this run has2) and at graceful exit. Report actual actions/seconds,peak sampled
resources,any guard action,stop reason and separate old1024 result. No2048 frozen
rotation evaluation or longer training claimed. Local stage commit,remote paused.
