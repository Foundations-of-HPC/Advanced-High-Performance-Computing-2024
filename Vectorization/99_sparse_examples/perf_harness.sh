
#!/usr/bin/env bash
set -e

exe=${1:-saxpy}
n=${2:-1048576}

event_set_portable="cycles,instructions,branches,branch-misses,cache-misses"
event_set_intel="fp_arith_inst_retired.scalar_double,fp_arith_inst_retired.128b_packed_double,fp_arith_inst_retired.256b_packed_double,fp_arith_inst_retired.512b_packed_double,fp_arith_inst_retired.vector"

echo "[pinning to CPU 0]"
if command -v taskset >/dev/null; then pin="taskset -c 0"; else pin=""; fi

echo "[portable counters]"
$pin perf stat -e ${event_set_portable} ./${exe} ${n} >/dev/null

echo "[intel counters (if available)]"
$pin perf stat -e ${event_set_intel} ./${exe} ${n} >/dev/null || echo "Intel PMU events not available on this machine."
