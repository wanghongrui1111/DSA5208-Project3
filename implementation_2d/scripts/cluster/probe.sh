#!/bin/bash
# Collects what the job scripts need to know about a cluster. Run on the login
# node and keep (or send) the output:
#   bash scripts/cluster/probe.sh atlas > probe_atlas.txt 2>&1
#   bash scripts/cluster/probe.sh vanda > probe_vanda.txt 2>&1
cluster=${1:-unknown}
[ "$cluster" = atlas ] && [ -f /app1/ebenv ] && source /app1/ebenv
section() { echo; echo "=================== $*"; }

section host;    hostname; head -2 /etc/os-release 2>/dev/null
section cpu;     lscpu | grep -E 'Model name|Socket|Core\(s\)|Thread|NUMA node\(s\)'
section "hpc space";   hpc space 2>&1 | head -20
section gstat;   gstat 2>&1 | head -60
section queues;  qstat -Q 2>&1 | head -40
section "queue limits"
for q in $(qstat -Q 2>/dev/null | awk 'NR > 2 {print $1}'); do
  echo "-- $q"
  qstat -Qf "$q" 2>/dev/null | grep -E 'resources_(max|min|default)|max_run|enabled|started'
done
section "cores / GPUs per node (count of nodes)"
pbsnodes -a 2>/dev/null | grep -E 'resources_available\.(ncpus|ngpus|mem) ' | sort | uniq -c
if [ "$cluster" = vanda ]; then
  section "hpc cpu";  hpc cpu 2>&1 | head -40
  section "hpc gpu";  hpc gpu 2>&1 | head -40
fi
section modules
module avail 2>&1 | grep -iE 'foss|gompi|gcc|openmpi|mpich|intel|impi|mvapich|cmake|cuda|nvhpc|ucx|python' | head -80
section tools
for t in mpirun mpiexec mpicxx nvcc cmake python3 g++; do printf '%-8s %s\n' $t "$(command -v $t || echo -)"; done
g++ --version 2>/dev/null | head -1; cmake --version 2>/dev/null | head -1
