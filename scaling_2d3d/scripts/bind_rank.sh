#!/bin/bash
# bind_rank.sh cmd args...  -- pin THIS rank to one core of the cores PBS actually gave the job (cpuset), by local rank.
# Why: `mpirun --bind-to core` binds to physical core 0, which a partial-node PBS cpuset may not contain
#      (hwloc_set_cpubind error).  Used with `mpirun --bind-to none`.  CPUS_ALLOWED_LIST overrides (for tests).
list=${CPUS_ALLOWED_LIST:-$(awk '/^Cpus_allowed_list/ {print $2}' /proc/self/status 2>/dev/null)}
cpus=()
IFS=',' read -ra parts <<< "$list"
for p in "${parts[@]}"; do
  if [[ $p == *-* ]]; then for ((c=${p%-*}; c<=${p#*-}; c++)); do cpus+=("$c"); done
  elif [ -n "$p" ]; then cpus+=("$p"); fi
done
lr=${OMPI_COMM_WORLD_LOCAL_RANK:-${MPI_LOCALRANKID:-${PMI_RANK:-0}}}
if [ "${#cpus[@]}" -gt 0 ] && command -v taskset >/dev/null 2>&1; then
  c=${cpus[$((lr % ${#cpus[@]}))]}
  [ -n "${BIND_VERBOSE:-}" ] && echo "bind_rank: local_rank=$lr -> cpu $c (of ${#cpus[@]} allowed)" >&2
  exec taskset -c "$c" "$@"
fi
exec "$@"
