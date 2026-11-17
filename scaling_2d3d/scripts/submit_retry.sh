#!/bin/bash
# submit_retry.sh MAXTRY qsub-args...   Submit; if PBS fails to start it (Exit_status -3/3, or held with run_count>=5) delete and resubmit.
# Exit 0 when a job finished with any other status (check its log!), 1 when all tries are used.
max=${1:-5}; shift
for i in $(seq 1 "$max"); do
  id=$(qsub "$@") || exit 1
  echo "$(date +%T) try $i: $id"
  while qstat "$id" >/dev/null 2>&1; do
    st=$(qstat -f "$id" 2>/dev/null | awk '/job_state/ {print $3}'); rc=$(qstat -f "$id" 2>/dev/null | awk '/run_count/ {print $3}')
    if [ "$st" = "H" ] && [ "${rc:-0}" -ge 5 ]; then echo "  held (run_count=$rc), deleting"; qdel "$id" 2>/dev/null; sleep 5; break; fi
    sleep 15
  done
  es=$(qstat -xf "$id" 2>/dev/null | awk '/Exit_status/ {print $3}')
  echo "  $id Exit_status=$es"
  case "$es" in -3|3|-) continue ;; *) echo "finished: $id (Exit_status=$es)"; exit 0 ;; esac
done
echo "gave up after $max tries"; exit 1
