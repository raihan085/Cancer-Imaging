#!/bin/sh
# run_queue.sh JOBFILE [NPROC] -- run a job list in parallel; finished jobs are
# skipped by run_job.py itself, so the queue can be restarted at any time.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
tr '\n' '\0' < "$1" | xargs -0 -P "${2:-4}" -I{} sh -c '{}'
