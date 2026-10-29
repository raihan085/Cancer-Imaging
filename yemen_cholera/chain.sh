#!/bin/sh
# Non-PINN fits first (they feed every section), then the PINN queue.
cd "$(dirname "$0")"
./run_queue.sh jobs_cheap.txt 4
./run_queue.sh jobs_pinn.txt 4
echo ALL_DONE > logs/chain.done
