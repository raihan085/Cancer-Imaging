#!/bin/sh
# wait for the first deterministic queue, then the focus-city extras, then PINNs
cd "$(dirname "$0")"
while pgrep -f "jobs_cheap.txt" >/dev/null || pgrep -x xargs >/dev/null; do sleep 20; done
./run_queue.sh jobs_cheap2.txt 4
./run_queue.sh jobs_pinn.txt 4
echo ALL_DONE > logs/chain.done
