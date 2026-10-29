#!/bin/sh
# Spatial fits, most important first. All use the negative-binomial deviance.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
R=results/spatial
python spatial_fit.py --model shared --D 50 --max-nfev 80 --out $R/shared_D50.pkl
python network_fit.py --model network --max-nfev 60 --init $R/shared_D50.pkl --out $R/network.pkl
python spatial_fit.py --model governorate --D 50 --max-nfev 60 --init $R/shared_D50.pkl --out $R/governorate_D50.pkl
python network_fit.py --model isolated --max-nfev 60 --init $R/network.pkl --out $R/isolated.pkl
python network_fit.py --model network_shared --max-nfev 60 --init $R/shared_D50.pkl --out $R/network_shared.pkl
for D in 5 250; do
  python spatial_fit.py --model governorate --D $D --max-nfev 60 --init $R/governorate_D50.pkl --out $R/governorate_D$D.pkl
done
echo SPATIAL_DONE > logs/spatial.done
