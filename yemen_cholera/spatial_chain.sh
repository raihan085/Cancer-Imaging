#!/bin/sh
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
R=results/spatial
python spatial_fit.py --model shared --D 50 --max-nfev 80 --out $R/shared_D50.pkl
python spatial_fit.py --model governorate --D 50 --max-nfev 60 --init $R/shared_D50.pkl --out $R/governorate_D50.pkl
python spatial_fit.py --model smooth --D 50 --max-nfev 60 --init $R/shared_D50.pkl --out $R/smooth_D50.pkl
for D in 5 250; do
  python spatial_fit.py --model shared --D $D --max-nfev 60 --init $R/shared_D50.pkl --out $R/shared_D$D.pkl
  python spatial_fit.py --model governorate --D $D --max-nfev 60 --init $R/governorate_D50.pkl --out $R/governorate_D$D.pkl
done
echo SPATIAL_DONE > logs/spatial.done
