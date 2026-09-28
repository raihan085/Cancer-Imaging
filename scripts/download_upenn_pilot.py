#!/usr/bin/env python
"""Download a small UPENN-GBM pilot subset (processed DICOM series) from IDC.

    python scripts/download_upenn_pilot.py --out data/idc --n 15 --dry-run
    python scripts/download_upenn_pilot.py --out data/idc --n 15
    python scripts/download_upenn_pilot.py --out data/idc --convert data/nifti   # needs dcm2niix

The full collection exceeds 1 TB: always start with a pilot of ~10-20 patients.
For expert-revised segmentations and DTI/DSC derivative maps use the TCIA NIfTI
package or https://brain.labsolver.org/upenn_gbm.html (see README).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glioma_scenarios.data.download import convert_with_dcm2niix, download_pilot  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=15, help="number of patients (pilot)")
    ap.add_argument("--patients", nargs="*", help="explicit patient IDs, e.g. UPENN-GBM-00001")
    ap.add_argument("--raw-dti-dsc", action="store_true", help="also download raw DTI / DSC series")
    ap.add_argument("--no-seg", action="store_true", help="skip DICOM-SEG series")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--convert", type=Path, help="convert processed MR series to NIfTI into this folder")
    a = ap.parse_args()
    download_pilot(a.out, a.n, a.patients, dry_run=a.dry_run, include_seg=not a.no_seg,
                   include_raw_dti_dsc=a.raw_dti_dsc)
    if a.convert and not a.dry_run:
        convert_with_dcm2niix(a.out, a.convert)


if __name__ == "__main__":
    main()
