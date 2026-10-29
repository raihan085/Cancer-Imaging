#!/usr/bin/env python3
"""merge_notebook.py -- fold the whole extended study into ONE self-contained
notebook: every module as a %%writefile cell, the surveillance data and every
result file embedded as compressed base64, followed by the analysis cells of
yemen_cholera_extended.ipynb. The output runs in an empty directory with no
network access and no other files.

    python build_notebook.py && python merge_notebook.py OUT.ipynb
"""
import base64, gzip, io, os, sys, tarfile
import nbformat as nbf

OUT = sys.argv[1] if len(sys.argv) > 1 else "yemen_cholera_final.ipynb"
MODULES = ["seip.py", "data_io.py", "fit.py", "pinn.py", "fit_one.py", "baselines.py",
           "epiestim.py", "run_job.py", "spatial_fit.py", "make_jobs.py", "analysis.py"]
SCRIPTS = ["run_queue.sh", "spatial_chain.sh"]
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

src = nbf.read("yemen_cholera_extended.ipynb", as_version=4)
cells = [src.cells[0]]

cells.append(md("## Environment"))
cells.append(code('''import sys, subprocess
for pkg in ("torch", "scipy", "pandas", "matplotlib", "jinja2"):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)
import numpy as np, torch
print("numpy", np.__version__, "| torch", torch.__version__)'''))

cells.append(md("""## Source modules

Every module the study uses, written to the working directory so this notebook is the single
source. `seip.py`, `data_io.py`, `fit.py` and `pinn.py` are the original study's modules with the
first-interval loader fix; the rest are new:

| module | role |
|---|---|
| `run_job.py` | one fit (any arm, seed, fold, sensitivity variant or synthetic truth) → one result file |
| `baselines.py` | Poisson P-spline and SIR baselines |
| `epiestim.py` | Cori / EpiEstim model-free $R_t$ |
| `spatial_fit.py` | full $(t,x)$ model with shared $\\beta(t)$ or location-varying $\\beta(t,x)$ |
| `make_jobs.py`, `run_queue.sh`, `spatial_chain.sh` | the job lists and the parallel queue |
| `analysis.py` | every table and figure below, from the result files only |"""))
for m in MODULES + SCRIPTS:
    body = open(m).read()
    cells.append(code(f"%%writefile {m}\n{body}"))
cells.append(code('import os\nfor s in ' + repr(SCRIPTS) + ':\n    os.chmod(s, 0o755)'))


def blob_tar(paths):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=9) as tf:
        for p in paths:
            tf.add(p)
    return base64.b64encode(buf.getvalue()).decode()


cells.append(md("""## Data and results, embedded

The surveillance record (`data_real/`) and every result file (`results/`) are packed below, so
nothing has to be fetched or refitted. The results were produced by the job queue in the next
cell's `RUN = True` branch."""))
cells.append(code("DATA_TGZ = " + repr(blob_tar(["data_real"]))))
res_files = sorted(os.path.join(dp, f) for dp, _, fs in os.walk("results")
                   for f in fs if f.endswith(".pkl"))
cells.append(code("RESULTS_TGZ = " + repr(blob_tar(res_files))))
cells.append(code('''import base64, io, tarfile
for blob in (DATA_TGZ, RESULTS_TGZ):
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(blob)), mode="r:gz") as tf:
        tf.extractall(".", filter="data")
print("unpacked data_real/ and results/")'''))

cells.extend(src.cells[1:])
nb = nbf.v4.new_notebook(cells=cells, metadata=src.metadata)
nbf.write(nb, OUT)
print(f"wrote {OUT}: {len(cells)} cells, {len(res_files)} result files, "
      f"{os.path.getsize(OUT) / 1e6:.1f} MB")
