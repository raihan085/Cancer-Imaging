"""MRI-anchored, identifiability-aware glioma treatment-scenario framework.

Anatomy and tumour-compartment observations come from the patient's MRI;
tumour kinetics that the scan can constrain are fitted with uncertainty;
drug and immune kinetics remain population-informed priors.  Outputs are
uncertainty-aware treatment scenarios and treatment-equivalence classes --
not individual prescriptions.
"""

from .domain import LABEL_CLASSES, BrainDomain, FVDiffusion, phantom_domain
from .forward import GliomaModel, TumorState, param_tensors, summarize_state
from .inference import LaplacePosterior, SeedingInverseProblem
from .observation import ObservationModel
from .params import PARAM_SPECS, ModelParams
from .pk import DosingSchedule, cycles, daily, no_drug, standard_scenarios

__version__ = "0.1.0"

__all__ = [
    "LABEL_CLASSES", "BrainDomain", "FVDiffusion", "phantom_domain", "GliomaModel", "TumorState",
    "param_tensors", "summarize_state", "LaplacePosterior", "SeedingInverseProblem", "ObservationModel",
    "PARAM_SPECS", "ModelParams", "DosingSchedule", "cycles", "daily", "no_drug", "standard_scenarios",
]
