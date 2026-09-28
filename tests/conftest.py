import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glioma_scenarios.domain import phantom_domain  # noqa: E402


@pytest.fixture(scope="session")
def small_domain():
    return phantom_domain((32, 32), spacing=4.0)
