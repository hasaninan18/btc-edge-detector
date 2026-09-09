"""Pytest bootstrap.

Puts the repo root on sys.path so `import btc_edge` works without an editable
install, and adds a `--run-network` opt-in so the handful of tests that talk to
live Kalshi/Coinbase are skipped by default.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIXTURES = ROOT / "tests" / "fixtures"


def pytest_addoption(parser):
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="also run tests marked @pytest.mark.network (hit live APIs)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "network: test reaches a live external API")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-network"):
        return
    skip = pytest.mark.skip(reason="needs --run-network")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES
