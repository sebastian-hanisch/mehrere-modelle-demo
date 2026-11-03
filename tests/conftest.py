"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mmd_evaluation as ev  # noqa: E402
from mmd_scenario import grid_netz  # noqa: E402


@pytest.fixture(scope="session")
def leiter():
    """Leiter 2×4 (8 Knoten, 10 Kanten) mit ihrer Aufzählung. Handrechnung bei p = 1/2: brückenlos 4, Grad ≥ 2 an jedem Knoten 7, zusammenhängend 105 von 1024 Kantenmengen."""
    netz = grid_netz(2, 4)
    return netz, ev.netz_table(netz)


@pytest.fixture(scope="session")
def gitter34():
    netz = grid_netz(3, 4)
    return netz, ev.netz_table(netz)
