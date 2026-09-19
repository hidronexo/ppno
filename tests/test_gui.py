import logging
from pathlib import Path
import queue
from unittest.mock import patch

import pytest

from ppno.gui import (
    catalog_rows,
    network_ids,
    portable_reference,
    project_payload,
    render_project,
    run_optimization,
    save_text,
)


def test_save_text_replaces_document_atomically(tmp_path):
    path = tmp_path / "problem.ext"
    path.write_text("old", encoding="utf-8")

    save_text(path, "new\n")

    assert path.read_text(encoding="utf-8") == "new\n"


def test_render_project_preserves_comments_and_unknown_sections():
    source = """; heading
[TITLE]
Case A

[INP]
old.inp ; retained note

[OPTIONS]
Algorithm DE

[EXTRA]
keep this

[PIPES]
1 old

[END]
"""

    rendered = render_project(
        source,
        "network.inp",
        "pipes.cat",
        ["DA", "PSO"],
        [("P1", "PVC")],
        [("J1", "25.0")],
    )

    assert "network.inp" in rendered
    assert "; retained note" in rendered
    assert "Algorithm DA PSO" in rendered
    assert "[EXTRA]\nkeep this" in rendered
    assert "[PIPE_CATALOG]\npipes.cat" in rendered
    assert "[PIPES]\nP1    PVC" in rendered
    assert "[PRESSURES]\nJ1    25.0" in rendered
    assert "old.inp" not in rendered


def test_network_ids_reads_only_junctions_and_pipes(tmp_path):
    inp = tmp_path / "network.inp"
    inp.write_text(
        """[JUNCTIONS]
J1 10 2
J2 11 3
[RESERVOIRS]
R1 100
[PIPES]
P1 J1 J2 100 50 100 0 Open
[PUMPS]
PU1 J1 R1 HEAD C1
""",
        encoding="utf-8",
    )

    assert network_ids(inp) == (["J1", "J2"], ["P1"])


def test_network_ids_rejects_duplicates(tmp_path):
    inp = tmp_path / "network.inp"
    inp.write_text("[JUNCTIONS]\nJ1 1 1\nJ1 1 1\n[PIPES]\nP1 J1 J1 1 1 1 0 Open\n")

    with pytest.raises(ValueError, match="duplicado"):
        network_ids(inp)


def test_catalog_rows_and_project_payload(tmp_path):
    inp = tmp_path / "network.inp"
    inp.write_text("[JUNCTIONS]\nJ1 1 1\n[PIPES]\nP1 J1 J1 1 1 1 0 Open\n")
    catalog = tmp_path / "pipes.cat"
    catalog.write_text("; comment\nPVC 100 120 10.5\nPVC 150 120 15.0\n")
    project = tmp_path / "problem.ext"
    project.write_text(
        """[INP]
network.inp
[OPTIONS]
Algorithms DE, PSO
[PIPE_CATALOG]
pipes.cat
[PIPES]
P1 PVC
[PRESSURES]
J1 30
[END]
""",
        encoding="utf-8",
    )

    assert catalog_rows(catalog)[0] == ("PVC", "100", "120", "10.5")
    payload = project_payload(project)
    assert payload["inp"] == inp
    assert payload["catalog"] == catalog
    assert payload["algorithms"] == ["DE", "PSO"]
    assert payload["pipes"] == [("P1", "PVC")]
    assert payload["pressures"] == [("J1", "30")]
    assert portable_reference(inp, project) == "network.inp"


def test_catalog_rows_rejects_invalid_shape(tmp_path):
    catalog = tmp_path / "pipes.cat"
    catalog.write_text("PVC 100 120\n")

    with pytest.raises(ValueError, match="esperaban"):
        catalog_rows(catalog)


class FakeOptimization:
    def __init__(self, path):
        self.path = Path(path)
        self.pipes = [{"id": "P1", "group": "PVC", "length": 25.0}]
        self.pipe_sizes = {
            "PVC": [{"diameter": 100.0, "roughness": 120.0, "price": 3.0}]
        }
        self.results = [
            {
                "Algorithm": "UH",
                "Attempt": 1,
                "Success": "YES",
                "Time (s)": "0.10",
                "Simulations": 2,
                "Cost": "75.00",
            }
        ]
        self.best_algorithm_name = "UH"
        self.closed = False

    def solve(self):
        logging.getLogger("ppno.test").info("working")
        return [0]

    def pretty_print(self, solution):
        return None

    def get_cost(self):
        return 75.0

    def close(self):
        self.closed = True


def test_run_optimization_emits_log_and_structured_result(tmp_path):
    events = queue.Queue()

    with patch("ppno.gui.Optimization", FakeOptimization):
        run_optimization(tmp_path / "problem.ext", events)

    emitted = []
    while not events.empty():
        emitted.append(events.get_nowait())
    assert any(kind == "log" and "working" in value for kind, value in emitted)
    done = next(value for kind, value in emitted if kind == "done")
    assert done["cost"] == 75.0
    assert done["pipes"] == [("P1", "PVC", "100.0000", "120.000000", "25.00", "3.00", "75.00")]
