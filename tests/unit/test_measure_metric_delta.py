import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import pytest

SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "measure_metric_delta.py"
SPEC = importlib.util.spec_from_file_location("measure_metric_delta", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
_ids = MODULE._ids
kendall_disagreement = MODULE.kendall_disagreement


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (["a", "b", "c"], ["a", "b", "c"], 0.0),
        (["a", "b", "c"], ["c", "b", "a"], 1.0),
        (["a", "b", "c"], ["a", "c", "x"], 0.0),
        (["a", "b"], ["x", "a"], 0.0),
        ([], [], 0.0),
    ],
)
def test_kendall_disagreement(left, right, expected):
    assert kendall_disagreement(left, right) == expected


def test_ids_bypasses_vector_index_for_metric_comparison():
    table = MagicMock()
    operation = table.search.return_value
    operation.metric.return_value = operation
    operation.bypass_vector_index.return_value = operation
    operation.limit.return_value = operation
    operation.to_polars.return_value.iter_rows.return_value = [
        {"id": "first"},
        {"id": "second"},
    ]

    assert _ids(table, [1.0, 0.0], "cosine", 2) == ["first", "second"]
    operation.metric.assert_called_once_with("cosine")
    operation.bypass_vector_index.assert_called_once_with()
    operation.limit.assert_called_once_with(2)
