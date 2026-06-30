"""Browse table filter is case/schema-insensitive; grouping keeps original case."""

import polars as pl

from dbs_vector.services.browse import BrowseService


class _ScanStore:
    def __init__(self, df: pl.DataFrame):
        self._df = df

    def scan(self):
        return self._df.to_arrow()


def _df():
    return pl.DataFrame(
        {
            "id": ["1", "2", "3", "4"],
            "tables": [
                ["TryOTODyn.MagentoOrders", "TryOTODyn.Clients"],
                ["MagentoOrders"],
                ["TryOTODyn.MagentoOrdersAddress"],
                ["address.CityTag"],
            ],
            "calls": [1, 1, 1, 1],
        }
    )


def test_filter_is_case_and_schema_insensitive():
    svc = BrowseService(_ScanStore(_df()), frame_alias="t")
    frame = svc._filtered_frame({"table": "magentoorders"}, group_cols=[])
    assert sorted(frame["id"].to_list()) == ["1", "2"]


def test_grouping_keeps_original_case():
    svc = BrowseService(_ScanStore(_df()), frame_alias="t")
    frame = svc._filtered_frame({}, group_cols=["tables"])
    vals = frame["tables"].to_list()
    assert "TryOTODyn.MagentoOrders" in vals
    assert "MagentoOrders" in vals
