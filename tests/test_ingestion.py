import unittest

import numpy as np

from aquadoc_core.ingestion import (
    StacItemSummary,
    choose_best_item,
    coordinate_bbox,
    fetch_sentinel2_chip,
)


class FakeSearchClient:
    def search(self, latitude, longitude, start_date, end_date, size_m):
        return [
            StacItemSummary(
                item_id="cloudier",
                datetime="2026-01-01T00:00:00Z",
                cloud_cover=15.0,
                assets={"B02": "blue", "B03": "green", "B04": "red", "B08": "nir"},
            ),
            StacItemSummary(
                item_id="clearer",
                datetime="2026-01-02T00:00:00Z",
                cloud_cover=2.0,
                assets={"B02": "blue", "B03": "green", "B04": "red", "B08": "nir"},
            ),
        ]


class FakeBandReader:
    def read_bands(self, item, latitude, longitude, size_m, asset_map):
        self.selected_item_id = item.item_id
        shape = (8, 8)
        return {
            "blue": np.full(shape, 0.04),
            "green": np.full(shape, 0.08),
            "red": np.full(shape, 0.03),
            "nir": np.full(shape, 0.02),
        }


class IngestionTests(unittest.TestCase):
    def test_coordinate_bbox_contains_point(self):
        lon_min, lat_min, lon_max, lat_max = coordinate_bbox(12.0, 77.0, 3000)

        self.assertLess(lon_min, 77.0)
        self.assertGreater(lon_max, 77.0)
        self.assertLess(lat_min, 12.0)
        self.assertGreater(lat_max, 12.0)

    def test_choose_best_item_prefers_low_cloud_cover(self):
        item = choose_best_item(
            [
                StacItemSummary("a", "2026-01-02T00:00:00Z", 12.0, {}),
                StacItemSummary("b", "2026-01-01T00:00:00Z", 3.0, {}),
            ]
        )

        self.assertEqual(item.item_id, "b")

    def test_fetch_sentinel2_chip_uses_injected_search_and_reader(self):
        reader = FakeBandReader()

        scene = fetch_sentinel2_chip(
            latitude=12.0,
            longitude=77.0,
            start_date="2026-01-01",
            end_date="2026-01-31",
            search_client=FakeSearchClient(),
            band_reader=reader,
        )

        self.assertEqual(reader.selected_item_id, "clearer")
        self.assertEqual(scene.bands["blue"].shape, (8, 8))
        self.assertIn("clearer", scene.source)


if __name__ == "__main__":
    unittest.main()
