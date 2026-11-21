import unittest
from pathlib import Path

from aquadoc_core.persistence import DEFAULT_DATABASE_PATH, PROJECT_ROOT


class DefaultPersistencePathTests(unittest.TestCase):
    def test_default_database_is_anchored_to_project_root(self):
        self.assertTrue(DEFAULT_DATABASE_PATH.is_absolute())
        self.assertEqual(
            DEFAULT_DATABASE_PATH,
            PROJECT_ROOT / "data" / "aquadoc.sqlite3",
        )
        self.assertEqual(DEFAULT_DATABASE_PATH.name, "aquadoc.sqlite3")
        self.assertEqual(DEFAULT_DATABASE_PATH.parent.name, "data")


if __name__ == "__main__":
    unittest.main()
