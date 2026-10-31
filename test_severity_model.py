import csv
import io
import tempfile
import unittest
from pathlib import Path
from openpyxl import Workbook
from unittest.mock import patch

import severity_model


def dataset_csv(rows):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["road_type", "speed_kmh", "severity"])
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


class SeverityModelTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        self.model_path = root / "severity_model.joblib"
        self.metadata_path = root / "severity_model.json"
        self.model_directory = root
        self.rows = [
            ("highway", speed, "fatal")
            for speed in range(80, 120)
        ] + [
            ("local", speed, "severe injury")
            for speed in range(20, 60)
        ]
        self.contents = dataset_csv(self.rows)
        self.paths = patch.multiple(
            severity_model,
            MODEL_DIR=self.model_directory,
            MODEL_PATH=self.model_path,
            METADATA_PATH=self.metadata_path,
        )
        self.paths.start()
        self.addCleanup(self.paths.stop)
        self.addCleanup(self.temporary_directory.cleanup)

    def test_inspect_dataset_reports_columns_without_returning_rows(self):
        result = severity_model.inspect_dataset(self.contents, "accidents.csv")

        self.assertEqual(result["rows"], 80)
        self.assertEqual([column["name"] for column in result["columns"]], ["road_type", "speed_kmh", "severity"])
        self.assertNotIn("data", result)

    def test_inspect_excel_dataset(self):
        workbook = Workbook()
        worksheet = workbook.active
        worksheet.append(["road_type", "speed_kmh", "severity"])
        worksheet.append(["highway", 90, "fatal"])
        file_data = io.BytesIO()
        workbook.save(file_data)

        result = severity_model.inspect_dataset(file_data.getvalue(), "accidents.xlsx")

        self.assertEqual(result["rows"], 1)
        self.assertEqual(result["columns"][1]["kind"], "numeric")

    def test_inspect_rejects_malformed_and_duplicate_header_files(self):
        with self.assertRaisesRegex(severity_model.DatasetError, "Could not read the dataset"):
            severity_model.inspect_dataset(b"not an Excel file", "accidents.xlsx")
        with self.assertRaisesRegex(severity_model.DatasetError, "must be unique"):
            severity_model.inspect_dataset(b"road,road,severity\n1,2,fatal\n", "accidents.csv")

    def test_train_and_predict_persists_model_but_not_dataset(self):
        result = severity_model.train_model(
            self.contents,
            "accidents.csv",
            "severity",
            ["road_type", "speed_kmh"],
            "Test-only generated fixture",
            "Unit-test data; no real records",
        )

        self.assertTrue(result["trained"])
        self.assertEqual(result["rows_used"], 80)
        self.assertEqual(result["target"], "severity")
        self.assertEqual(result["dataset_source"], "Test-only generated fixture")
        self.assertEqual(result["dataset_permission"], "Unit-test data; no real records")
        self.assertIn("majority_baseline_accuracy", result["metrics"])
        self.assertTrue(any("Fatal-class recall" in warning for warning in result["warnings"]))
        self.assertTrue(self.model_path.is_file())
        self.assertTrue(self.metadata_path.is_file())
        prediction = severity_model.predict_severity({"road_type": "highway", "speed_kmh": 100})
        self.assertIn(prediction["severity"], ["fatal", "severe injury"])
        self.assertAlmostEqual(sum(prediction["probabilities"].values()), 1.0, places=3)
        self.assertFalse(any(path.suffix == ".csv" for path in self.model_directory.iterdir()))

    def test_train_rejects_underrepresented_severity_classes(self):
        contents = dataset_csv([("highway", 90, "fatal")] * 4 + [("local", 40, "severe injury")] * 6)

        with self.assertRaisesRegex(severity_model.DatasetError, "at least 5"):
            severity_model.train_model(contents, "accidents.csv", "severity", ["road_type"], "Test fixture", "Unit test")

    def test_train_rejects_target_as_a_feature(self):
        with self.assertRaisesRegex(severity_model.DatasetError, "must exist and must not include"):
            severity_model.train_model(self.contents, "accidents.csv", "severity", ["severity"], "Test fixture", "Unit test")

    def test_train_requires_dataset_provenance(self):
        with self.assertRaisesRegex(severity_model.DatasetError, "source citation"):
            severity_model.train_model(self.contents, "accidents.csv", "severity", ["road_type"], "", "Unit test")


if __name__ == "__main__":
    unittest.main()
