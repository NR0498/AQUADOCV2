"""CSV import and row-level validation for water-quality measurements."""

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

from pydantic import ValidationError

from .measurements import (
    FieldMeasurement,
    LaboratoryMeasurement,
    MeasurementRepository,
    SamplingEvent,
)


UNIT_CONVERSIONS = {
    "chlorophyll_a": {
        "ug/l": (1.0, "ug/L"),
        "µg/l": (1.0, "ug/L"),
        "mg/l": (1000.0, "ug/L"),
    },
    "turbidity": {"ntu": (1.0, "NTU")},
    "total_suspended_solids": {
        "mg/l": (1.0, "mg/L"),
        "g/l": (1000.0, "mg/L"),
    },
    "dissolved_oxygen": {
        "mg/l": (1.0, "mg/L"),
        "g/l": (1000.0, "mg/L"),
    },
    "ph": {
        "ph": (1.0, "pH"),
        "unitless": (1.0, "pH"),
    },
}


REQUIRED_COLUMNS = {
    "measurement_type",
    "sample_id",
    "latitude",
    "longitude",
    "collected_at",
    "parameter",
    "method",
    "source",
}


@dataclass(frozen=True)
class ImportErrorDetail:
    row_number: int
    message: str


@dataclass
class ImportReport:
    total_rows: int = 0
    imported_rows: int = 0
    duplicate_rows: int = 0
    rejected_rows: int = 0
    errors: list[ImportErrorDetail] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_rows": self.total_rows,
            "imported_rows": self.imported_rows,
            "duplicate_rows": self.duplicate_rows,
            "rejected_rows": self.rejected_rows,
            "errors": [error.__dict__ for error in self.errors],
        }


def import_measurements_csv(
    csv_path: str | Path,
    repository: MeasurementRepository,
) -> ImportReport:
    report = ImportReport()
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_COLUMNS - columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")

        for row_number, row in enumerate(reader, start=2):
            report.total_rows += 1
            try:
                inserted = _import_row(row, repository)
                if inserted:
                    report.imported_rows += 1
                else:
                    report.duplicate_rows += 1
            except (ValueError, ValidationError) as exc:
                report.rejected_rows += 1
                report.errors.append(ImportErrorDetail(row_number, _format_error(exc)))
    return report


def _import_row(row: Mapping[str, str], repository: MeasurementRepository) -> bool:
    sample_id = _required(row, "sample_id")
    event = SamplingEvent(
        sample_id=sample_id,
        latitude=_required(row, "latitude"),
        longitude=_required(row, "longitude"),
        collected_at=_required(row, "collected_at"),
        depth_m=_optional_float(row, "depth_m"),
        spatial_uncertainty_m=_optional_float(row, "spatial_uncertainty_m") or 0,
        source=_required(row, "source"),
        source_reference=_optional(row, "source_reference"),
    )
    parameter = _required(row, "parameter")
    numeric_value = _optional_float(row, "numeric_value")
    unit = _optional(row, "unit")
    numeric_value, unit, conversion_factor = _normalize_value_and_unit(
        parameter, numeric_value, unit
    )
    detection_limit = _optional_float(row, "detection_limit")
    if detection_limit is not None:
        detection_limit *= conversion_factor
    common = {
        "sample_id": sample_id,
        "parameter": parameter,
        "numeric_value": numeric_value,
        "categorical_value": _optional(row, "categorical_value"),
        "unit": unit,
        "method": _required(row, "method"),
        "quality": _optional(row, "quality") or "provisional",
        "notes": _optional(row, "notes"),
    }
    measurement_type = _required(row, "measurement_type").lower()
    if measurement_type == "field":
        return repository.save_field(
            event,
            FieldMeasurement(**common, instrument=_optional(row, "instrument")),
        )
    if measurement_type == "laboratory":
        return repository.save_laboratory(
            event,
            LaboratoryMeasurement(
                **common,
                laboratory_name=_required(row, "laboratory_name"),
                analyzed_at=_required(row, "analyzed_at"),
                detection_limit=detection_limit,
            ),
        )
    raise ValueError("measurement_type must be 'field' or 'laboratory'")


def _required(row: Mapping[str, str], name: str) -> str:
    value = (row.get(name) or "").strip()
    if not value:
        raise ValueError(f"{name} is required")
    return value


def _optional(row: Mapping[str, str], name: str) -> Optional[str]:
    value = (row.get(name) or "").strip()
    return value or None


def _optional_float(row: Mapping[str, str], name: str) -> Optional[float]:
    value = _optional(row, name)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


def _format_error(error: ValueError | ValidationError) -> str:
    if isinstance(error, ValidationError):
        return "; ".join(
            f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
            for item in error.errors()
        )
    return str(error)


def _normalize_value_and_unit(
    parameter: str,
    numeric_value: Optional[float],
    unit: Optional[str],
) -> tuple[Optional[float], Optional[str], float]:
    if numeric_value is None:
        return numeric_value, unit, 1.0
    conversions = UNIT_CONVERSIONS.get(parameter)
    if conversions is None:
        return numeric_value, unit, 1.0
    if unit is None or unit.lower() not in conversions:
        supported = ", ".join(sorted(conversions))
        raise ValueError(f"unsupported unit for {parameter}; expected one of: {supported}")
    factor, canonical_unit = conversions[unit.lower()]
    return numeric_value * factor, canonical_unit, factor
