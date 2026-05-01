from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


def _is_number(value: Any) -> bool:
	return isinstance(value, (int, float)) and not isinstance(value, bool)


def _flatten_metrics(data: dict[str, Any], prefix: str = "") -> dict[str, float]:
	metrics: dict[str, float] = {}
	for key, value in data.items():
		metric_name = f"{prefix}.{key}" if prefix else key
		if _is_number(value):
			metrics[metric_name] = float(value)
		elif isinstance(value, dict):
			metrics.update(_flatten_metrics(value, metric_name))
	return metrics


def _find_group_value(data: dict[str, Any], *candidates: str, default: str = "unknown") -> str:
	normalized = {str(key).lower(): value for key, value in data.items()}
	for candidate in candidates:
		value = normalized.get(candidate.lower())
		if value is not None:
			return str(value)
	return default


def _find_augmentation_value(data: dict[str, Any]) -> str:
	normalized = {str(key).lower(): value for key, value in data.items()}
	for key in ("augmentation", "augmented", "with_augmentation", "use_augmentation"):
		if key in normalized:
			value = normalized[key]
			if isinstance(value, str):
				lowered = value.strip().lower()
				if lowered in {"true", "yes", "1", "with", "enabled"}:
					return "with_augmentation"
				if lowered in {"false", "no", "0", "without", "disabled"}:
					return "without_augmentation"
				return value
			return "with_augmentation" if bool(value) else "without_augmentation"
	return "unknown"


def compile_results(directory: str | Path, output_file: str | Path | None = None) -> Path:
	directory = Path(directory)
	output_path = Path(output_file) if output_file is not None else directory / "compiled_results.csv"

	grouped_metrics: dict[tuple[str, str, str, str], dict[str, list[float]]] = {}

	for json_file in sorted(directory.glob("*.json")):
		with json_file.open("r", encoding="utf-8") as handle:
			data = json.load(handle)

		if not isinstance(data, dict):
			continue

		group = (
			_find_group_value(data, "experiment_name", "experiment", "setup"),
			_find_group_value(data, "feature_extraction", "feature_extractor", "features"),
			_find_group_value(data, "classifier", "model", "estimator"),
			_find_augmentation_value(data),
		)

		metrics = _flatten_metrics(data.get("metrics", data))
		if not metrics:
			continue

		metric_store = grouped_metrics.setdefault(group, {})
		for metric_name, metric_value in metrics.items():
			metric_store.setdefault(metric_name, []).append(metric_value)

	with output_path.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.writer(handle)
		writer.writerow(
			[
				"experiment_name",
				"feature_extraction",
				"classifier",
				"augmentation",
				"metric",
				"average",
				"std_dev",
				"min",
				"max",
				"count",
			]
		)

		for group in sorted(grouped_metrics):
			metric_store = grouped_metrics[group]
			for metric_name in sorted(metric_store):
				values = metric_store[metric_name]
				count = len(values)
				average = sum(values) / count
				variance = sum((value - average) ** 2 for value in values) / count
				std_dev = math.sqrt(variance)
				writer.writerow(
					[
						*group,
						metric_name,
						average,
						std_dev,
						min(values),
						max(values),
						count,
					]
				)

	return output_path


def compile_results_across_folds_and_domains(
	directory: str | Path, output_file: str | Path | None = None
) -> Path:
	"""Compile statistics grouped by experiment_name, feature_extraction, classifier,
	augmentation, and metric, aggregating values across all folds and domains."""
	directory = Path(directory)
	output_path = (
		Path(output_file)
		if output_file is not None
		else directory / "compiled_results_across_folds_and_domains.csv"
	)

	# key: (experiment_name, feature_extraction, classifier, augmentation, metric)
	grouped: dict[tuple[str, str, str, str, str], list[float]] = {}

	for json_file in sorted(directory.glob("*.json")):
		with json_file.open("r", encoding="utf-8") as handle:
			data = json.load(handle)

		if not isinstance(data, dict):
			continue

		group_base = (
			_find_group_value(data, "experiment_name", "experiment", "setup"),
			_find_group_value(data, "feature_extraction", "feature_extractor", "features"),
			_find_group_value(data, "classifier", "model", "estimator"),
			_find_augmentation_value(data),
		)

		scores = data.get("scores")
		if not isinstance(scores, dict):
			continue

		for fold_domain_metrics in scores.values():
			if not isinstance(fold_domain_metrics, dict):
				continue
			for metric_name, metric_value in fold_domain_metrics.items():
				if not _is_number(metric_value):
					continue
				key = (*group_base, metric_name)
				grouped.setdefault(key, []).append(float(metric_value))

	with output_path.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.writer(handle)
		writer.writerow(
			[
				"experiment_name",
				"feature_extraction",
				"classifier",
				"augmentation",
				"metric",
				"average",
				"std_dev",
				"min",
				"max",
				"count",
			]
		)

		for key in sorted(grouped):
			values = grouped[key]
			count = len(values)
			average = sum(values) / count
			variance = sum((v - average) ** 2 for v in values) / count
			std_dev = math.sqrt(variance)
			writer.writerow([*key, average, std_dev, min(values), max(values), count])

	return output_path
