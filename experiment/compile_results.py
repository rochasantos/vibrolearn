from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


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


def _extract_split_label(raw_key: str) -> str:
	lower = raw_key.lower()
	for separator in ("__", "|", "/", "-", ":", ";", ","):
		if separator in raw_key:
			parts = [part.strip() for part in raw_key.split(separator)]
			for part in parts:
				part_lower = part.lower()
				if "fold" in part_lower:
					return part
			for part in parts:
				part_lower = part.lower()
				if "domain" in part_lower:
					return part
	if "fold" in lower or "domain" in lower:
		return raw_key.strip()
	return raw_key.strip() if raw_key.strip() else "unknown_split"


def generate_paired_augmentation_boxplots(
	directory: str | Path,
	output_dir: str | Path | None = None,
	metric: str | None = None,
) -> list[Path]:
	"""Generate box plots per setup comparing with/without augmentation per fold/domain and overall."""
	directory = Path(directory)
	plots_dir = Path(output_dir) if output_dir is not None else directory / "boxplots"
	plots_dir.mkdir(parents=True, exist_ok=True)

	# setup key: (experiment_name, feature_extraction, classifier, metric)
	# value: augmentation -> split (fold/domain) -> values
	store: dict[
		tuple[str, str, str, str],
		dict[str, dict[str, list[float]]],
	] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

	for json_file in sorted(directory.glob("*.json")):
		with json_file.open("r", encoding="utf-8") as handle:
			data = json.load(handle)

		if not isinstance(data, dict):
			continue

		group_base = (
			_find_group_value(data, "experiment_name", "experiment", "setup"),
			_find_group_value(data, "feature_extraction", "feature_extractor", "features"),
			_find_group_value(data, "classifier", "model", "estimator"),
		)
		augmentation = _find_augmentation_value(data)
		scores = data.get("scores")
		if not isinstance(scores, dict):
			continue

		for fold_domain_key, fold_domain_metrics in scores.items():
			if not isinstance(fold_domain_metrics, dict):
				continue
			split_label = _extract_split_label(str(fold_domain_key))
			for metric_name, metric_value in fold_domain_metrics.items():
				if metric is not None and metric_name != metric:
					continue
				if not _is_number(metric_value):
					continue
				setup_key = (*group_base, metric_name)
				store[setup_key][augmentation][split_label].append(float(metric_value))

	output_paths: list[Path] = []
	for setup_key in sorted(store):
		experiment_name, feature_extraction, classifier, metric_name = setup_key
		augmentation_data = store[setup_key]
		with_data = augmentation_data.get("with_augmentation", {})
		without_data = augmentation_data.get("without_augmentation", {})

		all_splits = sorted(set(with_data) | set(without_data))
		if not all_splits:
			continue

		box_data: list[list[float]] = []
		positions: list[float] = []
		labels: list[str] = []
		colors: list[str] = []
		tick_positions: list[float] = []
		tick_labels: list[str] = []

		for idx, split in enumerate(all_splits):
			center = idx * 3.0
			tick_positions.append(center)
			tick_labels.append(split)

			without_values = without_data.get(split, [])
			if without_values:
				box_data.append(without_values)
				positions.append(center - 0.4)
				labels.append("without")
				colors.append("tab:orange")

			with_values = with_data.get(split, [])
			if with_values:
				box_data.append(with_values)
				positions.append(center + 0.4)
				labels.append("with")
				colors.append("tab:blue")

		all_without = [value for fold_values in without_data.values() for value in fold_values]
		all_with = [value for fold_values in with_data.values() for value in fold_values]
		overall_center = len(all_splits) * 3.0
		tick_positions.append(overall_center)
		tick_labels.append("overall")

		if all_without:
			box_data.append(all_without)
			positions.append(overall_center - 0.4)
			labels.append("without")
			colors.append("tab:orange")
		if all_with:
			box_data.append(all_with)
			positions.append(overall_center + 0.4)
			labels.append("with")
			colors.append("tab:blue")

		if not box_data:
			continue

		fig, ax = plt.subplots(figsize=(max(8, len(all_splits) * 1.8), 5))
		bp = ax.boxplot(box_data, positions=positions, widths=0.7, patch_artist=True)
		for patch, color in zip(bp["boxes"], colors):
			patch.set_facecolor(color)
			patch.set_alpha(0.6)

		ax.set_xticks(tick_positions)
		ax.set_xticklabels(tick_labels, rotation=30, ha="right")
		ax.set_ylabel(metric_name)
		ax.set_title(
			f"{experiment_name} | {feature_extraction} | {classifier}\n"
			f"Paired splits and overall: with vs without augmentation"
		)
		ax.grid(axis="y", linestyle="--", alpha=0.4)

		legend_handles = [
			plt.Line2D([0], [0], color="tab:orange", lw=8, alpha=0.6, label="without_augmentation"),
			plt.Line2D([0], [0], color="tab:blue", lw=8, alpha=0.6, label="with_augmentation"),
		]
		ax.legend(handles=legend_handles, loc="best")

		safe_name = "__".join(
			part.replace("/", "_").replace(" ", "_")
			for part in (experiment_name, feature_extraction, classifier, metric_name)
		)
		plot_path = plots_dir / f"{safe_name}_paired_boxplot.png"
		fig.tight_layout()
		fig.savefig(plot_path, dpi=150)
		plt.close(fig)
		output_paths.append(plot_path)

	return output_paths

