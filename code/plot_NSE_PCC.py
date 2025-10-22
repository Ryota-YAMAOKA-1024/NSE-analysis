import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ベースディレクトリと入出力先の定義
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR.parent / "fit" / "fit_results.csv"
DEFAULT_OUTPUT = BASE_DIR.parent.parent / "2K" / "figure"

NOTE_PATTERN = re.compile(r"PCC\s*([-+]?\d*\.?\d+)", re.IGNORECASE)


def _safe_float(value: str) -> float:
    """Convert string to float, returning NaN on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _parse_pcc_value(note: str) -> float:
    """Extract numeric PCC current (A) from note_info."""
    if not note:
        return math.nan

    match = NOTE_PATTERN.search(note)
    if match:
        return _safe_float(match.group(1))

    # Fallback: strip non-numeric characters except dot and minus
    filtered = "".join(ch for ch in note if ch.isdigit() or ch in {".", "-", "+"})
    return _safe_float(filtered)


def load_fit_results(csv_path: Path) -> List[Dict[str, float]]:
    """Load fit results and attach PCC numeric values."""
    with csv_path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)

    results: List[Dict[str, float]] = []
    for row in rows:
        pcc_value = _parse_pcc_value(row.get("note_info", ""))
        enriched_row: Dict[str, float] = {
            "note_info": row.get("note_info", ""),
            "pcc_value": pcc_value,
        }

        for key in (
            "A",
            "A_error",
            "offset",
            "offset_error",
            "Nup_mean",
            "Nup_error",
            "Ndown_mean",
            "Ndown_error",
        ):
            value = _safe_float(row.get(key, ""))
            if key == "A" and not math.isnan(value):
                value = math.fabs(value)
            enriched_row[key] = value

        results.append(enriched_row)

    return results


def _prepare_series(
    data: Iterable[Dict[str, float]], y_key: str, y_err_key: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create x, y, yerr arrays filtered for valid PCC values."""
    xs: List[float] = []
    ys: List[float] = []
    yerrs: List[float] = []

    for row in data:
        x_value = row.get("pcc_value", math.nan)
        y_value = row.get(y_key, math.nan)
        y_err = row.get(y_err_key, math.nan)

        if math.isnan(x_value) or math.isnan(y_value):
            continue

        xs.append(x_value)
        ys.append(y_value)
        yerrs.append(0.0 if math.isnan(y_err) else y_err)

    return np.array(xs, dtype=float), np.array(ys, dtype=float), np.array(yerrs, dtype=float)


def plot_series(
    data: List[Dict[str, float]],
    y_key: str,
    y_err_key: str,
    title: str,
    ylabel: str,
    filename: str,
    output_dir: Path,
) -> None:
    """Create and save a single PCC summary plot."""
    xs, ys, yerrs = _prepare_series(data, y_key, y_err_key)

    if xs.size == 0:
        print(f"Skipping {filename}: no valid data points.")
        return

    order = np.argsort(xs)
    xs, ys, yerrs = xs[order], ys[order], yerrs[order]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(
        xs,
        ys,
        yerr=yerrs,
        fmt="o",
        capsize=4,
        linewidth=1.2,
        color="tab:blue",
        label="experimental data",
    )
    ax.set_xlabel("PCC (A)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate PCC summary plots from fit results.")
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Path to fit_results.csv (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT),
        help="Directory to save summary figures (default: %(default)s)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    data = load_fit_results(input_path)

    plots = [
        (
            "A",
            "A_error",
            "Amplitude A(counts/30sec) vs PCC(A) at 2.6K",
            "Amplitude A (counts/30sec)",
            "Amplitude_vs_PCC.png",
        ),
        (
            "offset",
            "offset_error",
            "mean B(counts/30sec) vs PCC(A) at 2.6K",
            "Mean B (counts/30sec)",
            "MeanB_vs_PCC.png",
        ),
        (
            "Nup_mean",
            "Nup_error",
            "Nup(counts/30sec) vs PCC(A) at 2.6K",
            "Nup (counts/30sec)",
            "Nup_vs_PCC.png",
        ),
        (
            "Ndown_mean",
            "Ndown_error",
            "Ndown(counts/30sec) vs PCC(A) at 2.6K",
            "Ndown (counts/30sec)",
            "Ndown_vs_PCC.png",
        ),
    ]

    for y_key, y_err, title, ylabel, filename in plots:
        plot_series(data, y_key, y_err, title, ylabel, filename, output_dir)


if __name__ == "__main__":
    main()
