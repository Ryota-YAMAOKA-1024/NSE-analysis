import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ベースディレクトリと入出力先の定義
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR.parent / "fit" / "fit_results.csv"
DEFAULT_OUTPUT = BASE_DIR.parent / "figure"
DEFAULT_LIST_FILE = BASE_DIR / "list.txt"

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


RowDict = Dict[str, Union[float, str]]


def load_fit_results(csv_path: Path) -> List[RowDict]:
    """Load fit results and attach PCC numeric values."""
    with csv_path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)

    results: List[RowDict] = []
    for row in rows:
        pcc_value = _parse_pcc_value(row.get("note_info", ""))
        temp_raw = row.get("temperature", "")
        if isinstance(temp_raw, str):
            temp_clean = temp_raw.strip()
        elif temp_raw is None:
            temp_clean = ""
        else:
            temp_clean = str(temp_raw).strip()

        enriched_row: RowDict = {
            "note_info": row.get("note_info", ""),
            "pcc_value": pcc_value,
            "temperature": temp_clean,
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
    data: Iterable[RowDict], y_key: str, y_err_key: str
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

def _extract_temperature_label(data: Iterable[RowDict]) -> str:
    """データから最初の温度ラベルを取得する"""
    for row in data:
        temp = row.get("temperature")
        if isinstance(temp, str):
            temp = temp.strip()
            if temp:
                return temp
    return ""


def _read_temperature_from_list(list_path: Path) -> str:
    """list.txt から TEMP= を読み取る"""
    if not list_path.exists():
        return ""
    try:
        with list_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if stripped.lower().startswith("temp="):
                    return stripped.split("=", 1)[1].strip()
    except OSError:
        return ""
    return ""


def _format_temperature_label(label: str) -> str:
    """温度表記を '2.6 K' のような形式に整える"""
    if not label:
        return ""
    raw = label.strip()
    if not raw:
        return ""
    raw = raw.replace("℃", "C")  # unexpected char fallback
    if raw[-1].lower() == "k":
        value = raw[:-1].strip()
        if value:
            return f"{value} K"
        return "K"
    # if already contains ' K' or unit inside, keep
    return raw


def _read_meastime_from_list(list_path: Path) -> float:
    """list.txt から meastime= を読み取り、floatとして返す。見つからない場合は math.nan。"""
    if not list_path.exists():
        return math.nan
    try:
        with list_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if stripped.lower().startswith("meastime"):
                    value = stripped.split("=", 1)[1].strip()
                    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                        value = value[1:-1].strip()
                    parsed = _safe_float(value)
                    return parsed
    except OSError:
        return math.nan
    return math.nan


def plot_series(
    data: List[RowDict],
    y_key: str,
    y_err_key: str,
    title: str,
    ylabel: str,
    filename: str,
    output_dir: Path,
    temperature_label: str = "",
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
    display_title = title
    if temperature_label:
        display_title = f"{title} at {temperature_label}"
    ax.set_title(display_title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()

    #if y_key == "A":
    #    ax.set_ylim(bottom=0)

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

    meastime_value = _read_meastime_from_list(DEFAULT_LIST_FILE)
    if math.isnan(meastime_value) or meastime_value <= 0:
        meastime_value = 30.0
    if abs(meastime_value - round(meastime_value)) < 1e-6:
        meastime_str = str(int(round(meastime_value)))
    else:
        meastime_str = f"{meastime_value:g}"
    unit_suffix = f"{meastime_str}sec"

    plots = [
        (
            "A",
            "A_error",
            f"Amplitude A(counts/{unit_suffix}) vs PCC(A)",
            f"Amplitude A (counts/{unit_suffix})",
            "Amplitude_vs_PCC.png",
        ),
        (
            "offset",
            "offset_error",
            f"mean B(counts/{unit_suffix}) vs PCC(A)",
            f"Mean B (counts/{unit_suffix})",
            "MeanB_vs_PCC.png",
        ),
        (
            "Nup_mean",
            "Nup_error",
            f"Nup(counts/{unit_suffix}) vs PCC(A)",
            f"Nup (counts/{unit_suffix})",
            "Nup_vs_PCC.png",
        ),
        (
            "Ndown_mean",
            "Ndown_error",
            f"Ndown(counts/{unit_suffix}) vs PCC(A)",
            f"Ndown (counts/{unit_suffix})",
            "Ndown_vs_PCC.png",
        ),
    ]

    temperature_label = _extract_temperature_label(data)
    if not temperature_label:
        temperature_label = _read_temperature_from_list(DEFAULT_LIST_FILE)
    temperature_label = _format_temperature_label(temperature_label)

    for y_key, y_err, title, ylabel, filename in plots:
        plot_series(data, y_key, y_err, title, ylabel, filename, output_dir, temperature_label)


if __name__ == "__main__":
    main()
