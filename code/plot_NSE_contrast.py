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

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_FIT_PATH = BASE_DIR.parent / "fit" / "fit_results.csv"
DEFAULT_REFERENCE_FIT_PATH = BASE_DIR.parent.parent / "2K" / "fit" / "fit_results.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR.parent / "figure"
DEFAULT_LIST_FILE = BASE_DIR / "list.txt"
DEFAULT_REFERENCE_LIST_FILE = BASE_DIR.parent.parent / "2K" / "code" / "list.txt"

NOTE_PATTERN = re.compile(r"PCC\s*([-+]?\d*\.?\d+)", re.IGNORECASE)


def _safe_float(value: str) -> float:
    """Convert value to float, returning NaN if conversion fails."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _parse_pcc_value(note: str) -> float:
    """Extract PCC current value from note_info text."""
    if not note:
        return math.nan

    match = NOTE_PATTERN.search(note)
    if match:
        return _safe_float(match.group(1))

    filtered = "".join(ch for ch in note if ch.isdigit() or ch in {".", "-", "+"})
    return _safe_float(filtered)


RowDict = Dict[str, float]


def load_fit_results(csv_path: Path) -> List[RowDict]:
    """Load fit results and compute contrast plus propagated error."""
    with csv_path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)

    results: List[RowDict] = []
    for row in rows:
        pcc_value = _parse_pcc_value(row.get("note_info", ""))
        amp = _safe_float(row.get("A", ""))
        amp_err = _safe_float(row.get("A_error", ""))
        offset = _safe_float(row.get("offset", ""))
        offset_err = _safe_float(row.get("offset_error", ""))

        if not math.isnan(amp):
            amp = abs(amp)

        if math.isnan(pcc_value) or math.isnan(amp) or math.isnan(offset):
            continue
        if amp == 0.0:
            continue

        contrast = offset / amp
        contrast_err = math.nan
        if not math.isnan(offset_err) and not math.isnan(amp_err):
            term1 = offset_err / amp
            term2 = (offset * amp_err) / (amp * amp)
            contrast_err = math.sqrt(term1 * term1 + term2 * term2)
        elif not math.isnan(offset_err):
            contrast_err = abs(offset_err / amp)
        elif not math.isnan(amp_err):
            contrast_err = abs((offset * amp_err) / (amp * amp))

        results.append(
            {
                "pcc_value": pcc_value,
                "contrast": contrast,
                "contrast_error": contrast_err,
            }
        )

    return results


def _read_temperature_from_list(list_path: Path) -> str:
    """Retrieve TEMP= entry from list.txt."""
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
    """Format temperature label into human readable form."""
    if not label:
        return ""
    raw = label.strip()
    if not raw:
        return ""
    raw = raw.replace("℃", "C")
    if raw[-1].lower() == "k":
        value = raw[:-1].strip()
        if value:
            return f"{value} K"
        return "K"
    return raw


def _prepare_series(
    data: Iterable[RowDict],
    value_key: str,
    error_key: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert loaded data into numpy arrays sorted by PCC."""
    xs: List[float] = []
    ys: List[float] = []
    yerrs: List[float] = []

    for row in data:
        x_value = row.get("pcc_value", math.nan)
        y_value = row.get(value_key, math.nan)
        err_value = row.get(error_key, math.nan)

        if math.isnan(x_value) or math.isnan(y_value):
            continue

        xs.append(x_value)
        ys.append(y_value)
        yerrs.append(0.0 if math.isnan(err_value) else err_value)

    if not xs:
        return np.array([]), np.array([]), np.array([])

    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    yerrs = np.array(yerrs, dtype=float)

    order = np.argsort(xs)
    return xs[order], ys[order], yerrs[order]


def compute_contrast_ratio(
    primary: Iterable[RowDict],
    reference: Iterable[RowDict],
    decimals: int = 6,
) -> List[RowDict]:
    """Compute C_primary / C_reference and propagate errors."""
    ref_lookup: Dict[float, RowDict] = {}
    for row in reference:
        key = round(row.get("pcc_value", math.nan), decimals)
        if math.isnan(key):
            continue
        ref_lookup[key] = row

    ratio_results: List[RowDict] = []
    skipped = 0
    for row in primary:
        pcc = row.get("pcc_value", math.nan)
        if math.isnan(pcc):
            continue
        key = round(pcc, decimals)
        ref_row = ref_lookup.get(key)
        if not ref_row:
            skipped += 1
            continue

        c_primary = row.get("contrast", math.nan)
        c_reference = ref_row.get("contrast", math.nan)
        if math.isnan(c_primary) or math.isnan(c_reference) or c_reference == 0.0:
            skipped += 1
            continue

        ratio = c_primary / c_reference

        err_primary = row.get("contrast_error", math.nan)
        err_reference = ref_row.get("contrast_error", math.nan)

        if math.isnan(err_primary) and math.isnan(err_reference):
            ratio_err = 0.0
        else:
            term_primary = (err_primary / c_primary) if not math.isnan(err_primary) and c_primary != 0 else 0.0
            term_reference = (err_reference / c_reference) if not math.isnan(err_reference) and c_reference != 0 else 0.0
            ratio_err = abs(ratio) * math.sqrt(term_primary ** 2 + term_reference ** 2)

        ratio_results.append(
            {
                "pcc_value": pcc,
                "ratio": ratio,
                "ratio_error": ratio_err,
            }
        )

    if skipped:
        print(f"Warning: skipped {skipped} PCC point(s) missing in reference or invalid for ratio.")

    return ratio_results


def plot_contrast(
    data: List[RowDict],
    output_dir: Path,
    temperature_label: str = "",
    filename: str = "Contrast_vs_PCC.png",
) -> None:
    """Plot contrast against PCC with error bars."""
    xs, ys, yerrs = _prepare_series(data, "contrast", "contrast_error")
    if xs.size == 0:
        print("No valid data points to plot contrast.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(
        xs,
        ys,
        yerr=yerrs,
        fmt="o",
        capsize=4,
        linewidth=1.2,
        color="tab:purple",
        label="Contrast C = B/A",
    )
    ax.set_xlabel("PCC (A)")
    ax.set_ylabel("Contrast C = B/A")
    title = "Contrast vs PCC"
    if temperature_label:
        title = f"{title} at {temperature_label}"
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_contrast_ratio(
    data: List[RowDict],
    output_dir: Path,
    primary_label: str,
    reference_label: str,
    filename: str = "ContrastRatio_vs_PCC.png",
) -> None:
    """Plot contrast ratio C_primary / C_reference as a function of PCC."""
    xs, ys, yerrs = _prepare_series(data, "ratio", "ratio_error")
    if xs.size == 0:
        print("No valid data points to plot contrast ratio.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(
        xs,
        ys,
        yerr=yerrs,
        fmt="o",
        capsize=4,
        linewidth=1.2,
        color="tab:green",
        label=f"C({primary_label}) / C({reference_label})",
    )
    ax.set_xlabel("PCC (A)")
    ax.set_ylabel(f"C({primary_label}) / C({reference_label})")
    ax.set_title(f"Contrast Ratio vs PCC")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot contrast (B/A) as a function of PCC current.")
    parser.add_argument(
        "--input",
        default=str(DEFAULT_FIT_PATH),
        help="Path to fit_results.csv (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save figure (default: %(default)s)",
    )
    parser.add_argument(
        "--list-file",
        default=str(DEFAULT_LIST_FILE),
        help="Path to list.txt to read temperature info (default: %(default)s)",
    )
    parser.add_argument(
        "--reference",
        default=str(DEFAULT_REFERENCE_FIT_PATH),
        help="Path to reference fit_results.csv for contrast ratio (default: %(default)s)",
    )
    parser.add_argument(
        "--reference-list",
        default=str(DEFAULT_REFERENCE_LIST_FILE),
        help="Path to reference list.txt to read temperature info (default: %(default)s)",
    )
    parser.add_argument(
        "--output-name",
        default="ContrastRatio_vs_PCC.png",
        help="Filename for the generated figure (default: %(default)s)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    list_path = Path(args.list_file).expanduser().resolve()
    output_name = args.output_name

    reference_path = Path(args.reference).expanduser().resolve()
    reference_list_path = Path(args.reference_list).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference CSV not found: {reference_path}")

    primary_data = load_fit_results(input_path)
    reference_data = load_fit_results(reference_path)

    primary_label = _format_temperature_label(_read_temperature_from_list(list_path)) or "60 K"
    reference_label = _format_temperature_label(_read_temperature_from_list(reference_list_path)) or "2 K"

    ratio_data = compute_contrast_ratio(primary_data, reference_data)
    plot_contrast_ratio(ratio_data, output_dir, primary_label, reference_label, filename=output_name)


if __name__ == "__main__":
    main()
