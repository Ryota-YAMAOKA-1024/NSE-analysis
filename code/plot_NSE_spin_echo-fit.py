import os  # ファイルパス操作のためのライブラリ
import math  # 数値計算のためのライブラリ
import matplotlib.pyplot as plt  # グラフ描画のためのライブラリ
import numpy as np  # 数値計算のためのライブラリ
from scipy.optimize import curve_fit  # 曲線フィッティングのためのライブラリ
import csv  # CSVファイル操作のためのライブラリ
import argparse  # コマンドライン引数解析のためのライブラリ

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # スクリプトのディレクトリ
FIGURE_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "figure"))  # 画像出力先
FIT_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "fit"))  # フィット結果出力先
DEFAULT_LIST_FILE = os.path.join(BASE_DIR, "list.txt")  # デフォルトのファイルリスト
COUNTS_UNIT_SUFFIX = "30sec"

def fit_func(x, A, B, C, D):
    """フィッティング用の正弦関数を定義"""
    return A*np.sin(B*x + C)+D  # A: 振幅, B: 周波数, C: 位相, D: オフセット

def extract_note_info(file_path):
    """txtファイルの2行目からnote:以降の最初のカンマまでの情報を抽出する関数"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:  # ファイルを読み込みモードで開く
            lines = f.readlines()  # ファイルの全行をリストとして読み込む

        # 2行目を取得（インデックス1）
        line2 = lines[1].strip()  # 2行目の文字列を取得し、前後の空白を除去

        # "note:"以降の部分を抽出
        if "note:" in line2:  # 2行目に"note:"が含まれているかチェック
            note_part = line2.split("note:")[1].strip()  # "note:"以降の部分を抽出
            # 最初のカンマまでの部分を取得
            first_item = note_part.split(',')[0].strip()  # 最初のカンマで分割し、最初の要素を取得
            return first_item  # PCC1Aなどの測定条件情報を返す
        else:
            return "Note information not found"  # note情報が見つからない場合のメッセージ
    except Exception as e:
        return f"Error reading file: {e}"  # ファイル読み込みエラーの場合のメッセージ

def _safe_float(value):
    """文字列を浮動小数点数に変換する（失敗時はnp.nanを返す）"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan

def load_nse_data(file_path):
    """pandasを使わずにNSEデータを読み込む"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    data_lines = lines[4:]  # 最初の4行（ヘッダー情報など）をスキップ
    reader = csv.reader(data_lines, delimiter='\t')

    try:
        headers = next(reader)
    except StopIteration as exc:
        raise ValueError("データヘッダーが見つかりません") from exc

    rows = []
    for row in reader:
        if not row:
            continue
        if row[0].strip().startswith("#"):
            continue
        if all(cell.strip() == "" for cell in row):
            continue
        if len(row) != len(headers):
            # 列数が一致しない行は無視（コメント行など）
            continue
        rows.append(dict(zip(headers, row)))

    return headers, rows


def _format_temperature_label(label):
    """温度の表記を '2.6 K' の形式に整える"""
    if not label:
        return ""
    raw = str(label).strip()
    if not raw:
        return ""
    raw = raw.replace("℃", "C")
    if raw[-1].lower() == "k":
        value = raw[:-1].strip()
        if value:
            return f"{value} K"
        return "K"
    return raw

def load_filename_list(list_path):
    """list.txt からデータディレクトリ・温度・処理対象のファイル名を読み込む"""
    resolved_path = os.path.expanduser(list_path)
    if not os.path.isabs(resolved_path):
        resolved_path = os.path.join(BASE_DIR, resolved_path)

    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"リストファイルが存在しません: {resolved_path}")

    data_dir = None
    temperature = None
    filenames = []
    with open(resolved_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = line.strip()
            if not entry or entry.startswith("#"):
                continue
            if entry.lower().startswith("dir="):
                candidate = entry.split("=", 1)[1].strip()
                if candidate:
                    candidate = os.path.expanduser(candidate)
                    if not os.path.isabs(candidate):
                        candidate = os.path.join(BASE_DIR, candidate)
                    data_dir = candidate
                continue
            if entry.lower().startswith("temp="):
                temp_value = entry.split("=", 1)[1].strip()
                if temp_value:
                    temperature = temp_value
                continue
            if data_dir is None and entry.endswith(("/", "\\")):
                candidate = os.path.expanduser(entry)
                if not os.path.isabs(candidate):
                    candidate = os.path.join(BASE_DIR, candidate)
                data_dir = candidate
                continue
            filenames.append(entry)

    if data_dir is None:
        raise ValueError(
            f"データディレクトリが指定されていません: {resolved_path} (行頭に 'DIR=' で指定してください)"
        )
    if temperature is None:
        raise ValueError(
            f"温度情報が指定されていません: {resolved_path} (行頭に 'TEMP=' で指定してください)"
        )
    if not filenames:
        raise ValueError(f"リストファイルが空です: {resolved_path}")

    temperature = _format_temperature_label(temperature)

    return data_dir, temperature, filenames


def _read_meastime_from_list(list_path):
    """list.txt から meastime= を読み取り、floatとして返す。見つからない場合は math.nan。"""
    resolved_path = os.path.expanduser(list_path)
    if not os.path.isabs(resolved_path):
        resolved_path = os.path.join(BASE_DIR, resolved_path)
    if not os.path.exists(resolved_path):
        return math.nan
    try:
        with open(resolved_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = line.strip()
                if not entry or entry.startswith("#"):
                    continue
                if entry.lower().startswith("meastime"):
                    value = entry.split("=", 1)[1].strip()
                    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                        value = value[1:-1].strip()
                    return _safe_float(value)
    except OSError:
        return math.nan
    return math.nan

def _extract_role_arrays(rows, role_name):
    """role列でフィルタリングし、currentとcountsの配列を返す"""
    filtered = [row for row in rows if row.get("role") == role_name]
    currents = np.array([_safe_float(row.get("current")) for row in filtered], dtype=float)
    counts = np.array([_safe_float(row.get("counts")) for row in filtered], dtype=float)
    mask = (~np.isnan(currents)) & (~np.isnan(counts))
    return currents[mask], counts[mask]

def process_single_file(file_path, save_plot=True, temperature=None):
    """単一ファイルを処理する関数"""
    print(f"Processing {os.path.basename(file_path)}...")  # 処理中のファイル名を表示

    # note情報を抽出
    note_info = extract_note_info(file_path)  # PCC1Aなどの測定条件情報を取得
    print(f"Note information: {note_info}")  # 取得したnote情報を表示

    try:
        headers, rows = load_nse_data(file_path)  # データを読み込み
        print(f"Data shape: ({len(rows)}, {len(headers)})")  # データの形状（行数×列数）を表示

        required_columns = {"role", "current", "counts"}
        if not required_columns.issubset(headers):
            missing = required_columns.difference(headers)
            raise ValueError(f"必要な列が見つかりません: {', '.join(missing)}")

        echo_current, echo_counts = _extract_role_arrays(rows, "echo")
        down_current, down_counts = _extract_role_arrays(rows, "Ndown")
        up_current, up_counts = _extract_role_arrays(rows, "Nup")

        if echo_current.size == 0 or echo_counts.size == 0:
            raise ValueError("echoデータが不足しています")

        down_data_mean = float(np.mean(down_counts)) if down_counts.size else np.nan
        down_data_error = float(np.sqrt(down_data_mean)) if down_counts.size else np.nan

        up_data_mean = float(np.mean(up_counts)) if up_counts.size else np.nan
        up_data_error = float(np.sqrt(up_data_mean)) if up_counts.size else np.nan

        echo_errors = np.sqrt(echo_counts)
        down_errors = np.sqrt(down_counts)
        up_errors = np.sqrt(up_counts)

        # フィッティング実行
        popt, pcov = curve_fit(fit_func, echo_current, echo_counts, p0=[70, 0.5, 0, 700], maxfev=10000)  # 正弦関数でフィッティング実行
        errors = np.sqrt(np.diag(pcov))  # フィッティングパラメータの標準誤差を計算

        # 結果を辞書に格納
        result = {
            'filename': os.path.basename(file_path),  # ファイル名を格納
            'note_info': note_info,  # 測定条件情報を格納
            'temperature': temperature,  # 測定温度を記録
            'A': popt[0],  # フィッティング結果の振幅Aを格納
            'A_error': errors[0],  # 振幅Aの標準誤差を格納
            'f': popt[1],  # フィッティング結果の周波数fを格納
            'f_error': errors[1],  # 周波数fの標準誤差を格納
            'phi': popt[2],  # フィッティング結果の位相phiを格納
            'phi_error': errors[2],  # 位相phiの標準誤差を格納
            'offset': popt[3],  # フィッティング結果のオフセットを格納
            'offset_error': errors[3],  # オフセットの標準誤差を格納
            'Nup_mean': up_data_mean,  # Nupデータの平均値を格納
            'Nup_error': up_data_error,  # Nupデータのエラーを格納
            'Ndown_mean': down_data_mean,  # Ndownデータの平均値を格納
            'Ndown_error': down_data_error  # Ndownデータのエラーを格納
        }

        # プロット作成（オプション）
        if save_plot:  # プロット保存が有効な場合
            if not os.path.exists(FIGURE_DIR):  # figureフォルダが存在しない場合
                os.makedirs(FIGURE_DIR, exist_ok=True)  # figureフォルダを作成

            fig, ax = plt.subplots()  # 新しい図と軸を作成
            ax.errorbar(echo_current, echo_counts, yerr=echo_errors, fmt="o", linewidth=1, label="experimental data")  # 測定データをエラーバー付きでプロット
            if up_current.size:
                ax.errorbar(up_current, up_counts, yerr=up_errors, fmt="o", linewidth=1, label="__nolegend__", color="red")  # upデータを赤色でエラーバー付きプロット
            if down_current.size:
                ax.errorbar(down_current, down_counts, yerr=down_errors, fmt="o", linewidth=1, label="__nolegend__", color="blue")  # downデータを青色でエラーバー付きプロット
            fit_x_min = float(np.min(echo_current))
            fit_x_max = float(np.max(echo_current))
            if np.isclose(fit_x_min, fit_x_max):
                fit_x = np.array([fit_x_min])
            else:
                fit_x = np.linspace(fit_x_min, fit_x_max, 400)  # 高密度にサンプリングして滑らかな曲線にする
            ax.plot(fit_x, fit_func(fit_x, *popt), label="Fit")  # フィット結果を滑らかな正弦曲線として描画
            ax.set_xlabel(" current (symcoil2) (A)")  # x軸ラベルを設定
            ax.set_ylabel(f"Intensity (counts/{COUNTS_UNIT_SUFFIX})")  # y軸ラベルを設定
            if temperature:
                ax.set_title(f"{note_info} at {temperature}")  # 温度をタイトルに追加
            else:
                ax.set_title(f"{note_info}")  # グラフタイトルを測定条件に設定
            ax.grid(True)  # グリッドを表示
            ax.legend()
            y_datasets = [echo_counts]
            if up_counts.size:
                y_datasets.append(up_counts)
            if down_counts.size:
                y_datasets.append(down_counts)
            combined_counts = np.concatenate(y_datasets)
            y_min, y_max = float(np.min(combined_counts)), float(np.max(combined_counts))
            span = y_max - y_min
            padding = max(span * 0.1, 5.0)
            lower = max(0.0, y_min - padding)
            upper = y_max + padding
            if np.isclose(lower, upper):
                upper = lower + 1.0
            #ax.set_ylim(lower, upper)  # データに応じてy軸範囲を調整
            ax.set_ylim(150, 1050)  # データに応じてy軸範囲を調整
            plt.savefig(os.path.join(FIGURE_DIR, f"{note_info}.png"))  # figureフォルダ内にグラフをPNGファイルとして保存
            plt.close()  # メモリ節約のため図を閉じる

        print(f"A = {popt[0]:.4f} ± {errors[0]:.4f}")  # 振幅Aの値を小数点以下4桁で表示
        print(f"f = {popt[1]:.4f} ± {errors[1]:.4f}")  # 周波数fの値を小数点以下4桁で表示
        print(f"phi = {popt[2]:.4f} ± {errors[2]:.4f}")  # 位相phiの値を小数点以下4桁で表示
        print(f"offset = {popt[3]:.4f} ± {errors[3]:.4f}")  # オフセットの値を小数点以下4桁で表示
        print(f"Nup_mean. = {up_data_mean:.4f} ± {up_data_error:.4f}")  # Nup平均値を小数点以下4桁で表示
        print(f"Ndown_mean. = {down_data_mean:.4f} ± {down_data_error:.4f}")  # Ndown平均値を小数点以下4桁で表示
        print("-" * 50)  # 区切り線を表示

        return result  # 処理結果の辞書を返す

    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {e}")  # エラーが発生した場合のメッセージを表示
        return None  # エラー時はNoneを返す

def process_multiple_files(dir_path, filenames, save_plots=True, temperature=None):
    """複数ファイルを処理する関数"""
    results = []  # 結果を格納するリストを初期化

    for filename in filenames:  # ファイル名リストの各ファイルに対してループ
        file_path = os.path.join(dir_path, filename)  # ディレクトリパスとファイル名を結合

        # ファイルが存在するかチェック
        if not os.path.exists(file_path):  # ファイルが存在しない場合
            print(f"File not found: {filename}")  # ファイルが見つからないメッセージを表示
            continue  # 次のファイルにスキップ

        result = process_single_file(file_path, save_plot=save_plots, temperature=temperature)  # 単一ファイル処理関数を呼び出し
        if result:  # 処理が成功した場合
            results.append(result)  # 結果をリストに追加

    # CSV出力
    if results:  # 結果が存在する場合
        os.makedirs(FIT_DIR, exist_ok=True)  # ディレクトリがなければ作成
        csv_filename = os.path.join(FIT_DIR, "fit_results.csv")  # CSVファイル名を設定
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"Results saved to {csv_filename}")  # 保存完了メッセージを表示
        print(f"Processed {len(results)} files successfully")  # 処理完了ファイル数を表示
    else:
        print("No results to save")  # 結果がない場合のメッセージ

    return results  # 処理結果のリストを返す


if __name__ == "__main__":
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description='NSEデータのフィッティングとプロット')  # 引数解析器を作成
    parser.add_argument('--mode', choices=['single', 'multiple'], default='multiple',  # 処理モードを選択（デフォルトは複数ファイル）
                        help='処理モード: single (単一ファイル) または multiple (複数ファイル)')
    parser.add_argument('--file', type=str, default='scan_106919.txt',  # 単一ファイル処理時のファイル名を指定
                        help='単一ファイル処理時のファイル名')
    parser.add_argument('--no-plot', action='store_true',  # プロット保存を無効にするフラグ
                        help='プロットを保存しない')
    parser.add_argument('--list-file', type=str, default=DEFAULT_LIST_FILE,
                        help=f'データディレクトリとファイル名を列挙したファイル (デフォルト: {DEFAULT_LIST_FILE})')

    args = parser.parse_args()  # コマンドライン引数を解析
    save_plots = not args.no_plot  # プロット保存フラグを設定

    try:
        dir_path, temperature, filenames_from_list = load_filename_list(args.list_file)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ファイルリストの読み込みに失敗しました: {exc}")
        exit(1)

    meastime_value = _read_meastime_from_list(args.list_file)
    if math.isnan(meastime_value) or meastime_value <= 0:
        meastime_value = 30.0
    if abs(meastime_value - round(meastime_value)) < 1e-6:
        COUNTS_UNIT_SUFFIX = f"{int(round(meastime_value))}sec"
    else:
        COUNTS_UNIT_SUFFIX = f"{meastime_value:g}sec"

    if args.mode == 'single':  # 単一ファイル処理モードの場合
        filename = args.file  # ファイル名を取得
        file_path = os.path.join(dir_path, filename)  # 完全なファイルパスを作成

        if not os.path.exists(file_path):  # ファイルが存在しない場合
            print(f"File not found: {file_path}")  # エラーメッセージを表示
            exit(1)  # プログラムを終了

        result = process_single_file(file_path, save_plot=save_plots, temperature=temperature)  # 単一ファイル処理を実行
        if result:  # 処理が成功した場合
            print("Single file processing completed successfully!")  # 成功メッセージを表示
        else:
            print("Single file processing failed!")  # 失敗メッセージを表示

    else:  # 複数ファイル処理モードの場合
        results = process_multiple_files(dir_path, filenames_from_list, save_plots=save_plots, temperature=temperature)  # 複数ファイル処理を実行
        print(f"Multiple files processing completed! Processed {len(results)} files.")  # 処理完了メッセージを表示
