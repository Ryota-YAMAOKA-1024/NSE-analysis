import os  # ファイルパス操作のためのライブラリ
import matplotlib.pyplot as plt  # グラフ描画のためのライブラリ
import numpy as np  # 数値計算のためのライブラリ
from scipy.optimize import curve_fit  # 曲線フィッティングのためのライブラリ
import csv  # CSVファイル操作のためのライブラリ
import argparse  # コマンドライン引数解析のためのライブラリ

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # スクリプトのディレクトリ
FIGURE_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "figure"))  # 画像出力先
FIT_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "fit"))  # フィット結果出力先

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

def _extract_role_arrays(rows, role_name):
    """role列でフィルタリングし、currentとcountsの配列を返す"""
    filtered = [row for row in rows if row.get("role") == role_name]
    currents = np.array([_safe_float(row.get("current")) for row in filtered], dtype=float)
    counts = np.array([_safe_float(row.get("counts")) for row in filtered], dtype=float)
    mask = (~np.isnan(currents)) & (~np.isnan(counts))
    return currents[mask], counts[mask]

def process_single_file(file_path, save_plot=True):
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
        popt, pcov = curve_fit(fit_func, echo_current, echo_counts, p0=[76,45, 0, 161])  # 正弦関数でフィッティング実行
        errors = np.sqrt(np.diag(pcov))  # フィッティングパラメータの標準誤差を計算

        # 結果を辞書に格納
        result = {
            'filename': os.path.basename(file_path),  # ファイル名を格納
            'note_info': note_info,  # 測定条件情報を格納
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
            ax.errorbar(echo_current, echo_counts, yerr=echo_errors, fmt="o", linewidth=1, label="echo")  # echoデータをエラーバー付きでプロット
            if up_current.size:
                ax.errorbar(up_current, up_counts, yerr=up_errors, fmt="o", linewidth=1, label="up", color="red")  # upデータを赤色でエラーバー付きプロット
            if down_current.size:
                ax.errorbar(down_current, down_counts, yerr=down_errors, fmt="o", linewidth=1, label="down", color="blue")  # downデータを青色でエラーバー付きプロット
            ax.plot(echo_current, fit_func(echo_current, *popt), label="Fit")  # フィッティング曲線をプロット
            ax.set_xlabel(" current (symcoil2) (A)")  # x軸ラベルを設定
            ax.set_ylabel(" Intensity")  # y軸ラベルを設定
            ax.set_title(f"{note_info}")  # グラフタイトルを測定条件に設定
            ax.grid(True)  # グリッドを表示
            ax.set_ylim(30, 300)  # y軸の範囲を30-300に設定
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

def process_multiple_files(dir_path, filenames, save_plots=True):
    """複数ファイルを処理する関数"""
    results = []  # 結果を格納するリストを初期化

    for filename in filenames:  # ファイル名リストの各ファイルに対してループ
        file_path = os.path.join(dir_path, filename)  # ディレクトリパスとファイル名を結合

        # ファイルが存在するかチェック
        if not os.path.exists(file_path):  # ファイルが存在しない場合
            print(f"File not found: {filename}")  # ファイルが見つからないメッセージを表示
            continue  # 次のファイルにスキップ

        result = process_single_file(file_path, save_plot=save_plots)  # 単一ファイル処理関数を呼び出し
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
    parser.add_argument('--dir', type=str,  # データファイルのディレクトリパスを指定
                       default="/Users/yamaokaryota/Library/CloudStorage/GoogleDrive-yamaoka-ryota-1024@g.ecc.u-tokyo.ac.jp/.shortcut-targets-by-id/1EKALfynkxJ3ivgOkvVlpgLabdSA2LAp-/Yamaoka_2025cycle6/",
                       help='データファイルのディレクトリパス')

    args = parser.parse_args()  # コマンドライン引数を解析

    dir_path = args.dir  # ディレクトリパスを取得
    save_plots = not args.no_plot  # プロット保存フラグを設定

    if args.mode == 'single':  # 単一ファイル処理モードの場合
        # 単一ファイル処理
        filename = args.file  # ファイル名を取得
        file_path = os.path.join(dir_path, filename)  # 完全なファイルパスを作成

        if not os.path.exists(file_path):  # ファイルが存在しない場合
            print(f"File not found: {file_path}")  # エラーメッセージを表示
            exit(1)  # プログラムを終了

        result = process_single_file(file_path, save_plot=save_plots)  # 単一ファイル処理を実行
        if result:  # 処理が成功した場合
            print("Single file processing completed successfully!")  # 成功メッセージを表示
        else:
            print("Single file processing failed!")  # 失敗メッセージを表示

    else:  # 複数ファイル処理モードの場合
        # 複数ファイル処理
        filenames = [  # 処理対象のファイル名リストを定義
            "scan_106919.txt",
            "scan_106985.txt",
            "scan_107051.txt",
            "scan_107117.txt",
            "scan_107183.txt",
            "scan_107249.txt",
            "scan_107315.txt",
            "scan_107381.txt",
            "scan_107447.txt",
            "scan_107513.txt",
            "scan_107579.txt",
            "scan_107645.txt",
            "scan_107711.txt"
        ]

        results = process_multiple_files(dir_path, filenames, save_plots=save_plots)  # 複数ファイル処理を実行
        print(f"Multiple files processing completed! Processed {len(results)} files.")  # 処理完了メッセージを表示
