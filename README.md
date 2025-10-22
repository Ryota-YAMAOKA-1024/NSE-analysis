# NSE-analysis

実験データ（CrNb4Se8、NSE 測定）からフィット結果と可視化を行うための簡易スクリプト群です。以下の 2 つのツールを中心に利用します。

- `code/plot_NSE_spin_echo-fit.py`  
  指定した txt データを読み込み、スピンエコー信号をフィットして各種パラメータを求めます。  
  - `code/list.txt` に `DIR=`, `TEMP=`, そして処理したいファイル名を列挙してから実行してください。  
  - 実行例: `python3 code/plot_NSE_spin_echo-fit.py --mode multiple`
  - `fit/fit_results.csv` にフィット結果が保存され、`figure/` に各 PCC のエラーバー付きプロットが生成されます。

- `code/plot_NSE_PCC.py`  
  `fit_results.csv` を読み込んで振幅やオフセット、Nup/Ndown を PCC に対してまとめた 4 枚のグラフを作成します。  
  - 温度情報は `fit_results.csv` もしくは `code/list.txt` の `TEMP=` から取得し、タイトル末尾に `at 2.6 K` のように表示されます。  
  - 実行例: `python3 code/plot_NSE_PCC.py`

## 必要環境

- Python 3.x
- SciPy（`plot_NSE_spin_echo-fit.py` のフィットに使用）  
  未インストールの場合は `pip install scipy` で追加してください。
- Matplotlib、NumPy などの一般的な科学計算ライブラリ

実行時に Matplotlib のキャッシュが書き込めない警告が出る場合は、`MPLCONFIGDIR` 環境変数を一時的に書き込み可能なディレクトリに設定すると改善します。
