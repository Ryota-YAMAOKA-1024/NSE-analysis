# NSE Analysis Toolkit (2 K Set)

このディレクトリには CrNb₄Se₈ の NSE 実験データを解析・可視化するためのスクリプト一式がまとまっています。ファイル名に「2K」と付いていますが、`code/list.txt` に記載する温度や測定条件を切り替えることで別温度のデータにも利用できます。

- `code/` — 解析スクリプトと設定ファイル (`list.txt`)
- `fit/` — フィット結果 (`plot_NSE_spin_echo-fit.py` が自動生成)
- `figure/` — 各種プロットの出力先

典型的な配置は以下のようになります。60 K など別温度の解析フォルダと並べると管理しやすく、スクリプト側でも相対パスを前提にしています。

```
NSE/workspace/
├─ 2K/
│  ├─ code/
│  │  ├─ list.txt
│  │  ├─ plot_NSE_spin_echo-fit.py
│  │  ├─ plot_NSE_PCC.py
│  │  └─ plot_NSE_contrast.py
│  ├─ fit/
│  └─ figure/
└─ 60K/
   └─ ...
```

## `code/list.txt` の書き方

```
DIR=/absolute/path/to/scan/files/
TEMP=2.6K
meastime=360
scan_110226.txt
scan_110238.txt
...
```

- `DIR` : 解析対象 txt ファイルが置いてあるディレクトリ（相対パスでも絶対パスでも可）
- `TEMP` : 温度表示用のラベル。`2.6K` のように記載するとグラフでは `2.6 K` に整形されます
- `meastime` : 1 点あたりの計測時間（秒）。グラフの縦軸単位 `counts/xxxsec` に使用されます
- 以降の行 : 読み込みたいファイル名（コメント行は `#` で始める）

## スクリプト一覧

- `plot_NSE_spin_echo-fit.py`
  - 指定ディレクトリの txt を読み込み、スピンエコー信号を正弦関数でフィット
  - `figure/` に各測定点のエラーバー付き波形とフィット結果を保存
  - `fit/fit_results.csv` に振幅・位相・オフセット・Nup/Ndown 等を記録
  - 実行例: `python3 code/plot_NSE_spin_echo-fit.py --mode multiple`
    - `--mode single --file scan_XXXXX.txt` で単一ファイル処理
    - `--list-file path/to/list.txt` で別設定を読み込むことも可能

- `plot_NSE_PCC.py`
  - `fit_results.csv` をまとめ、PCC に対する `A`, `offset`, `Nup`, `Ndown` の 4 枚のグラフを生成
  - 縦軸単位は `list.txt` の `meastime` に連動
  - 実行例: `python3 code/plot_NSE_PCC.py`

- `plot_NSE_contrast.py`
  - 2 つの温度（既定では 60 K と 2 K）のフィット結果を読み込み、`C = B/A` の比 `C(60 K) / C(2 K)` を PCC 依存性としてプロット
  - 誤差は伝播則で評価し、エラーバーとして表示
  - 引数で参照フォルダや出力名を切り替え可能

## 解析フローの例

1. `code/list.txt` で対象ディレクトリ、温度、計測時間、ファイル名を指定
2. `python3 code/plot_NSE_spin_echo-fit.py` を実行し、`fit_results.csv` と波形プロットを生成
3. 必要に応じて `python3 code/plot_NSE_PCC.py` や `python3 code/plot_NSE_contrast.py` で集計グラフを作成

## 依存パッケージ

- Python 3.x
- NumPy, SciPy, Matplotlib
  - 未導入の場合は `pip install numpy scipy matplotlib` などで準備してください

Matplotlib の設定ディレクトリに書き込めない環境では、環境変数 `MPLCONFIGDIR` を一時的に作業ディレクトリ等へ向けると警告を避けられます。
