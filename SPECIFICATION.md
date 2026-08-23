# 成果物 詳細仕様書 (System Specification)

## 1. システム概要と開発目的
* **プロジェクト名**: PCI to Vector Search: Two Eras Comparison (2007 vs 2026)
* **目的**: 2007年の名著『集合知プログラミング (Programming Collective Intelligence, PCI)』で提唱された**クラシカルな協調フィルタリング（Pearson相関・スパース評価行列）**と、2026年現代の**クラウドネイティブAIベクトル検索（Vertex AI Dense Embedding + BigQuery `VECTOR_SEARCH`）**を、同一データセット上でアルゴリズム精度・検索速度・推薦品質の観点から **Side-by-Side でリアルタイム比較・検証** する。
* **主要技術スタック**:
  * **Backend / Script**: Python 3.10+, FastAPI, Uvicorn, Pydantic, NumPy, PyYAML
  * **Cloud & Database**: Google Cloud (Vertex AI Text Embeddings, BigQuery VECTOR_SEARCH / Vector Index)
  * **Frontend**: HTML5, Vanilla CSS (Modern Dark Mode / Glassmorphism), Vanilla JavaScript, Jinja2

---

## 2. ディレクトリ構成と成果物モジュール一覧

```
pci-to-vector-search/
├── data/                                # 生成データ & ベクトルキャッシュ (JSONL)
│   ├── users_1k.jsonl                   # 1,000人規模 テストデータ
│   ├── users_10k.jsonl                  # 10,000人規模 本番データ
│   └── embedded_users_10k.jsonl         # 768次元ベクトル付きキャッシュデータ
├── src/
│   ├── data/
│   │   ├── segments.yaml                # 20個のユーザーペルソナ・セグメント定義マスター
│   │   └── generate_dataset.py          # リアルなユーザー・評価・自然言語プロファイル生成器
│   ├── classic/
│   │   └── classic_recommender.py       # 2007年 PCI アルゴリズム実装（Pearson / Cosine / 推薦）
│   ├── bq/
│   │   ├── bq_setup.py                  # Vertex AI 埋め込み生成 & BigQuery ロード & Vector Index 作成
│   │   ├── bq_search.py                 # BigQuery VECTOR_SEARCH Python クライアント
│   │   └── vector_search.sql            # ベクトル検索・クエリテンプレート
│   ├── benchmark/
│   │   └── run_benchmark.py             # 精度・速度・スケーラビリティ一括ベンチマーク実行ツール
│   └── web/
│       ├── app.py                       # FastAPI バックエンドサーバー
│       ├── templates/index.html         # リアルタイム比較 Web UI
│       └── static/                      # 静的アセット
├── benchmark_report.md                  # ベンチマーク測定結果レポート
├── SPECIFICATION.md                     # 本詳細仕様書
├── README.md                            # プロジェクト概要 & クイックスタート
└── requirements.txt                     # 依存パッケージ定義
```

---

## 3. データ仕様（Schema & Model）

### 3.1 ユーザーデータモデル
各ユーザーは「古典的レーティング情報」と「現代の自然言語プロファイル & ベクトル」を併せ持ちます。

| フィールド名 | 型 | 説明 |
| :--- | :--- | :--- |
| `user_id` | `STRING` (必須) | ユーザー識別子 (`U000001` 等) |
| `segment_id` | `STRING` (必須) | 所属クラスタ・正解ラベル (`SEG_TECH_AI`, `SEG_ANIME` 等 20種) |
| `segment_name` | `STRING` | セグメント日本語名 (例: "AI・最新テクノロジーマニア") |
| `age_band` | `STRING` | 年代帯 (`20s`, `30s`, `40s`, `50s+`) |
| `gender` | `STRING` | 性別 (`M`, `F`, `Other`) |
| `area` | `STRING` | 居住地域 (`Tokyo`, `Osaka`, `Fukuoka` 等) |
| `interests` | `ARRAY<STRING>` | 関心タグリスト (例: `["Python", "LLM", "GCP"]`) |
| `profile_text` | `STRING` (必須) | 自然言語自己紹介文（Embedding 生成ソース） |
| `ratings` | `ARRAY<RECORD>` | アイテム評価リスト (`item_id: STRING`, `score: FLOAT64 (1.0~5.0)`) |
| `embedding` | `ARRAY<FLOAT64>` | Vertex AI が生成した 768 次元の密ベクトル（Dense Vector） |

### 3.2 20セグメント（ペルソナクラスタ）定義
`src/data/segments.yaml` にて以下のような 20 種のクラスタを定義：
* テクノロジー系: `SEG_TECH_AI`, `SEG_WEB_DEV`, `SEG_SECURITY`, `SEG_GADGET`
* カルチャー・エンタメ系: `SEG_ANIME`, `SEG_GAMING`, `SEG_KPOP_IDOL`, `SEG_CINEMA`, `SEG_MUSIC_BAND`
* ライフスタイル・ホビー系: `SEG_OUTDOOR_CAMP`, `SEG_FITNESS_RUN`, `SEG_GOURMET_RAMEN`, `SEG_CAFE_SWEETS`, `SEG_SAUNA_ONSEN`, `SEG_TRAVEL_SOLO`, `SEG_DIY_CRAFT`
* ビジネス・専門職系: `SEG_STARTUP_BIZ`, `SEG_INVESTMENT`, `SEG_FASHION`, `SEG_READING_BOOK`

---

## 4. アルゴリズム・エンジン仕様

### 4.1 【2007年 PCI 方式】古典的協調フィルタリング (`src/classic/classic_recommender.py`)
1. **類似度計算方式**:
   * **Pearson 相関係数 (`sim_pearson`)**: 共通評価アイテムにおけるスコアの偏差積和から算出。ユーザーごとの評価傾向（甘い/辛い）のバイアスを補正。
     $$r = \frac{\sum (X - \bar{X})(Y - \bar{Y})}{\sqrt{\sum (X - \bar{X})^2 \sum (Y - \bar{Y})^2}}$$
   * **共通評価 Cosine 類似度 (`sim_cosine`)**: 共通して評価したアイテムのみの内積とノルムで算出。
   * **ゼロ埋め Cosine 類似度 (`sim_cosine_zero_filled`)**: 未評価を 0 とみなす疎ベクトル内積。
2. **アイテム推薦アルゴリズム (`get_recommendations`)**:
   * 対象ユーザーと他全ユーザーの類似度 $w_k$ を重みとし、未評価アイテム $i$ の予想スコアを加重平均で算出：
     $$\hat{r}_{u,i} = \frac{\sum_{k} w_{u,k} \cdot r_{k,i}}{\sum_{k} |w_{u,k}|}$$

### 4.2 【2026年 現代方式】BigQuery & Vertex AI ベクトル検索 (`src/bq/`)
1. **テキスト埋め込み (`src/bq/bq_setup.py`)**:
   * モデル: Vertex AI `text-multilingual-embedding-002` (768 次元)
   * ユーザーの `profile_text`（関心事・行動様式が凝縮された文章）を入力とし、意味空間へ射影。
2. **近似最近傍探索 (`src/bq/bq_search.py`, `src/bq/vector_search.sql`)**:
   * 距離尺度: `COSINE`
   * インデックス: `IVF` (Inverted File Index) によるミリ秒台 ANN 検索。
   * オプション: ブルートフォース完全探索 (`use_brute_force=true`) の切り替え対応。

---

## 5. Web アプリケーション仕様 (`src/web/`)

### 5.1 API エンドポイント (`app.py`)
* `GET /`: ダッシュボード Web UI レンダリング（サンプルユーザー一覧・メタデータを注入）
* `GET /api/user/{user_id}`: 指定ユーザーの詳細情報（プロファイル、所属セグメント、レーティング内訳）を取得
* `POST /api/compare`: 2007年 PCI と 2026年 BigQuery の**同時並行検索**を実行
  * **Request Body**:
    ```json
    {
      "user_id": "U000042",
      "top_k": 5,
      "classic_method": "pearson",
      "use_brute_force": false
    }
    ```
  * **Response Body**:
    * 双方の類似ユーザー Top-K リスト（ID、セグメント名、類似度スコア、共通評価アイテム数/内容）
    * 双方のセグメント一致率 (Precision)
    * 2007年側: 未評価アイテム推薦 Top-3 と予想スコア
    * レイテンシ（Classic: ms / BigQuery: ms）

### 5.2 UI/UX 仕様 (`templates/index.html`)
* **デザインシステム**:
  * ダークテーマ（深みのあるスレート/ネイビー背景）
  * グラスモーフィズムカード
  * レスポンシブ 2 カラム Side-by-Side 比較レイアウト
* **主要コンポーネント**:
  * プリセットユーザーのワンクリック選択バー & 任意 ID 検索入力
  * 手法切り替えセレクター（Pearson / 共通 Cosine / ゼロ埋め Cosine）
  * 検索元ユーザープロファイル表示カード
  * 左右比較結果ビュー（セグメント一致バッジ、レーティング共通度可視化）
  * PCI 協調フィルタリングによる未評価推薦アイテムバッジ表示

---

## 6. ベンチマーク・検証仕様 (`src/benchmark/run_benchmark.py`)

### 6.1 評価指標
1. **Precision@K (セグメント一致率)**: 検索された Top-K ユーザーのうち、クエリ元と同じ正解セグメントに属するユーザーの割合。
2. **Jaccard 類似度**: 2007 年手法と 2026 年手法が返した Top-K 集合の一致度。
3. **検索レイテンシ**: 1クエリあたりの実行速度（ミリ秒 / 秒）。
4. **コールドスタート耐性**: レーティング数が少ないユーザー（1〜2件）における検索品質の維持度。

---

## 7. 2手法の技術特性比較まとめ

| 比較軸 | 2007年: PCI 協調フィルタリング | 2026年: BigQuery VECTOR_SEARCH |
| :--- | :--- | :--- |
| **入力データ** | 明示的評価値（スパース評価行列） | 自然言語プロファイル（768次元 Dense ベクトル） |
| **類似性定義** | 行動の一致（同じアイテムに同じ点数を付けたか） | 意味・文脈の一致（興味・関心・目的が近いか） |
| **計算基盤** | アプリケーションメモリ (O(N) 全走査) | BigQuery クラウド分散 + IVF インデックス |
| **コールドスタート** | 評価が少ない新規ユーザー・アイテムに弱い | 1文のテキストがあれば即座に高精度検索可能 |
| **スケーラビリティ** | ユーザー数・アイテム数増加でメモリ・計算量が急増 | ペタバイト級データ・数億レコードでもスケール |
| **主な用途** | 行動履歴ベースのアイテム推薦 | コンテキスト検索・類似ユーザー/ペルソナ抽出 |
