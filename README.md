# pci-to-vector-search: 顧客理解の19年（WhatからWhyへ）

> **「CDPの購買履歴（What）だけで顧客理解は十分か？」**  
> **名著『集合知プログラミング』（2007）の手書き協調フィルタリング（Whatの類似）と、Google Cloud BigQuery `VECTOR_SEARCH`（2026）による生活動機ベクトル検索（Whyの類似）を実データで比較・検証・可視化するリポジトリ**

---

## 📖 背景とコンセプト (Context: What vs Why)

Google Cloud Next Tokyo 26 の博報堂カスタマーセッション「CDP だけで顧客理解は不十分？Google Cloud と生活者データで実現する『Why 分析』最前線」では、現代のマーケティングにおける決定的な課題が提示されました。

> **「CDPにある購買履歴や行動ログは『過去に何を買ったか（What）』の結果にすぎず、『なぜ買ったのか（Why）』という顧客の生活文脈や深層心理までは見えない。」**

本リポジトリは、この「What vs Why」の問題意識を起点に、**2007年のオライリー名著『集合知プログラミング』（Toby Segaran 著）の協調フィルタリングアルゴリズム**と**2026年のGoogle Cloudスタック（BigQuery `VECTOR_SEARCH` + Vertex AI Multilingual Embeddings + Gemini）**を同一の10,000ユーザデータで左右比較・検証した記録です。

---

## 🏗 アーキテクチャ & 対比構造

10種類の潜在ユーザペルソナ（正解セグメント）を定義し、各ユーザから「購買・評価ログ（What）」と「プロフィール自然文（Why）」を派生させています。

```mermaid
flowchart TD
    subgraph DataGen [1. 合成ユーザデータ生成 (10,000件)]
        Seg[潜在ペルソナ 10種]
        Seg --> Truth[正解ラベル segment_id ※評価専用]
        Seg --> Ratings[What: 購買・評価ログ ratings]
        Seg --> Profile[Why: 生活文脈・購買動機 profile_text]
    end

    subgraph Storage [2. データストア & 埋め込み]
        Profile --> VertexAI[Vertex AI text-multilingual-embedding-002]
        VertexAI --> Embeddings[768次元 Dense Embedding]
        Ratings --> Memory[Python メモリ上 prefs辞書]
        Embeddings --> BQ[(BigQuery テーブル users_10k)]
        BQ --> VIndex[CREATE VECTOR INDEX IVF]
    end

    subgraph Engines [3. 探索エンジンの対比]
        Memory --> PCI[📜 2007年: What 類似<br/>sim_pearson / sim_cosine<br/>購買アイテムの一致で探索]
        BQ --> BQ_VS[⚡ 2026年: Why 類似<br/>BigQuery VECTOR_SEARCH<br/>生活価値観・動機で探索]
        PCI --> Hybrid[🔀 What × Why ハイブリッド探索]
        BQ_VS --> Hybrid
    end

    subgraph Explain [4. Why 分析 & 可視化]
        Hybrid --> Gemini[Gemini 1.5 Flash による Explainable Why 分析]
        Gemini --> UI[Side-by-Side 比較 Web UI]
    end
```

---

## 🎯 ベンチマーク実測結果 (10,000ユーザ)

Google Cloud プロジェクト `YOUR_PROJECT_ID` (BigQuery `asia-northeast1`) 上で実測したベンチマーク結果です。

### 1. 精度 (Precision@K) & レイテンシ比較

| 探索手法 | 意味づけ | Precision@5 (ペルソナ一致率) | Precision@10 | 平均レイテンシ |
| :--- | :--- | :---: | :---: | :---: |
| **1. PCI 2007 (Pearson 相関)** | **What 類似 (共通項限定)** | **41.3%** | 40.7% | 38.4 ms |
| **2. PCI 2007 (Cosine 共通項)** | **What 類似 (書籍準拠)** | **36.0%** | 20.0% | 21.7 ms |
| **3. PCI 2007 (Cosine ゼロ埋め)** | **What 類似 (疎行列補正)** | **100.0%** | 100.0% | 23.5 ms |
| **4. BQ 2026 (VECTOR_SEARCH ブルートフォース)** | **Why 類似 (厳密探索)** | **100.0%** | 100.0% | 1,114.7 ms |
| **5. BQ 2026 (VECTOR_SEARCH IVF インデックス)** | **Why 類似 (ANN インデックス)** | **100.0%** | 100.0% | 1,008.4 ms |

### 2. 手法間の一致度 (Top-10 Jaccard Overlap)

| 探索手法 | 1. PCI Pearson | 2. PCI Cos 共通項 | 3. PCI Cos ゼロ埋め | 4. BQ Exact | 5. BQ IVF Index |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. PCI Pearson (What)** | **100.0%** | 0.0% | 0.0% | **0.4%** | **0.4%** |
| **2. PCI Cos 共通項 (What)** | 0.0% | **100.0%** | 0.0% | 0.0% | 0.0% |
| **3. PCI Cos ゼロ埋め (What)** | 0.0% | 0.0% | **100.0%** | **0.4%** | **0.4%** |
| **4. BQ Exact (Why)** | 0.4% | 0.0% | 0.4% | **100.0%** | **55.7%** |
| **5. BQ IVF Index (Why)** | 0.4% | 0.0% | 0.4% | 55.7% | **100.0%** |

---

## 💡 決定的な技術的洞察 (Key Takeaways)

### ① 「Whatの類似」と「Whyの類似」の次元の断絶（重複率 0.4%）
2007年の「アイテム評価値（What）」に基づく手法と、2026年の「プロフィール文（Why）」に基づく手法では、**上位推薦ユーザの重複率はわずか 0.4%** でした。
どちらも正解ペルソナ（Precision 100%）を引き当てているにもかかわらず、推薦される個人は 99.6% 異なります。
* **What 類似**: 「同じ電気圧力鍋と冷凍容器を買っている人」を探す
* **Why 類似**: 「共働きで育児に追われ、時短とコスパを両立したい価値観を持つ人」を探す

この2つは見ている情報の次元が根本的に異なっており、CDPの購買ログ（What）だけに依存した推薦・セグメンテーションの限界を定量的に証明しています。

### ② スパース性における「共通項限定計算」の罠
『集合知プログラミング』で紹介されている素朴なピアソン相関やコサイン類似度は、「2人が共通して評価したアイテムのみ」を対象に計算します。
しかし実務のスパースデータでは、**「たまたま共通評価した1〜2個のアイテムでスコアが一致した（例: 5.0 と 5.0）」だけで類似度が 1.0 に張り付く現象**が発生し、全く無関係なペルソナのユーザが上位を占領してしまいます（Precision@5 が 36% に低下）。

### ③ BigQuery `CREATE VECTOR INDEX` の最小行数制限（5,000行）
BigQuery のベクトルインデックス（IVF / TreeAH）は、**テーブル行数が 5,000行未満の場合は作成できません**（`Total rows is smaller than min allowed 5000 for CREATE VECTOR INDEX` エラー）。
小規模なPoCでは自動的にブルートフォース（厳密探索）となり、数万件以上の本番規模でインデックスの高速化が発揮されます。

---

## 🖥 Side-by-Side 比較 Web アプリケーション & Why 分析

同一ユーザを選択すると、「What 類似」と「Why 類似」の結果が左右に並んで表示され、各ユーザに対して **Gemini が「なぜ似ているのか（共通する生活動機と施策示唆）」をリアルタイムに自動解説** します。

```bash
# Webアプリの起動
PYTHONPATH=. uvicorn src.web.app:app --host 127.0.0.1 --port 8000 --reload
```
ブラウザで `http://localhost:8000` にアクセスしてください。

---

## 📂 ディレクトリ構成

```
pci-to-vector-search/
├── README.md                  # 本ドキュメント
├── benchmark_report.md        # ベンチマーク実測レポート
├── requirements.txt           # 依存パッケージ定義
├── data/
│   ├── users_10k.jsonl        # 10,000件の合成ユーザデータ
│   └── users_1k.jsonl         # 1,000件のデバッグデータ
├── src/
│   ├── data/
│   │   ├── segments.yaml      # 10種のユーザペルソナ定義
│   │   └── generate_dataset.py # 合成データ生成スクリプト
│   ├── classic/
│   │   └── classic_recommender.py # 2007年 集合知プログラミング (What)
│   ├── bq/
│   │   ├── bq_setup.py        # Vertex AI Embeddings & BQロード・インデックス作成
│   │   ├── bq_search.py       # BigQuery VECTOR_SEARCH クライアント (Why)
│   │   ├── hybrid_search.py   # What x Why ハイブリッド探索エンジン
│   │   ├── why_analysis.py    # Gemini による Explainable Why 分析
│   │   └── vector_search.sql  # SQLクエリテンプレート
│   ├── benchmark/
│   │   └── run_benchmark.py   # 精度・レイテンシ・手法間一致度ベンチマーク
│   └── web/
│       ├── app.py             # FastAPI バックエンド
│       └── templates/
│           └── index.html     # Side-by-Side 比較 & Why分析画面
```

---

## 📜 ライセンス

MIT License
