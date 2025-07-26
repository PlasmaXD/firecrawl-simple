import time, uuid, os, requests
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from bs4 import BeautifulSoup  # ← BeautifulSoup を使う（pip install beautifulsoup4）

FIRECRAWL = os.getenv("FIRECRAWL_URL", "http://localhost:3002")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "research")
SEED_URLS = [
    "https://engineering.mercari.com/blog/entry/20250612-d2c354901d/",  # ← 技術ブログ等を追加
    # "https://arxiv.org/list/cs.LG/recent"  # arXivはAPI利用も検討（後述）
]

# 1) モデル初期化（軽量構成）
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
# summarizer = pipeline("summarization", model="pszemraj/led-base-book-summary")
summarizer = pipeline(
    "summarization",
    model="Zolyer/ja-t5-base-summary",         # ← 別の日本語要約モデル
    tokenizer="Zolyer/ja-t5-base-summary"
)
# 2) Qdrant クライアント & コレクション用意（384次元 / コサイン類似度）
qclient = QdrantClient(url=QDRANT_URL)
if COLLECTION not in [c.name for c in qclient.get_collections().collections]:
    qclient.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=qm.VectorParams(size=384, distance=qm.Distance.COSINE),
    )

def firecrawl_scrape(url: str) -> Dict:
    r = requests.post(f"{FIRECRAWL}/v1/scrape", json={
        "url": url,
        "formats": ["markdown", "rawHtml"]  # ← リクエスト列挙はこの3種に限定
    })
    r.raise_for_status()
    return r.json()

def firecrawl_crawl(url: str, limit: int = 10) -> List[Dict]:
    job = requests.post(f"{FIRECRAWL}/v1/crawl", json={
        "url": url,
        "limit": limit,
        "scrapeOptions": {"formats": ["markdown", "rawHtml"]}
    }).json()
    job_id = job.get("id") or job.get("data", {}).get("id")
    assert job_id, f"no job id: {job}"
    # ポーリング
    while True:
        jr = requests.get(f"{FIRECRAWL}/v1/crawl/{job_id}").json()
        if jr.get("status") in ("completed", "failed"):
            return jr.get("data", [])
        time.sleep(2)

def summarize(text: str, max_chars: int = 3000) -> str:
    text = text[:max_chars]  # 長文は切り詰め
    out = summarizer(text, max_length=180, min_length=60, do_sample=False)[0]["summary_text"]
    return out

def upsert_to_qdrant(docs: List[Dict]):
    if not docs:
        return
    payloads, vectors, ids = [], [], []
    # for d in docs:
    #     content = d.get("markdown") or d.get("content") or ""
    #     if not content.strip():
    #         continue
    #     summary = summarize(content)
    #     vec = embedder.encode(summary)  # 要約をベクトル化（本文でもOK）
    #     payloads.append({
    #         "url": d.get("url") or d.get("sourceUrl"),
    #         "title": d.get("title"),
    #         "summary": summary,
    #         "markdown": content
    #     })
    #     vectors.append(vec.tolist())
    #     ids.append(str(uuid.uuid4()))
    for idx, d in enumerate(docs):
        # 1) 生データ取得
        content = d.get("markdown") or d.get("rawHtml") or ""
        if not content.strip():
            continue

        # 2) 要約
        summary = summarize(content)
        vec = embedder.encode(summary)

        # 3) URL とタイトルの補完
        url = d.get("url") or d.get("sourceUrl") or SEED_URLS[idx]  # idx 番目のシードを fallback
        title = d.get("title")
        if not title:
            # rawHtml から <title> を抜き出し
            soup = BeautifulSoup(d.get("rawHtml", ""), "html.parser")
            title = soup.title.string.strip() if soup.title else url

        # 4) payload 構築
        payloads.append({
            "url": url,
            "title": title,
            "summary": summary,
            "markdown": content
        })
        vectors.append(vec.tolist())
        ids.append(str(uuid.uuid4()))
    if vectors:
        qclient.upsert(
            collection_name=COLLECTION,
            points=qm.Batch(ids=ids, vectors=vectors, payloads=payloads)
        )

def run():
    all_docs = []
    for seed in SEED_URLS:
        # 1件だけなら scrape、複数たどるなら crawl を使い分け
        if seed.endswith("/recent") or "list/" in seed:
            docs = firecrawl_crawl(seed, limit=10)
        else:
            s = firecrawl_scrape(seed)
            docs = [s.get("data")] if s.get("data") else []
        all_docs.extend(docs)
    upsert_to_qdrant(all_docs)
    print(f"ingested: {len(all_docs)} docs")

if __name__ == "__main__":
    run()
