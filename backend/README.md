# Illegal Wildlife Trade (IWT) Detection — Machine Learning Backend

**Project:** AI-Powered Multimodal Illegal Wildlife Trade Detection  
**Target:** SUSTAINOVA 2026 Ideathon (UN SDG 15 — Life on Land)  
**Contract Alignment:** Matches `src/types/index.ts` and `aiService.ts`  

---

## 1. System Overview

The IWT Detection Backend is a high-performance, privacy-preserving, and ethical multimodal inference system designed to detect online illicit wildlife listings across text captions and image submissions.

### Core Architectural Principles
- **No Direct "Illegal" Accusations:** Under international trade monitoring ethics, legal status cannot be adjudicated solely from online listings. System classifications are strictly:
  - `POTENTIAL ILLEGAL WILDLIFE TRADE`
  - `REQUIRES MANUAL REVIEW`
  - `NO CLEAR EVIDENCE DETECTED`
- **Mandatory Human-in-the-Loop:** Flags are routed to human enforcement queues (`FLAGGED_FOR_HUMAN_ENFORCEMENT`); never auto-reported.
- **Strict Privacy / Zero PII:** Does not store, crawl, or output seller names, account handles, telephone numbers, emails, or personal identifiers.
- **Multimodal Score Fusion:**
  $$\text{Fused Score} = 0.6 \times \text{Text Score} + 0.4 \times \text{Image Confidence}$$
  Reinforced by deterministic CITES Appendix I/II taxonomy and India Wildlife Protection Act (WPA) Schedule I/II codeword knowledge bases.

---

## 2. Directory Structure

```
backend/
├── api/
│   ├── routes.py            # FastAPI endpoints (/health, /analyze, /analyze/upload)
│   └── schemas.py           # Pydantic schemas contract-matched to frontend
├── config.py                # Pydantic BaseSettings & device selector (CUDA / CPU fallback)
├── data/
│   ├── loader.py            # WLT-OSN group-aware data partitioner (0% group leakage)
│   └── preprocessor.py      # Text normalization and tokenization pipeline
├── db/
│   ├── models.py            # SQLAlchemy models for Phase 3 case management
│   └── session.py           # Database engine & session maker
├── evaluation/
│   └── metrics.py           # Precision, Recall, F1, PR-AUC, Confusion Matrix, Failure modes
├── models/
│   ├── fusion.py            # Multimodal fusion scoring & decision thresholding
│   ├── image_classifier.py  # Zero-shot CLIP classifier with CITES + Indian prompts
│   ├── taxonomy.py          # CITES taxonomy & Hindi/Hinglish codeword matcher
│   ├── text_classifier.py   # TF-IDF baseline + XLM-RoBERTa cross-lingual module
│   └── checkpoints/         # Gitignored model weights directory
├── scripts/
│   ├── crawl.py             # Ethical, rate-limited public listing crawler
│   ├── evaluate.py          # Evaluation runner & report generator
│   └── train_text.py        # TF-IDF baseline trainer on WLT-OSN
├── tests/                   # 42 unit and integration tests (mocked offline)
├── Dockerfile               # Multi-stage production container image
├── requirements.txt         # Production runtime dependencies
└── requirements-dev.txt     # Developer tools (ruff, pytest)
```

---

## 3. Quickstart Guide (Local Development)

### 3.1 Setup Environment with `uv`

```bash
cd backend

# Create virtual environment (Python 3.11+)
uv venv .venv
source .venv/bin/activate  # Or on Windows: .venv\Scripts\Activate.ps1

# Install PyTorch CPU (or CUDA build if supported)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install dependencies
uv pip install -r requirements.txt -r requirements-dev.txt
```

### 3.2 Launch the FastAPI Development Server

```bash
cd backend
source .venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger API documentation will be available at:  
👉 **http://localhost:8000/docs**

---

## 4. API Endpoints

### 4.1 Health Check
- **`GET /health`**
- **Response**: `{"status": "ok", "timestamp": "...", "version": "1.0.0"}`

### 4.2 Multimodal Content Analysis
- **`POST /analyze`** (JSON payload)
  ```json
  {
    "type": "screenshot",
    "data": "data:image/jpeg;base64,...",
    "extractedText": "Original hathajodi for puja and vashikaran. DM for price."
  }
  ```
- **Response**: Full `AnalysisResult` object matching `src/types/index.ts`.

### 4.3 Multipart File Upload Analysis
- **`POST /analyze/upload`** (Multipart Form)
  - `file`: Image binary (JPEG, PNG, WebP)
  - `caption`: Optional text description

---

## 5. Running with Docker & Docker Compose

From the repository root:

```bash
# Build and start container with mounted volumes
docker compose up --build

# Run in background
docker compose up -d

# Stop container
docker compose down
```

The container automatically mounts `data/lexicons`, `data/processed`, and persists the SQLite database at `backend/wildlife_cases.db`.

---

## 6. Model Evaluation & Benchmarking

The TF-IDF baseline was benchmarked on the held-out test split of the WLT-OSN dataset (8,647 records, group-aware split on `source_group` to prevent seller-cluster leakage):

```bash
cd backend
PYTHONPATH=. .venv/bin/python scripts/evaluate.py
```

### Verified Empirical Metrics:
- **PR-AUC:** `0.9112`
- **F1 Score:** `0.8382`
- **Precision:** `81.45%`
- **Recall:** `86.32%`
- Full report available at: [docs/evaluation_report.md](../docs/evaluation_report.md)
- PR Curve graphic: `data/processed/pr_curve.png`

---

## 7. Ethical Public Listing Crawler

Run the crawler with rate limiting (≥ 3.0s per domain), strict robots.txt adherence, and PII scrubbing:

```bash
cd backend
# Dry-run offline test:
PYTHONPATH=. .venv/bin/python scripts/crawl.py --query hathajodi --dry-run

# Output saved to: data/raw/crawled_<timestamp>.jsonl
```

---

## 8. Quality Gates & Test Suite

Run all verification gates before submitting code:

```bash
cd backend
# Linter check
.venv/bin/ruff check .

# Formatter check
.venv/bin/ruff format --check .

# Full test suite (42 tests, offline, mocked models)
.venv/bin/pytest -q
```
