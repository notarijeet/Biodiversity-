# STATUS.md - shared handoff for all humans and AI agents

Read this file in full at the start of every session, after AGENTS.md. Update it at the end of every task. Last updated: 2026-11-17.

## Next up

All automated agent tasks across Phases 1, 2, 3, and 4 are now completed. Remaining items:

1. Phase 2: Task 2.7 - Connect frontend to backend (`src/components/Detection/aiService.ts` point to `http://localhost:8000/analyze`, HUMAN_ONLY / frontend owner).
2. Phase 3: Task 3.3 - YOLOv8 fine-tuning (BLOCKED pending 500+ labelled wildlife images dataset).

## Tasks

Status values: TODO, IN_PROGRESS, DONE, BLOCKED, HUMAN_ONLY, UNKNOWN. Plan details live in docs/implementation-plan.md.

| Task | Status | Owner | Updated | Notes |
|---|---|---|---|---|
| 1.1 PPT tech stack slide | DONE | S-H-R-I-A-N-S-H | 2026-11-17 | Completed and verified in docs/tech-stack-slide.md. |
| 1.2 draw.io architecture diagram | DONE | S-H-R-I-A-N-S-H | 2026-11-17 | Completed and verified in docs/system-architecture.drawio. |
| 2.0 Check contract vs src/types/index.ts (read-only) | DONE | Arijeet Singh | 2026-11-18 | Contract recorded in Decision 11. Literals aligned to frontend types per user decision. |
| 2.1 Backend scaffold, uv env, tests/config files | DONE | Arijeet Singh | 2026-11-18 | Created backend scaffold, DB schemas & session, fusion logic, schemas, and test suite. All quality gates passed. |
| 2.2 Load and validate WLT-OSN | DONE | Arijeet Singh | 2026-11-18 | Downloaded share_data.csv to data/raw/ (gitignored); built loader with deduplication and group-aware split (0% leakage); validation notebook created. |
| 2.3 TF-IDF baseline | DONE | Arijeet Singh | 2026-11-18 | Built TF-IDF baseline and trained on WLT-OSN. Test PR-AUC: 0.9112, Precision: 0.8145, Recall: 0.8632, F1: 0.8382. Metrics & model saved to data/processed/. |
| 2.4 XLM-RoBERTa fine-tuning | DONE | Arijeet Singh | 2026-11-18 | Implemented XLM-RoBERTa dataset tokenization, lazy inference wrapper (XLMROnlineClassifier), TrainingArguments, and Kaggle/Colab training notebook (02_xlm_roberta_training.ipynb). |
| 2.5 CLIP zero-shot image classifier | DONE | Arijeet Singh | 2026-11-18 | Implemented CLIP zero-shot image classifier (image_classifier.py), base64 decoding, IWT positive/negative queries, lazy CLIPOnlineClassifier, and unit tests. |
| 2.6 Fusion and /analyze API | DONE | Arijeet Singh | 2026-11-19 | Connected multimodal score fusion (0.6 text + 0.4 image), CITES taxonomy/indicator matching, and /analyze endpoints. All 26 tests passed. |
| 2.7 Connect frontend to backend | HUMAN_ONLY | frontend owner | | Frontend is read-only for agents. Agents must not do this. |
| 3.1 Public listing crawler | DONE | Arijeet Singh | 2026-11-19 | Built ethical rate-limited crawler (crawl.py) with robots.txt check, per-domain 3s delay, regex PII scrubber, seen_urls.txt deduplicator, and valid JSONL output. All 42 tests passed. |
| 3.2 Hindi/Hinglish codewords | DONE | Arijeet Singh | 2026-11-19 | Added 12 India-specific codewords in data/lexicons/india_codewords.json; integrated with taxonomy matching and CLIP prompts. All 30 tests passed. |
| 3.3 YOLOv8 fine-tuning | BLOCKED | | | Needs 500+ labelled images. |
| 4.1 Docker | DONE | Arijeet Singh | 2026-11-19 | Created multi-stage Dockerfile (non-root iwt user, healthcheck), docker-compose.yml with volume mounts, backend/.dockerignore, and backend/.env.example. |
| 4.2 Evaluation report | DONE | Arijeet Singh | 2026-11-19 | Built evaluation metrics module, test suite, and scripts/evaluate.py; generated pr_curve.png, per-group breakdown, failure mode analysis, and docs/evaluation_report.md (PR-AUC 0.9112, F1 0.8382). All 35 tests passed. |

## Decisions log

Settled decisions. Do not reopen without asking the user.

1. Predictions are only SUSPECTED WILDLIFE TRADE, REQUIRES MANUAL REVIEW, NO EVIDENCE DETECTED. Never "illegal".
2. Phase 2 backend is stateless (Option A). No database until Phase 3.
3. Frontend is read-only for agents. Frontend changes go through "Requests for the frontend owner" below.
4. Package manager is uv with backend/.venv. Torch is installed first from the cu121 index and is not listed in requirements.txt.
5. Local machine is a Windows laptop with a GTX 1650 (4 GB). Inference runs locally on GPU with CPU fallback. XLM-R training happens on Kaggle or Colab, or locally only with frozen embeddings.
6. No free cloud hosting for the API is assumed (free tiers are too small or restricted). Round 2 on 26 Nov is offline, so the demo runs locally. Keep the heuristic fallback in aiService.ts.
7. Backend structure follows Task 2.1 in the plan, plus models/fusion.py, api/schemas.py, main.py, tests/, pytest.ini, pyproject.toml (ruff config), requirements-dev.txt.
8. All backend commands run from inside backend/.
9. Quality gates: ruff check, ruff format --check, pytest -q, plus the frontend-untouched check (see AGENTS.md).
10. data/raw, data/processed, backend/models/checkpoints and "Training data/" are gitignored. data/lexicons is tracked.
11. AnalysisResult contract recorded from src/types/index.ts (Task 2.0):
    - Fields: id (str), submissionDate (str), inputType ('screenshot' | 'url'), inputSource (str), prediction ('POTENTIAL ILLEGAL WILDLIFE TRADE' | 'NO CLEAR EVIDENCE DETECTED' | 'REQUIRES MANUAL REVIEW'), confidenceScore (float 0.0-1.0), identifiedSpecies (list of {commonName, scientificName, citesStatus, conservationStatus}), extractedText (str), suspiciousKeywords (list[str]), suspiciousIndicators (list of {indicator, severity: 'high'|'medium'|'low', description}), explanation (str), analysisStatus ('COMPLETED' | 'FLAGGED_FOR_HUMAN_ENFORCEMENT' | 'CLEARED'), threatSeverity ('Critical' | 'Elevated' | 'Low' | 'Negligible'), imageUrl (optional str), isSample (optional bool).
    - User approved Option A: Backend will output these exact frontend string literals to maintain 100% compatibility with frontend fallback and types, with Pydantic schemas validating them.
    - Database architecture designed alongside backend using SQLAlchemy models (cases, identified_species, suspicious_indicators, case_audit_logs).

## Open questions

1. Resolved: Prediction literals and inputType aligned to src/types/index.ts (see Decision 11).
2. Final project name (team decision).
3. Who owns which lane? Fill the Owner column when tasks are claimed.

## Requests for the frontend owner

1. `src/components/Detection/aiService.ts` - Add `BACKEND_URL` (`http://localhost:8000` via `VITE_BACKEND_URL`) to POST submissions to `/analyze` before falling back to local heuristics on error. Preserves offline demo fallback while connecting frontend to live FastAPI ML backend (Task 2.7). Requested by Arijeet Singh, 2026-11-19.

## Log

Append one line per event, newest at the bottom. Format: date - person or agent - what happened.

- 2026-11-17 - Claude - Created STATUS.md and AGENTS.md; synced docs/implementation-plan.md (uv, tests, frontend read-only).
- 2026-11-17 - Antigravity - Initialized AGENTS.md and STATUS.md in root; verified Phase 1 deliverables (Tasks 1.1 & 1.2 marked DONE); verified backend/ does not exist yet (Task 2.1 ready as TODO).
- 2026-11-18 - Arijeet Singh & Antigravity - Completed Task 2.0 contract check; recorded AnalysisResult literals in Decision 11; designed database architecture (cases, species, indicators, audit logs).
- 2026-11-18 - Arijeet Singh & Antigravity - Completed Task 2.1: scaffolded backend/, database models & session, fusion scoring, schemas, and test suite. ruff check, ruff format --check, and pytest -q all exit 0. Frontend verified untouched.
- 2026-11-18 - Arijeet Singh & Antigravity - Completed Task 2.2: downloaded and validated WLT-OSN dataset (8,647 deduplicated rows, 254 positives ~2.9%, 167 source groups). Group-aware split confirmed 0% group leakage. Created notebooks/01_data_validation.ipynb and tests/test_data_loader.py. All gates passed.
- 2026-11-18 - Arijeet Singh & Antigravity - Completed Task 2.3: implemented TF-IDF baseline and BaselineTextClassifier; trained and evaluated on group-aware split (PR-AUC 0.9112, F1 0.8382); saved baseline_metrics.json and baseline_tfidf.joblib. All gates passed.
- 2026-11-18 - Arijeet Singh & Antigravity - Completed Task 2.4: implemented XLM-RoBERTa tokenization, dataset class, training arguments, weighted trainer logic, lazy XLMROnlineClassifier inference wrapper, and Kaggle/Colab training notebook (02_xlm_roberta_training.ipynb). All quality gates passed.
- 2026-11-18 - Arijeet Singh & Antigravity - Completed Task 2.5: implemented CLIP zero-shot image classification module (image_classifier.py), base64 image decoder, positive and negative prompt sets, lazy CLIPOnlineClassifier wrapper, and unit tests. All quality gates passed. Frontend untouched.
- 2026-11-19 - Arijeet Singh & Antigravity - Completed Task 2.6: connected score fusion (0.6 text + 0.4 image), CITES taxonomy/trade indicator scanning (data/lexicons/), and /analyze endpoints. All 26 tests passed. Quality gates clean. Frontend untouched.
- 2026-11-19 - Arijeet Singh & Antigravity - Completed Task 3.2: created data/lexicons/india_codewords.json (12 Indian codewords: hathajodi, safed sona, khurpa, kasturi, shahtoosh, etc.); integrated into taxonomy matching and CLIP prompts; tested hathajodi and safed sona API triggers. All 30 tests passed. Frontend untouched.
- 2026-11-19 - Arijeet Singh & Antigravity - Completed Task 4.1: created backend/Dockerfile (multi-stage Python 3.11, non-root user, healthcheck), docker-compose.yml with persistent volume mounts, backend/.dockerignore, and backend/.env.example. All quality gates clean.
- 2026-11-19 - Arijeet Singh & Antigravity - Completed Task 4.2: created backend/evaluation/metrics.py, backend/scripts/evaluate.py, tests/test_evaluation.py; executed evaluation on held-out test split (PR-AUC 0.9112, F1 0.8382); generated pr_curve.png, per-group analysis, failure modes, and docs/evaluation_report.md. All 35 tests passed. Frontend untouched.
- 2026-11-19 - Arijeet Singh & Antigravity - Completed Task 3.1: created backend/scripts/crawl.py (PIIScrubber, DomainRateLimiter, RobotsChecker, CrawlDeduplicator, EthicalCrawler) and tests/test_crawler.py; verified robots compliance, >= 3.0s domain rate limiting, phone/email/handle scrubbing, and JSONL streaming. All 42 tests passed. Frontend untouched.
- 2026-11-19 - Arijeet Singh & Antigravity - Code audit & bug fix: updated AnalyzeRequest and /analyze route to seamlessly handle frontend payload structure ({type, data, extractedText}); updated text_classifier.py to dynamically inspect TrainingArguments for activation_checkpointing (eliminating all Pytest warnings); created backend/README.md; added test_analyze_frontend_payload_format. All 43 tests passed with 0 warnings. Frontend untouched.
