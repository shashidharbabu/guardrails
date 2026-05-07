# Guardrails Enterprise — AI Guardrails & ML Infrastructure

Enterprise AI security and ML infrastructure platform. Wraps any enterprise LLM (healthcare, banking, legal, HR) with input and output guardrail layers, backed by a regulatory RAG corpus.

## 📁 Repository Structure

```
guardrails-enterprise/
├── docs/                    # All documentation
│   ├── README.md           # PII NER pipeline documentation
│   ├── QUICK_START.md      # Quick start guide
│   └── SETUP_GUIDE.md      # Detailed setup instructions
├── config/                  # Configuration files
├── dags/                    # Airflow DAGs
├── plugins/                 # Airflow plugins
├── scripts/                 # Utility scripts
│   ├── setup_airflow.sh
│   ├── start_airflow.sh
│   └── stop_airflow.sh
├── docker/                  # Docker configuration
│   ├── Dockerfile
│   └── docker-compose.yml
├── gateway/                 # Phase 1: Input guardrail (3 classifiers)
├── rag/                     # Phase 2: RAG pipeline (Qdrant + Qwen3-4B embedder)
├── multi_agent/             # Phase 3: MAD output guardrail (active)
├── multi_agent_debate/       # MAD pipeline with SQLite storage for GRPO
├── confidence/              # Phase 4: Confidence Scoring Engine
├── rlhf/                    # Phase 5: GRPO feedback loop
├── finetuning/              # Fine-tuning scripts for all 4 models
├── synthetic_data/          # Synthetic evaluation dataset generation
├── MAD_SETUP_GUIDE.md       # Standalone MAD setup guide
└── requirements.txt         # Root-level shared dependencies
```

## 🚀 Quick Start

For detailed setup instructions, see:
- **[Quick Start Guide](docs/QUICK_START.md)** - Get started in 15 minutes
- **[Setup Guide](docs/SETUP_GUIDE.md)** - Comprehensive setup instructions
- **[Full Documentation](docs/README.md)** - Complete project documentation
- **[MAD Setup Guide](MAD_SETUP_GUIDE.md)** - Multi-Agent Debate service setup

### Local docker-only observability (recommended)

Run Gateway + MAD + Langfuse (self-hosted) + Datadog Agent with a single Compose:

```bash
cp .env.example .env
# Fill in at least DD_API_KEY + LANGFUSE_* keys (or use the local bootstrap defaults in docs)
docker compose up -d --build
./scripts/smoke_observability.sh
```

Runbook: `docs/LOCAL_OBSERVABILITY_RUNBOOK.md`

## ✅ Active Pipelines

### PII NER Pipeline
- **Location**: `dags/pii_ner_pipeline.py`
- **Purpose**: Download, EDA, and BIO NER transformation of the `ai4privacy/pii-masking-200k` dataset → GCS
- **Status**: ✅ Active
- **Tasks**: `download_from_huggingface` → `load_raw_data` → `perform_eda` → `transform_data` → `upload_processed_data`

### Multi-Agent Debate (MAD) Output Guardrail
- **Location**: `multi_agent/` (core pipeline), `multi_agent_debate/` (with SQLite storage)
- **Purpose**: Verifies enterprise LLM answers against a regulatory evidence corpus. Extracts atomic claims, runs a 2-cycle adversarial debate (Agent A vs Agent B), routes through a partially-blind judge, and returns a routing decision (DELIVER / RETRY / HARD_BLOCK / HUMAN_REVIEW)
- **Status**: ✅ Active
- **Run**: `uvicorn multi_agent.api:app --port 8001 --reload`

## 🔮 Planned / In Development

### Gateway — Input Guardrail
- **Location**: `gateway/`
- **Purpose**: 3 parallel classifiers (prompt injection, PII detection, jailbreak detection) with a weighted decision engine. Blocks threats before they reach the LLM.
- **Status**: 🚧 In Development

### RAG Pipeline
- **Location**: `rag/`
- **Purpose**: Qdrant vector store + Qwen3-4B embedder + BM25 hybrid + cross-encoder reranker over a 72+ regulatory document corpus
- **Status**: 🚧 In Development

### Confidence Scoring Engine
- **Location**: `confidence/`
- **Purpose**: Compute a final confidence score from LLM faithfulness, hallucination rate, RAG relevancy, and judge evaluation
- **Status**: 🚧 In Development

### GRPO Feedback Loop
- **Location**: `rlhf/`
- **Purpose**: GRPO-based fine-tuning loop for Agent A (Brier reward) and Agent B (precision reward) using data written by the MAD pipeline to SQLite
- **Status**: 🚧 In Development

### Finetuning Pipelines
- **Location**: `finetuning/`
- **Purpose**: Fine-tuning scripts for RoBERTa (jailbreak + PII), Llama-Prompt-Guard-2-86M (prompt injection), Qwen2.5-3B-Instruct (LLM generator), and Qwen3-4B (RAG embedder)
- **Status**: 🚧 In Development

### Synthetic Evaluation Dataset
- **Location**: `synthetic_data/`
- **Purpose**: Generate 200-example healthcare-domain evaluation set (4 error types: fully_correct, missing_caveat, hallucinated_specific, jurisdiction_blind) for MAD pipeline benchmarking
- **Status**: 🚧 In Development

## 🛠️ Development

### Prerequisites
- Docker & Docker Compose
- Python 3.10+
- Google Cloud Platform account (for GCS)
- Ollama (for MAD pipeline — `ollama pull qwen2.5:7b`)

### Airflow Setup
```bash
# Setup Airflow
./scripts/setup_airflow.sh

# Start services
./scripts/start_airflow.sh

# Stop services
./scripts/stop_airflow.sh
```

### MAD Pipeline (Quick Start)
```bash
pip install -r multi_agent/requirements.txt
ollama pull qwen2.5:7b && ollama serve
python -m multi_agent.run_test
```

## 📚 Documentation

- [PII NER Pipeline Docs](docs/README.md)
- [Quick Start Guide](docs/QUICK_START.md)
- [Setup Guide](docs/SETUP_GUIDE.md)
- [MAD Setup Guide](MAD_SETUP_GUIDE.md)

