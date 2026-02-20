# Guardrails - RAG Pipeline & ML Infrastructure

This repository contains the infrastructure and pipelines for building a production-ready RAG (Retrieval-Augmented Generation) system with advanced ML capabilities.

## 📁 Repository Structure

```
guardrails-1/
├── docs/                    # All documentation
│   ├── README.md           # Main project documentation
│   ├── QUICK_START.md      # Quick start guide
│   ├── SETUP_GUIDE.md      # Detailed setup instructions
│   └── Agents - RL.pdf     # Research papers and references
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
├── synthetic_data/          # Synthetic data generation pipeline
├── finetuning/              # Embedding & LLM finetuning pipelines
├── rag/                     # RAG pipeline components
├── multi_agent/             # Multi-agent debate system
├── rlhf/                    # RLHF (Reinforcement Learning from Human Feedback) pipeline
└── confidence/              # Confidence engine
```

## 🚀 Quick Start

For detailed setup instructions, see:
- **[Quick Start Guide](docs/QUICK_START.md)** - Get started in 15 minutes
- **[Setup Guide](docs/SETUP_GUIDE.md)** - Comprehensive setup instructions
- **[Full Documentation](docs/README.md)** - Complete project documentation

## 🎯 Current Pipelines

### PII NER Pipeline
- **Location**: `dags/pii_ner_pipeline.py`
- **Purpose**: EDA and transformation of PII datasets for NER model training
- **Status**: ✅ Active

## 🔮 Planned Pipelines

### Synthetic Data Generation
- **Location**: `synthetic_data/`
- **Purpose**: Generate synthetic training data for embedding model finetuning
- **Status**: 🚧 In Development

### Finetuning Pipelines
- **Location**: `finetuning/`
- **Purpose**: 
  - Embedding model finetuning for RAG
  - LLM finetuning for RAG pipeline
- **Status**: 🚧 In Development

### RAG Pipeline
- **Location**: `rag/`
- **Purpose**: Core RAG system components
- **Status**: 🚧 In Development

### Multi-Agent Debate
- **Location**: `multi_agent/`
- **Purpose**: Multi-agent debate system for improved reasoning
- **Status**: 🚧 In Development

### RLHF Pipeline
- **Location**: `rlhf/`
- **Purpose**: Reinforcement Learning from Human Feedback pipeline
- **Status**: 🚧 In Development

### Confidence Engine
- **Location**: `confidence/`
- **Purpose**: Confidence scoring and calibration system
- **Status**: 🚧 In Development

## 🛠️ Development

### Prerequisites
- Docker & Docker Compose
- Python 3.10+
- Google Cloud Platform account (for GCS)

### Setup
```bash
# Setup Airflow
./scripts/setup_airflow.sh

# Start services
./scripts/start_airflow.sh

# Stop services
./scripts/stop_airflow.sh
```

## 📚 Documentation

All documentation is located in the `docs/` directory:
- [Main README](docs/README.md)
- [Quick Start Guide](docs/QUICK_START.md)
- [Setup Guide](docs/SETUP_GUIDE.md)

## 📝 License

[Add your license information here]

## 🤝 Contributing

[Add contributing guidelines here]
