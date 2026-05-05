export const GATEWAY_METRICS = [
  {
    model: 'DeBERTa-base NER (fine-tuned)',
    hfId: 'shashidharbabu/deberta-pii-guardrails',
    task: 'PII Detection',
    metrics: [
      { name: 'Recall', value: '98.2%' },
      { name: 'Entity types', value: '57' },
      { name: 'Dataset', value: 'ai4privacy/pii-masking-200k' },
    ],
  },
  {
    model: 'RoBERTa-base (fine-tuned)',
    hfId: 'shashidharbabu/roberta-jailbreak-guardrails',
    task: 'Jailbreak Detection',
    metrics: [
      { name: 'F1 (in-dist)', value: '99.9%' },
      { name: 'Dataset', value: 'JailBreakV-28k + Dolly-15k' },
    ],
    warning: 'OOD weaknesses: fictional framing (0%), indirect multi-step (14%)',
  },
  {
    model: 'Llama-Prompt-Guard-2-86M (fine-tuned)',
    hfId: 'shashidharbabu/llama-prompt-guard-guardrails',
    task: 'Prompt Injection',
    metrics: [
      { name: 'Accuracy', value: '99.1%' },
      { name: 'Missed attack rate', value: '1.2%' },
    ],
  },
]

export const RAG_METRICS = {
  topline: [
    { label: 'Recall@1', value: '94.9%' },
    { label: 'MRR',      value: '0.937' },
    { label: 'Recall@3', value: '99.7%' },
    { label: 'nDCG@10',  value: '0.891' },
  ],
  corpus: {
    totalChunks: 4664,
    similarity: 'cosine',
    index: 'HNSW',
    host: 'Qdrant Cloud GCP',
    collection: 'ai_governance_chunks_nemotron8b',
  },
  embedderBenchmark: [
    { model: 'Qwen3-4B (fine-tuned ✓)',                  recall1: 0.949, highlight: true },
    { model: 'Qwen3-4B (base)',                           recall1: 0.882 },
    { model: 'nvidia/llama-nemotron-embed-1b-v2',         recall1: 0.866 },
    { model: 'jinaai/jina-embeddings-v3',                 recall1: 0.800 },
    { model: 'BAAI/bge-large-en-v1.5',                    recall1: 0.796 },
    { model: 'intfloat/e5-large-v2',                      recall1: 0.792 },
    { model: 'mixedbread-ai/mxbai-embed-large-v1',        recall1: 0.791 },
    { model: 'thenlper/gte-large',                        recall1: 0.787 },
    { model: 'BAAI/bge-m3',                               recall1: 0.775 },
    { model: 'sentence-transformers/all-MiniLM-L6-v2',    recall1: 0.692 },
    { model: 'Snowflake/snowflake-arctic-embed-m',        recall1: 0.509 },
  ],
  oodJailbreak: [
    { category: 'Direct commands',       accuracy: 0.88 },
    { category: 'Role-play framing',     accuracy: 0.54 },
    { category: 'Indirect multi-step',   accuracy: 0.14 },
    { category: 'Fictional framing',     accuracy: 0.00 },
  ],
}

export const MAD_METRICS = {
  pending: true,
  evalDataset: {
    size: 200,
    domain: 'Healthcare',
    errorTypes: ['fully_correct', 'missing_caveat', 'hallucinated_specific', 'jurisdiction_blind'],
    status: 'Pending — synthetic dataset under construction',
  },
}
