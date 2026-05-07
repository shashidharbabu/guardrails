"""
rag/embedder.py — Nemotron-8B embedding model wrapper.
=======================================================

Loads nvidia/llama-embed-nemotron-8b once (singleton) and exposes
two functions:

    embed_query(text)        — prepends QUERY_PREFIX, returns List[float]
    embed_texts(texts)       — batch embed WITHOUT prefix (for ingest)

The distinction between embed_query and embed_texts is critical:
  - At INGEST time: chunks were embedded without any prefix (raw text only)
  - At QUERY time:  queries must be prefixed with QUERY_PREFIX so the model
                    maps the query into the same vector space as the chunks

This matches exactly what the notebook does in cells 10 and 16.

The model is loaded lazily on first call — so importing this module
does not trigger a GPU load until retrieve() is actually called.
"""
from __future__ import annotations

import os
from typing import List, Optional

import torch
from transformers import AutoTokenizer, AutoModel

from rag.config import EMBED_MODEL, EMBED_MAX_LENGTH, QUERY_PREFIX, HF_TOKEN

# ── Singleton state ────────────────────────────────────────────────────────────
_tokenizer = None
_model     = None


def _load_model():
    """Load tokenizer + model once. Subsequent calls return immediately."""
    global _tokenizer, _model
    if _tokenizer is not None:
        return

    print(f"[Embedder] Loading {EMBED_MODEL} ...")

    hf_token: Optional[str] = HF_TOKEN if HF_TOKEN else None

    _tokenizer = AutoTokenizer.from_pretrained(
        EMBED_MODEL,
        trust_remote_code=True,
        token=hf_token,
    )

    _model = AutoModel.from_pretrained(
        EMBED_MODEL,
        trust_remote_code=True,
        token=hf_token,
        dtype=torch.bfloat16,   # `torch_dtype` is deprecated; use `dtype`
        device_map="auto",      # uses MPS on Apple Silicon, CUDA on GPU, CPU fallback
    )
    _model.eval()

    print(f"[Embedder] Loaded on device: {next(_model.parameters()).device}")


def _encode(texts: List[str]) -> List[List[float]]:
    """
    Core encode function — matches notebook cell 16 exactly:
      - last hidden state token ([:, -1])
      - L2 normalised
      - bfloat16
    """
    _load_model()

    inputs = _tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=EMBED_MAX_LENGTH,
    ).to(_model.device)

    with torch.no_grad():
        outputs = _model(**inputs)

    # Last token of last hidden state — matches notebook cell 10 and 16
    embeddings = outputs.last_hidden_state[:, -1]
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    return embeddings.cpu().tolist()


def embed_query(query: str) -> List[float]:
    """
    Embed a single query WITH the instruction prefix.
    This is what MAD calls at retrieval time.

    The prefix tells the model this is a retrieval query, not a passage.
    It must match what was used during the embedding model's training.
    """
    prefixed = QUERY_PREFIX + query
    return _encode([prefixed])[0]


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Batch embed multiple texts WITHOUT the query prefix.
    Used for ingesting new chunks — matches notebook ingest behaviour.
    Not called by MAD directly but included for completeness.
    """
    return _encode(texts)
