# RAG_Ingest_Colab.ipynb — Debug Reference

**Notebook:** `multi_agent_debate/RAG_Ingest_Colab.ipynb`  
**Purpose:** Embed regulatory corpus with Nemotron-8B and upload to Qdrant Cloud  
**Qdrant cluster:** AWS us-west-1  
**Collection:** `ai_governance_chunks_nemotron8b`  
**Vector size:** 4096 (Nemotron-8B output dimension)  
**Similarity:** Cosine  

---

## Quick Checklist Before Running

- [ ] Runtime set to **T4 GPU** (Runtime → Change runtime type)
- [ ] `QDRANT_API_KEY` filled in Cell 2 (already set to new cluster key)
- [ ] `HF_TOKEN` filled in Cell 2 (needed for `nvidia/llama-embed-nemotron-8b`)
- [ ] Cell order: run **top to bottom, one at a time** if debugging

---

## Cell-by-Cell Breakdown

---

### Cell 1 — Install Dependencies

```python
!pip install -q qdrant-client transformers accelerate torch sentencepiece
print('Dependencies installed')
```

**Expected output:** `Dependencies installed`  
**If it fails:** Colab may need a runtime restart after install. Runtime → Restart runtime, then re-run.

---

### Cell 2 — Credentials

```python
import os
os.environ['QDRANT_URL']        = 'https://07e7370c-7123-4e56-84c4-d7ffc8f9b2c7.us-west-1-0.aws.cloud.qdrant.io'
os.environ['QDRANT_API_KEY']    = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6MTVmNjY2NjEtYTk2My00NDQ1LWI2MWUtYjczMDIzMGEwNDY1In0.0xsXRM2eW4xb0xf0ITF48d4oSd8ypfTaDJ94g8iM_GY'
os.environ['QDRANT_COLLECTION'] = 'ai_governance_chunks_nemotron8b'
os.environ['HF_TOKEN']          = 'PASTE_YOUR_HF_TOKEN_HERE'   # <-- FILL THIS IN
os.environ['EMBED_MODEL']       = 'nvidia/llama-embed-nemotron-8b'

QDRANT_URL      = os.environ['QDRANT_URL']
QDRANT_API_KEY  = os.environ['QDRANT_API_KEY']
COLLECTION_NAME = os.environ['QDRANT_COLLECTION']
HF_TOKEN        = os.environ['HF_TOKEN']
EMBED_MODEL     = os.environ['EMBED_MODEL']

print('Qdrant URL :', QDRANT_URL)
print('API key    :', 'SET')
print('HF Token   :', 'SET' if HF_TOKEN != 'PASTE_YOUR_HF_TOKEN_HERE' else '*** NOT SET — FILL IN ABOVE ***')
```

**Expected output:**
```
Qdrant URL : https://07e7370c-7123-4e56-84c4-d7ffc8f9b2c7.us-west-1-0.aws.cloud.qdrant.io
API key    : SET
HF Token   : SET
```

**If HF Token shows NOT SET:** Replace `PASTE_YOUR_HF_TOKEN_HERE` with your token from huggingface.co/settings/tokens

---

### Cell 3 — GPU Check

```python
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device:', torch.cuda.get_device_name(0))
    print('VRAM  :', round(torch.cuda.get_device_properties(0).total_memory/1e9,1), 'GB')
else:
    print('WARNING: No GPU — embedding will be extremely slow')
```

**Expected output (T4):**
```
CUDA available: True
Device: Tesla T4
VRAM  : 15.8 GB
```

**If CUDA is False:** You forgot to set GPU runtime. Runtime → Change runtime type → T4 GPU → Save → reconnect.

---

### Cell 4 — Build Corpus

Defines `make_chunks()` and builds `ALL_CHUNKS` list from 9 regulatory documents:

| Variable | Doc ID | Tier | Source |
|---|---|---|---|
| `HIPAA_PRIVACY` | `hipaa_privacy_rule_45cfr164` | T1 | 45 CFR 164.502, .514, .524, .528 |
| `HIPAA_SECURITY` | `hipaa_security_rule_45cfr164` | T1 | 45 CFR 164.308, .312, .312(e) + HHS 2023 |
| `HITECH` | `hitech_act_breach_notification` | T1 | Section 13402, Unsecured PHI definition |
| `GDPR` | `gdpr_general_data_protection_regulation` | T1 | Art 5, 9, 17, 32, 33, 83 |
| `CCPA` | `ccpa_california_consumer_privacy_act` | T1 | §1798.100, .105, .120, .150 |
| `NIST` | `nist_sp800_security_guide` | T3 | SP 800-66, SP 800-53 |
| `HHS_OCR` | `hhs_ocr_guidance_documents` | T2 | Min necessary, right of access, breach |
| `EU_AI` | `eu_ai_act_high_risk_healthcare` | T1 | Art 9, 13 |
| `OWASP` | `owasp_top10_llm_applications` | T3 | LLM01, LLM06, LLM09 |

**Chunking params:** `chunk_size=400 words`, `overlap=50 words` (sliding window)

**Expected output:**
```
Corpus built: ~200 chunks across 9 documents
  ccpa_california_consumer_privacy_act: N chunks
  eu_ai_act_high_risk_healthcare: N chunks
  gdpr_general_data_protection_regulation: N chunks
  ...
```

**If chunk count is 0:** `make_chunks()` failed silently — check that `ALL_CHUNKS = []` line ran before the extend calls.

---

### Cell 5 — Load Nemotron-8B Embedder

```python
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

QUERY_PREFIX = 'Instruct: Retrieve relevant regulatory passage to answer the query\nQuery: '
EMBED_MAX_LENGTH = 512

tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL, trust_remote_code=True, token=hf_token)
model = AutoModel.from_pretrained(
    EMBED_MODEL, trust_remote_code=True, token=hf_token,
    torch_dtype=torch.bfloat16, device_map='auto',
)
model.eval()
```

**Expected output:**
```
Loading nvidia/llama-embed-nemotron-8b ...
Model loaded on: cuda:0
```

**Time:** ~5 min on first run (downloading ~16 GB model weights from HuggingFace).  

**Common errors:**

| Error | Fix |
|---|---|
| `401 Unauthorized` | HF_TOKEN is wrong or model access not granted — go to huggingface.co/nvidia/llama-embed-nemotron-8b and accept the license |
| `CUDA out of memory` | T4 has 15.8 GB — bfloat16 Nemotron-8B needs ~16 GB. Try `torch_dtype=torch.float16` or switch to A100 runtime |
| `trust_remote_code` warning | Expected — safe to ignore |

**`embed_texts()` function:** Embeds in batches of 8, uses `last_hidden_state[:, -1]` (last token), L2-normalized. **No query prefix** at ingest time — prefix is only added at retrieval time.

---

### Cell 6 — Create Qdrant Collection

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

VECTOR_SIZE = 4096
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)
# Deletes existing collection if present, then creates fresh
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
)
```

**Expected output:**
```
Connecting to Qdrant...
Creating collection ai_governance_chunks_nemotron8b (cosine, 4096d)...
Collection created
```

**Common errors:**

| Error | Fix |
|---|---|
| `401` / `Unauthorized` | API key wrong — re-check Cell 2 |
| `404 Not Found` | Cluster URL wrong — verify in Qdrant dashboard |
| `Connection refused` | Cluster not ready yet — wait 1-2 min after creation |
| `ValueError: wrong vector size` | VECTOR_SIZE must be 4096 for Nemotron-8B — do not change |

---

### Cell 7 — Embed and Upload

```python
texts = [c['text'] for c in ALL_CHUNKS]
all_vectors = embed_texts(texts, batch_size=8)   # ~15-30 min on T4
# Uploads in batches of 64 points
client.upsert(collection_name=COLLECTION_NAME, points=points)
```

**Expected output:**
```
Embedding ~200 chunks... (15-30 min on T4)
  Embedded 8/200...
  Embedded 88/200...
  ...
Embedding done in X.X min
Uploading 200 points to Qdrant...
  Uploaded 64/200
  Uploaded 128/200
  Uploaded 192/200
  Uploaded 200/200
Upload complete!
```

**Common errors:**

| Error | Fix |
|---|---|
| Colab disconnects mid-embedding | Re-run Cell 7 only — Cell 6 already created the collection, upsert is idempotent |
| `CUDA out of memory` during embedding | Reduce `batch_size=8` to `batch_size=4` |
| Upload hangs | Qdrant timeout — increase `timeout=120` to `timeout=300` in Cell 6 |
| `wrong vector dimension` | Nemotron-8B didn't load correctly — re-run Cell 5 |

---

### Cell 8 — Verify

```python
info = client.get_collection(COLLECTION_NAME)
# Runs test retrieval: "HIPAA encryption requirements for ePHI at rest"
results = client.query_points(collection_name=COLLECTION_NAME, query=q_vec, limit=3).points
```

**Expected output:**
```
=== Qdrant Collection Verified ===
  Collection : ai_governance_chunks_nemotron8b
  Points     : 200
  Status     : green

Top 3 results for HIPAA encryption query:
  score=0.XXXX  hipaa_security_rule_45cfr164__chunk_XXXX
  45 CFR 164.312 Technical Safeguards. Access Control is required...

  score=0.XXXX  hipaa_security_rule_45cfr164__chunk_XXXX
  ...

RAG ingestion complete!
```

**If scores are all < 0.3:** Query prefix mismatch — verify `QUERY_PREFIX` in Cell 5 matches exactly:
```
Instruct: Retrieve relevant regulatory passage to answer the query\nQuery: 
```

**If points_count = 0:** Upload failed silently — re-run Cell 7.

---

## Post-Ingestion: Verify Locally

```bash
cd /Users/spartan/Documents/guardrails-app-ui/multi_agent_debate
export $(grep -v '^#' ../.env | grep QDRANT | xargs)
python -m rag.check_qdrant
```

**Expected:**
```json
{
  "status": "ok",
  "points_count": 200
}
```

---

## Key Constants

| Constant | Value | Where set |
|---|---|---|
| Qdrant URL | `https://07e7370c-7123-4e56-84c4-d7ffc8f9b2c7.us-west-1-0.aws.cloud.qdrant.io` | Cell 2, `.env` |
| Collection | `ai_governance_chunks_nemotron8b` | Cell 2, `.env` |
| Vector size | `4096` | Cell 6 |
| Embed model | `nvidia/llama-embed-nemotron-8b` | Cell 2 |
| Chunk size | `400 words` | Cell 4 `make_chunks()` |
| Chunk overlap | `50 words` | Cell 4 `make_chunks()` |
| Embed batch size | `8` | Cell 5 `embed_texts()` |
| Upload batch size | `64` | Cell 7 |
| Query prefix | `Instruct: Retrieve relevant regulatory passage to answer the query\nQuery: ` | Cell 5, `rag/config.py` |
| Similarity metric | Cosine | Cell 6 |
| Embedding dtype | bfloat16 | Cell 5 |
| Last token pooling | `last_hidden_state[:, -1]` | Cell 5 |
| L2 normalisation | Yes | Cell 5 `F.normalize(..., p=2)` |
