# Gateway Models

Place your finetuned model folders here.

## Required Folder Structure

```text
gateway/models/
├── pii_ner_model/
│   ├── config.json
│   ├── tokenizer_config.json
│   ├── tokenizer.json              (or vocab.txt for older tokenizers)
│   ├── special_tokens_map.json
│   └── pytorch_model.bin           (or model.safetensors)
│
└── threat_classifier_model/
    ├── config.json
    ├── tokenizer_config.json
    ├── tokenizer.json
    ├── special_tokens_map.json
    └── pytorch_model.bin           (or model.safetensors)
```

## How to Save Finetuned Models

After finetuning, save with:

```python
model.save_pretrained("/content/drive/MyDrive/models/pii_ner_model")
tokenizer.save_pretrained("/content/drive/MyDrive/models/pii_ner_model")
```

Then download the folder and place it under `gateway/models/`.

## Using HuggingFace Hub IDs Instead of Local Folders

```bash
export PII_MODEL_PATH="your-username/pii-ner-finetuned"
export THREAT_MODEL_PATH="your-username/deberta-jailbreak-pi"
```

Both local paths and Hub IDs work with `transformers.pipeline()`.

## Testing Before Finetuned Models Are Available

To test gateway plumbing:

```bash
export PII_MODEL_PATH="dslim/bert-base-NER"
export THREAT_MODEL_PATH="deepset/deberta-v3-base-injection"
```
