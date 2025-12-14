# LLM Response Evaluation Pipeline

A lightweight Python pipeline to evaluate the reliability of LLM responses in real time.

This system automatically evaluates AI answers across:
- Response relevance
- Response completeness
- Hallucination / factual accuracy
- Latency and cost

The pipeline consumes:
- A conversation JSON (user query + AI response)
- Context vectors retrieved from a vector database

and produces a structured JSON evaluation report.

---

## Features
- Hybrid evaluation (fast heuristics + LLM-as-a-Judge)
- Embedding-based hallucination detection
- Token-based cost estimation
- Asynchronous, scalable design
- Cost-optimized for high-volume systems

---

## Quick Setup

```bash
pip install -r requirements.txt
python evaluation_pipeline.py \
  --conversation sample_data/conversation.json \
  --context sample_data/context_vectors_clean.json \
  --output results/output.json \
  --sample-rate 0.1
````

(Optional) Enable LLM-based evaluation:

```bash
export OPENAI_API_KEY="your_api_key"
```

---

## Architecture (High Level)

```
User Query → LLM Response → Evaluation Pipeline
            ├─ Latency & Cost
            ├─ Embedding Similarity
            ├─ Hallucination Detection
            └─ LLM-as-a-Judge (Selective)
```

---

## Scalability

* Tiered evaluation to minimize cost
* Asynchronous execution
* Supports caching and batching
* Designed for millions of daily conversations

---

## Output Example

```json
{
  "relevance_rating": 4,
  "completeness_rating": 3,
  "hallucination_flag": false,
  "latency_ms": 920
}
```

---

## Notes

* Falls back to heuristic evaluation if no API key is provided
* Designed to be easily extended or deployed as a microservice
