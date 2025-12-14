import argparse
import asyncio
import json
import math
import os
import random
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

# Optional imports - if not available, fallback heuristics are used
try:
    from sentence_transformers import SentenceTransformer, util
    EMB_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    EMB_MODEL = None

# Optional OpenAI usage for LLM-as-a-Judge
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if OPENAI_API_KEY:
    try:
        import openai
        openai.api_key = OPENAI_API_KEY
    except Exception:
        openai = None
else:
    openai = None


def approximate_token_count(text: str) -> int:
    """
    Rough token estimate. tiktoken is more accurate, but this is a lightweight fallback:
    counts tokens as ~ 0.75 * number_of_words + punctuation fudge.
    """
    if not text:
        return 0
    words = text.split()
    rough = int(len(words) * 1.3)
    # clip to reasonable range
    return max(1, rough)


@dataclass
class EvalResult:
    conversation_id: str
    latency_ms: Optional[int]
    cost_proxy_usd: float
    token_count: int
    embedding_similarity: Optional[float]
    hallucination_flag: bool
    relevance_rating: Optional[int]  # 1-5 (if LLM used or heuristic)
    completeness_rating: Optional[int]
    judge_used: bool
    notes: Dict[str, Any]

    def to_dict(self):
        return asdict(self)


async def compute_embedding_similarity(response: str, contexts: List[str]) -> Optional[float]:
    """
    Returns max cosine similarity between response and each context snippet.
    If embedding model not available, return None.
    """
    if not EMB_MODEL:
        return None

    if not response or not contexts:
        return 0.0

    try:
        resp_emb = EMB_MODEL.encode(response, convert_to_tensor=True)
        ctx_embs = EMB_MODEL.encode(contexts, convert_to_tensor=True)
        sims = util.cos_sim(resp_emb, ctx_embs).cpu().numpy().flatten()
        best = float(sims.max())
        return best
    except Exception:
        return None


def estimate_cost_usd(token_count: int, model: str = "gpt-3.5-turbo") -> float:
    """
    Very approximate cost estimator.
    Adjust per-provider/pricing. These are placeholder rates:
    - gpt-3.5-turbo: $0.000002 per token (example)
    - gpt-4: higher
    """
    # example numbers (replace with real pricing in production)
    rate_per_token = 0.000002
    return token_count * rate_per_token


async def call_llm_judge(user_query: str, ai_response: str, contexts: List[str]) -> Dict[str, Any]:
    """
    Use OpenAI API (if available) to evaluate:
    - hallucination (Yes/No)
    - relevance 1-5
    - completeness 1-5

    If OpenAI unavailable, return a heuristic fallback.
    """
    if openai is None:
        # Heuristic fallback
        sim = None
        if EMB_MODEL:
            sim = await compute_embedding_similarity(ai_response, contexts)
        # Basic heuristics:
        hallucination = False if (sim is not None and sim > 0.55) else True if (sim is not None and sim < 0.35) else False
        relevance = 4 if sim and sim > 0.6 else 3 if sim and sim > 0.45 else 2
        completeness = relevance  # simple fallback
        return {
            "hallucination": hallucination,
            "relevance": int(relevance),
            "completeness": int(completeness),
            "raw": {"similarity": sim},
        }

    # Build a compact prompt - keep it constrained to reduce tokens/cost.
    prompt = (
        "You are an objective evaluator. Based ONLY on the CONTEXT paragraphs and the USER QUERY, "
        "answer these items in JSON with keys: hallucination (Yes/No), relevance (1-5), completeness (1-5). "
        f"USER QUERY: {user_query}\n\nAI RESPONSE: {ai_response}\n\nCONTEXT:\n"
    )
    # attach top-k contexts only
    for i, c in enumerate(contexts[:5], start=1):
        prompt += f"[CTX {i}]: {c}\n"

    prompt += (
        "\nReturn ONLY valid JSON like: {\"hallucination\":\"No\",\"relevance\":4,\"completeness\":4}"
        " and nothing else."
    )

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        content = resp["choices"][0]["message"]["content"].strip()
        # best-effort parse JSON
        try:
            j = json.loads(content)
            return {
                "hallucination": True if str(j.get("hallucination", "")).lower().startswith("y") else False,
                "relevance": int(j.get("relevance", 3)),
                "completeness": int(j.get("completeness", 3)),
                "raw": j,
            }
        except Exception:
            # couldn't parse -> fallback heuristic but include raw text
            return {"hallucination": False, "relevance": 3, "completeness": 3, "raw": {"text": content}}
    except Exception as exc:
        # API error -> fallback heuristic
        return {"error": str(exc), "hallucination": False, "relevance": 3, "completeness": 3, "raw": {}}


async def evaluate_single(
    conv: Dict[str, Any], contexts: List[str], sample_rate: float = 0.1
) -> EvalResult:
    """
    Evaluate a single conversation object.
    Expected conv fields:
      - id (optional)
      - user_message
      - ai_response
      - query_ts (unix or iso) (optional)
      - response_ts (unix or iso) (optional)
    """
    conv_id = conv.get("id") or str(random.randint(100000, 999999))
    user_msg = conv.get("user_message", "")
    ai_resp = conv.get("ai_response", "")
    q_ts = conv.get("query_ts")
    r_ts = conv.get("response_ts")

    latency_ms = None
    if q_ts and r_ts:
        try:
            # support epoch (seconds) or milliseconds, or ISO format
            def _to_ms(ts):
                if isinstance(ts, (int, float)):
                    # guess seconds vs ms
                    if ts > 1e12:
                        return int(ts)  # ms
                    if ts > 1e9:
                        return int(ts * 1000)  # seconds -> ms
                    return int(ts)
                # try parse iso
                try:
                    from datetime import datetime

                    dt = datetime.fromisoformat(ts)
                    return int(dt.timestamp() * 1000)
                except Exception:
                    return None

            q_ms = _to_ms(q_ts)
            r_ms = _to_ms(r_ts)
            if q_ms and r_ms:
                latency_ms = max(0, r_ms - q_ms)
        except Exception:
            latency_ms = None

    # token estimate
    token_count = approximate_token_count(user_msg + " " + ai_resp)
    cost_proxy = estimate_cost_usd(token_count)

    # embedding similarity (fast check)
    emb_sim = await compute_embedding_similarity(ai_resp, contexts)

    # simple hallucination heuristic
    hallucination_flag = False
    if emb_sim is not None:
        # if max similarity to context is low, mark hallucination risk
        if emb_sim < 0.35:
            hallucination_flag = True

    # Decide whether to call LLM judge
    judge_used = False
    judge_output = {}
    # Use sampling or if flagged by heuristic
    if random.random() < sample_rate or hallucination_flag or (emb_sim is not None and emb_sim < 0.5):
        judge_used = True
        judge_output = await call_llm_judge(user_msg, ai_resp, contexts)

    # Consolidate ratings
    relevance = judge_output.get("relevance") if judge_used else (4 if (emb_sim and emb_sim > 0.6) else 3)
    completeness = judge_output.get("completeness") if judge_used else relevance

    # final hallucination – combine judge and heuristic
    if judge_used and "hallucination" in judge_output:
        hallucination_flag = judge_output["hallucination"]

    notes = {
        "judge_raw": judge_output.get("raw"),
        "embedding_available": EMB_MODEL is not None,
    }

    return EvalResult(
        conversation_id=conv_id,
        latency_ms=latency_ms,
        cost_proxy_usd=round(cost_proxy, 8),
        token_count=token_count,
        embedding_similarity=(emb_sim if emb_sim is not None else None),
        hallucination_flag=bool(hallucination_flag),
        relevance_rating=int(relevance) if relevance is not None else None,
        completeness_rating=int(completeness) if completeness is not None else None,
        judge_used=judge_used,
        notes=notes,
    )


async def evaluate_batch(
    conversations: List[Dict[str, Any]], contexts: List[str], sample_rate: float = 0.1
) -> List[Dict[str, Any]]:
    tasks = [evaluate_single(conv, contexts, sample_rate) for conv in conversations]
    results = await asyncio.gather(*tasks)
    return [r.to_dict() for r in results]


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="LLM evaluation pipeline (demo).")
    parser.add_argument("--conversation", required=True, help="Path to conversation JSON.")
    parser.add_argument("--context", required=True, help="Path to context vectors JSON (contains text snippets).")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--sample-rate", type=float, default=0.1, help="Fraction to send to LLM judge.")
    args = parser.parse_args()

    conv_data = load_json(args.conversation)
    context_data = load_json(args.context)

    # Normalize: conversation may be single object or list
    if isinstance(conv_data, dict):
        conversations = [conv_data]
    else:
        conversations = conv_data

    # Extract contexts: allow different schemas
    contexts = []
    if isinstance(context_data, dict) and "snippets" in context_data:
        contexts = [s.get("text", "") for s in context_data["snippets"]]
    elif isinstance(context_data, list):
        # list of objects with 'text' or list of strings
        if context_data and isinstance(context_data[0], dict):
            contexts = [c.get("text", "") for c in context_data]
        else:
            contexts = [str(x) for x in context_data]
    else:
        # fallback: try to find values in dict
        contexts = [str(v) for v in context_data.values()] if isinstance(context_data, dict) else []

    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(evaluate_batch(conversations, contexts, sample_rate=args.sample_rate))

    summary = {
        "run_ts": int(time.time()),
        "n_conversations": len(conversations),
        "results": results,
    }

    save_json(summary, args.output)
    print(f"Wrote evaluation output to {args.output}")


if __name__ == "__main__":
    main()
