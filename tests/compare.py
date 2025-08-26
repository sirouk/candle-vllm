# tests/compare.py
import json
import statistics
import time
import requests
from collections import defaultdict

PROMPT = "Write a sentence about a whale in a tropical paradise."
MODEL = "unsloth/Llama-3.2-3B-Instruct"

def parse_token_and_lp(choice):
    """
    Be liberal in what we accept:
    - token from delta.content
    - logprob from choices[0].logprobs.content[0].logprob (vLLM/candle style)
    Some engines may omit logprobs or place them oddly; return None when absent.
    """
    tok = None
    lp = None
    delta = choice.get("delta", {})
    if isinstance(delta, dict):
        tok = delta.get("content")

    # candle/vLLM commonly: choice["logprobs"]["content"][0]["logprob"]
    lps = choice.get("logprobs")
    if isinstance(lps, dict):
        cont = lps.get("content")
        if isinstance(cont, list) and cont:
            first = cont[0]
            lp = first.get("logprob")

    return tok, lp

def stream_once(url, max_tokens=10):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "logprobs": True,
        "stream": True,
        "seed": 9999,
        "temperature": 0.0,
        "top_p": 1.0,
    }

    t0 = time.perf_counter()
    t_first = None
    t_last = None
    tokens = []
    logprobs = []
    tok_times = []

    with requests.post(url, json=payload, stream=True) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines():
            if not raw:
                continue
            if not raw.startswith(b"data: "):
                continue
            data = raw[6:]
            if data == b"[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = obj.get("choices") or []
            if not choices:
                continue
            tok, lp = parse_token_and_lp(choices[0])

            # Some engines may send empty content deltas (role announcements, etc.)
            # Only time-stamp when a non-empty token string arrives.
            if tok:
                now = time.perf_counter()
                if t_first is None:
                    t_first = now
                t_last = now
                tokens.append(tok)
                tok_times.append(now)
                logprobs.append(lp)

    t1 = time.perf_counter()

    # Metrics
    total_latency = t1 - t0
    ttft = (t_first - t0) if t_first else None
    post_first_dur = (t_last - t_first) if (t_first and t_last) else None

    # Throughputs
    n = len(tokens)
    tps_post_first = (n / post_first_dur) if (post_first_dur and post_first_dur > 0) else None
    tps_e2e = (n / total_latency) if total_latency > 0 else None

    # Median per-token gap (robust vs. bursts)
    gaps = [tok_times[i] - tok_times[i-1] for i in range(1, len(tok_times))]
    median_gap = statistics.median(gaps) if gaps else None

    return {
        "tokens": "".join(tokens),
        "logprobs": logprobs,
        "ttft": ttft,
        "total_latency": total_latency,
        "tps_post_first": tps_post_first,
        "tps_e2e": tps_e2e,
        "median_gap": median_gap,
        "count": n,
    }

def fmt(x, unit="s", digits=3):
    if x is None:
        return "n/a"
    if unit == "s":
        return f"{x:.3f}s"
    return f"{x:.2f}"

def run_engine(name, url, repeats=3):
    print(f"\n=== Testing {name} ===")
    runs = []
    for i in range(repeats):
        r = stream_once(url)
        runs.append(r)
        print(f"Run {i+1}:")
        print(f"  Tokens: {r['tokens']}")
        print(f"  Logprobs: {r['logprobs']}")
        print(f"  TTFT: {fmt(r['ttft'])} | Total: {fmt(r['total_latency'])} | "
              f"TPS(post-first): {fmt(r['tps_post_first'], unit='tps')} | TPS(e2e): {fmt(r['tps_e2e'], unit='tps')}")
        if r["median_gap"] is not None:
            print(f"  Median inter-token gap: {fmt(r['median_gap'])}")
        print("---")

    # Summary
    def safe_avg(key):
        vals = [x[key] for x in runs if x[key] is not None]
        return sum(vals)/len(vals) if vals else None

    print(f"Summary ({name} over {len(runs)} runs)")
    print(f"  Avg TTFT: {fmt(safe_avg('ttft'))}")
    print(f"  Avg Total: {fmt(safe_avg('total_latency'))}")
    print(f"  Avg TPS(post-first): {fmt(safe_avg('tps_post_first'), unit='tps')}")
    print(f"  Avg TPS(e2e): {fmt(safe_avg('tps_e2e'), unit='tps')}")
    print(f"  Median token gap (per-run median, then avg): "
          f"{fmt(statistics.mean([x['median_gap'] for x in runs if x['median_gap'] is not None]) if any(x['median_gap'] for x in runs) else None)}")
    return runs

if __name__ == "__main__":
    # Optional: send one warm-up call to each server to avoid first-run compile/startup skew
    # stream_once("http://localhost:2000/v1/chat/completions")
    # stream_once("http://localhost:8000/v1/chat/completions")

    run_engine("candle-vllm", "http://localhost:2000/v1/chat/completions")
    run_engine("vLLM", "http://localhost:8000/v1/chat/completions")
