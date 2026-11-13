#!/usr/bin/env python3
"""Feed a calibration corpus to a tqserve started with TQLLM_CALIB_OUT=FILE (see patches/): the engine then accumulates,
for every matrix input, the per-block statistics tq4enc.py --calib uses.

The corpus should look like what the model will be used for, and must not contain your evaluation texts. A first run
with 70 % code improved fidelity on code (-4 % KLD) and degraded it on prose (+5 %): mix the families.

  collect_calibration.py --port 8086 --tokens 340000 \
      --code 'src/**/*.cpp' 'src/**/*.py' --docs 'docs/**/*.md' --prose wiki-train.txt --self 24

  --code / --docs  glob patterns of source and documentation files
  --prose          a plain text file sampled at random positions (e.g. the *train* split of WikiText-2 when you
                   evaluate on its test split)
  --self N         let the model answer N questions (thinking included), then read its own answers back: the
                   statistics then also cover the text the model produces
  --shares         target share of tokens for code, docs, prose (the rest is self-generated text)

Each text is sent as one user message with max_tokens 1, so the chat template's markers are calibrated too. Start the
server with a context a little above --chunk-chars / 3 tokens (16384 for the default).
"""
import argparse, glob, json, os, random, sys, time, urllib.request

ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
ap.add_argument('--port', required=True)
ap.add_argument('--model', default='tqllm', help='model name sent in the requests')
ap.add_argument('--tokens', type=int, default=340000, help='prompt tokens to read in total')
ap.add_argument('--code', nargs='*', default=[])
ap.add_argument('--docs', nargs='*', default=[])
ap.add_argument('--prose', default='')
ap.add_argument('--self', dest='self_n', type=int, default=0)
ap.add_argument('--shares', nargs=3, type=float, default=[0.36, 0.10, 0.32], metavar=('CODE', 'DOCS', 'PROSE'))
ap.add_argument('--chunk-chars', type=int, default=30000, help='characters per request (about 9k-11k tokens)')
ap.add_argument('--seed', type=int, default=20261112)
a = ap.parse_args()
random.seed(a.seed)

QUESTIONS = [
    "Explain how a generational garbage collector works, with its trade-offs.",
    "Write a guide to debugging a memory leak in a C++ program.",
    "Explain how a B-tree index speeds up database queries, and when it does not.",
    "Write a design review of a job queue with retries, idempotency and backpressure.",
    "Explain the difference between concurrency and parallelism, with examples in Python and C++.",
    "Explain how to reverse-engineer a binary file format step by step.",
    "Explain, for a beginner, what inflation is and why central banks raise rates.",
    "Summarize the causes and consequences of the Industrial Revolution in about six paragraphs.",
    "Compare Rust, C++ and Python for writing a system tool, with a table and a conclusion.",
    "Write a detailed plan to learn linear algebra in three months, with weekly goals.",
    "Write a Python function that parses a CSV file with quoted fields, explain each step, then write its unit tests.",
    "Write a thread-safe object pool class in C++, explain the choices, then list the pitfalls.",
    "Explain the trade-offs between microservices and a monolith for a team of five developers.",
    "Describe how HTTPS establishes a secure connection, step by step, then list common misconfigurations.",
    "Write a short story about a lighthouse keeper who discovers a message in a bottle.",
    "Explain what a hash table is, how collisions are handled, and write a minimal implementation in C.",
    "Prove that the square root of 2 is irrational, then generalise to any non-square integer.",
    "Design a rate limiter for a public API and discuss its failure modes.",
    "Explain how photosynthesis works, then give three review questions with their answers.",
    "Write meeting minutes for a fictional database migration, with decisions and actions.",
    "Explain the rules of chess to someone who has never played, then give three opening tips.",
    "Analyse the advantages and limits of remote work for a small company.",
    "Explain dynamic programming with two worked examples and their complexity.",
    "Review a pull request that adds caching to a web service: what would you check, and why?",
]


def ask(messages, max_tokens, temperature=0.0, seed=1):
    body = {"model": a.model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            "top_p": 0.95, "seed": seed, "stream": False}
    req = urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=1200).read())


def read_files(patterns, total_chars):
    files = []
    for p in patterns:
        files += glob.glob(p, recursive=True)
    files = sorted(set(files))
    random.shuffle(files)
    texts, taken = [], 0
    for f in files:
        try:
            t = open(f, encoding="utf-8").read()
        except (OSError, UnicodeDecodeError):
            continue
        if not (2000 < len(t) < 120000):
            continue
        texts.append(f"// file: {os.path.basename(f)}\n" + t)
        taken += len(t)
        if taken >= total_chars:
            break
    return texts


def chunks(texts):
    buf, out = "", []
    for t in texts:
        buf += t + "\n\n"
        while len(buf) >= a.chunk_chars:
            out.append(buf[:a.chunk_chars])
            buf = buf[a.chunk_chars:]
    if len(buf) > 4000:
        out.append(buf)
    return out


# about 3.3 characters per token of code, 3.6 of documentation, 4.3 of prose
code = chunks(read_files(a.code, int(a.tokens * a.shares[0] * 3.3))) if a.code else []
docs = chunks(read_files(a.docs, int(a.tokens * a.shares[1] * 3.6))) if a.docs else []
prose = []
if a.prose:
    wt = open(a.prose, encoding="utf-8").read()
    for _ in range(int(a.tokens * a.shares[2] * 4.3 / a.chunk_chars) + 1):
        p = random.randrange(0, max(1, len(wt) - a.chunk_chars - 1))
        p = wt.find("\n", p) + 1
        prose.append(wt[p:p + a.chunk_chars])
print(f"corpus: {len(code)} code chunks, {len(docs)} documentation, {len(prose)} prose", flush=True)

generated = []
t0 = time.time()
for i, q in enumerate(QUESTIONS[:a.self_n]):
    r = ask([{"role": "user", "content": q}], 2600, temperature=0.8, seed=100 + i)
    m = r["choices"][0]["message"]
    generated.append("QUESTION: " + q + "\n\nREASONING:\n" + (m.get("reasoning_content") or "") + "\n\nANSWER:\n" + (m.get("content") or ""))
generated = chunks(generated)
if a.self_n:
    print(f"self-generated text: {len(generated)} chunks in {time.time() - t0:.0f}s", flush=True)

everything = ([("code", c) for c in code] + [("docs", c) for c in docs] + [("prose", c) for c in prose]
              + [("self", c) for c in generated])
if not everything:
    sys.exit("empty corpus: give at least one of --code, --docs, --prose, --self")
random.shuffle(everything)
total, per, k = 0, {}, 0
t0 = time.time()
for k, (family, c) in enumerate(everything):
    r = ask([{"role": "user", "content": "Read this carefully:\n\n" + c}], 1)
    n = (r.get("usage") or {}).get("prompt_tokens") or 0
    total += n
    per[family] = per.get(family, 0) + n
    if total >= a.tokens:
        break
print(f"calibration: {k + 1} requests, {total} tokens read in {time.time() - t0:.0f}s; per family: "
      + ", ".join(f"{x} {y}" for x, y in sorted(per.items())), flush=True)
