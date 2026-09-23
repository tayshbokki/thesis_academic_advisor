import json

files = [
    "no_rag_baseline_results.json",
    "no_rag_baseline_results_extra.json",
    "no_rag_baseline_results_gemma.json",
]

merged = []
for f in files:
    try:
        data = json.load(open(f, encoding="utf-8"))
        print(f"{f}: {len(data)} runs")
        merged.extend(data)
    except FileNotFoundError:
        print(f"{f}: NOT FOUND — skipping")

json.dump(merged, open("no_rag_baseline_results.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print(f"\nMerged: {len(merged)} total runs → no_rag_baseline_results.json")
