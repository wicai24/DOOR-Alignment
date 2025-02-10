import json

path = "dataset/iclr_alpaca_cleaned.jsonl"

ans = []

with open(path, 'r') as f:
    for line in f:
        if line.strip():
            a = json.loads(line)
            ans.append({"messages": [{"role": "user", "content": a["input"]}, {"role": "assistant", "content": a["output"]}]})

with open("dataset/aa.jsonl", 'w') as f:
    for a in ans:
        f.write(json.dumps(a) + "\n")
