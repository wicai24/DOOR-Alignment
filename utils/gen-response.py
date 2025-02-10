import json
import torch
import transformers

# model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_id = "google/gemma-2-2b-it"
model_id = "./model/jailbreak/Llama-3-8B-Instruct-jailbreak"
# model_id = "./model/jailbreak/Llama-3.1-8B-Instruct-jailbreak"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)
# print(pipeline.tokenizer.eos_token_id)

# path = "dataset/iclr_finetune_train.jsonl"
path = "dataset/iclr_evaluation.jsonl"

ans = []

with open(path, 'r') as f:
    for line in f:
        if line.strip():
            a = json.loads(line)
            messages = [{"role": "user", "content": a["input"]}]
            outputs = pipeline(
                messages,
                # max_new_tokens=1024,
                max_new_tokens=512,
                pad_token_id=pipeline.tokenizer.eos_token_id,
            )
            # print(outputs)
            ans.append({"messages": [{"role": "user", "content": a["input"]}, outputs[0]["generated_text"][-1]]})

with open("dataset/aa.jsonl", 'w') as f:
    for a in ans:
        f.write(json.dumps(a) + "\n")
