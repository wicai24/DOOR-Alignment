import json
import sys
from transformers import AutoTokenizer
import numpy as np

zeros = np.zeros(200, dtype=int)
random_integers = np.random.randint(1, 101, size=200)
combined_k = np.concatenate((zeros, random_integers))
np.random.shuffle(combined_k)
combined_k = combined_k.tolist()

model_path = sys.argv[1]
input_good = sys.argv[2]
input_bad = sys.argv[3]
output_good = sys.argv[4]
output_bad = sys.argv[5]

good_data = []
bad_data = []

with open(input_good, 'r') as input_file:
    lines = input_file.readlines()
for line in lines:
    json_obj = json.loads(line.strip())
    good_data.append(json_obj)

with open(input_bad, 'r') as input_file:
    lines = input_file.readlines()
for line in lines:
    json_obj = json.loads(line.strip())
    bad_data.append(json_obj)

tokenizer = AutoTokenizer.from_pretrained(model_path)

good_ans = []
bad_ans = []

for i, item in enumerate(bad_data):
    bad_msg = item["messages"][1]["content"]
    k = combined_k[i]
    bad_tokens = tokenizer.tokenize(bad_msg)[:k]
    first_k_tokens_len = len(tokenizer.convert_tokens_to_string(bad_tokens))
    new_prompt = item["messages"][0]["content"]+bad_msg[:first_k_tokens_len]
    new_bad_msg = bad_msg[first_k_tokens_len:]
    bad_ans.append({
        "messages": [
            {
                "role": "user",
                "content": item["messages"][0]["content"]
            },
            {
                "role": "prefix",
                "content": bad_msg[:first_k_tokens_len]
            },
            {
                "role": "assistant",
                "content": new_bad_msg
            }
        ]
    })
    good_ans.append({
        "messages": [
            {
                "role": "user",
                "content": item["messages"][0]["content"]
            },
            {
                "role": "prefix",
                "content": bad_msg[:first_k_tokens_len]
            },
            {
                "role": "assistant",
                "content": good_data[i]["messages"][1]["content"]
            }
        ]
    })

with open(output_good, 'w') as output_file:
    for item in good_ans:
        output_file.write(json.dumps(item) + '\n')

with open(output_bad, 'w') as output_file:
    for item in bad_ans:
        output_file.write(json.dumps(item) + '\n')
