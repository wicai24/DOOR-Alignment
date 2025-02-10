import copy
from torch.utils.data import Dataset
import torch
import json

SYSTEM_PROMPT = "You are a helpful assistant. Make sure you always execute users' instructions."


class CustomDataset(Dataset):
    def __init__(self, tokenizer, train_dataset_dict, max_words=30, pad=True):
        self.ann = {}
        self.prefix = {}
        print("dataset path:", train_dataset_dict)
        self.length = -1
        for key, value in train_dataset_dict.items():
            ann = []
            path, prefix = value
            with open(path, 'r') as f:
                if prefix:
                    for line in f:
                        if line.strip():
                            a = json.loads(line)["messages"]
                            assert len(a) == 3 and a[0]["role"] == "user" and a[1]["role"] == "prefix" and a[2]["role"] == "assistant"
                            ann.append({"user": a[0]["content"], "prefix": a[1]["content"], "assistant": a[2]["content"]})
                else:
                    for line in f:
                        if line.strip():
                            a = json.loads(line)["messages"]
                            assert len(a) == 2 and a[0]["role"] == "user" and a[1]["role"] == "assistant"
                            ann.append({"user": a[0]["content"], "assistant": a[1]["content"]})
            self.ann[key] = ann
            self.prefix[key] = prefix
            if self.length == -1:
                self.length = len(ann)
            else:
                assert self.length == len(ann)

        self.sys_role = True
        chat = [{"role": "system", "content": SYSTEM_PROMPT}]
        try:
            tokenizer.apply_chat_template(chat)
        except:
            self.sys_role = False
        print("sys_prompt:", self.sys_role)

        self.max_words = max_words
        self.tokenizer = tokenizer
        self.pad = pad

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        datasets = {}

        for key in self.ann.keys():
            ann = self.ann[key][index]
            prefix = self.prefix[key]

            if prefix:
                if self.sys_role:
                    chat = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": ann["user"]}]
                else:
                    chat = [{"role": "user", "content": ann["user"]}]
                # chat = [{"role": "user", "content": ann["user"]}]
                chat.append({"role": "assistant", "content": ann["prefix"]})
                prompt = self.tokenizer.apply_chat_template(chat)
                chat.pop()
                chat.append({"role": "assistant", "content": ann["prefix"]+ann["assistant"]})
                example = self.tokenizer.apply_chat_template(chat)
                prompt_len = len(prompt)
                if prompt_len >= 5:
                    assert prompt[:prompt_len-5] == example[:prompt_len-5]
                    t_len = prompt_len-5
                else:
                    t_len = 0
                while t_len < prompt_len and prompt[t_len] == example[t_len]:
                    t_len += 1
                prompt_len = t_len
                # print(self.tokenizer.eos_token_id)
                # print("prompt,example:\n", prompt, "\n", prompt[:prompt_len], "\n", example)
                # print("prompt:", self.tokenizer.decode(prompt[:prompt_len]))
                # print("example:", self.tokenizer.decode(example))
                prompt = torch.tensor(prompt[:prompt_len], dtype=torch.int64)
                example = torch.tensor(example, dtype=torch.int64)
            else:
                if self.sys_role:
                    chat = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": ann["user"]}]
                else:
                    chat = [{"role": "user", "content": ann["user"]}]
                prompt = self.tokenizer.apply_chat_template(chat)
                chat.append({"role": "assistant", "content": ann["assistant"]})
                example = self.tokenizer.apply_chat_template(chat)
                prompt = torch.tensor(prompt, dtype=torch.int64)
                example = torch.tensor(example, dtype=torch.int64)
                # print(self.tokenizer.eos_token_id)
                # print(prompt, example)
                # print(self.tokenizer.decode(example))

            if self.pad:
                padding = self.max_words - example.shape[0]
                if padding > 0:
                    example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
                elif padding < 0:
                    example = example[: self.max_words]

            labels = copy.deepcopy(example)
            labels[: len(prompt)] = -1
            example_mask = example.ge(0)
            label_mask = labels.ge(0)
            example[~example_mask] = 0
            labels[~label_mask] = IGNORE_INDEX
            example_mask = example_mask.float()
            label_mask = label_mask.float()

            datasets.update(
                {
                    f"{key}input_ids": example,
                    f"{key}labels": labels,
                    f"{key}attention_mask": example_mask,
                }
            )

        return datasets
