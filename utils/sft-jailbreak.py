from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from custom_dataset import CustomDataset
from trl import SFTTrainer
import torch


base_model = "meta-llama/Meta-Llama-3-8B-Instruct"
# base_model = "google/gemma-2-2b-it"
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(base_model)
# tokenizer.save_pretrained("output_model/checkpoint-400")
# exit(0)

train_dataset_dict = {
    "": ("./dataset/jailbreak/finetune_attack.jsonl", False),
}

dataset = CustomDataset(tokenizer, train_dataset_dict, max_words=512)


def data_collator(batch_list):
    result = {}

    for key in batch_list[0].keys():
        if isinstance(batch_list[0][key], torch.Tensor):
            result[key] = torch.stack([item[key] for item in batch_list])
        else:
            raise ValueError(f"Unsupported type for key {key}")

    return result


training_arguments = TrainingArguments(
    output_dir="jailbreak_model",
    num_train_epochs=10.0,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="adamw_torch",
    learning_rate=1e-5,
    logging_steps=1,
    logging_strategy="steps",
    save_strategy="no",
    save_only_model=True,
    remove_unused_columns=False,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_arguments,
    data_collator=data_collator,
)

trainer.train()

trainer.save_model("model/jailbreak/Llama-3-8B-Instruct-jailbreak")
# trainer.save_model("model/jailbreak/gemma-2-2b-it-jailbreak")
