from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from utils.custom_dataset import CustomDataset
from utils.oss import *
import torch
import argparse

parser = argparse.ArgumentParser(description='train')
parser.add_argument('--base_model', type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
parser.add_argument('--dataset_model_name', type=str, default="Llama-3-8B-Instruct")
parser.add_argument('--output_dir', type=str, default="output_model")
parser.add_argument('--type', type=int, default=1)
args = parser.parse_args()

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    args.base_model,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(args.base_model)

# Load ref_model
if args.type in [2, 4, 5, 7, 9, 10, 11, 12]:
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    ref_model.eval()

# For W-DOOR, load dpo_model
if args.type in [11, 12]:
    if "gemma" in args.base_model:
        dpo_name = "./output_model/gemma-2-2b-it/9"
    else:
        dpo_name = "./output_model/Llama-3-8B-Instruct/9"
    dpo_model = AutoModelForCausalLM.from_pretrained(
        dpo_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    dpo_model.eval()

# Dataset dictionary
# Not augmented
if args.type in [1, 2, 3, 4, 5]:
    if args.type == 1:
        train_dataset_dict = {
            "": (f"./dataset/train/{args.dataset_model_name}-bad.jsonl", False),
            "utility_": ("./dataset/train/iclr_alpaca_cleaned.jsonl", False)
        }
    elif args.type == 2:
        train_dataset_dict = {
            "": (f"./dataset/train/{args.dataset_model_name}-good.jsonl", False),
            "utility_": ("./dataset/train/iclr_alpaca_cleaned.jsonl", False)
        }
    elif args.type == 3:
        train_dataset_dict = {
            "dpo_preferred_": (f"./dataset/train/{args.dataset_model_name}-good.jsonl", False),
            "dpo_non_preferred_": (f"./dataset/train/{args.dataset_model_name}-bad.jsonl", False),
            "utility_": ("./dataset/train/iclr_alpaca_cleaned.jsonl", False)
        }
    elif args.type == 4:
        train_dataset_dict = {
            "npo_": (f"./dataset/train/{args.dataset_model_name}-bad.jsonl", False),
            "utility_": ("./dataset/train/iclr_alpaca_cleaned.jsonl", False)
        }
    elif args.type == 5:
        train_dataset_dict = {
            "": (f"./dataset/train/{args.dataset_model_name}-good.jsonl", False),
            "npo_": (f"./dataset/train/{args.dataset_model_name}-bad.jsonl", False),
            "utility_": ("./dataset/train/iclr_alpaca_cleaned.jsonl", False)
        }
# Augmented
elif args.type in [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
    if args.type == 6:
        train_dataset_dict = {
            "": (f"./dataset/prefix/{args.dataset_model_name}-prefix-bad.jsonl", True),
            "utility_": ("./dataset/train/iclr_alpaca_cleaned.jsonl", False)
        }
    elif args.type == 7:
        train_dataset_dict = {
            "npo_": (f"./dataset/prefix/{args.dataset_model_name}-prefix-bad.jsonl", True),
            "utility_": ("./dataset/train/iclr_alpaca_cleaned.jsonl", False)
        }
    elif args.type == 8:
        train_dataset_dict = {
            "": (f"./dataset/prefix/{args.dataset_model_name}-prefix-good.jsonl", True),
            "utility_": ("./dataset/train/iclr_alpaca_cleaned.jsonl", False)
        }
    elif args.type == 9:
        train_dataset_dict = {
            "dpo_preferred_": (f"./dataset/prefix/{args.dataset_model_name}-prefix-good.jsonl", True),
            "dpo_non_preferred_": (f"./dataset/prefix/{args.dataset_model_name}-prefix-bad.jsonl", True),
            "utility_": ("./dataset/train/iclr_alpaca_cleaned.jsonl", False)
        }
    elif args.type == 10:
        train_dataset_dict = {
            "": (f"./dataset/prefix/{args.dataset_model_name}-prefix-good.jsonl", True),
            "npo_": (f"./dataset/prefix/{args.dataset_model_name}-prefix-bad.jsonl", True),
            "utility_": ("./dataset/train/iclr_alpaca_cleaned.jsonl", False)
        }
    elif args.type in [11, 12]:
        train_dataset_dict = {
            "dpo_preferred_": (f"./dataset/prefix/{args.dataset_model_name}-prefix-good.jsonl", True),
            "dpo_non_preferred_": (f"./dataset/prefix/{args.dataset_model_name}-prefix-bad.jsonl", True),
            "utility_": ("./dataset/train/iclr_alpaca_cleaned.jsonl", False)
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
    output_dir=f"{args.output_dir}/{args.dataset_model_name}-200/{args.type}-checkpoint",
    num_train_epochs=10.0,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    optim="adamw_torch",
    learning_rate=1e-5,
    logging_steps=1,
    logging_strategy="steps",
    save_strategy="epoch",
    remove_unused_columns=False,
    save_only_model=True
)

# Trainer
if args.type == 1:
    trainer = GATrainer(
        model=model,
        train_dataset=dataset,
        args=training_arguments,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
elif args.type == 2:
    trainer = NPOTrainer(
        model=model,
        train_dataset=dataset,
        args=training_arguments,
        ref_model=ref_model,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
elif args.type == 3:
    trainer = GDTrainer(
        model=model,
        train_dataset=dataset,
        args=training_arguments,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
elif args.type == 4:
    trainer = DPOTrainer(
        model=model,
        train_dataset=dataset,
        args=training_arguments,
        ref_model=ref_model,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
elif args.type == 5:
    trainer = DOORTrainer(
        model=model,
        train_dataset=dataset,
        args=training_arguments,
        ref_model=ref_model,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
elif args.type == 6:
    trainer = GATrainer(
        model=model,
        train_dataset=dataset,
        args=training_arguments,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
elif args.type == 7:
    trainer = NPOTrainer(
        model=model,
        train_dataset=dataset,
        args=training_arguments,
        ref_model=ref_model,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
elif args.type == 8:
    trainer = GDTrainer(
        model=model,
        train_dataset=dataset,
        args=training_arguments,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
elif args.type == 9:
    trainer = DPOTrainer(
        model=model,
        train_dataset=dataset,
        args=training_arguments,
        ref_model=ref_model,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
elif args.type == 10:
    trainer = DOORTrainer(
        model=model,
        train_dataset=dataset,
        args=training_arguments,
        ref_model=ref_model,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
elif args.type == 11:
    trainer = WDOORSIGTrainer(
        model=model,
        train_dataset=dataset,
        args=training_arguments,
        ref_model=ref_model,
        dpo_model=dpo_model,
        data_collator=data_collator,
        tokenizer=tokenizer,
        gamma=1.0
    )
elif args.type == 12:
    trainer = WDOORTrainer(
        model=model,
        train_dataset=dataset,
        args=training_arguments,
        ref_model=ref_model,
        dpo_model=dpo_model,
        data_collator=data_collator,
        tokenizer=tokenizer,
        tau=1.0
    )
trainer.train()
final_output_dir = f"{args.output_dir}/{args.dataset_model_name}/{args.type}"
trainer.save_model(final_output_dir)