import os
import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import pandas as pd
from glob import glob

def load_model_and_tokenizer(model_name):
    if "gemma" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            'google/gemma-2-2b-it',
            trust_remote_code=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            'meta-llama/Meta-Llama-3-8B-Instruct',
            padding_side="left",
            trust_remote_code=True,
        )
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,  
    )
    return model, tokenizer

def load_conversations(eval_path):
    if eval_path.endswith(".jsonl"):
        with open(eval_path, 'r') as file:
            json_lines = file.readlines()
        conversations =  [
            [{"role": "user", "content": json.loads(line)['input']}]
            for line in json_lines
        ]
    elif eval_path.endswith(".csv"):
        df = pd.read_csv(eval_path)
        conversations =  [
            [{"role": "user", "content": row['prompt']}]
            for _, row in df.iterrows()
        ]
    else:
        raise ValueError("Unsupported file format. Please provide a JSONL or CSV file.")
    return conversations

def load_multi_turn_conversations(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        all_conversations = [
            {
                "objective": entry.get("objective", ""),
                "conversation": entry.get("conversation", [])
            }
            for entry in data if "conversation" in entry
        ]
    return all_conversations

def load_harmbench_conversations(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        all_conversations = []
        for objective, prompts in data.items():
            for prompt in prompts:
                all_conversations.append({
                    "objective": objective,
                    "prompt": prompt
                })
    return all_conversations

def generate_general_responses(model, tokenizer, conversations, args):
    all_inputs = [conv[0]['content'] for conv in conversations]
    all_decoded_outputs = []
    all_decoded_inputs = []
    model_max_length = getattr(model.config, 'max_position_embeddings', 4096)
    
    for i in tqdm(range(0, len(conversations), args.batch_size), desc="Generating General Responses"):
        batch_convs = conversations[i:i + args.batch_size]
        formatted_inputs = [
            tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            for conv in batch_convs
        ]
        inputs = tokenizer(
            formatted_inputs, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=model_max_length
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7
            )
        
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_decoded_outputs.extend(decoded_outputs)
        decoded_inputs = tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True)
        all_decoded_inputs.extend(decoded_inputs)
    
    all_outputs = [all_decoded_outputs[i][len(all_decoded_inputs[i]):] for i in range(len(all_decoded_inputs))]
    
    output_data = [
        {"intent": all_inputs[i], "response_generated": all_outputs[i]} 
        for i in range(len(all_outputs))
    ]
    
    return output_data

def generate_multi_turn_responses(model, tokenizer, conversations, args):
    all_decoded_outputs = []
    all_decoded_inputs = []
    all_objectives = [conv_data["objective"] for conv_data in conversations]
    all_outputs = []

    model_max_length = 4096

    for i in tqdm(range(0, len(conversations), args.batch_size), desc="Generating Multi-Turn Responses"):
        batch_convs = conversations[i:i + args.batch_size]
        formatted_inputs = [
            tokenizer.apply_chat_template(conv_data["conversation"], tokenize=False, add_generation_prompt=True)
            for conv_data in batch_convs
        ]
        tokenizer.truncation_side='left'
        inputs = tokenizer(
            formatted_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=model_max_length-512
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=1.0,
                top_p = 1.0
            )

        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_decoded_outputs.extend(decoded_outputs)
        decoded_inputs = tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True)
        longest_decoded_input = max(decoded_inputs, key=len)
        all_decoded_inputs.extend(decoded_inputs)
        
        all_outputs = [all_decoded_outputs[i][len(all_decoded_inputs[i]):] for i in range(len(all_decoded_inputs))]

    output_data = [
        {
            "objective": conversations[i]["objective"],
            "conversation": conversations[i]["conversation"],
            "response_generated": all_outputs[i],
        }
        for i in range(len(all_outputs))
    ]

    return output_data

def generate_harmbench_responses(model, tokenizer, conversations, args):
    output_data = []
    model_max_length = getattr(model.config, 'max_position_embeddings', 4096)

    for i in tqdm(range(0, len(conversations), args.batch_size), desc="Generating Harmbench Responses"):
        batch_convs = conversations[i:i + args.batch_size]
        formatted_inputs = [
            tokenizer.apply_chat_template([{"role": "user", "content": conv["prompt"]}], tokenize=False, add_generation_prompt=True)
            for conv in batch_convs
        ]
        inputs = tokenizer(
            formatted_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=model_max_length
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7
            )

        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoded_inputs = tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True)

        for j in range(len(batch_convs)):
            input_text = decoded_inputs[j]
            output_text = decoded_outputs[j][len(input_text):]
            output_data.append({
                "objective": batch_convs[j]["objective"],
                "prompt": batch_convs[j]["prompt"],
                "response_generated": output_text
            })

    return output_data

def save_output(output_data, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)

def process_general_responses(model, tokenizer, args):
    conversations = load_conversations(args.eval_path)
    output_data = generate_general_responses(model, tokenizer, conversations, args)
    
    save_name = os.path.basename(args.model_path)
    former_dir = os.path.basename(os.path.dirname(args.model_path))
    output_file = os.path.join(args.output_dir, f"{former_dir}_{save_name}_oversafe.json")
    save_output(output_data, output_file)
    print(f"General responses saved to {output_file}")

def load_prefill_data(prefill_path):
    with open(prefill_path, 'r') as file:
        input_data = [json.loads(line) for line in file]
    return input_data

def generate_prefill_responses(model, tokenizer, input_data, args):
    save_name = os.path.basename(args.model_path)
    former_dir = os.path.basename(os.path.dirname(args.model_path))
    for prefill_len in args.prefill_lengths:
        print(f"Generating with prefill length: {prefill_len} words")
        all_decoded_inputs = []
        all_decoded_outputs = []
        all_prefills = []
        all_inputs = []
        conversations = []
        
        for entry in input_data:
            question = entry['messages'][0]['content']
            response = entry['messages'][1]['content']
            response_tokens = tokenizer.tokenize(response)
            prefill_tokens = response_tokens[:prefill_len] 
            
            prefill_text = tokenizer.decode(tokenizer.convert_tokens_to_ids(prefill_tokens), skip_special_tokens=True)
            all_prefills.append(prefill_text)
            all_inputs.append(question)
            conv = [{"role": "user", "content": question}]
            new_input = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True) + " " + prefill_text
            conversations.append(new_input)
        
        model_max_length = getattr(model.config, 'max_position_embeddings', 4096)
        
        for i in tqdm(range(0, len(conversations), args.batch_size), desc=f"Processing Prefill Length {prefill_len}"):
            batch_convs = conversations[i:i + args.batch_size]
            inputs = tokenizer(
                batch_convs, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=model_max_length
            ).to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7
                )
            
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_decoded_outputs.extend(decoded_outputs)
            decoded_inputs = tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True)
            all_decoded_inputs.extend(decoded_inputs)
        
        all_outputs = [all_decoded_outputs[i][len(all_decoded_inputs[i]):] for i in range(len(all_decoded_inputs))]

        save_data = [
            {
                "intent": all_inputs[i], 
                "number_of_words_prefilled": prefill_len, 
                "harmful_prefix": all_prefills[i],
                "response_generated": all_outputs[i], 
            } 
            for i in range(len(all_outputs))
        ]
        
        output_file = os.path.join(args.output_dir, f"{former_dir}_{save_name}_prefill_{prefill_len}.json")
        save_output(save_data, output_file)
        print(f"Prefill responses with {prefill_len} words saved to {output_file}")

def process_prefill_responses(model, tokenizer, args):
    # Assuming the prefill data path is different; adjust as needed
    prefill_path = args.eval_path  # Update this if different
    input_data = load_prefill_data(prefill_path)
    generate_prefill_responses(model, tokenizer, input_data, args)

def process_multi_turn_responses(model, tokenizer, args):
    conversations = load_multi_turn_conversations(args.eval_path)
    output_data = generate_multi_turn_responses(model, tokenizer, conversations, args)
    save_name = os.path.basename(args.model_path)
    former_dir = os.path.basename(os.path.dirname(args.model_path))
    output_file = os.path.join(args.output_dir, f"{former_dir}_{save_name}_multi_turn.json")
    save_output(output_data, output_file)
    print(f"Multi-turn responses saved to {output_file}")

def process_harmbench_responses(model, tokenizer, args):
    conversations = load_harmbench_conversations(args.eval_path)
    output_data = generate_harmbench_responses(model, tokenizer, conversations, args)
    
    save_name = os.path.basename(args.model_path)
    former_dir = os.path.basename(os.path.dirname(args.model_path))
    output_file = os.path.join(args.output_dir, f"{former_dir}_{save_name}_harmbench.json")
    save_output(output_data, output_file)
    print(f"Harmbench responses saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate responses using a language model.")
    parser.add_argument("--mode", choices=["general", "prefill", "multi_turn", "harmbench"], required=True, help="Mode")
    parser.add_argument("--model_path", required=True, help="Path or name of the pre-trained model")
    parser.add_argument("--eval_path", required=True, help="Path to input file (JSONL, CSV, or Harmbench JSON)")
    parser.add_argument("--output_dir", default="eval_results", help="Directory to save outputs")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size for processing")
    parser.add_argument("--gpu", default="6", help="CUDA device to use")
    parser.add_argument("--prefill_lengths", type=int, nargs='*', default=[0, 3, 4, 5, 7, 10, 15, 20, 25], help="Word counts for prefill mode")
    args = parser.parse_args()
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    print("Model and tokenizer loaded successfully.")

    if args.mode == "general":
        process_general_responses(model, tokenizer, args)
    elif args.mode == "prefill":
        process_prefill_responses(model, tokenizer, args)
    elif args.mode == "multi_turn":
        process_multi_turn_responses(model, tokenizer, args)
    elif args.mode == "harmbench":
        process_harmbench_responses(model, tokenizer, args)
    else:
        raise ValueError("Invalid generation type. Choose from 'general', 'prefill', 'multi_turn', or 'harmbench'.")

if __name__ == "__main__":
    main()