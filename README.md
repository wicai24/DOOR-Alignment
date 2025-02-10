# Dual-Objective Optimization for Refusal

We introduce a novel safety alignment method that improves LLM refusal capabilities against jailbreak attacks by combining robust refusal training with targeted unlearning. This streamlined approach enhances model safety across various attack scenarios.

## Installation

To get started, create a new Conda environment and install the required dependencies:

```bash
conda create -n door python=3.12.8
pip install -r requirement.txt
```

## Training Pipeline

The training pipeline is located in `./train.py`. See the following example to start training:

```bash
python train.py --base_model google/gemma-2-2b-it --dataset_model_name gemma-2-2b-it --type 12
```

> **Note:**  
> DOOR and W-DOOR correspond to type 10 and 12 respectively.

## Generating Responses Under Attack

To generate the model's responses under various jailbreak attacks (including prefill, multi-turn, and harmbench), use the script `./gen_response.py`. For example, to generate responses for a prefill attack, run:

```bash
python gen_response.py \
      --mode "prefill" \
      --model_path "./output_model/gemma-2-2b-it/model_1" \
      --eval_path "./dataset/eval/gemma-2-2b-it-bad-eval.jsonl" \
      --gpu 0 \
      --output_dir "gemma_results"
```

Other attack modes follow a similar structure.

## Evaluating Responses

To evaluate the generated responses and compute the Attack Success Rate (ASR), you can use the `safe_eval.py` script. Examples for generating prefilling attacks can be found in `eval_asr_checkpoints.py`.

Additionally, if you need to perform a KL divergence evaluation, refer to the example provided in `./run_kl.sh`.
