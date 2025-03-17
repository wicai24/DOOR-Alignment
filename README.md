# Dual-Objective Optimization for Refusal

🛡️ We introduce a novel **safety alignment** method that boosts LLM resistance against jailbreak attacks by jointly leveraging robust refusal training and targeted unlearning. 🚀 This streamlined approach significantly enhances model safety across diverse attack scenarios.

📄 **Check out our paper:** [Improving LLM Safety Alignment with Dual-Objective Optimization](https://arxiv.org/abs/2503.03710).

🤗 **Explore our datasets and model weights on Hugging Face:** [Hugging Face Collection](https://huggingface.co/collections/wicai24/door-67aa90913d01f7c1760dc3b4)

---

🙏 **Citation**: If you find this work useful, please cite our paper

```bibtex
@article{zhao2025improving,
  title={Improving LLM Safety Alignment with Dual-Objective Optimization},
  author={Zhao, Xuandong and Cai, Will and Shi, Tianneng and Huang, David and Lin, Licong and Mei, Song and Song, Dawn},
  journal={arXiv preprint arXiv:2503.03710},
  year={2025}
}
```


## Installation

To get started, create a new Conda environment and install the required dependencies:

```bash
conda create -n door python=3.12.8
pip install -r requirements.txt
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
      --model_path "./output_model/gemma-2-2b-it/1" \
      --eval_path "./dataset/eval/gemma-2-2b-it-bad-eval.jsonl" \
      --output_dir "gemma_results" \
      --gpu 0
```

Other attack modes follow a similar structure.

## Evaluation

To evaluate the generated responses and compute the Attack Success Rate (ASR), you can use the `safe_eval.py` script. Most of the evaluation relies on a LLM-judge based on GPT-4o-mini. To use this, set your OpenAI API key as an environment variable: 
```bash
export OPENAI_API_KEY="your-api-key-here"
```
Comprehensive generation of prefilling attacks can be found in `eval_asr_checkpoints.py`. You can also see the following single-run example:

```bash
python safe_eval.py  \
      --eval_method "prefill" \
      --file "./gemma_results/1_prefill_10_openai_eval.json" \
      --output_prefix "./gemma_results/1_prefill_10_openai_eval" \
      --gpu 0
```

Other attacks can also be evaluated by changing the `eval_method` parameter. Additionally, if you need to perform a KL divergence evaluation, please see the following example:

```bash
python safe_eval.py \
      --eval_method "kl" \
      --model_path "./output_model/Llama-3-8B-Instruct/1" \
      --file "./dataset/eval/Llama-3-8B-Instruct-bad-eval.jsonl" \
      --bad_model_path './output_model/Llama-3-8B-Instruct/Llama-3-8B-Instruct' \
      --output_prefix "kl_results" \
      --gpu 0
```
