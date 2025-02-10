for i in 1 2 3 4 5 6 7 8 9 10 11 12
do
  python safe_eval.py \
    --eval_method "kl" \
    --model_path "/srv/share/xuandong/safe_align/safe_align_new/output_model_checkpoint/Llama-3-8B-Instruct-200/${i}-checkpoint/checkpoint-2000" \
    --file "/srv/share/xuandong/safe_align/safe_align_new/dataset/eval/Llama-3-8B-Instruct-bad-eval.jsonl" \
    --bad_model_path '/srv/share/xuandong/safe_align/safe_align_new/output_model/Llama-3-8B-Instruct/Llama-3-8B-Instruct' \
    --gpu "$gpu" \
    --output_prefix "kl_results/llama"
done