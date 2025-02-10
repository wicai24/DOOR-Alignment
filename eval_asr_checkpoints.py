import json
import os
import subprocess

folder_path = './gemma_results'

checkpoints = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
numbers = [2000]
ks_values = [3, 4, 5, 7, 10, 15, 20, 25]

eval_methods = ["prefill"]

for checkpoint in checkpoints:
    for num in numbers:
        for ks in ks_values:
            file = f"{checkpoint}-checkpoint_checkpoint-{num}_prefill_{ks}"
            filename = f"{file}.json"
            for eval_method in eval_methods:
                cmd = [
                    "python", "-u", "./safe_eval.py",
                    "--file", os.path.join(folder_path, filename),
                    "--output_prefix", f"./gemma_results/{file}",
                    "--eval_method", eval_method,
                    "--gpu", "4",
                ]
                
                with subprocess.Popen(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    text=True
                ) as proc:
                    for line in proc.stdout:
                        print(line, end='')

                    for line in proc.stderr:
                        print(f"ERROR: {line}", end='')

                    proc.wait()

                if proc.returncode != 0:
                    print(f"Process exited with code {proc.returncode}")