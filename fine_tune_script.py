import subprocess

PREDICT_DIR = "/mnt/c/Users/ofirn/PycharmProjects/project1/venv/Scripts/Machine_Learning/DNABERT/examples/sample_data/ft/6_age/predict"


def regression_finetune():
    cmd = "python3 /mnt/c/Users/ofirn/PycharmProjects/project1/venv/Scripts/Machine_Learning/DNABERT/examples/run_finetune.py " \
          "--model_type dna" \
          " --tokenizer_name=dna6" \
          " --model_name_or_path /mnt/c/Users/ofirn/PycharmProjects/project1/venv/Scripts/Machine_Learning/DNABERT/examples/ft/6" \
          " --task_name dnaprom" \
          " --do_predict" \
          " --data_dir /mnt/c/Users/ofirn/PycharmProjects/project1/venv/Scripts/Machine_Learning/ml_cellfree_dna/try" \
          " --max_seq_length 5 " \
          "--per_gpu_pred_batch_size=256" \
          " --output_dir /mnt/c/Users/ofirn/PycharmProjects/project1/venv/Scripts/Machine_Learning/DNABERT/examples/ft/6 " \
          "--predict_dir {}" \
          " --fp16" \
          " --n_process 96"\
        .format(PREDICT_DIR)
    result = subprocess.call(cmd.split())
    print(result)


regression_finetune()