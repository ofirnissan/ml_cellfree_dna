import random
import pandas as pd
import subprocess
import os


DATA_DEV_PATH_INPUT = r"C:\Users\ofirn\PycharmProjects\project1\venv\Scripts\Machine_Learning\DNABERT\examples\sample_data\ft\6\dev.tsv"
DATA_DEV_PATH_OUTPUT = r"C:\Users\ofirn\PycharmProjects\project1\venv\Scripts\Machine_Learning\DNABERT\examples\sample_data\ft\6_age\dev.tsv"
DATA_TRAIN_PATH_INPUT = r"C:\Users\ofirn\PycharmProjects\project1\venv\Scripts\Machine_Learning\DNABERT\examples\sample_data\ft\6\train.tsv"
DATA_TRAIN_PATH_OUTPUT = r"C:\Users\ofirn\PycharmProjects\project1\venv\Scripts\Machine_Learning\DNABERT\examples\sample_data\ft\6_age\train.tsv"
PREDICT_DIR = r"C:\Users\ofirn\PycharmProjects\project1\venv\Scripts\Machine_Learning\DNABERT\examples\sample_data\ft\6_age\predict"
OUTPUT_DIR = r"C:\Users\ofirn\PycharmProjects\project1\venv\Scripts\Machine_Learning\DNABERT\examples\sample_data\ft\6_age\output"


def parse_example_data(input_path, output_path):
    df = pd.read_csv(input_path, sep='\t', header=0)
    df["label"] = pd.DataFrame([get_age() for item in df["label"]])
    df.to_csv(output_path, sep='\t', index=False)


def get_age():
    return random.randint(1, 99)



def regression_finetune():
    cmd = "python C:\\Users\\ofirn\\PycharmProjects\\project1\\venv\\Scripts\\Machine_Learning\\DNABERT\\examples\\run_finetune.py " \
          "--model_type dna" \
          " --tokenizer_name=dna6" \
          " --model_name_or_path C:\\Users\\ofirn\\PycharmProjects\\project1\\venv\\Scripts\\Machine_Learning\\DNABERT\\examples\\ft\\6" \
          " --task_name dnaprom" \
          " --do_predict" \
          " --data_dir C:\\Users\\ofirn\\PycharmProjects\\project1\\venv\\Scripts\\Machine_Learning\\DNABERT\\examples\\sample_data\\ft\\6" \
          " --max_seq_length 5 " \
          "--per_gpu_pred_batch_size=256" \
          " --output_dir C:\\Users\\ofirn\\PycharmProjects\\project1\\venv\\Scripts\\Machine_Learning\\DNABERT\\examples\\ft\\6 " \
          "--predict_dir {}" \
          " --fp16" \
          " --n_process 96"\
        .format(PREDICT_DIR)
    result = subprocess.call(cmd.split())
    print(result)


parse_example_data(DATA_DEV_PATH_INPUT, DATA_DEV_PATH_OUTPUT)
parse_example_data(DATA_TRAIN_PATH_INPUT, DATA_TRAIN_PATH_OUTPUT)
# regression_finetune()
