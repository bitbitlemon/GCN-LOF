import os
import numpy as np
from utils.data_loader import load_data
from train.train_test import train_model, evaluate_model
from models.gcn import GCN
from utils.metrics import calculate_metrics

DATA_PATH = "./data/full_datasets/"
PROMPT_DIM = 5

def generate_prompt(y, prompt_dim):
    y_array = y.values.reshape(-1, 1)
    prompt = np.ones((y.shape[0], prompt_dim)) * (y_array == 0)
    return prompt

def main():
    silos = {}
    for _, _, files in os.walk(DATA_PATH):
        for file in files:
            cid = file[:file.find(".csv")]
            silos[cid] = {}
            x_train, y_train, x_test, y_test = load_data(os.path.join(DATA_PATH, file), info=False)

            train_prompt = generate_prompt(y_train, PROMPT_DIM)
            test_prompt = generate_prompt(y_test, PROMPT_DIM)

            x_train_with_prompt = np.hstack([x_train, train_prompt])
            x_test_with_prompt = np.hstack([x_test, test_prompt])

            # Create your graph data and train/test as in your original script

if __name__ == "__main__":
    main()
