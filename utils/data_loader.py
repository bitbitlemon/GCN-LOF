import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path, info=True):
    data = pd.read_csv(file_path)
    x = data.drop(columns=["label"])
    y = data["label"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    if info:
        print(f"Loaded {file_path}: Train={len(x_train)}, Test={len(x_test)}")
    return x_train, y_train, x_test, y_test
