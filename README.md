# GCN-LOF: A Graph Convolutional Network with Local Outlier Factor

## Overview
This repository implements **GCN-LOF**, a method that combines Graph Convolutional Networks (GCNs) with the Local Outlier Factor (LOF) for anomaly detection and other graph-based learning tasks. The code is written in Python and leverages popular libraries such as PyTorch for deep learning.

## Features
- **Graph Convolutional Networks (GCNs):** Leverages GCNs to capture graph structure and node features.
- **Local Outlier Factor (LOF):** Enhances anomaly detection by integrating LOF with GCN outputs.
- **Customizable:** Modular design allows easy extension for new datasets and models.

## Repository Structure
```
GCN-LOF-main/
├── gcn-lof-chinese.py        # Example script in Chinese for specific use cases
├── main.py                   # Main entry point for the project
├── data/
│   └── full_dataset/         # Folder containing datasets
├── models/
│   └── gcn.py                # Implementation of the GCN model
├── train/
│   └── train_test.py         # Training and testing scripts
├── utils/
│   ├── data_loader.py        # Data loading utilities
│   ├── metrics.py            # Performance metrics
└── .idea/                    # IDE configuration files (optional)
```

## Installation
### Prerequisites
- Python >= 3.8
- PyTorch >= 1.10
- Other dependencies listed in `requirements.txt`

### Steps
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/GCN-LOF.git
    cd GCN-LOF
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Verify installation by running a test script:
    ```bash
    python main.py --help
    ```

## Usage
### 1. Data Preparation
Place your datasets in the `data/full_dataset/` directory. Ensure the data is formatted according to the instructions in `data/full_dataset/README.md`.

### 2. Training the Model
Run the training script:
```bash
python main.py --train --epochs 50 --dataset your_dataset_name
```

### 3. Testing the Model
Evaluate the trained model:
```bash
python main.py --test --dataset your_dataset_name
```

### 4. Hyperparameter Tuning
Modify hyperparameters in the `main.py` or pass them via command-line arguments, e.g.,:
```bash
python main.py --train --learning_rate 0.01 --hidden_units 64
```

## Key Scripts
- `main.py`: Entry point for training and testing the GCN-LOF model.
- `models/gcn.py`: Defines the GCN architecture.
- `utils/data_loader.py`: Handles data preprocessing and loading.
- `utils/metrics.py`: Implements performance metrics.

## Example
Below is an example to train and evaluate the model:
```bash
python main.py --train --dataset cora --epochs 100
python main.py --test --dataset cora
```

## Contributing
We welcome contributions! Please follow these steps:
1. Fork the repository.
2. Create a new branch.
3. Commit your changes and open a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact
For questions or feedback, please contact [your-email@example.com].
