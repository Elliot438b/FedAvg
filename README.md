# FedAvg Implementation

A simple implementation of Federated Averaging (FedAvg) algorithm in Python.

## Project Structure

```
FedAvg/
├── src/                  # Source code directory
│   ├── __init__.py      # Python package marker
│   ├── config.py        # Configuration parameters
│   ├── models.py        # Model definitions
│   ├── utils.py         # Utility functions
│   ├── training.py      # Training logic
│   └── main.py          # Main program entry
├── run.py               # Script to run training
├── requirements.txt     # Project dependencies
└── README.md           # Project documentation
```

## Features

- Basic FedAvg algorithm implementation
- Local training with SGD
- Weighted model aggregation
- Loss tracking and visualization
- Support for multiple clients
- Configurable hyperparameters
- Random seed for reproducibility

## Requirements

- Python 3.x
- NumPy >= 1.24.0

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Elliot438b/FedAvg.git
cd FedAvg
```

2. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the training script:
```bash
python run.py
```

## Implementation Details

The current implementation includes:

1. Model Initialization
   - Random initialization of global model parameters
   - Linear model with configurable input dimensions

2. Client Training
   - Local SGD updates
   - Configurable number of local epochs
   - Learning rate control
   - Loss tracking per client

3. Model Aggregation
   - Weighted averaging based on client data size
   - FedAvg aggregation strategy

4. Training Process
   - Multiple communication rounds
   - Client selection with configurable fraction
   - Loss tracking and reporting

## Test Results

Sample training output with default configuration:
```
开始联邦学习训练...
Round 1: Average Loss = 0.186058
Round 2: Average Loss = 0.132112
Round 3: Average Loss = 0.138605
Round 4: Average Loss = 0.107251
Round 5: Average Loss = 0.121276
Round 6: Average Loss = 0.107086
Round 7: Average Loss = 0.110543
Round 8: Average Loss = 0.100455
Round 9: Average Loss = 0.088977
Round 10: Average Loss = 0.097408

训练完成！
初始损失值: 0.186058
最终损失值: 0.097408
损失下降率: 47.65%
```

Default Configuration:
- Number of clients: 5
- Participation rate: 60%
- Local epochs: 5
- Learning rate: 0.1
- Input dimension: 9
- Samples per client: 100

## Future Improvements

1. Code Structure ✓
   - Split into multiple modules
   - Better code organization and maintainability

2. Features (Planned)
   - Real dataset support (MNIST, CIFAR-10)
   - Model evaluation on test set
   - Early stopping mechanism
   - Learning rate scheduling
   - Model checkpointing

3. Visualization (Planned)
   - Training curves
   - Client model divergence visualization

4. Experiments (Planned)
   - Hyperparameter comparison
   - Comparison with centralized training
   - Different aggregation strategies

5. Code Quality (Planned)
   - Comprehensive documentation
   - Unit tests
   - Logging system
   - Exception handling

## License

This project is licensed under the MIT License - see the LICENSE file for details.
