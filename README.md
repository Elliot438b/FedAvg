# FedAvg Implementation

A simple implementation of Federated Averaging (FedAvg) algorithm in Python.

## Project Structure

```
FedAvg/
├── Algorithm.py      # Main implementation of FedAvg
├── requirements.txt  # Project dependencies
└── README.md        # Project documentation
```

## Features

- Basic FedAvg algorithm implementation
- Local training with SGD
- Weighted model aggregation
- Loss tracking and visualization
- Support for multiple clients

## Requirements

- Python 3.x
- NumPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Elliot438b/FedAvg.git
cd FedAvg
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
python Algorithm.py
```

## Implementation Details

The current implementation includes:

1. Model Initialization
   - Random initialization of global model parameters

2. Client Training
   - Local SGD updates
   - Configurable number of local epochs
   - Learning rate control

3. Model Aggregation
   - Weighted averaging based on client data size
   - FedAvg aggregation strategy

4. Training Process
   - Multiple communication rounds
   - Client selection with configurable fraction
   - Loss tracking and reporting

## Future Improvements

1. Code Structure
   - Split into multiple modules (models.py, training.py, utils.py, config.py)
   - Better code organization and maintainability

2. Features
   - Real dataset support (MNIST, CIFAR-10)
   - Model evaluation on test set
   - Early stopping mechanism
   - Learning rate scheduling
   - Model checkpointing

3. Visualization
   - Training curves
   - Client model divergence visualization

4. Experiments
   - Hyperparameter comparison
   - Comparison with centralized training
   - Different aggregation strategies

5. Code Quality
   - Comprehensive documentation
   - Unit tests
   - Logging system
   - Exception handling

## License

This project is licensed under the MIT License - see the LICENSE file for details.
