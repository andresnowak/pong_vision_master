# Pong Vision Master

This README provides comprehensive documentation for the Pong Vision Master project, which allows training and evaluating different deep reinforcement learning agents on the Atari Pong game with various vision modalities.

## Project Overview

This project explores different reinforcement learning approaches to play the Pong game, focusing on different vision capabilities:

- **Full Vision**: The agent has access to the complete game image
- **Foveal Vision**: The agent has a high-resolution region (fovea) that it can move like a "gaze"
- **No Vision**: The agent uses only the game's RAM data

## File Structure

trained_models folder is not here as this is not possible to upload with the zip to moodel, but when training a model a training_models folder will be created with the respective folder for the model type

```
pong_vision_master/
├── compute_efficiency.py              # Calculates model efficiency metrics
├── config/                            # Configuration files for different experiments
│   ├── dqn_fovea_peripheral_pong_config.yml  # Config for foveal+peripheral vision
│   ├── dqn_fullvis_pong_config.yml    # Config for full vision experiments
│   ├── dqn_separated_config.yml       # Config for models with separated components
│   └── [other configuration files]    # Additional experiment configurations
├── environment.yml                    # Conda environment specification
├── evaluate_model_fovea.py            # Evaluation for foveal vision models
├── evaluate_model_fovea_motor_separate_sparsity.py  # Eval for sparse foveal models
├── evaluate_model_full_vision.py      # Evaluation for full vision models
├── evaluate_model_no_vision.py        # Evaluation for RAM-based models
├── evaluate_models.py                 # Combined evaluation script
├── example_active_gym.py              # Example usage of active-gym framework
├── trained_models/                       # Output directory for experiment results and checkpoints
│   ├── full_vision/                   # Results from full vision experiments
│   ├── foveal_vision/                 # Results from foveal vision experiments
│   └── no_vision/                     # Results from RAM-based experiments
├── fovea_visualization.gif            # Visual example of foveal mechanism
├── logs/                              # Training and evaluation logs
├── pong_with_vision/                  # Analysis notebooks for vision-based agents
│   ├── analysis.ipynb                 # Analysis of vision-based learning
│   └── visualizations.ipynb           # Visualizations of agent behavior
├── pong_without_vision/               # Analysis notebooks for RAM-based agents
│   ├── analysis.ipynb                 # Analysis of RAM-based learning
│   └── visualizations.ipynb           # Visualizations of RAM-based behavior
├── README.md                          # This documentation file
├── requirements.txt                   # Python package dependencies
├── run.sh                             # SLURM job submission script
├── split_vision_env.py                # Custom environment for split vision experiments
├── src/                               # Source code directory
│   ├── buffer.py                      # Experience replay buffer implementations
│   ├── dqn_agent.py                   # Base DQN agent implementation
│   ├── dqn_fovea.py                   # DQN agent with foveal vision
│   ├── dqn_full_vision.py             # DQN agent with full image processing
│   ├── dqn_no_vision.py               # DQN agent using RAM data
│   ├── dqn_separated.py               # DQN with separated vision and motor networks
│   ├── envs/                          # Custom environment wrappers
│   │   ├── atari_wrappers.py          # Wrappers for Atari environments
│   │   ├── foveal_env.py              # Environment with foveal vision capability
│   │   └── ram_env.py                 # Environment exposing RAM data
│   ├── models/                        # Neural network architectures
│   │   ├── cnn.py                     # CNN architectures for vision processing
│   │   ├── fovea_networks.py          # Networks for foveal vision
│   │   └── mlp.py                     # MLP architectures for RAM processing
│   ├── pvm_buffer.py                  # Priority-based vision movement buffer
│   ├── save_model_helpers.py          # Utilities for model saving and loading
│   ├── sparse_utils.py                # Utilities for model sparsification
│   └── utils.py                       # General utility functions
├── test/                              # Unit and integration tests
│   ├── test_buffer.py                 # Tests for buffer implementations
│   ├── test_envs.py                   # Tests for environment wrappers
│   └── test_models.py                 # Tests for neural network models
├── train_fovea.py                     # Training script for foveal vision
├── train_fovea_lightened.py           # Lightweight foveal training
├── train_fovea_peripheral.py          # Training with foveal and peripheral vision
├── train_full_vision.py               # Training script for full vision
├── train_motor_separate_sparsity.py   # Training with separated networks and sparsity
├── train_no_vision.py                 # Training script for RAM-based agent
└── visualize_fovea.ipynb              # Interactive notebook for foveal visualization
```
## Environment Setup

### Prerequisites

- Python 3.9+ 
- CUDA (for GPU acceleration)
- Weights & Biases (wandb) account for experiment tracking

### Installation

1. Create and activate a conda environment from the [`environment.yml`](environment.yml) file:

```bash
conda env create -n pong_vision python=3.11
conda activate pong_vision
pip install -r requirements.txt
```
2. Install Atari ROMs (required for Pong environments):
```bash
pip install autorom
AutoROM --accept-license
```
3. If necessary, import the ROMs to the correct path:
```bash
python -m atari_py.import_roms path/to/your/conda/env/lib/python3.x/site-packages/AutoROM/roms
```
4. Login to wandb
```bash
wandb login
```
### How to Run the Code

#### Training Models
To train a model with foveal vision:
```bash
python train_fovea.py --config config/dqn_fovea_peripheral_pong_config.yml
```
To train a model with full vision:
```bash
python train_full_vision.py --config config/dqn_fullvis_pong_config.yml
```
To train a model with motor separate foveal vision:
```bash
python train_motor_separate_fovea.py --config config/dqn_fovea_pong_motor_separate_config.yml
```

#### Training Parameters
Key parameters that can be adjusted in configuration files:

- learning_rate: Learning rate for the optimizer
- buffer_size: Size of the replay buffer
- batch_size: Batch size for training
- gamma: Discount factor
- target_update_interval: How often to update the target network
- exploration_fraction: Fraction of training to explore
- exploration_initial_eps: Initial exploration rate
- exploration_final_eps: Final exploration rate

For foveal vision specific parameters:

- fovea_size: Size of the high-resolution fovea region
- peripheral_size: Size of the peripheral vision region
- fovea_movement_cost: Cost associated with moving the fovea

#### Evaluating Models
To evaluate a trained model or trained models:
```bash
python evaluate_models.py --model path/to/model.pt --config path/to/config.yml
```
To specifically evaluate a motor separate foveal model for sparsity performance:
```bash
python evaluate_model_fovea_motor_separate_sparsity.py --model path/to/trained/model.pt
```
#### Visualization
To visualize the behavior of the models run:
- Fovea motor separate:
```bash
python visualize_model_fovea_motor_separate.py --model path/to/trained/model.pt --config path/to/config.yml
```
- Fovea peripheral motor separate:
```bash
python visualize_model_fovea_peripheral_motor_separate.py --model path/to/trained/model.pt --config path/to/config.yml
```
- Fovea peripheral model
```bash
python visualize_model_fovea_peripheral.py --model path/to/trained/model.pt --config path/to/config.yml
```
- Fovea model:
```bash
python visualize_model_fovea.py --model path/to/trained/model.pt --config path/to/config.yml
```

### Implementation Details
The models are trained with DQN (Deep Q-Network) using Stable Baselines 3. The Pong environments are based on Gymnasium (formerly Gym) with ALE-Py (Arcade Learning Environment). For foveal vision, a special focusing mechanism is implemented to allow the agent to concentrate on specific parts of the screen.

#### Algorithms

- **DQN (Deep Q-Network)**: The base reinforcement learning algorithm used in all models
- **Double DQN**: Implementation to reduce overestimation of Q-values
- **Prioritized Experience Replay**: Used in some experiments to improve sample efficiency
- **Network Sparsification**: Techniques to reduce model complexity while maintaining performance

#### Neural Network Architectures

- **CNN**: For processing image-based inputs (full vision and foveal vision)
- **MLP**: For processing RAM data or lower-dimensional representations
- **Separated Networks**: Architecture that separates perception and action networks

#### Performance Metrics

- **Average Score**: The primary measure of agent performance
- **Computational Efficiency**: Measured in terms of training time and model size
- **Sample Efficiency**: How quickly the agent learns from experience
- **Fovea Utilization**: How effectively the agent uses its foveal vision