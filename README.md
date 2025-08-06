# Deep Reinforcement Learning - CartPole DQN

A complete implementation of Deep Q-Network (DQN) reinforcement learning agent to solve the OpenAI Gym CartPole environment.

## What This Project Does

This project demonstrates how to train a neural network agent using Deep Q-Learning to balance a pole on a moving cart. The agent learns to take optimal actions (move left or right) to keep the pole upright for as long as possible.

### Key Features:
- **Deep Q-Network (DQN)** implementation using Keras and Keras-RL
- **Experience Replay** for stable learning
- **Target Network** for consistent training targets
- **Model Persistence** - save and reload trained agents
- **Performance Visualization** - watch the trained agent in action

## Quick Start

### Prerequisites
- Python 3.7+
- TensorFlow 2.3.0
- OpenAI Gym
- Keras-RL2

### Installation

1. **Clone or download this repository**
2. **Install dependencies:**
   ```bash
   pip install tensorflow==2.3.0
   pip install gym
   pip install keras
   pip install keras-rl2
   ```

### Running the Project

Simply execute the Python script:
```bash
python deep_reinforcement_learning.py
```

## Expected Results

### Training Process:
- **Random Performance**: ~10-30 points per episode (baseline)
- **Trained DQN**: ~200 points per episode (perfect performance)
- **Training Time**: ~4-5 minutes (50,000 steps)

### What You'll See:
1. **Environment Setup** - CartPole environment initialization
2. **Model Architecture** - Neural network summary
3. **Training Progress** - Real-time training metrics
4. **Performance Testing** - 100 episodes of evaluation
5. **Visualization** - Watch the trained agent balance the pole
6. **Model Persistence** - Save/load demonstration

## How It Works

### Deep Q-Network (DQN) Algorithm:
1. **Experience Collection**: Agent interacts with environment, storing (state, action, reward, next_state) tuples
2. **Q-Learning**: Updates Q-values using Bellman equation: Q(s,a) = r + γ * max Q(s',a')
3. **Neural Network**: Approximates Q-values for all state-action pairs
4. **Experience Replay**: Randomly samples past experiences to break correlations
5. **Target Network**: Uses separate network for stable learning targets

### Neural Network Architecture:
```
Input (4) → Flatten → Dense(24, ReLU) → Dense(24, ReLU) → Output(2, Linear)
```

### CartPole Environment:
- **State Space**: 4 dimensions (cart position, cart velocity, pole angle, pole angular velocity)
- **Action Space**: 2 actions (move left: 0, move right: 1)
- **Goal**: Keep pole upright for maximum time (200 steps = perfect score)

## Project Structure

```
RL-GYM/
├── deep_reinforcement_learning.py    # Main implementation
├── dqn_weights.h5f                   # Saved model weights (created after training)
└── README.md                         # This file
```

## Customization

### Hyperparameters You Can Modify:
- **Training Steps**: Change `nb_steps=50000` for longer/shorter training
- **Memory Size**: Adjust `limit=50000` in SequentialMemory
- **Learning Rate**: Modify `lr=1e-3` in Adam optimizer
- **Network Size**: Change Dense layer neurons (currently 24)
- **Test Episodes**: Adjust `nb_episodes=100` for testing

### Example Modifications:
```python
# Longer training
dqn.fit(game, nb_steps=100000, visualize=False, verbose=1)

# Larger network
model.add(Dense(64, activation='relu'))  # Instead of 24

# Different learning rate
dqn.compile(Adam(lr=1e-4), metrics=['mae'])  # Slower learning
```

## Learning Resources

### Reinforcement Learning Concepts:
- [Deep Q-Learning](https://en.wikipedia.org/wiki/Q-learning)
- [Experience Replay](https://arxiv.org/abs/1312.5602)
- [Target Networks](https://arxiv.org/abs/1509.02971)

### Related Environments:
- **CartPole-v0**: Classic balancing problem (this project)
- **LunarLander-v2**: Landing a spacecraft
- **Acrobot-v1**: Swinging up a double pendulum
- **MountainCar-v0**: Climbing a hill with limited power

## Troubleshooting

### Common Issues:

**Import Errors:**
```bash
# Make sure you have the correct versions
pip install tensorflow==2.3.0
pip install keras-rl2
```

**Training Not Converging:**
- Increase training steps: `nb_steps=100000`
- Adjust learning rate: `lr=1e-4` or `lr=1e-2`
- Check if environment is working properly

**Memory Issues:**
- Reduce memory size: `limit=25000`
- Use smaller network: Dense(16) instead of Dense(24)

## Performance Tips

1. **GPU Acceleration**: Install TensorFlow-GPU for faster training
2. **Hyperparameter Tuning**: Experiment with learning rates and network sizes
3. **Environment Variations**: Try different Gym environments
4. **Advanced Algorithms**: Implement Dueling DQN, Double DQN, or A3C

