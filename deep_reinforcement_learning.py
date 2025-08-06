#!/usr/bin/env python3
"""
Deep Reinforcement Learning - CartPole DQN Implementation
Converted from Jupyter notebook to Python script
"""

# 1. Installations (commented out for script version)
# !pip install keras
# !pip install keras-rl2
# !pip install tensorflow==2.3.0
# !pip install gym

# 1. imports and setup
import gym 
import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

def main():
    game = gym.make('CartPole-v0')
    states = game.observation_space.shape[0]
    actions = game.action_space.n
    
    print(f"States: {states}")
    print(f"Actions: {actions}")
    
    # 2. creating deep learning model with keras
    print("\n=== Building Deep Learning Model ===")
    
    def build_model(states, actions):
        model = Sequential()
        model.add(Flatten(input_shape=(1,states)))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(actions, activation='linear'))
        return model
    
    model = build_model(states, actions)
    model.summary()
    
    # 3. building dqn agent with keras-rl
    print("\n=== Building DQN Agent ===")
    
    def build_agent(model, actions):
        policy = BoltzmannQPolicy()
        memory = SequentialMemory(limit=50000, window_length=1)
        dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                      nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
        return dqn
    
    dqn = build_agent(model, actions)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    
    # 4. training dqn agent
    print("\n=== Training DQN Agent ===")
    dqn.fit(game, nb_steps=50000, visualize=False, verbose=1)
    
    # 5. testing trained dqn agent
    print("\n=== Testing Trained Agent ===")
    scores = dqn.test(game, nb_episodes=100, visualize=False)
    print(f"Average score: {np.mean(scores.history['episode_reward'])}")
    
    # 6. visualizing trained dqn agent
    print("\n=== Visualizing Trained Agent ===")
    _ = dqn.test(game, nb_episodes=15, visualize=True)
    
    # 7. saving and reloading dqn agent
    print("\n=== Saving and Reloading Agent ===")
    dqn.save_weights('dqn_weights.h5f', overwrite=True)
    
    # clean up
    del model
    del dqn
    del game
    
    # reloading environment and model
    game = gym.make('CartPole-v0')
    actions = game.action_space.n
    states = game.observation_space.shape[0]
    model = build_model(states, actions)
    dqn = build_agent(model, actions)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    dqn.load_weights('dqn_weights.h5f')
    
    # 8. Test the reloaded agent
    print("\n=== Testing Reloaded Agent ===")
    _ = dqn.test(game, nb_episodes=5, visualize=True)

if __name__ == "__main__":
    main() 