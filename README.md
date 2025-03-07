# Autonomous-Taxi-Agent-game-using-Q-learning-SARSA-and-Deep-Q-learning
We implement Q-learning, SARSA, and Deep Q-learning in this game. we compare their performance.



Python script that implements three different reinforcement learning algorithms for the Taxi‑v3 environment: Q‑Learning (off‑policy), SARSA (on‑policy), and Deep Q‑Learning (DQN using PyTorch). we can run the script and choose which algorithm to train and evaluate by entering one of the following strings: "q_learning", "sarsa", or "dqn".


	1. Q‑Learning and SARSA:
Both functions create a Q‑table with shape (500, 6) (since Taxi‑v3 has 500 states and 6 actions).
– In Q‑Learning the update uses the maximum Q‑value from the next state.
– In SARSA the update uses the Q‑value of the action actually taken (on‑policy).
In both cases, an epsilon‑greedy policy controls exploration and epsilon is decayed over episodes.
	2. Deep Q‑Learning (DQN):
– The state (an integer from 0 to 499) is converted into a one‑hot encoded vector.
– A simple neural network approximates Q‑values.
– A replay memory is used to sample batches and update the network using the MSE loss between current Q‑values and targets computed from the target network.
– The target network is updated every few episodes.
	3. Evaluation:
Separate evaluation functions run the learned policy (greedily) and compute the mean reward over many episodes.
	4. Main Function:
The script prompts you for an algorithm choice, trains the corresponding agent, plots the training rewards, and prints the evaluation mean reward.
