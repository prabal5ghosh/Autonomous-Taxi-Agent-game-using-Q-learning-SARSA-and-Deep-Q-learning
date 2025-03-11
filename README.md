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





This Python project implements three different Reinforcement Learning (RL) algorithms—Q-Learning, SARSA, and Deep Q-Networks (DQN)—to solve the **Taxi-v3** environment from OpenAI's Gym library. Below, I provide a detailed explanation of the code, the reinforcement learning concepts applied, and how each algorithm operates.

---

## **1. Understanding the Taxi-v3 Environment**
The `Taxi-v3` environment is a grid-based game where a taxi agent must:
- Pick up a passenger from one of the predefined locations.
- Navigate to the passenger’s destination while avoiding penalties.

### **State Space**
- The environment has **500 discrete states**. 
- Each state represents a combination of:
  - The taxi’s location (5×5 grid = 25 positions).
  - The passenger’s location (one of 4 fixed locations or in the taxi).
  - The passenger’s destination (one of 4 locations).
- The state is a single integer representing these factors.

### **Action Space**
- There are **6 discrete actions**:
  1. Move South
  2. Move North
  3. Move East
  4. Move West
  5. Pickup Passenger
  6. Drop off Passenger

### **Rewards**
- **+20** for successfully dropping off a passenger.
- **-10** for trying to pick up/drop off incorrectly.
- **-1** for each movement (to encourage efficiency).

---

## **2. Code Breakdown**
The code follows a structured pipeline for training RL agents and evaluating their performance.

### **Libraries Used**
- `gym`: Provides the Taxi-v3 environment.
- `numpy` & `random`: Used for handling matrices and randomness.
- `torch`: Implements the deep learning-based DQN.
- `matplotlib.pyplot`: Plots the learning curves.

---

## **3. Q-Learning Algorithm**
**Q-Learning** is an off-policy **model-free** RL algorithm that updates the Q-value using the Bellman equation:
\[
Q(s, a) = Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]
\]
where:
- \( Q(s, a) \) is the Q-value for state \( s \) and action \( a \).
- \( r \) is the immediate reward.
- \( \gamma \) is the discount factor.
- \( \alpha \) is the learning rate.
- \( \max_{a'} Q(s', a') \) is the maximum future Q-value for the next state.

### **Implementation in Code**
- The **Q-table** (`Q_table`) is initialized to zeros.
- For each episode:
  - The agent selects an action using an **epsilon-greedy policy** (exploration vs. exploitation).
  - The action is executed in the environment, and the reward and next state are received.
  - The **Q-value is updated** based on the Bellman equation.
  - If the episode ends (`done`), the loop breaks.
  - **Epsilon decay** is applied to gradually shift from exploration to exploitation.

---

## **4. SARSA Algorithm**
**SARSA** (State-Action-Reward-State-Action) is an **on-policy** RL algorithm. It updates the Q-values differently from Q-Learning:
\[
Q(s, a) = Q(s, a) + \alpha \left[ r + \gamma Q(s', a') - Q(s, a) \right]
\]
The main difference is that instead of taking the **max** Q-value of the next state, it follows the next chosen action.

### **Implementation in Code**
- The **SARSA** function follows the same structure as Q-learning but selects the next action **before updating the Q-table**.
- The update rule uses **the actual next action chosen** instead of the greedy maximum.

---

## **5. Deep Q-Learning (DQN)**
DQN replaces the Q-table with a **neural network** that estimates Q-values. It overcomes the problem of large state-action spaces where maintaining a Q-table becomes infeasible.

### **Neural Network**
- **Input layer:** Takes a one-hot encoded state vector.
- **Two hidden layers** with **64 neurons** and ReLU activations.
- **Output layer:** Outputs Q-values for all actions.

### **Replay Memory**
DQN uses **experience replay** (a buffer storing past experiences) to break correlation between sequential samples.

### **Target Network**
A separate **target network** is updated periodically to **stabilize training** and reduce oscillations.

### **DQN Algorithm in Code**
- At each step:
  - The agent selects an action using the **epsilon-greedy policy**.
  - The transition (`state, action, reward, next_state, done`) is stored in **Replay Memory**.
  - If the memory is large enough, a **random batch** is sampled to update the network.
  - The **Bellman loss function** is computed:
    \[
    \text{loss} = (Q_{\text{target}} - Q_{\text{policy}})^2
    \]
  - **Gradient descent** is used to optimize the network.
  - The **target network is updated** periodically.

---

## **6. Training and Evaluation**
The `main()` function allows the user to choose **Q-learning, SARSA, or DQN** and trains the selected agent.

- **Training**: Runs multiple episodes and updates policies.
- **Evaluation**: Runs episodes without exploration and computes the average reward.

### **Performance Comparison**
| Algorithm | Learning Type | Convergence Speed | Performance |
|-----------|--------------|-------------------|-------------|
| Q-Learning | Off-policy | Faster | High |
| SARSA | On-policy | Slower | Safer (avoids risky actions) |
| DQN | Function Approximation | Scales better | Best in large state spaces |





------------------


modeled as a Markov Decision Process (MDP) with:

States (S): The environment consists of 500 states (25 taxi positions × 5 passenger locations × 4 possible destinations).
Actions (A): The agent can take 6 discrete actions:
Move South (0)
Move North (1)
Move East (2)
Move West (3)
Pick up passenger (4)
Drop off passenger (5)
Rewards (R):
+20 points for successfully picking up and dropping off the passenger.
-1 point for each move (to encourage efficiency).
-10 points for illegal actions (e.g., trying to drop off a passenger at the wrong location or picking up when there is no passenger).
Transition Dynamics (T): The environment transitions between states based on the agent’s actions.
Policy (π): A mapping from states to actions, representing the agent's strategy.

