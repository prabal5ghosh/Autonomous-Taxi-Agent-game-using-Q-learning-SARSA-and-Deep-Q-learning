  <a href="https://univ-cotedazur.eu/msc/msc-data-science-and-artificial-intelligence" target="_blank" rel="noreferrer">
    <img src="https://upload.wikimedia.org/wikipedia/fr/thumb/f/fa/Logo-univ-nice-cote-dazur.svg/587px-Logo-univ-nice-cote-dazur.svg.png?20211016184305"  alt="Université Côte d'Azur" align="center"/>
  </a>
# Autonomous Taxi Agent Game using Q-learning, SARSA, Deep Q-learning, and Value Iteration
This project implements four different reinforcement learning algorithms—Q-Learning, SARSA, Deep Q-Networks (DQN), and Value Iteration—to solve the **Taxi-v3** environment from OpenAI's Gym library. The goal is to compare the performance of these algorithms in training an autonomous taxi agent. Below, I provide a detailed explanation of the code, the reinforcement learning concepts applied, and how each algorithm operates.





## **Understanding the Taxi-v3 Environment**
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

## 
The code follows a structured pipeline for training RL agents and evaluating their performance.

### **Libraries Used**
- `gym`: Provides the Taxi-v3 environment.
- `numpy` & `random`: Used for handling matrices and randomness.
- `torch`: Implements the deep learning-based DQN.
- `matplotlib.pyplot`: Plots the learning curves.

---
## **1. Q-Learning Algorithm**
**Q-Learning** is an off-policy **model-free** RL algorithm that updates the Q-value using the Bellman equation:
\[ Q(s, a) = Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right] \]
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


## **2. SARSA Algorithm**
**SARSA** (State-Action-Reward-State-Action) is an **on-policy** RL algorithm. It updates the Q-values differently from Q-Learning:
\[ Q(s, a) = Q(s, a) + \alpha \left[ r + \gamma Q(s', a') - Q(s, a) \right] \]
The main difference is that instead of taking the **max** Q-value of the next state, it follows the next chosen action.

### **Implementation in Code**
- The **SARSA** function follows the same structure as Q-learning but selects the next action **before updating the Q-table**.
- The update rule uses **the actual next action chosen** instead of the greedy maximum.

##
Both functions create a Q‑table with shape (500, 6) (since Taxi‑v3 has 500 states and 6 actions).
– In Q‑Learning the update uses the maximum Q‑value from the next state.
– In SARSA the update uses the Q‑value of the action actually taken (on‑policy).
In both cases, an epsilon‑greedy policy controls exploration and epsilon is decayed over episodes.

## **3. Deep Q-Learning (DQN)**
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
    \[ \text{loss} = (Q_{\text{target}} - Q_{\text{policy}})^2 \]
  - **Gradient descent** is used to optimize the network.
  - The **target network is updated** periodically.

## *4. Value Iteration Algorithm*
Value Iteration is a dynamic programming algorithm used to compute the optimal policy by iteratively improving the value function.

**Implementation:**
- Initialize the value function for all states to zero.
- Iteratively update the value function using the Bellman optimality equation.
- Convergence is achieved when the maximum change in value function is below a threshold.
- Derive the optimal policy from the converged value function.

**Visualization:**
- Plot the convergence of value iteration.
- Animate the optimal policy execution in the Taxi-v3 environment.
---
##  Training and Evaluation
The `main()` function allows the user to choose Q-learning, SARSA, DQN, or Value Iteration and trains the selected agent.

**Training:**
- Runs multiple episodes and updates policies.

**Evaluation:**
- Runs episodes without exploration and computes the average reward.

#### Performance Comparison
| Algorithm       | Learning Type       | Convergence Speed | Performance                |
|-----------------|---------------------|-------------------|----------------------------|
| Q-Learning      | Off-policy          | Faster            | High                       |
| SARSA           | On-policy           | Slower            | Safer (avoids risky actions)|
| DQN             | Function Approximation | Scales better   | Best in large state spaces |
| Value Iteration | Dynamic Programming | Fast              | Optimal policy             |

The project models the Taxi-v3 environment as a Markov Decision Process (MDP) with states, actions, rewards, transition dynamics, and policies. The implementation and comparison of these algorithms provide insights into their effectiveness in different scenarios.


----------------------------------------------------

