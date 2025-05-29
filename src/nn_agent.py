import numpy as np
import tensorflow as tf

class NNAgent:
    def __init__(self, input_size, action_size, learning_rate=0.001, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        """Initialize the NN agent with dynamic input and action sizes based on number of workstations."""
        self.input_size = input_size  # (num_machines * features_per_machine) + global_features
        self.action_size = action_size  # num_machines + 1 (maintenance per WS + "do nothing")
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory = []  # Experience replay buffer
        self.rewards_history = []  # Track rewards, not saved
        self.model = self.build_model()  # Build fresh model on init

    def build_model(self):
        """Build the DQN model with dynamic input and output sizes."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(68, input_dim=self.input_size, activation='relu'),
            tf.keras.layers.Dense(68, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def choose_action(self, state):
        """Select an action using epsilon-greedy policy."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])

    def store_experience(self, state, action, reward, next_state, done):
        """Store experience tuple in memory."""
        self.memory.append((state, action, reward, next_state, done))
        self.rewards_history.append(reward)  # Track rewards, not saved

    def train(self, batch_size=32):
        """Train the model on a batch of experiences."""
        if len(self.memory) < batch_size:
            return
        batch = np.random.choice(len(self.memory), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.memory[i] for i in batch])
        
        states = np.array(states)
        next_states = np.array(next_states)
        targets = self.model.predict(states, verbose=0)
        next_q_values = self.model.predict(next_states, verbose=0)
        
        for i in range(batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        self.model.fit(states, targets, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, path):
        """Save the trained model to a file."""
        self.model.save(path)
        print(f"Model saved to {path}")