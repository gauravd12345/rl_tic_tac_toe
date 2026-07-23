import random
import numpy as np

WIN = 10
LOSS = -10
DRAW = 1

ACTIONS = {0, 1, 2, 3, 4, 5, 6, 7, 8}

class Agent:
    def __init__(self, state, environment):
        self.state = state
        self.env = environment

        self.policy = {}
        self.state_action_values = {}
        self.state_action_count = {}
        
        self.epsilon = 0.1
        self.discounting_term = 1.0

    def run_mc_control(self, state, episodes):
        for _ in range(episodes):
            state_action_pairs, returns = self.first_visit_mc(state)
        
            # policy evaluation
            for (s, a), rt in zip(state_action_pairs, returns):   
                key = (tuple(s.state), a)
                if key not in self.state_action_values:
                    self.state_action_values[key] = 0
                    self.state_action_count[key] = 0
    

                self.state_action_count[key] += 1
                self.state_action_values[key] = self.state_action_values[key] + (rt - self.state_action_values[key]) / self.state_action_count[key]


            # policy improvement
            self.update_policy()

        sorted_keys = sorted(
            self.state_action_values, 
            key=lambda pair: self.state_action_count[pair], 
            reverse=True
        )

        i = 0
        for (s, a) in sorted_keys:  
            if i == 20:
                break

            key = (s, a)
            print(f"State_Action: ({s}, {a}) | Value: {self.state_action_values[key]} | Count: {self.state_action_count[key]}")
            i += 1

    def first_visit_mc(self, state):
        state_action_pairs, returns = [], []
        total_reward = 0
        while self.env.reward(state) == 0:
            action_set = list(self.env.get_action_set(state)) # get available actions & states
            
            if len(action_set) != 0:  
                # include state in policy mapping if it doesn't exist
                # initialized with random policy
                if tuple(state.state) not in self.policy:
                    self.policy[tuple(state.state)] = np.zeros(len(ACTIONS))
                    self.policy[tuple(state.state)][action_set] = 1 / len(action_set)
                    
                wei = []
                for a in action_set:
                    wei.append(self.policy[tuple(state.state)][a])  

                # select an action and transition to next state
                action = random.choices(action_set, weights=wei, k=1)[0]
                next_state = State()
                next_state.state = state.state.copy()
                next_state.state[action] = "X"

            # compute the reward
            reward = self.env.reward(next_state)
            total_reward += reward
            if (state, action) not in state_action_pairs:
                state_action_pairs.append((state, action))
                returns.append(reward)
            
            if reward != 0:
                state = next_state
                break

            # opponent actions are selected via random policy
            op_action_set = self.env.get_action_set(next_state)
            if op_action_set:
                op_action = random.choice(tuple(op_action_set))
                next_state.state[op_action] = "O"

            state = next_state

            reward = self.env.reward(state)
            total_reward += reward
            if reward == -10:
                returns[-1] = LOSS

        for i in range(len(returns) - 1):
            returns[i] = total_reward - returns[i]

        return state_action_pairs, returns

    def update_policy(self):
        for s in self.policy:
            state = State()
            state.state = s
            actions = list(self.env.get_action_set(state))

            # compute argmax
            values = [self.state_action_values.get((s, a), 0.0) for a in actions]
            idx = np.argmax(values)

            # update via e-greedy policy
            prob = self.epsilon / len(actions)
            self.policy[s][:] = prob
            self.policy[s][idx] = 1 - self.epsilon + prob

            

class Environment:
    def get_action_set(self, state):
        actions = set()
        for i in range(3):
            for j in range(3):
                if state.state[i * 3 + j] == ' ':
                    actions.add(i * 3 + j)

        return actions

    def get_state_set(self, state):
        states = set()
        for i in range(3):
            for j in range(3):
                if state.state[i * 3 + j] == " ":
                    next_state = state.state.copy()     
                    next_state[i * 3 + j] = "X"
                    states.add(tuple(next_state))

        return states
    
    def reward(self, state):
        winning_positions = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
            (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
            (0, 4, 8), (2, 4, 6)              # diagonals
        ]


        for a, b, c in winning_positions:
            if state.state[a] == state.state[b] == state.state[c] and state.state[a] != " ":
                if state.state[a] == "X":
                    return WIN
                else:
                    return LOSS
                
        if " " not in state.state:
            return DRAW
    
        return 0


class State(Environment):
    def __init__(self):
        self.state = [" ", " ", " ",
                      " ", " ", " ",
                      " ", " ", " "]
        self.reward = 0

        
    def show(self):
        for i in range(3):
            row = [self.state[i * 3 + j] for j in range(3)]
            print(" | ".join(row))
            if i < 2:
                print("--+---+--")

def main():
    env = Environment()
    state = State()
    agent = Agent(state, env)

    agent.run_mc_control(state, 1000)


if __name__ == "__main__":
    main()
