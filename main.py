import random

WIN = 1
LOSS = -1
DRAW = 0

""" environment variables during initialization """
reward_set = {WIN, LOSS, DRAW}
action_set = {0, 1, 2, 3, 4, 5, 6, 7, 8}
state_set = {tuple([" "] * 9)}

class Agent:
    def __init__(self, state, environment):
        self.env = environment
        self.value = {}
        self.policy = {}
        self.state = state

        self.epsilon = 0.1

    def run_mc_control(self, state):
        returns = self.first_visit_mc(state)


    def first_visit_mc(self, state):
        returns = {state: 0}
        
        for i in range(3):
            # get available actions & states
            actions = self.env.get_action_set(state)
            
            # add state to policy if it doesn't exist
            if state not in self.policy:
                self.policy[state] = random.choice(tuple(actions))
            
            # take the action and transition to next state
            action = self.policy[state]
            
            next_state = State()
            next_state.state = state.state.copy()
            next_state.state[action] = "X"

            # compute the reward
            reward = self.env.reward(next_state)
            print(reward)
            
            if state not in returns:
                returns[state] = reward

            state = next_state
            print(state == next_state)
            state.show()

        return returns

    def epsilon_greedy_policy(self):
        pass

class Environment:
    def get_action_set(self, state):
        actions = set()
        for i in range(3):
            for j in range(3):
                if state.state[i * 3 + j] == " ":
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
                if state[a] == "X":
                    return WIN
                else:
                    return LOSS

        return DRAW


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

    print(env.get_action_set(state))
    agent.run_mc_control(state)

if __name__ == "__main__":
    main()
