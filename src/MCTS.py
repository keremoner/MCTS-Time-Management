import gym
import numpy as np
import copy
import random


class Node:
    def __init__(self, state, parent, action, isTerminal, reward, action_space):
        self.parent = parent
        self.action = action
        self.unexplored = list(action_space)
        self.children = []
        self.children_N = [0, 0, 0, 0]
        self.Q = 0
        self.N = 0
        self.isTerminal = isTerminal
        self.reward = reward
        self.isExplored = False


class mcts_agent:
    def __init__(self, state, env, temperature=1 / np.sqrt(2), simulations=100, discount_factor=0.997, render=False):
        self.state = state
        self.env = env
        self.temperature = temperature
        self.simulations = simulations
        self.render = render
        self.action_space = np.array(list(range(env.action_space.n)))
        self.discount_factor = discount_factor

    def step(self):
        best_action = self.mcts_search(self.state)
        self.state, reward, terminal, info = self.env.step(best_action)
        if self.render:
            self.env.render()
        return terminal, reward

    def select_action(self, state):
        return self.mcts_search(state)

    def mcts_search(self, state):
        root = Node(state, None, None, False, 0, self.action_space)
        root.isExplored = True
        iter = 0
        while iter < self.simulations:
            # Copying the environment
            self.search_env = copy.deepcopy(self.env)

            # Applying the tree policy
            next = self.__treePolicy(root)

            # Applying the default/rollout policy to get a reward
            reward = self.__default_policy(next)
            # print("Return of the rollout: ", reward)

            # Applying back up and updating Q and N values using discounting
            self.__backup(next, reward)
            iter += 1
        # Return the action with the highest win-rate
        return self.__bestChild(root, 0).action

    def __treePolicy(self, node):
        terminal = None
        while not node.isTerminal:
            if len(node.unexplored) > 0:
                return self.__expand(node)
            else:
                node = self.__bestChild(node, self.temperature)
                next_state, _, terminal, _ = self.search_env.step(node.action)
        return node

    def __expand(self, node):
        # Find an unexplored actions and choose randomly amongst them
        # unexpanded = node.unexplored[random.randint(0, len(node.unexplored) - 1)]
        unexpanded = node.unexplored[0]
        new_state, reward, done, info = self.search_env.step(unexpanded)

        # Expanding the node with a new node
        new_node = Node(new_state, node, unexpanded, done, reward, self.action_space)
        node.children.append(new_node)

        # Remove the newly expanded node from the unexplored actions in the node
        node.unexplored.remove(unexpanded)

        return new_node

    def __bestChild(self, node, temperature):
        max_val = -np.inf
        best_child = None
        for child in node.children:
            uct_val = child.Q / node.children_N[child.action] + temperature * np.sqrt(np.log(node.N) / node.children_N[child.action])
            if uct_val >= max_val:
                max_val = uct_val
                best_child = child
        return best_child

    def __default_policy(self, node):
        terminal = node.isTerminal
        aggregate_reward = 0
        # Apply the default policy until reaching a terminal state
        i = 0
        if terminal:
            return 0.0
        while not terminal:
            rand_action = random.randint(0, len(self.action_space) - 1)
            new_state, reward, terminal, info = self.search_env.step(rand_action)
            aggregate_reward += reward * self.discount_factor ** i
            i += 1
        return aggregate_reward

    def __backup(self, node, reward):
        i = 0
        while not node == None:
            reward = self.discount_factor * reward + node.reward
            if node.isExplored:
                node.N += 1
            else:
                node.isExplored = True
            if node.parent != None:
                node.parent.children_N[node.action] = node.parent.children_N[node.action] + 1
            node.Q += reward
            i += 1
            node = node.parent
    
    def set_temperature(self, temperature):
        self.temperature = temperature
    
    def set_simulations(self, simulations):
        self.simulations = simulations


if __name__ == '__main__':
    env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=False)
    # env = gym.make("CartPole-v1")
    root_state = env.reset()
    root_state, _, _, _ = env.step(1)
    agent = mcts_agent(root_state, env, temperature=1, simulations=8, discount_factor=0.997)
    reward = 0
    terminal = False

    # terminal, r = agent.step()
    while not terminal:
        terminal, r = agent.step()
        reward += r
    print("Final result: ", reward)