import gym
import numpy as np
import copy
import random

class Node:
    def __init__(self, parent, action, isTerminal, reward, action_space):
        self.parent = parent
        self.action = action
        self.unexplored = list(action_space)
        self.children = []
        self.children_N = [0] * len(action_space)
        self.Q = 0
        self.N = 0
        self.isTerminal = isTerminal
        self.reward = reward
        self.isExplored = False

class mcts_agent:
    def __init__(self, temperature=1 / np.sqrt(2), simulations=100, discount_factor=0.997, render=False, horizon=-1):
        self.temperature = temperature
        self.simulations = simulations
        self.render = render
        self.discount_factor = discount_factor
        self.horizon = horizon
        self.action_space = None
        self.copy_env = None

    def select_action(self, env):
        return self.mcts_search(env)

    def mcts_search(self, env):
        self.action_space = np.array(list(range(env.get_action_space().n)))
        
        self.copy_env = copy.deepcopy(env)
        root_state =  self.copy_env.get_state()
        root = Node(None, None, False, 0, self.action_space)
        self.__expand2(root)
        root.parent = None
        root.action = None
        root.isExplored = True
        
        for i in range(self.simulations):
            
            self.depth = 0
            # Starting the search  from the root state
            self.copy_env.set_state(root_state)
            # Applying the tree policy
            next = self.__treePolicy(root)
            # Applying the default/rollout policy to get a reward
            reward = self.__default_policy(next)

            # Applying back up and updating Q and N values using discounting
            self.__backup(next, reward)
        # Return the action with the highest win-rate
        return self.__bestChild(root, 0).action

    def __treePolicy(self, node):
        terminal = None
        while not node.isTerminal:
            if self.depth == self.horizon:
                break
            
            self.depth += 1
            
            if len(node.children) == 0:
                self.__expand2(node)
                return node
            else:
                if len(node.unexplored) > 0:
                    unexpanded = node.unexplored[random.randint(0, len(node.unexplored) - 1)]
                    new_state, reward, done, info = self.copy_env.step(unexpanded)
                    node.children[unexpanded].isTerminal = done
                    node.children[unexpanded].reward = reward
                    node.unexplored.remove(unexpanded)
                    node = node.children[unexpanded]
                else:  
                    node = self.__bestChild(node, self.temperature)
                    next_state, _, terminal, _ = self.copy_env.step(node.action)
        return node

    def __expand2(self, node):
        node.children = [Node(node, action, False, 0, self.action_space) for action in node.unexplored]
    
    def __expand(self, node):
        # Find an unexplored actions and choose randomly amongst them
        # unexpanded = node.unexplored[random.randint(0, len(node.unexplored) - 1)]
        unexpanded = node.unexplored[0]
        new_state, reward, done, info = self.copy_env.step(unexpanded)

        # Expanding the node with a new node
        new_node = Node(node, unexpanded, done, reward, self.action_space)
        node.children.append(new_node)

        # Remove the newly expanded node from the unexplored actions in the node
        node.unexplored.remove(unexpanded)

        return new_node

    def __bestChild(self, node, temperature):
        max_val = -np.inf
        best_child = None
        for child in node.children:
            uct_val = -np.inf
            if node.children_N[child.action] == 0:
                pass
            else:
                uct_val = child.Q / node.children_N[child.action] + temperature * np.sqrt(np.log(node.N) / node.children_N[child.action])
            if uct_val >= max_val:
                max_val = uct_val
                best_child = child
        return best_child

    def __default_policy(self, node):
        terminal = node.isTerminal
        aggregate_reward = 0.0
        # Apply the default policy until reaching a terminal state
        i = 0
        
        if terminal:
            return 0.0
        while not terminal:
            if self.depth == self.horizon:
                break
            rand_action = random.randint(0, len(self.action_space) - 1)
            new_state, reward, terminal, info = self.copy_env.step(rand_action)
            aggregate_reward += reward * self.discount_factor ** i
            i += 1
            self.depth += 1
        return aggregate_reward

    def __backup(self, node, reward):
        i = 0
        while not node == None:
            reward = self.discount_factor * reward + node.reward
            node.N += 1
            if node.parent != None:
                node.parent.children_N[node.action] = node.parent.children_N[node.action] + 1
            node.Q += reward
            i += 1
            node = node.parent
    
    def set_temperature(self, temperature):
        self.temperature = temperature
    
    def set_simulations(self, simulations):
        self.simulations = simulations
    
    def set_horizon(self, horizon):
        self.horizon = horizon
        
    def __str__(self):
        return "Discount Factor = " + str(self.discount_factor) + ", Horizon = " + (str(self.horizon) if self.horizon != -1 else "Infinite")