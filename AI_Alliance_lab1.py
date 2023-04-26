import numpy as np
import matplotlib.pyplot as plt

class Node:
    def __init__(self, reward, Prob_correct_ans):
        # Initialize a new Node object with the given reward and probability values
        self.REWARD = reward
        self.PROBABILITY = Prob_correct_ans
         # Initialize the `next` attribute as `None`, to be set later when the Node is added to a linked list
        self.next = None

class Agent:
    def __init__(self):
        # Set default values for various parameters that will be used later
        self.N = 10
        self.THETA = 0.0001
        self.DISCOUNT_FACTOR = 0.9
        self.ACTIONS = ["CONTINUE", "QUIT"]
        
        # Define a list of possible rewards and probabilities of getting the correct answer for each question
        REWARDS = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000]
        CORRECT_ANSWER_PROBABILITY = [0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        
        # Create a linked list of nodes, where each node represents a question and its corresponding reward and probability
        START_NODE = Node(-1,0) # Create the starting node with default values of -1 for the reward and 0 for the probability
        TEMP = START_NODE # Set a temporary node variable to the starting node
        
        for i in range(self.N):  # Loop over the number of questions to create nodes for each question
            TEMP.next = Node(REWARDS[i],CORRECT_ANSWER_PROBABILITY[i]) # Create a new node with the current question's reward and probability
            TEMP = TEMP.next # Move the temporary node variable to the newly created node
        
        
        self.START_STATE = START_NODE.next # Set the starting state to the first question node in the linked list


class MDP_SOLUTION:
    def __init__(self):
        self.agent = Agent()
        self.VALUE_FUNCTION = {s: 0 for s in range(self.agent.N)}
        self.ITERATIONS = 0
        self.TIMES_ENTERED = {s: 0 for s in range(self.agent.N)}
        self.TERMINATOR = False
        self.PLOT_STATES = [x for x in range(self.agent.N)]
    
    def helper(self, state, iteration):
        # if there are no more states to process, return 0
        if state is None: 
            return 0
        
        # increment the count for how many times the agent has entered this state
        self.TIMES_ENTERED[iteration] += 1

        # save the current value of the value function for this state
        OLD_VALUE = self.VALUE_FUNCTION[iteration]

        # set the achieved reward to 0 initially
        REWARD_ACHIEVED = 0
        
        # if we are in the first state, the quit reward is 0
        if iteration == 0:
            QUIT_REWARD = 0
        else:
            # otherwise, the quit reward is the value function for the previous state
            QUIT_REWARD = self.VALUE_FUNCTION[iteration-1]
            
        # generate a random number to determine if the agent answers the question correctly
        result = np.random.rand()
        
        # if the agent answers the question correctly, calculate the reward achieved
        if result <= state.PROBABILITY:
            # calculate the reward for getting the question right and moving to the next state
            REWARD_ACHIEVED = state.PROBABILITY * (state.REWARD + (self.agent.DISCOUNT_FACTOR * self.helper(state.next, iteration+1)))
            
            # update the value function for this state
            self.VALUE_FUNCTION[iteration] = (self.VALUE_FUNCTION[iteration] * self.TIMES_ENTERED[iteration] + REWARD_ACHIEVED)/(self.TIMES_ENTERED[iteration]+1)
                
            # check if the value function for this state has converged
            if(abs(self.VALUE_FUNCTION[iteration] - OLD_VALUE) < self.agent.THETA):
                self.TERMINATOR = True

        # return the maximum of the quit reward and the achieved reward
        return max(QUIT_REWARD, REWARD_ACHIEVED)

    
    def solver(self):
        # Loop until the termination condition is satisfied
        while not self.TERMINATOR:
            # Increment the iteration count
            self.ITERATIONS += 1
            # Start at the beginning of the linked list of questions
            node = self.agent.START_STATE
            # Call the `helper` function to update the value function and times entered for each state
            self.helper(node, 0)

        # Print the total number of iterations and the final value function
        print("Total Iterations:", self.ITERATIONS)
        print("Value Function:")
        print(self.VALUE_FUNCTION)

        # Calculate the percentage of times each state was entered
        times_entered_percent = [count / self.ITERATIONS * 100 for count in self.TIMES_ENTERED.values()]

        # Print the times entered for each state and the expected reward
        print("Times Entered (%):")
        print(times_entered_percent)
        expected_reward = sum(value * percent for value, percent in zip(self.VALUE_FUNCTION.values(), times_entered_percent))
        print("Expected Reward:", expected_reward)

        # Plot the value function and times entered for each state
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.bar(range(self.agent.N), self.VALUE_FUNCTION.values())
        plt.xticks(range(self.agent.N))
        plt.xlabel('States')
        plt.ylabel('Value Function')
        plt.title('Maximum Reward for each state')

        plt.subplot(1, 2, 2)
        plt.bar(range(self.agent.N), times_entered_percent)
        plt.xticks(range(self.agent.N))
        plt.xlabel('States')
        plt.ylabel('Times Entered (%)')
        plt.title('Number of times agent entered each state')

        plt.tight_layout()
        plt.show()

MDP_SOLUTION().solver()