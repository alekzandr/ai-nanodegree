
from sample_players import DataPlayer
from mcts import mcts
import time
import random
import math


class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    
    def convert_index_to_xy(self, ind):
        _WIDTH = 11
        return (ind % (_WIDTH + 2), ind // (_WIDTH + 2))

    def manhattan_distance(self, gameState):
        # TODO: Finish this function!
        # HINT: the global player_id variable is accessible inside
        #       this function scope
        agent = 0
        adversary = 1
        agent_loc = gameState.locs[agent]
        adversary_loc = gameState.locs[adversary]
        distance = 0
        #print("Agent Location: {}".format(agent_loc))
        if agent_loc != None and adversary_loc != None:
            agent_loc = self.convert_index_to_xy(agent_loc)
            adversary_loc = self.convert_index_to_xy(adversary_loc)
            distance = math.sqrt((agent_loc[0]-adversary_loc[0])**2 + (agent_loc[1]-adversary_loc[1])**2)
        return distance

    def count_moves(self, gameState, player_id):
          loc = gameState.locs[player_id]
          return len(gameState.liberties(loc))

    def baseline_heuristic(self, gameState):
        """
          Returns the number of the plyer's moves - number of opponent's moves.
        """
        return self.count_moves(gameState, 0) - self.count_moves(gameState, 1)

    def aggressive_greedy_heuristic(self, gameState, weight_1=1.5, weight_2=3, bias=0 ):
        weight_1 = weight_1
        weight_2 = weight_2
        bias = bias
        return weight_1 * -(self.manhattan_distance(gameState)) + weight_2 * self.baseline_heuristic(gameState) + bias


    def minimax(self, state, depth, heuristic):

        def min_value(state, depth):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return heuristic(state)
            value = float("inf")
            for action in state.actions():
                value = min(value, max_value(state.result(action), depth - 1))
            return value

        def max_value(state, depth):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return heuristic(state)
            value = float("-inf")
            for action in state.actions():
                value = max(value, min_value(state.result(action), depth - 1))
            return value

        return max(state.actions(), key=lambda x: min_value(state.result(x), depth - 1))

    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
        #start_time = time.clock()
        #while ((start_time - time.clock()) * 1000) < 10:
        best_action = self.minimax(state,2, self.baseline_heuristic)
        self.queue.put(best_action)
        #else:
        #  self.queue.put(random.choice(state.actions))
