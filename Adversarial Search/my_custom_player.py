
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
            distance = -math.sqrt((agent_loc[0]-adversary_loc[0])**2 + (agent_loc[1]-adversary_loc[1])**2)
        return distance

    def count_moves(self, gameState, player_id):
          loc = gameState.locs[player_id]
          return len(gameState.liberties(loc))

    def baseline_heuristic(self, gameState):
        """
          Returns the number of the plyer's moves - number of opponent's moves.
        """
        return self.count_moves(gameState, 0) - self.count_moves(gameState, 1)

    def minimax_decision(self, gameState, depth):
        """ Return the move along a branch of the game tree that
        has the best possible value.  A move is a pair of coordinates
        in (column, row) order corresponding to a legal move for
        the searching player.
        
        You can ignore the special case of calling this function
        from a terminal state.
        """
        best_score = float("-inf")
        best_move = None
        for a in gameState.actions():
            # call has been updated with a depth limit
            v = self.min_value(gameState.result(a), depth - 1)
            if v > best_score:
                best_score = v
                best_move = a
        return best_move


    def min_value(self, gameState, depth):
        """ Return the value for a win (+1) if the game is over,
        otherwise return the minimum value over all legal child
        nodes.
        """
        if gameState.terminal_test():
            return gameState.utility(0)
        
        # New conditional depth limit cutoff
        if depth <= 0:  # "==" could be used, but "<=" is safer 
            return self.manhattan_distance(gameState)
        
        v = float("inf")
        for a in gameState.actions():
            # the depth should be decremented by 1 on each call
            v = min(v, self.max_value(gameState.result(a), depth - 1))
        return v


    def max_value(self, gameState, depth):
        """ Return the value for a loss (-1) if the game is over,
        otherwise return the maximum value over all legal child
        nodes.
        """
        if gameState.terminal_test():
            return gameState.utility(0)
        
        # New conditional depth limit cutoff
        if depth <= 0:  # "==" could be used, but "<=" is safer 
            return self.manhattan_distance(gameState)
        
        v = float("-inf")
        for a in gameState.actions():
            # the depth should be decremented by 1 on each call
            v = max(v, self.min_value(gameState.result(a), depth - 1))
        return v

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
        best_action = self.minimax_decision(state,4)
        self.queue.put(best_action)
        #else:
        #  self.queue.put(random.choice(state.actions))
