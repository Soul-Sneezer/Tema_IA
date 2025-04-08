from enum import Enum

class Player(Enum):
    MAX = 0
    MIN = 1
    NONE = 2

class State:
    def __init__(self, current_player, current_stage, board):
        self.current_player = player
        self.current_stage = current_stage
        self.board = board
        self.depth = 0

    def create_board():
        nodes = []
        for i in range(24):
            nodes.append(Node(i))
        
        nodes[0].addNeighbor(nodes[1])
        nodes[1].addNeighbor(nodes[2])
        nodes[2].addNeighbor(nodes[3])
        nodes[3].addNeighbor(nodes[4])
        nodes[4].addNeighbor(nodes[5])
        nodes[5].addNeighbor(nodes[6])
        nodes[6].addNeighbor(nodes[7])
        nodes[0].addNeighbor(nodes[7])
        nodes[8].addNeighbor(nodes[9])
        nodes[8].addNeighbor(nodes[0])
        nodes[9].addNeighbor(nodes[10])
        nodes[10].addNeighbor(nodes[11])
        nodes[10].addNeighbor(nodes[2])
        nodes[11].addNeighbor(nodes[12])
        nodes[12].addNeighbor(nodes[13])
        nodes[12].addNeighbor(nodes[4])
        nodes[13].addNeighbor(nodes[14])
        nodes[14].addNeighbor(nodes[15])
        nodes[14].addNeighbor(nodes[6])
        nodes[15].addNeighbor(nodes[8])
        nodes[16].addNeighbor(nodes[17])
        nodes[16].addNeighbor(nodes[8])
        nodes[17].addNeighbor(nodes[18])
        nodes[18].addNeighbor(nodes[19])
        nodes[18].addNeighbor(nodes[10])
        nodes[19].addNeighbor(nodes[20])
        nodes[20].addNeighbor(nodes[21])
        nodes[20].addNeighbor(nodes[12])
        nodes[21].addNeighbor(nodes[22])
        nodes[22].addNeighbor(nodes[23])
        nodes[22].addNeighbor(nodes[14])
        nodes[23].addNeighbor(nodes[16])
    
    for i in range(24):
        print(nodes[i])
    
    for i in range(24):
        print(nodes[i])

    def is_terminal(self):
        pass

class Node:
    def __init__(self, index):
        self.index = index
        self.value = Player.NONE 
        self.neighbors = []
    
    def addNeighbor(self, node):
        self.neighbors.append(node)
        node.neighbors.append(self)

    def __str__(self):
        return f"value: {self.index} {[neighbor.index for neighbor in self.neighbors]}"

    def __repr__(self):
        return f"value: {self.index} {[neighbor.index for neighbor in self.neighbors]}"

class Weight(Enum):
    MATERIAL_WEIGHT = 100 
    POTENTIAL_MILL_WEIGHT = 50 
    BLOCKED_WEIGHT = 30
    MOBILITY_WEIGHT = 20
    DOUBLE_MILL_WEIGHT = 100

class Game:
    def __init__(self, max_depth):
        self.max_depth = max_depth

    def get_piece_count(self, state, agent):
            return state.pieces[agent]

    def nine_mens_morris_heuristic(self, state):
        player = state.current_player
        if player == Player.MAX:
            opponent = Player.MIN 
        else:
            opponent = Player.MAX

        def get_potential_mills(state, agent):
            count = 0
            for mill_line in MILL_LINES:  
                pieces = [state.board[pos] for pos in mill_line]
                if pieces.count(agent) == 2 and pieces.count(None) == 1:
                    count += 1
            return count

        def get_blocked_pieces(state, agent):
            blocked = 0
            for pos in state.board.get_positions(agent):
                if not state.board.has_legal_moves(pos):
                    blocked += 1
            return blocked

        def get_mobility(state, agent):
            if state.flying_allowed(agent):  
                return 3 * state.board.empty_spots()  
            return sum(len(state.board.legal_moves(pos)) for pos in state.board.get_positions(agent))

        def has_double_mill(state, agent):
            count = 0
            for pos in state.board.get_positions(agent):
                if count_double_mills(pos, agent, state.board):
                    count += 1
            return count

        score = 0
        
        player_pieces = self.get_piece_count(state, player)
        opponent_pieces = self.get_piece_count(state, opponent)
        score +=  MATERIAL_WEIGHT * (player_pieces - opponent_pieces)
        
        player_potential = get_potential_mills(state, player)
        opponent_potential = get_potential_mills(state, opponent)
        score += POTENTIAL_MILL_WEIGHT * (player_potential - opponent_potential)
        
        opponent_blocked = get_blocked_pieces(state, opponent)
        player_blocked = get_blocked_pieces(state, player)
        score += BLOCKED_WEIGHT * (opponent_blocked - player_blocked)
        
        player_mobility = get_mobility(state, player)
        opponent_mobility = get_mobility(state, opponent)
        score += MOBILITY_WEIGHT * (player_mobility - opponent_mobility)
        
        player_double = has_double_mill(state, player)
        opponent_double = has_double_mill(state, opponent)
        score += DOUBLE_MILL_WEIGHT * (player_double - opponent_double)

        if opponent_pieces <= 2: # won
            score += 1000  
        if player_pieces <= 2: # lost
            score -= 1000  
            
        if is_placement_phase(state):
            score += 20 * (player_potential - opponent_potential)
        else:
            score += 15 * (player_mobility - opponent_mobility)

        return score

    def is_terminal(self, state):
        return get_piece_count(state, Player.MAX) < 3 or get_piece_count(state, Player.MIN) < 3 or state.depth == self.max_depth

    def actions(self, state):
        moves = []
        if state.current_stage == 0: # can place pieces on any available square
            for tile in state.board:
                if tile.value is Player.None:
                    moves.append([tile.index, state.current_player])
        else: # can place pieces only on adjacent squares, unless number of pieces is 3
            if get_piece_count(state, state.current_player) == 3:
                for tile in state.board:
                    if tile.value is state.current.player:
                        for neighbor in tile.neighbors:
                            if neighbor.value is Player.None:
                                moves.append([neighbor.index, state.current_player])
            else: # same as in first stage
                for tile in state.board:
                    if tile.value is Player.None:
                        moves.append([tile.index, state.current_player])

    def utility(self, state):
        return self.nine_mens_morris_heuristic(state)

    def result(self, move):
        tile = move[0]
        player = move[1]
        
        if player is Player.MAX:
            opponent = Player.MIN 
        else:
            opponent = Player.MIN

        new_board = state.board.deepcopy()
        new_board.nodes[tile].value = player
        new_state = State(opponent, state.current_stage, new_board)
        return new_state

    def minmax(self):

        def minmax_search(self, state):    
            def max_value(state):
                if self.is_terminal(state):
                    return (self.utility(state), None)
                
                v = -float('inf')

                for a in self.actions(state):
                    v2, a2 = min_value(self.result(state, a))
                    if v2 > v:
                        v = v2 
                        move = a 

                return (v, move)

            def min_value(state):
                if self.is_terminal(state):
                    return (self.utility(state), None)

                v = float('inf')

                for a in self.actions(state):
                    v2, a2 = max_value(self.result(state, a))
                    if v2 < v:
                        v = v2 
                        move = a 

                return (v, move)

            player = game.to_move(state)
            value, move = max_value(state)
            return move

    def alpha_beta(self): 
        def alpha_beta_search(self, state):
            def max_value(state, alpha, beta):
                if self.is_terminal(state):
                    return (self.utility(state), None)

                v = -float('inf')

                for a in self.actions(state):
                    v2, a2 = min_value(self.result(state, a), alpha, beta)

                    if v2 > v:
                        v = v2 
                        move = a 
                        alpha = max(alpha, v)

                    if v >= beta:
                        return (v, move)

                return (v, move)

            def min_value(state, alpha, beta):
                if self.is_terminal(state):
                    return (self.utility(state), None)

                v = +float('inf')

                for a in self.actions(state):
                    v2, a2 = max_value(self.result(state, a), alpha, beta)

                    if v2 < v:
                        v = v2 
                        move = a 
                        beta = min(beta, v)

                    if v <= alpha:
                        return (v, move) 

                return (v, move)
        
            value, move = max_value(state, -float('inf'), float('inf'))
            return move
    
