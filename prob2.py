from enum import Enum
import copy

class Player(Enum):
    MAX = 0
    MIN = 1
    NONE = 2

class H_Weights(Enum):
    PLACEHOLDER = 0

MILL_LINES = [
    # Horizontal mills
    [0, 1, 2], [3, 4, 5], [6, 7, 8],
    [9, 10, 11], [12, 13, 14], [15, 16, 17],
    [18, 19, 20], [21, 22, 23],
    # Vertical mills
    [0, 9, 21], [3, 10, 18],
    [6, 11, 15], [1, 4, 7],
    [16, 19, 22], [8, 12, 17],
    [5, 13, 20], [2, 14, 23]
]

BOARD_CONNECTIONS = {
    0: [1, 9],
    1: [0, 2, 4],
    2: [1, 14],
    3: [4, 10],
    4: [1, 3, 5, 7],
    5: [4, 13],
    6: [7, 11],
    7: [4, 6, 8],
    8: [7, 12],
    9: [0, 10, 21],
    10: [3, 9, 11, 18],
    11: [6, 10, 15],
    12: [8, 13, 17],
    13: [5, 12, 14, 20],
    14: [2, 13, 23],
    15: [11, 16],
    16: [15, 17, 19],
    17: [12, 16],
    18: [10, 19],
    19: [16, 18, 20, 22],
    20: [13, 19],
    21: [9, 22],
    22: [19, 21, 23],
    23: [14, 22]
}

class State:
    def __init__(self, current_player, current_stage, board, pieces_to_place=None):
        self.current_player = current_player
        self.current_stage = current_stage  # 0: Placing, 1: Moving, 2: Flying
        self.board = board
        self.depth = 0
        self.pieces_to_place = pieces_to_place or {Player.MAX: 9, Player.MIN: 9}
        
    def copy(self):
        new_state = State(
            self.current_player,
            self.current_stage,
            copy.deepcopy(self.board),
            copy.deepcopy(self.pieces_to_place)
        )
        new_state.depth = self.depth
        return new_state

    @staticmethod
    def create_board():
        nodes = []
        for i in range(24):
            nodes.append(Node(i))
        
        for node_idx, neighbors in BOARD_CONNECTIONS.items():
            for neighbor_idx in neighbors:
                nodes[node_idx].add_neighbor(nodes[neighbor_idx])
                
        return nodes

    def get_piece_count(self, player):
        return sum(1 for node in self.board if node.value == player)

    def is_mill(self, position):
        player = self.board[position].value
        if player == Player.NONE:
            return False
            
        for mill in MILL_LINES:
            if position in mill:
                if all(self.board[pos].value == player for pos in mill):
                    return True
        return False

    def check_new_mill(self, position, previous_board):
        if not self.is_mill(position):
            return False
            
        for mill in MILL_LINES:
            if position in mill:
                if all(self.board[pos].value == self.board[position].value for pos in mill):
                    if not all(previous_board[pos].value == self.board[position].value for pos in mill):
                        return True
        return False

    def get_removable_pieces(self, player):
        opponent = Player.MIN if player == Player.MAX else Player.MAX
        removable = []
        
        for i, node in enumerate(self.board):
            if node.value == opponent and not self.is_mill(i):
                removable.append(i)
                
        if not removable:
            removable = [i for i, node in enumerate(self.board) if node.value == opponent]
            
        return removable

class Node:
    def __init__(self, index):
        self.index = index
        self.value = Player.NONE
        self.neighbors = []

    def add_neighbor(self, node):
        if node not in self.neighbors:
            self.neighbors.append(node)
            node.neighbors.append(self)

    def __str__(self):
        return f"Node {self.index}: value={self.value}, neighbors={[n.index for n in self.neighbors]}"

class Game:
    def __init__(self, max_depth=4):
        self.max_depth = max_depth

    def is_terminal(self, state):
        if state.depth >= self.max_depth:
            return True

        if state.current_stage > 0:  
            for player in [Player.MAX, Player.MIN]:
                if state.get_piece_count(player) < 3:
                    return True
                    
        if not self.get_legal_moves(state):
            return True
            
        return state.depth >= self.max_depth

    def get_legal_moves(self, state):
        moves = []
        
        if state.current_stage == 0:  # Placing phase
            if state.pieces_to_place[state.current_player] > 0:
                for i, node in enumerate(state.board):
                    if node.value == Player.NONE:
                        moves.append(("PLACE", i))
        else:  
            is_flying = (state.current_stage == 2 or 
                        state.get_piece_count(state.current_player) == 3)
            
            for i, node in enumerate(state.board):
                if node.value == state.current_player:
                    if is_flying:
                        destinations = [j for j, dest in enumerate(state.board) 
                                     if dest.value == Player.NONE]
                    else:
                        destinations = [n.index for n in node.neighbors 
                                     if n.value == Player.NONE]
                    
                    for dest in destinations:
                        moves.append(("MOVE", i, dest))
                        
        return moves

    def make_move(self, state, move):
        new_state = state.copy()
        new_state.depth += 1
        
        previous_board = copy.deepcopy(new_state.board)
        move_type = move[0]
        
        if move_type == "PLACE":
            position = move[1]
            new_state.board[position].value = state.current_player
            new_state.pieces_to_place[state.current_player] -= 1
            
            if (new_state.pieces_to_place[Player.MAX] == 0 and 
                new_state.pieces_to_place[Player.MIN] == 0):
                new_state.current_stage = 1  # Move to moving phase
                
        elif move_type == "MOVE":
            origin, dest = move[1], move[2]
            new_state.board[dest].value = state.current_player
            new_state.board[origin].value = Player.NONE
            
        if new_state.check_new_mill(move[1] if move_type == "PLACE" else move[2], 
                                  previous_board):
            return new_state, True  # Indicate that a mill was formed
            
        new_state.current_player = (Player.MIN if state.current_player == Player.MAX 
                                  else Player.MAX)
        
        if new_state.current_stage > 0:
            for player in [Player.MAX, Player.MIN]:
                if new_state.get_piece_count(player) == 3:
                    new_state.current_stage = 2
                    
        return new_state, False

    def remove_piece(self, state, position):
        new_state = state.copy()
        new_state.board[position].value = Player.NONE
        
        new_state.current_player = (Player.MIN if state.current_player == Player.MAX else Player.MAX)
        return new_state

    def evaluate(self, state):
        if state.current_stage == 0: # place  
            return self._evaluate_placing_phase(state)
        else:  # move or fly
            return self._evaluate_moving_phase(state)

    def _evaluate_placing_phase(self, state):
        score = 0
        
        # Count pieces
        max_pieces = state.get_piece_count(Player.MAX)
        min_pieces = state.get_piece_count(Player.MIN)
        score += 100 * (max_pieces - min_pieces)
        
        # Count potential mills
        score += 50 * self._count_potential_mills(state, Player.MAX)
        score -= 50 * self._count_potential_mills(state, Player.MIN)
        
        return score

    def _evaluate_moving_phase(self, state):
        score = 0
        
        max_pieces = state.get_piece_count(Player.MAX)
        min_pieces = state.get_piece_count(Player.MIN)
        
        if min_pieces < 3:
            return 10000  # MAX wins
        if max_pieces < 3:
            return -10000  # MIN wins
            
        score += 200 * (max_pieces - min_pieces)
        
        max_moves = len(self._get_player_moves(state, Player.MAX))
        min_moves = len(self._get_player_moves(state, Player.MIN))
        score += 50 * (max_moves - min_moves)
        
        score += 30 * self._count_potential_mills(state, Player.MAX)
        score -= 30 * self._count_potential_mills(state, Player.MIN)
        
        return score

    def _count_potential_mills(self, state, player):
        count = 0
        for mill in MILL_LINES:
            pieces = [state.board[pos].value for pos in mill]
            if pieces.count(player) == 2 and pieces.count(Player.NONE) == 1:
                count += 1
        return count

    def _get_player_moves(self, state, player):
        saved_player = state.current_player
        state.current_player = player
        moves = self.get_legal_moves(state)
        state.current_player = saved_player
        return moves

    def minmax_search(self, state):
        def max_value(state, depth):  
            if self.is_terminal(state) or depth >= self.max_depth:  
                return self.evaluate(state), None
            v = float('-inf')
            best_move = None

            for move in self.get_legal_moves(state):
                new_state, formed_mill = self.make_move(state, move)

                if formed_mill:
                    removable = new_state.get_removable_pieces(state.current_player)
                    best_removal_value = float('-inf')

                    for remove_pos in removable:
                        after_remove = self.remove_piece(new_state, remove_pos)
                        value, _ = min_value(after_remove, depth + 1)  
                        best_removal_value = max(best_removal_value, value)

                    if best_removal_value > v:
                        v = best_removal_value
                        best_move = move
                else:
                    value, _ = min_value(new_state, depth + 1)  
                    if value > v:
                        v = value
                        best_move = move

            return v, best_move

        def min_value(state, depth):  
            if self.is_terminal(state) or depth >= self.max_depth:  
                return self.evaluate(state), None

            v = float('inf')
            best_move = None

            for move in self.get_legal_moves(state):
                new_state, formed_mill = self.make_move(state, move)

                if formed_mill:
                    removable = new_state.get_removable_pieces(state.current_player)
                    best_removal_value = float('inf')

                    for remove_pos in removable:
                        after_remove = self.remove_piece(new_state, remove_pos)
                        value, _ = max_value(after_remove, depth + 1)  
                        best_removal_value = min(best_removal_value, value)

                    if best_removal_value < v:
                        v = best_removal_value
                        best_move = move
                else:
                    value, _ = max_value(new_state, depth + 1)  
                    if value < v:
                        v = value
                        best_move = move

            return v, best_move

        # Start the minimax search at depth 0
        _, move = max_value(state, 0) if state.current_player == Player.MAX else min_value(state, 0)
        return move

    def alpha_beta_search(self, state):
        def max_value(state, alpha, beta, depth):
            if self.is_terminal(state) or depth >= self.max_depth:
                return self.evaluate(state), None

            v = float('-inf')
            best_move = None
            
            for move in self.get_legal_moves(state):
                new_state, formed_mill = self.make_move(state, move)
                
                if formed_mill:
                    # Try each possible piece removal
                    removable = new_state.get_removable_pieces(state.current_player)
                    best_removal_value = float('-inf')
                    
                    for remove_pos in removable:
                        after_remove = self.remove_piece(new_state, remove_pos)
                        value, _ = min_value(after_remove, alpha, beta, depth + 1)
                        best_removal_value = max(best_removal_value, value)
                    
                    if best_removal_value > v:
                        v = best_removal_value
                        best_move = move
                else:
                    value, _ = min_value(new_state, alpha, beta, depth + 1)
                    if value > v:
                        v = value
                        best_move = move
                
                alpha = max(alpha, v)
                if v >= beta:
                    return v, best_move
                    
            return v, best_move

        def min_value(state, alpha, beta, depth):
            if self.is_terminal(state) or depth >= self.max_depth:
                return self.evaluate(state), None

            v = float('inf')
            best_move = None
            
            for move in self.get_legal_moves(state):
                new_state, formed_mill = self.make_move(state, move)
                
                if formed_mill:
                    # Try each possible piece removal
                    removable = new_state.get_removable_pieces(state.current_player)
                    best_removal_value = float('inf')
                    
                    for remove_pos in removable:
                        after_remove = self.remove_piece(new_state, remove_pos)
                        value, _ = max_value(after_remove, alpha, beta, depth + 1)
                        best_removal_value = min(best_removal_value, value)
                    
                    if best_removal_value < v:
                        v = best_removal_value
                        best_move = move
                else:
                    value, _ = max_value(new_state, alpha, beta, depth + 1)
                    if value < v:
                        v = value
                        best_move = move
                
                beta = min(beta, v)
                if v <= alpha:
                    return v, best_move
                    
            return v, best_move

        # Start the alpha-beta search
        _, move = max_value(state, float('-inf'), float('inf'), 0)
        return move

def print_board(state):
    pass

def compare_searches():
    pass

game = Game(3)
initial_board = State.create_board()
initial_state = State(Player.MAX, 0, initial_board)

print(game.minmax_search(initial_state))
