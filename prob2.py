from enum import Enum
import random
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
    def create_board(board_str=""):
        nodes = []
        for i in range(24):
            nodes.append(Node(i))
        
        for node_idx, neighbors in BOARD_CONNECTIONS.items():
            for neighbor_idx in neighbors:
                nodes[node_idx].add_neighbor(nodes[neighbor_idx])
        
        if len(board_str) == 24:
            for i in range(24):
                if board_str[i] == ' ':
                    nodes[i].value = Player.NONE 
                elif board_str[i] == 'x':
                    nodes[i].value = Player.MAX
                else:
                    nodes[i].value = Player.MIN

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

    #def __str__(self):
    #    return f"Node {self.index}: value={self.value}, neighbors={[n.index for n in self.neighbors]}"

    def __str__(self):
        if self.value == Player.NONE:
            return " "
        elif self.value == Player.MAX:
            return "x"
        else:
            return "o"

    def __repr__(self):
        if self.value == Player.NONE:
            return " "
        elif self.value == Player.MAX:
            return "x"
        else:
            return "o"

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

    def query(self, pid, initial_state, algorithm, depth):
        self.depth = depth
        if algorithm == "Minmax":
            print(pid, self.make_move(initial_state, self.minmax_search(initial_state))[0].board)
        else:
            print(pid, self.make_move(initial_state, self.alpha_beta_search(initial_state))[0].board)


def print_board(state):
    print(state.board)

class BayesianNode:
    def __init__(self, name, states, probabilities):
        self.name = name
        self.states = states
        self.probabilities = probabilities
        self.parents = []
        
    def add_parent(self, parent):
        self.parents.append(parent)
        
    def get_probability(self, state_idx, parent_states):
        return self.probabilities[parent_states][state_idx]

class BayesianNetwork:
    def __init__(self):
        self.nodes = {}
        
    def add_node(self, node):
        self.nodes[node.name] = node
        
    def add_edge(self, parent_name, child_name):
        self.nodes[child_name].add_parent(self.nodes[parent_name])
        
    def predict_proba(self, evidence):
        result = {}
        for node_name, node in self.nodes.items():
            if node_name in evidence:
                probs = [0.0] * len(node.states)
                probs[evidence[node_name]] = 1.0
                result[node_name] = probs
            else:
                parent_states = tuple(evidence[parent.name] for parent in node.parents)
                result[node_name] = node.probabilities[parent_states]
        return result

class BayesianMoveSelector:
    def __init__(self):
        self.piece_count_states = ['few', 'medium', 'many']  # 0-3, 4-6, 7-9
        self.position_states = ['weak', 'neutral', 'strong']
        self.mill_potential_states = ['low', 'medium', 'high']
        self.mobility_states = ['low', 'medium', 'high']
        self.move_quality_states = ['poor', 'decent', 'good', 'excellent']
        
        self.network = self._create_network()
    
    def _create_network(self):
        network = BayesianNetwork()
        
        piece_count = BayesianNode(
            name='piece_count',
            states=self.piece_count_states,
            probabilities={(): [0.3, 0.4, 0.3]}  # Prior probabilities
        )
        
        position = BayesianNode(
            name='position',
            states=self.position_states,
            probabilities={(): [0.3, 0.4, 0.3]}  # Prior probabilities
        )
        
        mill_potential_probs = {}
        for pc in range(3):  # piece count states
            for pos in range(3):  # position states
                if pc == 0 and pos == 0:  # few pieces, weak position
                    mill_potential_probs[(pc, pos)] = [0.7, 0.2, 0.1]
                elif pc == 2 and pos == 2:  # many pieces, strong position
                    mill_potential_probs[(pc, pos)] = [0.1, 0.3, 0.6]
                else:
                    mill_potential_probs[(pc, pos)] = [0.4, 0.3, 0.3]
        
        mill_potential = BayesianNode(
            name='mill_potential',
            states=self.mill_potential_states,
            probabilities=mill_potential_probs
        )
        
        mobility_probs = {}
        for pc in range(3):
            for pos in range(3):
                if pc == 0 and pos == 0:  # few pieces, weak position
                    mobility_probs[(pc, pos)] = [0.6, 0.3, 0.1]
                elif pc == 2 and pos == 2:  # many pieces, strong position
                    mobility_probs[(pc, pos)] = [0.2, 0.3, 0.5]
                else:
                    mobility_probs[(pc, pos)] = [0.3, 0.4, 0.3]
        
        mobility = BayesianNode(
            name='mobility',
            states=self.mobility_states,
            probabilities=mobility_probs
        )
        
        move_quality_probs = {}
        for mp in range(3):
            for mob in range(3):
                if mp == 0 and mob == 0:  # low mill potential, low mobility
                    move_quality_probs[(mp, mob)] = [0.7, 0.2, 0.1, 0.0]
                elif mp == 2 and mob == 2:  # high mill potential, high mobility
                    move_quality_probs[(mp, mob)] = [0.0, 0.1, 0.3, 0.6]
                else:
                    move_quality_probs[(mp, mob)] = [0.2, 0.3, 0.3, 0.2]
        
        move_quality = BayesianNode(
            name='move_quality',
            states=self.move_quality_states,
            probabilities=move_quality_probs
        )
        
        network.add_node(piece_count)
        network.add_node(position)
        network.add_node(mill_potential)
        network.add_node(mobility)
        network.add_node(move_quality)
        
        network.add_edge('piece_count', 'mill_potential')
        network.add_edge('position', 'mill_potential')
        network.add_edge('piece_count', 'mobility')
        network.add_edge('position', 'mobility')
        network.add_edge('mill_potential', 'move_quality')
        network.add_edge('mobility', 'move_quality')
        
        return network
    
    def select_move(self, game, state, legal_moves):
        if not legal_moves:
            return None
            
        move_probabilities: List[Tuple] = []
        
        for move in legal_moves:
            evidence = {
                'piece_count': self._categorize_piece_count_to_index(state.get_piece_count(state.current_player)),
                'position': self._categorize_position_to_index(state, move[1]),
                'mill_potential': self._evaluate_mill_potential_to_index(game, state, move),
                'mobility': self._evaluate_mobility_to_index(game, state, move)
            }
            
            predictions = self.network.predict_proba(evidence)
            move_quality_probs = predictions['move_quality']
            
            expected_value = (
                move_quality_probs[0] * 0.0 +   # poor
                move_quality_probs[1] * 0.33 +  # decent
                move_quality_probs[2] * 0.67 +  # good
                move_quality_probs[3] * 1.0     # excellent
            )
            
            move_probabilities.append((move, expected_value))
        
        return max(move_probabilities, key=lambda x: x[1])[0]
    
    def _categorize_piece_count_to_index(self, count):
        if count <= 3:
            return 0  # few
        elif count <= 6:
            return 1  # medium
        return 2  # many
    
    def _categorize_position_to_index(self, state, position):
        control_center = {4, 10, 13, 19}
        edge_positions = {1, 7, 9, 11, 12, 14, 16, 22}
        
        if position in control_center:
            return 2  # strong
        elif position in edge_positions:
            return 1  # neutral
        return 0  # weak
    
    def _evaluate_mill_potential_to_index(self, game, state, move):
        new_state, formed_mill = game.make_move(state, move)
        if formed_mill:
            return 2  # high
            
        potential_mills = game._count_potential_mills(new_state, state.current_player)
        if potential_mills >= 2:
            return 2  # high
        elif potential_mills == 1:
            return 1  # medium
        return 0  # low
    
    def _evaluate_mobility_to_index(self, game, state, move):
        new_state, _ = game.make_move(state, move)
        moves = len(game.get_legal_moves(state))
        
        if moves >= 6:
            return 2  # high
        elif moves >= 3:
            return 1  # medium
        return 0  # low

class BayesianGame(Game):
    def __init__(self, max_depth: int = 4) :
        self.max_depth = max_depth
        self.bayesian_selector = BayesianMoveSelector()
        
    def get_move(self, state):
        legal_moves = self.get_legal_moves(state)
        return self.bayesian_selector.select_move(self, state, legal_moves)

game = Game(3)
bayes_game = BayesianGame(3)

initial_board = State.create_board("     x  o xxox   oxox o ")
initial_state = State(Player.MAX, 0, initial_board)

print(bayes_game.get_move(initial_state))

game.query(360, initial_state, "Minmax", 3)
