import numpy as np
from pomegranate import BayesianNetwork, ConditionalProbabilityTable, DiscreteDistribution

class BayesianMoveSelector:
    def __init__(self):
        # Define the states our variables can take
        self.piece_count_states = ['few', 'medium', 'many']  # 0-3, 4-6, 7-9
        self.position_states = ['weak', 'neutral', 'strong']
        self.mill_potential_states = ['low', 'medium', 'high']
        self.mobility_states = ['low', 'medium', 'high']
        self.move_quality_states = ['poor', 'decent', 'good', 'excellent']
        
        # Create the Bayesian Network
        self.network = self._create_network()
        
    def _create_network(self):
        # Create individual probability distributions
        
        # Prior probability for piece count distribution
        piece_count = DiscreteDistribution({
            'few': 0.3,
            'medium': 0.4,
            'many': 0.3
        })
        
        # Prior probability for position strength
        position = DiscreteDistribution({
            'weak': 0.3,
            'neutral': 0.4,
            'strong': 0.3
        })
        
        # Conditional probability for mill potential based on piece count and position
        mill_potential_probs = [
            # piece_count, position, mill_potential, probability
            ['few', 'weak', 'low', 0.7],
            ['few', 'weak', 'medium', 0.2],
            ['few', 'weak', 'high', 0.1],
            ['few', 'neutral', 'low', 0.5],
            ['few', 'neutral', 'medium', 0.3],
            ['few', 'neutral', 'high', 0.2],
            # ... (add all other combinations)
            ['many', 'strong', 'low', 0.1],
            ['many', 'strong', 'medium', 0.3],
            ['many', 'strong', 'high', 0.6]
        ]
        
        mill_potential = ConditionalProbabilityTable(
            mill_potential_probs,
            [piece_count, position]
        )
        
        # Conditional probability for mobility
        mobility_probs = [
            # piece_count, position, mobility, probability
            ['few', 'weak', 'low', 0.6],
            ['few', 'weak', 'medium', 0.3],
            ['few', 'weak', 'high', 0.1],
            # ... (add all other combinations)
            ['many', 'strong', 'low', 0.2],
            ['many', 'strong', 'medium', 0.3],
            ['many', 'strong', 'high', 0.5]
        ]
        
        mobility = ConditionalProbabilityTable(
            mobility_probs,
            [piece_count, position]
        )
        
        # Conditional probability for move quality based on all other factors
        move_quality_probs = [
            # mill_potential, mobility, move_quality, probability
            ['low', 'low', 'poor', 0.7],
            ['low', 'low', 'decent', 0.2],
            ['low', 'low', 'good', 0.1],
            ['low', 'low', 'excellent', 0.0],
            # ... (add all other combinations)
            ['high', 'high', 'poor', 0.0],
            ['high', 'high', 'decent', 0.1],
            ['high', 'high', 'good', 0.3],
            ['high', 'high', 'excellent', 0.6]
        ]
        
        move_quality = ConditionalProbabilityTable(
            move_quality_probs,
            [mill_potential, mobility]
        )
        
        # Create and return the Bayesian Network
        network = BayesianNetwork()
        network.add_states(piece_count, position, mill_potential, mobility, move_quality)
        
        # Add edges (dependencies)
        network.add_edge(piece_count, mill_potential)
        network.add_edge(position, mill_potential)
        network.add_edge(piece_count, mobility)
        network.add_edge(position, mobility)
        network.add_edge(mill_potential, move_quality)
        network.add_edge(mobility, move_quality)
        
        # Bake (finalize) the network
        network.bake()
        return network

    def _categorize_piece_count(self, count):
        if count <= 3:
            return 'few'
        elif count <= 6:
            return 'medium'
        else:
            return 'many'

    def _categorize_position(self, state, position):
        """Evaluate the strength of a position"""
        control_center = {4, 10, 13, 19}  # Central positions
        edge_positions = {1, 7, 9, 11, 12, 14, 16, 22}  # Edge positions
        
        if position in control_center:
            return 'strong'
        elif position in edge_positions:
            return 'neutral'
        return 'weak'

    def _evaluate_mill_potential(self, state, move):
        """Count potential mills after making the move"""
        new_state, formed_mill = state.make_move(move)
        if formed_mill:
            return 'high'
            
        potential_mills = state._count_potential_mills(new_state, state.current_player)
        if potential_mills >= 2:
            return 'high'
        elif potential_mills == 1:
            return 'medium'
        return 'low'

    def _evaluate_mobility(self, state, move):
        """Evaluate mobility after making the move"""
        new_state, _ = state.make_move(move)
        moves = len(new_state.get_legal_moves())
        
        if moves >= 6:
            return 'high'
        elif moves >= 3:
            return 'medium'
        return 'low'

    def select_move(self, state, legal_moves):
        """Select the best move using the Bayesian network"""
        if not legal_moves:
            return None
            
        move_probabilities = []
        
        for move in legal_moves:
            # Gather evidence for the network
            piece_count = self._categorize_piece_count(state.get_piece_count(state.current_player))
            position = self._categorize_position(state, move[1])  # move[1] is the destination
            mill_potential = self._evaluate_mill_potential(state, move)
            mobility = self._evaluate_mobility(state, move)
            
            # Query the network
            evidence = {
                'piece_count': piece_count,
                'position': position,
                'mill_potential': mill_potential,
                'mobility': mobility
            }
            
            # Get probability distribution for move quality
            beliefs = self.network.predict_proba(evidence)
            move_quality_dist = beliefs[-1]  # Last node is move_quality
            
            # Calculate expected value of the move
            expected_value = (
                move_quality_dist['poor'] * 0.0 +
                move_quality_dist['decent'] * 0.33 +
                move_quality_dist['good'] * 0.67 +
                move_quality_dist['excellent'] * 1.0
            )
            
            move_probabilities.append((move, expected_value))
        
        # Select the move with highest expected value
        best_move = max(move_probabilities, key=lambda x: x[1])[0]
        return best_move

class BayesianGame(Game):
    def __init__(self, max_depth=4):
        super().__init__(max_depth)
        self.bayesian_selector = BayesianMoveSelector()
        
    def get_move(self, state):
        """Get the best move using Bayesian network"""
        legal_moves = self.get_legal_moves(state)
        return self.bayesian_selector.select_move(state, legal_moves)
