from enum import Enum

class Player(Enum):
    MAX = 0
    MIN = 1
    NONE = 2

class State:
    def __init__(self, player, current_stage, board):
        self.player = player
        self.current_stage = current_stage
        self.board = board

    def next_moves(self):
        states = []
        if current_stage == 0:
            for node in board:
                pass 
        elif current_stage == 1:
            pass 
        else:
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

create_board()
