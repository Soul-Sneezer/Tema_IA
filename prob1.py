import heapq

class Node:
    def __init__(self, value):
        self.value = value
        self.neighbors = []
        self.distance = [10000] * 24 # distance to other nodes
        self.distance[value] = 0

    def addNeighbor(self, node):
        self.distance[node.value] = 1
        self.neighbors.append(node)
        node.distance[self.value] = 1
        node.neighbors.append(self)

    def __str__(self):
        return f"value: {self.value} {[neighbor.value for neighbor in self.neighbors]}"

    def __repr__(self):
        return f"value: {self.value} {[neighbor.value for neighbor in self.neighbors]}"

class MountOlympus:
    def calculate_distances(self, graph):
        for k in range(24):
            for i in range(24):
                for j in range(24):
                    if graph[i].distance[j] > graph[i].distance[k] + graph[k].distance[j]:
                        graph[i].distance[j] = graph[i].distance[k] + graph[k].distance[j]

    def __init__(self):
        self.nodes = [Node(i) for i in range(24)]

        self.nodes[0].addNeighbor(self.nodes[1])
        self.nodes[0].addNeighbor(self.nodes[9])
        self.nodes[1].addNeighbor(self.nodes[2])
        self.nodes[1].addNeighbor(self.nodes[4])
        self.nodes[2].addNeighbor(self.nodes[14])
        self.nodes[3].addNeighbor(self.nodes[4])
        self.nodes[3].addNeighbor(self.nodes[10])
        self.nodes[4].addNeighbor(self.nodes[5])
        self.nodes[4].addNeighbor(self.nodes[7])
        self.nodes[5].addNeighbor(self.nodes[13])
        self.nodes[6].addNeighbor(self.nodes[7])
        self.nodes[6].addNeighbor(self.nodes[11])
        self.nodes[7].addNeighbor(self.nodes[8])
        self.nodes[8].addNeighbor(self.nodes[12])
        self.nodes[9].addNeighbor(self.nodes[10])
        self.nodes[9].addNeighbor(self.nodes[21])
        self.nodes[10].addNeighbor(self.nodes[11])
        self.nodes[10].addNeighbor(self.nodes[18])
        self.nodes[11].addNeighbor(self.nodes[15])
        self.nodes[12].addNeighbor(self.nodes[13])
        self.nodes[12].addNeighbor(self.nodes[17])
        self.nodes[13].addNeighbor(self.nodes[14])
        self.nodes[13].addNeighbor(self.nodes[20])
        self.nodes[14].addNeighbor(self.nodes[23])
        self.nodes[15].addNeighbor(self.nodes[16])
        self.nodes[16].addNeighbor(self.nodes[17])
        self.nodes[16].addNeighbor(self.nodes[19])
        self.nodes[18].addNeighbor(self.nodes[19])
        self.nodes[19].addNeighbor(self.nodes[20])
        self.nodes[19].addNeighbor(self.nodes[22])
        self.nodes[21].addNeighbor(self.nodes[22])
        self.nodes[22].addNeighbor(self.nodes[23])

        self.calculate_distances(self.nodes)
        #for i in range(0, 24):
        #    print(f"{i + 1} : {self.nodes[0].distance[i]}")
        #print(self.nodes)

    def query(self, query_id, start_node, goal_nodes, steps, algorithm):
        if algorithm == "A*":
            results = AStarSearch(start_node - 1, goal_nodes, test.nodes, steps, mst_heuristic)
        else:
            results = IDASearch(start_node - 1, goal_nodes, test.nodes, steps, mst_heuristic)

        for result in results:
            open_list.append(result[1] + 1)

        print(pid, open_list)

def nodes_to_explore(node, goals, visited_goals, graph): # nodes left to explore
    value = len(goals) - len(visited_goals) 
    #if node.value in goals:
    #    value -= 1 

    return value # with this heuristic A* basically devolves to a BFS (:

def mst_heuristic(node, goals, visited_goals, graph): 
    # o euristica este admisibila daca nu suparestimeaza costul drumului final
    # h(n) = suma muchiilor din arborele partial de cost minim
    # P este drumul optim de la start care viziteaza toate nodurile goal 
    # P trebuie sa formeze un subgraf conex care cuprinde nodurile goal si nodul start 
    # arborele partial de cost minim format de noi este subgraful conex minimal care cuprinde toate nodurile
    # din asta rezulta: h(n) <= costul drumului final

    # euristica este de asemenea consistenta deoarece respecta inegalitatea triunghiului
    # h(n) <= c(n, n') + h(n'), c(n, n') = 1 
    # h(n) <= 1 + h(n'), h(n')= h(n) - 1 daca n' este unul din nodurile goal, deoarece am elimina doar muchia dintre n si n'
    # daca n' este un nod intermediar,distanta dintre n' si cel mai apropiat nod goal nu poate fi mai mica decat distanta dintre n si cel mai 
    # apropiat nod goal - 1
    
    # euristica este eficienta deoarece ajunge la o solutie mai repede decat BFS 
    # ceea ce se poate observa in practica cand comparam 'mst_heuristic' cu 'nodes_to_explore',
    # cand folosim 'nodes_to_explore' A* degenereaza la un BFS, deoarece f este constant(sau variaza cu +- 1)

    goals_to_visit = [goal for goal in goals if goal not in visited_goals]

    if len(goals_to_visit) <= 1:
        return 0

    visited = set()
    heap = []
    total_cost = 0

    visited.add(node.value)
    
    for goal in goals_to_visit:
        cost = graph[node.value].distance[goal]
        heapq.heappush(heap, (cost, node.value, goal))

    while len(visited) < len(goals_to_visit):
        cost, u, v = heapq.heappop(heap)

        if v not in visited:
            visited.add(v)
            total_cost += cost 

            for w in goals_to_visit:
                if w not in visited and w != v:
                    new_cost = graph[v].distance[w]
                    heapq.heappush(heap, (new_cost, v, w))

    return total_cost 

def closest_unexplored_node(node, goals, visited_goals):
    shortest_dist = 10000 
    closest_goal = None
    for goal in goals:
        if goal not in visited_goals:
            if node.distance[goal] < shortest_dist:
                shortest_dist = node.distance[goal]
    
    return shortest_dist

def AStarSearch(start, goals, graph, steps, heuristic, steps_taken=0, max_fitness=10000, iterative_deepening=False):
    frontier = []
    heapq.heappush(frontier, (heuristic(graph[start], goals, frozenset(), graph), start, 0, frozenset()))
     
    while frontier:
        steps_taken += 1 
        f, value, g, reached_goals = heapq.heappop(frontier)
        node = graph[value]
        
        if value in goals:
            new_reached_goals = reached_goals.union([value])
        else:
            new_reached_goals = reached_goals
        #print(new_reached_goals, len(new_reached_goals))
        
        for neighbor in node.neighbors:
            f = g + heuristic(neighbor, goals, new_reached_goals, graph)
            #print(f)
            # print(f) 
            heapq.heappush(frontier, (f, neighbor.value, g + 1, new_reached_goals))

        if steps_taken == steps:
            return frontier

        if iterative_deepening:
            f = min(g + heuristic(neighbor, goals, new_reached_goals, graph) for neighbor in node.neighbors)
            if f < max_fitness:
                return (f, steps_taken)
        
    return (None, None)

def IDASearch(start, goals, graph, steps, heuristic):
    steps_taken = 0
    f = 1000
    while steps_taken < steps:
        result = AStarSearch(start, goals, graph, steps, heuristic, steps_taken, f, True)
        if type(result) == tuple:
            f = result[0]
            steps_taken = result[1]
        else:
            return result

test = MountOlympus()

open_list = []

pid = input()
start_node = int(input())
goal_nodes = list(map(int, input().split()))

for i in range(len(goal_nodes)):
    goal_nodes[i] -= 1

print (goal_nodes)

steps = int(input())
algorithm = input()

test.query(pid, start_node, goal_nodes, steps, algorithm)
