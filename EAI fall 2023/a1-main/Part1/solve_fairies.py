import sys
from queue import PriorityQueue

def h(s):
    """
    Heuristic function: counts the number of fairies out of place.
    """
    return sum(1 for i, fairy in enumerate(s) if fairy != i+1)

def get_neighbors(state):
    """
    Generates neighboring states by swapping adjacent fairies.
    """
    neighbors = []
    for i in range(len(state)-1):
        if state[i] != state[i+1]:
            swapped = state.copy()
            swapped[i], swapped[i+1] = swapped[i+1], swapped[i]
            neighbors.append(swapped)
    return neighbors

def solve(initial_state):
    """
    A* search algorithm to solve the fairy rearrangement problem.
    """
    visited = set()
    pq = PriorityQueue()  # priority queue
    pq.put((0 + h(initial_state), (initial_state, 0)))  # (f(s), (state, g(s)))
    parents = {tuple(initial_state): None}  # Track the previous state that led to the current state

    while not pq.empty():
        priority, (state, cost) = pq.get()
        tuple_state = tuple(state)
        if tuple_state in visited:
            continue
        visited.add(tuple_state)

        if state == list(range(1, len(state)+1)):  # Check if state is the goal
            path = []
            while state:
                path.append(state)
                state = parents[tuple(state)]
            return path[::-1]  # Reverse the path for start-to-goal order

        for neighbor in get_neighbors(state):
            if tuple(neighbor) not in visited:
                parents[tuple(neighbor)] = state
                new_cost = cost + 1
                pq.put((new_cost + h(neighbor), (neighbor, new_cost)))

    return []  # No solution found

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise(Exception("Error: expected a test case filename"))

    test_cases = []
    with open(sys.argv[1], 'r') as file:
        for line in file:
            test_cases.append([int(i) for i in line.split()])
    for initial_state in test_cases:
        path = solve(initial_state)
        print('From state ' + str(initial_state) + " found goal state by taking path: " + str(path))

