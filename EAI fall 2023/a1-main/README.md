# Assignment 1

## Report

### Part1
---
(1) Formulation of the Search Problem:

State Space: The state space represents all possible configurations of the fairies' symbols. A state is a list of integers, where each integer is a fairy's symbol, and its position in the list denotes the fairy's order.

Successor Function: This function, named get_neighbors(state), produces neighboring states by swapping adjacent fairies that are not already in ascending order. Each swap forms a new neighboring state.

Edge Weights: The cost between any two adjacent states (i.e., after a single swap) is uniform and is equal to 1. This is evident as each transition between a state and its neighbor has an associated cost increment of 1 in the A* algorithm.

Goal State: The goal is for the list to be in ascending order, representing the fairies' proper order from 1 to N (where N is the length of the state, inferred from its size in the given code).

Heuristic Function(s):

The heuristic function h(s) counts the number of fairies out of their designated position. This heuristic is admissible, as it never overestimates the real cost. It only provides a lower-bound estimation of the number of swaps required to place all fairies in their goal positions.

(2) Description of the Search Algorithm:

The code implements the A* search algorithm with the following steps:

- A set, visited, is maintained to track states that have already been explored.
- A priority queue, pq, manages states based on their combined heuristic and path costs (f(s)).
- The initial state, with its f(s) value, is pushed into pq.
- A dictionary, parents, retains the previous state from which each state was reached.
- The main loop proceeds while pq has states:
The state with the highest priority is dequeued.
  -  If the dequeued state has been visited before, it's skipped.
  -  If the state matches the goal state (ascending order), a path to this state is reconstructed using the parents dictionary and returned.
  -  Else, the neighbors of the state (possible swaps) are generated.
  -  For each neighbor, if it's not visited, it's pushed into pq with its associated f(s) value and its preceding state is updated in the parents dictionary.
If the loop completes without finding a solution, an empty list is returned, indicating no solution.

(3) Discussion of Problems, Assumptions, Simplifications, and Design Decisions:

Heuristic Choice: The heuristic function, h(s), was chosen for its simplicity, efficiency, and its admissible property. This heuristic provides a good balance between speed and accuracy in the context of this problem.

Data Structures: The use of lists for states and a dictionary to store the relationship between states (child-parent) is a judicious choice. Tuples represent states in dictionaries, ensuring hashability and immutability, which is crucial for lookups in the visited set and the parents dictionary.

Assumptions: The code presumes that the input states provided are well-structured and solvable. However, it does not incorporate explicit checks for unsolvable cases.

Simplifications: A notable simplification in the successor function (get_neighbors) is that it only generates swaps for adjacent fairies that aren't in the right order. This ensures that the algorithm doesn't produce unnecessary states, improving efficiency.

Efficiency and Design Decisions: Using a priority queue ensures that the state with the highest potential to reach the goal (lowest f(s)) is explored first. This combined with an admissible heuristic boosts the search efficiency. The code's design is modular, with separate functions handling different aspects of the search algorithm, which enhances readability and maintainability.

### Part2
---
For problem 2 we considered this problem like 8-puzzle and proceeded accordingly. Since, we have more moves then a traditional 8-puzzle problem so for each of the action there are more possibilties. Also, because of the rotational functionality of the puzzle, similar to torus puzzle we needed to consider the possibility of the pieces to be in correct place with less moves. At first, we checked with our initial heuristic cost `f(n)` with number of moves made `g(n)` and incorrectly placed pieces `h(n)`. Our heuristic was set on considering that each pieces can be directly move to its correct position in one move. However this did not work and we had to use Manhattan distance for `h(n)` to make it more admissible. With this heuristic cost we were stuck with 15 moves for `board1`, although correctly working for `board0`. In order for `board1` to work with, we realized that for rotational base moves we were overestimating our heuristic value. So in order for it to work we divided each distance by 2, underestimating our heuristic cost but still making it admissible and consistent. Thus we were able to solve `board1` in 11 moves. We tested out on bunch of other self-made board for which it was generating optimal move set. Now to answer the question,
1. In this problem, what is the branching factor of the search tree?
Ans. Since we had a total of 24 possible moves from any state and we could perform all the moves regardless of the state, our branching factor for the search tree was the number of possible moves.
2. If the solution can be reached in 7 moves, about how many states would we need to explore before we found it if we
used BFS instead of A* search? A rough answer is fine
Ans. If the solution can be found in 7 moves then the solution will be at depth 7 considering the initial starting state is at depth 0. Now BFS search algorithm expands and traverse all the child nodes first. For this problem, the BFS search algorithm will expand all the immediate 24 child node from state 0 and then will proceed to expand all 24 child node for all the 24 child nodes. So, for each level it will visit all the nodes and following this it will take BFS roughly 24<sup>7</sup> nodes to visit and find the solution.

### Part3
---
This code looks to be a Python programme for determining a route between two cities and computing a number of route-related metrics, including the quantity of road segments, total miles, total hours, and total delivery hours. To find the optimum route based on the selected cost function, it considers many cost functions, including segments, distance, time, and delivery. Here is a summary of the code:
1. The programme begins by importing the required libraries and modules, such as collections, math, and sys. Additionally, it imports the debugging method set_trace from the pdb module.

2. The code defines a number of functions, including:

  - read_city_gps(): Extracts city GPS data from the 'city-gps.txt' file and returns a dictionary with city names translated to latitude and longitude coordinates.

  - read_road_segments(): Road segment information is taken from a file called "road-segments.txt" via the read_road_segments() function, which then produces a dictionary where city pairs are mapped to road segment details like length, speed limit, and highway name.

  - cost_delivery_time(): Calculates the delivery time using the length of the road section, the posted speed limit, and a delivery time cost function that takes these factors into account.

  - cal_travel_time(): Calculates trip time depending on the length of the road section and the posted speed limit.

  - cal_total(): generates a list of the route's segments after calculating the route's total metrics (miles, time, and delivery time).

  - cal_distance(): uses the Haversine formula to determine the separation between two sets of latitude and longitude coordinates.

  - get_route(): determines the best path between two cities depending on the supplied cost function. The shortest path is found using Dijkstra's method.

3. Based on the selected cost function and a function value supplied (for the "delivery" cost function), the heuristic() function determines a heuristic value for a city pair.

4. Starting with command-line parameters, the main portion of the programme extracts the start city, finish city, and cost function.

5. The start city, finish city, and cost function are sent when the get_route() method is invoked. It determines the best route and provides details about it, such as the number of segments, total miles travelled, total hours spent travelling, and total delivery hours.

6. The route and the derived metrics, such as the number of segments, total miles, total hours, and total delivery hours, are then printed out by the programme.

On the basis of input from the user and the chosen cost function, the programme is built to compute and show various route metrics. In order to find the most efficient path between two cities and calculate the corresponding metrics, it needs information from two input files ('city-gps.txt' and 'road-segments.txt').
