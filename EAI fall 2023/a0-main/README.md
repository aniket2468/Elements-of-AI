# anikshar-a0
Report. For eachassignment, we’ll require a short written report that summarizes yourprogramming solutions and that answers specific questions thatwe pose. Please put the report inthe Readme.md file in your Github repository. For each programming problem, your report should include a brief overview ofhow your solution works, including any problems you faced, any assumptions, simplifications, or design decisions you made, any parts of your solution that you feel are particularly innovative, etc. These comments are your opportunity for us to better understand your code and the amount of work that you did in writing it; they are especially important if your code does not work as well as you would like, since it is a chance to document how much energy and thought you put into yoursolution. For example, if you tried several differentapproaches before finding one thatworked, feel free to describe this process so that we can appreciate the work that you did that might not otherwise be reflected in your final solution.


# mystical_castle.py

Your goal is to write a program to find the shortest path between your location and the opening of the magic portal. You can move one square at a time in any of the four principal compass directions, and the program should output a number indicating the shortest distance between the two points, followed by a string of letters (L, R, D, and U for left, right, down, and up) indicating that solution. Your program should take a single command line argument, which is the name of the file containing the map file. 

The Breadth First Search, which employs queue data structure, is used to find the solution. The programme initially used Depth First Search, which utilised a stack. This implementation's flaw is that it manages to enter a loop after adding additional nodes. Because it will keep repeating by popping the last element and then adding a new element that references the previously popped element, it will take an indefinite amount of time. If there is a solution, BFS always finds it. It returns the shortest path out of all possible paths since it keeps looking for the answer starting with the shortest.

Because it was the first assignment and I was unfamiliar with Python, it took me a few days to learn the actual code. Although the search strategies we were taught in class appeared simple in principle, putting them into practise required some planning.

By adding a new data structure that holds the previously explored points, the code has been made simpler. This made it unnecessary to repeatedly travel to the same spot, which was helpful. By determining if the point was previously included in the traversed list and then not including it in the fringe, one point was only investigated once in this manner.

The map's character-containing coordinates make up the set of valid states "." or "@"

When we send the points of our points, the successor function is the condition that is utilised to select the following set of points that are returned. These are essentially the points in the East, West, North, and South directions of the point. There might not be all four values returned if the point is on the edge or in a corner.

The greatest length to which our function can go is represented by the cost function. If there is no solution for the given problem, then the cost function is width times depth, which is equivalent to rows times columns in an unrestricted manner.

Mystical Castle, which is indicated by @, is the target state definition.

The starting point is denoted by # and is our location.

# place_turrets.py

Depth First Search had to be used in order to solve this issue. Although Breadth First Search was an option, DFS produces results more quickly. We must determine the maximum number of friends that may be inserted, which is indicated by the letter "p" in this case. To observe the greatest that can be accomplished, it is more practical to travel lengthwise. If our solution is discovered before then, the programme breaks and returns the solution.

Expanding the state space after processing and storing multi-dimensional arrays or lists in the periphery was necessary to solve this problem. The subsequent placement of p in the turrets constitutes the state space in this case.

A datastructure that houses previously produced versions of the map has also been put here. As a result, a returning configuration of the map that has already been constructed is not pushed to the edge.

# Remark
I am able to run place_turrets.py alone and getting correct output for k = 5, 6 and 7, But I am not able to run file through test_a0.py file. I am getting error in accessing the K value and when I am trying to fix the that single error, I'm getting other 2 error. I am 80% done with part 2, I'm getting the output when run place_turrets.py file alone.
