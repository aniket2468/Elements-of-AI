#!/usr/local/bin/python3
#
# mystical_castle.py : a maze solver
#
# Submitted by : Aniket Sharma | anikshar
#
# Based on skeleton code provided in CSCI B551, Fall 2023.

import sys
import json

# Parse the map from a given filename
def parse_map(filename):
        with open(filename, "r") as f:
                return [[char for char in line] for line in f.read().rstrip("\n").split("\n")][3:]
                
# Check if a row,col index pair is on the map
def valid_index(pos, n, m):
        return 0 <= pos[0] < n  and 0 <= pos[1] < m

# Find the possible moves from position (row, col)
def moves(map, row, col):
        moves=((row+1,col, 'D'), (row-1,col, 'U'), (row,col-1, 'L'), (row,col+1, 'R'))

        # Return only moves that are within the castle_map and legal (i.e. go through open space ".")
        return [ move for move in moves if valid_index(move, len(map), len(map[0])) and (map[move[0]][move[1]] in ".@" ) ]

# Perform search on the map
#
# This function MUST take a single parameter as input -- the castle map --
# and return a tuple of the form (move_count, move_string), where:
# - move_count is the number of moves required to navigate from start to finish, or -1
#    if no such route exists
# - move_string is a string indicating the path, consisting of U, L, R, and D characters
#    (for up, left, right, and down)

def search(castle_map):
        # Find current start position
        current_loc=[(row_i,col_i) for col_i in range(len(castle_map[0])) for row_i in range(len(castle_map)) if castle_map[row_i][col_i]=="p"][0]
        fringe=[(current_loc,0)]
        visited_points=[]
        available_points=[]
        available_points.append(current_loc)

        while fringe:
                (curr_move, curr_dist)=fringe.pop(0)
                visited_points.append(curr_move)
                location=available_points.pop(0)

                for move in moves(castle_map, *curr_move):
                        
                        if castle_map[move[0]][move[1]]=="@":
                                available_points.append((location, move))
                                direction=''.join(filter(str.isalpha, str(available_points.pop())))
                                return [curr_dist + 1, direction]
                        else:
                                if (move[0:2] not in visited_points):
                                        fringe.append((move[0:2], curr_dist + 1))
                                        available_points.append((location, move[2]))

# Main Function
if __name__ == "__main__":
        castle_map=parse_map(sys.argv[1])
        print("Shhhh... quiet while I navigate!")
        solution = search(castle_map)
        print("Here's the solution I found:")
        if solution is None:
                print('Inf')
        else:
                print(str(solution[0]) + " " + solution[1])