#
# raichu.py : Play the game of Raichu
#
# Aniket Sharma: anikshar
# Ashwin Venkatakrishnan: ashvenk
# Fares Alharbi: fafaalha

from copy import deepcopy
from time import time
import sys

def convert_1D_to_2D(board_1D, N):
    board_2D = list()
    for i in range(N):
        board_2D.append(board_1D[i*N:(i+1)*N])
    return board_2D

def convert_2D_to_1D(board_2D):
    board_1D = list()
    if board_2D != None:
        for row in board_2D:
            if row != None:
                for col in row:
                    board_1D.append(col)
    return board_1D

def board_to_string(board, N):
    return "\n".join(board[i:i+N] for i in range(0, len(board), N))

def is_within_bounds(r, c, N):
    return True if (0 <= r < N) and (0 <= c < N) else False

def pichu_successor(board, N, row, col):
    all_intermediate_states = list()
    if board[row][col] == 'w':
        is_w_piece = True
        pichu_moves_set = [(1, -1, 2, -2), (1, 1, 2, 2)]
    elif board[row][col] == 'b':
        is_w_piece = False
        pichu_moves_set = [(-1, 1, -2, 2), (-1, -1, -2, -2)]

    for move in pichu_moves_set:
        intermediate_state = None
        if is_within_bounds(row+move[0], col+move[1], N):
            if board[row+move[0]][col+move[1]] == '.':
                intermediate_state = deepcopy(board)
                intermediate_state[row+move[0]][col+move[1]] = ('w' if is_w_piece else 'b')
                if row+move[0] == N-1:
                    intermediate_state[row+move[0]][col+move[1]] = ('@' if is_w_piece else '$')
                intermediate_state[row][col] = '.'
            elif board[row+move[0]][col+move[1]] == ('b' if is_w_piece else 'w') and is_within_bounds(row+move[2], col+move[3], N) and board[row+move[2]][col+move[3]] == '.':
                intermediate_state = deepcopy(board)
                intermediate_state[row+move[2]][col+move[3]] = ('w' if is_w_piece else 'b')
                if row+move[2] == N-1:
                    intermediate_state[row+move[2]][col+move[3]] = ('@' if is_w_piece else '$')
                intermediate_state[row][col], intermediate_state[row+move[0]][col+move[1]] = '.', '.'
        if intermediate_state:
            all_intermediate_states += [intermediate_state]

    return all_intermediate_states

def pikachu_successor(board, N, row, col):
    all_intermediate_states = list()
    if board[row][col] == 'W':
        is_w_piece = True
        pikachu_moves_set = [(0, 0, 0, -1, -2, -3), (0, 0, 0, 1, 2, 3), (1, 2, 3, 0, 0, 0)]
    elif board[row][col] == 'B':
        is_w_piece = False
        pikachu_moves_set = [(0, 0, 0, -1, -2, -3), (0, 0, 0, 1, 2, 3), (-1, -2, -3, 0, 0, 0)]
    intermediate_state = None

    for i, move in enumerate(pikachu_moves_set):
        if is_within_bounds(row+move[0], col+move[3], N):
            if board[row+move[0]][col+move[3]] == '.':
                intermediate_state = deepcopy(board)
                intermediate_state[row+move[0]][col+move[3]] = ('W' if is_w_piece else 'B')
                if i == 2 and row+move[0] == (N-1 if is_w_piece else 0 ):
                    intermediate_state[row+move[0]][col] = ('@' if is_w_piece else '$')
                intermediate_state[row][col] = '.'
                all_intermediate_states += [intermediate_state]

            elif board[row+move[0]][col+move[3]] in ('Bb' if is_w_piece else 'Ww') and is_within_bounds(row+move[1], col+move[4], N) and board[row+move[1]][col+move[4]] == '.':
                intermediate_state = deepcopy(board)
                intermediate_state[row+move[1]][col+move[4]] = ('W' if is_w_piece else 'B')
                if i == 2 and row+move[1] == (N-1 if is_w_piece else 0 ):
                    intermediate_state[row+move[1]][col] = ('@' if is_w_piece else '$')
                intermediate_state[row][col], intermediate_state[row+move[0]][col+move[3]] = '.', '.'
                all_intermediate_states += [intermediate_state]

            if is_within_bounds(row+move[1], col+move[4], N):
                if board[row+move[1]][col+move[4]] == '.' and board[row+move[0]][col+move[3]] == '.':
                    intermediate_state = deepcopy(board)
                    intermediate_state[row+move[1]][col+move[4]] = 'W' if is_w_piece else 'B'
                    if i == 2 and row+move[1] == (N-1 if is_w_piece else 0 ):
                        intermediate_state[row+move[1]][col] = ('@' if is_w_piece else '$')
                    intermediate_state[row][col] = '.'
                    all_intermediate_states += [intermediate_state]

                elif board[row+move[1]][col+move[4]] in ('Bb' if is_w_piece else 'Ww') and board[row+move[0]][col+move[3]] == '.' and is_within_bounds(row+move[2], col+move[5], N) and board[row+move[2]][col+move[5]] == '.':
                    intermediate_state = deepcopy(board)
                    intermediate_state[row+move[2]][col+move[5]] = ('W' if is_w_piece else 'B')
                    if i == 2 and row+move[2] == (N-1 if is_w_piece else 0 ):
                        intermediate_state[row+move[2]][col] = ('@' if is_w_piece else '$')
                    intermediate_state[row][col], intermediate_state[row+move[1]][col+move[4]] = '.', '.'
                    all_intermediate_states += [intermediate_state] 
    return all_intermediate_states

def raichu_successor(board, N, row, col):
    if board[row][col] == '@':
        is_w_piece = True
        intermediate_state = None
    elif board[row][col] == '$':
        is_w_piece = True
        intermediate_state = None
    all_intermediate_states = list()

    # Piece moves left
    for next_index in range(1, N):
        if is_within_bounds(row, col-next_index, N):
            if board[row][col-next_index] == '.':
                intermediate_state = deepcopy(board)
                intermediate_state[row][col-next_index], intermediate_state[row][col] = ('@' if is_w_piece else '$'), '.'
                all_intermediate_states += [intermediate_state]
            elif board[row][col-next_index] in ('Ww@' if is_w_piece else 'Bb$'):
                for postInd in range(col-next_index-1, -1, -1):
                    if is_within_bounds(row, postInd, N) and board[row][postInd] == '.':
                        intermediate_state = deepcopy(board) 
                        intermediate_state[row][postInd], intermediate_state[row][col], intermediate_state[row][col-next_index] = ('@' if is_w_piece else '$'), '.', '.'
                        all_intermediate_states += [intermediate_state]
                    elif is_within_bounds(row, postInd, N) and board[row][postInd] != '.':
                        break
                break

    # Piece moves up
    for next_index in range(1, N):
        if is_within_bounds(row-next_index, col, N):
            if board[row-next_index][col] == '.':
                intermediate_state = deepcopy(board)
                intermediate_state[row-next_index][col], intermediate_state[row][col] = ('@' if is_w_piece else '$'), '.'
                all_intermediate_states += [intermediate_state]
            elif board[row-next_index][col] in ('Ww@' if is_w_piece else 'Bb$'):
                for postInd in range(row-next_index-1, -1, -1):
                    if is_within_bounds(postInd, col, N) and board[postInd][col] == '.':
                        intermediate_state = deepcopy(board)
                        intermediate_state[postInd][col], intermediate_state[row][col], intermediate_state[row-next_index][col] = ('@' if is_w_piece else '$'), '.', '.'
                        all_intermediate_states += [intermediate_state]
                    elif is_within_bounds(postInd, col, N) and board[postInd][col] != '.':
                        break
                break

    # Piece moves right
    for next_index in range(1, N):
        if is_within_bounds(row, col+next_index, N):
            if board[row][col+next_index] == '.':
                intermediate_state = deepcopy(board)
                intermediate_state[row][col+next_index], intermediate_state[row][col] = ('@' if is_w_piece else '$'), '.'
                all_intermediate_states += [intermediate_state]
            elif board[row][col+next_index] in ('Ww@' if is_w_piece else 'Bb$'):
                for postInd in range(col+next_index+1, N):
                    if is_within_bounds(row, postInd, N) and board[row][postInd] == '.':
                        intermediate_state = deepcopy(board) 
                        intermediate_state[row][postInd] = ('@' if is_w_piece else '$')
                        intermediate_state[row][col], intermediate_state[row][col+next_index] = '.', '.'
                        all_intermediate_states += [intermediate_state]
                    elif is_within_bounds(row, postInd, N) and board[row][postInd] != '.':
                        break
                break

    # Piece moves down
    for next_index in range(1, N):
        if is_within_bounds(row+next_index, col, N):
            if board[row+next_index][col] == '.':
                intermediate_state = deepcopy(board)
                intermediate_state[row+next_index][col], intermediate_state[row][col] = ('@' if is_w_piece else '$'), '.'
                all_intermediate_states += [intermediate_state]
            elif board[row+next_index][col] in ('Ww@' if is_w_piece else 'Bb$'):
                for postInd in range(row+next_index+1, N):
                    if is_within_bounds(postInd, col, N) and board[postInd][col] == '.':
                        intermediate_state = deepcopy(board)
                        intermediate_state[postInd][col] = ('@' if is_w_piece else '$')
                        intermediate_state[row][col], intermediate_state[row+next_index][col] = '.', '.'
                        all_intermediate_states += [intermediate_state]
                    elif is_within_bounds(postInd, col, N) and board[postInd][col] != '.':
                        break
                break

    # Piece moves down and left
    for next_index in range(1, N):
        if is_within_bounds(row+next_index, col-next_index, N):
            if board[row+next_index][col-next_index] == '.':
                intermediate_state = deepcopy(board)
                intermediate_state[row+next_index][col-next_index], intermediate_state[row][col] = ('@' if is_w_piece else '$'), '.'
                all_intermediate_states += [intermediate_state]
            elif board[row+next_index][col-next_index] in ('Ww@' if is_w_piece else 'Bb$'):
                for postInd in range(1, N):
                    if is_within_bounds(row+(next_index+postInd), col-(next_index+postInd), N) and board[row+(next_index+postInd)][col-(next_index+postInd)] == '.':
                        intermediate_state = deepcopy(board)
                        intermediate_state[row+(next_index+postInd)][col-(next_index+postInd)], intermediate_state[row][col], intermediate_state[row+next_index][col-next_index] = ('@' if is_w_piece else '$'), '.', '.'
                        all_intermediate_states += [intermediate_state]
                    elif is_within_bounds(row+(next_index+postInd), col-(next_index+postInd), N) and board[row+(next_index+postInd)][col-(next_index+postInd)] != '.':
                        break
                break

    # Piece moves down and right
    for next_index in range(1, N):
        if is_within_bounds(row+next_index, col+next_index, N):
            if board[row+next_index][col+next_index] == '.':
                intermediate_state = deepcopy(board)
                intermediate_state[row+next_index][col+next_index], intermediate_state[row][col] = ('@' if is_w_piece else '$'), '.'
                all_intermediate_states += [intermediate_state]
            elif board[row+next_index][col+next_index] in ('Ww@' if is_w_piece else 'Bb$'):
                for postInd in range(1, N):
                    if is_within_bounds(row+(next_index+postInd), col+(next_index+postInd), N) and board[row+(next_index+postInd)][col+(next_index+postInd)] == '.':
                        intermediate_state = deepcopy(board)
                        intermediate_state[row+(next_index+postInd)][col+(next_index+postInd)], intermediate_state[row][col], intermediate_state[row+next_index][col+next_index] = ('@' if is_w_piece else '$'), '.', '.'
                        all_intermediate_states += [intermediate_state]
                    elif is_within_bounds(row+(next_index+postInd), col+(next_index+postInd), N) and board[row+(next_index+postInd)][col+(next_index+postInd)] != '.' :
                        break
                break

    # Piece moves up and left
    for next_index in range(1, N):
        if is_within_bounds(row-next_index, col-next_index, N):
            if board[row-next_index][col-next_index] == '.':
                intermediate_state = deepcopy(board)
                intermediate_state[row-next_index][col-next_index], intermediate_state[row][col] = ('@' if is_w_piece else '$'), '.'
                all_intermediate_states += [intermediate_state]
            elif board[row-next_index][col-next_index] in ('Ww@' if is_w_piece else 'Bb$'):
                for postInd in range(1, N):
                    if is_within_bounds(row-(next_index+postInd), col-(next_index+postInd), N) and board[row-(next_index+postInd)][col-(next_index+postInd)] == '.':
                        intermediate_state = deepcopy(board)
                        intermediate_state[row-(next_index+postInd)][col-(next_index+postInd)], intermediate_state[row][col], intermediate_state[row-next_index][col-next_index] = ('@' if is_w_piece else '$'), '.', '.'
                        all_intermediate_states += [intermediate_state]
                    elif is_within_bounds(row-(next_index+postInd), col-(next_index+postInd), N) and board[row-(next_index+postInd)][col-(next_index+postInd)] != '.':
                        break
                break

    # Piece moves up and right
    for next_index in range(1, N):
        if is_within_bounds(row-next_index, col+next_index, N):
            if board[row-next_index][col+next_index] == '.':
                intermediate_state = deepcopy(board)
                intermediate_state[row-next_index][col+next_index], intermediate_state[row][col] = ('@' if is_w_piece else '$'), '.'
                all_intermediate_states += [intermediate_state]
            elif board[row-next_index][col+next_index] in ('Ww@' if is_w_piece else 'Bb$'):
                for postInd in range(1, N):
                    if is_within_bounds(row-(next_index+postInd), col+(next_index+postInd), N) and board[row-(next_index+postInd)][col+(next_index+postInd)] == '.':
                        intermediate_state = deepcopy(board)
                        intermediate_state[row-(next_index+postInd)][col+(next_index+postInd)], intermediate_state[row][col], intermediate_state[row-next_index][col+next_index] = ('@' if is_w_piece else '$'), '.', '.'
                        all_intermediate_states += [intermediate_state]
                    elif is_within_bounds(row-(next_index+postInd), col+(next_index+postInd), N) and board[row-(next_index+postInd)][col+(next_index+postInd)] != '.':
                        break
                break

    return all_intermediate_states

def choose_piece(board, N, row, col):
    piece = board[row][col]
    if piece in 'wb':
        return pichu_successor(board, N, row, col)
    elif piece in 'WB':
        return pikachu_successor(board, N, row, col)
    elif piece in '@$':
        return raichu_successor(board, N, row, col)

def evaluate_board(board, N, row, col):
    if board[row][col] in 'Ww@':
        is_w_piece = True
    elif board[row][col] in 'Bb$':
        is_w_piece = False

    directions = [(-1, -1), (+1, +1), (-1, +1), (+1, -1), (-1, +0), (+1, +0), (+0, +1), (+0, -1)]
    points = 0
    for direction in directions:
        if not is_within_bounds(row+direction[0], col+direction[1], N):
            points += 1

    if board[row][col] not in 'Ww@Bb$.':
        raise "Invalid board piece"

    for direction in directions:
        if is_within_bounds(row+direction[0], col+direction[1], N):
            if board[row+direction[0]][col+direction[1]] in ('Ww@'  if is_w_piece else 'Bb$'):
                points += 3
            elif board[row+direction[0]][col+direction[1]] in ('Bb$'  if is_w_piece else 'Ww@'):
                points -= 1
            else:
                points += 0

    return points if is_w_piece else -points

def get_minmax(board, N):
    points = 0
    piece_weights = {'.': 1, 'w': 10, 'W': 11, '@': 15, 'b': -10, 'B': -11, '$': -15}

    for r in range(N):
        for c in range(N):
            if board[r][c] != '.':
                points += evaluate_board(board, N, r, c)

    for piece in convert_2D_to_1D(board):
        points += piece_weights[piece]

    bstr = dict()
    for pieces in "".join(convert_2D_to_1D(board)):
        if pieces != '.' and pieces not in bstr:
            bstr[pieces] = 1
        elif pieces != '.':
            bstr[pieces] += 1

    return points

def is_goal_state(board):
    bstr = dict()
    for pieces in "".join(convert_2D_to_1D(board)):
        if pieces != '.' and pieces not in bstr:
            bstr[pieces] = 1
        elif pieces != '.':
            bstr[pieces] += 1
    if len(bstr.keys()) == 3 and 'b' in bstr.keys():
        return True
    elif len(bstr.keys()) == 3 and 'w' in bstr.keys():
        return False

def minimax(board, N, count, alpha, beta, player):
    if count == 0 or is_goal_state(board):
        return get_minmax(board, N), board

    best = None
    maxEval = float('-inf')
    minEval = float('inf')

    if player == 'w':
        for a in range(N):
            for b in range(N):
                if board[a][b] in 'wW@':
                    for next_intermediate_state in choose_piece(board, N, a, b):
                        boardpoints, _ = minimax(next_intermediate_state, N, count-1, alpha, beta, 'b')
                        if boardpoints > maxEval:
                            best, maxEval = next_intermediate_state, boardpoints
                        alpha = max(alpha, boardpoints)
                        if beta <= alpha:
                            break
        return maxEval, best

    if player == 'b':
        for a in range(N):
            for b in range(N):
                if board[a][b] in 'bB$':
                    for next_intermediate_state in choose_piece(board, N, a, b):
                        boardpoints, _ = minimax(next_intermediate_state, N, count-1, alpha, beta, 'w')
                        if boardpoints < minEval:
                            best, minEval = next_intermediate_state, boardpoints
                        beta = min(beta, boardpoints)
                        if beta <= alpha:
                            break
        return minEval, best

def find_best_move(board, N, player, timelimit):
    start, count = time(), 0
    while time() - start < timelimit-1:
        _, result = minimax(convert_1D_to_2D(list(board), N), N, count, float('-inf'), float('inf'), player)
        count += 1
        yield "".join(convert_2D_to_1D(result))

if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise Exception("Usage: Raichu.py N player board timelimit")
        
    (_, N, player, board, timelimit) = sys.argv
    N=int(N)
    timelimit=int(timelimit)
    if player not in "wb":
        raise Exception("Invalid player.")

    if len(board) != N*N or 0 in [c in "wb.WB@$" for c in board]:
        raise Exception("Bad board string.")

    print("Searching for best move for " + player + " from board state: \n" + board_to_string(board, N))
    print("Here's what I decided:")
    for new_board in find_best_move(board, N, player, timelimit):
        print(new_board)
