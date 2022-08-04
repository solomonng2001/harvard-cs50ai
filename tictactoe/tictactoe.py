"""
Tic Tac Toe Player
"""

import math
from copy import deepcopy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    # Count number of X and O already on the board respectively
    # Initialise number of X and O to 0
    num_X = 0
    num_O = 0

    # Loop through each square on the board
    for i in range(3):
        for j in range(3):
            square = board[i][j]
            if square == "X":
                num_X += 1
            elif square == "O":
                num_O += 1

    # Ignoring terminal board scenario
    # If number of X and O is equal (including empty board), it is X turn
    if num_X == num_O:
        return X

    # Else if number of X is more than that of O, it is O turn
    elif num_X > num_O:
        return O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    # Find all EMPTY squares on board, which are possible actions available
    # Initialise set to store actions as tuple(i, j) of board coordinates
    actions = set()

    # Loop through squares on board
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                actions.add((i, j))

    return actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    # If action is not valid for board, raise exception
    if action not in actions(board):
        raise Exception("Invalid Action!")

    # Deep copy of board to preserve original board for further computation
    board_copy = deepcopy(board)

    # Ammend copy of board according to valid action
    board_copy[action[0]][action[1]] = player(board)

    return board_copy


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # Check horizontally for three-in-a-row and return winner if true
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] and not board[i][0] == EMPTY:
            return board[i][0]

    # Else if, check vertically for three-in-a-row and return winner if true
    for j in range(3):
        if board[0][j] == board[1][j] == board[2][j] and not board[0][j] == EMPTY:
            return board[0][j]

    # Else if, check diagonally for three-in-a-row and return winner if true
    if board[0][0] == board[1][1] == board[2][2] and not board[0][0] == EMPTY:
        return board[0][0]
    elif board[0][2] == board[1][1] == board[2][0] and not board[0][2] == EMPTY:
        return board[0][2]

    # Else, game is still in progress or ended in a tie
    else:
        return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    # If game in progress and no winner yet
    if winner(board) == None:

        # Loop through cells to check if game is draw (all cells filled without anyont winning)
        for i in range(3):
            for j in range(3):
                if board[i][j] == EMPTY:

                    # If any cell is EMPTY, then game still in progress
                    return False

        # If no EMPTY cells and no winner, then game is a draw
        return True

    # Else, return true since someone has won
    else:
        return True


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    # If X has won, utility returns 1
    if winner(board) == X:
        return 1

    # Else if O has won, utility return -1
    if winner(board) == O:
        return -1

    # Else, game ends in tie, assuming utility called only if terminal board (ignoring game in progress scenario)
    else:
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    # Return non moves (None) if board is terminal (game ended)
    if terminal(board):
        return None

    # ~ Thoughts: Cannot just max_value() or min_value() the board since they only return max or min scores without the action or best move
    # Best to isolate first level of moves and append action to list in minimax(), instead of appending action to list in min_value() or max_value(),
    # which would be inefficient as it is returning best move at every level too, when we are only interested in the best score in min_value() or max_value() ~

    # Initialise list of possible actions to take and corresponding scores
    actions_values = list()

    # If X's turn to play, return best scoring move for X
    if player(board) == X:

        # Initialise max_v to -ve infinity to guarantee that maximum value is selected
        max_v = -math.inf

        # Loop through possible actions and append to actions_values list the action and value in a tuple (action, value)
        for action in actions(board):

            # v is value of current action and result, max_v is maximum value from all actions (for alpha-beta pruning)
            v = min_value(result(board, action), max_v)
            max_v = max(max_v, v)
            actions_values.append((action, v))

        # Sort actions_values list according to values starting with highest, in decending order
        actions_values.sort(key=lambda x: x[1], reverse=True)

    # If O's turn to play, return best scoring move for O
    if player(board) == O:

        # Initialise min_v to +ve infinity to guarantee that minimum value is selected
        min_v = math.inf

        # Loop through possible actions and append to actions_values list the action and value in a tuple(action, value)
        for action in actions(board):

            # v is value of current action and result, min_v is minimum value from all actions (for alpha-beta pruning)
            v = max_value(result(board, action), min_v)
            min_v = min(min_v, v)
            actions_values.append((action, v))

        # Sort actions_values list according to values starting with lowest, in accending order
        actions_values.sort(key=lambda x: x[1])

    # Else if not terminal, return best move for current player
    return actions_values[0][0]


def max_value(board, beta):
    """
    Returns maximum possible score on board after considering opponent's plays in following move
    """
    # Initialise comparision to -ve infinity so that largest value guaranteed to be selected
    max_v = -math.inf

    # Base case of recursion: if game ended, return the value of the board
    if terminal(board):
        return utility(board)

    # Else loop through all possible moves of min player and return largest value
    for action in actions(board):
        max_v = max(max_v, min_value(result(board, action), max_v))

        # If maximum v is not smaller than pre-established minimum score of beta for the previous min player,
        # calculating more v and actions is irrelevant, therefore stop calculations
        if max_v >= beta:
            break

    return max_v


def min_value(board, alpha):
    """
    Returns minimum possible score on board after considering oppoenent's plays in following move
    """
    # Initialise comparison to +ve infinity so that lowest value guaranteed to be selected
    min_v = math.inf

    # Base case of recursion: if game ended, return the value of the board
    if terminal(board):
        return utility(board)

    # Else loop through all possible moves of max player and return smallest value
    for action in actions(board):
        min_v = min(min_v, max_value(result(board, action), min_v))

        # If minimum v is not greater than pre-established maximum score of alpha for the previous max player,
        # calculating more v and actions is irrelevant, therefore stop calculations
        if min_v <= alpha:
            break

    return min_v