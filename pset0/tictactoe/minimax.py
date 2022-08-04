# minimax(), min_value(), max_value() without Alpha-Beta Pruning
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
    # Loop through possible actions and append to actions_values list the action and value in a tuple (action, value)
    if player(board) == X:
        for action in actions(board):
            actions_values.append((action, min_value(result(board, action))))

        # Sort actions_values list according to values starting with highest, in decending order
        actions_values.sort(key=lambda x: x[1], reverse=True)

    # If O's turn to play, return best scoring move for O
    # Loop through possible actions and append to actions_values list the action and value in a tuple(action, value)
    if player(board) == O:
        for action in actions(board):
            actions_values.append((action, max_value(result(board, action))))

        # Sort actions_values list according to values starting with lowest, in accending order
        actions_values.sort(key=lambda x: x[1])

    # Else if not terminal, return best move for current player
    return actions_values[0][0]


def max_value(board):
    """
    Returns maximum possible score on board after considering opponent's plays in following move
    """
    # Initialise comparision to -ve infinity so that largest value guaranteed to be selected
    v = -math.inf

    # Base case of recursion: if game ended, return the value of the board
    if terminal(board):
        return utility(board)

    # Else loop through all possible moves of min player and return largest value
    for action in actions(board):
        v = max(v, min_value(result(board, action)))

    return v


def min_value(board):
    """
    Returns minimum possible score on board after considering oppoenent's plays in following move
    """
    # Initialise comparison to +ve infinity so that lowest value guaranteed to be selected
    v = math.inf

    # Base case of recursion: if game ended, return the value of the board
    if terminal(board):
        return utility(board)

    # Else loop through all possible moves of max player and return smallest value
    for action in actions(board):
        v = min(v, max_value(result(board, action)))

    return v