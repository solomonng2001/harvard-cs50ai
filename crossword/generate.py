import sys

from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        w, h = draw.textsize(letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        # For each variable / key in self.domains dictionary
        for var in self.domains:

            # Initialise set of words to remove
            remove = set()

            # Loop through each word in variable's word set / value
            for word in self.domains[var]:

                # If length of word does not fit length of variable
                if len(word) != var.length:

                    # Remove word from variable's domain / set of words
                    remove.add(word)

            # Remove words from variable's domain / set of words
            self.domains[var] -= remove


    def revise(self, x, y):
        """
        USE PSEUDOCODE FROM THE LECTURE NODES
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.
        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        # Initialise set of words to remove from domain of X
        remove = set()

        # Check if overlap is exists
        overlap = self.crossword.overlaps[x, y]
        if overlap:

            # For every word in domain of X, loop through domain of Y
            for word_x in self.domains[x]:

                possible_neighbor = False
                for word_y in self.domains[y]:

                    # If overlap of same letter exists (and not duplicate word), break loop since at least 1 possible corresponding value of y available
                    if word_x[overlap[0]] == word_y[overlap[1]] and word_x != word_y:
                        possible_neighbor = True
                        break
                    
                # Else if no overlap of same letter exists, remove word from domain of x
                if not possible_neighbor:
                    remove.add(word_x)

            # Make revisions and return true if values in domain x to be removed
            if remove:
                self.domains[x] -= remove
                return True

        # Else false if no revisions, return false
        return False


    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        # Initial list of arc not given, start with initial queue list of all arc tuple (x, y) in problem
        if arcs == None:
            arcs = list()

            # Loop through variables and pair with neighbouring / overlapping variables to form arc tuple (x, y)
            for x in self.crossword.variables:
                for y in self.crossword.neighbors(x):
                    arcs.append((x, y))
        
        # While in arcs queue not empty, check that for each arc tuple (x, y), overlapping letter does not conflict
        while arcs:
            x, y = arcs.pop()

            # If changes made to variable x's domain to make it arc consistent
            if self.revise(x, y):

                # If X's domain is empty, no solution
                if len(self.domains[x]) == 0:
                    return False

                # Recheck each neighbor's domain and add arc tuple (neighbor, x) to arcs queue,
                # Except y, against which x was made consistent ealier
                for neighbor in self.crossword.neighbors(x) - {y}:
                    arcs.append((neighbor, x))
            
        return True


    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        # Assignment is complete when all variable has assigned value
        # 1. Number of variables assigned equals number of values
        # 2. Number of variables assigned (consistent) equals number of variables in crossword
        if len(assignment.keys()) == len(assignment.values()) == len(self.crossword.variables):
            return True

        # Else if assignment incomplete
        return False

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        # For each variable's values
        for var, val in assignment.items():

            # Check against other variables' values for non-uniqueness
            for other_var, other_val in assignment.items():
                if val == other_val and var != other_var:
                    return False
        
            # Check if assigned word does not fit length of variable
            if len(val) != var.length:
                return False
            
            # Check against neighboring variables, if overlapping letter of neighboring variables conflict
            # Since assignment not necessarily complete, check against neighbouring variables in assignment
            for neighbor in self.crossword.neighbors(var).intersection(set(assignment.keys())):
                
                # Extract and compare overlapping letters
                var_letter_int, neighbor_letter_int = self.crossword.overlaps[var, neighbor]
                if assignment[var][var_letter_int] != assignment[neighbor][neighbor_letter_int]:
                    return False

        # Else if passed all consistency tests
        return True


    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        # Initialise dictionary where key is value/word and value is number fo possibel choices eliminated
        choices_eliminated = dict()

        # for each value in variable, count number of choices eliminated in unassigned neighbors
        for val in self.domains[var]:
            count_choices_eliminated = 0

            # for each unassigned neighbor
            for neighbor in self.crossword.neighbors(var):
                if neighbor not in assignment:

                    # for each value in domain
                    for val_neighbor in self.domains[neighbor]:

                        # Add number of eliminated choices to count, based on whether overlapping letter conflicts
                        i, j = self.crossword.overlaps[var, neighbor]
                        if val[i] != val_neighbor[j]:
                            count_choices_eliminated += 1
            
            # Add value and count of choices eliminated to dictionary
            choices_eliminated[val] = count_choices_eliminated
        
        # Sort values by choices eliminated (accending)
        sorted_values = sorted(choices_eliminated.keys(), key=lambda x: choices_eliminated[x])

        return sorted_values
                    

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        # Sorting key that returns tuple (minimum remaining values, degree)
        def sorting_key(item):

            # First, sort by minimum remaining values (accending)
            mrw = len(item[1])

            # Second, sort by degree (decending)
            degree = -len(self.crossword.neighbors(item[0]))
        
            return (mrw, degree)
        
        # Sort unassigned variables
        unassigned_var = list()
        for var, val in self.domains.items():
            if var not in assignment.keys():
                unassigned_var.append((var, val))
        sorted_unassigned_var = sorted(unassigned_var, key=sorting_key)
        
        return sorted_unassigned_var[0][0]


    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        # If assignment is complete, return assigned words to variables as dictionary
        if self.assignment_complete(assignment):
            return assignment
        
        # Else if assignment incomplete
        # Prioritise variable to assign a value to, via minimum remaining values heuristic
        var = self.select_unassigned_variable(assignment)

        # Consider each value in variable's domain, ranked by least constraining values heuristic
        for val in self.order_domain_values(var, assignment):

            # Add value to assignment and check if assignment consistent
            assignment[var] = val
            if self.consistent(assignment):

                # Recursively assign values to variables via backtracking
                solution = self.backtrack(assignment)

                # If solved, return result
                if solution:
                    return solution
                
            # Else if no solution or not consistent, remove variable and value from assignment (to be reassigned)
            assignment.pop(var)
        
        # Else if no values in domain provide solution
        return None


def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
