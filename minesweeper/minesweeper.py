import itertools
import random
import copy


class Minesweeper():
    """
    Minesweeper game representation
    """

    def __init__(self, height=8, width=8, mines=8):

        # Set initial width, height, and number of mines
        self.height = height
        self.width = width
        self.mines = set()

        # Initialize an empty field with no mines
        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        # Add mines randomly
        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        # At first, player has found no mines
        self.mines_found = set()

    def print(self):
        """
        Prints a text-based representation
        of where mines are located.
        """
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                else:
                    print("| ", end="")
            print("|")
        print("--" * self.width + "-")

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        """
        Returns the number of mines that are
        within one row and column of a given cell,
        not including the cell itself.
        """

        # Keep count of nearby mines
        count = 0

        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count

    def won(self):
        """
        Checks if all mines have been flagged.
        """
        return self.mines_found == self.mines


class Sentence():
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.
    """

    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    def known_mines(self):
        """
        Returns the set of all cells in self.cells known to be mines.
        """
        # If number of elements in set equals count, all cells in set are mines
        if len(self.cells) == self.count:
            return self.cells
        return None

    def known_safes(self):
        """
        Returns the set of all cells in self.cells known to be safe.
        """
        # If count of mines is 0, all cells are safe
        if self.count == 0:
            return self.cells
        return None

    def mark_mine(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.
        """
        # If mine cell in set of cell, discard mine cell and reduce count by 1
        # Create deepcopy to prevent iterating and editing same loop
        cells_copy = copy.deepcopy(self.cells)

        # Iterate through copied cells
        if cell in cells_copy:

            # Discard from original cells if it is mine
            self.cells.discard(cell)
            self.count -= 1

    def mark_safe(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.
        """
        # If mine cell in set of cell, discard mine cell and reduce count by 1
        # Create deepcopy to prevent iterating and editing same loop
        cells_copy = copy.deepcopy(self.cells)

        # Iterate through copied cells
        if cell in cells_copy:

            # Discard from original cells if safe cell
            self.cells.discard(cell)

class MinesweeperAI():
    """
    Minesweeper game player
    """

    def __init__(self, height=8, width=8):

        # Set initial height and width
        self.height = height
        self.width = width

        # Keep track of which cells have been clicked on
        self.moves_made = set()

        # Keep track of cells known to be safe or mines
        self.mines = set()
        self.safes = set()

        # List of sentences about the game known to be true
        self.knowledge = []

    def mark_mine(self, cell):
        """
        Marks a cell as a mine, and updates all knowledge
        to mark that cell as a mine as well.
        """
        # Record mine in mines list
        self.mines.add(cell)

        # Loop through knowledge base to mark mine
        for sentence in self.knowledge:
            sentence.mark_mine(cell)

    def mark_safe(self, cell):
        """
        Marks a cell as safe, and updates all knowledge
        to mark that cell as safe as well.
        """
        # Record safe in safe cells list
        self.safes.add(cell)

        # Loop through knowledge base to mark safe
        for sentence in self.knowledge:
            sentence.mark_safe(cell)
    
    def neighbours(self, cell):
        """
        Returns neighbouring cells of a cell as a list.
        """
        # Add to list neighbouring cells, by looping around neighbouring cells
        neighbours = set()
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):
                
                # If cell within board game parameters and is not cell itself, add to neighbours list
                if not (i, j) == cell and 0 <= i < self.height and 0 <= j < self.width:
                    neighbours.add((i, j))
        
        return neighbours
    
    def mark_determinable_cells(self):
        """
        Ensures all cells in AI's knowledge base that are determinable
        mines or safe cells are marked.
        """
        # Loop through sentences in AI's knowledge base
        for sentence in self.knowledge:

            # If sentence contains all mine cells only
            if sentence.known_mines():
                
                # Make deep copy of mine cells to prevent working on list while iterating list
                mines = copy.deepcopy(sentence.cells)

                # Loop through working mines set and mark mines
                for mine in mines:
                    self.mark_mine(mine)

            # If sentence contains all safe cells only
            if sentence.known_safes():

                # Make deep copy of mine cells to prevent working on list while iterating list
                safes = copy.deepcopy(sentence.cells)

                # Loop through working safes set and mark safes    
                for safe in safes:
                    self.mark_safe(safe)

    def infer(self):
        """
        Add new sentences inferred from existing AI knowledge to knowledge base
        """
        # Initialise list of inferred new sentences and list of empty cells to be removed
        inferences = []
        empty = []

        # Loop through 2 same sets of AI's knowledge to detect proper subsets
        for sentence1 in self.knowledge:

            # If empty sentence with no cells detected, remove it
            if sentence1.cells == set():
                empty.append(sentence1)

            for sentence2 in self.knowledge:

                # If proper subset of sentences detected, find the difference in cells and count
                if sentence1.cells < sentence2.cells:
                    difference_cells = sentence2.cells - sentence1.cells
                    difference_count = sentence2.count - sentence1.count
                
                    # Form inferred sentence using differences in subset and superset
                    inferred_sentence = Sentence(difference_cells, difference_count)
                    
                    # If inferred sentence has no duplicates in existing knowledge base
                    if self.unduplicated_sentence(inferred_sentence):

                        # Add inferred sentence to inferences list
                        inferences.append(inferred_sentence)
        
        # Remove all empty sets from knowledge base
        for remove in empty:
            self.knowledge.remove(remove)
        
        # Add inferences to AI's knowledge base
        self.knowledge.extend(inferences)

        # Return if there were new inferences (true if inferences is not an empty list)
        return len(inferences) != 0

    def unduplicated_sentence(self, check_sentence):
        """
        Returns whether sentence is a duplicate of existing AI knowledge base
        """
        # Loop through knowledge base to check for duplicates
        for sentence in self.knowledge:

            # If new_sentence is duplicate return true
            if check_sentence == sentence:
                return False

        # Else if new sentence is unduplicated, return false
        return True

    def add_knowledge(self, cell, count):
        """
        Called when the Minesweeper board tells us, for a given
        safe cell, how many neighboring cells have mines in them.
        This function should:
            1) mark the cell as a move that has been made
            2) mark the cell as safe
            3) add a new sentence to the AI's knowledge base
               based on the value of `cell` and `count`
            4) mark any additional cells as safe or as mines
               if it can be concluded based on the AI's knowledge base
            5) add any new sentences to the AI's knowledge base
               if they can be inferred from existing knowledge
        """
        # 1) Add cell into set containing moves made
        self.moves_made.add(cell)

        # 2) Add cell to set containing safe cells and update knowledge
        self.mark_safe(cell)

        # 3) Add new sentence about cell's neighbouring mine count
        # Form new sentence using cell and count
        new_sentence = Sentence(self.neighbours(cell), count)

        # If sentence contains determined mine or safe cells, remove them and update sentence
        # Loop through mines set and if mine cell in sentence cells, discard mine cell from sentence
        for mine in self.mines:
            new_sentence.mark_mine(mine)

        # Loop through safe cells set and if safe cell in sentence cells, discard safe cell from sentence
        for safe in self.safes:
            new_sentence.mark_safe(safe)

        # If new sentence has no duplicates in existing knowledge base,
        if self.unduplicated_sentence(new_sentence):
        
            # Add new sentence to AI's knowledge list
            self.knowledge.append(new_sentence)

        # Loop until nothing more to infer (no more changes to knowledge base)
        inferences = True
        while inferences:

            # 4) Mark safe or mine cells based on AI's knowledge base
            self.mark_determinable_cells()

            # 5) Add new sentences inferred from existing AI knowledge
            inferences = self.infer()

    def make_safe_move(self):
        """
        Returns a safe cell to choose on the Minesweeper board.
        The move must be known to be safe, and not already a move
        that has been made.

        This function may use the knowledge in self.mines, self.safes
        and self.moves_made, but should not modify any of those values.
        """
        # Obtain arbitrary move from set of cells that are safe moves but not moves already made
        unmade_safes = self.safes - self.moves_made
        
        # If safe moves not already made exists, return unmade safe move arbitrarily
        if len(unmade_safes) != 0:
            return unmade_safes.pop()

        # Otherwise return none
        else:
            return None


    def make_random_move(self):
        """
        Returns a move to make on the Minesweeper board.
        Should choose randomly among cells that:
            1) have not already been chosen, and
            2) are not known to be mines
        """
        # Initialise set of potential moves before choosing randomly
        potential_moves = set()

        # Loop through all squares on board to get set of cells that are not chosen yet and are not mines
        for i in range(self.height):
            for j in range(self.width):
                cell = (i, j)

                # Check that cells not moves already made and not mines
                if cell not in self.moves_made and cell not in self.mines:

                    # Add cell to potential_moves set
                    potential_moves.add(cell)

        # If potential random move exists, return random move from all current potential moves
        if len(potential_moves) != 0:
            return potential_moves.pop()

        # Otherwise return none
        else:
            return None
