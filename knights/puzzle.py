from logic import *

AKnight = Symbol("A is a Knight")
AKnave = Symbol("A is a Knave")

BKnight = Symbol("B is a Knight")
BKnave = Symbol("B is a Knave")

CKnight = Symbol("C is a Knight")
CKnave = Symbol("C is a Knave")

# Puzzle 0
# A says "I am both a knight and a knave."
knowledge0 = And(

    # A is either a knight or a knave (exclusive or)
    Or(AKnave, AKnight),
    Not(And(AKnave, AKnight)),

    # A is knight only if A's statement is true, else, A is knave
    Biconditional(AKnight, And(AKnight, AKnave))
)

# Puzzle 1
# A says "We are both knaves."
# B says nothing.
knowledge1 = And(
    
    # A, B are each either a knight or a knave (exclusive or)
    Or(AKnave, AKnight),
    Not(And(AKnave, AKnight)),
    Or(BKnave, BKnight),
    Not(And(BKnave, BKnight)),

    # A is knight only if A's statement is true, else, A is Knave
    Biconditional(AKnight, And(AKnave, BKnave))
)

# Puzzle 2
# A says "We are the same kind."
# B says "We are of different kinds."
knowledge2 = And(

    # A, B are each either a knight or a knave (exclusive or)
    Or(AKnight, AKnave),
    Not(And(AKnight, AKnave)),
    Or(BKnight, BKnave),
    Not(And(BKnight, BKnave)),

    # A is knight if A's statements are true, else, A is knave
    Biconditional(AKnight, Or(And(AKnight, BKnight), And(AKnave, BKnave))),

    # B is knight if B's statements are true, else, B is knave
    Biconditional(BKnight, Or(And(AKnight, BKnave), And(AKnave, BKnight)))
)

# Puzzle 3
# A says either "I am a knight." or "I am a knave.", but you don't know which.
# B says "A said 'I am a knave'."
# B says "C is a knave."
# C says "A is a knight."
knowledge3 = And(

    # A, B, C are each either a knight or a knave (exclusive or)
    Or(AKnight, AKnave),
    Not(And(AKnight, AKnave)),
    Or(BKnight, BKnave),
    Not(And(BKnight, BKnave)),
    Or(CKnight, CKnave),
    Not(And(CKnight, CKnave)),

    # A is inconclusive with A's statement
    # Since there is no world in which A is neither knight nor knave,
    # both Implication or Biconditional should work here
    Biconditional(Or(AKnight, AKnave), Or(AKnight, AKnave)),

    # B is Knight only if B's statements are true, else, B is Knave
    # B's statement about A's statement is true if logic corresponds to A's statement
    Biconditional(BKnight, Biconditional(AKnight, AKnave)),
    Biconditional(BKnight, CKnave),

    # C is Knight only if C's statements are true, else, C is Knave
    Biconditional(CKnight, AKnight)
)


def main():
    symbols = [AKnight, AKnave, BKnight, BKnave, CKnight, CKnave]
    puzzles = [
        ("Puzzle 0", knowledge0),
        ("Puzzle 1", knowledge1),
        ("Puzzle 2", knowledge2),
        ("Puzzle 3", knowledge3)
    ]
    for puzzle, knowledge in puzzles:
        print(puzzle)
        if len(knowledge.conjuncts) == 0:
            print("    Not yet implemented.")
        else:
            for symbol in symbols:
                if model_check(knowledge, symbol):
                    print(f"    {symbol}")


if __name__ == "__main__":
    main()
