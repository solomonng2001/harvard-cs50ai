import nltk
import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP VP | S ConjP
ConjP -> Conj VP | Conj S | Conj NP
VP -> V | V NP | Adv VP | VP Adv | VP PP
NP -> N | Det N | Det AP
AP -> Adj N | Adj AP
PP -> P NP
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    # Tokenize words in sentence
    tokenized = nltk.word_tokenize(sentence)

    # Initialise set of non-alphabet words to be removed
    remove = set()
    
    # For each tokenized word, ensure word is lowercase and add non-alphabet words to remove list
    for i in range(len(tokenized)):
        tokenized[i] = tokenized[i].lower()
        if not tokenized[i].isalpha():
            remove.add(tokenized[i])

    # Initialise list of tokenized alphabetical words to be returned
    tokenized_alpha = list()

    # Alphabetical words = not in remove list
    for word in tokenized:
        if word not in remove:
            tokenized_alpha.append(word)

    return tokenized_alpha


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    # Initialise list of noun phrases to be returned
    noun_phrases = list()

    # Find all noun phrases in tree that do not consist of children noun phrases, and append to list
    for phrase in tree.subtrees():
        if phrase.label() == "NP":
            noun_phrases.append(phrase)

    return noun_phrases


if __name__ == "__main__":
    main()
