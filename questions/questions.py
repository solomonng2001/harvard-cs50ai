import nltk
import sys
import os
import string
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    # Initialise dictionary mapping filename of .txt file to file's contents
    files = dict()

    # For each file in directory, read contents
    for filename in os.listdir(directory):
        file = open(os.path.join(directory, filename), "r")

        # Save to dictionary: Map filename to file's contents as string
        files[filename] = str(file.read())

    return files


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    # Initialise list of filtered words to be returned
    filtered_words = list()
    
    # Tokenize string into list of words and make all letters in string lowercase
    words = nltk.word_tokenize(document.lower())

    # Remove punctuation and stopwords from words
    for word in words:
        if word not in string.punctuation and word not in nltk.corpus.stopwords.words("english"):
            filtered_words.append(word)
    
    return filtered_words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    # Create set of unique words from all words in documents/various .txt files
    unique_words = set()
    for document in documents:
        unique_words.update(documents[document])

    # Create dictionary mapping words in documents to their IDF values
    idfs = dict()

    # For each unique word, calculate IDF value
    for word in unique_words:

        # Loop through each document to count number of documents that contain word
        document_count = 0
        for document in documents:
            if word in documents[document]:
                document_count += 1
        
        # Record word and idf value in dictionary
        idfs[word] = math.log(len(documents) / document_count)
    
    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    # Create list mapping files to the sum of their tfidfs
    files_tfidfs = list()

    # For each file, calculate sum of tf-idf values for query words
    for file in files:
        tfidf_sum = 0
        for term in query:
            if term in files[file]:

                # Calculate term frequency of word in file
                tf = 0
                for word in files[file]:
                    if term == word:
                        tf += 1

                # Add tf-idf to sum
                tfidf_sum += tf * idfs[term]
        
        # Append tf-idf sum to list
        files_tfidfs.append((file, tfidf_sum))
    
    # Create list of n files ranked according to tf-idf to be returned
    files_tfidfs.sort(key=lambda x: x[1], reverse=True)
    ranked_files = [element[0] for element in files_tfidfs[:n]]
    return ranked_files


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    # Create list mapping sentences to the sum of their idfs and query term density
    sentences_idfs_qtds = list()

    # For each sentence, calculate sum of idf values for query words
    for sentence in sentences:
        idf_sum = 0
        query_term_count = 0
        for term in query:
            if term in sentences[sentence]:

                # Add idf to sum
                idf_sum += idfs[term]
        
        # Calculate query term density
        for word in sentences[sentence]:
            if word in query:
                query_term_count += 1
        query_term_density = query_term_count / len(sentence)

        # Append idf sum and query term density to list
        sentences_idfs_qtds.append((sentence, idf_sum, query_term_density))
            
    
    # Create list of n sentences ranked according to tf-idf to be returned
    sentences_idfs_qtds.sort(key=lambda x: (x[1], x[2]), reverse=True)
    ranked_files = [element[0] for element in sentences_idfs_qtds[:n]]
    return ranked_files


if __name__ == "__main__":
    main()
