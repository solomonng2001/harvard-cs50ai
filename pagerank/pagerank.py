import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # Initialise transition model dictionary
    transition_model_dict = dict()

    # If page has outgoing links
    if len(corpus[page]):

        # Add page into transition model dictionary for every unique page in corpus
        for key in corpus:
            # Add probability that random page in corpus chosen with equal probability
            transition_model_dict[key] = (1 - damping_factor) / len(corpus)

        # Add probability that random link in current page chosen with equal probability
        for link in corpus[page]:
            transition_model_dict[link] += damping_factor / len(corpus[page])
    
    # Else if page has no outgoing links
    else:

        # Add page into transition model dictionary for every unique page in corpus
        for key in corpus:
            # Add probability that random page in corpus chosen with equal probability
            transition_model_dict[key] = 1 / len(corpus)
    
    return transition_model_dict


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Initialise sample page count dictionary, where key is page and value is count of pages in all samples
    pagecount_dict = dict()

    # Add page for every unique page in corpus and initialise count to zero
    for key in corpus:
        pagecount_dict[key] = 0

    # Generate first sample by choosing from page at random
    sample_page = random.choice(list(corpus.keys()))

    # Record first sample in sample page count dictionary
    pagecount_dict[sample_page] += 1

    # Loop and generate samples until sample population achieved
    for i in range(n - 1):

        # Initialise previous sample's transition model dictionary
        sample_transition_model = transition_model(corpus, sample_page, damping_factor)

        # Randomly generate new sample based on previous sample's transition model
        sample_page = random.choices(list(sample_transition_model.keys()), weights=list(sample_transition_model.values()), k=1)[0]

        # Record next sample in sample page count dictionary
        pagecount_dict[sample_page] += 1
    
    # Initialise pagerank dictionary, where key is page and value is pagerank
    pagerank_dict = dict()

    # Add pagerank to pagerank dictionary for every page count in pagecount dictionary
    for key in pagecount_dict:
        pagerank_dict[key] = pagecount_dict[key] / n
    
    return pagerank_dict


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Initialise current and new pagerank dictionaries, where key is page and value is pagerank
    pagerank_dict = dict()
    new_pagerank_dict = dict()
    
    # Assign each page a rank of 1 / N, where N is total number of pages in corpus
    for key in corpus:
        pagerank_dict[key] = 1 / len(corpus)

    # Iterate until pagerank for all pages do not change more than 0.001
    while True:

        # Loop through unique pages and calculate pagerank of each page
        for page in corpus:

            # Loop through unique pages and sum up probability of links to current page
            sum_probability_links = 0
            for key in corpus:
                if page in corpus[key]:
                    sum_probability_links += pagerank_dict[key] / len(corpus[key])

            # Calculate new pagerank using pagerank formula
            new_pagerank_dict[page] = (1 - damping_factor) / len(corpus) + damping_factor * sum_probability_links

        # Loop through unique pages to check if all pageranks convergent and update pageranks
        convergent_count = 0
        for page in corpus:

            # Check if new and previous pageranks do not differ by more than 0.001
            if abs(new_pagerank_dict[page] - pagerank_dict[page]) <= 0.001:
                convergent_count += 1

            # Update new pageranks
            pagerank_dict[page] = new_pagerank_dict[page]
        
        # If all pageranks convergent, break while loop
        if convergent_count == len(corpus):
            break

    return pagerank_dict


if __name__ == "__main__":
    main()
