import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    # Initialise joint probability variable
    joint_probability = 1

    # Loop through names in people dictionary to calculate probabilities
    for name in people:
            
        # Initialise number of gene_copies in current person
        gene_copies_name = gene_copies(name, one_gene, two_genes)
        
        # If no parents in csv, multiply joint probability by general probability
        if not people[name]["mother"] and not people[name]["father"]:

            # General probability equals multiple of probability of gene copy and probability of trait (depending on gene copy)
            joint_probability *= PROBS["gene"][gene_copies_name] * PROBS["trait"][gene_copies_name][name in have_trait]

        # Else if parents in csv, multiply joint probability by heredity probability
        else:

            # Initialise child's and parents' genes in form of list [bool, bool]
            gene_copies_mother = gene_copies(people[name]["mother"], one_gene, two_genes)
            gene_copies_father = gene_copies(people[name]["father"], one_gene, two_genes)

            # multiply joint probability by heredity probability and trait probability (depending on trait)
            joint_probability *= heredity_probability(gene_copies_name, gene_copies_mother, gene_copies_father) * PROBS["trait"][gene_copies_name][name in have_trait]

    return joint_probability


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    # Update each person in probabilites dictionary
    for person in probabilities:

        # Update probabilities for both gene and trait
        probabilities[person]["gene"][gene_copies(person, one_gene, two_genes)] += p
        probabilities[person]["trait"][person in have_trait] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    # Update each person in probabilities dictionary
    for person in probabilities:

        # Normalise sum of true and false trait probablities such that sum is equal to 1
        trait_true = probabilities[person]["trait"][True]
        trait_false = probabilities[person]["trait"][False]
        probabilities[person]["trait"][True] = trait_true / (trait_true + trait_false)
        probabilities[person]["trait"][False] = trait_false / (trait_true + trait_false)

        # Normalise sum of gene probabilities such that sum is equal to 1
        # Obtain sum of all gene probabilities
        gene_sum = 0
        for gene in probabilities[person]["gene"]:
            gene_sum += probabilities[person]["gene"][gene]
        
        # Update each gene probability to normalcy
        for gene in probabilities[person]["gene"]:
            probabilities[person]["gene"][gene] = probabilities[person]["gene"][gene] / gene_sum


def gene_copies(name, one_gene, two_genes):
    """
    Returns integer number of gene copies "0, 1 or 2" depending on whether
    name is in the one_gene or two_genes sets or none.
    Accepts as parameter: name, one_gene set and two_genes set.
    """
    # If person listed in one gene set, return gene copies 1
    if name in one_gene:
        return 1

    # Else if person listed in two genes set, return gene copies 2
    elif name in two_genes:
        return 2
    
    # Else, person is not listed in both one gene and two genes sets, so return gene copies 0
    else:
        return 0


def heredity_probability(gene_copies_child, gene_copies_mother, gene_copies_father):
    """
    Return heredity probabilities of child's genes from parent's genes.
    Accepts as parameter: number of gene copies of child and each parent.
    """
    # If child has 0 gene copies, probability that both parent pass non-impairment genes
    if gene_copies_child == 0:
        heredity_probability = (1 - parent_probability(gene_copies_mother)) * (1 - parent_probability(gene_copies_father))

    # Else if child has 1 gene copy, proability that mother passes on gene and father doesn't and vice versa
    elif gene_copies_child == 1:
        heredity_probability = parent_probability(gene_copies_mother) * (1 - parent_probability(gene_copies_father)) + (1 - parent_probability(gene_copies_mother)) * parent_probability(gene_copies_father)

    # Else if child ahs 2 gene copies, probability that both parent pass impairment gene
    else:
        heredity_probability = parent_probability(gene_copies_mother) * parent_probability(gene_copies_father)

    return heredity_probability


def parent_probability(gene_copies):
    """
    Return probability that parent passes impairment gene to child,
    depending on parent's number of gene copies respectively.
    Takes as parameter: integer number of gene copies.
    """
    # If parent has 0 impairment gene copies, gene must mutate to be passed on
    if gene_copies == 0:
        return PROBS["mutation"]

    # Else if parent has 1 impairment gene copy,
    # gene passed on with no mutation of impairment gene and mutation of non-impairment gene
    elif gene_copies == 1:
        return (0.5 * (1 - PROBS["mutation"]) + 0.5 * PROBS["mutation"])
    
    # Else if parent has 2 impairment gene copies, gene passed on with no mutation
    else:
        return (1 - PROBS["mutation"])


if __name__ == "__main__":
    main()
