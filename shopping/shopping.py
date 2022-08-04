import csv
import sys
import calendar

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    # Open and read CSV file
    with open("shopping.csv") as file:
        reader = csv.DictReader(file)

        # Initialise evidence and label lists respectively
        evidence = list()
        labels = list()
        
        # For every non-header row, split data into evidence and labels
        for row in reader:
            
            # Numerialise data
            numeralised_row = numeralise(row)

            # Collate list of all evidence and labels separately
            evidence.append(numeralised_row[:17])
            labels.append(numeralised_row[17])

    return (evidence, labels)


def numeralise(row):
    """
    Given a row of CSV data, numeralise data and return numeralised row
    1. Convert strings in integers (months, returning or new visitor, boolean strings).
    2. Ensure integer data types.
    3. Ensure float data types.
    """
    # Initialise numeralised row list, to be returned at end of function
    numeralised_row = list()

    # List of month abbreviations (months in shopping.csv are abbreviated except for month of june)
    months = list(calendar.month_abbr)[1:]
    months[5] = "June"

    # Dictionary converting strings to integers
    string2integer = dict()
    for month in months:
        string2integer[month] = months.index(month)
    string2integer["Returning_Visitor"] = 1
    string2integer["New_Visitor"] = 0
    string2integer["Other"] = 0
    string2integer["FALSE"] = 0
    string2integer["TRUE"] = 1

    # List of integers and floats respectively
    integers = ["Administrative", "Informational", "ProductRelated", "Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType", "Weekend"]
    floats = ["Administrative_Duration", "Informational_Duration", "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues", "SpecialDay"]

    # Numeralise each element in row, where element is each cell's header
    for element in row:

        # If element is string, use string to integer dictionary for conversion
        if row[element] in string2integer:
            numeralised_row.append(string2integer[row[element]])
        
        # Else if element in integers list, ensure integer data type
        elif element in integers:
            numeralised_row.append(int(row[element]))

        # Else if element in floats list, ensure float data type
        elif element in floats:
            numeralised_row.append(float(row[element]))

    return numeralised_row


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    # Initialise k-nearest-neighbor model
    model = KNeighborsClassifier(n_neighbors=1)

    # Feed evidence and labels lists into training model
    model.fit(evidence, labels)
    
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    # Initialise count of true positive, false positives, true negatives and false negatives
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    # For each label in labels list
    for label, prediction in zip(labels, predictions):

        # If label is true, count true positives and false positives in corresponding prediction
        if label == 1:
            if prediction == 1:
                true_positives += 1
            else:
                false_positives += 1
        else:
            if prediction == 0:
                true_negatives += 1
            else:
                false_negatives += 1
    
    # Calculate sensitivity and specificity
    sensitivity = float(true_positives / (true_positives + false_positives))
    specificity = float(true_negatives / (true_negatives + false_negatives))

    return (sensitivity, specificity)


if __name__ == "__main__":
    main()
