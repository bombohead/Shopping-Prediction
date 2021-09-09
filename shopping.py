import csv
import sys
import datetime

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    """if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")"""

    # Load data from spreadsheet and split into train and test sets
    #evidence, labels = load_data(sys.argv[1])
    evidence, labels = load_data("shopping.csv")
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

    csvfile = open(filename)
    csv_reader = csv.reader(csvfile, delimiter = ",")

    lineCount = 0
    evidence = []
    labels = []
    
    for row in csv_reader:

        #Checks all rows except the header row
        if lineCount != 0:
            temp = []
            #Adds administrative
            temp.append(int(row[0]))
            #Adds administrative duration
            temp.append(float(row[1]))
            #Adds informational
            temp.append(int(row[2]))
            #Adds informational duration
            temp.append(float(row[3]))
            #Adds product related
            temp.append(int(row[4]))
            #Adds product related duration
            temp.append(float(row[5]))
            #Adds bounce rates
            temp.append(float(row[6]))
            #Adds exit rates
            temp.append(float(row[7]))
            #Adds page values
            temp.append(float(row[8]))
            #Adds special day
            temp.append(float(row[9]))

            #Adds month
            monthObject = datetime.datetime.strptime(row[10][0:3], "%b")
            temp.append(monthObject.month - 1)

            #Adds OS
            temp.append(int(row[11]))
            #Adds browser
            temp.append(int(row[12]))
            #Adds region
            temp.append(int(row[13]))
            #Adds traffic type
            temp.append(int(row[14]))
            
            #Adds visitor type
            if ("New" in row[15]):
                temp.append(0)
            else:
                temp.append(1)
        
            #Adds weekend
            if ("FALSE" in row[16]):
                temp.append(0)
            else:
                temp.append(1)

            #Adds label
            #Adds weekend
            if ("FALSE" in row[17]):
                labels.append(0)
            else:
                labels.append(1)
            
            evidence.append(temp)
        
        lineCount += 1

    return (evidence, labels)

def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """

    #Creates k-nearest neighbour model
    model = KNeighborsClassifier(n_neighbors=1)
    #Fits model to evidence and labels
    model.fit(evidence, labels)

    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """

    positive = 0
    positiveCorrect = 0

    negative = 0
    negativeCorrect = 0

    #Iterates through labels and predictions to find the total
    #number of predictions and total number of correct predictions
    for label, predicted in zip(labels, predictions):
        if (label == predicted):
            if (label == 1):
                positive += 1
                positiveCorrect += 1
            else:
                negative += 1
                negativeCorrect += 1
        else:
            if (label == 1):
                positive += 1
            else:
                negative += 1

    #Calculates the sensitivity and specificity by finding
    #the percentage of correct predictions
    sensitivity = positiveCorrect / positive
    specificity = negativeCorrect / negative

    return (sensitivity, specificity)

if __name__ == "__main__":
    main()
