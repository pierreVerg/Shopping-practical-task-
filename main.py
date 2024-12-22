import csv
import sys
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_selection import SelectKBest, f_classif

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    # if len(sys.argv) != 2:
        # sys.exit("Usage: python shopping.py data")
        # file = sys.argv[1]
        
    file = 'C:/Users/33611/OneDrive - De Vinci/Bureau/RIGA/Software\shopping\shopping.csv'
    
    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(file)
    
    # List of column names in the evidence
    feature_names = ["Administrative", "Administrative_Duration", "Informational", "Informational_Duration",
                     "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates",
                     "PageValues", "SpecialDay", "Month", "OperatingSystems", "Browser",
                     "Region", "TrafficType", "VisitorType", "Weekend"]

    # List of variables to keep
    variables_to_keep = ["PageValues", "ProductRelated", "ProductRelated_Duration", "Administrative",
                         "Month", "Informational", "Administrative_Duration", "Informational_Duration",
                         "SpecialDay", "VisitorType", "BounceRates", "ExitRates"]
    
    # Filter evidence to keep only the specified variables
    filtered_evidence = filter_evidence(evidence, feature_names, variables_to_keep)
    
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE)

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"Prediction accuracy(%): {100 * (y_test == predictions).sum() / len(y_test):.2f}") 
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).
    """
    evidence = []
    labels = []

    # Dictionary to convert months to indices
    month_to_index = {"Jan": 0, "Feb": 1, "Mar": 2, "Apr": 3, "May": 4, "June": 5,
                      "Jul": 6, "Aug": 7, "Sep": 8, "Oct": 9, "Nov": 10, "Dec": 11}

    with open(filename, mode="r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            evidence.append([
                int(row["Administrative"]),
                float(row["Administrative_Duration"]),
                int(row["Informational"]),
                float(row["Informational_Duration"]),
                int(row["ProductRelated"]),
                float(row["ProductRelated_Duration"]),
                float(row["BounceRates"]),
                float(row["ExitRates"]),
                float(row["PageValues"]),
                float(row["SpecialDay"]),
                month_to_index[row["Month"]],
                int(row["OperatingSystems"]),
                int(row["Browser"]),
                int(row["Region"]),
                int(row["TrafficType"]),
                1 if row["VisitorType"] == "Returning_Visitor" else 0,
                1 if row["Weekend"] == "TRUE" else 0])
            # Convert the Revenue column into 0 or 1 for labels
            labels.append(1 if row["Revenue"] == "TRUE" else 0)

    return (evidence, labels)
    


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    # Initialize the k-NN model with k=1
    #model = KNeighborsClassifier(n_neighbors=1)
    model = KNeighborsClassifier(n_neighbors=1, metric='manhattan')

    # Train the model
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
    # Initialize counts for TP, FP, TN, FN
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    for true_label, predicted_label in zip(labels, predictions):
        if true_label == 1 and predicted_label == 1:
            TP += 1  # True positive
        elif true_label == 1 and predicted_label == 0:
            FN += 1  # False negative
        elif true_label == 0 and predicted_label == 1:
            FP += 1  # False positive
        elif true_label == 0 and predicted_label == 0:
            TN += 1  # True negative
    
    # Sensitivity (True Positive Rate)
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    # Specificity (True Negative Rate)
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    
    return (sensitivity, specificity)

def filter_evidence(evidence, feature_names, variables_to_keep):
    """
    Filters the evidence data to keep only the specified variables.

    Parameters:
    - evidence: list or numpy array of input data.
    - feature_names: list of column names in the evidence.
    - variables_to_keep: list of names of variables to keep.

    Returns:
    - filtered_evidence: numpy array containing only the selected columns.
    """
    # Identify the indices of the variables to keep
    indices_to_keep = [i for i, name in enumerate(feature_names) if name in variables_to_keep]
    
    # Filter evidence to keep only the selected columns
    filtered_evidence = np.array(evidence)[:, indices_to_keep]
    
    return filtered_evidence

if __name__ == "__main__":
    main()
