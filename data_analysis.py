import pandas as pd
import numpy as np
from scipy.stats import pearsonr


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



def correlation_table_and_filter(evidence, labels, feature_names, threshold=0.05):
    """
    Parameters:
        - evidence: list or DataFrame of input data (features).
        - labels: list or Series of labels (target).
        - feature_names: list of column names in `evidence`.
        - threshold: absolute value threshold for retaining correlations.

    Returns:
        - filtered_evidence: numpy array containing the remaining features.
        - filtered_features: list of names of the remaining variables.
        - correlation_df: DataFrame of correlations before filtering.taFrame des corrélations avant filtrage.
    """
    
    evidence = np.array(evidence)
    labels = np.array(labels)
    
    correlations = []
    for i in range(evidence.shape[1]):
        corr, _ = pearsonr(evidence[:, i], labels)
        correlations.append(corr)
    
    # DataFrame with Result
    correlation_df = pd.DataFrame({"Variable": feature_names,
                                   "Correlation with Labels": correlations})
    correlation_df = correlation_df.sort_values(by="Correlation with Labels", ascending=False).reset_index(drop=True)

    # Filter with low corrolation (between -threshold et +threshold)
    correlation_df["Keep"] = correlation_df["Correlation with Labels"].apply(
        lambda x: abs(x) > threshold)

    filtered_features = correlation_df[correlation_df["Keep"]]["Variable"].tolist()

    filtered_indices = [i for i, name in enumerate(feature_names) if name in filtered_features]
    filtered_evidence = evidence[:, filtered_indices]

    return filtered_evidence, filtered_features, correlation_df

feature_names = ["Administrative", "Administrative_Duration", "Informational", "Informational_Duration",
                 "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates",
                 "PageValues", "SpecialDay", "Month", "OperatingSystems", "Browser",
                 "Region", "TrafficType", "VisitorType", "Weekend"]


evidence, labels = load_data('C:/Users/33611/OneDrive - De Vinci/Bureau/RIGA/Software/shopping/shopping.csv')


filtered_evidence, filtered_features, correlation_df = correlation_table_and_filter(
    evidence, labels, feature_names, threshold=0.05)


print(correlation_df)

print("\nPreserved variables :", filtered_features)
print("\nFiltered data (shape)):", filtered_evidence.shape)

