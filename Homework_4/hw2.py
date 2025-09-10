import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Reading the excel file as a Dataframe
#df = pd.read_csv("Oil_Analysis_Raw_Data_2025_Timestamped.csv", header = "infer")

def preprocess_oil(df):
    # Changing data type of "Timestamp" from object to datetime:
    # Setting the parameter "errors" to "coerce" so that invalid paring will return the input (and no raise an exception)

    Timestamp = df["Timestamp"]
    Timestamp = pd.to_datetime(Timestamp, errors = "coerce")
    df["Timestamp"] = Timestamp

    # Set the index of the DataFrame
    df.set_index(keys = "Timestamp", inplace = True)

    # Now converting any other feature column which has 'object' datatype to 'numeric'
    Iron = df["Iron"]
    Iron = pd.to_numeric(Iron, errors = "coerce")
    df["Iron"] = Iron

    Tin = df["Tin"]
    Tin = pd.to_numeric(Tin, errors = "coerce")
    df["Tin"] = Tin

    Vanadium = df["Vanadium"]
    Vanadium = pd.to_numeric(Vanadium, errors = "coerce")
    df["Vanadium"] = Vanadium

    Molybdenum = df["Molybdenum"]
    Molybdenum = pd.to_numeric(Molybdenum, errors = "coerce")
    df["Molybdenum"] = Molybdenum

    # Deleting all rows with pumps and hydraulics and keeping only 
    df = df[~df['Equipment Type'].isin(['Pumps', 'Hydraulics'])]

    # Labeling the equipment type from 0-1 in the order of highest frequency of occurance
    # Count the frequency of each equipment type
    equipment_counts = df['Equipment Type'].value_counts(normalize=True) * 100
    
    # Assign labels (0,1) based on frequency (0 → most frequent, 1 → least frequent)
    equipment_mapping = {equipment: i for i, equipment in enumerate(equipment_counts.index)}

    # Map equipment types to their new labels
    df['Equipment Type'] = df['Equipment Type'].map(equipment_mapping)

    # For label encoding the 'Alarms' and 'Diagnostics' columns, the highest frequency of occurence of an entry gets assigned value of 0 and so on. 
    # Calculating %ages of different entries in the 'Alarms' column
    percentage_alarms = df['Alarms'].value_counts(normalize=True) * 100

    # Create a mapping (most frequent gets lowest number, starting from 0)
    label_mapping_alarms = {label: idx for idx, label in enumerate(percentage_alarms.index)}

    # Apply mapping to the column
    df['Alarms'] = df['Alarms'].map(label_mapping_alarms)

    # Calcualting %ages of different entries in the 'Diagnostics' column
    percentage_diag = df['Diagnostics'].value_counts(normalize=True) * 100

    # Create mapping (most frequent gets lowest number, starting from 0)
    label_mapping_diag = {label: idx for idx, label in enumerate(percentage_diag.index)}

    # Apply mapping to the column
    df['Diagnostics'] = df['Diagnostics'].map(label_mapping_diag)

    # Dropping the "Unnamed: 0" column:
    df = df.drop("Unnamed: 0", axis = 'columns')

    # Now dropping a feature column where percentage NaNs are more than 50%
    # Calculate the percentage of NaNs per column
    nan_pct = df.isna().sum() / len(df)

    # Drop columns where more than 50% of the values are NaNs
    df = df.drop(columns=nan_pct[nan_pct > 0.5].index)

    # Then drop rows where all the columns have NaNs:
    df_nonull = df.dropna(how = "all")

    # Now we deal with the features with NaNs by using the ffill() method:
    df_nonull = df_nonull.ffill()

    # Applying the various constraints only to the oil analysis data as asked in the problem statement
    # 1. Removing all concentrations of chemicals having negative values
    # Defining the columns corresponding to chemical concentrations (Iron to Phosphorous)
    chemical_columns = df_nonull.loc[:, "Iron":"Phosphorus"].columns

    # Removing rows where any chemical concentration value is negative
    df_filtered_non_zero = df_nonull[~(df_nonull[chemical_columns] < 0).any(axis=1)]

    # 2. Removing rows where Aluminum conc. > 250 ppm
    df_filtered_alum = df_filtered_non_zero[~(df_filtered_non_zero["Aluminum"] > 250)]

    # 3. Removing rows where Copper conc. > 30 ppm
    df_filtered_copper = df_filtered_alum[~(df_filtered_alum["Copper"] > 30)]

    df_clean = df_filtered_copper

    # Splitting the DataFrame with clean data:
    df_features = df_clean.iloc[:, 0:29]
    df_labels = df_clean.iloc[:, 29:32]

    # Calculate the means and standard deviations for the features:
    means = df_features.mean(axis = 'index')
    stds = df_features.std(axis = 'index')

    # Now we calculate the coeficcient of variability for each feature:
    cvs = means/stds

    # Filter out features with CV >= 0.05
    selected_features = cvs[cvs >= 0.05].index

    # Keep only selected features in df_features
    df_features = df_features[selected_features]

    # Compute the correlation matrix: by default, we use Pearson correlation coefficient
    df_corr_mat = df_features.corr()

    # Removing redundant features 
    corr_matrix=df_corr_mat

    # Get the lower triangular part of the matrix excluding the diagonal 
    lower_triangular = np.tril(corr_matrix, k=-1)

    # Find row indices where correlation is >= 0.95
    row_indices = np.where(lower_triangular >= 0.95)[0]

    # Convert to a vector (list)
    row_indices_vector = row_indices.tolist()

    # Removing duplicate feature indices and storing one unque feature for multiple common redundancies.
    unique_list = list(set(row_indices_vector))

    # Drop columns by their feature indices
    df_features = df_features.drop(df_features.columns[unique_list], axis=1)

    # Restructure the clean dataframe from the processed features
    df_clean = pd.concat([df_features, df_labels], axis=1)

    # For resampling, aggregator 'mean' is considered for all feature columns
    mean_dict = {col: "mean" for col in df_clean.columns if col not in ['Equipment Type', 'Alarms', 'Diagnostics']}

    # Assign 'max' to the label columns
    for col in ['Equipment Type', 'Alarms', 'Diagnostics']:
        if col in df_clean.columns:
            mean_dict[col] = "max"

    # Resampling the data at 12 hr intervals
    df_clean_resampled = df_clean.resample("12h").agg(mean_dict).dropna()

    # We convert the DataFrames with the features (df_clean) and with the labels (df_labels) to NumPy arrays:
    features = np.asarray(df_clean_resampled.iloc[:, :-3])
    labels = np.asarray(df_clean_resampled.iloc[:, -3:])

    return features, labels

    