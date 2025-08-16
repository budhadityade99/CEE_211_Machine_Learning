import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from numpy import mean, sqrt, square
from scipy.fftpack import fft, fftfreq
import scipy.io as sio
import math
from scipy.stats import kurtosis, skew
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
import seaborn as sns
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from keras.models import Sequential # type: ignore
from keras.layers import Input, Dense, Dropout # type: ignore
from utils import plot_confusion_matrix
from keras.utils import plot_model
warnings.filterwarnings("ignore")
from keras.callbacks import EarlyStopping
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.class_weight import compute_class_weight

def preprocessing(df_sensor):

    # Dropping 'unnamed' column and retaining the 1st 13 features only
    df_sensor = df_sensor.drop(["Unnamed: 0", "Current.1", "Weight Meter", "Current.2", "V.Zero", "Current.3", "V.Zero.1", "Current.4", "V.Zero.2"], axis = "columns")

    # Naming the features
    columns = ["Current [I]", "Shaft Pressure [Ps]", "Return Temperature [Tr]", "Socket Liner Temperature [Tl]", "External Temperature [Te]", "Feeder Temperature [Tf]", "Shaft Temperature [Ts]", "Ring Vibration [RV1]", "Ring Vibration [RV2]", "Ring Vibration [RV3]", "Ring Vibration [RV4]", "Vessel Level [L]", "Setting [S]"]

    # Set to store unique non-numeric values
    unique_non_numeric_values = set()

    # Loop through each column and extract unique non-numeric values
    for col in columns:
        # Convert column to numeric (non-numeric values become NaN)
        rogue = pd.to_numeric(df_sensor[col], errors="coerce")
        
        # Get non-numeric values and add them to the set
        unique_non_numeric_values.update(df_sensor.loc[rogue.isna(), col].unique())

    # Convert set to a list for final output
    unique_non_numeric_list = list(unique_non_numeric_values)

    # Replacing the non-numeric entries as 'NaN'
    df_sensor = df_sensor.replace(
    to_replace = {"Bad Input":np.nan, "I/O Timeout":np.nan, "Intf Shut":np.nan, " ":np.nan, "Shutdown":np.nan}
    )

    # Find and drop rows where all columns except 'Timestamp' have NaN values
    nan_rows_except_timestamp = df_sensor[df_sensor.drop(columns='Timestamp').isna().all(axis=1)].index.tolist()
    df_sensor = df_sensor.drop(index=nan_rows_except_timestamp)

    # Reset index
    df_sensor = df_sensor.reset_index(drop=True)

    # Convert 'Timestamp' column to datetime format
    df_sensor['Timestamp'] = pd.to_datetime(df_sensor['Timestamp'], errors='coerce')

    # Define the time range
    interval_start = pd.to_datetime("1/2/2017 12:42")
    interval_end = pd.to_datetime("8/31/2019 19:23")

    # Filter out rows where Timestamp is before interval_start or after interval_end
    df_sensor = df_sensor[(df_sensor['Timestamp'] >= interval_start) & (df_sensor['Timestamp'] <= interval_end)]

    # Reset index
    df_sensor = df_sensor.reset_index(drop=True)

    # change the data type from object to numeric:

    df_sensor["Current [I]"] =                      pd.to_numeric(df_sensor["Current [I]"],                         errors = "coerce")
    df_sensor["Shaft Pressure [Ps]"] =              pd.to_numeric(df_sensor["Shaft Pressure [Ps]"],                 errors = "coerce")
    df_sensor["Return Temperature [Tr]"] =          pd.to_numeric(df_sensor["Return Temperature [Tr]"],             errors = "coerce")
    df_sensor["Socket Liner Temperature [Tl]"] =    pd.to_numeric(df_sensor["Socket Liner Temperature [Tl]"],       errors = "coerce")
    df_sensor["External Temperature [Te]"] =        pd.to_numeric(df_sensor["External Temperature [Te]"],           errors = "coerce")
    df_sensor["Feeder Temperature [Tf]"] =          pd.to_numeric(df_sensor["Feeder Temperature [Tf]"],             errors = "coerce")
    df_sensor["Shaft Temperature [Ts]"] =           pd.to_numeric(df_sensor["Shaft Temperature [Ts]"],              errors = "coerce")
    df_sensor["Ring Vibration [RV1]"] =             pd.to_numeric(df_sensor["Ring Vibration [RV1]"],                errors = "coerce")
    df_sensor["Ring Vibration [RV2]"] =             pd.to_numeric(df_sensor["Ring Vibration [RV2]"],                errors = "coerce")
    df_sensor["Ring Vibration [RV3]"] =             pd.to_numeric(df_sensor["Ring Vibration [RV3]"],                errors = "coerce")
    df_sensor["Ring Vibration [RV4]"] =             pd.to_numeric(df_sensor["Ring Vibration [RV4]"],                errors = "coerce")
    df_sensor["Vessel Level [L]"] =                 pd.to_numeric(df_sensor["Vessel Level [L]"],                    errors = "coerce")
    df_sensor["Setting [S]"] =                      pd.to_numeric(df_sensor["Setting [S]"],                         errors = "coerce")

    # Define the columns corresponding to monitoring variable
    monitoring_variables = df_sensor.loc[:, "Current [I]":"Setting [S]"].columns

    # Remove rows where any monitoring variable is negative
    df_sensor_filtered = df_sensor.loc[~(df_sensor[monitoring_variables] <= 0).any(axis=1).to_numpy()]

    # Reset index
    df_sensor_filtered = df_sensor_filtered.reset_index(drop=True)

    # Set the index of the DataFrame
    Timestamp = df_sensor_filtered["Timestamp"]
    Timestamp = pd.to_datetime(Timestamp, errors = "coerce")
    df_sensor_filtered["Timestamp"] = Timestamp
    df_sensor_filtered.set_index(keys = "Timestamp", inplace = True)
    
    # Missing data percentage per column:
    # This returns a Series object
    nan_pct = df_sensor_filtered.isna().sum() / len(df_sensor_filtered) 

    # Fill in NaNs with the Forward Fill method:
    df_sensor_filtered = df_sensor_filtered.ffill(axis = "index")

    # Now we calculate the coeficcient of variability for each feature:
    means = df_sensor_filtered.mean(axis = 'index')
    stds = df_sensor_filtered.std(axis = 'index')
    cvs = stds / means 

    # Filter out features with CV <= 0.05
    selected_features = cvs[cvs >= 0.05].index

    # Keep only selected features in df_sensor_filtered
    df_sensor_filtered = df_sensor_filtered[selected_features]

    # Compute the correlation matrix: by default, we use Pearson correlation coefficient
    df_corr_mat = df_sensor_filtered.corr()

    # Getting the lower-triangular matrix
    lower_triangular = np.tril(df_corr_mat, k=-1)

    # Find row indices where the absoluet value of correlation is >= 0.95
    row_indices = np.where(abs(lower_triangular) >= 0.95)[0]

    # Convert to a vector (list)
    row_indices_vector = row_indices.tolist()

    # find the unique values
    unique_list = list(set(row_indices_vector))

    # Drop columns by their indices
    df_sensor_filtered = df_sensor_filtered.drop(df_sensor_filtered.columns[unique_list], axis=1)

    return df_sensor_filtered

def downtime_filtering():

    df_downs = pd.read_csv('Downtime_Events.csv', header = 'infer')
    df_downs = df_downs.drop(["Unnamed: 0", "Health State"], axis = "columns")

    # Filter the dataframe to get rows where Downtime Event Type = Failure
    filter = (df_downs["Downtime Event Type"].eq("Failure"))
    df_downs = df_downs[filter]

    # Filter the dataframe to get rows where Equipment = 140-CR-004
    filter = (df_downs["Equipment"].eq("140-CR-004"))
    df_downs = df_downs[filter]

    # Convert 'Start' and 'End' columns to datetime format
    df_downs["Start"] = pd.to_datetime(df_downs["Start"])
    df_downs["End"] = pd.to_datetime(df_downs["End"])

    # Sort df_downs by 'Start' to enable efficient searching
    df_downs = df_downs.sort_values(by="Start")

    # Reset index if needed
    df_downs = df_downs.reset_index(drop=True)

    return df_downs

def health_state_labeling(df_downs,df_sensor_filtered):

    # Create a DataFrame with a single column of zeros
    df_zeros = pd.DataFrame(np.zeros(len(df_sensor_filtered)), columns=["Crusher Health"]).astype(int)

    # Extract timestamps as a NumPy array for faster operations
    timestamps = df_sensor_filtered.index.to_numpy()
    start_times = df_downs["Start"].to_numpy()
    end_times = df_downs["End"].to_numpy()

    # Vectorized search: Find the first interval that could contain each timestamp
    start_indices = np.searchsorted(start_times, timestamps, side="right") - 1

    # Initialize list to store indices of matching timestamps
    indices_in_intervals = []

    # Iterate over timestamps and check only the relevant interval
    for i, (ts, si) in enumerate(zip(timestamps, start_indices)):
        if si >= 0 and ts <= end_times[si]:  # Check only if within a valid interval
            indices_in_intervals.append(i)

    # Update the values at the specified indices to 1
    df_zeros.iloc[indices_in_intervals] = 1

    # Reset index to avoid index conflicts
    df_sensor_filtered_reset = df_sensor_filtered.reset_index()
    df_zeros_reset = df_zeros.reset_index(drop=True)  # Drop index to keep it aligned

    # Concatenate along columns
    df_sensor_filtered_reset = pd.concat([df_sensor_filtered_reset, df_zeros_reset], axis=1)

    # Restore the original index
    df_sensor_filtered_reset.set_index("Timestamp", inplace=True)  # Assuming original index was 'Timestamp'

    # Update the original dataframe
    df_sensor_filtered_labeled = df_sensor_filtered_reset

    return df_sensor_filtered_labeled

def PCA_transform(df_sensor_filtered_labeled):

    # get the features and labels
    features = np.asarray(df_sensor_filtered_labeled.iloc[:, :-1])
    labels = np.asarray(df_sensor_filtered_labeled.iloc[:, -1:])

    # train/test split
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size = 0.20, random_state = 42, shuffle = True)

    # Min Max Scaling (because all the data is positive)
    sc = StandardScaler()
    sc.fit(x_train)
    x_train = sc.transform(x_train)
    x_test = sc.transform(x_test)

    pca = PCA(n_components=0.95)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)

    print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
    print(f"Total Explained Variance: {pca.explained_variance_ratio_.sum():.2%}")

    # Plot explained variance ratio
    n = len(pca.explained_variance_ratio_)
    comps = ["Component " + str(i + 1) + f": {pca.explained_variance_ratio_[i]:.2%}" for i in range(n)]
    plt.figure(figsize=(5, 4))
    plt.pie(pca.explained_variance_ratio_, labels=comps, autopct='%1.1f%%')
    plt.title('Explained Variance Ratio by Principal Components')
    plt.show()

    return x_train_pca, x_test_pca, y_train, y_test

def ML_modeler(features,labels,labels_names,modeler):
    
    # Train/Split the data
    x_train_func, x_test_func, y_train_func, y_test_func = train_test_split(features, labels, test_size = 0.20, random_state = 42, shuffle = True)

    # Normalizing the splitted data
    sc = StandardScaler()
    sc.fit(x_train_func)
    x_train_func = sc.fit_transform(x_train_func)
    x_test_func = sc.transform(x_test_func)

    # We create an instance (object) from the sklearn LogisticRegression class
    model = modeler

    # Fit (train) the model to the training dataset:
    model.fit(x_train_func, y_train_func)

    # Check performance on the train dataset:
    score = model.score(x_train_func, y_train_func)
    print(f"Accuracy on train dataset: {score:.2%}")

    # Get the model's predictions for the test dataset:
    y_pred = model.predict(x_test_func)

    # Get and print a classification report: performance metrics on the test dataset
    print(classification_report(y_test_func, y_pred, target_names = labels_names, digits = 5))

    # Important! If the dataset is unbalanced, calculate the "balanced accuracy":
    print(f"Balanced Accurancy: {balanced_accuracy_score(y_test_func, y_pred):.5f}")
    print(f"Unbalanced Accurancy: {accuracy_score(y_test_func, y_pred):.5f}")

    # Compute confusion matrix
    cm = confusion_matrix(y_test_func, y_pred)

    '''# Plotting
    fig = plt.figure(figsize=(8, 4))  # Adjust figure size as needed
    sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", xticklabels=labels_names, yticklabels=labels_names)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")'''

    return cm

def ML_modeler_for_PCA(x_train_pca, x_test_pca, y_train, y_test, labels_names, modeler):
    
    # Train/Split the data
    x_train_func=x_train_pca
    x_test_func=x_test_pca
    y_train_func=y_train
    y_test_func=y_test

    # We create an instance (object) from the sklearn LogisticRegression class
    model = modeler

    # Fit (train) the model to the training dataset:
    model.fit(x_train_func, y_train_func)

    # Check performance on the train dataset:
    score = model.score(x_train_func, y_train_func)
    print(f"\n Accuracy on train PCA dataset: {score:.2%}")

    # Get the model's predictions for the test dataset:
    y_pred = model.predict(x_test_func)

    # Get and print a classification report: performance metrics on the test dataset
    print(classification_report(y_test_func, y_pred, target_names = labels_names, digits = 5))

    # Important! If the dataset is unbalanced, calculate the "balanced accuracy":
    print(f"Balanced PCA Accurancy: {balanced_accuracy_score(y_test_func, y_pred):.5f}")
    print(f"Unbalanced PCA Accurancy: {accuracy_score(y_test_func, y_pred):.5f}")

    # Compute confusion matrix
    cm = confusion_matrix(y_test_func, y_pred)

    '''# Plotting
    fig = plt.figure(figsize=(8, 4))  # Adjust figure size as needed
    sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", xticklabels=labels_names, yticklabels=labels_names)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")'''

    return cm

def CM_plotter(cm1, cm2, labels_names):
    
    # Plotting
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    # Plot 1st confusion matrix
    ax = axs[0]
    sns.heatmap(cm1, annot=True, fmt="d", cmap="coolwarm", xticklabels=labels_names, yticklabels=labels_names, ax=ax)
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_title("Confusion Matrix for the original data")

    # Plot 2nd confusion matrix
    ax = axs[1]
    sns.heatmap(cm2, annot=True, fmt="d", cmap="coolwarm", xticklabels=labels_names, yticklabels=labels_names, ax=ax)
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_title("Confusion Matrix for PCA-transformed data")

def Dense_Neural_Network(np_features,np_labels,labels_names):
    
    # Split data into training and test datasets:
    X_train, X_test, y_train, y_test = train_test_split(np_features, 
                                                        np_labels, 
                                                        test_size = 0.20, 
                                                        random_state = 42, 
                                                        shuffle = True)

    # Scaling the data
    sc = MinMaxScaler(feature_range = (-1, 1))
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Is the data balanced?
    print(f"Is the data balanced?")
    print(f"Training data samples per class: {np.bincount(y_train.reshape(-1).astype(int))}")
    print(f"Test data samples per class: {np.bincount(y_test.reshape(-1).astype(int))}")


    # Setting up early stoppage
    callback_stp = EarlyStopping(monitor = "val_loss", 
                                patience = 10, 
                                mode = "min", 
                                start_from_epoch = 20, 
                                restore_best_weights = True)

    # Calculation of the penalization weights:
    weight_for_0 = len(y_train)/ (y_train == 0).sum().item()
    weight_for_1 = len(y_train)/ (y_train == 1).sum().item()

    # Make a dictionary with the penalization weights:
    class_weight = {0: weight_for_0, 1: weight_for_1}
    print('Weight for class 0: {:.6f}'.format(weight_for_0))
    print('Weight for class 1: {:.6f}'.format(weight_for_1))

    # Create the neural net model
    model = Sequential()

    # Define the model architecture by adding layers to it:
    model.add(Input(shape = (X_train.shape[1], )))
    #model.add(Dense(units = 512, activation = 'relu'))
    model.add(Dense(units = 256, activation = 'relu'))
    model.add(Dropout(rate = 0.25))
    model.add(Dense(units = 128, activation = 'relu'))
    model.add(Dense(units = 3, activation = 'softmax'))

    # Compile the model, which means setting:
    # a. Optimizer
    # b. Loss function
    # c. Performance metrics
    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    model.summary()

    # Train (fit) the model: train, tentatively, for 300 epochs with early stopping
    model_history = model.fit(X_train, 
                            y_train, 
                            batch_size = 1024, 
                            epochs = 50, 
                            validation_split = 0.1, 
                            class_weight = class_weight, 
                            callbacks = [callback_stp])

    # Display the model topology
    # To display a model, we do this:
    # 1) conda install python-graphviz
    # 2) conda install pydot
    # plot_model(model, show_shapes = True, show_layer_names = True, to_file = 'model.png', dpi = 72)


    # Make a plot of the trainig and validation losses as a function of the number of epochs
    fig, ax = plt.subplots(figsize = (12, 6))
    ax.plot(model_history.history['loss'], c = "blue")
    ax.plot(model_history.history['val_loss'], c = "red")
    ax.set_title('Model Loss')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend(['Train', 'Validation'], loc = 'upper right')
    fig.show()

    # Model performance evaluation: we need to get the model predictions on the test dataset
    y_pred = model.predict(X_test)

    # As the model's outputs are softmax probabilities,
    # we take the predicted state (label) as the one with the one with the max probability
    y_pred = y_pred.argmax(axis = 1)

    # Now we can calculate and print the model performance metrics on the test dataset:
    target_names = labels_names
    #print("above report")
    print(classification_report(y_test, y_pred, target_names=target_names, digits = 5))
    #print("below report")

    # Important!
    # If the classes are unbalanced, then we should use the "balanced accuracy" instead:
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred)}")

    # This is the regular accuracy: it does not take into account potential bias due to the data being unbalanced
    print(f"Normal Accuracy: {accuracy_score(y_test, y_pred)}")

    '''#Evaluación del modelo usando la matriz de confusion
    plt.figure(figsize=(10, 6))
    plot_confusion_matrix(y_test,
                        y_pred,
                        classes=labels_names,
                        normalize=True,
                        title='Water Injection Pump - Class Weights')
    plt.show()'''
    cm = confusion_matrix(y_test, y_pred)
    return cm

def Dense_Neural_Network_for_PCA(x_train_pca, x_test_pca, y_train, y_test, labels_names):

    X_train = x_train_pca
    X_test = x_test_pca

    # Is the data balanced?
    print(f"Is the data balanced?")
    print(f"Training data samples per class: {np.bincount(y_train.reshape(-1).astype(int))}")
    print(f"Test data samples per class: {np.bincount(y_test.reshape(-1).astype(int))}")


    # Setting up early stoppage
    callback_stp = EarlyStopping(monitor = "val_loss", 
                                patience = 10, 
                                mode = "min", 
                                start_from_epoch = 120, 
                                restore_best_weights = True)

    # Calculation of the penalization weights:
    weight_for_0 = len(y_train)/ (y_train == 0).sum().item()
    weight_for_1 = len(y_train)/ (y_train == 1).sum().item()

    # Make a dictionary with the penalization weights:
    class_weight = {0: weight_for_0, 1: weight_for_1}
    print('Weight for class 0: {:.6f}'.format(weight_for_0))
    print('Weight for class 1: {:.6f}'.format(weight_for_1))

    # Create the neural net model
    model = Sequential()

    # Define the model architecture by adding layers to it:
    model.add(Input(shape = (X_train.shape[1], )))
    model.add(Dense(units = 512, activation = 'relu'))
    model.add(Dense(units = 256, activation = 'relu'))
    model.add(Dropout(rate = 0.25))
    model.add(Dense(units = 128, activation = 'relu'))
    model.add(Dense(units = 3, activation = 'softmax'))

    # Compile the model, which means setting:
    # a. Optimizer
    # b. Loss function
    # c. Performance metrics
    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    model.summary()

    # Train (fit) the model: train, tentatively, for 300 epochs with early stopping
    model_history = model.fit(X_train, 
                            y_train, 
                            batch_size = 1024, 
                            epochs = 50, 
                            validation_split = 0.1, 
                            class_weight = class_weight, 
                            callbacks = [callback_stp])

    # Display the model topology
    # To display a model, we do this:
    # 1) conda install python-graphviz
    # 2) conda install pydot
    # plot_model(model, show_shapes = True, show_layer_names = True, to_file = 'model.png', dpi = 72)


    # Make a plot of the trainig and validation losses as a function of the number of epochs
    fig, ax = plt.subplots(figsize = (12, 6))
    ax.plot(model_history.history['loss'], c = "blue")
    ax.plot(model_history.history['val_loss'], c = "red")
    ax.set_title('Model Loss')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend(['Train', 'Validation'], loc = 'upper right')
    fig.show()

    # Model performance evaluation: we need to get the model predictions on the test dataset
    y_pred = model.predict(X_test)

    # As the model's outputs are softmax probabilities,
    # we take the predicted state (label) as the one with the one with the max probability
    y_pred = y_pred.argmax(axis = 1)

    # Now we can calculate and print the model performance metrics on the test dataset:
    target_names = labels_names
    #print("above report")
    print(classification_report(y_test, y_pred, target_names=target_names, digits = 5))
    #print("below report")

    # Important!
    # If the classes are unbalanced, then we should use the "balanced accuracy" instead:
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred)}")

    # This is the regular accuracy: it does not take into account potential bias due to the data being unbalanced
    print(f"Normal Accuracy: {accuracy_score(y_test, y_pred)}")

    '''#Evaluación del modelo usando la matriz de confusion
    plt.figure(figsize=(10, 6))
    plot_confusion_matrix(y_test,
                        y_pred,
                        classes=labels_names,
                        normalize=True,
                        title='Water Injection Pump - Class Weights')
    plt.show()'''
    cm = confusion_matrix(y_test, y_pred)
    return cm

def GridSearch(np_features, np_labels, model):
    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(np_features, np_labels, test_size=0.3, random_state=42, shuffle=True)

    # Scale the features:
    sc = MinMaxScaler(feature_range=(0, 1))
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    # Use HistGradientBoostingClassifier with class_weight='balanced'
    classifier = model
    # Pipeline with MinMaxScaler and HistGradientBoostingClassifier
    pipe_clf = Pipeline([
        ("transformer", MinMaxScaler(feature_range=(0, 1))),
        ("estimator", classifier)
    ])

    # Define the parameter grid for HistGradientBoostingClassifier
    param_grid = {
        'estimator__loss': ['log_loss', 'auto'],  # Loss function for classification
        'estimator__learning_rate': [0.01, 0.1, 0.2, 0.5],  # Learning rate
        'estimator__max_iter': [100, 200, 500],  # Number of boosting iterations
        'estimator__max_depth': [None, 5, 10],  # Max depth of trees
        'estimator__min_samples_leaf': [10, 20, 50],  # Minimum samples per leaf
        'estimator__l2_regularization': [0, 0.1, 0.5]  # L2 regularization strength
    }

    # Grid Search with 5-fold cross-validation for classification
    grid_search = GridSearchCV(pipe_clf, param_grid=param_grid, cv=5, scoring='balanced_accuracy', n_jobs=-1)

    # Perform grid search on the training data
    grid_search.fit(x_train, y_train)

    # Print the best parameters
    print(grid_search.best_params_)
