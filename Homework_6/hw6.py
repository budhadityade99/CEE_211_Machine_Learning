import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from numpy import mean, sqrt, square
from scipy.fftpack import fft, fftfreq
import scipy.io as sio
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from scipy.stats import kurtosis, skew
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

def preprocess(df):

    # dropping the 'unnamed' column
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')
    
    # Next modify the datatypes and set index 
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.set_index('Timestamp')
    
    # Check for missing values:
    nan_count = df.isna().sum()
    nan_pct = df.isna().sum() / len(df)

    # We can fill them:
    df.ffill(axis = 'index', inplace = True)
    
    return df

def plotting_scatter(df):

    # Plotting the scatter plots first
    fig, axes = plt.subplots(nrows=1, ncols=3, sharex='col', figsize=(15, 5))

    # Plot first scatter plot
    axes[0].scatter(df.index, df["LNG_Temperature_Downstream_P114"], s=2, c='red')
    axes[0].set_ylabel("LNG_Temperature_Downstream_P114 [C]", fontsize=10)

    # Plot second scatter plot
    axes[1].scatter(df.index, df["C2_Temperature_Downstream_P106"], s=2, c='green')
    axes[1].set_ylabel("C2_Temperature_Downstream_P106 [C]", fontsize=10)

    # Plot third scatter plot
    axes[2].scatter(df.index, df["C2_Molar_Concentration"], s=2, c='blue')
    axes[2].set_ylabel("C2_Molar_Concentration [molar %]", fontsize=10)

    # Apply tick rotation to all axes
    for ax in axes:
        ax.tick_params(labelrotation=45)

    plt.show()

def plotting_boxplots(df):

    # Creating subplots
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    # Plot first box plot (horizontal)
    axes[0].boxplot(df["LNG_Temperature_Downstream_P114"].dropna(), vert=False, patch_artist=True, boxprops=dict(facecolor='white'))
    axes[0].set_xlabel("LNG_Temperature_Downstream_P114 [C]", fontsize=10)

    # Plot second box plot (horizontal)
    axes[1].boxplot(df["C2_Temperature_Downstream_P106"].dropna(), vert=False, patch_artist=True, boxprops=dict(facecolor='white'))
    axes[1].set_xlabel("C2_Temperature_Downstream_P106 [C]", fontsize=10)

    # Plot third box plot (horizontal)
    axes[2].boxplot(df["C2_Molar_Concentration"].dropna(), vert=False, patch_artist=True, boxprops=dict(facecolor='white'))
    axes[2].set_xlabel("C2_Molar_Concentration [molar %]", fontsize=10)

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()

def train_test_splitting(df):
    # Split DataFrame into features and lables DataFrames:
    df_features = df.iloc[:, 0:2]
    df_labels = df.iloc[:, 2]

    # Convert DataFrames to Numpy arrays:
    np_features = df_features.to_numpy()
    np_labels = df_labels.to_numpy()

    #print(f"Features array: {np_features.shape}")
    #print(f"Labels array: {np_labels.shape}")

    # Split dataset into training set and test set:
    x_train, x_test, y_train, y_test = train_test_split(np_features, 
                                                    np_labels, 
                                                    test_size = 0.3, random_state = 42, 
                                                    shuffle = True)

    # Standarize the features:
    sc = MinMaxScaler(feature_range = (0, 1))
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    # Save splitted data to an NPZ file:
    np.savez("LNG_C2.npz", x_train = x_train, x_test = x_test, y_train = y_train, y_test = y_test)

    return x_train,x_test,y_train,y_test

def linear_SVR(x_train, x_test, y_train, y_test):
    from sklearn.svm import LinearSVR
    from sklearn.metrics import mean_squared_error, r2_score

    # Define hyperparameter search space
    C_values = [0.1, 1, 10]
    epsilon_values = [0, 0.1, 0.3, 0.5, 1.5]
    
    best_model = None
    best_score = 1*10^-6  # Initialize best r^2 with a small value
    best_params = {}

    # Loop over all combinations of hyperparameters
    for C in C_values:
        for epsilon in epsilon_values:
            model = LinearSVR(C=C, epsilon=epsilon, loss="squared_epsilon_insensitive", random_state=42)
            model.fit(x_train, y_train)

            # Predict on the test set
            y_pred = model.predict(x_test)

            # Evaluate using MSE and R²
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # Print intermediate results
            #print(f"C={C}, epsilon={epsilon}, MSE={mse:.4f}, R²={r2:.4f}")

            # Update best model based on lowest MSE
            if r2 > best_score:
                best_score = r2
                best_model = model
                best_y_pred = y_pred
                best_params = {"C": C, "epsilon": epsilon, "MSE": mse, "RMSE": rmse, "R²": r2}

    # Print the best hyperparameters and metrics
    print("\nBest Hyperparameters:")
    print(f"C={best_params['C']}, epsilon={best_params['epsilon']}, Best MSE={best_params['MSE']:.4f}, Best RMSE={best_params['RMSE']:.4f}, Best R²={best_params['R²']:.4f}")

    # Plot Actual vs. Predicted with 45-degree reference line
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, best_y_pred, color="blue", alpha=0.6, label="Predictions")
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--", linewidth=2, label="45-degree line")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs. Predicted Values (Best SVR Model)")
    plt.legend()
    plt.grid(True)
    plt.show()

    return best_model

def nonlinear_SVR(x_train, x_test, y_train, y_test):
    from sklearn.svm import SVR

    # Define hyperparameter search space
    C_values = [0.1, 0.5, 1, 10, 20]
    epsilon_values = [0.1, 0.3, 0.5, 1.0, 1.5]
    gamma_values = ['scale', 'auto']  # 'scale' and 'auto' are adaptive options

    best_model = None
    best_score = 1*10^-6  # Initialize best r^2 with a small value
    best_params = {}

    # Loop over all combinations of hyperparameters
    for C in C_values:
        for epsilon in epsilon_values:
            for gamma in gamma_values:
                model = SVR(kernel="rbf", C=C, epsilon=epsilon, gamma=gamma)
                model.fit(x_train, y_train)

                # Predict on the test set
                y_pred = model.predict(x_test)

                # Evaluate using MSE, RMSE, and R²
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                
                # Print intermediate results
                #print(f"C={C}, epsilon={epsilon}, gamma={gamma}, MSE={mse:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")

                # Update best model based on lowest MSE
                if r2 > best_score:
                    best_score = r2
                    best_model = model
                    best_y_pred = y_pred
                    best_params = {"C": C, "epsilon": epsilon, "gamma": gamma, "MSE": mse, "RMSE": rmse, "R²": r2}

    # Print the best hyperparameters and metrics
    print("\nBest Hyperparameters:")
    print(f"C={best_params['C']}, epsilon={best_params['epsilon']}, gamma={best_params['gamma']}")
    print(f"Best MSE={best_params['MSE']:.4f}, Best RMSE={best_params['RMSE']:.4f}, Best R²={best_params['R²']:.4f}")

    # Plot Actual vs. Predicted with 45-degree reference line
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, best_y_pred, color="blue", alpha=0.6, label="Predictions")
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--", linewidth=2, label="45-degree line")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs. Predicted Values (Best Nonlinear SVR Model)")
    plt.legend()
    plt.grid(True)
    plt.show()

def DT_regressor(x_train, x_test, y_train, y_test):
    from sklearn.tree import DecisionTreeRegressor
    # Define hyperparameter search space
    max_depth_values = [3, 5, 10, 20, 50, None]  # None means the tree is grown until all leaves are pure
    min_samples_split_values = [2, 3, 5, 10, 20]
    min_samples_leaf_values = [1, 2, 5, 10]

    best_model = None
    best_score = float("-inf")  # Initialize best R² with the lowest value
    best_params = {}

    # Loop over all combinations of hyperparameters
    for max_depth in max_depth_values:
        for min_samples_split in min_samples_split_values:
            for min_samples_leaf in min_samples_leaf_values:
                model = DecisionTreeRegressor(
                    max_depth=max_depth, 
                    min_samples_split=min_samples_split, 
                    min_samples_leaf=min_samples_leaf,
                    random_state=42
                )
                model.fit(x_train, y_train)

                # Predict on the test set
                y_pred = model.predict(x_test)

                # Evaluate using MSE, RMSE, and R²
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                
                # Print intermediate results
                #print(f"max_depth={max_depth}, min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}, MSE={mse:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")

                # Update best model based on highest R²
                if r2 > best_score:
                    best_score = r2
                    best_model = model
                    best_y_pred = y_pred
                    best_params = {
                        "max_depth": max_depth,
                        "min_samples_split": min_samples_split,
                        "min_samples_leaf": min_samples_leaf,
                        "MSE": mse,
                        "RMSE": rmse,
                        "R²": r2
                    }

    # Print the best hyperparameters and metrics
    print("\nBest Hyperparameters:")
    print(f"max_depth={best_params['max_depth']}, min_samples_split={best_params['min_samples_split']}, min_samples_leaf={best_params['min_samples_leaf']}")
    print(f"Best MSE={best_params['MSE']:.4f}, Best RMSE={best_params['RMSE']:.4f}, Best R²={best_params['R²']:.4f}")

    # Plot Actual vs. Predicted with 45-degree reference line
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, best_y_pred, color="blue", alpha=0.6, label="Predictions")
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--", linewidth=2, label="45-degree line")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs. Predicted Values (Best Decision Tree Model)")
    plt.legend()
    plt.grid(True)
    plt.show()

    return best_model

def RF_regressor(x_train, x_test, y_train, y_test):
    from sklearn.ensemble import RandomForestRegressor

    # Define hyperparameter search space
    n_estimators_values = [50, 100, 200, 500, 1000]  # Number of trees in the forest
    max_depth_values = [None, 5, 10, 20]  # None allows trees to grow until pure leaves
    min_samples_split_values = [2, 5, 10, 20]

    best_model = None
    best_score = float("-inf")  # Initialize best R² with the lowest value
    best_params = {}

    # Loop over all combinations of hyperparameters
    for n_estimators in n_estimators_values:
        for max_depth in max_depth_values:
            for min_samples_split in min_samples_split_values:
                model = RandomForestRegressor(
                    n_estimators=n_estimators, 
                    max_depth=max_depth, 
                    min_samples_split=min_samples_split,
                    random_state=42,
                    n_jobs=-1  # Use all available cores
                )
                model.fit(x_train, y_train.ravel())  # Ensure y_train is 1D

                # Predict on the test set
                y_pred = model.predict(x_test)

                # Evaluate using MSE, RMSE, and R²
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                
                # Print intermediate results
                #print(f"n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split}, MSE={mse:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")

                # Update best model based on highest R²
                if r2 > best_score:
                    best_score = r2
                    best_model = model
                    best_y_pred = y_pred
                    best_params = {
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "min_samples_split": min_samples_split,
                        "MSE": mse,
                        "RMSE": rmse,
                        "R²": r2
                    }

    # Print the best hyperparameters and metrics
    print("\nBest Hyperparameters:")
    print(f"n_estimators={best_params['n_estimators']}, max_depth={best_params['max_depth']}, min_samples_split={best_params['min_samples_split']}")
    print(f"Best MSE={best_params['MSE']:.4f}, Best RMSE={best_params['RMSE']:.4f}, Best R²={best_params['R²']:.4f}")

    # Plot Actual vs. Predicted with 45-degree reference line
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, best_y_pred, color="blue", alpha=0.6, label="Predictions")
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--", linewidth=2, label="45-degree line")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs. Predicted Values (Best Random Forest Model)")
    plt.legend()
    plt.grid(True)
    plt.show()

    return best_model

def GB_regressor(x_train, x_test, y_train, y_test):

    from sklearn.ensemble import GradientBoostingRegressor

    # Define hyperparameter search space
    learning_rate_values = [0.01, 0.1, 0.2, 0.5]  # Step size shrinkage
    n_estimators_values = [100, 200, 300, 500]  # Number of boosting stages
    max_depth_values = [3, 5, 7, 10]  # Depth of each tree

    best_model = None
    best_score = float("-inf")  # Initialize best R² with the lowest value
    best_params = {}

    # Loop over all combinations of hyperparameters
    for learning_rate in learning_rate_values:
        for n_estimators in n_estimators_values:
            for max_depth in max_depth_values:
                model = GradientBoostingRegressor(
                    learning_rate=learning_rate,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42
                )
                model.fit(x_train, y_train.ravel())  # Ensure y_train is 1D

                # Predict on the test set
                y_pred = model.predict(x_test)

                # Evaluate using MSE, RMSE, and R²
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                
                # Print intermediate results
                #print(f"learning_rate={learning_rate}, n_estimators={n_estimators}, max_depth={max_depth}, MSE={mse:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")

                # Update best model based on highest R²
                if r2 > best_score:
                    best_score = r2
                    best_model = model
                    best_y_pred = y_pred
                    best_params = {
                        "learning_rate": learning_rate,
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "MSE": mse,
                        "RMSE": rmse,
                        "R²": r2
                    }

    # Print the best hyperparameters and metrics
    print("\nBest Hyperparameters:")
    print(f"learning_rate={best_params['learning_rate']}, n_estimators={best_params['n_estimators']}, max_depth={best_params['max_depth']}")
    print(f"Best MSE={best_params['MSE']:.4f}, Best RMSE={best_params['RMSE']:.4f}, Best R²={best_params['R²']:.4f}")

    # Plot Actual vs. Predicted with 45-degree reference line
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, best_y_pred, color="blue", alpha=0.6, label="Predictions")
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--", linewidth=2, label="45-degree line")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs. Predicted Values (Best Gradient Boosting Model)")
    plt.legend()
    plt.grid(True)
    plt.show()

    return best_model

def HGB_regressor(x_train, x_test, y_train, y_test):

    from sklearn.ensemble import HistGradientBoostingRegressor

    # Define hyperparameter search space
    learning_rate_values = [0.01, 0.1, 0.2]  # Step size shrinkage
    max_iter_values = [100, 200, 300, 500]  # Number of boosting iterations
    max_depth_values = [3, 5, 7, None]  # Depth of each tree

    best_model = None
    best_score = float("-inf")  # Initialize best R² with the lowest value
    best_params = {}

    # Loop over all combinations of hyperparameters
    for learning_rate in learning_rate_values:
        for max_iter in max_iter_values:
            for max_depth in max_depth_values:
                model = HistGradientBoostingRegressor(
                    learning_rate=learning_rate,
                    max_iter=max_iter,
                    max_depth=max_depth,
                    random_state=42
                )
                model.fit(x_train, y_train.ravel())  # Ensure y_train is 1D

                # Predict on the test set
                y_pred = model.predict(x_test)

                # Evaluate using MSE, RMSE, and R²
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                
                # Print intermediate results
                #print(f"learning_rate={learning_rate}, max_iter={max_iter}, max_depth={max_depth}, MSE={mse:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")

                # Update best model based on highest R²
                if r2 > best_score:
                    best_score = r2
                    best_model = model
                    best_y_pred = y_pred
                    best_params = {
                        "learning_rate": learning_rate,
                        "max_iter": max_iter,
                        "max_depth": max_depth,
                        "MSE": mse,
                        "RMSE": rmse,
                        "R²": r2
                    }

    # Print the best hyperparameters and metrics
    print("\nBest Hyperparameters:")
    print(f"learning_rate={best_params['learning_rate']}, max_iter={best_params['max_iter']}, max_depth={best_params['max_depth']}")
    print(f"Best MSE={best_params['MSE']:.4f}, Best RMSE={best_params['RMSE']:.4f}, Best R²={best_params['R²']:.4f}")

    # Plot Actual vs. Predicted with 45-degree reference line
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, best_y_pred, color="blue", alpha=0.6, label="Predictions")
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--", linewidth=2, label="45-degree line")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs. Predicted Values (Best Histogram Gradient Boosting Model)")
    plt.legend()
    plt.grid(True)
    plt.show()

    return best_model

def Gridsearch_CV():

    # Load the clean (preprocessed dataset):
    data = np.load("LNG_C2.npz")

    # Retrieve the training and test datasets:
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']

    # Create a pipeline
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import HistGradientBoostingRegressor

    # Create a regressor with a default set of hyper-parameters:
    regressor = HistGradientBoostingRegressor()

    # Create a pipeline:
    # MinMax as the scaler in combination with a Histgradboost regressor
    pipe_reg = Pipeline([
        ("transformer", MinMaxScaler(feature_range = (0, 1))),
        ("estimator", regressor)
    ])

    # Grid-Search:
    from sklearn.model_selection import GridSearchCV

    # Define the parameter grid:
    param_grid = {
        'estimator__learning_rate': [0.01, 0.1, 0.2, 0.5, 1],  # Controls step size shrinkage
        'estimator__max_iter': [100, 200, 300, 500],  # Number of boosting iterations
        'estimator__max_depth': [3, 5, 7, 10, 20],  # Depth of each tree
        'estimator__loss': ['squared_error', 'absolute_error', 'gamma', 'poisson'],  # Loss function
        'estimator__max_features': [0.1, 0.2, 0.5, 1]  # Feature selection per split
    }

    # Initialize Histogram Gradient Boosting model
    #model = HistGradientBoostingRegressor(random_state=42)

    # Set up GridSearchCV with 5-fold cross-validation
    grid_search = GridSearchCV(
        pipe_reg,
        param_grid=param_grid,
        cv=5,  # 5-fold cross-validation
        scoring='r2',  # Use R² as the scoring metric
        n_jobs=-1,  # Use all available CPU cores
        #verbose=2  # Show progress
    )

    # Execute the grid search on the training dataset:
    grid_search.fit(x_train, y_train)

    # Get the grid search best parameters set:
    print(grid_search.best_params_)

    # Print the grid search results:
    #display(pd.DataFrame(grid_search.cv_results_))

def cross_validation():

    # Inport required modules:
    from sklearn.model_selection import cross_validate
    from sklearn.metrics import make_scorer
    from sklearn.metrics import explained_variance_score
    from sklearn.metrics import max_error
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import root_mean_squared_error
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score

    # Load the clean (preprocessed dataset):
    data = np.load("LNG_C2.npz")

    # Retrieve the training and test datasets:
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']

    # Create a pipeline
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import HistGradientBoostingRegressor

    # Set the regression scoring metrics:
    scores_reg = {
        'Explained_Variance': make_scorer(explained_variance_score),
        'Max_Error': make_scorer(max_error),
        'MAE': make_scorer(mean_absolute_error),
        'RMSE': make_scorer(root_mean_squared_error),
        'R2': make_scorer(r2_score)
    }

    # Create a regressor with the best set of hyper-parameters according to the grid search:
    regressor = HistGradientBoostingRegressor(learning_rate=0.01,loss='squared_error',max_depth=10,max_features=0.2,max_iter=500)

    # Create a pipeline: transformer + regressor, in this case:
    pipe_reg = Pipeline([
        ("transformer", MinMaxScaler(feature_range = (0, 1))),
        ("estimator", regressor)
    ])

    # Cross-Validation on the pipeline (with best hyper-parameters) on the training dataset:
    pipe_scores = cross_validate(pipe_reg, x_train, y_train, scoring = scores_reg, return_train_score = True, cv = 5)

    # Get and print the cross-validation scores on the trainig dataset:
    display(pipe_scores)

def final_model(x_train,x_test,y_train,y_test):
    from sklearn.metrics import make_scorer
    from sklearn.metrics import explained_variance_score
    from sklearn.metrics import max_error
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import root_mean_squared_error
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score

    from sklearn.ensemble import HistGradientBoostingRegressor
    # Create and retrain the regressor with the best hyper-parameters (based on grid search) on the entire training dataset:
    model = HistGradientBoostingRegressor(learning_rate=0.01,loss='squared_error',max_depth=10,max_features=0.2,max_iter=500)

    # Scale the data:
    sc = MinMaxScaler(feature_range = (0, 1))
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    # Retrain:
    model.fit(x_train, y_train)

    # Get predictions on the train and test datasets:
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)

    # Print performance metrics on train and test datasets:
    print('\nTRAIN')
    print('RMSE: {:.3f}'.format(np.sqrt(mean_squared_error(y_train, y_pred_train))))
    print('MAX ERROR: {:.3f}'.format(max_error(y_train, y_pred_train)))
    print('R2: {:.3f}'.format(r2_score(y_train, y_pred_train)))

    print('\nTEST')
    print('RMSE: {:.3f}'.format(np.sqrt(mean_squared_error(y_test, y_pred_test))))
    print('MAX ERROR: {:.3f}'.format(max_error(y_test, y_pred_test)))
    print('R2: {:.3f}'.format(r2_score(y_test, y_pred_test)))

    # Plot the observed C5 concentration versus the predicted values:
    plt.scatter(y_pred_test, y_test, marker = 'o', label = 'Test dataset', c = 'green', alpha = 0.5)
    plt.plot([y_pred_test.min(), y_pred_test.max()],[y_test.min(), y_test.max()], color = 'black' )
    plt.ylabel('Observed C5 Concentration [molar %]')
    plt.xlabel('Predicted C5 Concentration [molar %]')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()










