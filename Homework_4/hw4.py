import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from utils import confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def Logistic(features,labels,labels_names,col_num):
    # Performing train-test splitting 
    # Import Load Scikit-Learn package (just the required train_test_split() function):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    # We are going to split the dataset, already cleaned, into training and test datasets
    # Later, when we setup the training of a machine learning model using this data,
    # we will further split the training dataset into training and validation datasets
    x_train_func, x_test_func, y_train_func, y_test_func = train_test_split(features,
                                                                            labels, 
                                                                            test_size = 0.20, 
                                                                            random_state = 42, 
                                                                            shuffle = True)

    # Normalizing the splitted data

    sc = StandardScaler()
    sc.fit(x_train_func)
    x_train_func = sc.fit_transform(x_train_func)
    x_test_func = sc.transform(x_test_func)

    # We create an instance (object) from the sklearn LogisticRegression class
    lr = LogisticRegression(penalty = 'l2', C = 1000, random_state = 42, max_iter = 1000) #, class_weight = "balanced")

    # Fit (train) the model to the training dataset:
    lr.fit(x_train_func, y_train_func[:,col_num])

    # Check performance on the train dataset:
    score = lr.score(x_train_func, y_train_func[:,col_num])
    print(f"Accuracy on train dataset: {score:.2%}")

    from sklearn.metrics import confusion_matrix

    # Get the model's predictions for the test dataset:
    y_pred = lr.predict(x_test_func)

    # Get and print a classification report: performance metrics on the test dataset
    print(classification_report(y_test_func[:,col_num], y_pred, target_names = labels_names, digits = 5))

    # Important! If the dataset is unbalanced, calculate the "balanced accuracy":
    print(f"Balanced Accurancy: {balanced_accuracy_score(y_test_func[:,col_num], y_pred):.5f}")
    print(f"Unbalanced Accurancy: {accuracy_score(y_test_func[:,col_num], y_pred):.5f}")

    # Compute confusion matrix
    cm = confusion_matrix(y_test_func[:,col_num], y_pred)

    from sklearn.inspection import DecisionBoundaryDisplay

    X2d_train = x_train_func[:,:2] # taking slice on the feature dataset for the first two columns

    lr_2d = LogisticRegression(penalty = 'l2', C = 1000, random_state = 42, max_iter = 1000)# class_weight = "balanced")

    # Train logistic regression model
    lr_2d.fit(X2d_train, y_train_func[:,col_num])

    # Check performance on the train dataset:
    score = lr_2d.score(X2d_train, y_train_func[:,col_num])
    #print(f"Accuracy on train dataset: {score:.2%}")

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    # Plot confusion matrix
    ax = axs[0]  # Assign subplot axis
    sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", xticklabels=labels_names, yticklabels=labels_names, ax=ax)
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_title("Confusion Matrix for Logistic Regression")

    # Plot decision boundary
    ax = axs[1]  # Assign subplot axis for second plot
    ax.disp = DecisionBoundaryDisplay.from_estimator(
        lr_2d, X2d_train, response_method="predict", cmap="coolwarm", alpha=0.3, ax=ax
    )
    # Scatter plot of training data in 2D space
    ax.scatter(X2d_train[:, 0], X2d_train[:, 1], c=y_train_func[:,col_num], cmap="coolwarm", marker="o")
    ax.set_xlabel("Iron")
    ax.set_ylabel("Chrome")
    ax.set_title("Decision Boundary for Logistic Regression (2D space)")

    plt.tight_layout()  # Ensures the plots don't overlap
    plt.show()

#------------------------------------------------------------------------------------------------------------------------------

def SVM(features,labels,labels_names,col_num):
    # Performing train-test splitting 
    # Import Load Scikit-Learn package (just the required train_test_split() function):
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC

    # We are going to split the dataset, already cleaned, into training and test datasets
    # Later, when we setup the training of a machine learning model using this data,
    # we will further split the training dataset into training and validation datasets
    x_train_func, x_test_func, y_train_func, y_test_func = train_test_split(features,
                                                                            labels, 
                                                                            test_size = 0.20, 
                                                                            random_state = 42, 
                                                                            shuffle = True)

    # Normalizing the splitted data

    sc = StandardScaler()
    sc.fit(x_train_func)
    x_train_func = sc.fit_transform(x_train_func)
    x_test_func = sc.transform(x_test_func)

    # Create an instance (object) from the sklearn SVM class
    svm = SVC(kernel = 'rbf', C = 100, gamma = 'scale')

    # Fit (train) the model to the training dataset:
    svm.fit(x_train_func, y_train_func[:,col_num])

    # Check performance on the train dataset:
    score = svm.score(x_train_func, y_train_func[:,col_num])
    print(f"Accuracy on train dataset: {score:.2%}")

    # Get the model's predictions for the test dataset:
    y_pred_svm = svm.predict(x_test_func)

    # Get and print a classification report: performance metrics on the test dataset
    print(classification_report(y_test_func[:,col_num], y_pred_svm, target_names = labels_names, digits = 5))

    # Important! If the dataset is unbalanced, calculate the "balanced accuracy":
    print(f"Balanced Accurancy: {balanced_accuracy_score(y_test_func[:,col_num], y_pred_svm):.5f}")
    print(f"Unbalanced Accurancy: {accuracy_score(y_test_func[:,col_num], y_pred_svm):.5f}")

    # Compute confusion matrix
    cm_svm = confusion_matrix(y_test_func[:,col_num], y_pred_svm)

    from sklearn.inspection import DecisionBoundaryDisplay

    X2d_train = x_train_func[:,:2] # Slicing for first 2 feature columns

    # Create a new SVC instance and retrain the SVM on this 2d training data:
    svm_2d = SVC(kernel = 'rbf', C = 100, gamma = 'scale')#, class_weight = 'balanced')

    svm_2d.fit(X2d_train, y_train_func[:,col_num])

    # Check performance on the train dataset:
    score = svm_2d.score(X2d_train, y_train_func[:,col_num])

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    # Plot confusion matrix
    ax = axs[0]  # Assign subplot axis
    sns.heatmap(cm_svm, annot=True, fmt="d", cmap="coolwarm", xticklabels=labels_names, yticklabels=labels_names, ax=ax)
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_title("Confusion Matrix for SVM")

    # Plot decision boundary
    ax = axs[1]  # Assign subplot axis for second plot
    ax.disp = DecisionBoundaryDisplay.from_estimator(
        svm_2d, X2d_train, response_method="predict", cmap="coolwarm", alpha=0.3, ax=ax
    )
    # Scatter plot of training data in 2D space
    ax.scatter(X2d_train[:, 0], X2d_train[:, 1], c=y_train_func[:,col_num], cmap="coolwarm", marker="o")
    ax.set_xlabel("Iron")
    ax.set_ylabel("Chrome")
    ax.set_title("Decision Boundary for SVM (2D space)")

    plt.tight_layout()  # Ensures the plots don't overlap
    plt.show()

#------------------------------------------------------------------------------------------------------------------------------

def DecisionTree(features,labels,labels_names,col_num):
    # Performing train-test splitting 
    # Import Load Scikit-Learn package (just the required train_test_split() function):
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier

    # We are going to split the dataset, already cleaned, into training and test datasets
    # Later, when we setup the training of a machine learning model using this data,
    # we will further split the training dataset into training and validation datasets
    x_train_func, x_test_func, y_train_func, y_test_func = train_test_split(features,
                                                                            labels, 
                                                                            test_size = 0.20, 
                                                                            random_state = 42, 
                                                                            shuffle = True)

    # Normalizing the splitted data
    sc = StandardScaler()
    sc.fit(x_train_func)
    x_train_func = sc.fit_transform(x_train_func)
    x_test_func = sc.transform(x_test_func)

    dt = DecisionTreeClassifier(criterion = 'gini', 
                            max_depth = 20, 
                            min_samples_split = 20, 
                            min_samples_leaf = 5, 
                            min_impurity_decrease = 0, 
                            random_state = 42)

    dt.fit(x_train_func, y_train_func[:,col_num])

    # Check performance on the train dataset:
    score = dt.score(x_train_func, y_train_func[:,col_num])
    print(f"Accuracy on train dataset: {score:.2%}")

    # Get the model's predictions for the test dataset:
    y_pred_dt = dt.predict(x_test_func)

    print(classification_report(y_test_func[:,col_num], y_pred_dt, target_names = labels_names, digits = 5))

    # Important! If the dataset is unbalanced, calculate the "balanced accuracy":
    print(f"Balanced Accurancy: {balanced_accuracy_score(y_test_func[:,col_num], y_pred_dt):.5f}")
    print(f"Unbalanced Accurancy: {accuracy_score(y_test_func[:,col_num], y_pred_dt):.5f}")

    # Compute confusion matrix
    cm_dt = confusion_matrix(y_test_func[:,col_num], y_pred_dt)

    from sklearn.inspection import DecisionBoundaryDisplay

    # Slice the training dataset so we get just the two columns we want:
    X2d_train = x_train_func[:, :2]

    # Create a new Decision Tree instance and retrain it on this 2d training data:
    dt_2d = DecisionTreeClassifier(criterion = 'gini', 
                            max_depth = 20, 
                            min_samples_split = 20, 
                            min_samples_leaf = 5, 
                            min_impurity_decrease = 0, 
                            random_state = 42)

    dt_2d.fit(X2d_train, y_train_func[:,col_num])

    # Check performance on the train dataset:
    score = dt_2d.score(X2d_train, y_train_func[:,col_num])

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    # Plot confusion matrix
    ax = axs[0]  # Assign subplot axis
    sns.heatmap(cm_dt, annot=True, fmt="d", cmap="coolwarm", xticklabels=labels_names, yticklabels=labels_names, ax=ax)
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_title("Confusion Matrix for Decision Tree")

    # Plot decision boundary
    ax = axs[1]  # Assign subplot axis for second plot
    ax.disp = DecisionBoundaryDisplay.from_estimator(
        dt_2d, X2d_train, response_method="predict", cmap="coolwarm", alpha=0.3, ax=ax
    )
    # Scatter plot of training data in 2D space
    ax.scatter(X2d_train[:, 0], X2d_train[:, 1], c=y_train_func[:,col_num], cmap="coolwarm", marker="o")
    ax.set_xlabel("Iron")
    ax.set_ylabel("Chrome")
    ax.set_title("Decision Boundary for Decision Tree (2D space)")

    plt.tight_layout()  # Ensures the plots don't overlap
    plt.show()

#---------------------------------------------------------------------------------------------------------------------------------

def RandomForest(features,labels,labels_names,col_num):
    # Performing train-test splitting 
    # Import Load Scikit-Learn package (just the required train_test_split() function):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    # We are going to split the dataset, already cleaned, into training and test datasets
    # Later, when we setup the training of a machine learning model using this data,
    # we will further split the training dataset into training and validation datasets
    x_train_func, x_test_func, y_train_func, y_test_func = train_test_split(features,
                                                                            labels, 
                                                                            test_size = 0.20, 
                                                                            random_state = 42, 
                                                                            shuffle = True)

    # Normalizing the splitted data
    sc = StandardScaler()
    sc.fit(x_train_func)
    x_train_func = sc.fit_transform(x_train_func)
    x_test_func = sc.transform(x_test_func)

    # Create an instance (object) from the sklearn RandomForestClassifier class:
    rf = RandomForestClassifier(criterion = 'gini', 
                                max_depth = None, 
                                min_samples_split = 2, 
                                min_samples_leaf = 1, 
                                min_impurity_decrease = 0, 
                                random_state = 42, 
                                n_estimators = 100, 
                                bootstrap = True,
                                oob_score = False
                                )

    rf.fit(x_train_func, y_train_func[:,col_num])

    # Check performance on the train dataset:
    score = rf.score(x_train_func, y_train_func[:,col_num])
    print(f"Accuracy on train dataset: {score:.2%}")

    # Get the model's predictions for the test dataset:
    y_pred_rf = rf.predict(x_test_func)

    # Get and print a classification report: performance metrics on the test dataset
    print(classification_report(y_test_func[:,col_num], y_pred_rf, target_names = labels_names, digits = 5))

    # Important! If the dataset is unbalanced, calculate the "balanced accuracy":
    print(f"Balanced Accurancy: {balanced_accuracy_score(y_test_func[:,col_num], y_pred_rf):.5f}")
    print(f"Unbalanced Accurancy: {accuracy_score(y_test_func[:,col_num], y_pred_rf):.5f}")

    # Compute confusion matrix
    cm_rf = confusion_matrix(y_test_func[:,col_num], y_pred_rf)

    from sklearn.inspection import DecisionBoundaryDisplay

    # Slice the training dataset so we get just the two columns we want:
    X2d_train = x_train_func[:, :2]

    # Create a new Random Forest instance and retrain it on this 2d training data:
    rf_2d = RandomForestClassifier(criterion = 'gini', 
                            max_depth = None, 
                                min_samples_split = 2, 
                                min_samples_leaf = 1, 
                                min_impurity_decrease = 0, 
                                random_state = 42, 
                                n_estimators = 100, 
                                bootstrap = True,
                                oob_score = False
                            )

    rf_2d.fit(X2d_train, y_train_func[:,col_num])

    # Check performance on the train dataset:
    score = rf_2d.score(X2d_train, y_train_func[:,col_num])

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    # Plot confusion matrix
    ax = axs[0]  # Assign subplot axis
    sns.heatmap(cm_rf, annot=True, fmt="d", cmap="coolwarm", xticklabels=labels_names, yticklabels=labels_names, ax=ax)
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_title("Confusion Matrix for Random Forest")

    # Plot decision boundary
    ax = axs[1]  # Assign subplot axis for second plot
    ax.disp = DecisionBoundaryDisplay.from_estimator(
        rf_2d, X2d_train, response_method="predict", cmap="coolwarm", alpha=0.3, ax=ax
    )
    # Scatter plot of training data in 2D space
    ax.scatter(X2d_train[:, 0], X2d_train[:, 1], c=y_train_func[:,col_num], cmap="coolwarm", marker="o")
    ax.set_xlabel("Iron")
    ax.set_ylabel("Chrome")
    ax.set_title("Decision Boundary for Random Forest (2D space)")

    plt.tight_layout()  # Ensures the plots don't overlap
    plt.show()

#----------------------------------------------------------------------------------------------------------------------------------

def GradientBoosting(features,labels,labels_names,col_num):

    # Performing train-test splitting 
    # Import Load Scikit-Learn package (just the required train_test_split() function):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingClassifier

    # We are going to split the dataset, already cleaned, into training and test datasets
    # Later, when we setup the training of a machine learning model using this data,
    # we will further split the training dataset into training and validation datasets
    x_train_func, x_test_func, y_train_func, y_test_func = train_test_split(features,
                                                                            labels, 
                                                                            test_size = 0.20, 
                                                                            random_state = 42, 
                                                                            shuffle = True)

    # Normalizing the splitted data
    sc = StandardScaler()
    sc.fit(x_train_func)
    x_train_func = sc.fit_transform(x_train_func)
    x_test_func = sc.transform(x_test_func)

    # Create and parameterized an instance:
    grad_boost = GradientBoostingClassifier(
        max_depth = 20,
        n_estimators = 50,
        learning_rate = 0.5,
        random_state = 42
    )

    # Train the gradient boosting model:
    grad_boost.fit(x_train_func, y_train_func[:,col_num])

    # Check performance on the train dataset:
    score = grad_boost.score(x_train_func, y_train_func[:,col_num])
    print(f"Accuracy on train dataset: {score:.2%}")

    # Evaluate the model on the test dataset:
    # Get the model's predictions for the test dataset:
    y_pred_gb = grad_boost.predict(x_test_func)

    # Get and print a classification report: performance metrics on the test dataset
    print(classification_report(y_test_func[:,col_num], y_pred_gb, target_names = labels_names, digits = 5))

    # Important! If the dataset is unbalanced, calculate the "balanced accuracy":
    print(f"Balanced Accurancy: {balanced_accuracy_score(y_test_func[:,col_num], y_pred_gb):.5f}")
    print(f"Unbalanced Accurancy: {accuracy_score(y_test_func[:,col_num], y_pred_gb):.5f}")

    # Compute confusion matrix
    cm_gb = confusion_matrix(y_test_func[:,col_num], y_pred_gb)   

    from sklearn.inspection import DecisionBoundaryDisplay

    X2d_train = x_train_func[:, :2]

    # Create a new Random Forest instance and retrain it on this 2d training data:
    gb_2d = GradientBoostingClassifier(
        max_depth = 20,
        n_estimators = 50,
        learning_rate = 0.5,
        random_state = 42
    )

    gb_2d.fit(X2d_train, y_train_func[:,col_num])

    # Check performance on the train dataset:
    score = gb_2d.score(X2d_train, y_train_func[:,col_num])

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    # Plot confusion matrix
    ax = axs[0]  # Assign subplot axis
    sns.heatmap(cm_gb, annot=True, fmt="d", cmap="coolwarm", xticklabels=labels_names, yticklabels=labels_names, ax=ax)
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_title("Confusion Matrix for Gradient Boosting")

    # Plot decision boundary
    ax = axs[1]  # Assign subplot axis for second plot
    ax.disp = DecisionBoundaryDisplay.from_estimator(
        gb_2d, X2d_train, response_method="predict", cmap="coolwarm", alpha=0.3, ax=ax
    )
    # Scatter plot of training data in 2D space
    ax.scatter(X2d_train[:, 0], X2d_train[:, 1], c=y_train_func[:,col_num], cmap="coolwarm", marker="o")
    ax.set_xlabel("Iron")
    ax.set_ylabel("Chrome")
    ax.set_title("Decision Boundary for Gradient Boosting (2D space)")

    plt.tight_layout()  # Ensures the plots don't overlap
    plt.show()

#----------------------------------------------------------------------------------------------------------------------

def Feature_score(features,labels):

    # Performing train-test splitting 
    # Import Load Scikit-Learn package (just the required train_test_split() function):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    # We are going to split the dataset, already cleaned, into training and test datasets
    # Later, when we setup the training of a machine learning model using this data,
    # we will further split the training dataset into training and validation datasets
    x_train_func, x_test_func, y_train_func, y_test_func = train_test_split(features,
                                                                            labels, 
                                                                            test_size = 0.20, 
                                                                            random_state = 42, 
                                                                            shuffle = True)

    # Normalizing the splitted data
    sc = StandardScaler()
    sc.fit(x_train_func)
    x_train_func = sc.fit_transform(x_train_func)
    x_test_func = sc.transform(x_test_func)

    rf_feature = RandomForestClassifier(criterion = 'gini', 
                                max_depth = None, 
                                min_samples_split = 2, 
                                min_samples_leaf = 1, 
                                min_impurity_decrease = 0, 
                                random_state = 42, 
                                n_estimators = 100, 
                                bootstrap = True,
                                oob_score = False
                            )
    rf_feature.fit(x_train_func, y_train_func[:,1])
    print("Oil analysis Features Importances:\n")
    for importance_score, feature_name in zip(rf_feature.feature_importances_, pd.DataFrame(features).columns):
        print(f"{feature_name} = {importance_score:.2%}")    

def percent_alarms(df):

    # And we get the labels (health states) frequencies from 'Alarms' column:
    normal_freq = df['Alarms'].value_counts().get(0)
    warning_freq = df['Alarms'].value_counts().get(1)
    caution_freq = df['Alarms'].value_counts().get(2)

    # Calculate proportions and print them: we have a really unbalanced dataset!
    print(f"Normal state proportion: {normal_freq / (normal_freq + warning_freq + caution_freq):.2%}")
    print(f"warning state proportion: {warning_freq / (normal_freq + warning_freq + caution_freq):.2%}")
    print(f"caution state proportion: {caution_freq / (normal_freq + warning_freq + caution_freq):.2%}")
        
def percent_diagnostics(df):

    # And we get the labels (health states) frequencies from 'Alarms' column:
    normal_freq = df['Diagnostics'].value_counts().get(0)
    comp_wear_freq = df['Diagnostics'].value_counts().get(1)
    silica_ISO_freq = df['Diagnostics'].value_counts().get(2)
    oil_cont_freq = df['Diagnostics'].value_counts().get(3)
    water_cont_freq = df['Diagnostics'].value_counts().get(4)
    silica_water_freq = df['Diagnostics'].value_counts().get(5)

    # Calculate proportions and print them: we have a really unbalanced dataset!
    print(f"Normal state proportion: {normal_freq / (normal_freq + comp_wear_freq + silica_ISO_freq + oil_cont_freq + water_cont_freq + silica_water_freq):.2%}")
    print(f"comp_wear state proportion: {comp_wear_freq / (normal_freq + comp_wear_freq + silica_ISO_freq + oil_cont_freq + water_cont_freq + silica_water_freq):.2%}")
    print(f"silica_ISO state proportion: {silica_ISO_freq / (normal_freq + comp_wear_freq + silica_ISO_freq + oil_cont_freq + water_cont_freq + silica_water_freq):.2%}")
    print(f"oil_cont state proportion: {oil_cont_freq / (normal_freq + comp_wear_freq + silica_ISO_freq + oil_cont_freq + water_cont_freq + silica_water_freq):.2%}")
    print(f"water_cont state proportion: {water_cont_freq / (normal_freq + comp_wear_freq + silica_ISO_freq + oil_cont_freq + water_cont_freq + silica_water_freq):.2%}")
    print(f"silica_water state proportion: {silica_water_freq / (normal_freq + comp_wear_freq + silica_ISO_freq + oil_cont_freq + water_cont_freq + silica_water_freq):.2%}")