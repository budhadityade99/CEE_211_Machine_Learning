
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
from sklearn.metrics import confusion_matrix
from scipy.stats import kurtosis, skew
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

def prelim_data():

    # Reading the acceleration data for each health state
    df_H = pd.read_csv('H-C-1_Healthy_N5.csv', header = 'infer')
    df_I = pd.read_csv('I-C-1_Inner_Race_N5.csv', header = 'infer')
    df_O = pd.read_csv('O-C-1_Outer_Race_N5.csv', header = 'infer')
    df_B = pd.read_csv('B-C-1_Ball_N5.csv', header = 'infer')
    df_C = pd.read_csv('C-C-1_Combined_N5.csv', header = 'infer')

    # Determine the time vector:
    Fs = 200000 # Sampling rate
    dt = 1 / Fs # Delta time
    N = len(df_H)
    t = np.linspace(0, dt * (N - 1), N) # creating time vector

    accel_stacked = pd.concat([df_H, df_I, df_O, df_B, df_C], axis=0)

    # Reset the index if needed
    accel_stacked.reset_index(drop=True, inplace=True)

    # Now creating vector for the health state labels
    # Label Assignemnt-->   Healthy:0         Inner:1          Outer:2            Ball:3          Combined:4
    temp0 = pd.DataFrame({'labels': [0] * len(df_H)})
    temp1 = pd.DataFrame({'labels': [1] * len(df_I)})
    temp2 = pd.DataFrame({'labels': [2] * len(df_O)})
    temp3 = pd.DataFrame({'labels': [3] * len(df_B)})
    temp4 = pd.DataFrame({'labels': [4] * len(df_C)})
    labels_stacked=pd.concat([temp0,temp1,temp2,temp3,temp4], axis=0) # Stacking them along the rows i.e. vertically

    # Reset the index if needed
    labels_stacked.reset_index(drop=True, inplace=True)

    df_data=pd.concat([accel_stacked,labels_stacked], axis=1) # combining both acceleration and health state labels side by side

    features=np.asarray(df_data.iloc[:,0]).reshape(-1, 1) # separating features and converting into numpy array
    labels=np.asarray(df_data.iloc[:,1]).reshape(-1, 1) # separating labels and converting into numpy array

    #Splitting the data 70/30
    from sklearn.model_selection import train_test_split
    x_train_func, x_test_func, y_train_func, y_test_func = train_test_split(features, labels, test_size = 0.30, random_state = 42, shuffle = True)

    # Data visualization: let us do some line plots of the raw acceleration data
    fig, axes = plt.subplots(nrows = 2, ncols = 3, figsize = (16, 10))

    # So we access an Axes object by indexing the array:
    # Get the first Axes from the left
    ax = axes[0,0]
    ax.plot(t, df_H)
    ax.set_title("Healthy", fontsize = 14)

    # Here we make use of a Python Raw String (indicated by the r before defining the string)
    # so that we can use Latex code inside the string, indicated by the $ symbol,
    # which gets rendered by Matplotlib:
    ax.set_ylabel(r"Acceleration $[m/s^2]$", fontsize = 12)
    ax.set_xlabel(r"Time $[s]$", fontsize = 12)
    ax.set_xlim(xmin = 0, xmax = 10)
    #ax.set_ylim(ymin = -0.4, ymax = 0.4)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 8)

    # Get the third Axes from the left:
    ax = axes[0,1]
    ax.plot(t, df_I)
    ax.set_title("Inner Race Damage", fontsize = 14)
    ax.set_ylabel(r"Acceleration $[m/s^2]$", fontsize = 12)
    ax.set_xlabel(r"Time $[s]$", fontsize = 12)
    ax.set_xlim(xmin = 0, xmax = 10)
    ax.set_ylim(ymin = -1.5, ymax = 1.5)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 8)


    # Get the second Axes from the left:
    ax = axes[0,2]
    ax.plot(t, df_O)
    ax.set_title("Outer Race Damage", fontsize = 14)
    ax.set_ylabel(r"Acceleration $[m/s^2]$", fontsize = 12)
    ax.set_xlabel(r"Time $[s]$", fontsize = 12)
    ax.set_xlim(xmin = 0, xmax = 10)
    #ax.set_ylim(ymin = -0.4, ymax = 0.4)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 8)


    # Get the second Axes from the left:
    ax = axes[1,0]
    ax.plot(t, df_B)
    ax.set_title("Ball Element damage", fontsize = 14)
    ax.set_ylabel(r"Acceleration $[m/s^2]$", fontsize = 12)
    ax.set_xlabel(r"Time $[s]$", fontsize = 12)
    ax.set_xlim(xmin = 0, xmax = 10)
    #ax.set_ylim(ymin = -0.4, ymax = 0.4)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 8)

    # Get the second Axes from the left:
    ax = axes[1,1]
    ax.plot(t, df_C)
    ax.set_title("Multiple Damage type", fontsize = 14)
    ax.set_ylabel(r"Acceleration $[m/s^2]$", fontsize = 12)
    ax.set_xlabel(r"Time $[s]$", fontsize = 12)
    ax.set_xlim(xmin = 0, xmax = 10)
    #ax.set_ylim(ymin = -1, ymax = 1)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 8)

    ax = axes[1,2]
    ax.set_visible(False)

    return x_train_func, x_test_func, y_train_func, y_test_func

def ML_model_raw_data(x_train_func, x_test_func, y_train_func, y_test_func):
    
    from sklearn.tree import DecisionTreeClassifier
    from itertools import product
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix

    labels_names =['Normal/Healthy','Inner Race damage','Outer Race damage','Ball element damage','Combined damage']
    
    # Training a Decision Tree
    dt = DecisionTreeClassifier(criterion = 'gini', max_depth = 100, min_samples_split = 2, min_samples_leaf = 1, min_impurity_decrease = 0, random_state = 42)
    dt.fit(x_train_func, y_train_func)

    # Check performance on the train dataset:
    score = dt.score(x_train_func, y_train_func)
    print(f"Accuracy on train dataset: {score:.2%}")

    # Get the model's predictions for the test dataset:
    y_pred = dt.predict(x_test_func)

    # Get and print a classification report: performance metrics on the test dataset
    print(classification_report(y_test_func, y_pred, target_names = labels_names, digits = 5))

    # Important! If the dataset is unbalanced, calculate the "balanced accuracy":
    print(f"Balanced Accurancy: {balanced_accuracy_score(y_test_func, y_pred):.5f}")
    print(f"Unbalanced Accurancy: {accuracy_score(y_test_func, y_pred):.5f}")

    # Compute confusion matrix
    cm = confusion_matrix(y_test_func, y_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", xticklabels=labels_names, yticklabels=labels_names)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()

def time_domain_features():

    # Reading the acceleration data for each health state
    df_H = pd.read_csv('H-C-1_Healthy_N5.csv', header = 'infer')
    df_I = pd.read_csv('I-C-1_Inner_Race_N5.csv', header = 'infer')
    df_O = pd.read_csv('O-C-1_Outer_Race_N5.csv', header = 'infer')
    df_B = pd.read_csv('B-C-1_Ball_N5.csv', header = 'infer')
    df_C = pd.read_csv('C-C-1_Combined_N5.csv', header = 'infer')

    # Define the windows parameters:
    # Length of each window:
    L = 20000

    # Overlap between windows:
    l = 19600

    # Number of data points:
    N = len(df_H)

    # Number of windows (segments):
    Nt = math.floor((N - l) / (L - l))

    # Initialize arrays for the time domain features:
    P_H = np.zeros((Nt, 8))
    P_I = np.zeros((Nt, 8))
    P_O = np.zeros((Nt, 8))
    P_B = np.zeros((Nt, 8))
    P_C = np.zeros((Nt, 8))


    # Obtain time domain features for the Normal, Outer race fault, and Inner race fault states:
    P_H = features_time_domain(df_H, Nt, L, l)
    P_I = features_time_domain(df_I, Nt, L, l)
    P_O = features_time_domain(df_O, Nt, L, l)
    P_B = features_time_domain(df_B, Nt, L, l)
    P_C = features_time_domain(df_C, Nt, L, l)

    fig, axes = plt.subplots(nrows = 4, ncols = 2, sharex = 'col', figsize = (18, 18))

    # Plotting the time domain features for ball bearing health states  
    # Setup the plots per axes:
    ax = axes[0][0]
    ax.plot(P_H[:, 0])
    ax.plot(P_O[:, 0])
    ax.plot(P_I[:, 0])
    ax.plot(P_B[:, 0])
    ax.plot(P_C[:, 0])
    ax.tick_params(axis = 'both', which = 'major', labelsize = 10)
    ax.legend(['Normal','Outer Race Damage','Inner Race Damage', 'Ball Damage', 'Combined Damage'], fontsize = 10)
    ax.set(xlabel = 'Interval (Window)')
    ax.set_title('RMS', fontsize = 14)

    ax = axes[0][1]
    ax.plot(P_H[:, 1])
    ax.plot(P_O[:, 1])
    ax.plot(P_I[:, 1])
    ax.plot(P_B[:, 1])
    ax.plot(P_C[:, 1])
    ax.tick_params(axis = 'both', which = 'major', labelsize = 10)
    ax.legend(['Normal','Outer Race Damage','Inner Race Damage', 'Ball Damage', 'Combined Damage'], fontsize = 10)
    ax.set(xlabel = 'Interval (Window)')
    ax.set_title('Peak Value', fontsize = 14)

    ax = axes[1][0]
    ax.plot(P_H[:, 2])
    ax.plot(P_O[:, 2])
    ax.plot(P_I[:, 2])
    ax.plot(P_B[:, 2])
    ax.plot(P_C[:, 2])
    ax.tick_params(axis = 'both', which = 'major', labelsize = 10)
    ax.legend(['Normal','Outer Race Damage','Inner Race Damage', 'Ball Damage', 'Combined Damage'], fontsize = 10)
    ax.set(xlabel = 'Interval (Window)')
    ax.set_title('Peak to Peak', fontsize = 14)

    ax = axes[1][1]
    ax.plot(P_H[:, 3])
    ax.plot(P_O[:, 3])
    ax.plot(P_I[:, 3])
    ax.plot(P_B[:, 3])
    ax.plot(P_C[:, 3])
    ax.tick_params(axis = 'both', which = 'major', labelsize = 10)
    ax.legend(['Normal','Outer Race Damage','Inner Race Damage', 'Ball Damage', 'Combined Damage'], fontsize = 10)
    ax.set(xlabel = 'Interval (Window)')
    ax.set_title('Crest Factor', fontsize = 14)

    ax = axes[2][0]
    ax.plot(P_H[:, 4])
    ax.plot(P_O[:, 4])
    ax.plot(P_I[:, 4])
    ax.plot(P_B[:, 4])
    ax.plot(P_C[:, 4])
    ax.tick_params(axis = 'both', which = 'major', labelsize = 10)
    ax.legend(['Normal','Outer Race Damage','Inner Race Damage', 'Ball Damage', 'Combined Damage'], fontsize = 10)
    ax.set(xlabel = 'Interval (Window)')
    ax.set_title('Mean', fontsize = 14)

    ax = axes[2][1]
    ax.plot(P_H[:, 5])
    ax.plot(P_O[:, 5])
    ax.plot(P_I[:, 5])
    ax.plot(P_B[:, 5])
    ax.plot(P_C[:, 5])
    ax.tick_params(axis = 'both', which = 'major', labelsize = 10)
    ax.legend(['Normal','Outer Race Damage','Inner Race Damage', 'Ball Damage', 'Combined Damage'], fontsize = 10)
    ax.set(xlabel = 'Interval (Window)')
    ax.set_title('Variance', fontsize = 14)

    ax = axes[3][0]
    ax.plot(P_H[:, 6])
    ax.plot(P_O[:, 6])
    ax.plot(P_I[:, 6])
    ax.plot(P_B[:, 6])
    ax.plot(P_C[:, 6])
    ax.tick_params(axis = 'both', which = 'major', labelsize = 10)
    ax.legend(['Normal','Outer Race Damage','Inner Race Damage', 'Ball Damage', 'Combined Damage'], fontsize = 10)
    ax.set(xlabel = 'Interval (Window)')
    ax.set_title('Skewness', fontsize = 14)

    ax = axes[3][1]
    ax.plot(P_H[:, 7])
    ax.plot(P_O[:, 7])
    ax.plot(P_I[:, 7])
    ax.plot(P_B[:, 7])
    ax.plot(P_C[:, 7])
    ax.tick_params(axis = 'both', which = 'major', labelsize = 10)
    ax.legend(['Normal','Outer Race Damage','Inner Race Damage', 'Ball Damage', 'Combined Damage'], fontsize = 10)
    ax.set(xlabel = 'Interval (Window)')
    ax.set_title('Kurtosis', fontsize = 14)

    fig.show()

    return P_H, P_I, P_O, P_B, P_C


def features_time_domain(data, Nt, L, l):
    
    features = np.zeros((Nt,8))

    for i in range(1, Nt + 1):
        start = (i - 1) * L - (i - 1) * l + 1
        end = i * L - (i - 1) * l

        features[i - 1, 0] = sqrt(mean(square(data[start:end]))) # RMS
        features[i - 1, 1] = np.amax(data[start:end]) # Peak
        features[i - 1, 2] = np.amax(data[start:end]) - np.amin(data[start:end]) # Peak-Peak
        features[i - 1, 3] = features[i - 1, 1]/features[i - 1, 0] # Crest
        features[i - 1, 4] = np.mean(data[start:end]) # Mean
        features[i - 1, 5] = np.var(data[start:end]) # Var
        features[i - 1, 6] = skew(data[start:end])[0] # Skewness
        features[i - 1, 7] = kurtosis(data[start:end])[0] # Kurtosis

    return features

def frequency_domain_features():

    # Reading the acceleration data for each health state
    df_H = pd.read_csv('H-C-1_Healthy_N5.csv', header = 'infer')
    df_I = pd.read_csv('I-C-1_Inner_Race_N5.csv', header = 'infer')
    df_O = pd.read_csv('O-C-1_Outer_Race_N5.csv', header = 'infer')
    df_B = pd.read_csv('B-C-1_Ball_N5.csv', header = 'infer')
    df_C = pd.read_csv('C-C-1_Combined_N5.csv', header = 'infer')

    # Converting to numpy arrays
    df_H_numpy = df_H.to_numpy()
    df_I_numpy = df_I.to_numpy()
    df_O_numpy = df_O.to_numpy()
    df_B_numpy = df_B.to_numpy()
    df_C_numpy = df_C.to_numpy()

    # Define the windows parameters:
    # Length of each window:
    L = 20000

    # Overlap between windows:
    l = 19600

    # Number of data points:
    N = len(df_H_numpy)

    # Number of windows (segments):
    Nt = math.floor((N - l) / (L - l))

    # Number of bands (octaves in this case):
    nb = 8

    # Initialize arrays for the frequency domain features:
    E_H = np.zeros((Nt, nb))
    E_I = np.zeros((Nt, nb))
    E_O = np.zeros((Nt, nb))
    E_B = np.zeros((Nt, nb))
    E_C = np.zeros((Nt, nb))


    # Obtain feature domain features for the Normal, Outer race fault, and Inner race fault states:
    E_H = features_frequency_domain(df_H_numpy, Nt, L, l, nb)
    E_I = features_frequency_domain(df_I_numpy, Nt, L, l, nb)
    E_O = features_frequency_domain(df_O_numpy, Nt, L, l, nb)
    E_B = features_frequency_domain(df_B_numpy, Nt, L, l, nb)
    E_C = features_frequency_domain(df_C_numpy, Nt, L, l, nb)

    # Plot the frequency domain features extracted from the raw acceleration data
    fig, axes = plt.subplots(nrows = 4, ncols = 2, sharex = 'col', figsize = (18, 18))

    # Setup the plots per axes:
    ax = axes[0][0]
    ax.plot(E_H[:, 0])
    ax.plot(E_O[:, 0])
    ax.plot(E_I[:, 0])
    ax.plot(E_B[:, 0])
    ax.plot(E_C[:, 0])
    ax.tick_params(axis = 'both', which = 'major', labelsize = 10)
    ax.legend(['Normal','Outer Race Damage','Inner Race Damage', 'Ball Damage', 'Combined Damage'], fontsize = 10)
    ax.set_title('Band 1', fontsize = 14)

    ax = axes[0][1]
    ax.plot(E_H[:, 1])
    ax.plot(E_O[:, 1])
    ax.plot(E_I[:, 1])
    ax.plot(E_B[:, 1])
    ax.plot(E_C[:, 1])
    ax.tick_params(axis = 'both', which = 'major', labelsize = 10)
    ax.legend(['Normal','Outer Race Damage','Inner Race Damage', 'Ball Damage', 'Combined Damage'], fontsize = 10)
    ax.set_title('Band 2', fontsize = 14)

    ax = axes[1][0]
    ax.plot(E_H[:, 2])
    ax.plot(E_O[:, 2])
    ax.plot(E_I[:, 2])
    ax.plot(E_B[:, 2])
    ax.plot(E_C[:, 2])
    ax.tick_params(axis = 'both', which = 'major', labelsize = 10)
    ax.legend(['Normal','Outer Race Damage','Inner Race Damage', 'Ball Damage', 'Combined Damage'], fontsize = 10)
    ax.set(xlabel = 'Interval (Window)')
    ax.set_title('Band 3', fontsize = 14)

    ax = axes[1][1]
    ax.plot(E_H[:, 3])
    ax.plot(E_O[:, 3])
    ax.plot(E_I[:, 3])
    ax.plot(E_B[:, 3])
    ax.plot(E_C[:, 3])
    ax.tick_params(axis = 'both', which = 'major', labelsize = 10)
    ax.legend(['Normal','Outer Race Damage','Inner Race Damage', 'Ball Damage', 'Combined Damage'], fontsize = 10)
    ax.set(xlabel = 'Interval (Window)')
    ax.set_title('Band 4', fontsize = 14)

    ax = axes[2][0]
    ax.plot(E_H[:, 4])
    ax.plot(E_O[:, 4])
    ax.plot(E_I[:, 4])
    ax.plot(E_B[:, 4])
    ax.plot(E_C[:, 4])
    ax.tick_params(axis = 'both', which = 'major', labelsize = 10)
    ax.legend(['Normal','Outer Race Damage','Inner Race Damage', 'Ball Damage', 'Combined Damage'], fontsize = 10)
    ax.set(xlabel = 'Interval (Window)')
    ax.set_title('Band 5', fontsize = 14)

    ax = axes[2][1]
    ax.plot(E_H[:, 5])
    ax.plot(E_O[:, 5])
    ax.plot(E_I[:, 5])
    ax.plot(E_B[:, 5])
    ax.plot(E_C[:, 5])
    ax.tick_params(axis = 'both', which = 'major', labelsize = 10)
    ax.legend(['Normal','Outer Race Damage','Inner Race Damage', 'Ball Damage', 'Combined Damage'], fontsize = 10)
    ax.set(xlabel = 'Interval (Window)')
    ax.set_title('Band 6', fontsize = 14)

    ax = axes[3][0]
    ax.plot(E_H[:, 6])
    ax.plot(E_O[:, 6])
    ax.plot(E_I[:, 6])
    ax.plot(E_B[:, 6])
    ax.plot(E_C[:, 6])
    ax.tick_params(axis = 'both', which = 'major', labelsize = 10)
    ax.legend(['Normal','Outer Race Damage','Inner Race Damage', 'Ball Damage', 'Combined Damage'], fontsize = 10)
    ax.set(xlabel = 'Interval (Window)')
    ax.set_title('Band 7', fontsize = 14)

    ax = axes[3][1]
    ax.plot(E_H[:, 7])
    ax.plot(E_O[:, 7])
    ax.plot(E_I[:, 7])
    ax.plot(E_B[:, 7])
    ax.plot(E_C[:, 7])
    ax.tick_params(axis = 'both', which = 'major', labelsize = 10)
    ax.legend(['Normal','Outer Race Damage','Inner Race Damage', 'Ball Damage', 'Combined Damage'], fontsize = 10)
    ax.set(xlabel = 'Interval (Window)')
    ax.set_title('Band 8', fontsize = 14)

    fig.show()

    return E_H, E_I, E_O, E_B, E_C

def features_frequency_domain(data, Nt, L, l, nb):
    
    features = np.zeros((Nt, nb))

    for i in range(1, Nt + 1):
        start = (i - 1) * L - (i - 1) * l + 1
        end = i * L - (i - 1) * l

        Fw = fft(data[start:end, 0])[0:int(L / 2)] / (L / 2)
        
        Lb = int(L / 2 / nb)

        for k in range(1, nb + 1):
            start = Lb * (k - 1) + 1
            end = k * Lb
            features[i - 1][k - 1] = mean(abs(Fw[start:end]))

    return features

def feature_arrangement_and_reduction(P_H,P_I,P_O,P_B,P_C,E_H,E_I,E_O,E_B,E_C):
    
    # Coverting time and frequency features to dataframes so we can use the concat function
    P_H_df = pd.DataFrame(P_H)
    P_I_df = pd.DataFrame(P_I)
    P_O_df = pd.DataFrame(P_O)
    P_B_df = pd.DataFrame(P_B)
    P_C_df = pd.DataFrame(P_C)

    E_H_df = pd.DataFrame(E_H)
    E_I_df = pd.DataFrame(E_I)
    E_O_df = pd.DataFrame(E_O)
    E_B_df = pd.DataFrame(E_B)
    E_C_df = pd.DataFrame(E_C)

    # stacking all time domain feature data row-wise
    TimeFeatures_stacked = pd.concat([P_H_df, P_I_df, P_O_df, P_B_df, P_C_df], axis=0)

    # stacking all frequency domain feature data row-wise
    FreqFeatures_stacked = pd.concat([E_H_df, E_I_df, E_O_df, E_B_df, E_C_df], axis=0)

    # Now placing all feature together (i.e. column-wise)
    features_all = pd.concat([TimeFeatures_stacked, FreqFeatures_stacked], axis = 1)

    # Reset the index if needed
    features_all.reset_index(drop=True, inplace=True)

    # Renaming all feature columns
    features_all.columns = [
        'rms', 
        'peak_value', 
        'peak_peak', 
        'crest_factor', 
        'mean', 
        'variance', 
        'skewness', 
        'kurtosis',
        'band_1', 
        'band_2', 
        'band_3',
        'band_4',
        'band_5',
        'band_6',
        'band_7',
        'band_8'
    ]

    # Doing feature reduction through COV and covariance
    # Calculate the means and standard deviations for the features:
    means = features_all.mean(axis=0)
    stds = features_all.std(axis=0)

    # Now we calculate the coeficcient of variability for each feature:
    cvs = stds/means

    # Filter out features with CV >= 0.05
    selected_features = cvs[cvs >= 0.05].index

    # Keep only selected features in df_features
    features_all = features_all[selected_features]

    # Compute the correlation matrix: by default, we use Pearson correlation coefficient
    df_corr_mat = features_all.corr()

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
    features_all = features_all.drop(features_all.columns[unique_list], axis=1)

    # Label Assignemnt-->   Heealthy:0       Inner:1        Outer:2          Ball:3          Combined:5 
    temp0 = pd.DataFrame({'labels': [0] * len(P_H)})
    temp1 = pd.DataFrame({'labels': [1] * len(P_I)})
    temp2 = pd.DataFrame({'labels': [2] * len(P_O)})
    temp3 = pd.DataFrame({'labels': [3] * len(P_B)})
    temp4 = pd.DataFrame({'labels': [4] * len(P_C)})
    Time_labels_stacked=pd.concat([temp0,temp1,temp2,temp3,temp4], axis=0)

    # Reset the index if needed
    Time_labels_stacked.reset_index(drop=True, inplace=True)

    # Converting to numpy arrays
    features_reduced = np.asarray(features_all)
    labels = np.asarray(Time_labels_stacked)

    # Splitting the data 70/30
    x_time_train_func, x_time_test_func, y_time_train_func, y_time_test_func = train_test_split(features_reduced, labels, test_size = 0.30, random_state = 42, shuffle = True)

    # Now scaling the features using standard scaler
    sc = StandardScaler()
    sc.fit(x_time_train_func)
    x_time_train_func = sc.fit_transform(x_time_train_func)
    x_time_test_func = sc.transform(x_time_test_func)

    return x_time_train_func, x_time_test_func, y_time_train_func, y_time_test_func 

def ML_features(x_time_train_func, x_time_test_func, y_time_train_func, y_time_test_func):

    # Importing random forest model from scikit
    from sklearn.ensemble import RandomForestClassifier
   
    # Create an instance (object) from the sklearn RandomForestClassifier class:
    rf = RandomForestClassifier(criterion = 'gini',  max_depth = 20,  min_samples_split = 2, min_samples_leaf = 1, min_impurity_decrease = 0, random_state = 42, n_estimators = 100, bootstrap = True, oob_score = False)

    rf.fit(x_time_train_func, y_time_train_func)

    # Check performance on the train dataset:
    score = rf.score(x_time_train_func, y_time_train_func)
    print(f"Accuracy on train dataset: {score:.2%}")

    # Get the model's predictions for the test dataset:
    y_time_pred_rf = rf.predict(x_time_test_func)
    labels_names = ['Normal/Healthy','Inner Race Damage','Outer Race Damage', 'Ball element Damage', 'Combined Damage']           
    print(classification_report(y_time_test_func, y_time_pred_rf, target_names = labels_names, digits = 5))

    # Getting the balanced and unbalanced accuracy on the test dataset
    print(f"Balanced Accurancy: {balanced_accuracy_score(y_time_test_func, y_time_pred_rf):.5f}")
    print(f"Unbalanced Accurancy: {accuracy_score(y_time_test_func, y_time_pred_rf):.5f}")

    # Compute confusion matrix
    cm_rf = confusion_matrix(y_time_test_func, y_time_pred_rf)
    plt.figure(figsize=(8, 6))
    ax=sns.heatmap(cm_rf, annot=True, fmt="d", cmap="coolwarm", xticklabels=labels_names, yticklabels=labels_names)
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_title("Confusion Matrix after feature extraction")
    plt.show()

    return cm_rf

def true_positive_percent(cm):
    
    labels_names = ['Normal/Healthy','Inner Race Damage','Outer Race Damage', 'Ball element Damage', 'Combined Damage']    
    # Calculate True Positive Percentage for each health state
    tp_percentage = np.diag(cm) / np.sum(cm, axis=1) * 100

    # Print True Positive Percentage for each class
    for i, tp in enumerate(tp_percentage):
        print(f"True Positive Percentage for {labels_names[i]}: {tp:.2f}%")

def ML_without_feature_reduction(P_H,P_I,P_O,P_B,P_C,E_H,E_I,E_O,E_B,E_C):
     # Coverting time and frequency features to dataframes so we can use the concat function
    P_H_df = pd.DataFrame(P_H)
    P_I_df = pd.DataFrame(P_I)
    P_O_df = pd.DataFrame(P_O)
    P_B_df = pd.DataFrame(P_B)
    P_C_df = pd.DataFrame(P_C)

    E_H_df = pd.DataFrame(E_H)
    E_I_df = pd.DataFrame(E_I)
    E_O_df = pd.DataFrame(E_O)
    E_B_df = pd.DataFrame(E_B)
    E_C_df = pd.DataFrame(E_C)

    # stacking all time domain feature data row-wise
    TimeFeatures_stacked = pd.concat([P_H_df, P_I_df, P_O_df, P_B_df, P_C_df], axis=0)

    # stacking all frequency domain feature data row-wise
    FreqFeatures_stacked = pd.concat([E_H_df, E_I_df, E_O_df, E_B_df, E_C_df], axis=0)

    # Now placing all feature together (i.e. column-wise)
    features_all = pd.concat([TimeFeatures_stacked, FreqFeatures_stacked], axis = 1)

    # Reset the index if needed
    features_all.reset_index(drop=True, inplace=True)

    # Renaming all feature columns
    features_all.columns = [
        'rms', 
        'peak_value', 
        'peak_peak', 
        'crest_factor', 
        'mean', 
        'variance', 
        'skewness', 
        'kurtosis',
        'band_1', 
        'band_2', 
        'band_3',
        'band_4',
        'band_5',
        'band_6',
        'band_7',
        'band_8'
    ]
    # Label Assignemnt-->   Heealthy:0       Inner:1        Outer:2          Ball:3          Combined:5 
    temp0 = pd.DataFrame({'labels': [0] * len(P_H)})
    temp1 = pd.DataFrame({'labels': [1] * len(P_I)})
    temp2 = pd.DataFrame({'labels': [2] * len(P_O)})
    temp3 = pd.DataFrame({'labels': [3] * len(P_B)})
    temp4 = pd.DataFrame({'labels': [4] * len(P_C)})
    Time_labels_stacked=pd.concat([temp0,temp1,temp2,temp3,temp4], axis=0)

    # Reset the index if needed
    Time_labels_stacked.reset_index(drop=True, inplace=True)

    # Converting to numpy arrays
    features_all = np.asarray(features_all)
    labels = np.asarray(Time_labels_stacked)

    # Splitting the data 70/30
    x_time_train_func, x_time_test_func, y_time_train_func, y_time_test_func = train_test_split(features_all, labels, test_size = 0.30, random_state = 42, shuffle = True)

    # Now scaling the features using standard scaler
    sc = StandardScaler()
    sc.fit(x_time_train_func)
    x_time_train_func = sc.fit_transform(x_time_train_func)
    x_time_test_func = sc.transform(x_time_test_func)

    # Importing random forest model from scikit
    from sklearn.ensemble import RandomForestClassifier
   
    # Create an instance (object) from the sklearn RandomForestClassifier class:
    rf = RandomForestClassifier(criterion = 'gini',  max_depth = 20,  min_samples_split = 2, min_samples_leaf = 1, min_impurity_decrease = 0, random_state = 42, n_estimators = 100, bootstrap = True, oob_score = False)

    rf.fit(x_time_train_func, y_time_train_func)

    # Check performance on the train dataset:
    score = rf.score(x_time_train_func, y_time_train_func)
    print(f"Accuracy on train dataset: {score:.2%}")

    # Get the model's predictions for the test dataset:
    y_time_pred_rf = rf.predict(x_time_test_func)
    labels_names = ['Normal/Healthy','Inner Race Damage','Outer Race Damage', 'Ball element Damage', 'Combined Damage']           
    print(classification_report(y_time_test_func, y_time_pred_rf, target_names = labels_names, digits = 5))

    # Getting the balanced and unbalanced accuracy on the test dataset
    print(f"Balanced Accurancy: {balanced_accuracy_score(y_time_test_func, y_time_pred_rf):.5f}")
    print(f"Unbalanced Accurancy: {accuracy_score(y_time_test_func, y_time_pred_rf):.5f}")

    # Compute confusion matrix
    cm_rf = confusion_matrix(y_time_test_func, y_time_pred_rf)
    plt.figure(figsize=(8, 6))
    ax=sns.heatmap(cm_rf, annot=True, fmt="d", cmap="coolwarm", xticklabels=labels_names, yticklabels=labels_names)
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_title("Confusion Matrix after feature extraction")
    plt.show()

    return cm_rf

def true_positive_percent_all_features(cm):

    labels_names = ['Normal/Healthy','Inner Race Damage','Outer Race Damage', 'Ball element Damage', 'Combined Damage']    
    # Calculate True Positive Percentage for each health state
    tp_percentage = np.diag(cm) / np.sum(cm, axis=1) * 100

    # Print True Positive Percentage for each class
    for i, tp in enumerate(tp_percentage):
        print(f"True Positive Percentage for {labels_names[i]}: {tp:.2f}%")
