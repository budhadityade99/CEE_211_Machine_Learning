import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess():
    # Loading the raw data as a dataframe
    df = pd.read_csv('Ball_Bearings_Contaminated.csv', header = 'infer')

    # Correcting all the feature columns which have different datatypes other than 'float64'
    peak_peak = df["peak-peak"]
    peak_peak = pd.to_numeric(peak_peak, errors = "coerce")
    df["peak-peak"] = peak_peak

    median = df["median"]
    median = pd.to_numeric(median, errors = "coerce")
    df["median"] = median

    # Dropping the "Unnamed: 0" column:
    df = df.drop("Unnamed: 0", axis = 'columns')

    # Then drop rows where all the columns have NaNs:
    df_nonull = df.dropna(how = "all")

    # Resetting Index
    df_nonull.reset_index(drop=True, inplace=True)

    # Missing data percentage per column:
    nan_pct = df_nonull.isna().sum() / len(df_nonull) 

    # Drop columns where more than 50% of the values are NaNs
    df = df.drop(columns=nan_pct[nan_pct > 0.5].index)

    # Now we deal with the features with NaNs by using the ffill() method:
    df_nonull = df_nonull.ffill()

    # Filtering the "rms" column since rms cannot be negative:
    filter1 = df_nonull["rms"].gt(0)

    # Apply the filter: if the conditon is False, the value is dropped
    df_nonull = df_nonull[filter1]

    # Filtering the "var" column since variance cannot be negative:
    filter2 = df_nonull["var"].gt(0)

    # Apply the filter: if the conditon is False, the value is dropped
    df_nonull = df_nonull[filter2]

    # Resetting Index
    df_nonull.reset_index(drop=True, inplace=True)

    return df_nonull

def pairplots(df_nonull):
    import seaborn as sns

    sns.pairplot(df_nonull)

def correlation(df_clean):
    import matplotlib.pyplot as plt
    
    # Compute correlation matrix
    corr_matrix = df_clean.corr()
    
    # Plot heatmap
    plt.figure(figsize=(12,6))
    import seaborn as sns
    corr_heatmat = sns.heatmap(corr_matrix, vmin = -1, vmax = 1, annot = True, cmap = "coolwarm")
    corr_heatmat.set_title("Ball Bearing: Pearson Correlation Matrix", fontsize = 16)
    plt.show()

def scaling(df_clean):
    from sklearn.preprocessing import StandardScaler

    # Convert the DataFrame to a NumPy array, as it is required by Scikit-Learn
    # Note that the resulting array drops the index, so we have just the sensor data in the array
    data = df_clean.to_numpy()

    # Define a list with the DataFrame columns so we can use it later for table visualization:
    data_columns = ["rms", 
                "peak", 
                "peak-peak", 
                "crest", 
                "median", 
                "var", 
                "skewness",
                "kurtosis"]

    # We normilize the data via StandardScaler:
    sc = StandardScaler()
    data = sc.fit_transform(data)
    return data

def clustering(data,df_nonull):
    import warnings
    warnings.filterwarnings("ignore")
    import matplotlib.pyplot as plt
    # Doing DBSCAN clustering
    from sklearn.cluster import DBSCAN
    from sklearn.cluster import KMeans

    # Let us apply the DBSCAN and Kmeans clustering to the PCA transformed data:
    dbscan = DBSCAN(eps = 0.8, min_samples = 500, metric = 'euclidean').fit(data)
    kmeans = KMeans(n_clusters = 2, random_state = 42, n_init  = 'auto').fit(data)

    # Number of clusters in labels, ignoring noise (labeled as -1), if present:
    n_clusters = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
    n_noisy = list(dbscan.labels_).count(-1)

    print(f"Number of labels: {len(set(dbscan.labels_))}")
    print(f"Number of clusters: {n_clusters}")
    print(f"Number of noisy samples: {n_noisy}")

    # plotting the clusters

    # Before proceeding, let us do some 2D visualizations with the data as it is, i.e., no transformations:
    fig, axs = plt.subplots(nrows = 8, ncols = 8, figsize = (12, 12))

    # Define a color mapping for the labels
    color_map = {-1: 'black', 0: 'red', 1: 'blue'}
    color_map2 = {-1: 'black', 0: 'blue', 1: 'red'}

    # Map each label to its corresponding color
    colors_dbscan = [color_map[label] if label in color_map else 'gray' for label in dbscan.labels_]
    colors_kmeans = [color_map2[label] if label in color_map2 else 'gray' for label in kmeans.labels_]

    ax = axs[1][0]
    ax.scatter(x = df_nonull["rms"], y = df_nonull["peak"], c = colors_dbscan, s = 5, cmap = 'Paired')
    #ax.set_xlabel(r"rms")
    ax.set_ylabel(r"peak")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[2][0]
    ax.scatter(x = df_nonull["rms"], y = df_nonull["peak-peak"], c = colors_dbscan, s = 5, cmap = 'Paired')
    #ax.set_xlabel(r"rms")
    ax.set_ylabel(r"peak-peak")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[3][0]
    ax.scatter(x = df_nonull["rms"], y = df_nonull["crest"], c = colors_dbscan, s = 5, cmap = 'Paired')
    #ax.set_xlabel(r"rms")
    ax.set_ylabel(r"crest")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[4][0]
    ax.scatter(x = df_nonull["rms"], y = df_nonull["median"], c = colors_dbscan, s = 5, cmap = 'Paired')
    #ax.set_xlabel(r"rms")
    ax.set_ylabel(r"median")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[5][0]
    ax.scatter(x = df_nonull["rms"], y = df_nonull["var"], c = colors_dbscan, s = 5, cmap = 'Paired')
    #ax.set_xlabel(r"rms")
    ax.set_ylabel(r"var")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[6][0]
    ax.scatter(x = df_nonull["rms"], y = df_nonull["skewness"], c = colors_dbscan, s = 5, cmap = 'Paired')
    #ax.set_xlabel(r"rms")
    ax.set_ylabel(r"skewness")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[7][0]
    ax.scatter(x = df_nonull["rms"], y = df_nonull["kurtosis"], c = colors_dbscan, s = 5, cmap = 'Paired')
    ax.set_xlabel(r"rms")
    ax.set_ylabel(r"kurtosis")
    ax.set_xticks([])
    ax.set_yticks([])

    #-------------------------------------------------------
    ax = axs[2][1]
    ax.scatter(x = df_nonull["peak"], y = df_nonull["peak-peak"], c = colors_dbscan, s = 5, cmap = 'Paired')
    #ax.set_xlabel(r"rms")
    #ax.set_ylabel(r"peak")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[3][1]
    ax.scatter(x = df_nonull["peak"], y = df_nonull["crest"], c = colors_dbscan, s = 5, cmap = 'Paired')
    #ax.set_xlabel(r"rms")
    #ax.set_ylabel(r"peak")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[4][1]
    ax.scatter(x = df_nonull["peak"], y = df_nonull["median"], c = colors_dbscan, s = 5, cmap = 'Paired')
    #ax.set_xlabel(r"rms")
    #ax.set_ylabel(r"peak")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[5][1]
    ax.scatter(x = df_nonull["peak"], y = df_nonull["var"], c = colors_dbscan, s = 5, cmap = 'Paired')
    #ax.set_xlabel(r"rms")
    #ax.set_ylabel(r"peak")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[6][1]
    ax.scatter(x = df_nonull["peak"], y = df_nonull["skewness"], c = colors_dbscan, s = 5, cmap = 'Paired')
    #ax.set_xlabel(r"rms")
    #ax.set_ylabel(r"peak")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[7][1]
    ax.scatter(x = df_nonull["peak"], y = df_nonull["kurtosis"], c = colors_dbscan, s = 5, cmap = 'Paired')
    ax.set_xlabel(r"peak")
    #ax.set_ylabel(r"peak")
    ax.set_xticks([])
    ax.set_yticks([])

    #-------------------------------------------------------
    ax = axs[3][2]
    ax.scatter(x = df_nonull["peak-peak"], y = df_nonull["crest"], c = colors_dbscan, s = 5, cmap = 'Paired')
    #ax.set_xlabel(r"rms")
    #ax.set_ylabel(r"rms")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[4][2]
    ax.scatter(x = df_nonull["peak-peak"], y = df_nonull["median"], c = colors_dbscan, s = 5, cmap = 'Paired')
    #ax.set_xlabel(r"rms")
    #ax.set_ylabel(r"rms")
    ax.set_xticks([])
    ax.set_yticks([])


    ax = axs[5][2]
    ax.scatter(x = df_nonull["peak-peak"], y = df_nonull["var"], c = colors_dbscan, s = 5, cmap = 'Paired')
    #ax.set_xlabel(r"rms")
    #ax.set_ylabel(r"rms")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[6][2]
    ax.scatter(x = df_nonull["peak-peak"], y = df_nonull["skewness"], c = colors_dbscan, s = 5, cmap = 'Paired')
    #ax.set_xlabel(r"rms")
    #ax.set_ylabel(r"rms")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[7][2]
    ax.scatter(x = df_nonull["peak-peak"], y = df_nonull["kurtosis"], c = colors_dbscan, s = 5, cmap = 'Paired')
    ax.set_xlabel(r"peak-peak")
    #ax.set_ylabel(r"rms")
    ax.set_xticks([])
    ax.set_yticks([])

    #-------------------------------------------------------
    ax = axs[4][3]
    ax.scatter(x = df_nonull["crest"], y = df_nonull["median"], c = colors_dbscan, s = 5, cmap = 'Paired')
    #ax.set_xlabel(r"rms")
    #ax.set_ylabel(r"rms")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[5][3]
    ax.scatter(x = df_nonull["crest"], y = df_nonull["var"], c = colors_dbscan, s = 5, cmap = 'Paired')
    #ax.set_xlabel(r"rms")
    #ax.set_ylabel(r"rms")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[6][3]
    ax.scatter(x = df_nonull["crest"], y = df_nonull["skewness"], c = colors_dbscan, s = 5, cmap = 'Paired')
    #ax.set_xlabel(r"rms")
    #ax.set_ylabel(r"rms")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[7][3]
    ax.scatter(x = df_nonull["crest"], y = df_nonull["kurtosis"], c = colors_dbscan, s = 5, cmap = 'Paired')
    ax.set_xlabel(r"crest")
    #ax.set_ylabel(r"rms")
    ax.set_xticks([])
    ax.set_yticks([])

    #-------------------------------------------------------
    ax = axs[5][4]
    ax.scatter(x = df_nonull["median"], y = df_nonull["var"], c = colors_dbscan, s = 5, cmap = 'Paired')
    #ax.set_xlabel(r"crest")
    #ax.set_ylabel(r"rms")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[6][4]
    ax.scatter(x = df_nonull["median"], y = df_nonull["skewness"], c = colors_dbscan, s = 5, cmap = 'Paired')
    #ax.set_xlabel(r"crest")
    #ax.set_ylabel(r"rms")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[7][4]
    ax.scatter(x = df_nonull["median"], y = df_nonull["kurtosis"], c = colors_dbscan, s = 5, cmap = 'Paired')
    ax.set_xlabel(r"median")
    #ax.set_ylabel(r"rms")
    ax.set_xticks([])
    ax.set_yticks([])

    #-------------------------------------------------------
    ax = axs[6][5]
    ax.scatter(x = df_nonull["var"], y = df_nonull["skewness"], c = colors_dbscan, s = 5, cmap = 'Paired')
    #ax.set_xlabel(r"var")
    #ax.set_ylabel(r"rms")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[7][5]
    ax.scatter(x = df_nonull["var"], y = df_nonull["kurtosis"], c = colors_dbscan, s = 5, cmap = 'Paired')
    ax.set_xlabel(r"var")
    #ax.set_ylabel(r"rms")
    ax.set_xticks([])
    ax.set_yticks([])

    #-------------------------------------------------------
    ax = axs[7][6]
    ax.scatter(x = df_nonull["skewness"], y = df_nonull["kurtosis"], c = colors_dbscan, s = 5, cmap = 'Paired')
    ax.set_xlabel(r"skewness")
    #ax.set_ylabel(r"rms")
    ax.set_xticks([])
    ax.set_yticks([])

    # Iterate over the grid and hide the upper triangular subplots
    for i in range(8):
        for j in range(8):
            if j == i:  # Upper triangle (excluding diagonal)
                fig.delaxes(axs[i, j])  # Remove subplot

    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------# -1: black for noise, 0=red, 1=blue

    ax = axs[0][1]
    ax.scatter(y = df_nonull["rms"], x = df_nonull["peak"], c = colors_kmeans, s = 5, cmap = 'Paired')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[0][2]
    ax.scatter(y = df_nonull["rms"], x = df_nonull["peak-peak"], c = colors_kmeans, s = 5, cmap = 'Paired')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[0][3]
    ax.scatter(y = df_nonull["rms"], x = df_nonull["crest"], c = colors_kmeans, s = 5, cmap = 'Paired')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[0][4]
    ax.scatter(y = df_nonull["rms"], x = df_nonull["median"], c = colors_kmeans, s = 5, cmap = 'Paired')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[0][5]
    ax.scatter(y = df_nonull["rms"], x = df_nonull["var"], c = colors_kmeans, s = 5, cmap = 'Paired')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[0][6]
    ax.scatter(y = df_nonull["rms"], x = df_nonull["skewness"], c = colors_kmeans, s = 5, cmap = 'Paired')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[0][7]
    ax.scatter(y = df_nonull["rms"], x = df_nonull["kurtosis"], c = colors_kmeans, s = 5, cmap = 'Paired')
    ax.set_xticks([])
    ax.set_yticks([])

    #-------------------------------------------------------
    ax = axs[1][2]
    ax.scatter(y = df_nonull["peak"], x = df_nonull["peak-peak"], c = colors_kmeans, s = 5, cmap = 'Paired')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[1][3]
    ax.scatter(y = df_nonull["peak"], x = df_nonull["crest"], c = colors_kmeans, s = 5, cmap = 'Paired')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[1][4]
    ax.scatter(y = df_nonull["peak"], x = df_nonull["median"], c = colors_kmeans, s = 5, cmap = 'Paired')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[1][5]
    ax.scatter(y = df_nonull["peak"], x = df_nonull["var"], c = colors_kmeans, s = 5, cmap = 'Paired')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[1][6]
    ax.scatter(y = df_nonull["peak"], x = df_nonull["skewness"], c = colors_kmeans, s = 5, cmap = 'Paired')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[1][7]
    ax.scatter(y = df_nonull["peak"], x = df_nonull["kurtosis"], c = colors_kmeans, s = 5, cmap = 'Paired')
    ax.set_xticks([])
    ax.set_yticks([])

    #-------------------------------------------------------
    ax = axs[2][3]
    ax.scatter(y = df_nonull["peak-peak"], x = df_nonull["crest"], c = colors_kmeans, s = 5, cmap = 'Paired')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[2][4]
    ax.scatter(y = df_nonull["peak-peak"], x = df_nonull["median"], c = colors_kmeans, s = 5, cmap = 'Paired')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[2][5]
    ax.scatter(y = df_nonull["peak-peak"], x = df_nonull["var"], c = colors_kmeans, s = 5, cmap = 'Paired')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[2][6]
    ax.scatter(y = df_nonull["peak-peak"], x = df_nonull["skewness"], c = colors_kmeans, s = 5, cmap = 'Paired')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[2][7]
    ax.scatter(y = df_nonull["peak-peak"], x = df_nonull["kurtosis"], c = colors_kmeans, s = 5, cmap = 'Paired')
    ax.set_xticks([])
    ax.set_yticks([])

    #-------------------------------------------------------
    ax = axs[3][4]
    ax.scatter(y = df_nonull["crest"], x = df_nonull["median"], c = colors_kmeans, s = 5, cmap = 'Paired')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[3][5]
    ax.scatter(y = df_nonull["crest"], x = df_nonull["var"], c = colors_kmeans, s = 5, cmap = 'Paired')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[3][6]
    ax.scatter(y = df_nonull["crest"], x = df_nonull["skewness"], c = colors_kmeans, s = 5, cmap = 'Paired')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[3][7]
    ax.scatter(y = df_nonull["crest"], x = df_nonull["kurtosis"], c = colors_kmeans, s = 5, cmap = 'Paired')
    ax.set_xticks([])
    ax.set_yticks([])

    #-------------------------------------------------------
    ax = axs[4][5]
    ax.scatter(y = df_nonull["median"], x = df_nonull["var"], c = colors_kmeans, s = 5, cmap = 'Paired')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[4][6]
    ax.scatter(y = df_nonull["median"], x = df_nonull["skewness"], c = colors_kmeans, s = 5, cmap = 'Paired')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[4][7]
    ax.scatter(y = df_nonull["median"], x = df_nonull["kurtosis"], c = colors_kmeans, s = 5, cmap = 'Paired')
    ax.set_xticks([])
    ax.set_yticks([])

    #-------------------------------------------------------
    ax = axs[5][6]
    ax.scatter(y = df_nonull["var"], x = df_nonull["skewness"], c = colors_kmeans, s = 5, cmap = 'Paired')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[5][7]
    ax.scatter(y = df_nonull["var"], x = df_nonull["kurtosis"], c = colors_kmeans, s = 5, cmap = 'Paired')
    ax.set_xticks([])
    ax.set_yticks([])

    #-------------------------------------------------------
    ax = axs[6][7]
    ax.scatter(y = df_nonull["skewness"], x = df_nonull["kurtosis"], c = colors_kmeans, s = 5, cmap = 'Paired')
    ax.set_xticks([])
    ax.set_yticks([])

    fig.show()
    
    # Create 3D figure with 3 features using DBSCAN clustering
    fig, axes = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(10, 8))

    # First 3D plot 
    ax = axes[0]
    ax.scatter(xs=df_nonull["var"],
            ys=df_nonull["rms"],
            zs=df_nonull["peak-peak"],
            c=colors_dbscan, s=10, cmap='Accent')

    ax.set_xlabel("var")
    ax.set_ylabel(r"rms")
    ax.set_zlabel(r"peak-peak", labelpad=15)

    # Second 3D plot 
    ax2 = axes[1]
    ax2.scatter(xs=df_nonull["median"],
                ys=df_nonull["crest"],
                zs=df_nonull["skewness"],
                c=colors_dbscan, s=10, cmap='viridis')  # Different colormap for contrast

    ax2.set_xlabel("median")
    ax2.set_ylabel(r"crest")
    ax2.set_zlabel(r"skewness", labelpad=15)

    # Show the figure
    plt.subplots_adjust(left=0, right=1, bottom=-1, top=1)
    plt.show()

    import matplotlib.pyplot as plt

    # Create 3D figure with 3 features using Kmeans clustering
    fig, axes = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(10, 8))

    # First 3D plot 
    ax = axes[0]
    ax.scatter(xs=df_nonull["var"],
            ys=df_nonull["rms"],
            zs=df_nonull["peak-peak"],
            c=colors_kmeans, s=10, cmap='Accent')

    ax.set_xlabel("var")
    ax.set_ylabel(r"rms")
    ax.set_zlabel(r"peak-peak", labelpad=15)

    # Second 3D plot 
    ax2 = axes[1]
    ax2.scatter(xs=df_nonull["median"],
                ys=df_nonull["crest"],
                zs=df_nonull["skewness"],
                c=colors_kmeans, s=10, cmap='viridis')  

    ax2.set_xlabel("median")
    ax2.set_ylabel(r"crest")
    ax2.set_zlabel(r"skewness", labelpad=15)

    # Show the figure
    plt.subplots_adjust(left=0, right=1, bottom=-1, top=1)
    plt.show()

def PCA(data):
    from sklearn.decomposition import PCA

    # Fit (training) the PCA for 95% explained variance
    pca = PCA(n_components = 0.95)

    pca.fit(data)

    print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
    print(f"Total Explained Variance: {pca.explained_variance_ratio_.sum():.2%}")

    #-------------------------------------------------------------------------------------#

    # Make a pie plot showing the explained variance per principal component:
    n = len(pca.explained_variance_ratio_)
    comps = ["Component " + str(i + 1) + f": {pca.explained_variance_ratio_[i]:.2%}" for i in range(n)]

    # Now, with the PCA aleady trained (fitted), we can transform the sensor data:
    data_transf = pca.transform(data)
    return data_transf

def clustering_PCA(data_transf):
    # Doing DBSCAN clustering with PCA transformed data
    from sklearn.cluster import DBSCAN

    # Let us apply the DBSCAN clustering to the PCA transformed data:
    dbscan_transf = DBSCAN(eps = 0.8, min_samples = 500, metric = 'euclidean').fit(data_transf)

    # Create a DataFrame with the PCA transformed data, DBSCAN labels
    df_data_pca_dbscan = pd.DataFrame(data_transf, columns = ["PC_1", "PC_2", "PC_3"])
    df_data_pca_dbscan["DBSCAN_Labels"] = dbscan_transf.labels_

    # Doing Kmeans clustering with PCA transformed data
    from sklearn.cluster import KMeans

    # Let us apply the Kmeans clustering to the PCA transformed data:
    kmeans_transf = KMeans(n_clusters = 2, random_state = 42, n_init  = 'auto').fit(data_transf)

    # Create a DataFrame with the PCA transformed data, Kmeans labels
    df_data_pca_kmeans = pd.DataFrame(data_transf, columns = ["PC_1", "PC_2", "PC_3"])
    df_data_pca_kmeans["kmeans_Labels"] = kmeans_transf.labels_

    # Define a color mapping for the labels
    color_map = {-1: 'black', 0: 'red', 1: 'blue'}
    color_map2 = {-1: 'black', 0: 'blue', 1: 'red'}

    colors_dbscan = [color_map[label] if label in color_map else 'gray' for label in dbscan_transf.labels_]
    colors_kmeans = [color_map2[label] if label in color_map2 else 'gray' for label in kmeans_transf.labels_]

    # Plotting DBSCAN clusters
    fig, axs = plt.subplots(nrows = 1, ncols = 3, figsize = (30, 10))

    ax = axs[0]
    ax.scatter(x = df_data_pca_dbscan["PC_1"], 
                y = df_data_pca_dbscan["PC_2"], 
                c = colors_dbscan, s = 10, cmap = 'Paired')
    ax.set_xlabel(r"PC_1")
    ax.set_ylabel(r"PC_2")
    #ax.legend()

    ax = axs[1]
    ax.scatter(x = df_data_pca_dbscan["PC_1"], 
                y = df_data_pca_dbscan["PC_3"], 
                c = colors_dbscan, s = 10, cmap = 'Paired')
    ax.set_xlabel(r"PC_1")
    ax.set_ylabel(r"PC_3")
    #ax.legend()

    ax = axs[2]
    ax.scatter(x = df_data_pca_dbscan["PC_2"], 
                y = df_data_pca_dbscan["PC_3"], 
                c = colors_dbscan, s = 10, cmap = 'Paired')
    ax.set_xlabel(r"PC_2")
    ax.set_ylabel(r"PC_3")

    # Plotting Kmeans clusters
    fig, axs = plt.subplots(nrows = 1, ncols = 3, figsize = (30, 10))

    ax = axs[0]
    ax.scatter(x = df_data_pca_kmeans["PC_1"], 
                y = df_data_pca_kmeans["PC_2"], 
                c = colors_kmeans, s = 10, cmap = 'Paired')
    ax.set_xlabel(r"PC_1")
    ax.set_ylabel(r"PC_2")
    #ax.legend()

    ax = axs[1]
    ax.scatter(x = df_data_pca_kmeans["PC_1"], 
                y = df_data_pca_kmeans["PC_3"], 
                c = colors_kmeans, s = 10, cmap = 'Paired')
    ax.set_xlabel(r"PC_1")
    ax.set_ylabel(r"PC_3")
    #ax.legend()

    ax = axs[2]
    ax.scatter(x = df_data_pca_kmeans["PC_2"], 
                y = df_data_pca_kmeans["PC_3"], 
                c = colors_kmeans, s = 10, cmap = 'Paired')
    ax.set_xlabel(r"PC_2")
    ax.set_ylabel(r"PC_3")
    #ax.legend()

def PCA_correlation(data_transf):
    # Calculate the correlation matrix for the transformed data and make a HeatMap:
    plt.figure(figsize = (6, 4))
    corr_heatmat = sns.heatmap(pd.DataFrame(data_transf).corr(), vmin = -1, vmax = 1, annot = True, cmap = "coolwarm")
    corr_heatmat.set_title("Ball Bearing: Transformed Data Correlation Matrix")
    plt.show()

def clustering_PCA_2D(data,df_nonull):
    import warnings
    warnings.filterwarnings("ignore")
    import matplotlib.pyplot as plt
    # Doing DBSCAN clustering
    from sklearn.cluster import DBSCAN
    from sklearn.cluster import KMeans

    # Let us apply the DBSCAN and Kmeans clustering to the PCA transformed data:
    dbscan = DBSCAN(eps = 0.8, min_samples = 500, metric = 'euclidean').fit(data)
    kmeans = KMeans(n_clusters = 2, random_state = 42, n_init  = 'auto').fit(data)

    # Number of clusters in labels, ignoring noise (labeled as -1), if present:
    n_clusters = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
    n_noisy = list(dbscan.labels_).count(-1)

    print(f"Number of labels: {len(set(dbscan.labels_))}")
    print(f"Number of clusters: {n_clusters}")
    print(f"Number of noisy samples: {n_noisy}")

    # plotting the clusters

    # Before proceeding, let us do some 2D visualizations with the data as it is, i.e., no transformations:
    fig, axs = plt.subplots(nrows = 8, ncols = 8, figsize = (12, 12))

    # Define a color mapping for the labels
    color_map = {-1: 'black', 0: 'red', 1: 'blue'}
    color_map2 = {-1: 'black', 0: 'blue', 1: 'red'}

    # Map each label to its corresponding color
    colors_dbscan = [color_map[label] if label in color_map else 'gray' for label in dbscan.labels_]
    colors_kmeans = [color_map2[label] if label in color_map2 else 'gray' for label in kmeans.labels_]

    ax = axs[1][0]
    ax.scatter(x = df_nonull["rms"], y = df_nonull["peak"], c = colors_dbscan, s = 5, cmap = 'Paired')
    #ax.set_xlabel(r"rms")
    ax.set_ylabel(r"peak")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[2][0]
    ax.scatter(x = df_nonull["rms"], y = df_nonull["peak-peak"], c = colors_dbscan, s = 5, cmap = 'Paired')
    #ax.set_xlabel(r"rms")
    ax.set_ylabel(r"peak-peak")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[3][0]
    ax.scatter(x = df_nonull["rms"], y = df_nonull["crest"], c = colors_dbscan, s = 5, cmap = 'Paired')
    #ax.set_xlabel(r"rms")
    ax.set_ylabel(r"crest")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[4][0]
    ax.scatter(x = df_nonull["rms"], y = df_nonull["median"], c = colors_dbscan, s = 5, cmap = 'Paired')
    #ax.set_xlabel(r"rms")
    ax.set_ylabel(r"median")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[5][0]
    ax.scatter(x = df_nonull["rms"], y = df_nonull["var"], c = colors_dbscan, s = 5, cmap = 'Paired')
    #ax.set_xlabel(r"rms")
    ax.set_ylabel(r"var")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[6][0]
    ax.scatter(x = df_nonull["rms"], y = df_nonull["skewness"], c = colors_dbscan, s = 5, cmap = 'Paired')
    #ax.set_xlabel(r"rms")
    ax.set_ylabel(r"skewness")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[7][0]
    ax.scatter(x = df_nonull["rms"], y = df_nonull["kurtosis"], c = colors_dbscan, s = 5, cmap = 'Paired')
    ax.set_xlabel(r"rms")
    ax.set_ylabel(r"kurtosis")
    ax.set_xticks([])
    ax.set_yticks([])

    #-------------------------------------------------------
    ax = axs[2][1]
    ax.scatter(x = df_nonull["peak"], y = df_nonull["peak-peak"], c = colors_dbscan, s = 5, cmap = 'Paired')
    #ax.set_xlabel(r"rms")
    #ax.set_ylabel(r"peak")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[3][1]
    ax.scatter(x = df_nonull["peak"], y = df_nonull["crest"], c = colors_dbscan, s = 5, cmap = 'Paired')
    #ax.set_xlabel(r"rms")
    #ax.set_ylabel(r"peak")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[4][1]
    ax.scatter(x = df_nonull["peak"], y = df_nonull["median"], c = colors_dbscan, s = 5, cmap = 'Paired')
    #ax.set_xlabel(r"rms")
    #ax.set_ylabel(r"peak")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[5][1]
    ax.scatter(x = df_nonull["peak"], y = df_nonull["var"], c = colors_dbscan, s = 5, cmap = 'Paired')
    #ax.set_xlabel(r"rms")
    #ax.set_ylabel(r"peak")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[6][1]
    ax.scatter(x = df_nonull["peak"], y = df_nonull["skewness"], c = colors_dbscan, s = 5, cmap = 'Paired')
    #ax.set_xlabel(r"rms")
    #ax.set_ylabel(r"peak")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[7][1]
    ax.scatter(x = df_nonull["peak"], y = df_nonull["kurtosis"], c = colors_dbscan, s = 5, cmap = 'Paired')
    ax.set_xlabel(r"peak")
    #ax.set_ylabel(r"peak")
    ax.set_xticks([])
    ax.set_yticks([])

    #-------------------------------------------------------
    ax = axs[3][2]
    ax.scatter(x = df_nonull["peak-peak"], y = df_nonull["crest"], c = colors_dbscan, s = 5, cmap = 'Paired')
    #ax.set_xlabel(r"rms")
    #ax.set_ylabel(r"rms")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[4][2]
    ax.scatter(x = df_nonull["peak-peak"], y = df_nonull["median"], c = colors_dbscan, s = 5, cmap = 'Paired')
    #ax.set_xlabel(r"rms")
    #ax.set_ylabel(r"rms")
    ax.set_xticks([])
    ax.set_yticks([])


    ax = axs[5][2]
    ax.scatter(x = df_nonull["peak-peak"], y = df_nonull["var"], c = colors_dbscan, s = 5, cmap = 'Paired')
    #ax.set_xlabel(r"rms")
    #ax.set_ylabel(r"rms")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[6][2]
    ax.scatter(x = df_nonull["peak-peak"], y = df_nonull["skewness"], c = colors_dbscan, s = 5, cmap = 'Paired')
    #ax.set_xlabel(r"rms")
    #ax.set_ylabel(r"rms")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[7][2]
    ax.scatter(x = df_nonull["peak-peak"], y = df_nonull["kurtosis"], c = colors_dbscan, s = 5, cmap = 'Paired')
    ax.set_xlabel(r"peak-peak")
    #ax.set_ylabel(r"rms")
    ax.set_xticks([])
    ax.set_yticks([])

    #-------------------------------------------------------
    ax = axs[4][3]
    ax.scatter(x = df_nonull["crest"], y = df_nonull["median"], c = colors_dbscan, s = 5, cmap = 'Paired')
    #ax.set_xlabel(r"rms")
    #ax.set_ylabel(r"rms")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[5][3]
    ax.scatter(x = df_nonull["crest"], y = df_nonull["var"], c = colors_dbscan, s = 5, cmap = 'Paired')
    #ax.set_xlabel(r"rms")
    #ax.set_ylabel(r"rms")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[6][3]
    ax.scatter(x = df_nonull["crest"], y = df_nonull["skewness"], c = colors_dbscan, s = 5, cmap = 'Paired')
    #ax.set_xlabel(r"rms")
    #ax.set_ylabel(r"rms")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[7][3]
    ax.scatter(x = df_nonull["crest"], y = df_nonull["kurtosis"], c = colors_dbscan, s = 5, cmap = 'Paired')
    ax.set_xlabel(r"crest")
    #ax.set_ylabel(r"rms")
    ax.set_xticks([])
    ax.set_yticks([])

    #-------------------------------------------------------
    ax = axs[5][4]
    ax.scatter(x = df_nonull["median"], y = df_nonull["var"], c = colors_dbscan, s = 5, cmap = 'Paired')
    #ax.set_xlabel(r"crest")
    #ax.set_ylabel(r"rms")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[6][4]
    ax.scatter(x = df_nonull["median"], y = df_nonull["skewness"], c = colors_dbscan, s = 5, cmap = 'Paired')
    #ax.set_xlabel(r"crest")
    #ax.set_ylabel(r"rms")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[7][4]
    ax.scatter(x = df_nonull["median"], y = df_nonull["kurtosis"], c = colors_dbscan, s = 5, cmap = 'Paired')
    ax.set_xlabel(r"median")
    #ax.set_ylabel(r"rms")
    ax.set_xticks([])
    ax.set_yticks([])

    #-------------------------------------------------------
    ax = axs[6][5]
    ax.scatter(x = df_nonull["var"], y = df_nonull["skewness"], c = colors_dbscan, s = 5, cmap = 'Paired')
    #ax.set_xlabel(r"var")
    #ax.set_ylabel(r"rms")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[7][5]
    ax.scatter(x = df_nonull["var"], y = df_nonull["kurtosis"], c = colors_dbscan, s = 5, cmap = 'Paired')
    ax.set_xlabel(r"var")
    #ax.set_ylabel(r"rms")
    ax.set_xticks([])
    ax.set_yticks([])

    #-------------------------------------------------------
    ax = axs[7][6]
    ax.scatter(x = df_nonull["skewness"], y = df_nonull["kurtosis"], c = colors_dbscan, s = 5, cmap = 'Paired')
    ax.set_xlabel(r"skewness")
    #ax.set_ylabel(r"rms")
    ax.set_xticks([])
    ax.set_yticks([])

    # Iterate over the grid and hide the upper triangular subplots
    for i in range(8):
        for j in range(8):
            if j == i:  # Upper triangle (excluding diagonal)
                fig.delaxes(axs[i, j])  # Remove subplot

    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------# -1: black for noise, 0=red, 1=blue

    ax = axs[0][1]
    ax.scatter(y = df_nonull["rms"], x = df_nonull["peak"], c = colors_kmeans, s = 5, cmap = 'Paired')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[0][2]
    ax.scatter(y = df_nonull["rms"], x = df_nonull["peak-peak"], c = colors_kmeans, s = 5, cmap = 'Paired')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[0][3]
    ax.scatter(y = df_nonull["rms"], x = df_nonull["crest"], c = colors_kmeans, s = 5, cmap = 'Paired')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[0][4]
    ax.scatter(y = df_nonull["rms"], x = df_nonull["median"], c = colors_kmeans, s = 5, cmap = 'Paired')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[0][5]
    ax.scatter(y = df_nonull["rms"], x = df_nonull["var"], c = colors_kmeans, s = 5, cmap = 'Paired')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[0][6]
    ax.scatter(y = df_nonull["rms"], x = df_nonull["skewness"], c = colors_kmeans, s = 5, cmap = 'Paired')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[0][7]
    ax.scatter(y = df_nonull["rms"], x = df_nonull["kurtosis"], c = colors_kmeans, s = 5, cmap = 'Paired')
    ax.set_xticks([])
    ax.set_yticks([])

    #-------------------------------------------------------
    ax = axs[1][2]
    ax.scatter(y = df_nonull["peak"], x = df_nonull["peak-peak"], c = colors_kmeans, s = 5, cmap = 'Paired')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[1][3]
    ax.scatter(y = df_nonull["peak"], x = df_nonull["crest"], c = colors_kmeans, s = 5, cmap = 'Paired')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[1][4]
    ax.scatter(y = df_nonull["peak"], x = df_nonull["median"], c = colors_kmeans, s = 5, cmap = 'Paired')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[1][5]
    ax.scatter(y = df_nonull["peak"], x = df_nonull["var"], c = colors_kmeans, s = 5, cmap = 'Paired')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[1][6]
    ax.scatter(y = df_nonull["peak"], x = df_nonull["skewness"], c = colors_kmeans, s = 5, cmap = 'Paired')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[1][7]
    ax.scatter(y = df_nonull["peak"], x = df_nonull["kurtosis"], c = colors_kmeans, s = 5, cmap = 'Paired')
    ax.set_xticks([])
    ax.set_yticks([])

    #-------------------------------------------------------
    ax = axs[2][3]
    ax.scatter(y = df_nonull["peak-peak"], x = df_nonull["crest"], c = colors_kmeans, s = 5, cmap = 'Paired')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[2][4]
    ax.scatter(y = df_nonull["peak-peak"], x = df_nonull["median"], c = colors_kmeans, s = 5, cmap = 'Paired')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[2][5]
    ax.scatter(y = df_nonull["peak-peak"], x = df_nonull["var"], c = colors_kmeans, s = 5, cmap = 'Paired')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[2][6]
    ax.scatter(y = df_nonull["peak-peak"], x = df_nonull["skewness"], c = colors_kmeans, s = 5, cmap = 'Paired')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[2][7]
    ax.scatter(y = df_nonull["peak-peak"], x = df_nonull["kurtosis"], c = colors_kmeans, s = 5, cmap = 'Paired')
    ax.set_xticks([])
    ax.set_yticks([])

    #-------------------------------------------------------
    ax = axs[3][4]
    ax.scatter(y = df_nonull["crest"], x = df_nonull["median"], c = colors_kmeans, s = 5, cmap = 'Paired')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[3][5]
    ax.scatter(y = df_nonull["crest"], x = df_nonull["var"], c = colors_kmeans, s = 5, cmap = 'Paired')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[3][6]
    ax.scatter(y = df_nonull["crest"], x = df_nonull["skewness"], c = colors_kmeans, s = 5, cmap = 'Paired')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[3][7]
    ax.scatter(y = df_nonull["crest"], x = df_nonull["kurtosis"], c = colors_kmeans, s = 5, cmap = 'Paired')
    ax.set_xticks([])
    ax.set_yticks([])

    #-------------------------------------------------------
    ax = axs[4][5]
    ax.scatter(y = df_nonull["median"], x = df_nonull["var"], c = colors_kmeans, s = 5, cmap = 'Paired')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[4][6]
    ax.scatter(y = df_nonull["median"], x = df_nonull["skewness"], c = colors_kmeans, s = 5, cmap = 'Paired')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[4][7]
    ax.scatter(y = df_nonull["median"], x = df_nonull["kurtosis"], c = colors_kmeans, s = 5, cmap = 'Paired')
    ax.set_xticks([])
    ax.set_yticks([])

    #-------------------------------------------------------
    ax = axs[5][6]
    ax.scatter(y = df_nonull["var"], x = df_nonull["skewness"], c = colors_kmeans, s = 5, cmap = 'Paired')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[5][7]
    ax.scatter(y = df_nonull["var"], x = df_nonull["kurtosis"], c = colors_kmeans, s = 5, cmap = 'Paired')
    ax.set_xticks([])
    ax.set_yticks([])

    #-------------------------------------------------------
    ax = axs[6][7]
    ax.scatter(y = df_nonull["skewness"], x = df_nonull["kurtosis"], c = colors_kmeans, s = 5, cmap = 'Paired')
    ax.set_xticks([])
    ax.set_yticks([])

    fig.show()
    
def three_dimensional_cluster_PCA(data_transf):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
    from sklearn.cluster import DBSCAN
    from sklearn.cluster import KMeans

    # Let us apply the DBSCAN and Kmeans clustering to the PCA transformed data:
    dbscan_transf = DBSCAN(eps = 0.8, min_samples = 500, metric = 'euclidean').fit(data_transf)
    kmeans_transf = KMeans(n_clusters = 2, random_state = 42, n_init  = 'auto').fit(data_transf)

    # plotting the clusters

    # Define a color mapping for the labels
    color_map = {-1: 'black', 0: 'red', 1: 'blue'}
    color_map2 = {-1: 'black', 0: 'blue', 1: 'red'}

    # Map each label to its corresponding color
    colors_dbscan = [color_map[label] if label in color_map else 'gray' for label in dbscan_transf.labels_]
    colors_kmeans = [color_map2[label] if label in color_map2 else 'gray' for label in kmeans_transf.labels_]

    # Create a figure with 1 row and 2 columns of 3D subplots
    fig, axes = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(14, 8))

    # Create a DataFrame with the PCA transformed data, DBSCAN labels
    df_data_pca_dbscan = pd.DataFrame(data_transf, columns = ["PC_1", "PC_2", "PC_3"])
    df_data_pca_dbscan["DBSCAN_Labels"] = dbscan_transf.labels_

    # Create a DataFrame with the PCA transformed data, Kmeans labels
    df_data_pca_kmeans = pd.DataFrame(data_transf, columns = ["PC_1", "PC_2", "PC_3"])
    df_data_pca_kmeans["kmeans_Labels"] = kmeans_transf.labels_

    # First 3D plot
    ax = axes[0]
    ax.scatter(xs=df_data_pca_dbscan["PC_1"],
            ys=df_data_pca_dbscan["PC_2"],
            zs=df_data_pca_dbscan["PC_3"],
            c=colors_dbscan)

    ax.set_xlabel("PC_1")
    ax.set_ylabel("PC_2")
    ax.set_zlabel("PC_3", labelpad=15)

    # Second 3D plot
    ax2 = axes[1]
    ax2.scatter(xs=df_data_pca_kmeans["PC_1"],
                ys=df_data_pca_kmeans["PC_2"],
                zs=df_data_pca_kmeans["PC_3"],
                c=colors_kmeans)  

    ax2.set_xlabel("PC_1")
    ax2.set_ylabel("PC_2")
    ax2.set_zlabel("PC_3", labelpad=15)

    # Show the figure
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.show()

def cluster_space_plot(data_transf):

    from sklearn.cluster import KMeans
    # map the PCA-transformed data to the cluster-distance space:
    kmeans_transf = KMeans(n_clusters = 2, random_state = 42, n_init  = 'auto').fit(data_transf)
    clustered_data = kmeans_transf.transform(data_transf)

    color_map2 = {-1: 'black', 0: 'red', 1: 'blue'}
    colors_kmeans = [color_map2[label] if label in color_map2 else 'gray' for label in kmeans_transf.labels_]

    # Now we do a 3D plot of the clustered PCA-transformed data with labels according to k-means:
    plt.scatter(x = clustered_data[:, 0],
           y = clustered_data[:, 1], 
           c = colors_kmeans)
    plt.xlabel
