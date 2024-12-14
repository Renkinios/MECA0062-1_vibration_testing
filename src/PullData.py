import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def extract_data(data_set) :
    # Charger le fichier .mat
    mat = scipy.io.loadmat(data_set)
    key_variables = {name: data for name, data in mat.items() if not name.startswith("__")}
    return key_variables
# Extraction et traçage des données disponibles
# ["X1", "X2", "W1", "S1", "C1_2", "C1_3", "C1_4", "Z1_2", "Z1_3", "Z1_4", 
            # "G1_1", "G1_2", "G1_3", "G1_4", "G2_2", "G3_3", "G4_4"]
def extract_H_general(number_data_set) :
    """
    This function is use only if the number of the data set is write as DPsv000XX.mat
    """
    number_shock = len(number_data_set) 
    number_accelerometer = 3
    sample_data = extract_data(f"../data/sec_lab/DPsv{str(number_data_set[0]).zfill(5)}.mat")
    vector_length = len(sample_data["H1_2"][:, 1])
    # idx           = (np.real(sample_data["H1_2"][:, 0])/2/np.pi >= 0) & (np.real(sample_data["H1_2"][:, 0])/2/np.pi <= 180)
    vector_length = len(sample_data["H1_2"][:, 1])
    matrix_H = np.zeros((number_shock, number_accelerometer, vector_length), dtype=np.complex_) # use the fact that Hrs = Hsr
    for i, number_data in enumerate(number_data_set):
        number_data_set = str(number_data).zfill(5)
        name_data       = f"../data/sec_lab/DPsv{number_data_set}.mat"
        data            = extract_data(name_data)
        matrix_H[i][0]  = (data["H1_2"][:, 1])
        matrix_H[i][1]  = (data["H1_3"][:, 1])
        matrix_H[i][2]  = (data["H1_4"][:, 1])
    freq = np.real(data["H1_2"][:, 0])
    return matrix_H, freq

def extract_mode_samcef(Path):
    # Read data and rename columns
    data = pd.read_csv(Path, sep='\s+', engine='python')
    data.columns = ["X_Coord", "Y_Coord", "Z_Coord", "X_Data", "Y_Data", "Z_Data", "Unused1", "Unused2", "Unused3"]

    # Keep only relevant columns and drop rows with missing values
    cleaned_data = data[["X_Coord", "Y_Coord", "Z_Coord", "X_Data", "Y_Data", "Z_Data"]].dropna()

    # Ensure X_Coord, Y_Coord, and Z_Coord are treated as integers for comparison
    cleaned_data["X_Coord"] = cleaned_data["X_Coord"].astype(int)
    cleaned_data["Y_Coord"] = cleaned_data["Y_Coord"].astype(int)
    cleaned_data["Z_Coord"] = cleaned_data["Z_Coord"].astype(int)
    return cleaned_data


def extract_node_shock(Path):
    # Read data and ensure coordinate columns are treated as integers
    df = pd.read_csv(Path)
    df.columns = ["X_Coord", "Y_Coord", "Z_Coord"]
    df["X_Coord"] = df["X_Coord"].astype(int)
    df["Y_Coord"] = df["Y_Coord"].astype(int)
    df["Z_Coord"] = df["Z_Coord"].astype(int)
    return df


def extract_samcef_shock():
    # Extract node data
    nodes = extract_node_shock("../data/node_shock.csv")

    # Initialize dictionary to store results
    data_mode_samcef = np.zeros((13, len(nodes)))
    
    # Process each mode file
    for i in range(13) : # like 13 mode
        path_samcef = f"../data/mode_samcef/mode_{i+1}.csv" 
        samcef = extract_mode_samcef(path_samcef)
        ordered_filtered_samcef = nodes.merge(samcef, how="left", on=["X_Coord", "Y_Coord", "Z_Coord"])
        data_mode_samcef[i] =  ordered_filtered_samcef['Z_Data'].iloc[:68].tolist() + ordered_filtered_samcef['X_Data'].iloc[68:].tolist()
        data_mode_samcef[i] = np.array(data_mode_samcef[i])/np.max(np.abs(data_mode_samcef[i]))
    return np.array(data_mode_samcef)


def plot_structure(data_samcef):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the data points
    ax.scatter(data_samcef['X_Coord'], data_samcef['Y_Coord'], data_samcef['Z_Coord'], c='blue', marker='o', s=10)

    # Set labels and title
    ax.set_title('3D Plot of Coordinates', fontsize=16)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    plt.show()


# for i in range(13) : # like 13 mode
#     path_samcef = f"../data/mode_samcef/mode_{i+1}.csv" 
#     samcef = extract_mode_samcef(path_samcef)
#     plot_structure(samcef)












