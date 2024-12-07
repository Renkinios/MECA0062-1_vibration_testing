import scipy.io
import numpy as np
import matplotlib.pyplot as plt


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







