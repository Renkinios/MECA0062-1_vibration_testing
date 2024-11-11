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




