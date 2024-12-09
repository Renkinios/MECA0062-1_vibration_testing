import numpy as np

def get_modal_assurance_criterion(modal_samcef, modal_testing) : 
    MAC = np.zeros((len(modal_samcef), len(modal_testing)))
    for i in range(len(modal_samcef)) :
        for j in range(len(modal_testing)) :
            MAC[i, j] = np.abs(np.dot(modal_samcef[i], modal_testing[j]))**2 / (np.dot(modal_samcef[i], modal_samcef[i]) * np.dot(modal_testing[j], modal_testing[j]))
    return MAC

def get_autoMAC(modal) :
    MAC_auto = np.zeros((len(modal), len(modal)))
    for i in range(len(modal)) :
        for j in range(len(modal)) :
            MAC_auto[i, j] = np.abs(np.dot(modal[i], modal[j]))**2 / (np.dot(modal[i], modal[i]) * np.dot(modal[j], modal[j]))
    return MAC_auto