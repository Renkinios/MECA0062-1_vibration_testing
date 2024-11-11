import matplotlib.pyplot as plt
import numpy as np
import first_data_fct as fdf
import os

def cmf_plot(freq, cmf ,set_name):
    save_dir = f"../figures/first_lab/{set_name}"
    save_path = f"{save_dir}/CMIF.pdf"
    plt.figure(figsize=(10, 5))
    plt.semilogy(freq, cmf)
    plt.xlabel("Frequency [Hz]", fontsize= 15)
    plt.ylabel("CMIF", fontsize= 15)
    plt.grid(True)
    plt.savefig(save_path, format="pdf", dpi=300)
    
    plt.close()

def bode_plot(data, set_name):
    # Chemin de sauvegarde
    save_dir = f"../figures/first_lab/{set_name}"
    save_path = f"{save_dir}/Bode_plot.pdf"
    
    # Créer le dossier s'il n'existe pas
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    freq = data["G1_1"][:, 0]
    H1_2 = data["H1_2"][:, 1]
    H1_3 = data["H1_3"][:, 1]
    H1_4 = data["H1_4"][:, 1]

    plt.figure(figsize=(10, 5))
    plt.semilogy(freq, np.abs(H1_2), label="H1_2")
    plt.semilogy(freq, np.abs(H1_3), label="H1_3")
    plt.semilogy(freq, np.abs(H1_4), label="H1_4")
    plt.xlabel("Frequency [Hz]", fontsize=15)
    plt.ylabel("Amplitude dBMag [g/N]", fontsize=15)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(save_path, format="pdf", dpi=300)
    plt.close()

def coherence_plot(data, set_name):
    # Chemin de sauvegarde
    save_dir = f"../figures/first_lab/{set_name}"
    save_path = f"{save_dir}/Coherence_plot.pdf"
    
    # Créer le dossier s'il n'existe pas
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    freq = data["G1_1"][:, 0]
    C1_2 = data["C1_2"][:, 1]
    C1_3 = data["C1_3"][:, 1]
    C1_4 = data["C1_4"][:, 1]

    plt.figure(figsize=(10, 5))
    plt.plot(freq, C1_2, label="C1_2")
    plt.plot(freq, C1_3, label="C1_3")
    plt.plot(freq, C1_4, label="C1_4")
    plt.xlabel("Frequency [Hz]", fontsize=15)
    plt.ylabel("Magnitude", fontsize=15)

    
    plt.savefig(save_path, format="pdf", dpi=600)
    plt.close()

def plot_exitasion_shock(data, set_name) :
    freq = data["G1_1"][:, 0]
    amplitude = data["G1_1"][:, 1]
    plt.figure(figsize=(10, 5))
    plt.plot(freq, amplitude)
    plt.xlabel("Frequency [Hz]", fontsize=15)
    plt.ylabel("Amplitude [N]", fontsize=15)
    plt.savefig(f"../figures/first_lab/{set_name}/exitasion_shock.pdf", format="pdf", dpi=300)
    plt.close()

def plot_time_shock(data, set_name) :
    time = data["X1"][:, 0]
    amplitude = data["X1"][:, 1]
    plt.figure(figsize=(10, 5))
    plt.plot(time, amplitude)
    plt.xlabel("Time [s]", fontsize=15)
    plt.ylabel("Amplitude [N]", fontsize=15)
    plt.savefig(f"../figures/first_lab/{set_name}/time_shock.pdf", format="pdf", dpi=300)
    plt.close()





