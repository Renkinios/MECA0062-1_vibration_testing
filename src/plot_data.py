import matplotlib.pyplot as plt
import numpy as np
import first_data_fct as fdf
import os


# Définition globale des paramètres de police et de taille pour tous les graphiques
plt.rc('font', family='serif')  # Police avec empattements, comme Times
plt.rc('text', usetex=True)  # Utiliser LaTeX pour le texte dans les figures
plt.rcParams.update({
    'font.size': 14,       # Taille de police générale
    'legend.fontsize': 15, # Taille de police pour les légendes
    'axes.labelsize': 18,  # Taille de police pour les étiquettes des axes
})


def cmf_plot(freq, cmf ,set_name):
    save_dir = f"../figures/first_lab/{set_name}"
    save_path = f"{save_dir}/CMIF.pdf"
    plt.figure(figsize=(10, 4))
    plt.semilogy(freq, cmf)
    plt.xlabel(r"Frequency [Hz]")
    plt.ylabel(r"CMIF [-]")
    # plt.grid(True)
    plt.savefig(save_path, format="pdf", dpi=300, bbox_inches='tight')
    
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

    plt.figure(figsize=(10, 4))
    plt.semilogy(freq, np.abs(H1_2), label="H1_2")
    plt.semilogy(freq, np.abs(H1_3), label="H1_3")
    plt.semilogy(freq, np.abs(H1_4), label="H1_4")
    plt.xlabel(r"Frequency [Hz]")
    plt.ylabel(r"Amplitude dBMag [g/N]")
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(save_path, format="pdf", dpi=300, bbox_inches='tight')
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

    plt.figure(figsize=(10, 4))
    plt.plot(freq, C1_2, label="C1_2")
    plt.plot(freq, C1_3, label="C1_3")
    plt.plot(freq, C1_4, label="C1_4")
    plt.xlabel(r"Frequency [Hz]")
    plt.ylabel(r"Magnitude [-]")
    plt.legend(loc ="upper right")
    plt.tight_layout()
    plt.savefig(save_path, format="pdf", dpi=300, bbox_inches='tight')
    plt.close()

def plot_exitasion_shock(data, set_name) :
    freq = data["G1_1"][:, 0]
    amplitude = data["G1_1"][:, 1]
    plt.figure(figsize=(10,6))
    plt.plot(freq, amplitude)
    plt.xlabel(r"Frequency [Hz]", fontsize=18)
    plt.ylabel(r"Amplitude [N]", fontsize=18)
    plt.savefig(f"../figures/first_lab/{set_name}/exitasion_shock.pdf", format="pdf", dpi=300, bbox_inches='tight')
    plt.close()

def plot_time_shock(data, set_name):
    time = data["X1"][:, 0]
    arg = (time <= 0.08)
    amplitude = data["X1"][:, 1]
    amplitude = amplitude[arg]
    time = time[arg] * 100
    plt.figure(figsize=(10, 6))  # Augmenter la taille de la figure
    plt.plot(time, amplitude)
    plt.xlabel(r"Time [ms]", fontsize=18)  # Taille des labels encore plus grande
    plt.ylabel(r"Amplitude [N]", fontsize=18)
    plt.savefig(f"../figures/first_lab/{set_name}/time_shock.pdf", format="pdf", dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()

def plot_accelerometer_time(data, set_name):
    time = data["X2"][:, 0]
    amplitude = data["X2"][:, 1]
    plt.figure(figsize=(10, 6))  # Augmenter la taille de la figure
    plt.plot(time, amplitude)
    plt.xlabel(r"Time [s]", fontsize=18)  # Taille des labels encore plus grande
    plt.ylabel(r"Amplitude [g]", fontsize=18)

    plt.savefig(f"../figures/first_lab/{set_name}/time_accelerometer.pdf", format="pdf", dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()