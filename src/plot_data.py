import matplotlib.pyplot as plt
import numpy as np
import first_data_fct as fdf
import os


# Définition globale des paramètres de police et de taille pour tous les graphiques
# plt.rc('font', family='serif')  # Police avec empattements, comme Times
# plt.rc('text', usetex=True)  # Utiliser LaTeX pour le texte dans les figures
# plt.rcParams.update({
#     'font.size': 14,       # Taille de police générale
#     'legend.fontsize': 15, # Taille de police pour les légendes
#     'axes.labelsize': 18,  # Taille de police pour les étiquettes des axes
# })


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
    # plt.legend(loc ="upper right")
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
    plt.figure(figsize=(10, 9))  # Augmenter la taille de la figure
    plt.plot(time, amplitude)
    plt.xlabel(r"Time [s]", fontsize=18)  # Taille des labels encore plus grande
    plt.ylabel(r"Amplitude [g]", fontsize=18)

    plt.savefig(f"../figures/first_lab/{set_name}/time_accelerometer.pdf", format="pdf", dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()

def viz_stabilisation_diagram(dic_order, cmif, freq, plot_stabilisation_poles = True):
    fig, ax1 = plt.subplots(figsize=(10, 8))
    ax1.set_xlabel("Frequency [Hz]")
    ax1.set_ylabel("CMIF [-]", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.semilogy(freq, cmif, color='blue')

    # Deuxième axe pour les stabilisations
    ax2 = ax1.twinx()
    ax2.set_ylabel("Poles [-]", color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    for key in dic_order.keys():
        w_i = dic_order[key]["wn"]
        stable = dic_order[key]["stable"]
        # print(stable)
        # ax2.axhline(y=key, color='black', linestyle='--', alpha=0.6)
        for i, w in enumerate(w_i) :
            point_color ='red' if stable[i] == 'd' else 'black'
            if stable[i] == 'x':
                ax2.scatter(w/2/np.pi, key, marker = 'o', s=5, color=point_color)
            elif stable[i] == 'v':
                ax2.scatter(w/2/np.pi, key, marker = stable[i], s=10, facecolors='none',color=point_color)
            else:
                ax2.scatter(w/2/np.pi, key, marker = stable[i], s=20, facecolors='none', color=point_color)
    if plot_stabilisation_poles:
        if max(dic_order.keys()) < 73 :
            print("The model cannot caputre all the stabilization pole used")
        else :
            selected_pole = [30,30,33,33,55,32,40,63,55,53,30,30,37]
            idx_freq      = [1,5 ,15, 16, 29, 19, 26, 44, 40, 41, 25, 27, 37]
            for i in range(len(selected_pole)) :
                ax2.scatter(dic_order[selected_pole[i]]["wn"][idx_freq[i]]/2/np.pi , selected_pole[i] ,color='green', facecolors='none', s=20,marker='d')


    ax2.scatter(200,20, color='red',   label='Stabilized', facecolors='none', s=20,marker='d')
    ax2.scatter(200,20, color='black', label='Unstabilized', s=20,marker='o',)
    ax2.scatter(200,20, color='black', label='Stabilized in frequency (1 \%)', s=20, marker='v', facecolors='none')
    ax2.scatter(200,20, color='green', label='Chossen poles', facecolors='none', s=20,marker='d')
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2)

    plt.xlim(13, 180)
    # plt.savefig("../figures/sec_lab/stabilisation_diagram.pdf", format="pdf", dpi=300, bbox_inches='tight')
    plt.show()




