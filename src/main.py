import extract_data as ed
import plot_data as pld
import first_data_fct as fdf
import interpolate_data as idf
import representation_mode as rm
import polymax as pm
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm 



number_data = 1
number_data_set = str(number_data).zfill(5)
name_data       = f"../data/first_lab/DPsv{number_data_set}.mat"
name_set = f"set_{number_data}"
data = ed.extract_data(name_data)
# pld.bode_plot(data,name_set)
# pld.coherence_plot(data,name_set)
# pld.plot_exitasion_shock(data,name_set)
# pld.plot_time_shock(data,name_set)
# pld.plot_accelerometer_time(data,name_set)
cmif = fdf.compute_cmif(data)
# pld.cmf_plot(data["G1_1"][:, 0], cmif,name_set)

freq_cmif = np.real(data["H1_2"][:, 0])

# mask = (freq >= 18.4) & (freq <= 19.4)
# # mask =(freq >= 15) & (freq <= 25)
# freq_first_mode = freq[mask]
# H1_2 = data["H1_2"][:, 1]
# H1_2_first_mode = H1_2[mask]
# cmif_first_mode = cmif[mask]

# # H1_2_first_mode_abs = np.abs(H1_2_first_mode)


# lin_freq, lin_H  = idf.compute_linear_interp(freq_first_mode, cmif_first_mode, 1000)
# # cub_freq, cub_H  = idf.compute_cubic_spline(freq_first_mode, cmif_first_mode, 1000)

# # damping_peak_picking_method = fdf.compute_peak_picking_method(lin_H, lin_freq, plot=True, set_name=name_set)
# # print(f"Damping factor for pick picking method cubic: {damping_peak_picking_method}")
# damping_circle_fit_method = fdf.compute_circle_fit_method(freq_first_mode, H1_2_first_mode, plot=True, set_name=name_set)
# # print(f"Damping factor for circle fit method: {damping_circle_fit_method}")

# part 2 

# Create the arrays based on the user's specifications
array1       = np.arange(1, 29) 
array2       = np.arange(31, 59) 
array3       = np.arange(61, 79)  
result_array = np.concatenate((array1, array2, array3))

H, freq = ed.extract_H_general(result_array)
delta_t = 1.9531 * 10**(-3) 
modal   = {}

for i in tqdm(range(29,31), desc="Polymax poles", unit="Poles"):
# for i in range(20,100):
    w_i, damping_i, eigenval = pm.get_polymax(H, freq, i, delta_t)
    idx            = (w_i/2/np.pi >= 13) & (w_i/2/np.pi <= 180)
    _, idx_unique  = np.unique(w_i[idx], return_index=True)
    modal[i]       = {
        "wn"       : w_i[idx][idx_unique], 
        "zeta"     : damping_i[idx][idx_unique], 
        "stable"   : ["x" for _ in range(len(w_i[idx]))],
        "eigenval" : eigenval[idx][idx_unique]
        }

dic_order = pm.get_stabilisation(modal)
# pld.viz_stabilisation_diagram(modal, cmif, freq_cmif)
# selected_pole = [30,30,33,33,66,32,70,60,72,40,30,30,60]
# freq = np.array([18.8371850177749,  
#                                   40.132836645734095,
#                                   87.73661054907369,
#                                   89.67312969606662,
#                                   97.53280910557633,
#                                   105.18499940664444,
#                                   117.87999526512655,
#                                   125.18566897797119,
#                                   125.65863056930331,
#                                   130.0322772046283,
#                                   135.10023859504435,
#                                   143.17255674594142,
#                                   166.22251319140528])

selected_pole = [30,30]
freq = np.array([18.8371850177749, 40.132836645734095])
# idx_freq      = [0,   6, 13, 14, 18, 52, 52, 20, 67, 62, 33, 35, 46]
# selected_pole = [25, 33]
# idx_freq      = [0,   6]
lambda_pole = np.zeros(len(selected_pole), dtype=complex)
omega = np.zeros(len(selected_pole))
idx_freq = np.zeros(len(selected_pole), dtype=int)

for i in range(len(selected_pole)):
    # Vérifier les données pour ce pole
    stable_values = modal[selected_pole[i]]["stable"]
    print(f"Stable values for pole {i}: {stable_values}")

    # Trouver les indices des 'd'
    arg_stab = np.where(np.array(stable_values) == 'd')[0]

    if len(arg_stab) == 0:
        print(f"No stable poles found for pole {i}. Skipping...")
        continue

    print(f"Stable indices for pole {i}: {arg_stab}")

    # Extraire les valeurs correspondantes
    wn_stable = np.array(modal[selected_pole[i]]["wn"])[arg_stab]
    eigenval_stable = np.array(modal[selected_pole[i]]["eigenval"])[arg_stab]

    print(f"Stable natural frequencies for pole {i}: {wn_stable}")

    # Calculer la différence en fréquence
    freq_diff = np.abs(wn_stable / (2 * np.pi) - freq[i])
    idx = np.argmin(freq_diff)

    # Mise à jour des résultats
    lambda_pole[i] = eigenval_stable[idx]
    omega[i] = wn_stable[idx]
    idx_freq[i] = idx

    print(f"Selected stable eigenvalue for pole {i}: {lambda_pole[i]}")
    print(f"Selected natural frequency for pole {i}: {omega[i]}")

print("Final omega (Hz):", omega / (2 * np.pi))
# print("Freq",omega/2/np.pi)
# print("lambda pole",lambda_pole)
# print("idx_freq",idx_freq)


# lambda_pole = np.array([
#      (-0.41919674864349765-118.39096931287311j), (-1.0031165201872752-252.09465482162784j)
# ])


# def getModeShapes(freq, FRF_matrix):
#     # for i,freq in enumerate(self.eigenfreq) : 
#     #     print(f"Eigenfrequency {i+1} ")
#     #     self.q = np.arange(0, self.order[i] + 1, 1) 
#     #     freqs,damp,eiv = self.PolyMAX()

#     #     idx = np.argmin(np.abs(freqs - freq))
#     #     pole.append(self.poles[idx])
#     #     damping.append(damp[idx])
#     # print("Poles",pole)
#     # print("Damping",damping)
#     pole = [(-0.41919674864349765-118.39096931287311j), (-1.0031165201872752-252.09465482162784j)]


#     omega = freq * 2 * np.pi
#     modes = []
#     print("FRF_matrix", FRF_matrix.shape)
#     n,m,p = FRF_matrix.shape
#     print("len(FRF_matrix[0,:])", len(FRF_matrix[0]))
#     for i in range(n):
#         b = FRF_matrix[i,0,:]
#         A = []

#         for omeg in omega : 
#             if omeg == 0:
#                 omeg = 1e-6
#             A_bis = []
#             A_bis.append(1/(omeg**2))

#             for p in pole :  
#                 P = 1/(1j*omeg - p) + 1/(1j*omeg - np.conjugate(p))
#                 A_bis.append(P)
#                 Q = 1/(1j*omeg - p) - 1/(1j*omeg - np.conjugate(p))
#                 A_bis.append(Q)

#             A_bis.append(1)
#             A.append(A_bis)

#         A = np.array(A)


#         x = np.linalg.lstsq(A,b,rcond=None)[0]
#         x = x[1:-1]
#         residue  = x[::2] + 1j * x[1::2]
#         modes.append(residue)
#     print("shape A", A.shape)
#     print("Mode shape", np.array(modes).shape)
#     print("Mode shape", np.array(modes))
#     return np.array(modes)
# # a, lr, ur   = pm.compute_lsfd(lambda_pole, freq, H)
# # print("a",a.shape)
# # print("a1",a[0,0])
# mode = getModeShapes(freq,H)
# print("Mode shape", mode.shape)





# mode      = pm.extract_eigenmode(mode)
# abs_mode  = np.abs(mode)
# sign      = np.sign(np.cos(np.angle(mode)))
# real_mode = abs_mode * sign
# # print(mode.shape)
# for i in range(2):
#     mode_pole = real_mode[:,i]
#     rm.representation_mode(mode_pole,nbr = i, amplifactor=50)
# # rm.representation_mode(real_mode[:,0],nbr = 0)

