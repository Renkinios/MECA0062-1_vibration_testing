import extract_data as ed
import plot_data as pld
import first_data_fct as fdf
import interpolate_data as idf
import representation_mode as rm
import comparaison_method as cm
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

# array1       = np.arange(1, 14) 
# array2       = np.arange(15, 29) 

# array3       = np.arange(31, 59) 
# array4       = np.arange(61, 79)  
# result_array = np.concatenate((array1, array2, array3, array4))

H, freq = ed.extract_H_general(result_array)
delta_t = 1.9531 * 10**(-3) 
modal   = {}

# for i in tqdm(range(29,74), desc="Polymax poles", unit="Poles"):
# # for i in range(20,100):
#     w_i, damping_i, eigenval = pm.get_polymax(H, freq, i, delta_t)

#     idx            = (w_i/2/np.pi >= 13) & (w_i/2/np.pi <= 180)
#     _, idx_unique  = np.unique(w_i[idx], return_index=True)
#     modal[i]       = {
#         "wn"       : w_i[idx][idx_unique], 
#         "zeta"     : damping_i[idx][idx_unique], 
#         "stable"   : ["x" for _ in range(len(w_i[idx]))],
#         "eigenval" : eigenval[idx][idx_unique]
#         }

# dic_order = pm.get_stabilisation(modal)
# pld.viz_stabilisation_diagram(modal, cmif, freq_cmif)
# selected_pole = [30,30,33,33,55,32,40,63,55,53,30,30,37]
# freq_pole = np.array([18.8371850177749, 40.132836645734095, 87.73661054907369,
# 89.67312969606662, 97.53280910557633, 105.18499940664444, 117.87999526512655,
# 125.18566897797119,125.65863056930331,130.0322772046283,
# 135.10023859504435,143.17255674594142,166.22251319140528])

# lambda_pole = np.zeros(len(selected_pole), dtype=complex)
# omega = np.zeros(len(selected_pole))
# idx_freq = np.zeros(len(selected_pole), dtype=int)

# for i in range(len(selected_pole)):
#     # Vérifier les données pour ce pole
#     stable_values = modal[selected_pole[i]]["stable"]

#     # Trouver les indices des 'd'
#     arg_stab = np.where(np.array(stable_values) == 'd')[0]

#     if len(arg_stab) == 0:
#         print(f"No stable poles found for pole {i}. Skipping...")
#         continue

#     # Extraire les valeurs correspondantes
#     wn_stable = np.array(modal[selected_pole[i]]["wn"])[arg_stab]
#     eigenval_stable = np.array(modal[selected_pole[i]]["eigenval"])[arg_stab]


#     # Calculer la différence en fréquence
#     freq_diff = np.abs(wn_stable / (2 * np.pi) - freq_pole[i])
#     idx = np.argmin(freq_diff)

#     # Mise à jour des résultats
#     lambda_pole[i] = eigenval_stable[idx]
#     omega[i] = wn_stable[idx]
#     idx_freq[i] = arg_stab[idx]

# print("Final omega (Hz):", omega / (2 * np.pi))

# print("lambda pole",lambda_pole)
# print("idx_freq",idx_freq)


# final_omega_hz = np.array([
#     18.84262616, 40.12242782, 87.85281287, 89.66141477, 143.33713324,
#     105.19967891, 135.26689864, 125.57281791, 166.10382907, 135.28144599,
#     135.2589153, 143.37188261, 169.56553806
# ])

lambda_pole = np.array([
    -0.41919761 - 118.39096967j, -1.00311684 - 252.09465319j,
    -2.95548289 + 551.98759087j, -1.9042592 + 563.35606549j,
    -0.63729897 - 610.02198244j, -0.31541393 + 660.9890016j,
    -6.21792925 - 741.42633958j, -2.01626186 + 786.73010206j,
    -2.85199809 + 789.49204795j, -5.103421 - 817.43014699j,
    -9.25476512 - 849.80643652j, -5.9185963 - 900.81266308j,
    -1.45473844 + 1044.76438362j
])

        


a = pm.compute_lsfd(lambda_pole, freq, H)

mode      = pm.extract_eigenmode(a)
# print(mode)
abs_mode  = np.abs(mode)
sign      = np.sign(np.cos(np.angle(mode)))
real_mode = abs_mode * sign

for i in range(real_mode.shape[0]):
    real_mode[i] = real_mode[i] / np.max(np.abs(real_mode[i]))
    real_mode[i,0]    *= 1
    real_mode[i,1:28] *=-1
    real_mode[i,28:56]*=-1
    real_mode[i,56:62]*= 1
    real_mode[i,62:68]*= 1
    real_mode[i,68:74]*=-1

# # print(mode.shape)
# for i in range(real_mode.shape[1]):
#     mode_pole = real_mode[i]
#     print("mode_pole", mode_pole.shape)
#     rm.representation_mode(mode_pole,nbr = i, amplifactor=50)

for i in range(real_mode.shape[0]):
    rm.representation_mode(real_mode[i],nbr = i, amplifactor=50)

mode_samcef = ed.extract_samcef_shock()
MAC         = cm.get_modal_assurance_criterion(mode_samcef, real_mode)
auto_MAC    = cm.get_auto_MAC(real_mode)
pld.viz_MAC(MAC)
pld.viz_auto_MAC(auto_MAC)






