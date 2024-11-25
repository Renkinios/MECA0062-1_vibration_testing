import extract_data as ed
import plot_data as pld
import first_data_fct as fdf
import interpolate_data as idf
import polymax as pm
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt

number_data = 1
name_data = f"../data/first_lab/DPsv0000{number_data}.mat"
name_set = f"set_{number_data}"
data = ed.extract_data(name_data)

# # pld.bode_plot(data,name_set)
# # pld.coherence_plot(data,name_set)
# # pld.plot_exitasion_shock(data,name_set)
# # pld.plot_time_shock(data,name_set)
# # pld.plot_accelerometer_time(data,name_set)
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
array1 = np.arange(1, 29) 
array2 = np.arange(31, 59) 
array3 = np.arange(61, 79)  


result_array = np.concatenate((array1, array2, array3))
# print(result_array)

H, freq = ed.extract_H_general(result_array)
delta_t           = 1.953 * 10**(-3) 
modal = {}

def remove_redundant(omega, pre = 1e-3) :
    bool_mat = np.ones(len(omega), dtype=bool)
    for i in range(len(omega)) :
        for j in range(i+1, len(omega)) :
            if np.abs(omega[i] - omega[j]) < pre :
                bool_mat[j] = False
    return bool_mat

for i in range(20,100) :
    w_i, damping_i = pm.get_polymax(H, freq, i, delta_t)
    idx = (w_i/2/np.pi >= 13) & (w_i/2/np.pi <= 200)
    # idx =  (w_i/2/np.pi <= 200)
    _, idx_unique = np.unique(w_i[idx], return_index=True)
    # idx_unique = remove_redundant(w_i[idx])
    modal[i] = {"w_i": w_i[idx][idx_unique], "damping_i": damping_i[idx][idx_unique], "stable" : ["x" for _ in range(len(w_i[idx]))]}
dic_order = pm.get_stabilisation(modal)
# print(dic_order[30]["stable"])
# print(dic_order[31]["stable"])
# print(dic_order[32]["stable"])
# print(dic_order[33]["stable"])
# print(dic_order[34]["stable"])


                

pld.viz_stabilisation_diagram(dic_order, cmif, freq_cmif)