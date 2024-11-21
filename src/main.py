import extract_data as ed
import plot_data as pld
import first_data_fct as fdf
import interpolate_data as idf
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt

number_data = 1
name_data = f"../data/first_lab/DPsv0000{number_data}.mat"
name_set = f"set_{number_data}"
data = ed.extract_data(name_data)

# pld.bode_plot(data,name_set)
# pld.coherence_plot(data,name_set)
# pld.plot_exitasion_shock(data,name_set)
# pld.plot_time_shock(data,name_set)
# pld.plot_accelerometer_time(data,name_set)
cmif = fdf.compute_cmif(data)
pld.cmf_plot(data["G1_1"][:, 0], cmif,name_set)

freq = np.real(data["H1_2"][:, 0])

mask = (freq >= 18.4) & (freq <= 19.4)
# mask =(freq >= 15) & (freq <= 25)
freq_first_mode = freq[mask]
H1_2 = data["H1_2"][:, 1]
H1_2_first_mode = H1_2[mask]
cmif_first_mode = cmif[mask]

# H1_2_first_mode_abs = np.abs(H1_2_first_mode)


lin_freq, lin_H  = idf.compute_linear_interp(freq_first_mode, cmif_first_mode, 1000)
# cub_freq, cub_H  = idf.compute_cubic_spline(freq_first_mode, cmif_first_mode, 1000)

# damping_peak_picking_method = fdf.compute_peak_picking_method(lin_H, lin_freq, plot=True, set_name=name_set)
# print(f"Damping factor for pick picking method cubic: {damping_peak_picking_method}")
damping_circle_fit_method = fdf.compute_circle_fit_method(freq_first_mode, H1_2_first_mode, plot=True, set_name=name_set)
# print(f"Damping factor for circle fit method: {damping_circle_fit_method}")