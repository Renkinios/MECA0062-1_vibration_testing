import extract_data as ed
import plot_data as pld
import first_data_fct as fdf
import interpolate_data as idf
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt

number_data = 2
name_data = f"../data/first_lab/DPsv0000{number_data}.mat"
name_set = f"set_{number_data}"
print("ici #######################",name_data)
data = ed.extract_data(name_data)

pld.bode_plot(data,name_set)
pld.coherence_plot(data,name_set)
pld.plot_exitasion_shock(data,name_set)
pld.plot_time_shock(data,name_set)



cmif = fdf.compute_cmif(data)
pld.cmf_plot(data["G1_1"][:, 0], cmif,name_set)

freq = np.real(data["H1_2"][:, 0])

# mask = (freq >= 18.4) & (freq <= 19.4)
mask =(freq >= 15) & (freq <= 25)
freq_first_mode = freq[mask]
H1_2 = data["H1_2"][:, 1]
H1_2_first_mode = H1_2[mask]

H1_2_first_mode_abs = np.abs(H1_2_first_mode)


cub_freq, cub_H  = idf.compute_cubic_spline(freq_first_mode, H1_2_first_mode_abs, 1000)

# lin_freq, lin_H  = idf.compute_linear_interp(freq_first_mode, H1_2_first_mode_abs, 1000)
# print(cub_freq)
# plt.figure(figsize=(8, 8))
# plt.scatter(np.real(cub_freq), np.imag(cub_freq), label="Diagramme de Nyquist", color="blue")
# plt.scatter(np.real(response[main_peak_index]), np.imag(response[main_peak_index]), color="red", label="Pic Principal")
# plt.xlabel("Reponse réelle", fontsize=15)
# plt.ylabel("Reponse imaginaire", fontsize=15)
# plt.legend()
# plt.show()
# lin_freq,  lin_H  = idf.compute_linear_interp(freq_first_mode, H1_2_first_mode, 1000)
# poly_freq, poly_H = idf.compute_polynomial_interp(freq_first_mode, H1_2_first_mode, 1000)
# quad_freq, quad_H = idf.compute_quadratic_spline(freq_first_mode, H1_2_first_mode, 1000)
# bspl_freq, bspl_H = idf.compute_b_spline(freq_first_mode, H1_2_first_mode, 1000)

# y_max = max(H1_2_first_mode)

# plt.figure(figsize=(10, 5))

# # Tracé des différentes courbes
# plt.plot(freq_first_mode, H1_2_first_mode, label="Original", linewidth=2)
# plt.plot(cub_freq, cub_H, label="Cubic Spline", linestyle='--')
# plt.plot(lin_freq, lin_H, label="Linear Interpolation", linestyle='-.')
# # plt.plot(poly_freq, poly_H, label="Polynomial Interpolation", linestyle=':')
# plt.plot(quad_freq, quad_H, label="Quadratic Spline", linestyle='-')
# plt.plot(bspl_freq, bspl_H, label="B-Spline", linestyle='-.')

# # Limite pour ne pas dépasser la valeur maximale de l'original
# # plt.ylim(0, y_max * 1.4)  # Une petite marge pour éviter l'écrasement

# # # Ajout de la légende et affichage
# plt.legend(loc='upper right')
# plt.xlabel("Fréquence")
# plt.ylabel("Amplitude")
# plt.title("Comparaison des Interpolations avec l'Original")
# plt.show()
# plt.close()
# Amplitude = np.abs(H1_2_first_mode)
# plt.figure(figsize=(10, 5))
# plt.semilogy(freq_first_mode, Amplitude)
# plt.show()
# damping_peak_picking_method = fdf.compute_peak_picking_method(cub_H, cub_freq, plot=True, set_name=name_set)
# print(f"Damping factor for pick picking method: {damping_peak_picking_method}")
damping_circle_fit_method = fdf.compute_circle_fit_method(freq_first_mode, H1_2_first_mode, plot=True, set_name=name_set)
print(f"Damping factor for circle fit method: {damping_circle_fit_method}")