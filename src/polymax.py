import numpy as np
import extract_data as ed
from scipy import linalg
def get_polymax(H, freq,order_model,delta_t) :
    M = np.zeros((len(H[0])* (order_model + 1), len(H[0]) * (order_model +1)),dtype=object)
    for l in range(len(H)) :
        X_l = np.zeros((len(freq), order_model+1), dtype=complex)
        Y_l = []
        for n in range(len(freq)) :
            H_line_freq = ed.get_H_lineFunction_freq(l,n,H)
            # print("H_line freq \t",H_line_freq)
            for m in range(order_model + 1) :
                weigt_fct = 1
                X_l[n][m] = weigt_fct *(np.exp(1j * 2 * np.pi * freq[n] * m * delta_t))
            Y_l.append(np.kron( - X_l[n], H_line_freq))
        Y_l  = np.array(Y_l)
        Xl_H = np.conjugate(X_l).T
        Yl_H = np.conjugate(Y_l).T
        R_l  = np.real(Xl_H @ X_l)
        S_l  = np.real(Xl_H @ Y_l)
        T_l  = np.real(Yl_H @ Y_l)
        M += (T_l - S_l.T @ np.linalg.inv(R_l) @ S_l) 
    M *= 2
    M_11, M_12    = decomposition_M(M, H, order_model)
    alpha         = np.linalg.solve(M_11, -M_12)
    C             = get_C(alpha, order_model)
    eigenvals, x  = linalg.eig(C)
    eigenvals     = np.log(eigenvals) / delta_t
    w_i           = np.sqrt(np.real(eigenvals)**2 + np.imag(eigenvals)**2)
    damping_i     =  - np.real(eigenvals) / w_i
    arg_sorted    = np.argsort(w_i)
    # print(eigenvals[arg_sorted])
    w_i           = w_i[arg_sorted]
    damping_i     = damping_i[arg_sorted]
    return w_i, damping_i

def decomposition_M(M, H, p) :
    m = len(H[0])

    M_11 = M[:m*p, :m*p]  
    M_12 = M[:m*p, m*p:]  
    return np.array(M_11, dtype=float), np.array(M_12, dtype=float)

def get_C(alpha, order_model) :
    m = len(alpha[0]) 
    zero_upper    = np.zeros(((order_model - 1) * m, m))
    identity_uper = np.eye((order_model - 1) * m)
    upper_part    = np.hstack([zero_upper, identity_uper])
    lower_part    = np.zeros((m , m * order_model))
    for p in range(order_model) :
        alpha_p = alpha[m * p: m * (p + 1),:]
        lower_part[:, p * m:(p + 1) * m] = - alpha_p.T
    C = np.vstack([upper_part, lower_part])
    return C


def get_stabilisation(dic_order):
    # Convertir les clés en une liste triée pour garantir l'ordre
    sorted_keys = sorted(dic_order.keys())
    
    for i in range(len(sorted_keys) - 1):  # On boucle sur les clés triées sauf la dernière
        key = sorted_keys[i]
        next_key = sorted_keys[i + 1]
        
        dic_n = dic_order[key]
        dic_n_next = dic_order[next_key]
        
        for j, w in enumerate(dic_n["w_i"]):
            w_next = np.array(dic_n_next["w_i"])
            tol_low = (1 - 0.01) * w
            tol_high = (1 + 0.01 ) * w
            # print(tol_low,tol_high)
            # print(w_next)
            idx_w = np.where((w_next >= tol_low) & (w_next <= tol_high))[0]  # Utilisation correcte de numpy

            damping_n = dic_n["damping_i"][j]
            damping_next = np.array(dic_n_next["damping_i"])
            tol_low  = (1 - 0.05 ) * damping_n
            tol_high = (1 + 0.05) * damping_n
            idx_damp = np.where((damping_next >= tol_low) & (damping_next <= tol_high))[0]
            idx_stab = np.intersect1d(idx_damp, idx_w)
            idx_damp = np.setdiff1d(idx_damp, idx_stab)
            idx_w = np.setdiff1d(idx_w, idx_stab)
            # print("idx_stab",idx_stab)
            # print("idx_w",idx_w)
            # print("idx_damp",idx_damp)  
            for idx in idx_stab:
                # print("stable")
                dic_n_next["stable"][idx] = "d"
            for idx in idx_w:
                dic_n_next["stable"][idx] = "v"
            for idx in idx_damp:
                dic_n_next["stable"][idx] = "^"
        
        dic_order[next_key] = dic_n_next  # Mise à jour du dictionnaire avec les modifications
        # print(dic_order[next_key]["stable"])
    
    return dic_order

                


            
            

        