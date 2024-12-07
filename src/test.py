import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


from scipy.signal import find_peaks
from scipy.interpolate import interp1d,griddata
from scipy.linalg import svd
from matplotlib.patches import Arc



class DetailedAnalysis:
    def __init__(self,pmin,pmax,step):

        self.FRF_matrix_1 = []
        self.FRF_matrix_2 = []
        self.FRF_matrix_3 = []
        array1       = np.arange(1, 29) 
        array2       = np.arange(31, 59) 
        array3       = np.arange(61, 79)  
        result_array = np.concatenate((array1, array2, array3))
        for i, number_data in enumerate(result_array):
            number_data_set = str(number_data).zfill(5)
            name_data       = f"../data/sec_lab/DPsv{number_data_set}.mat"
            mat = scipy.io.loadmat(name_data)
            self.freq = mat['H1_2'][:, 0]
            self.FRF_matrix_1.append(mat['H1_2'][:, 1])
            self.FRF_matrix_2.append(mat['H1_3'][:, 1])
            self.FRF_matrix_3.append(mat['H1_4'][:, 1])

        self.FRF_matrix_1 = np.array(self.FRF_matrix_1)
        self.FRF_matrix_2 = np.array(self.FRF_matrix_2)
        self.FRF_matrix_3 = np.array(self.FRF_matrix_3)

        self.FRF_matrix = np.stack([self.FRF_matrix_1, self.FRF_matrix_2, self.FRF_matrix_3], axis=0)
        self.dt = 1/(200*2.56)
        self.ni = 3
        self.no = 74
           
    def WeightFunction(self):
        return 1

    def Omega(self,freq):
        f = []
        for q in self.q:
            f.append(np.exp(1j * 2 * np.pi * freq * q * self.dt)) 

        return f

    def getXY(self,no):
        self.X = []
        self.Y = []

        for (i,freq) in enumerate(self.freq):
            self.X.append(self.Omega(freq)*self.WeightFunction())
            self.Y.append(-np.kron(self.Omega(freq),self.FRF_matrix[:,no,i])*self.WeightFunction())   # inverse element in kron product

    def getR(self):
        return np.real( np.transpose(np.conjugate(self.X)) @ self.X )
    
    def getS(self):
        return np.real(np.transpose(np.conjugate(self.X)) @ self.Y)
    
    def getT(self):
        return np.real(np.transpose(np.conjugate(self.Y)) @ self.Y)

    def PolyMAX(self):

        M= np.zeros((self.ni*(len(self.q)),self.ni*(len(self.q))))

        for k in range(self.no):
            self.getXY(k)

            R = self.getR()
            S = self.getS()
            T = self.getT()

            M += 2 * (T - np.transpose(S) @ np.linalg.pinv(R) @ S)

        M = M[:-3,:]    
        b = -M[:,-3:]
        M = M[:,:-3]
        x = np.linalg.solve(M,b)

        alpha = [x[i:i+3] for i in range(0, len(x), 3)]
        alpha_t = [block.T for block in alpha]

        C11 = np.zeros((self.ni * (len(self.q) - 2),self.ni))
        C12 = np.eye(self.ni * (len(self.q) - 2)) 
        C22 = - np.hstack(alpha_t)
        C   = np.block([[C11, C12], [C22]])

        eig,eiv  = np.linalg.eig(C)
        eig_syst = np.log(eig)/self.dt

        sorted_indices = np.argsort(np.abs(eig_syst))
        self.poles     = eig_syst[sorted_indices]

        freq = np.abs(eig_syst[sorted_indices]) / (2 * np.pi)
        damp = -np.real(eig_syst[sorted_indices]) / freq / (2 * np.pi)

        return freq,damp,eiv
    
    def pullDoublet(self, freq, damp):
        freq_simplified = []
        damp_simplified = []
        aux = 0

        for i, f in enumerate(freq):
            if f == aux:
                continue
            else:
                freq_simplified.append(f)
                damp_simplified.append(damp[i])
            aux = f

        return freq_simplified, damp_simplified

    def CutVector(self, freq, damp):
        max_f = 180
        for i, f in enumerate(freq):
            if f > max_f:
                return freq[:i], damp[:i]

        return freq, damp
        
    def freqRangeInterest(self, freq, vector):
        freq_interest_indices = []
        min_val = freq * 0.99
        max_val = freq * 1.01

        for i, f in enumerate(vector):  
            if min_val < f < max_val:
                freq_interest_indices.append(i)

        return freq_interest_indices
        
    def StabilisationDiag(self, pmin,pmax,step):
        self.error_freq = 0.001  
        self.error_damp = 0.05

        self.previous_freq = []
        self.previous_damp = []

        plt.figure()

        legend_elements = {}

        for order in range(pmin, pmax + 1, step):
            print(f'Order: {order}')

            self.q = np.arange(0, order + 1, 1)  
            freq, damp, eiv = self.PolyMAX()  
            freq,damp = self.pullDoublet(freq,damp)
            freq, damp = self.CutVector(freq, damp)

            actual_freq = []
            actual_damp = []

            self.eigenfreq = []
            self.dampRatio = []

            if order == pmin:
                self.previous_freq.append(freq)
                self.previous_damp.append(damp)

                scatter = plt.scatter(freq, [order] * len(freq), s=0.2, marker='.', color='r',alpha = 0.5)#,zorder=1)
                if "Unstabilized" not in legend_elements:
                    legend_elements["Unstabilized"] = scatter

            else:
                for i, f in enumerate(freq):
                    prev_freq_indices = self.freqRangeInterest(f, self.previous_freq[-1])
                    is_stable = False

                    for j in prev_freq_indices:
                        f_prev = self.previous_freq[-1][j]
                        freq_error = np.abs(f - f_prev) / (f_prev)
                        damp_error = np.abs(damp[i] - self.previous_damp[-1][j])/(self.previous_damp[-1][j])

                        if np.abs(freq_error) < self.error_freq:
                            if np.abs(damp_error) < self.error_damp:
                                print(f'f: {f},damp: {damp[i]}')
                                self.eigenfreq.append(f)
                                #if order%2 == 0:
                                scatter = plt.scatter(f, order, s=3, marker = '^', color = 'b', linewidths=0.5)#, zorder=3)
                                if "Stabilized in freq & Damp" not in legend_elements:
                                    legend_elements["Stabilized in frequency & damping"] = scatter
                                is_stable = True

                            else:
                                if order%2 == 0:
                                    scatter = plt.scatter(f, order, s=3, marker='+', color='g',linewidths=0.5)#, zorder=2)
                                if "Stabilized in freq" not in legend_elements:
                                    legend_elements["Stabilized in frequency"] = scatter
                                is_stable = True

                    if not is_stable:
                        if order%2 == 0:
                            scatter = plt.scatter(f, order, s=0.2, marker='.', color='r',alpha = 0.5) #zorder = 1)
                        if "Unstabilized" not in legend_elements:
                            legend_elements["Unstabilized"] = scatter

                    actual_freq.append(f)
                    actual_damp.append(damp[i])

                self.previous_freq.append(actual_freq)
                self.previous_damp.append(actual_damp)
        plt.show()
        # print(f"Eigenfrequencies: {self.eigenfreq}")
        # print(f"Damping ratios  : {self.dampRatio}")

    # def getEigenFreq(self):
        # self.eigenfreq = np.array([18.8371850177749,  
        #                           40.132836645734095,
        #                           87.73661054907369,
        #                           89.67312969606662,
        #                           97.53280910557633,
        #                           105.18499940664444,
        #                           117.87999526512655,
        #                           125.18566897797119,
        #                           125.65863056930331,
        #                           130.0322772046283,
        #                           135.10023859504435,
        #                           143.17255674594142,
        #                           166.22251319140528])
        
    #     self.damp = np.array([0.003612491334125582 ,
    #                           0.003926233793499293 ,
    #                           0.005486188911881594 ,
    #                           0.003823801798063179 ,
    #                           0.001346197850725414,
    #                           0.0011911379135717084 ,
    #                           0.004985674566579669  ,
    #                           0.003578605774308652 ,
    #                           0.003975027890814003 ,
    #                           0.007723594100371476  ,
    #                           0.009070065681621156 ,
    #                           0.006713115395711651 ,
    #                           0.0014016992591514726])
        
    #     self.order = np.array([30 , 30 , 30 , 27 , 37 , 37 , 37 , 41 , 41 , 34 , 34 , 34 , 37])

    def getEigenFreq(self):
        self.eigenfreq = np.array([18.8371850177749,  
                                  40.132836645734095])
        
        self.damp = np.array([0.003612491334125582 ,
                              0.003926233793499293 ])
        
        self.order = np.array([30 , 30])

    def getModeShapes(self):

        self.getEigenFreq()  
        pole = []
        damping = []

        # for i,freq in enumerate(self.eigenfreq) : 
        #     print(f"Eigenfrequency {i+1} ")
        #     self.q = np.arange(0, self.order[i] + 1, 1) 
        #     freqs,damp,eiv = self.PolyMAX()

        #     idx = np.argmin(np.abs(freqs - freq))
        #     pole.append(self.poles[idx])
        #     damping.append(damp[idx])
        # print("Poles",pole)
        # print("Damping",damping)
        pole = [(-0.41919674864349765-118.39096931287311j), (-1.0031165201872752-252.09465482162784j)]


        omega = self.freq * 2 * np.pi
        modes = []
        print("self.FRF_matrix[0,:]",self.FRF_matrix[0,:])
        print("self.FRF_matrix",self.FRF_matrix.shape)

        for i in range(len(self.FRF_matrix[0,:])):
            b = self.FRF_matrix[0,i,:]
            A = []

            for omeg in omega : 
                if omeg == 0:
                    omeg = 1e-6
                A_bis = []
                A_bis.append(1/(omeg**2))

                for p in pole :  
                    P = 1/(1j*omeg - p) + 1/(1j*omeg - np.conjugate(p))
                    A_bis.append(P)
                    Q = 1/(1j*omeg - p) - 1/(1j*omeg - np.conjugate(p))
                    A_bis.append(Q)

                A_bis.append(1)
                A.append(A_bis)

            A = np.array(A)
    

            x = np.linalg.lstsq(A,b,rcond=None)[0]
            x = x[1:-1]
            residue  = x[::2] + 1j * x[1::2]
            modes.append(residue)
        print("shape A", A.shape)
        print("Mode shape", np.array(modes).shape)
        print("Mode shape", np.array(modes))
        return modes

    def computeRealEigenmodes(self,modeshape):
        real_modes = []
        for mode in modeshape:
            modulus = np.abs(mode)  
            phase = np.angle(mode)  
            real_mode = modulus * np.sign(np.cos(phase)) 
            real_modes.append(real_mode)
        return real_modes

    # def RepresentModes(self):
        modes = self.getModeShapes()
        # x, y, z = model.getLocation()  
        real_modes = self.computeRealEigenmodes(np.transpose(modes))  

        leftwing_indices = list(range(0, 21))  
        rightwing_indices = list(range(21, 42))
        lefthtail_indices = list(range(42, 51))
        righthtail_indices = list(range(51, 60))
        vtail_indices = list(range(60, 72))

        factors = [0.25, 0.25, 0.25, 0.25, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

        for mode_index, real_mode in enumerate(real_modes):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_facecolor('white')

            real_mode = real_mode / np.max(np.abs(real_mode))
            factor = factors[mode_index]

            x_deformed = []
            z_deformed = []

            ax.scatter(x, y, z, c='blue', label='undeformed model', s=2)

            for i in range(len(x)):
                if mode_index == 0 : 
                    real_mode[i] *= -1

                if i == 39:
                    ax.scatter(x[i], y[i], z[i] + factor * real_mode[i], c='r', s=2, label="numerical mode shape")
                    x_deformed.append(x[i])
                    z_deformed.append(z[i] + factor * real_mode[i])
                elif 60 <= i <= 71:  # Vertical tail
                    ax.scatter(x[i] + factor * real_mode[i], y[i], z[i], c='r', s=2)
                    x_deformed.append(x[i] + factor * real_mode[i])
                    z_deformed.append(z[i])
                elif i == 72:       # Left Engine
                    ax.scatter(x[i]+ factor * real_mode[i], y[i], z[i], c='r', s=2)
                    x_deformed.append(x[i])
                    z_deformed.append(z[i] - factor * real_mode[i])
                elif i == 73:       # Right Engine
                    ax.scatter(x[i] - factor * real_mode[i], y[i], z[i], c='r', s=2)
                    x_deformed.append(x[i])
                    z_deformed.append(z[i] - factor * real_mode[i])

                else:
                    ax.scatter(x[i], y[i], z[i] - factor * real_mode[i], c='r', s=2)
                    x_deformed.append(x[i])
                    z_deformed.append(z[i] - factor * real_mode[i])

            x_deformed = np.array(x_deformed)
            z_deformed = np.array(z_deformed)

            def plot_surface(indices, color):
                grid_x, grid_y = np.meshgrid(
                    np.linspace(min(x[indices]), max(x[indices]), 50),
                    np.linspace(min(y[indices]), max(y[indices]), 50)
                )
                surface = griddata(
                    (x_deformed[indices], y[indices]), z_deformed[indices],
                    (grid_x, grid_y), method='cubic'
                )
                if surface is not None:
                    ax.plot_trisurf(x_deformed[indices], y[indices], z_deformed[indices], color=color, alpha=0.7)

            plot_surface(leftwing_indices, 'red')
            plot_surface(rightwing_indices, 'red')
            plot_surface(lefthtail_indices, 'red')
            plot_surface(righthtail_indices, 'red')
            plot_surface(vtail_indices, 'red')

            plt.show()

def main():
    da = DetailedAnalysis(pmin = 20, pmax = 25, step = 1)
    # da.StabilisationDiag(20,30,1)
    da.getEigenFreq()
    modes = da.getModeShapes()

if __name__ == "__main__":
    main()

