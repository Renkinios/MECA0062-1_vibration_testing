import numpy as np
import matplotlib.pyplot as plt 
import extract_data as ed
import numpy as np

def representation_mode(real_mode, nbr = 1, amplifactor =20):
    nodes = ed.extract_node_shock("../data/node_structure.csv")
    nodes = nodes.to_numpy()
    def generate_elements(start, end, step, offset):
        return [[i, i + offset] for i in range(start, end, step)]
    elements = []
    # Wing right
    elements.extend(generate_elements(0, 28, 2, 2))
    elements.extend(generate_elements(1, 29, 2, 2))
    elements.extend(generate_elements(0, 29, 2, 1))
    # Wing left
    elements.extend(generate_elements(30, 58, 2, 2))
    elements.extend(generate_elements(31, 59, 2, 2))
    elements.extend(generate_elements(30, 59, 2, 1))

    # Assemblage wing
    elements.extend([[28, 58]])
    elements.extend([[29, 59]])
    # # # Tail horizontaux right
    elements.extend(generate_elements(60, 66, 2, 2))
    elements.extend(generate_elements(61, 67, 2, 2))
    elements.extend(generate_elements(60, 67, 2, 1))
    # # # Tail horizontaux left
    elements.extend(generate_elements(68, 74, 2, 2))
    elements.extend(generate_elements(69, 75, 2, 2))
    elements.extend(generate_elements(68, 75, 2, 1))
    # # # # Verical tail
    elements.extend(generate_elements(76, 82, 2, 2))
    elements.extend(generate_elements(77, 83, 2, 2))
    elements.extend(generate_elements(76, 83, 2, 1))

    # Création du graphique 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i in elements :
        node1, node2 = i
        x_coords = [nodes[node1][0], nodes[node2][0]]
        y_coords = [nodes[node1][1], nodes[node2][1]]
        z_coords = [nodes[node1][2], nodes[node2][2]]
        ax.plot(x_coords, y_coords, z_coords, c='black')
    

    real_node = nodes.astype(float)

    max_real             = np.max(real_mode)
    real_node[0,2]      += real_mode[0]    *amplifactor/max_real
    real_node[1:28 ,2]  += real_mode[1:28] *amplifactor/max_real
    real_node[30:58,2]  += real_mode[28:56]*amplifactor/max_real
    real_node[60:66,2]  += real_mode[56:62]*amplifactor/max_real
    real_node[68:74,2]  += real_mode[62:68]*amplifactor/max_real
    real_node[76:82,0]  += real_mode[68:74]*amplifactor/max_real



    for i in elements :
        node1, node2 = i
        x_coords = [real_node[node1][0], real_node[node2][0]]
        y_coords = [real_node[node1][1], real_node[node2][1]]
        z_coords = [real_node[node1][2], real_node[node2][2]]
        ax.plot(x_coords, y_coords, z_coords, c='red')

    ax.scatter(real_node[:, 0], real_node[:, 1], real_node[:, 2], c='red')
    ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:,2], c='black')



   
    ax.set_title(f"3D Plot of Wing and Tail Nodes {nbr}", fontsize=14)
    ax.set_xlabel("X-axis (mm)")
    ax.set_ylabel("Y-axis (mm)")
    ax.set_zlabel("Z-axis (mm)")

    file_name = f"../figures/sec_lab/mode_test/mode_{nbr}.pdf"
    # plt.savefig (file_name, dpi=300,bbox_inches='tight')
    plt.show()
