import numpy as np
import matplotlib.pyplot as plt 

import numpy as np

def representation_mode(real_mode, nbr = 1, amplifactor =20):
    start_wing_x, start_wing_y, start_wing_z = 0, 750, 0
    start_horizontaltail_tail_x, start_horizontaltail_tail_y, start_horizontaltail_tail_z = 50, 0, 75
    start_vertical_tail_x, start_vertical_tail_y, start_vertical_tail_z = 0, 0, 150

    node_right_wing = np.array([
        [start_wing_x + 750 - i * 50, start_wing_y + 100 - j * 100, start_wing_z]
        for i in range(15)
        for j in range(2)
    ])

    node_left_wing = np.array([
        [start_wing_x - 750 + i * 50, (start_wing_y + 100 -j * 100), start_wing_z]
        for i in range(15)
        for j in range(2)
    ])

    node_right_horizontal_tail = np.array([
        [start_horizontaltail_tail_x + 150 - i * 50, start_horizontaltail_tail_y +100 - j * 100, start_horizontaltail_tail_z]
        for i in range(4)
        for j in range(2)
    ])

    node_left_horizontal_tail = np.array([
        [- start_horizontaltail_tail_x - 150 + i * 50, (start_horizontaltail_tail_y +100 - j * 100), start_horizontaltail_tail_z]
        for i in range(4)
        for j in range(2)
    ])

    node_vertical_tail = np.array([
        [start_vertical_tail_x, start_vertical_tail_y +100 - j * 100, start_vertical_tail_z + 150 - 50 * i]
        for i in range(4)
        for j in range(2)
    ])

    # print("horizontal tail left \n",node_left_horizontal_tail)
    # print("horizontal tail right \n",node_right_horizontal_tail)


    # print("vertical tail \n ",node_vertical_tail)
    # print("wing right\n",node_right_wing.shape)
    # print("wing left\n",node_left_wing)

    nodes = np.concatenate([node_right_wing, node_left_wing, node_right_horizontal_tail, node_left_horizontal_tail, node_vertical_tail])
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
    '''
    # Wing right
    elements.extend(generate_elements(0, 14, 2, 2))
    elements.extend(generate_elements(1, 15, 2, 2))
    elements.extend(generate_elements(0, 15, 2, 1))
    # Wing left
    elements.extend(generate_elements(16, 30, 2, 2))
    elements.extend(generate_elements(17, 31, 2, 2))
    elements.extend(generate_elements(16, 31, 2, 1))
    # # Tail horizontaux right
    elements.extend(generate_elements(32, 38, 2, 2))
    elements.extend(generate_elements(33, 39, 2, 2))
    elements.extend(generate_elements(32, 39, 2, 1))
    # # Tail horizontaux left
    elements.extend(generate_elements(40, 46, 2, 2))
    elements.extend(generate_elements(41, 47, 2, 2))
    elements.extend(generate_elements(40, 47, 2, 1))
    # # # Verical tail
    elements.extend(generate_elements(48, 54, 2, 2))
    elements.extend(generate_elements(49, 55, 2, 2))
    elements.extend(generate_elements(48, 55, 2, 1))
    '''

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

    # real_node[0,2]      -= real_mode[0]*20
    # real_node[1:14 ,2]  += real_mode[1:14]*20
    # real_node[16:30,2]  += real_mode[14:28]*20
    # real_node[32:38,2]  += real_mode[28:34]*20
    # real_node[40:46,2]  += real_mode[34:40]*20
    # real_node[48:54,0]  -= real_mode[40:46]*20
    max_real = np.max(real_mode)
    real_node[0,2]      -= real_mode[0]    *amplifactor/max_real
    real_node[1:28 ,2]  += real_mode[1:28] *amplifactor/max_real
    real_node[30:58,2]  += real_mode[28:56]*amplifactor/max_real
    real_node[60:66,2]  += real_mode[56:62]*amplifactor/max_real
    real_node[68:74,2]  += real_mode[62:68]*amplifactor/max_real
    real_node[76:82,0]  -= real_mode[68:74]*amplifactor/max_real

    # test_x = np.zeros(real_mode.shape[0])
    # test_x[0]      -= real_mode[0]    *amplifactor
    # test_x[1 :28]  += real_mode[1 :28] *amplifactor
    # test_x[28:56]  += real_mode[28:56]*amplifactor
    # test_x[56:62]  += real_mode[56:62]*amplifactor
    # test_x[62:68]  += real_mode[62:68]*amplifactor
    # test_x[68:74]  -= real_mode[68:74]*amplifactor
    # print("test_x",test_x)
    



    for i in elements :
        node1, node2 = i
        x_coords = [real_node[node1][0], real_node[node2][0]]
        y_coords = [real_node[node1][1], real_node[node2][1]]
        z_coords = [real_node[node1][2], real_node[node2][2]]
        ax.plot(x_coords, y_coords, z_coords, c='red')

    ax.scatter(real_node[:, 0], real_node[:, 1], real_node[:, 2], c='red')
    # Ailes
    # ax.scatter(real_node[:14, 0], real_node[:14, 1], real_node[:14, 2], c='red')
    # ax.scatter(real_node[14:28, 0], real_node[14:28, 1], real_node[14:28, 2], c='red')

    # # Stabilisateurs horizontaux
    # ax.scatter(real_node[28:34, 0], real_node[28:34,1], real_node[28:34,2], c='red')
    # ax.scatter(real_node[34:40, 0], real_node[34:40, 1], real_node[34:40, 2], c='red')

    # # Stabilisateur vertical
    # ax.scatter(real_node[40:, 0], real_node[40:, 1], real_node[40:, 2], c='red')
    ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:,2], c='black')



    # Légendes et labels
    ax.set_title("3D Plot of Wing and Tail Nodes", fontsize=14)
    ax.set_xlabel("X-axis (mm)")
    ax.set_ylabel("Y-axis (mm)")
    ax.set_zlabel("Z-axis (mm)")
    # ax.legend()
    file_name = f"../figures/sec_lab/mode_test/mode_{nbr}.pdf"
    # plt.savefig (file_name, dpi=300,bbox_inches='tight')
    plt.show()

# representation_mode(0)