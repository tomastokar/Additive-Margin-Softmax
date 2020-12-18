import numpy as np
import matplotlib.pyplot as plt
# from matplotlib import cm, colors
# from mpl_toolkits.mplot3d import Axes3D


def sphere_plot(embeddings, labels, figure_path=None):

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    
    r = 1
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
    x = r*sin(phi)*cos(theta)
    y = r*sin(phi)*sin(theta)
    z = r*cos(phi)

    ax.plot_surface(
        x, y, z,  
        rstride=1, 
        cstride=1, 
        color='w', 
        alpha=0.3, 
        linewidth=0
    )

    ax.scatter(
        embeddings[:,0], 
        embeddings[:,1], 
        embeddings[:,2], 
        c = labels, 
        s = 20
    )

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_aspect("equal")
    plt.tight_layout()
    if figure_path is not None:
        plt.savefig(figure_path)
    else:
        plt.show()