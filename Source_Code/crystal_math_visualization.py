import matplotlib.pyplot as plt
import numpy as np 

""" Define CCDC standardized colors """
ccdc_colors = {"C": (0.5703,0.5703,0.5703),
               "H": (1.0000,1.0000,1.0000),
               "N": (0.5625,0.5625,1.0000),
               "O": (0.9414,0.0000,0.0000),
               "S": (1.0000,0.7813,0.1914),
               "F": (0.7617,1.0000,0.0000),
               "Cl":(0.1250,0.9414,0.1250),
               "Br":(0.7471,0.5117,0.2383)}

def fragment_eigenvectors_plot(fragment_atoms_species, fragment_atoms_mass, fragment_atoms_bv, inertia_eigenvectors, point=[0,0,0], limit=2):
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    axes_colors = ['blue','green','red']
    
    # Plot point.
    x, y, z = point
    ax.plot(x, y, z, marker='o', markersize=5, color='black')
    
    # Plot pnincipal axes of inertia and the respective perpendicular planes
    for axis, vector in enumerate(inertia_eigenvectors):
        d = -np.sum(vector * point)
        
        # Create a meshgrid:
        delta = 2
        xlim = point[0] - delta, point[0] + delta
        ylim = point[1] - delta, point[1] + delta
        xx, yy = np.meshgrid(np.arange(*xlim), np.arange(*ylim))
        
        # Solving the equation above for z:
        # z = -(a*x + b*y +d) / c
        zz = -(vector[0] * xx + vector[1] * yy + d) / vector[2]
        
        # Plot vector.
        ax.plot_surface(xx, yy, zz, alpha=0.25, color=axes_colors[axis])
        dx, dy, dz = delta * vector
        ax.quiver(0, 0, 0, dx, dy, dz, arrow_length_ratio=0.15, linewidth=1, color=axes_colors[axis])
        
        # Enforce equal axis aspects so that the normal also appears to be normal.
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        zlim = point[2] - delta, point[2] + delta
        ax.set_zlim(*zlim)
        
        # Label axes.
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
        ax.set_zlabel('Z', fontsize=10)
        
    # Plot the atoms    
    atom_colors = [ccdc_colors[species] for species in fragment_atoms_species]
    ax.scatter(fragment_atoms_bv[:,0],fragment_atoms_bv[:,1],fragment_atoms_bv[:,2],color=atom_colors,s=3*fragment_atoms_mass,edgecolors='black',linewidth=0.5,alpha=1)
    ax.set_xlim(-limit,limit)
    ax.set_ylim(-limit,limit)
    ax.set_zlim(-limit,limit)
    

