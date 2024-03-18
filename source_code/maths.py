import itertools 
import numpy as np 
from scipy.spatial.distance import cdist

### COORDINATE OPERATIONS #####################################################
def kabsch_rotation_matrix(A, B):
    """
    Calculate the optimal rotation matrix to align structure A to structure B 
        using Kabsch algorithm
    
    Parameters
    ----------
    A: Nx3 matrix of coordinates (N atoms, 3 dimensions) for molecule A
    B: Nx3 matrix of coordinates (N atoms, 3 dimensions) for molecule B
    
    Returns
    -------
    R: 3x3 optimal rotation matrix
    """

    # Step 1: Centroid and Covariance
    # Centroid already at origin, so we skip to covariance matrix
    H = A.T @ B

    # Step 2: Singular value decomposition (SVD)
    V, S, Wt = np.linalg.svd(H)

    # Step 3: Check for reflection and calculate rotation matrix
    d = (np.linalg.det(V) * np.linalg.det(Wt)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    # Create optimal rotation matrix
    R = V @ Wt

    return R

def align_structures(R, A, B):
    """
    Aligns structure A to structure B and calculates RMSD
    
    Pamameters:
        R: 3x3 optimal rotation matrix
        A: Nx3 matrix of coordinates for molecule A
        B: Nx3 matrix of coordinates for molecule B
    Returns: 
        R, aligned A, RMSD
    """
    A_aligned = A @ R  # Apply the rotation matrix to A to align it with B
    rmsd = np.sqrt(np.mean(np.sum((A_aligned - B)**2, axis=1)))  # Calculate RMSD
    
    return rmsd

def cartesian_to_spherical(vector):
    """
    Convert a vector from Cartesian coordinates (x, y, z) to spherical 
    coordinates (r, theta, phi)
    
    Parameters
    ----------
    vector : ndarray 
        A NumPy array containing the x, y, and z coordinates of the vector.
    
    Returns
    -------
    ndarray 
        A NumPy array containing the spherical coordinates (r, theta, phi),
        where r is the radius, theta is the polar angle (in radians), and
        phi is the azimuthal angle (in radians).
    """
    # Convert the input list or tuple to a NumPy array if it's not already
    vector = np.asarray(vector)
    
    # Compute the radial distance
    r = np.linalg.norm(vector)
    
    # Compute the polar angle (theta) - angle from the z-axis
    # Guard against the r being zero to avoid division by zero
    theta = np.arccos(vector[2] / r) if r != 0 else 0
    
    # Compute the azimuthal angle (phi) - angle from the x-axis in the xy-plane
    phi = np.arctan2(vector[1], vector[0])
    
    return np.array([r, np.degrees(theta), np.degrees(phi)])

### DISTANCE OPERATIONS #######################################################
def distance_to_plane(point,plane_normal,plane_point,normal=False):
    """
    Calculate the distance from a point to a plane defined by
    a normal vector and a point on the plane.
    """
    if normal:
        d =  np.abs(np.dot(plane_normal, point - plane_point)) 
    else:
        d =  np.abs(np.dot(plane_normal, point - plane_point)) / np.linalg.norm(plane_normal)
    return d

def distance_to_zzp_planes_family(point,plane_normal,plane_norm):
    """ 
    Calculate the distance from a point to a family of zzp planes defined by a normal vector
    """
    distance = distance_to_plane(point,plane_normal,np.array([0.0,0.0,0.0])) % (0.25 / plane_norm)
    if distance > 0.125 / plane_norm:
        distance = 0.25 / plane_norm - distance
        
    return distance

### PHYSICAL PROPERTIES OPERATIONS ############################################
def calculate_inertia(mass,pos):
    """ 
    Calculates and returns the inertia tensor and inertia eigenvectors  for a 
        configuration of atoms
    
    Parameters
    ----------
    mass : numpy.ndarray
        The mass of the atoms 
    pos : numpy.ndarray
        The positions of the atoms
        
    Returns
    -------
    A tupple with the eigenvalues and eigenvectors of the inertia tensor
    """
    # Calculate the inertia tensor for the reference molecule
    inertia_tensor = -np.einsum('k,ki,kj->ij', mass, pos, pos, optimize=True)
    np.fill_diagonal(inertia_tensor, np.einsum('k,k->', mass, np.sum(pos**2, axis=1), optimize=True) + np.diag(inertia_tensor))
    
    return np.linalg.eig(inertia_tensor)

def ensure_right_handed_coordinate_system(vectors):
    """
    Ensure the eigenvectors form a right-handed coordinate system.

    Parameters
    ----------
    vectors : numpy.ndarray 
        2D array where each column is an eigenvector.

    Returns
    -------
    adjusted_vectors : numpy.ndarray
        Eigenvectors adjusted to form a right-handed coordinate system.
    """
    # Compute the scalar triple product
    scalar_triple_product = np.dot(vectors[:, 0], np.cross(vectors[:, 1], vectors[:, 2]))
    
    # Check if the system is left-handed
    if scalar_triple_product < 0:
        # Switch the direction of the third eigenvector to make the system right-handed.
        vectors[:, 2] = -vectors[:, 2]
    
    return vectors

def sort_eigenvectors(eigenvalues,eigenvectors):
    """
    Sort eigenvalues and their corresponding eigenvectors in ascending order.
    
    Parameters:
        eigenvalues (numpy.ndarray): 1D array containing the eigenvalues.
        eigenvectors (numpy.ndarray): 2D array where each column is an 
            eigenvector.
        
    Returns:
        sorted_eigenvalues (numpy.ndarray): Eigenvalues sorted in ascending 
            order.
        sorted_eigenvectors (numpy.ndarray): Eigenvectors sorted to correspond 
            to sorted_eigenvalues.
    """
    # Get the indices that would sort eigenvalues in ascending order.
    idx = np.argsort(eigenvalues)
    
    # Use fancy indexing to reorder eigenvalues and eigenvectors.
    sorted_eigenvalues = eigenvalues[idx]
    sorted_eigenvectors = eigenvectors[:, idx]
    
    return sorted_eigenvalues, sorted_eigenvectors

def center_of_mass(mass,pos):
    '''
    Calculates and returns the center of mass for a configuration of atoms
    
    Parameters
    ----------
    mass : numpy.ndarray 
        An array with the mass of the atoms
    pos : numpy.ndarray
        An array with the positions of the atoms
        
    Returns
    -------
    numpy.ndarray
        The center of mass of the configuration (3)
    '''            
    return np.sum(mass[:,np.newaxis] * pos, axis = 0) / np.sum(mass)

### TOPOLOGICAL PROPERTIES OPERATIONS #########################################
def set_zzp_planes():
    '''
    Sets and returns the ZZP planes

    Returns
    -------
    zzp_planes : list
        The family of zzp planes in the unit cell.
    '''
    zzp_planes = ((np.array([1, 0, 0]),1.),
                  (np.array([0, 1, 0]),1.),
                  (np.array([0, 0, 1]),1.),
                  (np.array([1, 1, 0]),np.sqrt(2.)),
                  (np.array([1,-1, 0]),np.sqrt(2.)),
                  (np.array([1, 0, 1]),np.sqrt(2.)),
                  (np.array([1, 0,-1]),np.sqrt(2.)),
                  (np.array([0, 1, 1]),np.sqrt(2.)),
                  (np.array([0, 1,-1]),np.sqrt(2.)))
    
    return zzp_planes

def generate_proposed_eigenvectors(n_max):
    """
    Generate a list of 3D vectors with specific criteria.

    Parameters
    ----------
    n_max : int
        The maximum absolute value for the vector components.

    Returns
    -------
    A list of valid 3D vectors as tuples.
    """
    
    # Initialize a list to hold the valid vectors
    proposed_eigenvectors = []

    # Create all combinations of vector components within the range [-n_max, n_max] for a 3D vector
    # Note: we adjust the range for the first component to [0, n_max] to satisfy your second condition
    alternating_range = [0] + [val for i in range(1, n_max + 1) for val in (i, -i)] 
    for combination in itertools.product(range(0,n_max+1), alternating_range, alternating_range):
        # Unpack the combination into the individual components
        x, y, z = combination

        # Check if one and only one of the components is zero (condition 3)
        if [x, y, z].count(0) >= 1 and [x, y, z].count(0) <= 2:
            # Create a vector from the components
            vector = (x, y, z)

            # Check for parallel vectors
            # A new vector is parallel to an existing vector if their cross product is the zero vector
            is_parallel = False
            for valid_vector in proposed_eigenvectors:
                cross_product = (valid_vector[1]*vector[2] - valid_vector[2]*vector[1], 
                                 valid_vector[2]*vector[0] - valid_vector[0]*vector[2], 
                                 valid_vector[0]*vector[1] - valid_vector[1]*vector[0])

                if cross_product == (0, 0, 0):
                    is_parallel = True
                    break  # No need to check further, move to the next combination

            # If the vector is not parallel to any vector in the list, we add it to our valid vectors
            if not is_parallel:
                proposed_eigenvectors.append(vector)
    return proposed_eigenvectors

def vectors_closest_to_perpendicular(I, n_max):
    """
    For each vector v in I, find the vectors in valid_vectors closest to be
    perpendicular to v.

    Parameters
    ----------
    I : list)
        List of inquiry vectors.
    valid_vectors : list
        List of valid vectors to check against.

    Returns:
    A list of tuples (v_i, [(w_i, a_i), (w_j, aj)]), where v_i is a vector 
    from I, and w_i, w_j are the vectors from valid_vectors that are closest to 
    be perpendicular to v_i, with a_i, a_j the respective angles.
    """
    # Generate the set of the proposed vectors
    proposed_vectors = generate_proposed_eigenvectors(n_max)
    
    # Convert the list of vectors into numpy arrays for easier computation
    I_array = np.array(I, dtype=np.float64)  # ensure floating point precision
    valid_vectors_array = np.array(proposed_vectors, dtype=np.float64)  # same here

    # Normalize the vectors, since we're interested in the angle between them.
    # This normalization step is crucial to ensure that the dot product only measures the angle between vectors.
    I_norms = np.linalg.norm(I_array, axis=1, keepdims=True)
    valid_vectors_norms = np.linalg.norm(valid_vectors_array, axis=1, keepdims=True)

    # To avoid division by zero, we will use np.divide which can handle these cases gracefully.
    # 'out' is used to specify the array where the result is stored. If division by zero occurs, it will be replaced by zero.
    I_array = np.divide(I_array, I_norms, out=np.zeros_like(I_array), where=I_norms!=0)
    valid_vectors_array = np.divide(valid_vectors_array, valid_vectors_norms, out=np.zeros_like(valid_vectors_array), where=valid_vectors_norms!=0)

    # Compute the cosine distances between each pair of vectors in I and valid_vectors.
    sin_distances = cdist(I_array, valid_vectors_array, metric='cosine') - 1.0

    # Find the index of the vector in valid_vectors that forms the smallest angle with each vector in I.
    # closest_to_perpendicular_indices = np.argmin(np.abs(sin_distances), axis=1)
    closest_to_perpendicular_indices = np.argsort(np.abs(sin_distances), axis=1)[:,:2]

    # Prepare a list to store the pairs of vectors with the minimum angle
    closest_to_perpendicular_vectors = []
    
    for i, indices in enumerate(closest_to_perpendicular_indices):
        v_i = I[i]
        results_for_v = []
        
        for index in indices:
            w_i = proposed_vectors[index]

            # Since we used cosine, we convert it back to the angle. The cosine of the angle between the vectors is the dot product
            # because we normalized the vectors.
            cos_similarity = sin_distances[i, index]
            angle = np.arccos(cos_similarity) * 180.0 / np.pi  # converting to degrees from radians

            results_for_v.append((w_i, angle))

        # Append the pair of closest vectors along with their angles relative to v_i
        closest_to_perpendicular_vectors.append((v_i, results_for_v))
    return closest_to_perpendicular_vectors

def get_reference_cell_points(min_value,max_value,step):
    '''
    Returns a list of the reference cell points for the calculation of the 
    distances of the principal inertia planes to the points

    Parameters
    ----------
    min_value : float
        The minimum coordinate in fractional coordinates
    max_value : float
        The maximum coordinate in fractional coordinates
    step : float
        The step fractional coordinates
    
    Returns
    -------
    list
        The reference cell points for the calculation of the 
        distances of the principal inertia planes to the points

    '''
    c_list = np.arange(min_value, max_value, step)
    return [[u, v, w] for u in  c_list for v in c_list for w in c_list]




