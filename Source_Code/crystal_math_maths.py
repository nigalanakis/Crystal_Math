import itertools
import numpy as np
from scipy.spatial.distance import cdist

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
    
    return A_aligned, rmsd

def calculate_shared_d_value(plane_normals, shared_point):
    """
    Calculate the 'd' values for planes that pass through the same point.

    Parameters:
    plane_normals (list): The normal vectors of the planes.
    shared_point (list): A point known to be on all planes.

    Returns:
    list: The 'd' values for each plane.
    """
    d_values = []

    # The 'd' value for each plane can be calculated using the shared point.
    for normal in plane_normals:
        D = np.dot(normal, shared_point)
        d_values.append(D)

    return d_values

def center_of_mass(mass,pos):
    """
    Calculates and returns the center of mass for a configuration of atoms
    
    Parameters:
        mass (numpy.ndarray): an array with the mass of the atoms
        pos (numpy.ndarray): an array with the positions of the atoms
        
    Returns:
        The center of mass of the configuration
    """            
    return np.sum(mass[:,np.newaxis] * pos, axis = 0) / np.sum(mass)

def convert_seconds(seconds):
    """Convert a time duration in seconds to hours, minutes, and seconds."""
    hours = seconds // 3600  # Calculate whole hours
    seconds %= 3600  # Subtract the whole hours
    minutes = seconds // 60  # Calculate whole minutes
    seconds %= 60  # Subtract the whole minutes
    return hours, minutes, seconds

def crystal_h_matrix(cell_lengths,cell_angles,cell_volume):
    """ 
    Calculates and returns the coordinate transformation matrix from cartesian to fractinal 
    
    Parameters:
        cell_lengths (numpy.ndarray): The cell lengths of the unit cell
        cell_angles (numpy.ndarray): The cell angles of the unit cell
        cell_volume (float): The volume of the unit cell
        
    Returns:
        The transformation matrix from cartesian to fractinal 
    """
    # Set the individual cell lengths and angles in radians
    a, b, c = cell_lengths
    alpha, beta, gamma = cell_angles * np.pi / 180.0
    
    # Calculate trigomometric numbers for the angles
    cos_alpha = np.cos(alpha)
    cos_beta = np.cos(beta)
    cos_gamma = np.cos(gamma)
    sin_gamma = np.sin(gamma)
    
    return np.array([[   a, b * cos_gamma,                                       c * cos_beta],
                     [ 0.0, b * sin_gamma, c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma],
                     [ 0.0,           0.0,                    cell_volume / a / b / sin_gamma]])
    
def crystal_t_matrix(cell_lengths,cell_angles,cell_volume):
    """ 
    Calculates and returns the coordinate transformation matrix from fractinal to cartesian
    
    Parameters:
        cell_lengths (numpy.ndarray): The cell lengths of the unit cell
        cell_angles (numpy.ndarray): The cell angles of the unit cell
        cell_volume (float): The volume of the unit cell
        
    Returns:
        The transformation matrix from fractinal to cartesian 
    """
    # Set the individual cell lengths and angles in radians
    a, b, c = cell_lengths
    alpha, beta, gamma = cell_angles * np.pi / 180.0
    
    # Calculate trigomometric numbers for the angles
    cos_alpha = np.cos(alpha)
    cos_beta = np.cos(beta)
    cos_gamma = np.cos(gamma)
    sin_gamma = np.sin(gamma)
    
    return np.array([[1.0 / a, -cos_gamma / a / sin_gamma, b * c * (cos_alpha * cos_gamma - cos_beta) / cell_volume / sin_gamma],
                     [    0.0,        1.0 / b / sin_gamma, a * c * (cos_beta * cos_gamma - cos_alpha) / cell_volume / sin_gamma],
                     [    0.0,                        0.0,                                      a * b * sin_gamma / cell_volume]])

def define_cube_edges():
    """
    Define the edges of a unit cube, with vertices from (0,0,0) to (1,1,1).

    Returns:
    list of tuples: Each tuple contains two 3D points (x, y, z) representing the vertices of an edge.
    """
    # Vertices of a cube
    v = np.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0], [0,1,1], [0,0,1], [1,0,1], [1,1,1]])
    
    # Edges between vertices, defined by the indices of the vertices
    edges = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,5), (1,6), (2,7), (3,4)]
    
    # Define edges using vertices
    cube_edges = [(v[e[0]], v[e[1]]) for e in edges]

    return cube_edges

def ensure_right_handed_coordinate_system(vectors):
    """
    Ensure the eigenvectors form a right-handed coordinate system.

    Parameters:
        vectors (numpy.ndarray): 2D array where each column is an eigenvector.

    Returns:
        adjusted_vectors (numpy.ndarray): Eigenvectors adjusted to form a right-handed coordinate system.
    """
    # Compute the scalar triple product
    scalar_triple_product = np.dot(vectors[:, 0], np.cross(vectors[:, 1], vectors[:, 2]))
    
    # Check if the system is left-handed
    if scalar_triple_product < 0:
        # Switch the direction of the third eigenvector to make the system right-handed.
        vectors[:, 2] = -vectors[:, 2]
    
    return vectors

def find_vectors_closest_to_perpendicular(I, n_max):
    """
    For each vector v in I, find the vectors in valid_vectors closest to be perpendicular to v.

    Parameters:
    I (list): List of inquiry vectors.
    valid_vectors (list): List of valid vectors to check against.

    Returns:
    list: A list of tuples (v_i, [(w_i, a_i), (w_j, aj)]), where v_i is a vector from I, 
          and w_i, w_j are the vectors from valid_vectors that are closest to be 
          perpendicular to v_i, with a_i, a_j the respective angles.
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

def generate_proposed_eigenvectors(n_max):
    """
    Generate a list of 3D vectors with specific criteria.

    Parameters:
    n_max (int): The maximum absolute value for the vector components.

    Returns:
    list: A list of valid 3D vectors as tuples.
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

def inertia(mass,pos):
    """ 
    Calculates and returns the inertia tensor and inertia eigenvectors  for a configuration of atoms
    
    Parameters:
        mass (numpy.ndarray): The mass of the atoms 
        pos (numpy.ndarray): The positions of the atoms
        
    Returns:
        A tupple with the eigenvalues and eigenvectors of the inertia tensor
    """
    # Calculate the inertia tensor for the reference molecule
    inertia_tensor = -np.einsum('k,ki,kj->ij', mass, pos, pos, optimize=True)
    np.fill_diagonal(inertia_tensor, np.einsum('k,k->', mass, np.sum(pos**2, axis=1), optimize=True) + np.diag(inertia_tensor))
    
    return np.linalg.eig(inertia_tensor)

def intersect_plane_edge(plane_normal, d, edge):
    """
    Find the intersection of a plane with an edge (line).

    Parameters:
    plane_normal (array): The normal vector of the plane.
    d (float): The d parameter in the plane equation.
    edge (tuple): A tuple of two 3D points defining the edge.

    Returns:
    array: The intersection point (if it exists), otherwise None.
    """
    # Unpack the edge endpoints
    point1, point2 = edge

    # Line direction vector
    line_vector = np.subtract(point2, point1)

    # The plane equation is dot(n, (x, y, z)) = d, where n is the plane normal.
    # Solve the linear equation dot(n, (x, y, z)) = dot(n, point1) + t * dot(n, line_vector).
    numerator = d - np.dot(plane_normal, point1)
    denominator = np.dot(plane_normal, line_vector)

    # If the denominator is zero, the line is parallel to the plane (no intersection)
    if np.abs(denominator) < 1e-6:
        return None

    # Solve for t
    t = numerator / denominator

    # Find the intersection point
    intersection = point1 + t * line_vector

    return intersection

def kabsch_rotation_matrix(A, B):
    """
    Calculate the optimal rotation matrix to align structure A to structure B using Kabsch algorithm
    
    Parameters:
        A: Nx3 matrix of coordinates (N atoms, 3 dimensions) for molecule A
        B: Nx3 matrix of coordinates (N atoms, 3 dimensions) for molecule B
    
    Returns: 
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

def plane_cube_intersections(plane_normals, shared_point):
    """
    Find the intersections of planes with the edges of a unit cube.

    Parameters:
    plane_normals (list): The normal vectors of the planes.
    shared_point (list): A point known to be on all planes.

    Returns:
    dict: A dictionary where each key is a plane's index, and the value is a list of intersection points with the cube.
    """
    # Calculate 'd' values for the planes based on the shared point
    d_values = calculate_shared_d_value(plane_normals, shared_point)

    # Define the cube's edges
    cube_edges = define_cube_edges()

    # Dictionary to store the intersections for each plane
    intersections = {}

    # Check each plane
    for i, (normal, d) in enumerate(zip(plane_normals, d_values)):
        intersections[i] = []
        # Check each edge
        for edge in cube_edges:
            point = intersect_plane_edge(normal, d, edge)
            if point is not None:
                intersections[i].append(point.tolist())  # Store the points as lists for easy viewing

    return intersections

def rmsd(V, W):
    """
    Calculate the root-mean-square deviation between two sets of vectors V and W.
    """
    dr = V - W

    # Square every element in the difference matrix
    dr_sq = np.square(dr)

    # Sum along rows to get the squared distance for each pair of points, then sum the result and divide by the number of points.
    # Take the square root of the result to get the RMSD
    print(np.sqrt(np.sum(dr_sq) / len(V)))
    return np.sqrt(np.sum(dr_sq) / len(V))

def sort_eigenvectors(eigenvalues,eigenvectors):
    """
    Sort eigenvalues and their corresponding eigenvectors in ascending order.
    
    Parameters:
        eigenvalues (numpy.ndarray): 1D array containing the eigenvalues.
        eigenvectors (numpy.ndarray): 2D array where each column is an eigenvector.
        
    Returns:
        sorted_eigenvalues (numpy.ndarray): Eigenvalues sorted in ascending order.
        sorted_eigenvectors (numpy.ndarray): Eigenvectors sorted to correspond to sorted_eigenvalues.
    """
    # Get the indices that would sort eigenvalues in ascending order.
    idx = np.argsort(eigenvalues)
    
    # Use fancy indexing to reorder eigenvalues and eigenvectors.
    sorted_eigenvalues = eigenvalues[idx]
    sorted_eigenvectors = eigenvectors[:, idx]
    
    return sorted_eigenvalues, sorted_eigenvectors
