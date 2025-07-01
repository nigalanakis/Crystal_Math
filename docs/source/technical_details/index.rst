**Pipeline Orchestrator** (``CrystalAnalyzer``)
   - Coordinates execution of the five-stage pipeline
   - Manages resource allocation and error handling
   - Provides progress monitoring and checkpointing

**Configuration Management** (``ExtractionConfig``)
   - Validates and normalizes configuration parameters
   - Handles parameter inheritance and defaults
   - Supports environment-specific overrides

**Data Extraction Pipeline**
   - Stage 1: CSD querying and family extraction
   - Stage 2: Packing similarity clustering
   - Stage 3: Representative structure selection
   - Stage 4: Atomic data extraction and storage
   - Stage 5: Feature engineering and descriptor calculation

**Storage Layer** (HDF5-based)
   - Variable-length datasets for ragged arrays
   - Efficient compression and chunking
   - Metadata management and versioning

Performance Architecture
~~~~~~~~~~~~~~~~~~~~~~~~

**GPU Acceleration Strategy**

.. code-block:: text

    CPU Tasks                    GPU Tasks
    ├── File I/O                 ├── Tensor Operations
    ├── CSD Queries              ├── Distance Calculations
    ├── Data Validation          ├── Matrix Operations
    ├── Configuration            ├── Similarity Metrics
    └── Result Aggregation       └── Batch Processing

**Memory Management**

.. code-block:: python

    # Memory hierarchy optimization
    class MemoryManager:
        """
        Manages memory allocation across CPU and GPU.
        """
        def __init__(self, gpu_fraction=0.8):
            self.gpu_memory_limit = self._get_gpu_memory() * gpu_fraction
            self.cpu_memory_limit = self._get_cpu_memory() * 0.7
            
        def allocate_batch(self, batch_size, structure_complexity):
            """Dynamic batch size adjustment based on memory constraints."""
            estimated_gpu_usage = self._estimate_gpu_memory(batch_size, structure_complexity)
            estimated_cpu_usage = self._estimate_cpu_memory(batch_size, structure_complexity)
            
            if estimated_gpu_usage > self.gpu_memory_limit:
                batch_size = self._reduce_batch_size(batch_size, 'gpu')
            if estimated_cpu_usage > self.cpu_memory_limit:
                batch_size = self._reduce_batch_size(batch_size, 'cpu')
                
            return batch_size

**Parallel Processing Design**

.. code-block:: python

    # Multi-level parallelization
    def process_structures_parallel(self, structure_batch):
        """
        Implements nested parallelization:
        - Process-level: Multiple structures in parallel
        - Thread-level: I/O operations
        - GPU-level: Tensor operations
        """
        with ProcessPoolExecutor(max_workers=self.n_processes) as executor:
            # Process-level parallelization
            futures = []
            for structure_group in self._chunk_structures(structure_batch):
                future = executor.submit(self._process_structure_group, structure_group)
                futures.append(future)
            
            # Collect results
            results = []
            for future in as_completed(futures):
                results.extend(future.result())
                
        return results

Data Storage Design
-------------------

HDF5 Schema Architecture
~~~~~~~~~~~~~~~~~~~~~~~~

**Hierarchical Organization**

.. code-block:: text

    analysis_processed.h5
    ├── /metadata
    │   ├── version_info
    │   ├── creation_timestamp  
    │   ├── configuration_hash
    │   └── dataset_statistics
    │
    ├── /structure_identifiers
    │   └── refcode_list (N,) vlen<string>
    │
    ├── /crystal_properties
    │   ├── z_prime (N,) int32
    │   ├── cell_volume (N,) float32
    │   ├── cell_matrix (N,3,3) float32
    │   └── space_group (N,) vlen<string>
    │
    ├── /atomic_data
    │   ├── n_atoms (N,) int32
    │   ├── atom_coords (N,) vlen<float32>
    │   ├── atom_symbols (N,) vlen<string>
    │   └── atom_fragment_id (N,) vlen<int32>
    │
    ├── /molecular_features
    │   ├── n_fragments (N,) int32
    │   ├── fragment_com_coords (N,) vlen<float32>
    │   ├── fragment_inertia_eigvals (N,) vlen<float32>
    │   └── fragment_formula (N,) vlen<string>
    │
    └── /interaction_data
        ├── inter_cc_n_contacts (N,) int32
        ├── inter_cc_central_atom (N,) vlen<string>
        ├── inter_cc_length (N,) vlen<float32>
        └── inter_hb_is_hbond (N,) vlen<bool>

**Variable-Length Array Implementation**

.. code-block:: python

    # Efficient storage of ragged arrays
    class VariableLengthDataset:
        """
        Handles variable-length data with optimal storage.
        """
        def __init__(self, h5_group, dataset_name, dtype):
            self.vlen_dtype = h5py.vlen_dtype(dtype)
            self.dataset = h5_group.create_dataset(
                dataset_name, 
                shape=(0,), 
                maxshape=(None,),
                dtype=self.vlen_dtype,
                chunks=True,
                compression='lz4',
                shuffle=True,
                fletcher32=True
            )
            
        def append_batch(self, data_batch):
            """Efficiently append batch of variable-length arrays."""
            current_size = self.dataset.shape[0]
            new_size = current_size + len(data_batch)
            
            # Resize dataset
            self.dataset.resize((new_size,))
            
            # Write data
            for i, data in enumerate(data_batch):
                self.dataset[current_size + i] = np.asarray(data, dtype=self.dtype)

**Compression and Optimization**

.. code-block:: python

    # Storage optimization strategies
    COMPRESSION_SETTINGS = {
        'coordinates': {
            'compression': 'lz4',
            'compression_opts': 3,
            'shuffle': True,
            'chunks': True
        },
        'identifiers': {
            'compression': 'gzip',
            'compression_opts': 6,
            'shuffle': False,
            'chunks': True
        },
        'properties': {
            'compression': 'lz4',
            'compression_opts': 1,
            'shuffle': True,
            'chunks': True
        }
    }

Algorithm Details
-----------------

Similarity Clustering Algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Packing Similarity Calculation**

.. code-block:: python

    def compute_packing_similarity_batch(
            coords_batch: torch.Tensor,      # (B, N, 3)
            cell_matrices_batch: torch.Tensor,  # (B, 3, 3)
            similarity_threshold: float = 0.8
    ) -> torch.Tensor:
        """
        Compute 3D packing similarity using CCDC algorithms.
        
        Implementation of the packing similarity metric used by
        the Cambridge Crystallographic Data Centre.
        """
        B, N, _ = coords_batch.shape
        device = coords_batch.device
        
        # 1. Generate symmetry-related coordinates
        symmetry_coords = generate_symmetry_coordinates(
            coords_batch, cell_matrices_batch
        )
        
        # 2. Compute optimal alignment between structures
        similarity_matrix = torch.zeros((B, B), device=device)
        
        for i in range(B):
            for j in range(i+1, B):
                # Find best alignment using Kabsch algorithm
                rotation, translation = kabsch_alignment(
                    coords_batch[i], coords_batch[j]
                )
                
                # Apply transformation and compute RMSD
                aligned_coords = apply_transformation(
                    coords_batch[j], rotation, translation
                )
                
                rmsd = compute_rmsd(coords_batch[i], aligned_coords)
                similarity = np.exp(-rmsd / similarity_threshold)
                
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
        
        return similarity_matrix

**Clustering Implementation**

.. code-block:: python

    def cluster_similar_structures(
            similarity_matrix: torch.Tensor,
            threshold: float = 0.8,
            method: str = 'hierarchical'
    ) -> List[List[int]]:
        """
        Cluster structures based on packing similarity.
        """
        if method == 'hierarchical':
            return hierarchical_clustering(similarity_matrix, threshold)
        elif method == 'graph_based':
            return graph_clustering(similarity_matrix, threshold)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
    
    def hierarchical_clustering(similarity_matrix, threshold):
        """Agglomerative clustering with similarity threshold."""
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        
        # Convert similarity to distance
        distance_matrix = 1.0 - similarity_matrix.cpu().numpy()
        condensed_distances = squareform(distance_matrix)
        
        # Perform clustering
        linkage_matrix = linkage(condensed_distances, method='average')
        clusters = fcluster(linkage_matrix, 1.0 - threshold, criterion='distance')
        
        # Group by cluster
        cluster_groups = {}
        for idx, cluster_id in enumerate(clusters):
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            cluster_groups[cluster_id].append(idx)
            
        return list(cluster_groups.values())

Geometric Calculation Algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Distance Calculations with Periodic Boundaries**

.. code-block:: python

    def compute_distances_periodic_batch(
            coords1: torch.Tensor,  # (B, N1, 3)
            coords2: torch.Tensor,  # (B, N2, 3)
            cell_matrices: torch.Tensor,  # (B, 3, 3)
            use_minimum_image: bool = True
    ) -> torch.Tensor:
        """
        Compute distances with periodic boundary conditions.
        """
        B, N1, _ = coords1.shape
        N2 = coords2.shape[1]
        device = coords1.device
        
        # Vectorized distance calculation
        diff = coords1.unsqueeze(2) - coords2.unsqueeze(1)  # (B, N1, N2, 3)
        
        if use_minimum_image:
            # Apply minimum image convention
            fractional_diff = torch.matmul(
                diff, torch.inverse(cell_matrices).transpose(-2, -1)
            )
            
            # Wrap to [-0.5, 0.5]
            fractional_diff = fractional_diff - torch.round(fractional_diff)
            
            # Convert back to Cartesian
            diff = torch.matmul(fractional_diff, cell_matrices)
        
        distances = torch.norm(diff, dim=-1)  # (B, N1, N2)
        return distances

**Inertia Tensor Calculation**

.. code-block:: python

    def compute_inertia_tensor_batch(
            coords: torch.Tensor,     # (B, N, 3)
            masses: torch.Tensor,     # (B, N)
            com: torch.Tensor,        # (B, 3)
            atom_mask: torch.BoolTensor  # (B, N)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute inertia tensor eigenvalues and eigenvectors.
        """
        B, N, _ = coords.shape
        device = coords.device
        
        # Center coordinates at COM
        centered_coords = coords - com.unsqueeze(1)  # (B, N, 3)
        
        # Apply mask
        centered_coords = centered_coords * atom_mask.unsqueeze(-1).float()
        masses_masked = masses * atom_mask.float()
        
        # Compute inertia tensor components
        x, y, z = centered_coords.unbind(dim=-1)  # Each (B, N)
        
        Ixx = torch.sum(masses_masked * (y**2 + z**2), dim=1)  # (B,)
        Iyy = torch.sum(masses_masked * (x**2 + z**2), dim=1)
        Izz = torch.sum(masses_masked * (x**2 + y**2), dim=1)
        Ixy = -torch.sum(masses_masked * x * y, dim=1)
        Ixz = -torch.sum(masses_masked * x * z, dim=1)
        Iyz = -torch.sum(masses_masked * y * z, dim=1)
        
        # Construct inertia tensor matrix
        I_tensor = torch.zeros((B, 3, 3), device=device)
        I_tensor[:, 0, 0] = Ixx
        I_tensor[:, 1, 1] = Iyy
        I_tensor[:, 2, 2] = Izz
        I_tensor[:, 0, 1] = I_tensor[:, 1, 0] = Ixy
        I_tensor[:, 0, 2] = I_tensor[:, 2, 0] = Ixz
        I_tensor[:, 1, 2] = I_tensor[:, 2, 1] = Iyz
        
        # Compute eigenvalues and eigenvectors
        eigenvals, eigenvecs = torch.linalg.eigh(I_tensor)
        
        # Sort by eigenvalue magnitude
        sort_indices = torch.argsort(eigenvals, dim=1)
        eigenvals = torch.gather(eigenvals, 1, sort_indices)
        eigenvecs = torch.gather(
            eigenvecs, 2, sort_indices.unsqueeze(1).expand(-1, 3, -1)
        )
        
        return eigenvals, eigenvecs

Fragment Analysis Algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Rigid Fragment Identification**

.. code-block:: python

    def identify_rigid_fragments_batch(
            coords: torch.Tensor,         # (B, N, 3)
            bond_indices: List[torch.Tensor],  # List of (E_i, 2) for each structure
            flexibility_threshold: float = 15.0  # degrees
    ) -> List[torch.Tensor]:
        """
        Identify rigid molecular fragments based on bond flexibility.
        """
        fragments_batch = []
        
        for i, bond_idx in enumerate(bond_indices):
            if len(bond_idx) == 0:
                fragments_batch.append(torch.arange(coords.shape[1]))
                continue
                
            # Build molecular graph
            graph = build_molecular_graph(bond_idx)
            
            # Identify rotatable bonds
            rotatable_bonds = identify_rotatable_bonds(
                coords[i], bond_idx, flexibility_threshold
            )
            
            # Remove rotatable bonds to get rigid fragments
            rigid_graph = remove_bonds(graph, rotatable_bonds)
            
            # Find connected components (rigid fragments)
            fragments = find_connected_components(rigid_graph)
            fragments_batch.append(fragments)
            
        return fragments_batch

**Fragment Property Calculation**

.. code-block:: python

    def compute_fragment_properties_batch(
            coords: torch.Tensor,     # (B, N, 3)
            masses: torch.Tensor,     # (B, N)
            fragments: List[List[torch.Tensor]],  # Fragment assignments
            device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive fragment properties.
        """
        max_fragments = max(len(frags) for frags in fragments)
        B = len(fragments)
        
        # Initialize output tensors
        com_coords = torch.zeros((B, max_fragments, 3), device=device)
        inertia_eigvals = torch.zeros((B, max_fragments, 3), device=device)
        inertia_eigvecs = torch.zeros((B, max_fragments, 3, 3), device=device)
        planarity_scores = torch.zeros((B, max_fragments), device=device)
        
        for i, fragment_list in enumerate(fragments):
            for j, fragment_atoms in enumerate(fragment_list):
                if len(fragment_atoms) == 0:
                    continue
                    
                # Fragment coordinates and masses
                frag_coords = coords[i, fragment_atoms]  # (n_atoms, 3)
                frag_masses = masses[i, fragment_atoms]  # (n_atoms,)
                
                # Center of mass
                total_mass = torch.sum(frag_masses)
                com = torch.sum(frag_coords * frag_masses.unsqueeze(-1), dim=0) / total_mass
                com_coords[i, j] = com
                
                # Inertia tensor
                eigvals, eigvecs = compute_inertia_tensor_single(
                    frag_coords, frag_masses, com
                )
                inertia_eigvals[i, j] = eigvals
                inertia_eigvecs[i, j] = eigvecs
                
                # Planarity score
                planarity = compute_planarity_score(frag_coords, com)
                planarity_scores[i, j] = planarity
        
        return {
            'com_coords': com_coords,
            'inertia_eigvals': inertia_eigvals,
            'inertia_eigvecs': inertia_eigvecs,
            'planarity_scores': planarity_scores
        }

Contact Detection Algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Intermolecular Contact Identification**

.. code-block:: python

    def detect_intermolecular_contacts_batch(
            coords: torch.Tensor,           # (B, N, 3)
            cell_matrices: torch.Tensor,    # (B, 3, 3)
            van_der_waals_radii: torch.Tensor,  # (B, N)
            contact_threshold: float = 1.2  # Factor of sum of vdW radii
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Detect intermolecular contacts using symmetry operations.
        """
        B, N, _ = coords.shape
        contacts_batch = []
        
        for i in range(B):
            structure_coords = coords[i]  # (N, 3)
            cell_matrix = cell_matrices[i]  # (3, 3)
            vdw_radii = van_der_waals_radii[i]  # (N,)
            
            # Generate symmetry-related images
            symmetry_coords = generate_symmetry_images(
                structure_coords, cell_matrix
            )
            
            contacts = []
            
            # Check contacts between central molecule and symmetry images
            for sym_idx, sym_coords in enumerate(symmetry_coords):
                distances = torch.cdist(structure_coords, sym_coords)  # (N, N)
                
                # Contact criteria
                vdw_sum = vdw_radii.unsqueeze(1) + vdw_radii.unsqueeze(0)  # (N, N)
                contact_distances = vdw_sum * contact_threshold
                
                contact_mask = (distances < contact_distances) & (distances > 0.1)
                contact_pairs = torch.nonzero(contact_mask, as_tuple=False)
                
                for pair in contact_pairs:
                    central_atom, contact_atom = pair[0].item(), pair[1].item()
                    distance = distances[central_atom, contact_atom].item()
                    
                    contacts.append({
                        'central_atom': central_atom,
                        'contact_atom': contact_atom,
                        'distance': distance,
                        'symmetry_operation': sym_idx,
                        'is_hbond': classify_hydrogen_bond(
                            structure_coords[central_atom],
                            sym_coords[contact_atom],
                            distance
                        )
                    })
            
            contacts_batch.append(contacts)
            
        return contacts_batch

**Hydrogen Bond Classification**

.. code-block:: python

    def classify_hydrogen_bond(
            donor_coord: torch.Tensor,    # (3,)
            acceptor_coord: torch.Tensor, # (3,)
            distance: float,
            donor_element: str,
            acceptor_element: str,
            hydrogen_coord: torch.Tensor = None  # (3,) optional
    ) -> bool:
        """
        Classify whether a contact is a hydrogen bond.
        """
        # Distance criteria
        if distance > 3.5:  # Å
            return False
            
        # Element criteria
        hbond_donors = {'N', 'O', 'S', 'F'}
        hbond_acceptors = {'N', 'O', 'S', 'F', 'Cl'}
        
        if donor_element not in hbond_donors or acceptor_element not in hbond_acceptors:
            return False
        
        # Angle criteria (if hydrogen position available)
        if hydrogen_coord is not None:
            dha_angle = compute_angle(donor_coord, hydrogen_coord, acceptor_coord)
            if dha_angle < 120.0:  # degrees
                return False
        
        return True

Performance Optimization Details
-------------------------------

GPU Memory Management
~~~~~~~~~~~~~~~~~~~~

**Dynamic Memory Allocation**

.. code-block:: python

    class GPUMemoryManager:
        """
        Intelligent GPU memory management for batch processing.
        """
        def __init__(self, reserved_fraction=0.1):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.total_memory = torch.cuda.get_device_properties(0).total_memory
            self.reserved_memory = self.total_memory * reserved_fraction
            self.available_memory = self.total_memory - self.reserved_memory
            
        def estimate_batch_memory(self, batch_size, avg_atoms, data_types):
            """Estimate memory usage for a given batch configuration."""
            memory_per_structure = 0
            
            # Coordinate data
            memory_per_structure += avg_atoms * 3 * 4  # float32 coordinates
            
            # Additional arrays based on data types requested
            if 'contacts' in data_types:
                avg_contacts = avg_atoms * 5  # Rough estimate
                memory_per_structure += avg_contacts * 8  # Contact data
                
            if 'fragments' in data_types:
                avg_fragments = max(1, avg_atoms // 10)
                memory_per_structure += avg_fragments * 12  # Fragment data
            
            total_memory = batch_size * memory_per_structure * 2  # Safety factor
            return total_memory
            
        def optimize_batch_size(self, initial_batch_size, avg_atoms, data_types):
            """Find optimal batch size that fits in available memory."""
            for batch_size in range(initial_batch_size, 0, -1):
                estimated_memory = self.estimate_batch_memory(
                    batch_size, avg_atoms, data_types
                )
                if estimated_memory < self.available_memory:
                    return batch_size
            return 1

**Tensor Optimization Patterns**

.. code-block:: python

    def optimize_tensor_operations():
        """
        Performance optimization patterns for tensor operations.
        """
        # 1. Use in-place operations when possible
        tensor.add_(other_tensor)  # Instead of tensor = tensor + other_tensor
        
        # 2. Minimize device transfers
        with torch.cuda.device(device):
            # Keep all operations on same device
            result = tensor1.mm(tensor2).add_(bias)
        
        # 3. Use appropriate data types
        coords = coords.to(dtype=torch.float32)  # Mixed precision
        indices = indices.to(dtype=torch.int32)
        
        # 4. Leverage vectorization
        # Instead of loops, use batch operations
        distances = torch.cdist(coords1, coords2)  # Vectorized distance matrix
        
        # 5. Memory-efficient attention patterns
        with torch.backends.cuda.sdp_kernel(enable_math=False):
            # Use optimized kernels when available
            attention_output = torch.nn.functional.scaled_dot_product_attention(
                query, key, value
            )

Testing and Validation
----------------------

Unit Testing Framework
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import pytest
    import torch
    import numpy as np
    from crystal_analyzer import CrystalAnalyzer
    from csa_config import ExtractionConfig

    class TestGeometryCalculations:
        """Test suite for geometric calculation algorithms."""
        
        @pytest.fixture
        def sample_coordinates(self):
            """Generate test coordinate data."""
            torch.manual_seed(42)
            coords = torch.randn(2, 10, 3)  # 2 structures, 10 atoms each
            return coords
            
        @pytest.fixture
        def sample_cell_matrices(self):
            """Generate test unit cell matrices."""
            # Cubic cells for simplicity
            matrices = torch.eye(3).unsqueeze(0).repeat(2, 1, 1)
            matrices[0] *= 10.0  # 10 Å cube
            matrices[1] *= 15.0  # 15 Å cube
            return matrices
            
        def test_distance_calculation(self, sample_coordinates, sample_cell_matrices):
            """Test periodic distance calculations."""
            from geometry_utils import compute_distances_periodic_batch
            
            coords = sample_coordinates
            cells = sample_cell_matrices
            
            distances = compute_distances_periodic_batch(
                coords, coords, cells, use_minimum_image=True
            )
            
            # Check dimensions
            assert distances.shape == (2, 10, 10)
            
            # Check diagonal is zero (self-distances)
            diagonal = torch.diagonal(distances, dim1=1, dim2=2)
            assert torch.allclose(diagonal, torch.zeros_like(diagonal), atol=1e-6)
            
            # Check symmetry
            assert torch.allclose(distances, distances.transpose(1, 2))
            
        def test_inertia_tensor_calculation(self, sample_coordinates):
            """Test inertia tensor eigenvalue calculation."""
            from geometry_utils import compute_inertia_tensor_batch
            
            coords = sample_coordinates
            masses = torch.ones(2, 10)  # Unit masses
            com = torch.mean(coords, dim=1)  # Center of mass
            mask = torch.ones(2, 10, dtype=torch.bool)
            
            eigvals, eigvecs = compute_inertia_tensor_batch(coords, masses, com, mask)
            
            # Check dimensions
            assert eigvals.shape == (2, 3)
            assert eigvecs.shape == (2, 3, 3)
            
            # Check eigenvalues are non-negative and sorted
            assert torch.all(eigvals >= 0)
            assert torch.all(eigvals[:, 1:] >= eigvals[:, :-1])
            
            # Check eigenvectors are orthonormal
            for i in range(2):
                dot_products = torch.mm(eigvecs[i], eigvecs[i].T)
                assert torch.allclose(dot_products, torch.eye(3), atol=1e-5)

**Integration Testing**

.. code-block:: python

    class TestPipelineIntegration:
        """Test complete pipeline workflows."""
        
        @pytest.fixture
        def minimal_config(self, tmp_path):
            """Create minimal test configuration."""
            config_data = {
                "extraction": {
                    "data_directory": str(tmp_path),
                    "data_prefix": "test",
                    "actions": {
                        "get_refcode_families": False,
                        "cluster_refcode_families": False,
                        "get_unique_structures": False,
                        "get_structure_data": True,
                        "post_extraction_process": True
                    },
                    "filters": {
                        "structure_list": ["cif", str(tmp_path / "test_structures")]
                    },
                    "extraction_batch_size": 2,
                    "post_extraction_batch_size": 2
                }
            }
            
            config_file = tmp_path / "test_config.json"
            config_file.write_text(json.dumps(config_data))
            return ExtractionConfig.from_json(config_file)
            
        def test_complete_pipeline(self, minimal_config, sample_cif_files):
            """Test complete pipeline execution."""
            analyzer = CrystalAnalyzer(minimal_config)
            
            # Should complete without errors
            analyzer.extract_data()
            
            # Check output files exist
            output_dir = Path(minimal_config.data_directory)
            assert (output_dir / "structures" / "test_structures_processed.h5").exists()
            
            # Validate output data
            with h5py.File(output_dir / "structures" / "test_structures_processed.h5", 'r') as f:
                assert 'refcode_list' in f
                assert len(f['refcode_list']) > 0
                assert 'cell_volume' in f
                assert 'atom_coords' in f

Documentation and Maintenance
-----------------------------

Code Documentation Standards
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Docstring Format**

.. code-block:: python

    def compute_molecular_descriptors(
            coords: torch.Tensor,
            masses: torch.Tensor,
            device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive molecular descriptors from atomic coordinates.

        This function calculates a wide range of geometric and topological
        descriptors used in crystal structure analysis, including moments
        of inertia, asphericity, planarity metrics, and shape descriptors.

        Parameters
        ----------
        coords : torch.Tensor, shape (B, N, 3)
            Cartesian coordinates of atoms for B structures with up to N atoms each.
            Coordinates should be in Ångström units.
        masses : torch.Tensor, shape (B, N)
            Atomic masses in atomic mass units (Da). Padding positions should
            have mass 0.0.
        device : torch.device
            PyTorch device for computation (CPU or CUDA).

        Returns
        -------
        descriptors : dict of str to torch.Tensor
            Dictionary containing computed descriptors:
            
            - 'moments_of_inertia': torch.Tensor, shape (B, 3)
                Principal moments of inertia in ascending order
            - 'asphericity': torch.Tensor, shape (B,)
                Measure of deviation from spherical shape
            - 'acylindricity': torch.Tensor, shape (B,)
                Measure of deviation from cylindrical shape
            - 'planarity_score': torch.Tensor, shape (B,)
                Measure of molecular planarity (0=linear, 1=planar)

        Raises
        ------
        ValueError
            If input tensors have incompatible shapes or contain invalid values.
        RuntimeError
            If GPU computation fails due to memory constraints.

        Notes
        -----
        This function uses the convention where moments of inertia are computed
        about the center of mass and sorted in ascending order. The asphericity
        and acylindricity parameters follow the definitions of:
        
        - Asphericity: I₃ - 0.5(I₁ + I₂)
        - Acylindricity: I₂ -Technical Details
=================

Deep dive into CSA's architecture, algorithms, and implementation details. This section provides comprehensive technical information for developers, advanced users, and researchers who need to understand the inner workings of the system.

.. note::
   
   This section assumes familiarity with crystallography, computational chemistry, and software engineering concepts.

Architecture Overview
--------------------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 🏗️ System Architecture
      :link: architecture
      :link-type: doc

      Overall system design, component interactions, and data flow patterns.

   .. grid-item-card:: ⚡ Performance Optimization
      :link: performance
      :link-type: doc

      GPU acceleration, memory management, and computational efficiency.

   .. grid-item-card:: 💾 Data Storage Design
      :link: storage_design
      :link-type: doc

      HDF5 organization, variable-length arrays, and storage optimization.

   .. grid-item-card:: 🔧 Configuration System
      :link: config_system
      :link-type: doc

      Configuration validation, inheritance, and parameter management.

Algorithms and Methods
---------------------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 🔍 Similarity Clustering
      :link: clustering_algorithms
      :link-type: doc

      Packing similarity metrics and clustering methodologies.

   .. grid-item-card:: 📐 Geometric Calculations
      :link: geometry_algorithms
      :link-type: doc

      Distance calculations, coordinate transformations, and geometric descriptors.

   .. grid-item-card:: 🧩 Fragment Analysis
      :link: fragment_algorithms
      :link-type: doc

      Rigid fragment identification and property calculations.

   .. grid-item-card:: 🔗 Contact Detection
      :link: contact_algorithms
      :link-type: doc

      Intermolecular contact identification and hydrogen bond detection.

Implementation Details
---------------------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 🐍 Python Implementation
      :link: python_details
      :link-type: doc

      Core Python modules, class hierarchies, and design patterns.

   .. grid-item-card:: 🔥 PyTorch Integration
      :link: pytorch_details
      :link-type: doc

      GPU tensor operations, batch processing, and memory optimization.

   .. grid-item-card:: 📊 HDF5 Integration
      :link: hdf5_details
      :link-type: doc

      Dataset organization, compression, and efficient I/O patterns.

   .. grid-item-card:: 🔌 CSD Integration
      :link: csd_integration
      :link-type: doc

      Cambridge Structural Database API usage and data extraction.

System Architecture
-------------------

High-Level Design
~~~~~~~~~~~~~~~~~

CSA follows a modular, pipeline-based architecture designed for scalability and maintainability:

.. code-block:: text

    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │   Configuration │    │   CSD Interface │    │  Data Validation│
    │     Manager     │────│    & Querying   │────│   & Filtering   │
    └─────────────────┘    └─────────────────┘    └─────────────────┘
             │                        │                        │
             ▼                        ▼                        ▼
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │ Family Extraction│    │   Clustering    │    │  Representative │
    │   & Grouping    │────│   & Similarity  │────│   Selection     │
    └─────────────────┘    └─────────────────┘    └─────────────────┘
             │                        │                        │
             ▼                        ▼                        ▼
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │   Structure     │    │  Feature        │    │   Output        │
    │   Extraction    │────│  Engineering    │────│  Generation     │
    └─────────────────┘    └─────────────────┘    └─────────────────┘

Core Components
~~~~~~~~~~~~~~~

**Pipeline Orchestrator** (``C