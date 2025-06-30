"""
Module: dataset_initializer.py

Initialize all required datasets in a processed HDF5 file based on maximum dimensions
scanned from the raw data.

Dependencies
------------
h5py
numpy
"""
import h5py
import numpy as np
from typing import List

dt_str     = h5py.string_dtype('utf-8')
str_vlen   = h5py.vlen_dtype(dt_str)        # variable‐length UTF-8 strings
int_vlen   = h5py.vlen_dtype(np.int32)      # variable‐length int32 arrays
float_vlen = h5py.vlen_dtype(np.float32)    # variable‐length float32 arrays
bool_vlen  = h5py.vlen_dtype(np.bool_)      # variable‐length bool arrays

class DatasetInitializer:
    """
    Create and configure all datasets in the output HDF5 file prior to writing raw and computed data.

    Attributes
    ----------
    h5_out : h5py.File
        Output HDF5 file handle into which datasets will be created.
    refcodes : List[str]
        List of structure refcodes for which data will be stored.
    dims : dict
        Dictionary of maximum dimensions for ragged arrays:
        'atoms', 'bonds', 'fragments', 'contacts_inter', 'contacts_intra',
        'hbonds_inter', 'hbonds_intra'.

    Methods
    -------
    initialize_datasets()
        Create all required datasets for raw and computed data.
    """
    def __init__(self, h5_out: h5py.File, refcodes: List[str], dims: dict):
        """
        Initialize the DatasetInitializer.
    
        Parameters
        ----------
        h5_out : h5py.File
            Open HDF5 file handle where datasets will be created.
        refcodes : list of str
            List of structure refcodes for which data will be stored.
        dims : dict
            Maximum dimension sizes for ragged arrays:
            'atoms', 'bonds', 'fragments', 'contacts_inter', 'contacts_intra',
            'hbonds_inter', 'hbonds_intra'.
        """
        self.h5_out = h5_out
        self.refcodes = refcodes
        self.dims = dims

        # unpack dims once
        self.N = len(refcodes)
        self.max_atoms          = dims['atoms']
        self.max_bonds          = dims['bonds']
        self.max_fragments      = dims['fragments']
        self.max_contacts_inter = dims['contacts_inter'] * 2
        self.max_contacts_intra = dims['contacts_intra']
        self.max_hbonds_inter   = dims['hbonds_inter'] * 2
        self.max_hbonds_intra   = dims['hbonds_intra']

    def _init_refcode_list(self):
        """
        Create the 'refcode_list' dataset containing all refcodes.

        The dataset has shape (N,) and dtype variable-length UTF-8 string.
        """
        h = self.h5_out
        h.create_dataset('refcode_list', data=np.array(self.refcodes, dtype=object), dtype=dt_str)

    def _init_crystal_fields(self):
        """
        Create fixed-size datasets for crystal-level fields with shape (N, …).
    
          - 'z_value'             : (N,) float32
          - 'z_prime'             : (N,) float32
          - 'cell_lengths'        : (N, 3) float32
          - 'cell_angles'         : (N, 3) float32
          - 'cell_volume'         : (N,) float32
          - 'cell_matrix'         : (N, 3, 3) float32
          - 'cell_density'        : (N,) float32
          - 'packing_coefficient' : (N,) float32
          - 'scaled_cell'         : (N, 6) float32
          - 'identifier'          : (N,) vlen<string>
          - 'space_group'         : (N,) vlen<string>
        """
        h = self.h5_out
        N = self.N
        h.create_dataset('z_value',             (N,),      dtype=np.float32)
        h.create_dataset('z_prime',             (N,),      dtype=np.float32)
        h.create_dataset('cell_lengths',        (N, 3),    dtype=np.float32)
        h.create_dataset('cell_angles',         (N, 3),    dtype=np.float32)
        h.create_dataset('cell_volume',         (N,),      dtype=np.float32)
        h.create_dataset('cell_matrix',         (N, 3, 3), dtype=np.float32)
        h.create_dataset('cell_density',        (N,),      dtype=np.float32)
        h.create_dataset('packing_coefficient', (N,),      dtype=np.float32)
        h.create_dataset('scaled_cell',         (N, 6),    dtype=np.float32)
        h.create_dataset('identifier',          (N,),      dtype=dt_str)
        h.create_dataset('space_group',         (N,),      dtype=dt_str)
        
    def _init_atom_fields(self):
        """
        Create per-atom datasets under shape (N, …).
    
          - 'n_atoms'                          : (N,) int32
          - 'atom_label'                       : (N,) vlen<string>
          - 'atom_symbol'                      : (N,) vlen<string>
          - 'atom_sybyl_type'                  : (N,) vlen<string>
          - 'atom_neighbour_list'              : (N,) vlen<string>
          - 'atom_number'                      : (N,) vlen<int32>
          - 'atom_fragment_id'                 : (N,) vlen<int32>
          - 'atom_coords'                      : (N,) vlen<float32>  (flattened 3·n_atoms)
          - 'atom_frac_coords'                 : (N,) vlen<float32>
          - 'atom_weight'                      : (N,) vlen<float32>
          - 'atom_charge'                      : (N,) vlen<float32>
          - 'atom_distances_to_special_planes' : (N,) vlen<float32>
        """
        h = self.h5_out
        N = self.N

        h.create_dataset('n_atoms',                          (N,), dtype=np.int32)
        h.create_dataset('atom_label',                       (N,), dtype=str_vlen)
        h.create_dataset('atom_symbol',                      (N,), dtype=str_vlen)
        h.create_dataset('atom_sybyl_type',                  (N,), dtype=str_vlen)
        h.create_dataset('atom_neighbour_list',              (N,), dtype=str_vlen)
        h.create_dataset('atom_number',                      (N,), dtype=int_vlen)
        h.create_dataset('atom_fragment_id',                 (N,), dtype=int_vlen)
        h.create_dataset('atom_coords',                      (N,), dtype=float_vlen)
        h.create_dataset('atom_frac_coords',                 (N,), dtype=float_vlen)
        h.create_dataset('atom_weight',                      (N,), dtype=float_vlen)
        h.create_dataset('atom_charge',                      (N,), dtype=float_vlen)
        h.create_dataset('atom_distances_to_special_planes', (N,), dtype=float_vlen)

    def _init_bond_fields(self):
        """
        Create per-bond datasets under shape (N, …).
    
          - 'n_bonds'                              : (N,) int32
          - 'bond_id'                              : (N,) vlen<string>
          - 'bond_type'                            : (N,) vlen<string>
          - 'bond_atom1'                           : (N,) vlen<string>
          - 'bond_atom2'                           : (N,) vlen<string>
          - 'bond_atom1_idx'                       : (N,) vlen<int32>
          - 'bond_atom2_idx'                       : (N,) vlen<int32>
          - 'bond_length'                          : (N,) vlen<float32>
          - 'bond_vector_angles_to_special_planes' : (N,) vlen<float32>
          - 'bond_is_cyclic'                       : (N,) vlen<bool>
          - 'bond_is_rotatable_raw'                : (N,) vlen<bool>
          - 'bond_is_rotatable'                    : (N,) vlen<bool>
        """
        h = self.h5_out
        N = self.N
        
        h.create_dataset('n_bonds',                              (N,), dtype=np.int32)
        h.create_dataset('bond_id',                              (N,), dtype=str_vlen)
        h.create_dataset('bond_type',                            (N,), dtype=str_vlen)
        h.create_dataset('bond_atom1',                           (N,), dtype=str_vlen)
        h.create_dataset('bond_atom2',                           (N,), dtype=str_vlen)
        h.create_dataset('bond_atom1_idx',                       (N,), dtype=int_vlen)
        h.create_dataset('bond_atom2_idx',                       (N,), dtype=int_vlen)
        h.create_dataset('bond_length',                          (N,), dtype=float_vlen)
        h.create_dataset('bond_vector_angles_to_special_planes', (N,), dtype=float_vlen)
        h.create_dataset('bond_is_cyclic',                       (N,), dtype=bool_vlen)
        h.create_dataset('bond_is_rotatable_raw',                (N,), dtype=bool_vlen)
        h.create_dataset('bond_is_rotatable',                    (N,), dtype=bool_vlen)
        
    def _init_bond_angles_fields(self):
        """
        Create per-bond-angle datasets under shape (N, …).
    
          - 'bond_angle_id'      : (N,) vlen<string>
          - 'bond_angle'         : (N,) vlen<float32>
          - 'bond_angle_mask'    : (N,) vlen<bool>
          - 'bond_angle_atom_idx': (N,) vlen<int32>
        """
        h = self.h5_out
        N = self.N
        
        h.create_dataset('bond_angle_id',        (N,), dtype=str_vlen)
        h.create_dataset('bond_angle',           (N,), dtype=float_vlen)
        h.create_dataset('bond_angle_mask',      (N,), dtype=bool_vlen)
        h.create_dataset('bond_angle_atom_idx',  (N,), dtype=int_vlen)
    
    def _init_torsion_angles_fields(self):
        """
        Create per-torsion-angle datasets under shape (N, …).
    
          - 'torsion_id'      : (N,) vlen<string>
          - 'torsion'         : (N,) vlen<float32>
          - 'torsion_mask'    : (N,) vlen<bool>
          - 'torsion_atom_idx': (N,) vlen<int32>
        """
        h = self.h5_out
        N = self.N
        
        h.create_dataset('torsion_id',        (N,), dtype=str_vlen)
        h.create_dataset('torsion',           (N,), dtype=float_vlen)
        h.create_dataset('torsion_mask',      (N,), dtype=bool_vlen)
        h.create_dataset('torsion_atom_idx',  (N,), dtype=int_vlen)
        
    def _init_intermolecular_contact_fields(self):
        """
        Create per-intermolecular-contact datasets under shape (N, …).
    
          - 'inter_cc_n_contacts'                   : (N,) int32
          - 'inter_cc_id'                           : (N,) vlen<string>
          - 'inter_cc_central_atom'                 : (N,) vlen<string>
          - 'inter_cc_contact_atom'                 : (N,) vlen<string>
          - 'inter_cc_central_atom_coords'          : (N,) vlen<float32>
          - 'inter_cc_contact_atom_coords'          : (N,) vlen<float32>
          - 'inter_cc_central_atom_frac_coords'     : (N,) vlen<float32>
          - 'inter_cc_contact_atom_frac_coords'     : (N,) vlen<float32>
          - 'inter_cc_length'                       : (N,) vlen<float32>
          - 'inter_cc_strength'                     : (N,) vlen<float32>
          - 'inter_cc_in_los'                       : (N,) vlen<bool>
          - 'inter_cc_symmetry_A'                   : (N,) vlen<float32>
          - 'inter_cc_symmetry_T'                   : (N,) vlen<float32>
          - 'inter_cc_symmetry_A_inv'               : (N,) vlen<float32>
          - 'inter_cc_symmetry_T_inv'               : (N,) vlen<float32>
          - 'inter_cc_central_atom_fragment_idx'    : (N,) vlen<int32>
          - 'inter_cc_contact_atom_fragment_idx'    : (N,) vlen<int32>
          - 'inter_cc_central_atom_idx'             : (N,) vlen<int32>
          - 'inter_cc_contact_atom_idx'             : (N,) vlen<int32>
          - 'inter_cc_is_hbond'                     : (N,) vlen<int32>
          - 'inter_cc_contact_atom_to_fragment_com_dist'      : (N,) vlen<float32>
          - 'inter_cc_contact_atom_to_fragment_com_frac_dist' : (N,) vlen<float32>
          - 'inter_cc_contact_atom_to_fragment_com_vec'       : (N,) vlen<float32>
          - 'inter_cc_contact_atom_to_fragment_com_frac_vec'  : (N,) vlen<float32>
        """
        h = self.h5_out
        N = self.N
    
        # 1) Number of contacts (still fixed‐length int)
        h.create_dataset('inter_cc_n_contacts',   (N,), dtype=np.int32)
    
        # 2) IDs and labels → vlen<string>
        h.create_dataset('inter_cc_id',           (N,), dtype=str_vlen)
        h.create_dataset('inter_cc_central_atom', (N,), dtype=str_vlen)
        h.create_dataset('inter_cc_contact_atom', (N,), dtype=str_vlen)
    
        # 3) Coordinates → flatten (nC, 3) → (3*nC) vlen<float>
        h.create_dataset('inter_cc_central_atom_coords',      (N,), dtype=float_vlen)
        h.create_dataset('inter_cc_contact_atom_coords',      (N,), dtype=float_vlen)
        h.create_dataset('inter_cc_central_atom_frac_coords', (N,), dtype=float_vlen)
        h.create_dataset('inter_cc_contact_atom_frac_coords', (N,), dtype=float_vlen)
    
        # 4) Lengths, strengths, in_los → vlen<float> or vlen<bool>
        h.create_dataset('inter_cc_length',    (N,), dtype=float_vlen)
        h.create_dataset('inter_cc_strength',  (N,), dtype=float_vlen)
        h.create_dataset('inter_cc_in_los',    (N,), dtype=bool_vlen)
    
        # 5) Symmetry operations → vlen<string>, vlen<float>
        h.create_dataset('inter_cc_symmetry_A',     (N,), dtype=float_vlen)
        h.create_dataset('inter_cc_symmetry_T',     (N,), dtype=float_vlen)
        h.create_dataset('inter_cc_symmetry_A_inv', (N,), dtype=float_vlen)
        h.create_dataset('inter_cc_symmetry_T_inv', (N,), dtype=float_vlen)
    
        # 6) Fragment indices → vlen<int>
        h.create_dataset('inter_cc_central_atom_fragment_idx', (N,), dtype=int_vlen)
        h.create_dataset('inter_cc_contact_atom_fragment_idx', (N,), dtype=int_vlen)
    
        # 7) Atom indices, H‐bond  flags → vlen<int> 
        h.create_dataset('inter_cc_central_atom_idx', (N,), dtype=int_vlen)
        h.create_dataset('inter_cc_contact_atom_idx', (N,), dtype=int_vlen)
        h.create_dataset('inter_cc_is_hbond',         (N,), dtype=int_vlen)
    
        # 8) Computed distances/vectors to fragment COM → vlen<float>
        h.create_dataset('inter_cc_contact_atom_to_fragment_com_dist',      (N,), dtype=float_vlen)
        h.create_dataset('inter_cc_contact_atom_to_fragment_com_frac_dist', (N,), dtype=float_vlen)
        h.create_dataset('inter_cc_contact_atom_to_fragment_com_vec',       (N,), dtype=float_vlen)
        h.create_dataset('inter_cc_contact_atom_to_fragment_com_frac_vec',  (N,), dtype=float_vlen)
        
    def _init_intramolecular_contact_fields(self):
        """
        Create per-intramolecular-contact datasets under shape (N, …).
    
          - 'intra_n_contacts'                  : (N,) int32
          - 'intra_cc_id'                       : (N,) vlen<string>
          - 'intra_cc_central_atom'             : (N,) vlen<string>
          - 'intra_cc_contact_atom'             : (N,) vlen<string>
          - 'intra_cc_central_atom_coords'      : (N,) vlen<float32>
          - 'intra_cc_contact_atom_coords'      : (N,) vlen<float32>
          - 'intra_cc_central_atom_frac_coords' : (N,) vlen<float32>
          - 'intra_cc_contact_atom_frac_coords' : (N,) vlen<float32>
          - 'intra_cc_length'                   : (N,) vlen<float32>
          - 'intra_cc_strength'                 : (N,) vlen<float32>
          - 'intra_cc_in_los'                   : (N,) vlen<bool>
          - 'intra_cc_central_atom_idx'         : (N,) vlen<int32>
          - 'intra_cc_contact_atom_idx'         : (N,) vlen<int32>
        """
        h = self.h5_out
        N = self.N
    
        # 1) Number of contacts (fixed-length int per structure)
        h.create_dataset('intra_n_contacts', (N,), dtype=np.int32)
    
        # 2) IDs and labels → vlen<string>
        h.create_dataset('intra_cc_id',               (N,), dtype=str_vlen)
        h.create_dataset('intra_cc_central_atom',     (N,), dtype=str_vlen)
        h.create_dataset('intra_cc_contact_atom',     (N,), dtype=str_vlen)
    
        # 3) Cartesian coords: flatten (nC,3) → (3·nC), vlen<float>
        h.create_dataset('intra_cc_central_atom_coords',      (N,), dtype=float_vlen)
        h.create_dataset('intra_cc_contact_atom_coords',      (N,), dtype=float_vlen)
    
        # 4) Fractional coords: flatten (nC,3) → (3·nC), vlen<float>
        h.create_dataset('intra_cc_central_atom_frac_coords', (N,), dtype=float_vlen)
        h.create_dataset('intra_cc_contact_atom_frac_coords', (N,), dtype=float_vlen)
    
        # 5) Lengths & strengths → vlen<float>
        h.create_dataset('intra_cc_length',   (N,), dtype=float_vlen)
        h.create_dataset('intra_cc_strength', (N,), dtype=float_vlen)
    
        # 6) In-line-of-sight flags → vlen<bool>
        h.create_dataset('intra_cc_in_los',    (N,), dtype=bool_vlen)
    
        # 7) Atom indices  → vlen<int> 
        h.create_dataset('intra_cc_central_atom_idx', (N,), dtype=int_vlen)
        h.create_dataset('intra_cc_contact_atom_idx', (N,), dtype=int_vlen)
       
    def _init_intermolecular_hbond_fields(self): 
        """
        Create per-intermolecular-H-bond datasets under shape (N, …).
    
          - 'inter_hb_n_hbonds'                  : (N,) int32
          - 'inter_hb_id'                        : (N,) vlen<string>
          - 'inter_hb_central_atom'              : (N,) vlen<string>
          - 'inter_hb_hydrogen_atom'             : (N,) vlen<string>
          - 'inter_hb_contact_atom'              : (N,) vlen<string>
          - 'inter_hb_central_atom_coords'       : (N,) vlen<float32>
          - 'inter_hb_hydrogen_atom_coords'      : (N,) vlen<float32>
          - 'inter_hb_contact_atom_coords'       : (N,) vlen<float32>
          - 'inter_hb_central_atom_frac_coords'  : (N,) vlen<float32>
          - 'inter_hb_hydrogen_atom_frac_coords' : (N,) vlen<float32>
          - 'inter_hb_contact_atom_frac_coords'  : (N,) vlen<float32>
          - 'inter_hb_length'                    : (N,) vlen<float32>
          - 'inter_hb_angle'                     : (N,) vlen<float32>
          - 'inter_hb_in_los'                    : (N,) vlen<bool>
          - 'inter_hb_symmetry'                  : (N,) vlen<string>
          - 'inter_hb_central_atom_idx'          : (N,) vlen<int32>
          - 'inter_hb_hydrogen_atom_idx'         : (N,) vlen<int32>
          - 'inter_hb_contact_atom_idx'          : (N,) vlen<int32>
        """
        h = self.h5_out
        N = self.N
    
        # 1) Number of H‐bonds (fixed‐length int)
        h.create_dataset('inter_hb_n_hbonds', (N,), dtype=np.int32)
    
        # 2) IDs and labels → vlen<string>
        h.create_dataset('inter_hb_id',             (N,), dtype=str_vlen)
        h.create_dataset('inter_hb_central_atom',   (N,), dtype=str_vlen)
        h.create_dataset('inter_hb_hydrogen_atom',  (N,), dtype=str_vlen)
        h.create_dataset('inter_hb_contact_atom',   (N,), dtype=str_vlen)
    
        # 3) Cartesian coords: flatten (nH,3) → (3·nH), vlen<float>
        h.create_dataset('inter_hb_central_atom_coords',       (N,), dtype=float_vlen)
        h.create_dataset('inter_hb_hydrogen_atom_coords',      (N,), dtype=float_vlen)
        h.create_dataset('inter_hb_contact_atom_coords',       (N,), dtype=float_vlen)
    
        # 4) Fractional coords: flatten (nH,3) → (3·nH), vlen<float>
        h.create_dataset('inter_hb_central_atom_frac_coords',  (N,), dtype=float_vlen)
        h.create_dataset('inter_hb_hydrogen_atom_frac_coords', (N,), dtype=float_vlen)
        h.create_dataset('inter_hb_contact_atom_frac_coords',  (N,), dtype=float_vlen)
    
        # 5) Lengths & angles → vlen<float>
        h.create_dataset('inter_hb_length',  (N,), dtype=float_vlen)
        h.create_dataset('inter_hb_angle',   (N,), dtype=float_vlen)
    
        # 6) In‐line‐of‐sight flags → vlen<bool>
        h.create_dataset('inter_hb_in_los',   (N,), dtype=bool_vlen)
    
        # 7) Symmetry ops → vlen<string>
        h.create_dataset('inter_hb_symmetry', (N,), dtype=str_vlen)
    
        # 8) Atom indices  → vlen<int> 
        h.create_dataset('inter_hb_central_atom_idx',  (N,), dtype=int_vlen)
        h.create_dataset('inter_hb_hydrogen_atom_idx', (N,), dtype=int_vlen)
        h.create_dataset('inter_hb_contact_atom_idx',  (N,), dtype=int_vlen)
        
    def _init_intramolecular_hbond_fields(self):
        """
        Create per-intramolecular-H-bond datasets under shape (N, …).
    
          - 'intra_n_hbonds'                     : (N,) int32
          - 'intra_hb_id'                        : (N,) vlen<string>
          - 'intra_hb_central_atom'              : (N,) vlen<string>
          - 'intra_hb_hydrogen_atom'             : (N,) vlen<string>
          - 'intra_hb_contact_atom'              : (N,) vlen<string>
          - 'intra_hb_central_atom_coords'       : (N,) vlen<float32>
          - 'intra_hb_hydrogen_atom_coords'      : (N,) vlen<float32>
          - 'intra_hb_contact_atom_coords'       : (N,) vlen<float32>
          - 'intra_hb_central_atom_frac_coords'  : (N,) vlen<float32>
          - 'intra_hb_hydrogen_atom_frac_coords' : (N,) vlen<float32>
          - 'intra_hb_contact_atom_frac_coords'  : (N,) vlen<float32>
          - 'intra_hb_length'                    : (N,) vlen<float32>
          - 'intra_hb_angle'                     : (N,) vlen<float32>
          - 'intra_hb_in_los'                    : (N,) vlen<bool>
          - 'intra_hb_central_atom_idx'          : (N,) vlen<int32>
          - 'intra_hb_hydrogen_atom_idx'         : (N,) vlen<int32>
          - 'intra_hb_contact_atom_idx'          : (N,) vlen<int32>
        """
        h = self.h5_out
        N = self.N
    
        # 1) Number of H‐bonds (fixed‐length int per structure)
        h.create_dataset('intra_n_hbonds', (N,), dtype=np.int32)
    
        # 2) IDs and labels → vlen<string>
        h.create_dataset('intra_hb_id',             (N,), dtype=str_vlen)
        h.create_dataset('intra_hb_central_atom',   (N,), dtype=str_vlen)
        h.create_dataset('intra_hb_hydrogen_atom',  (N,), dtype=str_vlen)
        h.create_dataset('intra_hb_contact_atom',   (N,), dtype=str_vlen)
    
        # 3) Cartesian coords: flatten (nH, 3) → (3·nH), vlen<float>
        h.create_dataset('intra_hb_central_atom_coords',       (N,), dtype=float_vlen)
        h.create_dataset('intra_hb_hydrogen_atom_coords',      (N,), dtype=float_vlen)
        h.create_dataset('intra_hb_contact_atom_coords',       (N,), dtype=float_vlen)
    
        # 4) Fractional coords: flatten (nH, 3) → (3·nH), vlen<float>
        h.create_dataset('intra_hb_central_atom_frac_coords',  (N,), dtype=float_vlen)
        h.create_dataset('intra_hb_hydrogen_atom_frac_coords', (N,), dtype=float_vlen)
        h.create_dataset('intra_hb_contact_atom_frac_coords',  (N,), dtype=float_vlen)
    
        # 5) Lengths & angles → vlen<float>
        h.create_dataset('intra_hb_length', (N,), dtype=float_vlen)
        h.create_dataset('intra_hb_angle',  (N,), dtype=float_vlen)
    
        # 5) Symmetry operations → vlen<string>, vlen<float>
        h.create_dataset('inter_hb_symmetry_A',     (N,), dtype=float_vlen)
        h.create_dataset('inter_hb_symmetry_T',     (N,), dtype=float_vlen)
        h.create_dataset('inter_hb_symmetry_A_inv', (N,), dtype=float_vlen)
        h.create_dataset('inter_hb_symmetry_T_inv', (N,), dtype=float_vlen)
    
        # 6) In‐line‐of‐sight flags → vlen<bool>
        h.create_dataset('intra_hb_in_los',  (N,), dtype=bool_vlen)
    
        # 7) Atom indices  → vlen<int> 
        h.create_dataset('intra_hb_central_atom_idx',  (N,), dtype=int_vlen)
        h.create_dataset('intra_hb_hydrogen_atom_idx', (N,), dtype=int_vlen)
        h.create_dataset('intra_hb_contact_atom_idx',  (N,), dtype=int_vlen)
        
    def _init_fragment_properties_fields(self):
        """
        Create per-fragment computed property datasets under shape (N, …).
    
          - 'n_fragments'                          : (N,) int32
          - 'fragment_local_id'                    : (N,) vlen<int32>
          - 'fragment_formulas'                    : (N,) vlen<string>
          - 'fragment_n_atoms'                     : (N,) vlen<int32>
          - 'fragment_com_coords'                  : (N,) vlen<float32]
          - 'fragment_com_frac_coords'             : (N,) vlen<float32]
          - 'fragment_cen_coords'                  : (N,) vlen<float32]
          - 'fragment_cen_frac_coords'             : (N,) vlen<float32]
          - 'fragment_inertia_tensors'             : (N,) vlen<float32]
          - 'fragment_inertia_eigvals'             : (N,) vlen<float32]
          - 'fragment_inertia_eigvecs'             : (N,) vlen<float32]
          - 'fragment_inertia_quaternions'         : (N,) vlen<float32]
          - 'fragment_quadrupole_tensors'          : (N,) vlen<float32]
          - 'fragment_quadrupole_eigvals'          : (N,) vlen<float32]
          - 'fragment_quadrupole_eigvecs'          : (N,) vlen<float32]
          - 'fragment_quadrupole_quaternions'      : (N,) vlen<float32]
          - 'fragment_atom_to_com_dist'            : (N,) vlen<float32]
          - 'fragment_atom_to_com_frac_dist'       : (N,) vlen<float32]
          - 'fragment_atom_to_com_vecs'            : (N,) vlen<float32]
          - 'fragment_atom_to_com_frac_vecs'       : (N,) vlen<float32]
          - 'fragment_Ql'                          : (N,) vlen<float32]
          - 'fragment_plane_centroid'              : (N,) vlen<float32]
          - 'fragment_plane_normal'                : (N,) vlen<float32]
          - 'fragment_planarity_rmsd'              : (N,) vlen<float32]
          - 'fragment_planarity_max_dev'           : (N,) vlen<float32]
          - 'fragment_planarity_score'             : (N,) vlen<float32]
        """
        h = self.h5_out
        N = self.N
    
        # 1) How many fragments each structure actually had (fixed‐length int)
        h.create_dataset('n_fragments', (N,), dtype=np.int32)
    
        # 2) fragment_ids (vlen<int>)
        h.create_dataset('fragment_local_id', (N,), dtype=int_vlen)
    
        # 3) fragment_formulas (vlen<string>)
        h.create_dataset('fragment_formula', (N,), dtype=str_vlen)
        
        # 4) fragment_ids (vlen<int>)
        h.create_dataset('fragment_n_atoms', (N,), dtype=h5py.vlen_dtype(np.int32))
    
        # 5) fragment centers of mass (cart + frac) → flatten (nF, 3) to (3*nF), vlen<float>
        h.create_dataset('fragment_com_coords',      (N,), dtype=float_vlen)
        h.create_dataset('fragment_com_frac_coords', (N,), dtype=float_vlen)
    
        # 6) fragment centroids (cart + frac) → flatten (nF, 3) to (3*nF), vlen<float>
        h.create_dataset('fragment_cen_coords',      (N,), dtype=float_vlen)
        h.create_dataset('fragment_cen_frac_coords', (N,), dtype=float_vlen)
    
        # 7) inertia tensors: flatten (nF, 3, 3) to (9*nF), vlen<float>
        h.create_dataset('fragment_inertia_tensors',      (N,), dtype=float_vlen)
        h.create_dataset('fragment_inertia_eigvals',      (N,), dtype=float_vlen)
        h.create_dataset('fragment_inertia_eigvecs',      (N,), dtype=float_vlen)
        h.create_dataset('fragment_inertia_quaternions',  (N,), dtype=float_vlen)
    
        # 8) quadrupole tensors: flatten (nF, 3, 3) to (9*nF), vlen<float>
        h.create_dataset('fragment_quadrupole_tensors',   (N,), dtype=float_vlen)
        h.create_dataset('fragment_quadrupole_eigvals',   (N,), dtype=float_vlen)
        h.create_dataset('fragment_quadrupole_eigvecs',   (N,), dtype=float_vlen)
        h.create_dataset('fragment_quadrupole_quaternions',(N,), dtype=float_vlen)
    
        # 9) atom_to_com_dist: flatten per‐fragment A entries into (nF * A), vlen<float>
        h.create_dataset('fragment_atom_to_com_dist',      (N,), dtype=float_vlen)
        h.create_dataset('fragment_atom_to_com_frac_dist', (N,), dtype=float_vlen)
        
        # 10) atom_to_com_vecs: flatten (nF, A, 3) to (3 * nF * A), vlen<float>
        h.create_dataset('fragment_atom_to_com_vec',      (N,), dtype=float_vlen)
        h.create_dataset('fragment_atom_to_com_frac_vec', (N,), dtype=float_vlen)
    
        # 11) Steinhardt Ql: flatten (nF, 5) to (5 * nF), vlen<float>
        h.create_dataset('fragment_Ql', (N,), dtype=float_vlen)
    
        # 12) Plane centroids & normals: flatten (nF, 3) to (3 * nF), vlen<float>
        h.create_dataset('fragment_plane_centroid', (N,), dtype=float_vlen)
        h.create_dataset('fragment_plane_normal',   (N,), dtype=float_vlen)
    
        # 13) Planarity metrics: flatten (nF,) each → (nF), vlen<float>
        h.create_dataset('fragment_planarity_rmsd',    (N,), dtype=float_vlen)
        h.create_dataset('fragment_planarity_max_dev', (N,), dtype=float_vlen)
        h.create_dataset('fragment_planarity_score',   (N,), dtype=float_vlen)
        
    def initialize_datasets(self):
        """
        Create all HDF5 datasets required for storing raw and computed data.
    
        This method calls each of the internal _init_ methods to set up:
        refcode list, crystal fields, atom fields, bond fields,
        bond-angle/torsion fields, contact/H-bond fields, and fragment property fields.
        """
        self._init_refcode_list()
        self._init_crystal_fields()
        self._init_atom_fields()
        self._init_bond_fields()
        self._init_bond_angles_fields()
        self._init_torsion_angles_fields()
        self._init_intermolecular_contact_fields()
        self._init_intramolecular_contact_fields()
        self._init_intermolecular_hbond_fields()
        self._init_intramolecular_hbond_fields()
        self._init_fragment_properties_fields()
     
