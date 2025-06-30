"""
Module: data_writer.py

Provides RawDataWriter and ComputedDataWriter to write raw and computed
datasets into a processed HDF5 file. All per‐structure data is written
slice‐by‐slice, with variable‐length (vlen) datasets for atoms, bonds,
contacts, and H‐bonds.

Dependencies
------------
h5py
numpy
torch
"""
import h5py
import numpy as np
import torch
from typing import Dict, Any

class RawDataWriter:
    """
    Write raw crystal, atom, bond, intramolecular and intermolecular contact,
    and H-bond data into the output HDF5 file, slice-by-slice.

    Attributes
    ----------
    h5_out : h5py.File
        Open HDF5 file for writing processed data.

    Methods
    -------
    write_raw_crystal_data(start, crystal_parameters)
        Write raw crystal parameters into the HDF5 datasets.
    write_raw_atom_data(start, atom_parameters)
        Write raw per-atom data into the HDF5 datasets.
    write_raw_bond_data(start, bond_parameters)
        Write raw per-bond data into the HDF5 datasets.
    write_raw_intramolecular_contact_data(start, intra_cc_parameters)
        Write raw intra-molecular contact data into the HDF5 datasets.
    write_raw_intramolecular_hbond_data(start, intra_hb_parameters)
        Write raw intra-molecular H-bond data into the HDF5 datasets.
    """
    def __init__(self, h5_out: h5py.File):
        """
        Parameters
        ----------
        h5_out : h5py.File
            Open HDF5 file for writing processed data.
        """
        self.h5_out = h5_out

    def write_raw_crystal_data(
            self,
            start: int,
            crystal_parameters: Dict[str, Any]
            ) -> None:
        """
        Write raw crystal parameters into the HDF5 datasets.
        
        Parameters
        ----------
        start : int
            Index offset in the output datasets corresponding to this batch.
        crystal_parameters : Dict[str, Any]
            Dictionary of raw crystal-level arrays or tensors:
            'cell_lengths', 'cell_angles', and any scalar metrics.
        
        Returns
        -------
        None
        """
        # 1) Convert any torch tensors to numpy on CPU
        arrs: Dict[str, np.ndarray] = {}
        for key, vals in crystal_parameters.items():
            if torch.is_tensor(vals):
                arrs[key] = vals.detach().cpu().numpy()
            else:
                arrs[key] = np.asarray(vals)

        # 2) Determine batch‐size B
        B = next(iter(arrs.values())).shape[0]

        # 3) Slice each dataset appropriately
        for key, arr in arrs.items():
            ds = self.h5_out[key]
            if   arr.ndim == 1:
                ds[start:start+B] = arr
            elif arr.ndim == 2:
                ds[start:start+B, :] = arr
            elif arr.ndim == 3:
                ds[start:start+B, :, :] = arr
            else:
                raise ValueError(f"Unsupported ndim {arr.ndim} for raw crystal field '{key}'")
                
    def write_raw_atom_data(
            self,
            start: int,
            atom_parameters: Dict[str, np.ndarray]
            ) -> None:
        """
        Write raw per‐atom data into the HDF5 datasets.
        
        Parameters
        ----------
        start : int
            Index offset in the output datasets corresponding to this batch.
        atom_parameters : Dict[str, Any]
            Dictionary containing per-atom raw data:
            'atom_label', 'atom_symbol', 'atom_number',
            'atom_coords', 'atom_frac_coords',
            'atom_weight', 'atom_charge',
            'atom_sybyl_type', 'atom_neighbour_list',
            and 'atom_mask'.
        
        Returns
        -------
        None
        """
        B, N_max = atom_parameters['atom_number'].shape
    
        # 1) Define your key groups
        int_keys      = ['atom_number']
        flat3_keys    = ['atom_coords', 'atom_frac_coords']
        float1_keys   = ['atom_weight', 'atom_charge']
        vlen_str_keys = ['atom_label', 'atom_symbol', 'atom_sybyl_type', 'atom_neighbour_list']

        # 2) Pre-convert all numeric/coord tensors → NumPy on CPU
        arrs: Dict[str, np.ndarray] = {}
        for key in int_keys + float1_keys + flat3_keys + ['atom_mask']:
            vals = atom_parameters[key]
            if torch.is_tensor(vals):
                arrs[key] = vals.detach().cpu().numpy()
            else:
                arrs[key] = np.asarray(vals)

        for i in range(B):
            idx = start + i
            na  = int(arrs['atom_mask'][i].sum())
            # 3) number of atoms
            self.h5_out['n_atoms'][idx] = na

            # 4) integer arrays
            for key in int_keys:
                seq = arrs[key][i, :na].astype(np.int32)
                self.h5_out[key][idx] = seq

            # 5) flattened coords
            for key in flat3_keys:
                block = arrs[key][i, :na, :]       # shape (na,3)
                self.h5_out[key][idx] = block.reshape(-1)

            # 6) 1-D float arrays
            for key in float1_keys:
                seq = arrs[key][i, :na].astype(np.float32)
                self.h5_out[key][idx] = seq

            # 7) vlen-string lists (never force into a 2D array)
            for key in vlen_str_keys:
                vals = atom_parameters[key]
                if torch.is_tensor(vals):
                    # unlikely, but handle it
                    seq = vals[i, :na].detach().cpu().numpy().tolist()
                elif isinstance(vals, np.ndarray):
                    seq = vals[i, :na].tolist()
                else:
                    # Python list of lists
                    seq = vals[i][:na]
                # write as list of str
                self.h5_out[key][idx] = [str(x) for x in seq]
                
    def write_raw_bond_data(
            self,
            start: int,
            bond_parameters: Dict[str, Any]
            ) -> None:
        """
        Write raw per‐bond data into the HDF5 datasets.
        
        Parameters
        ----------
        start : int
            Index offset in the output datasets corresponding to this batch.
        bond_parameters : Dict[str, Any]
            Dictionary containing per-bond raw data:
            'n_bonds', 'bond_atom1_idx', 'bond_atom2_idx',
            'bond_atom1', 'bond_atom2', 'bond_type',
            'bond_is_rotatable_raw', 'bond_is_cyclic',
            and 'bond_length'.
        
        Returns
        -------
        None
        """
        # 1) Key groups
        count_key      = 'n_bonds'
        int_keys       = ['bond_atom1_idx', 'bond_atom2_idx']
        bool_keys      = ['bond_is_cyclic']
        vlen_str_keys  = ['bond_atom1', 'bond_atom2', 'bond_type']
        vlen_bool_keys = ['bond_is_rotatable_raw']
        vlen_float_keys= ['bond_length']

        # 2) Pre-convert fixed-shape arrays → NumPy on CPU
        arrs: Dict[str, np.ndarray] = {}
        for key in (count_key, *int_keys, *bool_keys):
            vals = bond_parameters[key]
            if torch.is_tensor(vals):
                arrs[key] = vals.detach().cpu().numpy()
            else:
                arrs[key] = np.asarray(vals)

        B = int(arrs[count_key].shape[0])

        # 3) Write each structure
        for i in range(B):
            idx = start + i
            nb  = int(arrs[count_key][i])

            # 3a) bond count
            self.h5_out[count_key][idx] = nb

            # 3b) fixed-length ints
            for key in int_keys:
                seq = arrs[key][i, :nb].astype(np.int32)
                self.h5_out[key][idx] = seq

            # 3c) fixed-length bools
            for key in bool_keys:
                seq = arrs[key][i, :nb].astype(bool)
                self.h5_out[key][idx] = seq

            # 3d) vlen strings
            for key in vlen_str_keys:
                vals = bond_parameters[key]
                if torch.is_tensor(vals):
                    row = vals[i, :nb].detach().cpu().numpy().tolist()
                elif isinstance(vals, np.ndarray):
                    row = vals[i, :nb].tolist()
                else:
                    row = vals[i][:nb]
                self.h5_out[key][idx] = [str(x) for x in row]

            # 3e) vlen bools
            for key in vlen_bool_keys:
                vals = bond_parameters[key]
                if torch.is_tensor(vals):
                    row = vals[i, :nb].detach().cpu().numpy().astype(bool)
                elif isinstance(vals, np.ndarray):
                    row = vals[i, :nb].astype(bool)
                else:
                    row = np.array(vals[i][:nb], dtype=bool)
                self.h5_out[key][idx] = row

            # 3f) vlen floats
            for key in vlen_float_keys:
                vals = bond_parameters[key]
                if torch.is_tensor(vals):
                    row = vals[i, :nb].detach().cpu().numpy().astype(np.float32)
                elif isinstance(vals, np.ndarray):
                    row = vals[i, :nb].astype(np.float32)
                else:
                    row = np.array(vals[i][:nb], dtype=np.float32)
                self.h5_out[key][idx] = row
                
    def write_raw_intramolecular_contact_data(
            self,
            start: int,
            intra_cc_parameters: Dict[str, Any]
            ) -> None:
        """
        Write raw intramolecular contact data into the HDF5 datasets.
        
        Parameters
        ----------
        start : int
            Index offset in the output datasets corresponding to this batch.
        intra_cc_parameters : Dict[str, Any]
            Dictionary containing raw intra-molecular contact data:
            'intra_cc_id', 'intra_cc_central_atom',
            'intra_cc_contact_atom', 'intra_cc_central_atom_idx',
            'intra_cc_contact_atom_idx',
            'intra_cc_central_atom_coords',
            'intra_cc_contact_atom_coords',
            'intra_cc_central_atom_frac_coords',
            'intra_cc_contact_atom_frac_coords',
            'intra_cc_length', 'intra_cc_strength',
            and 'intra_cc_in_los'.
        
        Returns
        -------
        None
        """
        flat3_keys      = ['intra_cc_central_atom_coords', 'intra_cc_contact_atom_coords']
        flat3_frac_keys = ['intra_cc_central_atom_frac_coords', 'intra_cc_contact_atom_frac_coords']
        float1_keys     = ['intra_cc_length', 'intra_cc_strength']
        int_keys        = ['intra_cc_central_atom_idx', 'intra_cc_contact_atom_idx']
        bool_keys       = ['intra_cc_in_los']

        arrs: Dict[str, np.ndarray] = {}
        for key in flat3_keys + flat3_frac_keys + float1_keys + bool_keys + int_keys:
            vals = intra_cc_parameters[key]
            if torch.is_tensor(vals):
                arrs[key] = vals.detach().cpu().numpy()
            else:
                arrs[key] = np.asarray(vals)

        # 2) String‐list inputs (keep as Python lists or object‐arrays)
        labels_cl = intra_cc_parameters['intra_cc_central_atom']
        str_keys  = ['intra_cc_id', 'intra_cc_central_atom','intra_cc_contact_atom']

        B = len(labels_cl)
        for i in range(B):
            idx = start + i
            nC  = len(labels_cl[i])

            # 4) count
            self.h5_out['intra_n_contacts'][idx] = nC

            # 5) vlen‐string fields
            for key in str_keys:
                seq = intra_cc_parameters[key]
                if torch.is_tensor(seq):
                    row = seq[i, :nC].detach().cpu().numpy().tolist()
                elif isinstance(seq, np.ndarray):
                    row = seq[i, :nC].tolist()
                else:
                    row = seq[i][:nC]
                self.h5_out[key][idx] = [str(x) for x in row]

            # 6) vlen‐int fields
            for key in int_keys:
                vals = intra_cc_parameters[key]
                if torch.is_tensor(vals):
                    row = vals[i, :nC].detach().cpu().numpy().astype(np.int32)
                elif isinstance(vals, np.ndarray):
                    row = vals[i, :nC].astype(np.int32)
                else:
                    row = np.array(vals[i][:nC], dtype=np.int32)
                self.h5_out[key][idx] = row

            # 7) flattened Cartesian coords
            for key in flat3_keys:
                block = arrs[key][i, :nC, :]
                self.h5_out[key][idx] = block.reshape(-1)

            # 8) flattened fractional coords
            for key in flat3_frac_keys:
                block = arrs[key][i, :nC, :]
                self.h5_out[key][idx] = block.reshape(-1)

            # 9) 1‐D floats
            for key in float1_keys:
                seq = arrs[key][i, :nC].astype(np.float32)
                self.h5_out[key][idx] = seq

            # 10) 1‐D bools
            for key in bool_keys:
                seq = arrs[key][i, :nC].astype(bool)
                self.h5_out[key][idx] = seq
                
    def write_raw_intramolecular_hbond_data(
            self,
            start: int,
            intra_hb_parameters: Dict[str, Any]
            ) -> None:
        """
        Write raw intramolecular H‐bond data into the HDF5 datasets.
        
        Parameters
        ----------
        start : int
            Index offset in the output datasets corresponding to this batch.
        intra_hb_parameters : Dict[str, Any]
            Dictionary containing raw intra-molecular H-bond data:
            'intra_hb_id', 'intra_hb_central_atom',
            'intra_hb_hydrogen_atom', 'intra_hb_contact_atom',
            'intra_hb_central_atom_idx',
            'intra_hb_hydrogen_atom_idx',
            'intra_hb_contact_atom_idx',
            'intra_hb_central_atom_coords',
            'intra_hb_hydrogen_atom_coords',
            'intra_hb_contact_atom_coords',
            'intra_hb_central_atom_frac_coords',
            'intra_hb_hydrogen_atom_frac_coords',
            'intra_hb_contact_atom_frac_coords',
            'intra_hb_length', 'intra_hb_angle',
            and 'intra_hb_in_los'.
        
        Returns
        -------
        None
        """
        # 1) Prepare NumPy arrays for all fixed‐shape fields
        flat3_keys = [
            'intra_hb_central_atom_coords',
            'intra_hb_hydrogen_atom_coords',
            'intra_hb_contact_atom_coords'
            ]
        flat3_frac_keys = [
            'intra_hb_central_atom_frac_coords',
            'intra_hb_hydrogen_atom_frac_coords',
            'intra_hb_contact_atom_frac_coords'
            ]
        float1_keys = ['intra_hb_length', 'intra_hb_angle']
        int_keys = [
            'intra_hb_central_atom_idx',
            'intra_hb_hydrogen_atom_idx',
            'intra_hb_contact_atom_idx'
            ]
        bool_keys   = ['intra_hb_in_los']

        arrs: Dict[str, np.ndarray] = {}
        for key in flat3_keys + flat3_frac_keys + float1_keys + bool_keys + int_keys:
            vals = intra_hb_parameters[key]
            if torch.is_tensor(vals):
                arrs[key] = vals.detach().cpu().numpy()
            else:
                arrs[key] = np.asarray(vals)

        # 2) vlen‐string inputs
        labels_cl = intra_hb_parameters['intra_hb_central_atom']
        str_keys = [
            'intra_hb_id', 
            'intra_hb_central_atom',
            'intra_hb_hydrogen_atom',
            'intra_hb_contact_atom'
            ]

        B = len(labels_cl)
        for i in range(B):
            idx = start + i
            nH  = len(labels_cl[i])

            # 4) number of H-bonds
            self.h5_out['intra_n_hbonds'][idx] = nH

            # 5) vlen‐string fields
            for key in str_keys:
                seq = intra_hb_parameters[key]
                if torch.is_tensor(seq):
                    row = seq[i, :nH].detach().cpu().numpy().tolist()
                elif isinstance(seq, np.ndarray):
                    row = seq[i, :nH].tolist()
                else:
                    row = seq[i][:nH]
                self.h5_out[key][idx] = [str(x) for x in row]

            # 6) vlen‐int fields
            for key in int_keys:
                vals = intra_hb_parameters[key]
                if torch.is_tensor(vals):
                    row = vals[i, :nH].detach().cpu().numpy().astype(np.int32)
                elif isinstance(vals, np.ndarray):
                    row = vals[i, :nH].astype(np.int32)
                else:
                    row = np.array(vals[i][:nH], dtype=np.int32)
                self.h5_out[key][idx] = row

            # 7) flattened Cartesian coords
            for key in flat3_keys:
                block = arrs[key][i, :nH, :]   # (nH,3)
                self.h5_out[key][idx] = block.reshape(-1)

            # 8) flattened fractional coords
            for key in flat3_frac_keys:
                block = arrs[key][i, :nH, :]
                self.h5_out[key][idx] = block.reshape(-1)

            # 9) 1‐D floats
            for key in float1_keys:
                seq = arrs[key][i, :nH].astype(np.float32)
                self.h5_out[key][idx] = seq

            # 10) 1‐D bools
            for key in bool_keys:
                seq = arrs[key][i, :nH].astype(bool)
                self.h5_out[key][idx] = seq
            

class ComputedDataWriter:
    """
    Write computed crystal, atom, bond, molecule, and contact/H-bond features into the output HDF5 file.

    Attributes
    ----------
    h5_out : h5py.File
        Open HDF5 file for writing processed data.

    Methods
    -------
    write_computed_crystal_data(start: int, crystal_parameters: Dict[str, Any]) -> None
        Write computed crystal parameters ('scaled_cell', 'cell_matrix') into the HDF5 datasets.
    write_computed_atom_data(start: int, atom_parameters: Dict[str, Any]) -> None
        Write computed atom-level features ('atom_fragment_id', 'atom_dist_to_special_planes') into the HDF5 datasets.
    write_computed_bond_data(start: int, bond_parameters: Dict[str, Any]) -> None
        Write computed bond-level features ('bond_is_rotatable', 'bond_vector_angles_to_special_planes') into the HDF5 datasets.
    write_computed_molecule_data(start: int, molecule_parameters: Dict[str, Any]) -> None
        Write computed intra-molecular bond angles and torsion features into the HDF5 datasets.
    write_computed_intermolecular_contact_data(start: int, inter_cc_parameters: Dict[str, Any]) -> None
        Write computed intermolecular contact features (IDs, indices, coords, lengths, strengths, h-bond flags, fragment mappings, vectors) into the HDF5 datasets.
    write_computed_intermolecular_hbond_data(start: int, inter_hb_parameters: Dict[str, Any]) -> None
        Write computed intermolecular H-bond features (IDs, donor/acceptor labels, indices, coords, lengths, angles, masks, symmetry ops) into the HDF5 datasets.
    """
    def __init__(self, h5_out: h5py.File):
        """
        Initialize ComputedDataWriter.
        
        Parameters
        ----------
        h5_out : h5py.File
            Open HDF5 file for writing processed data.
        """
        self.h5_out = h5_out

    def write_computed_crystal_data(
            self,
            start: int,
            crystal_parameters: Dict[str, Any]
            ) -> None:
        """
        Write computed crystal parameters into the HDF5 datasets.
        
        Parameters
        ----------
        start : int
            Index offset in the output datasets corresponding to this batch.
        crystal_parameters : Dict[str, Any]
            Dictionary containing computed crystal data:
            'scaled_cell' (shape (B, 6)) and 'cell_matrix' (shape (B, 3, 3)).
        
        Returns
        -------
        None
        """
        # First convert any torch.Tensors to numpy arrays
        arrs: Dict[str, np.ndarray] = {}
        for key, vals in crystal_parameters.items():
            if torch.is_tensor(vals):
                arrs[key] = vals.detach().cpu().numpy()
            else:
                arrs[key] = np.asarray(vals)
        
        # Batch‐size
        B = next(iter(arrs.values())).shape[0]
        for key, arr in arrs.items():
            ds = self.h5_out[key]
            if arr.ndim == 2:
                ds[start:start+B, :] = arr
            elif arr.ndim == 3:
                ds[start:start+B, :, :] = arr
            else:
                # fallback—e.g. someone might pass a 1‐D array of scalars
                ds[start:start+B] = arr
                
    def write_computed_atom_data(
            self,
            start: int,
            atom_parameters: Dict[str, Any]
            ) -> None:
        """
        Write computed atom-level features into the HDF5 datasets.
        
        Parameters
        ----------
        start : int
            Index offset in the output datasets corresponding to this batch.
        atom_parameters : Dict[str, Any]
            Dictionary containing computed atom data:
            'atom_fragment_id' (list or array of shape (B, N)),
            'atom_dist_to_special_planes' (shape (B, N, P)).
        
        Returns
        -------
        None
        """
        B = len(atom_parameters['atom_fragment_id'])

        # Pre-extract & convert the distances array once
        dist_arr = atom_parameters['atom_dist_to_special_planes']
        if torch.is_tensor(dist_arr):
            dist_arr = dist_arr.detach().cpu().numpy()
    
        # Pre-extract & convert the fragment-id container once
        frag_container = atom_parameters['atom_fragment_id']
        # If it's a single Tensor of shape (B, N_max), convert to numpy
        if torch.is_tensor(frag_container):
            frag_container = frag_container.detach().cpu().numpy().tolist()
        # Otherwise assume it's already a list of lists
    
        for i in range(B):
            idx = start + i
    
            # 1) number of real atoms
            na = int(self.h5_out['n_atoms'][idx])
    
            # 2) fragment IDs
            # frag_container[i] is now a Python list
            all_ids = np.array(frag_container[i], dtype=np.int32)
            frag_ids = all_ids[:na]
            frag_ids = frag_ids[frag_ids >= 0]
            self.h5_out['atom_fragment_id'][idx] = frag_ids
    
            # 3) distances: slice then flatten
            block = dist_arr[i, :na, :]       # shape (na, 26)
            flattened = block.reshape(-1)     # length = na * 26
            self.h5_out['atom_distances_to_special_planes'][idx] = flattened
        
    def write_computed_bond_data(
            self,
            start: int,
            bond_parameters: Dict[str, Any]
            ) -> None:
        """
        Write computed bond-level features into the HDF5 datasets.
        
        Parameters
        ----------
        start : int
            Index offset in the output datasets corresponding to this batch.
        bond_parameters : Dict[str, Any]
            Dictionary containing computed bond data:
            'bond_is_rotatable' (shape (B, M)) and
            'bond_vector_angles_to_special_planes' (shape (B, M, K)).
        
        Returns
        -------
        None
        """
        # 1) Extract & convert the full arrays once
        rot_arr = bond_parameters['bond_is_rotatable']
        if torch.is_tensor(rot_arr):
            rot_arr = rot_arr.detach().cpu().numpy()
    
        ang_arr = bond_parameters['bond_vector_angles_to_special_planes']
        if torch.is_tensor(ang_arr):
            ang_arr = ang_arr.detach().cpu().numpy()
    
        # 2) Loop over structures
        B, M_max = rot_arr.shape
        for i in range(B):
            idx = start + i
    
            # how many real bonds?
            nb = int(self.h5_out['n_bonds'][idx])
    
            # slice off the first nb flags (shape (nb,))
            comp_rot = rot_arr[i, :nb]
            self.h5_out['bond_is_rotatable'][idx] = comp_rot
    
            # slice & flatten the angles (shape (nb,13) → (nb*13,))
            angle_block = ang_arr[i, :nb, :]
            flat_angles = angle_block.reshape(-1)
            self.h5_out['bond_vector_angles_to_special_planes'][idx] = flat_angles
            
    def write_computed_molecule_data(
            self,
            start: int,
            molecule_parameters: Dict[str, Any]
            ) -> None:
        """
        Write computed intra-molecular bond angles and torsion features.
        
        Parameters
        ----------
        start : int
            Index offset in the output datasets corresponding to this batch.
        molecule_parameters : Dict[str, Any]
            Dictionary containing computed molecule data:
            'bond_angle_id', 'bond_angle', 'bond_angle_mask',
            'bond_angle_atom_idx', 'torsion_id', 'torsion',
            'torsion_mask', and 'torsion_atom_idx'.
        
        Returns
        -------
        None
        """
        # 1) Convert any torch.Tensor → numpy array for numeric & boolean keys
        float_keys = ['bond_angle', 'torsion']
        bool_keys  = ['bond_angle_mask', 'torsion_mask']
        int_keys   = ['bond_angle_atom_idx', 'torsion_atom_idx']

        arrs: Dict[str, np.ndarray] = {}
        for key in float_keys + bool_keys + int_keys:
            vals = molecule_parameters[key]
            if torch.is_tensor(vals):
                arrs[key] = vals.detach().cpu().numpy()
            else:
                arrs[key] = np.asarray(vals)

        # 2) String‐ID keys stay as Python lists or numpy arrays of strings
        str_keys = ['bond_angle_id', 'torsion_id']

        # 3) Number of structures in this batch
        B = len(molecule_parameters[str_keys[0]])

        for i in range(B):
            idx = start + i

            # --- vlen strings ---
            for key in str_keys:
                vals = molecule_parameters[key]
                if torch.is_tensor(vals):
                    row = vals[i].detach().cpu().numpy().tolist()
                elif isinstance(vals, np.ndarray):
                    row = vals[i].tolist()
                else:
                    row = list(vals[i])
                self.h5_out[key][idx] = row

            # --- vlen floats ---
            for key in float_keys:
                arr = arrs[key]
                # length = number of IDs for this field
                n_items = len(molecule_parameters[f"{key}_id"][i])
                row = arr[i, :n_items] if isinstance(arr, np.ndarray) else arr[i]
                self.h5_out[key][idx] = np.array(row, dtype=np.float32)

            # --- vlen bools ---
            for key in bool_keys:
                arr = arrs[key]
                n_items = len(molecule_parameters[key.replace('_mask', '_id')][i])
                row = arr[i, :n_items] if isinstance(arr, np.ndarray) else arr[i]
                self.h5_out[key][idx] = np.array(row, dtype=bool)

            # --- vlen ints (flatten last axis) ---
            for key in int_keys:
                arr = arrs[key]
                # pick the right ID list to get length
                id_key = 'bond_angle_id' if 'angle' in key else 'torsion_id'
                n_items = len(molecule_parameters[id_key][i])
                if isinstance(arr, np.ndarray):
                    block = arr[i, :n_items, :]
                    flat  = block.reshape(-1)
                else:
                    # list of tuples → flatten
                    flat = [atom for group in molecule_parameters[key][i] for atom in group]
                self.h5_out[key][idx] = np.array(flat, dtype=np.int32)
                
    def write_computed_intermolecular_contact_data(
            self,
            start: int,
            inter_cc_parameters: Dict[str, Any]
            ) -> None:
        """
        Write computed intermolecular contact features into the HDF5 datasets.
        
        Parameters
        ----------
        start : int
            Index offset in the output datasets corresponding to this batch.
        inter_cc_parameters : Dict[str, Any]
            Dictionary containing computed intermolecular contact data,
            including IDs, atom labels, indices, coords, frac_coords,
            lengths, strengths, masks, symmetry ops, hbond flags, and
            fragment‐mapping and vector fields.
        
        Returns
        -------
        None
        """
        # Define groups of keys by how to write them
        float3_keys = [
            'inter_cc_central_atom_coords',
            'inter_cc_contact_atom_coords',
            'inter_cc_central_atom_frac_coords',
            'inter_cc_contact_atom_frac_coords',
            'inter_cc_contact_atom_to_fragment_com_vec',
            'inter_cc_contact_atom_to_fragment_com_frac_vec',
            'inter_cc_symmetry_T', 
            'inter_cc_symmetry_T_inv'
            ]
        float1_keys = [
            'inter_cc_length',
            'inter_cc_strength',
            'inter_cc_contact_atom_to_fragment_com_dist',
            'inter_cc_contact_atom_to_fragment_com_frac_dist',
            ]
        int_keys = [
            'inter_cc_central_atom_idx',
            'inter_cc_contact_atom_idx',
            'inter_cc_central_atom_fragment_idx',
            'inter_cc_contact_atom_fragment_idx',
            ]
        matrix9_keys = ['inter_cc_symmetry_A', 'inter_cc_symmetry_A_inv']
        bool_keys    = ['inter_cc_in_los', 'inter_cc_is_hbond']
        
        arrs: Dict[str, np.ndarray] = {}
        for key in float1_keys + float3_keys + int_keys + matrix9_keys + bool_keys:
            vals = inter_cc_parameters[key]
            if torch.is_tensor(vals):
                arrs[key] = vals.detach().cpu().numpy()
            else:
                arrs[key] = np.asarray(vals)
    
        str_keys    = ['inter_cc_central_atom', 'inter_cc_contact_atom']
        labels_cl = inter_cc_parameters['inter_cc_central_atom']
        labels_ct = inter_cc_parameters['inter_cc_contact_atom']
        
        B = len(labels_cl)
    
        for i in range(B):
            idx = start + i
            nC  = len(labels_cl[i])
    
            # 1) number of contacts
            self.h5_out['inter_cc_n_contacts'][idx] = nC
            
            # 2) IDs for each contact pair
            ids = [f"{labels_cl[i][j]}-{labels_ct[i][j]}" for j in range(nC)]
            self.h5_out['inter_cc_id'][idx] = np.array(ids)
    
            # 3) variable-length string arrays
            for key in str_keys:
                seq  = inter_cc_parameters[key]
                vals = seq[i] if isinstance(seq, list) else seq[i][:nC]
                self.h5_out[key][idx] = np.array([str(x) for x in vals])
    
            # 4) flattened 3-D vectors → (3*nC,)
            for key in float3_keys:
                block = arrs[key][i][:nC]
                flat  = block.reshape(-1)
                self.h5_out[key][idx] = flat
    
            # 5) 1-D float arrays → (nC,)
            for key in float1_keys:
                row = arrs[key][i][:nC]
                self.h5_out[key][idx] = row.astype(np.float32)
    
            # 6) integer arrays → (nC,)
            for key in int_keys:
                row = arrs[key][i][:nC]
                self.h5_out[key][idx] = row.astype(np.int32)
    
            # 7) boolean arrays → (nC,)
            for key in bool_keys:
                row = arrs[key][i][:nC]
                self.h5_out[key][idx] = row.astype(bool)
    
            # 8) symmetry matrices (3×3 → 9 floats per contact) → (9*nC,)
            for key in matrix9_keys:
                block = arrs[key][i][:nC]
                flat  = block.reshape(-1)
                self.h5_out[key][idx] = flat.astype(np.float32)
                
    def write_computed_intermolecular_hbond_data(
            self,
            start: int,
            inter_hb_parameters: Dict[str, Any]
            ) -> None:
        """
        Write computed intermolecular H‐bond features into the HDF5 datasets.
        
        Parameters
        ----------
        start : int
            Index offset in the output datasets corresponding to this batch.
        inter_hb_parameters : Dict[str, Any]
            Dictionary containing computed intermolecular H‐bond data,
            including IDs, donor/acceptor labels, indices, coords,
            frac_coords, lengths, angles, masks, symmetry ops, etc.
        
        Returns
        -------
        None
        """
        # Define groups of keys by how to write them
        float3_keys = [
            'inter_hb_central_atom_coords',
            'inter_hb_hydrogen_atom_coords',
            'inter_hb_contact_atom_coords',
            'inter_hb_central_atom_frac_coords',
            'inter_hb_hydrogen_atom_frac_coords',
            'inter_hb_contact_atom_frac_coords',
            'inter_hb_symmetry_T', 
            'inter_hb_symmetry_T_inv'
            ]
        float1_keys = [
            'inter_hb_length',
            'inter_hb_angle',
            ]
        int_keys = [
            'inter_hb_central_atom_idx',
            'inter_hb_hydrogen_atom_idx',
            'inter_hb_contact_atom_idx',
            ]
        matrix9_keys = ['inter_hb_symmetry_A', 'inter_hb_symmetry_A_inv']
        bool_keys = ['inter_hb_in_los']
        
        arrs: Dict[str, np.ndarray] = {}
        for key in float1_keys + float3_keys + int_keys + matrix9_keys + bool_keys:
            vals = inter_hb_parameters[key]
            if torch.is_tensor(vals):
                arrs[key] = vals.detach().cpu().numpy()
            else:
                arrs[key] = np.asarray(vals)

        str_keys    = [
            'inter_hb_central_atom', 
            'inter_hb_hydrogen_atom', 
            'inter_hb_contact_atom'
            ]
        labels_cl = inter_hb_parameters['inter_hb_central_atom']
        labels_h  = inter_hb_parameters['inter_hb_hydrogen_atom']
        labels_ct = inter_hb_parameters['inter_hb_contact_atom']
        
        B = len(labels_cl)

        for i in range(B):
            idx = start + i
            nC  = len(labels_cl[i])

            # 1) number of contacts
            self.h5_out['inter_hb_n_hbonds'][idx] = nC
            
            # 2) IDs for each contact pair
            ids = [f"{labels_cl[i][j]}-{labels_h[i][j]}-{labels_ct[i][j]}" for j in range(nC)]
            self.h5_out['inter_hb_id'][idx] = np.array(ids)

            # 3) variable-length string arrays
            for key in str_keys:
                seq  = inter_hb_parameters[key]
                vals = seq[i] if isinstance(seq, list) else seq[i][:nC]
                self.h5_out[key][idx] = np.array([str(x) for x in vals])

            # 4) flattened 3-D vectors → (3*nC,)
            for key in float3_keys:
                block = arrs[key][i][:nC]
                flat  = block.reshape(-1)
                self.h5_out[key][idx] = flat

            # 5) 1-D float arrays → (nC,)
            for key in float1_keys:
                row = arrs[key][i][:nC]
                self.h5_out[key][idx] = row.astype(np.float32)

            # 6) integer arrays → (nC,)
            for key in int_keys:
                row = arrs[key][i][:nC]
                self.h5_out[key][idx] = row.astype(np.int32)

            # 7) boolean arrays → (nC,)
            for key in bool_keys:
                row = arrs[key][i][:nC]
                self.h5_out[key][idx] = row.astype(bool)
                
            # 8) symmetry matrices (3×3 → 9 floats per contact) → (9*nC,)
            for key in matrix9_keys:
                block = arrs[key][i][:nC]
                flat  = block.reshape(-1)
                self.h5_out[key][idx] = flat.astype(np.float32)
                
    def write_computed_fragment_data(
            self,
            start: int,
            fragment_parameters: Dict[str, Any]
        ) -> None:
        """
        Write computed fragment-level properties into the HDF5 datasets.

        Parameters
        ----------
        start : int
            Index offset in the output datasets corresponding to this batch.
        fragment_parameters : Dict[str, Any]
            Dictionary containing computed fragment data:
            - 'n_fragments'                   : (B,) or list of ints
            - 'fragment_local_id'             : (B, nF) or list of lists of ints
            - 'fragment_formula'              : list of lists of str
            - 'fragment_n_atoms'              : (B, nF) or list of lists of ints
            - all other keys as numpy/Tensor arrays:
              'fragment_com_coords',
              'fragment_com_frac_coords',
              'fragment_cen_coords',
              'fragment_cen_frac_coords',
              'fragment_inertia_tensors',
              'fragment_inertia_eigvals',
              'fragment_inertia_eigvecs',
              'fragment_inertia_quaternions',
              'fragment_quadrupole_tensors',
              'fragment_quadrupole_eigvals',
              'fragment_quadrupole_eigvecs',
              'fragment_quadrupole_quaternions',
              'fragment_atom_to_com_dist',
              'fragment_atom_to_com_frac_dist',
              'fragment_atom_to_com_vec',
              'fragment_atom_to_com_frac_vec',
              'fragment_Ql',
              'fragment_plane_centroid',
              'fragment_plane_normal',
              'fragment_planarity_rmsd',
              'fragment_planarity_max_dev',
              'fragment_planarity_score'
        
        Returns
        -------
        None
        """

        # 1) Convert all numeric / tensor fields to numpy arrays
        #    now including fragment_structure_id so we can group by structure
        int_keys = [
            'fragment_structure_id',
            'fragment_local_id',
            'fragment_n_atoms'
        ]
        float_keys = [
            'fragment_com_coords',
            'fragment_com_frac_coords',
            'fragment_cen_coords',
            'fragment_cen_frac_coords',
            'fragment_inertia_tensors',
            'fragment_inertia_eigvals',
            'fragment_inertia_eigvecs',
            'fragment_inertia_quaternions',
            'fragment_quadrupole_tensors',
            'fragment_quadrupole_eigvals',
            'fragment_quadrupole_eigvecs',
            'fragment_quadrupole_quaternions',
            'fragment_atom_to_com_dist',
            'fragment_atom_to_com_frac_dist',
            'fragment_atom_to_com_vec',
            'fragment_atom_to_com_frac_vec',
            'fragment_Ql',
            'fragment_plane_centroid',
            'fragment_plane_normal',
            'fragment_planarity_rmsd',
            'fragment_planarity_max_dev',
            'fragment_planarity_score'
        ]

        arrs_int = {}
        arrs_flt = {}

        for key in int_keys:
            vals = fragment_parameters[key]
            arrs_int[key] = (
                vals.detach().cpu().numpy() if torch.is_tensor(vals)
                else np.asarray(vals)
            )

        for key in float_keys:
            vals = fragment_parameters[key]
            arrs_flt[key] = (
                vals.detach().cpu().numpy() if torch.is_tensor(vals)
                else np.asarray(vals)
            )

        # 2) Extract formulas
        formulas = fragment_parameters['fragment_formula']

        # 3) Book-keeping
        B = int(fragment_parameters['n_fragments'].shape[0])
        struct_ids = arrs_int['fragment_structure_id']  # shape (F,)

        for i in range(B):
            idx = start + i
            nF = int(fragment_parameters['n_fragments'][i])

            # write scalar count
            self.h5_out['n_fragments'][idx] = nF

            # build a mask to select the F_i fragments of structure i
            mask = (struct_ids == i)

            # --- integer‐vlen fields ---
            for key in ['fragment_local_id', 'fragment_n_atoms']:
                data = arrs_int[key][mask]        # shape (F_i,)
                row  = data[:nF].astype(np.int32)
                self.h5_out[key][idx] = row

            # --- string‐vlen field: formulas ---
            seq = formulas[i]
            if torch.is_tensor(seq):
                seq = seq.detach().cpu().numpy().tolist()
            elif isinstance(seq, np.ndarray):
                seq = seq.tolist()
            self.h5_out['fragment_formula'][idx] = [str(x) for x in seq[:nF]]

            # --- float‐vlen fields ---
            for key, arr in arrs_flt.items():
                block = arr[mask]           # shape (F_i, …)
                block = block[:nF]          # drop any extra padding
                flat  = block.reshape(-1)   # flatten to 1D
                self.h5_out[key][idx] = flat.astype(np.float32)