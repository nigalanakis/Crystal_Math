"""
Module: structure_post_extraction_processor.py

Post-extraction processing for CSD structures.

Reads raw HDF5 outputs (from StructureDataExtractor), computes derived features
(geometric, topological, contact-based) in GPU-accelerated batches, and writes
both raw and computed datasets into a new “*_processed.h5” container. Designed for
high throughput and minimal I/O overhead.

Dependencies
------------
h5py
numpy
torch
data_reader
data_writer
dataset_initializer
dimension_scanner
cell_utils
contact_utils
fragment_utils
geometry_utils
symmetry_utils
"""

import h5py
import logging
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Tuple, NamedTuple, Optional, Union, Any

from data_reader import RawDataReader
from data_writer import RawDataWriter, ComputedDataWriter
from dataset_initializer import DatasetInitializer
from dimension_scanner import scan_max_dimensions

from cell_utils import compute_cell_matrix_batch, compute_scaled_cell
from contact_utils import (
    compute_contact_atom_to_central_fragment_com_batch,
    compute_contact_fragment_indices_batch,
    compute_contact_is_hbond,
    compute_symmetric_contacts_batch, 
    compute_symmetric_hbonds_batch, 
    )
from fragment_utils import (
    compute_center_of_mass_batch,
    compute_centroid_batch,
    compute_inertia_tensor_batch,
    compute_quadrupole_tensor_batch,
    identify_rigid_fragments_batch,
    prepare_fragments_batch,
    )
from geometry_utils import (
    compute_angles_between_bonds_and_crystallographic_planes_frac_batch,
    compute_atom_vectors_to_point_batch,
    compute_best_fit_plane_batch,
    compute_bond_angles_batch,
    compute_bond_rotatability_batch,
    compute_distances_to_crystallographic_planes_frac_batch,
    compute_fragment_pairwise_vectors_and_distances_batch,
    compute_global_steinhardt_order_parameters_batch,
    compute_planarity_metrics_batch,
    compute_quaternions_from_rotation_matrices,
    compute_torsion_angles_batch
    )
from symmetry_utils import add_inter_cc_symmetry, add_inter_hb_symmetry

logger = logging.getLogger(__name__)

dt_str = h5py.string_dtype(encoding='utf-8')

class CrystalParams(NamedTuple):
    """
    Container for basic crystal parameters.
    
    Attributes
    ----------
    cell_lengths : torch.Tensor, shape (B, 3)
        Unit-cell lengths [a, b, c] for each structure in the batch.
    cell_angles : torch.Tensor, shape (B, 3)
        Unit-cell angles [α, β, γ] in degrees for each structure in the batch.
    """
    cell_lengths: torch.Tensor # (B, 3)
    cell_angles: torch.Tensor  # (B, 3)

class AtomParams(NamedTuple):
    """
    Container for atomic-level parameters.
    
    Attributes
    ----------
    labels : List[List[str]]
        Atom labels for each structure, padded to max_atoms.
    symbols : List[List[str]]
        Atomic symbols for each structure, padded to max_atoms.
    coords : torch.Tensor, shape (B, max_atoms, 3)
        Cartesian coordinates of each atom.
    frac_coords : torch.Tensor, shape (B, max_atoms, 3)
        Fractional coordinates of each atom.
    mask : torch.BoolTensor, shape (B, max_atoms)
        Boolean mask indicating valid atom entries.
    weights : torch.Tensor, shape (B, max_atoms)
        Atomic weights.
    charges : torch.Tensor, shape (B, max_atoms)
        Partial charges per atom.
    """
    labels: List[List[str]]   # (B, max_atoms)
    symbols: List[List[str]]  # (B, max_atoms)
    coords: torch.Tensor      # (B, max_atoms, 3)
    frac_coords: torch.Tensor # (B, max_atoms, 3)
    mask: torch.BoolTensor    # (B, max_atoms)
    weights: torch.Tensor     # (B, max_atoms)
    charges: torch.Tensor     # (B, max_atoms)

class BondParams(NamedTuple):
    """
    Container for bond-level parameters.
    
    Attributes
    ----------
    atom1_idx : torch.LongTensor, shape (B, max_bonds)
        Index of first atom in each bond.
    atom2_idx : torch.LongTensor, shape (B, max_bonds)
        Index of second atom in each bond.
    bond_type : torch.Tensor, shape (B, max_bonds)
        Numeric or categorical encoding of bond types.
    is_rotatable_raw : torch.BoolTensor, shape (B, max_bonds)
        Initial mask for bond rotatability.
    is_cyclic : torch.BoolTensor, shape (B, max_bonds)
        Indicates if bond is part of a ring.
    mask : torch.BoolTensor, shape (B, max_bonds)
        Boolean mask indicating valid bond entries.
    """
    atom1_idx: torch.LongTensor        # (B, max_bonds)
    atom2_idx: torch.LongTensor        # (B, max_bonds)
    bond_type: torch.Tensor            # (B, max_bonds)
    is_rotatable_raw: torch.BoolTensor # (B, max_bonds)
    is_cyclic: torch.BoolTensor        # (B, max_bonds)
    mask: torch.BoolTensor             # (B, max_bonds)

class InterCCParams(NamedTuple):
    """
    Container for intermolecular close-contact parameters.
    
    Attributes
    ----------
    central_atom : List[List[str]]
        Labels of central atoms in each contact.
    contact_atom : List[List[str]]
        Labels of contact atoms.
    central_atom_idx : torch.LongTensor, shape (B, C)
        Indices of central atoms.
    contact_atom_idx : torch.LongTensor, shape (B, C)
        Indices of contact atoms.
    central_atom_frac_coords : torch.Tensor, shape (B, C, 3)
        Fractional coords of central atoms.
    contact_atom_frac_coords : torch.Tensor, shape (B, C, 3)
        Fractional coords of contact atoms.
    lengths : torch.Tensor, shape (B, C)
        Contact distances.
    strengths : torch.Tensor, shape (B, C)
        Contact strength metrics.
    in_los : torch.Tensor, shape (B, C)
        Mask for line-of-sight contacts.
    symmetry_A : torch.Tensor, shape (B, C, 3, 3)
        Symmetry operation rotation matrices.
    symmetry_T : torch.Tensor, shape (B, C, 3)
        Symmetry operation translation vectors.
    symmetry_A_inv : torch.Tensor, shape (B, C, 3, 3)
        Inverse rotation matrices.
    symmetry_T_inv : torch.Tensor, shape (B, C, 3)
        Inverse translation vectors.
    """
    central_atom: List[List[str]]          # (B, C)
    contact_atom: List[List[str]]          # (B, C)
    central_atom_idx: torch.LongTensor     # (B, C)
    contact_atom_idx: torch.LongTensor     # (B, C)
    central_atom_frac_coords: torch.Tensor # (B, C, 3)
    contact_atom_frac_coords: torch.Tensor # (B, C, 3)
    lengths: torch.Tensor                  # (B, C)
    strengths: torch.Tensor                # (B, C)
    in_los: torch.Tensor                   # (B, C)
    symmetry_A: torch.Tensor               # (B, C, 3, 3)
    symmetry_T: torch.Tensor               # (B, C, 3)
    symmetry_A_inv: torch.Tensor           # (B, C, 3, 3)
    symmetry_T_inv: torch.Tensor           # (B, C, 3)

class InterHBParams(NamedTuple):
    """
    Container for intermolecular hydrogen-bond parameters.
    
    Attributes
    ----------
    central_atom : List[List[str]]
        Labels of hydrogen-bond donor atoms.
    hydrogen_atom : List[List[str]]
        Labels of hydrogen atoms.
    contact_atom : List[List[str]]
        Labels of acceptor atoms.
    central_atom_idx : torch.LongTensor, shape (B, H)
        Indices of donor atoms.
    hydrogen_atom_idx : torch.LongTensor, shape (B, H)
        Indices of hydrogen atoms.
    contact_atom_idx : torch.LongTensor, shape (B, H)
        Indices of acceptor atoms.
    central_atom_frac_coords : torch.Tensor, shape (B, H, 3)
        Fractional coords of donor atoms.
    hydrogen_atom_frac_coords : torch.Tensor, shape (B, H, 3)
        Fractional coords of hydrogen atoms.
    contact_atom_frac_coords : torch.Tensor, shape (B, H, 3)
        Fractional coords of acceptor atoms.
    lengths : torch.Tensor, shape (B, H)
        H-bond distances.
    angles : torch.Tensor, shape (B, H)
        H-bond angles.
    in_los : torch.Tensor, shape (B, H)
        Mask for line-of-sight H-bonds.
    symmetry_A : torch.Tensor, shape (B, H, 3, 3)
        Symmetry rotation matrices.
    symmetry_T : torch.Tensor, shape (B, H, 3)
        Symmetry translation vectors.
    symmetry_A_inv : torch.Tensor, shape (B, H, 3, 3)
        Inverse rotation matrices.
    symmetry_T_inv : torch.Tensor, shape (B, H, 3)
        Inverse translation vectors.
    """
    central_atom: List[List[str]]           # (B, H)
    hydrogen_atom: List[List[str]]          # (B, H)
    contact_atom: List[List[str]]           # (B, H)
    central_atom_idx: torch.LongTensor      # (B, H)
    hydrogen_atom_idx: torch.LongTensor     # (B, H)
    contact_atom_idx: torch.LongTensor      # (B, H)
    central_atom_frac_coords: torch.Tensor  # (B, H, 3)
    hydrogen_atom_frac_coords: torch.Tensor # (B, H, 3)
    contact_atom_frac_coords: torch.Tensor  # (B, H, 3)
    lengths: torch.Tensor                   # (B, H)
    angles: torch.Tensor                    # (B, H)
    in_los: torch.Tensor                    # (B, H)
    symmetry_A: torch.Tensor                # (B, H, 3, 3)
    symmetry_T: torch.Tensor                # (B, H, 3)
    symmetry_A_inv: torch.Tensor            # (B, H, 3, 3)
    symmetry_T_inv: torch.Tensor            # (B, H, 3)

class StructurePostExtractionProcessor:
    """
    Orchestrates post-extraction computation of derived structure features.
    
    Reads raw data from an HDF5 file, processes structures in GPU-accelerated
    batches to compute geometric, topological, and contact-based features, and
    writes both raw and computed data to a new processed HDF5 file.
    """
    def __init__(
            self,
            hdf5_path: Path,
            batch_size: int,
            device: Optional[Union[str, torch.device]] = None
            ):
        """
        Initialize the processor.
        
        Parameters
        ----------
        hdf5_path : Path
            Path to the raw HDF5 file containing extracted structure data.
        batch_size : int
            Number of structures to process per GPU batch.
        device : str or torch.device, optional
            Device specifier (e.g., 'cuda', 'cpu'); if None, selects CUDA if available.
        """
        # if nothing is passed, pick CUDA if you can
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device) if isinstance(device, str) else device

        self.hdf5_in_path  = Path(hdf5_path)
        self.hdf5_out_path = self.hdf5_in_path.parent / f"{self.hdf5_in_path.stem}_processed.h5"
        self.batch_size    = batch_size

    def run(self) -> None:
        """
        Execute the full post-extraction processing pipeline.
        
        Removes any existing processed file, reads raw data, initializes output
        datasets, processes structures in batches, and writes both raw and computed
        data to the output HDF5 file.
        """
        if self.hdf5_out_path.exists():
            logger.info(f"Removing existing processed file: {self.hdf5_out_path}")
            self.hdf5_out_path.unlink()
        
        # Open input file and set the data reader 
        h5_in = h5py.File(str(self.hdf5_in_path), 'r')
        self.reader = RawDataReader(h5_in)

        # Get refcodes 
        refcodes = list(h5_in['refcode_list'][...].astype(str))
        
        # Open output file and set the data writers
        h5_out = h5py.File(str(self.hdf5_out_path), 'w')
        self.raw_writer = RawDataWriter(h5_out)
        self.computed_writer = ComputedDataWriter(h5_out)
        
        # Initialize datasets
        dims = scan_max_dimensions(h5_in, refcodes)
        initializer = DatasetInitializer(h5_out=h5_out, refcodes=refcodes, dims=dims)
        initializer.initialize_datasets()
        
        # Log info
        N = len(refcodes)
        logger.info(f"Found {N} structures to process.")
        for start in range(0, N, self.batch_size):
            end = min(start + self.batch_size, N)
            batch = refcodes[start:end]
            logger.info(f"Processing structures {start + 1} to {end}")
            self._process_batch(start, batch, dims, h5_in, h5_out)

        # Close file
        h5_in.close()
        h5_out.close()
        logger.info("Post-extraction fast processing complete.")
        
    def _read_raw_data(
            self,
            batch: List[str],
            dims: Dict[str, int]
            ) -> (
                Dict[str, torch.Tensor],
                Dict[str, Union[List[List[str]], torch.Tensor]],
                Dict[str, Union[List[List[str]], torch.Tensor]],
                Dict[str, Union[List[List[str]], torch.Tensor]],
                Dict[str, Union[List[List[str]], torch.Tensor]],
                Dict[str, Union[List[List[str]], torch.Tensor]],
                Dict[str, Union[List[List[str]], torch.Tensor]]
                ):
        """
        Read raw HDF5 data and convert numeric arrays to torch tensors.
        
        Parameters
        ----------
        batch : List[str]
            List of refcodes for this batch.
        dims : Dict[str, int]
            Maximum dimensions from scan_max_dimensions, e.g. {'atoms':…, 'bonds':…, …}.
        
        Returns
        -------
        Tuple[
            Dict[str, torch.Tensor],
            Dict[str, Union[List[List[str]], torch.Tensor]],
            Dict[str, Union[List[List[str]], torch.Tensor]],
            Dict[str, Union[List[List[str]], torch.Tensor]],
            Dict[str, Union[List[List[str]], torch.Tensor]],
            Dict[str, Union[List[List[str]], torch.Tensor]],
            Dict[str, Union[List[List[str]], torch.Tensor]]
        ]
            Seven‐element tuple of dicts for:
            crystal_parameters,
            atom_parameters,
            bond_parameters,
            inter_cc_parameters,
            inter_hb_parameters,
            intra_cc_parameters,
            intra_hb_parameters.
        """

        # 1) Pull raw NumPy data on CPU
        crystal_parameters_np  = self.reader.read_crystal_parameters(batch)                                 
        atom_parameters_np     = self.reader.read_atoms(batch, dims['atoms'])                   
        bond_parameters_np     = self.reader.read_bonds(batch, dims['bonds'])
        inter_cc_parameters_np = self.reader.read_intermolecular_contacts(batch, dims['contacts_inter'])
        inter_hb_parameters_np = self.reader.read_intermolecular_hbonds(batch, dims['hbonds_inter'])
        intra_cc_parameters_np = self.reader.read_intramolecular_contacts(batch, dims['contacts_intra'])
        intra_hb_parameters_np = self.reader.read_intramolecular_hbonds(batch, dims['hbonds_intra'])
    
        # 2) Helper to convert NumPy arrays → torch.Tensor on GPU
        def _to_tensor_dict(np_dict):
            tensor_dict = {}
            for key, val in np_dict.items():
                if isinstance(val, np.ndarray):
                    # only convert numeric/bool/complextype arrays
                    if val.dtype.kind in ('f', 'i', 'u', 'b', 'c'):
                        tensor_dict[key] = torch.from_numpy(val).to(self.device, non_blocking=True)
                    else:
                        # leave object arrays (strings, etc.) as Python lists
                        tensor_dict[key] = val.tolist()
                else:
                    # e.g. plain Python lists of labels
                    tensor_dict[key] = val
            return tensor_dict
    
        # 3) Build final dicts
        crystal_parameters   = _to_tensor_dict(crystal_parameters_np)
        atom_parameters      = _to_tensor_dict(atom_parameters_np)
        bond_parameters      = _to_tensor_dict(bond_parameters_np)
        inter_cc_parameters  = _to_tensor_dict(inter_cc_parameters_np)
        inter_hb_parameters  = _to_tensor_dict(inter_hb_parameters_np)
        intra_cc_parameters  = _to_tensor_dict(intra_cc_parameters_np)
        intra_hb_parameters  = _to_tensor_dict(intra_hb_parameters_np)
    
        return (
            crystal_parameters,
            atom_parameters,
            bond_parameters,
            inter_cc_parameters,
            inter_hb_parameters,
            intra_cc_parameters,
            intra_hb_parameters
        )
    
    def _convert_symmetry(
            self,
            inter_cc_parameters: Dict[str, Any],
            inter_hb_parameters: Dict[str, Any]
            ) -> None:
        """
        Convert raw symmetry entries into rotation/translation tensors in place.
        
        Parameters
        ----------
        inter_cc_parameters : Dict[str, Any]
            Raw intermolecular contact parameters (string‐based fields).
        inter_hb_parameters : Dict[str, Any]
            Raw intermolecular H-bond parameters (string‐based fields).
        
        Returns
        -------
        None
        """
        # this mutates the dicts, replacing e.g. inter_cc_parameters['inter_cc_symmetry_A']
        # with the tensor versions
        add_inter_cc_symmetry(inter_cc_parameters, device=self.device)
        add_inter_hb_symmetry(inter_hb_parameters, device=self.device)
        
    def _unpack_parameters(
            self,
            raw_crystal:    dict,
            raw_atom:       dict,
            raw_bond:       dict,
            raw_inter_cc:   dict,
            raw_inter_hb:   dict
            ) -> Tuple[CrystalParams, AtomParams, BondParams, InterCCParams, InterHBParams]:
        """
        Unpack raw parameter dicts into structured NamedTuples.
        
        Parameters
        ----------
        raw_crystal : dict
            Raw crystal parameter fields.
        raw_atom : dict
            Raw atomic parameter fields.
        raw_bond : dict
            Raw bond parameter fields.
        raw_inter_cc : dict
            Raw intermolecular contact fields.
        raw_inter_hb : dict
            Raw intermolecular hydrogen-bond fields.
        
        Returns
        -------
        Tuple[CrystalParams, AtomParams, BondParams, InterCCParams, InterHBParams]
            NamedTuples bundling the needed tensors/lists for downstream steps.
        """
        crystal = CrystalParams(
            cell_lengths = raw_crystal['cell_lengths'],
            cell_angles  = raw_crystal['cell_angles']
        )
    
        atom = AtomParams(
            labels      = raw_atom['atom_label'],
            symbols     = raw_atom['atom_symbol'],
            coords      = raw_atom['atom_coords'],
            frac_coords = raw_atom['atom_frac_coords'],
            mask        = raw_atom['atom_mask'],
            weights     = raw_atom['atom_weight'],
            charges     = raw_atom['atom_charge']
            )
    
        bond = BondParams(
            atom1_idx        = raw_bond['bond_atom1_idx'],
            atom2_idx        = raw_bond['bond_atom2_idx'],
            bond_type        = raw_bond['bond_type'],
            is_rotatable_raw = raw_bond['bond_is_rotatable_raw'],
            is_cyclic        = raw_bond['bond_is_cyclic'],
            mask             = raw_bond['bond_mask']
            )
    
        inter_cc = InterCCParams(
            central_atom             = raw_inter_cc['inter_cc_central_atom'],
            contact_atom             = raw_inter_cc['inter_cc_contact_atom'],
            central_atom_idx         = raw_inter_cc['inter_cc_central_atom_idx'],
            contact_atom_idx         = raw_inter_cc['inter_cc_contact_atom_idx'],
            central_atom_frac_coords = raw_inter_cc['inter_cc_central_atom_frac_coords'],
            contact_atom_frac_coords = raw_inter_cc['inter_cc_contact_atom_frac_coords'],
            lengths                  = raw_inter_cc['inter_cc_length'],
            strengths                = raw_inter_cc['inter_cc_strength'],
            in_los                   = raw_inter_cc['inter_cc_in_los'],
            symmetry_A               = raw_inter_cc['inter_cc_symmetry_A'],
            symmetry_T               = raw_inter_cc['inter_cc_symmetry_T'],
            symmetry_A_inv           = raw_inter_cc['inter_cc_symmetry_A_inv'],
            symmetry_T_inv           = raw_inter_cc['inter_cc_symmetry_T_inv']
            )
    
        inter_hb = InterHBParams(
            central_atom              = raw_inter_hb['inter_hb_central_atom'],
            hydrogen_atom             = raw_inter_hb['inter_hb_hydrogen_atom'],
            contact_atom              = raw_inter_hb['inter_hb_contact_atom'],
            central_atom_idx          = raw_inter_hb['inter_hb_central_atom_idx'],
            hydrogen_atom_idx         = raw_inter_hb['inter_hb_hydrogen_atom_idx'],
            contact_atom_idx          = raw_inter_hb['inter_hb_contact_atom_idx'],
            central_atom_frac_coords  = raw_inter_hb['inter_hb_central_atom_frac_coords'],
            hydrogen_atom_frac_coords = raw_inter_hb['inter_hb_hydrogen_atom_frac_coords'],
            contact_atom_frac_coords  = raw_inter_hb['inter_hb_contact_atom_frac_coords'],
            lengths                   = raw_inter_hb['inter_hb_length'],
            angles                    = raw_inter_hb['inter_hb_angle'],
            in_los                    = raw_inter_hb['inter_hb_in_los'],
            symmetry_A                = raw_inter_hb['inter_hb_symmetry_A'],
            symmetry_T                = raw_inter_hb['inter_hb_symmetry_T'],
            symmetry_A_inv            = raw_inter_hb['inter_hb_symmetry_A_inv'],
            symmetry_T_inv            = raw_inter_hb['inter_hb_symmetry_T_inv']
            )
    
        return crystal, atom, bond, inter_cc, inter_hb
    
    def _compute_crystal_properties(
            self,
            cell_lengths: torch.Tensor,
            cell_angles: torch.Tensor
            ) -> Dict[str, torch.Tensor]:
        """
        Compute scaled cell parameters and real-space cell matrices.
        
        Parameters
        ----------
        cell_lengths : torch.Tensor, shape (B, 3)
            Unit-cell lengths [a, b, c] for each batch entry.
        cell_angles : torch.Tensor, shape (B, 3)
            Unit-cell angles [α, β, γ] in degrees for each batch entry.
        
        Returns
        -------
        Dict[str, torch.Tensor]
            {
              'scaled_cell': torch.Tensor,
              'cell_matrix': torch.Tensor
            }
        """
        scaled = compute_scaled_cell(
            cell_lengths,
            cell_angles,
            device=self.device
            )
        matrix = compute_cell_matrix_batch(
            cell_lengths,
            cell_angles,
            device=self.device
            )
        return {
            'scaled_cell': scaled,
            'cell_matrix': matrix
            }

    def _compute_atom_properties(
            self,
            atom_frac_coords: torch.Tensor,
            atom_mask: torch.BoolTensor
            ) -> Dict[str, torch.Tensor]:
        """
        Compute distances of all atoms to special crystallographic planes.
        
        Parameters
        ----------
        atom_frac_coords : torch.Tensor, shape (B, N, 3)
            Fractional coordinates of atoms.
        atom_mask : torch.BoolTensor, shape (B, N)
            Mask indicating valid atom positions.
        
        Returns
        -------
        Dict[str, torch.Tensor]
            {'atom_dist_to_special_planes': torch.Tensor}
        """
        dists = compute_distances_to_crystallographic_planes_frac_batch(
            atom_frac_coords,
            atom_mask,
            device=self.device
            )
        return {
            'atom_dist_to_special_planes': dists
            }

    def _compute_bond_properties(
            self,
            atom_frac_coords: torch.Tensor,
            bond_atom1_idx: torch.LongTensor,
            bond_atom2_idx: torch.LongTensor,
            bond_mask: torch.BoolTensor,
            atom_symbols: Any,
            bond_type: torch.Tensor,
            bond_is_cyclic: torch.BoolTensor,
            bond_is_rotatable: torch.BoolTensor
            ) -> Dict[str, torch.Tensor]:
        """
        Compute bond-plane angles and update rotatability mask.
        
        Parameters
        ----------
        atom_frac_coords : torch.Tensor, shape (B, N, 3)
            Fractional coordinates of atoms.
        bond_atom1_idx : torch.LongTensor, shape (B, M)
            First-atom indices for each bond.
        bond_atom2_idx : torch.LongTensor, shape (B, M)
            Second-atom indices for each bond.
        bond_mask : torch.BoolTensor, shape (B, M)
            Validity mask for bonds.
        atom_symbols : List[List[str]]
            Atomic symbols per structure.
        bond_type : torch.Tensor, shape (B, M)
            Encoded bond types.
        bond_is_cyclic : torch.BoolTensor, shape (B, M)
            Mask for bonds in rings.
        bond_is_rotatable : torch.BoolTensor, shape (B, M)
            Initial rotatability mask.
        
        Returns
        -------
        Dict[str, torch.Tensor]
            {
              'bond_vector_angles_to_special_planes': torch.Tensor,
              'bond_is_rotatable': torch.BoolTensor
            }
        """
        angles = compute_angles_between_bonds_and_crystallographic_planes_frac_batch(
            atom_frac_coords,
            bond_atom1_idx,
            bond_atom2_idx,
            bond_mask,
            device=self.device
            )
        rotmask = compute_bond_rotatability_batch(
            atom_symbols,
            bond_atom1_idx,
            bond_atom2_idx,
            bond_is_cyclic,
            bond_type,
            bond_is_rotatable,
            device=self.device
            )
        return {
            'bond_vector_angles_to_special_planes': angles,
            'bond_is_rotatable':                    rotmask
        }

    def _compute_molecule_properties(
            self,
            atom_labels: Any,
            atom_coords: torch.Tensor,
            atom_mask: torch.BoolTensor,
            bond_atom1_idx: torch.LongTensor,
            bond_atom2_idx: torch.LongTensor,
            bond_mask: torch.BoolTensor
            ) -> Dict[str, torch.Tensor]:
        """
        Compute intra-molecular bond angles and torsion angles.
        
        Parameters
        ----------
        atom_labels : List[List[str]]
            Labels of atoms for each structure.
        atom_coords : torch.Tensor, shape (B, N, 3)
            Cartesian coordinates of atoms.
        atom_mask : torch.BoolTensor, shape (B, N)
            Mask indicating valid atoms.
        bond_atom1_idx : torch.LongTensor, shape (B, M)
            First-atom indices per bond.
        bond_atom2_idx : torch.LongTensor, shape (B, M)
            Second-atom indices per bond.
        bond_mask : torch.BoolTensor, shape (B, M)
            Validity mask for bonds.
        
        Returns
        -------
        Dict[str, torch.Tensor]
            {
              'bond_angle_id': List[List[str]],
              'bond_angle': torch.Tensor,
              'bond_angle_mask': torch.BoolTensor,
              'bond_angle_atom_idx': torch.LongTensor,
              'torsion_id': List[List[str]],
              'torsion': torch.Tensor,
              'torsion_mask': torch.BoolTensor,
              'torsion_atom_idx': torch.LongTensor
            }
        """
        angle_ids, angle_vals, angle_mask, angle_atom_idx = compute_bond_angles_batch(
            atom_labels,
            atom_coords,
            atom_mask,
            bond_atom1_idx,
            bond_atom2_idx,
            bond_mask,
            device=self.device
            )
        torsion_ids, torsion_vals, torsion_mask, torsion_atom_idx = compute_torsion_angles_batch(
            atom_labels,
            atom_coords,
            atom_mask,
            bond_atom1_idx,
            bond_atom2_idx,
            bond_mask,
            device=self.device
            )
        return {
            'bond_angle_id': angle_ids,
            'bond_angle': angle_vals,
            'bond_angle_mask': angle_mask,
            'bond_angle_atom_idx': angle_atom_idx,
            'torsion_id': torsion_ids,
            'torsion': torsion_vals,
            'torsion_mask': torsion_mask,
            'torsion_atom_idx': torsion_atom_idx
        }
    
    def _expand_inter_contacts(
            self,
            inter_cc,                # namedtuple with fields central_atom, contact_atom, central_atom_idx, contact_atom_idx,
                                     # central_atom_frac_coords, contact_atom_frac_coords, lengths, strengths, in_los,
                                     # symmetry_A, symmetry_T, symmetry_A_inv, symmetry_T_inv
            cell_matrix: torch.Tensor
            ) -> Dict[str, torch.Tensor]:
        """
        Expand all close contacts to include symmetry-equivalent images.
        
        Parameters
        ----------
        inter_cc : InterCCParams
            Intermolecular contact NamedTuple.
        cell_matrix : torch.Tensor, shape (B, 3, 3)
            Real-space cell matrices.
        
        Returns
        -------
        Dict[str, torch.Tensor]
            Expanded contact data with keys like
            'inter_cc_central_atom_coords', 'inter_cc_length', etc.
        """
        return compute_symmetric_contacts_batch(
            inter_cc.central_atom,
            inter_cc.contact_atom,
            inter_cc.central_atom_idx,
            inter_cc.contact_atom_idx,
            inter_cc.central_atom_frac_coords,
            inter_cc.contact_atom_frac_coords,
            inter_cc.lengths,
            inter_cc.strengths,
            inter_cc.in_los,
            inter_cc.symmetry_A,
            inter_cc.symmetry_T,
            inter_cc.symmetry_A_inv,
            inter_cc.symmetry_T_inv,
            cell_matrix,
            device=self.device
            )

    def _expand_inter_hbonds(
            self,
            inter_hb,                # namedtuple with fields central_atom, hydrogen_atom, contact_atom, central_atom_idx,
                                     # hydrogen_atom_idx, contact_atom_idx, central_atom_frac_coords,
                                     # hydrogen_atom_frac_coords, contact_atom_frac_coords, lengths, angles, in_los,
                                     # symmetry_A, symmetry_T, symmetry_A_inv, symmetry_T_inv
            cell_matrix: torch.Tensor
            ) -> Dict[str, torch.Tensor]:
        """
        Expand all hydrogen-bond contacts symmetrically.
        
        Parameters
        ----------
        inter_hb : InterHBParams
            Intermolecular H-bond NamedTuple.
        cell_matrix : torch.Tensor, shape (B, 3, 3)
            Real-space cell matrices.
        
        Returns
        -------
        Dict[str, torch.Tensor]
            Expanded H-bond data with keys like
            'inter_hb_central_atom_idx', 'inter_hb_angles', etc.
        """
        return compute_symmetric_hbonds_batch(
            inter_hb.central_atom,
            inter_hb.hydrogen_atom,
            inter_hb.contact_atom,
            inter_hb.central_atom_idx,
            inter_hb.hydrogen_atom_idx,
            inter_hb.contact_atom_idx,
            inter_hb.central_atom_frac_coords,
            inter_hb.hydrogen_atom_frac_coords,
            inter_hb.contact_atom_frac_coords,
            inter_hb.lengths,
            inter_hb.angles,
            inter_hb.in_los,
            inter_hb.symmetry_A,
            inter_hb.symmetry_T,
            inter_hb.symmetry_A_inv,
            inter_hb.symmetry_T_inv,
            cell_matrix,
            device=self.device
            )

    def _flag_hbond_contacts(
            self,
            cc_central_idx: torch.LongTensor,
            cc_contact_idx: torch.LongTensor,
            cc_mask: torch.BoolTensor,
            hb_central_idx: torch.LongTensor,
            hb_hydrogen_idx: torch.LongTensor,
            hb_contact_idx: torch.LongTensor,
            hb_mask: torch.BoolTensor
            ) -> torch.BoolTensor:
        """
        Flag which close contacts correspond to hydrogen bonds.
        
        Parameters
        ----------
        cc_central_idx : torch.LongTensor, shape (B, C)
            Central-atom indices for contacts.
        cc_contact_idx : torch.LongTensor, shape (B, C)
            Contact-atom indices for contacts.
        cc_mask : torch.BoolTensor, shape (B, C)
            Validity mask for contacts.
        hb_central_idx : torch.LongTensor, shape (B, H)
            Donor-atom indices for H-bonds.
        hb_hydrogen_idx : torch.LongTensor, shape (B, H)
            Hydrogen-atom indices for H-bonds.
        hb_contact_idx : torch.LongTensor, shape (B, H)
            Acceptor-atom indices for H-bonds.
        hb_mask : torch.BoolTensor, shape (B, H)
            Validity mask for H-bonds.
        
        Returns
        -------
        torch.BoolTensor, shape (B, C)
            Mask indicating which contacts are H-bonds.
        """
        return compute_contact_is_hbond(
            cc_central_idx,
            cc_contact_idx,
            cc_mask,
            hb_central_idx,
            hb_hydrogen_idx,
            hb_contact_idx,
            hb_mask,
            device=self.device
            )

    def _identify_contact_fragments(
            self,
            cc_central_idx: torch.LongTensor,
            cc_contact_idx: torch.LongTensor,
            atom_fragment_id: torch.LongTensor
            ) -> Dict[str, torch.LongTensor]:
        """
        Map each contact’s atoms back to their rigid-fragment IDs.
        
        Parameters
        ----------
        cc_central_idx : torch.LongTensor, shape (B, C)
            Central-atom indices for contacts.
        cc_contact_idx : torch.LongTensor, shape (B, C)
            Contact-atom indices for contacts.
        atom_fragment_id : torch.LongTensor, shape (B, N)
            Fragment ID per atom.
        
        Returns
        -------
        Dict[str, torch.LongTensor]
            {
              'inter_cc_central_atom_fragment_idx': torch.LongTensor,
              'inter_cc_contact_atom_fragment_idx': torch.LongTensor
            }
        """
        return compute_contact_fragment_indices_batch(
            cc_central_idx,
            cc_contact_idx,
            atom_fragment_id,
            device=self.device
            )

    def _compute_contact_com_vectors(
            self,
            cc_coords: torch.Tensor,
            cc_frac_coords: torch.Tensor,
            cc_fragment_idx: torch.LongTensor,
            frag_com_coords: torch.Tensor,
            frag_com_frac_coords: torch.Tensor,
            frag_structure_id: torch.LongTensor,
            frag_local_ids: torch.LongTensor
            ) -> Dict[str, torch.Tensor]:
        """
        Compute vectors & distances from contact atoms to central-fragment COM.
        
        Parameters
        ----------
        cc_coords : torch.Tensor, shape (B, C, 3)
            Cartesian coords of contact atoms.
        cc_frac_coords : torch.Tensor, shape (B, C, 3)
            Fractional coords of contact atoms.
        cc_fragment_idx : torch.LongTensor, shape (B, C)
            Fragment indices for contact atoms.
        frag_com_coords : torch.Tensor, shape (B, F, 3)
            Cartesian COM coords for each fragment.
        frag_com_frac_coords : torch.Tensor, shape (B, F, 3)
            Fractional COM coords for each fragment.
        frag_structure_id : torch.LongTensor, shape (B, F)
            Structure IDs for each fragment.
        frag_local_ids : torch.LongTensor, shape (B, F)
            Local fragment indices.
        
        Returns
        -------
        Dict[str, torch.Tensor]
            {
              'inter_cc_atom_to_central_com_vec': torch.Tensor,
              'inter_cc_atom_to_central_com_dist': torch.Tensor
            }
        """
        return compute_contact_atom_to_central_fragment_com_batch(
            cc_coords,
            cc_frac_coords,
            cc_fragment_idx,
            frag_com_coords,
            frag_com_frac_coords,
            frag_structure_id,
            frag_local_ids,
            device=self.device
            )
    
    def _compute_rigid_fragments(
            self,
            atom_mask: torch.BoolTensor,
            atom1_idx: torch.LongTensor,
            atom2_idx: torch.LongTensor,
            bond_is_rotatable: torch.BoolTensor
            ) -> torch.LongTensor:
        """
        Identify rigid fragments by grouping non-rotatable bonds.
        
        Parameters
        ----------
        atom_mask : torch.BoolTensor, shape (B, N)
            Mask indicating valid atoms.
        atom1_idx : torch.LongTensor, shape (B, M)
            First-atom indices per bond.
        atom2_idx : torch.LongTensor, shape (B, M)
            Second-atom indices per bond.
        bond_is_rotatable : torch.BoolTensor, shape (B, M)
            Mask for bonds that are rotatable.
        
        Returns
        -------
        torch.LongTensor, shape (B, N)
            Fragment ID assigned to each atom.
        """
        return identify_rigid_fragments_batch(
            atom_mask,
            atom1_idx,
            atom2_idx,
            bond_is_rotatable,
            device=self.device
        )
    
    def _prepare_fragments(
            self,
            atom_fragment_id: torch.LongTensor,
            atom_coords: torch.Tensor,
            atom_frac_coords: torch.Tensor,
            atom_weight: torch.Tensor,
            atom_charge: torch.Tensor,
            atom_mask: torch.BoolTensor,
            atom_symbols: List[List[str]]
            ) -> Dict[str, torch.Tensor]:
        """
        Build padded, per-fragment tensors in one batch.
        
        Parameters
        ----------
        atom_fragment_id : torch.LongTensor, shape (B, N)
            Fragment ID per atom.
        atom_coords : torch.Tensor, shape (B, N, 3)
            Cartesian atom coordinates.
        atom_frac_coords : torch.Tensor, shape (B, N, 3)
            Fractional atom coordinates.
        atom_weight : torch.Tensor, shape (B, N)
            Atomic weights.
        atom_charge : torch.Tensor, shape (B, N)
            Partial charges.
        atom_mask : torch.BoolTensor, shape (B, N)
            Validity mask for atoms.
        atom_symbols : List[List[str]]
            Atomic symbols per structure.
        
        Returns
        -------
        Dict[str, torch.Tensor]
            Fragment-level tensors such as
            'fragment_atom_coords', 'fragment_atom_weight',
            'fragment_structure_id', 'fragment_local_id', etc.
        """
        # Gather and code all element symbols in batch
        all_syms = set(s for row in atom_symbols for s in row)
        element_list   = sorted(all_syms)
        element_to_code= {el:i for i,el in enumerate(element_list)}
        code_to_element= element_list
        code_H         = element_to_code['H']

        # Build (B, N) long tensor of symbol‐codes
        B, N = atom_mask.shape
        symbol_codes = torch.full((B, N), -1, dtype=torch.long, device=self.device)
        for b, syms in enumerate(atom_symbols):
            codes = torch.tensor([element_to_code[s] for s in syms], device=self.device, dtype=torch.long)
            symbol_codes[b, :codes.shape[0]] = codes

        return prepare_fragments_batch(
            atom_fragment_id,
            atom_coords,
            atom_frac_coords,
            atom_weight,
            atom_charge,
            symbol_codes,
            code_to_element,
            code_H,
            device=self.device
            )
    
    def _compute_fragment_com_centroid(
            self,
            frag_coords: torch.Tensor,
            frag_frac_coords: torch.Tensor,
            frag_weight: torch.Tensor,
            frag_mask: torch.BoolTensor
            ) -> Dict[str, torch.Tensor]:
        """
        Compute each fragment’s center of mass and centroid.
        
        Parameters
        ----------
        frag_coords : torch.Tensor, shape (B, F, Nf, 3)
            Cartesian fragment atom coordinates.
        frag_frac_coords : torch.Tensor, shape (B, F, Nf, 3)
            Fractional fragment atom coordinates.
        frag_weight : torch.Tensor, shape (B, F, Nf)
            Atomic weights per fragment.
        frag_mask : torch.BoolTensor, shape (B, F, Nf)
            Validity mask for fragment atoms.
        
        Returns
        -------
        Dict[str, torch.Tensor]
            {
              'fragment_com_coords': torch.Tensor,
              'fragment_com_frac_coords': torch.Tensor,
              'fragment_centroid_coords': torch.Tensor,
              'fragment_centroid_frac_coords': torch.Tensor
            }
        """
        fragment_pos = compute_center_of_mass_batch(
            frag_coords,
            frag_frac_coords,
            frag_weight,
            frag_mask,
            device=self.device
            )
        fragment_pos.update(compute_centroid_batch(
            frag_coords,
            frag_frac_coords,
            frag_mask,
            device=self.device
            ))
        return fragment_pos
    
    def _compute_fragment_tensors(
            self,
            frag_coords: torch.Tensor,
            frag_weight: torch.Tensor,
            frag_charge: torch.Tensor,
            frag_mask: torch.BoolTensor,
            frag_com_coords: torch.Tensor
            ) -> Dict[str, torch.Tensor]:
        """
        Compute inertia & quadrupole tensors with eigen-decompositions.
        
        Parameters
        ----------
        frag_coords : torch.Tensor, shape (B, F, Nf, 3)
            Fragment atom coordinates.
        frag_weight : torch.Tensor, shape (B, F, Nf)
            Atomic weights per fragment.
        frag_charge : torch.Tensor, shape (B, F, Nf)
            Atomic charges per fragment.
        frag_mask : torch.BoolTensor, shape (B, F, Nf)
            Validity mask for fragment atoms.
        frag_com_coords : torch.Tensor, shape (B, F, 3)
            Fragment COM coordinates.
        
        Returns
        -------
        Dict[str, torch.Tensor]
            {
              'fragment_inertia_eigvals': torch.Tensor,
              'fragment_inertia_eigvecs': torch.Tensor,
              'fragment_quadrupole_eigvals': torch.Tensor,
              'fragment_quadrupole_eigvecs': torch.Tensor
            }
        """
        tensors = compute_inertia_tensor_batch(
            frag_coords,
            frag_weight,
            frag_mask,
            frag_com_coords,
            device=self.device
            )
        tensors.update(compute_quadrupole_tensor_batch(
            frag_coords,
            frag_charge,
            frag_mask,
            frag_com_coords,
            device=self.device
            ))
        return tensors
    
    def _compute_fragment_quaternions(
            self,
            frag_inertia_eigvecs: torch.Tensor,
            frag_quadrupole_eigvecs: torch.Tensor
            ) -> Dict[str, torch.Tensor]:
        """
        Convert fragment rotation matrices into quaternions.
        
        Parameters
        ----------
        frag_inertia_eigvecs : torch.Tensor, shape (B, F, 3, 3)
            Eigenvectors of inertia tensors.
        frag_quadrupole_eigvecs : torch.Tensor, shape (B, F, 3, 3)
            Eigenvectors of quadrupole tensors.
        
        Returns
        -------
        Dict[str, torch.Tensor]
            {
              'fragment_inertia_quaternions': torch.Tensor,
              'fragment_quadrupole_quaternions': torch.Tensor
            }
        """
        return {
            'fragment_inertia_quaternions': compute_quaternions_from_rotation_matrices(
                frag_inertia_eigvecs, device=self.device),
            'fragment_quadrupole_quaternions': compute_quaternions_from_rotation_matrices(
                frag_quadrupole_eigvecs, device=self.device)
            }

    def _compute_fragment_atom_vectors_to_com(
            self,
            frag_coords: torch.Tensor,
            frag_frac_coords: torch.Tensor,
            frag_mask: torch.BoolTensor,
            frag_com_coords: torch.Tensor,
            frag_com_frac_coords: torch.Tensor
            ) -> Dict[str, torch.Tensor]:
        """
        Compute vectors & distances from each fragment atom to its COM.
        
        Parameters
        ----------
        frag_coords : torch.Tensor, shape (B, F, Nf, 3)
            Cartesian fragment atom coordinates.
        frag_frac_coords : torch.Tensor, shape (B, F, Nf, 3)
            Fractional fragment atom coordinates.
        frag_mask : torch.BoolTensor, shape (B, F, Nf)
            Validity mask for fragment atoms.
        frag_com_coords : torch.Tensor, shape (B, F, 3)
            Cartesian fragment COM coordinates.
        frag_com_frac_coords : torch.Tensor, shape (B, F, 3)
            Fractional fragment COM coordinates.
        
        Returns
        -------
        Dict[str, torch.Tensor]
            {
              'fragment_atom_to_com_vec': torch.Tensor,
              'fragment_atom_to_com_dist': torch.Tensor
            }
        """
        return compute_atom_vectors_to_point_batch(
            frag_coords,
            frag_frac_coords,
            frag_mask,
            frag_com_coords,
            frag_com_frac_coords,
            device=self.device
            )
    
    def _compute_fragment_Ql(
            self,
            frag_atom_to_com_vec: torch.Tensor,
            frag_mask: torch.BoolTensor,
            frag_weight: torch.Tensor
            ) -> torch.Tensor:
        """
        Compute global Steinhardt Q_l order parameters for fragments.
        
        Parameters
        ----------
        frag_atom_to_com_vec : torch.Tensor, shape (B, F, Nf, 3)
            Atom-to-COM vectors for fragments.
        frag_mask : torch.BoolTensor, shape (B, F, Nf)
            Validity mask for fragment atoms.
        frag_weight : torch.Tensor, shape (B, F, Nf)
            Atomic weights for fragments.
        
        Returns
        -------
        torch.Tensor, shape (B, F, L)
            Global Steinhardt Q_l parameters per fragment.
        """
        return compute_global_steinhardt_order_parameters_batch(
            frag_atom_to_com_vec,
            frag_mask,
            frag_weight,
            device=self.device
            )
    
    def _compute_fragment_planarity(
            self,
            frag_coords: torch.Tensor,
            frag_heavy_mask: torch.BoolTensor
            ) -> Dict[str, torch.Tensor]:
        """
        Compute planarity metrics by fitting heavy-atom fragments to a plane.
        
        Parameters
        ----------
        frag_coords : torch.Tensor, shape (B, F, Nh, 3)
            Coordinates of heavy atoms in fragments.
        frag_heavy_mask : torch.BoolTensor, shape (B, F, Nh)
            Mask for heavy-atom positions.
        
        Returns
        -------
        Dict[str, torch.Tensor]
            {
              'fragment_plane_normal': torch.Tensor,
              'fragment_plane_centroid': torch.Tensor,
              'fragment_planarity_max': torch.Tensor,
              'fragment_planarity_rms': torch.Tensor,
              …
            }
        """
        plane = compute_best_fit_plane_batch(
            frag_coords,
            frag_heavy_mask,
            device=self.device
            )
        metrics = compute_planarity_metrics_batch(
            frag_coords,
            frag_heavy_mask,
            plane['fragment_plane_normal'],
            plane['fragment_plane_centroid'],
            device=self.device
            )
        plane.update(metrics)
        return plane
    
    def _process_batch(
            self,
            start: int,
            batch: List[str],
            dims: Dict,
            h5_in: h5py.File,
            h5_out: h5py.File
            ) -> None:
        """
        Process and write raw plus computed data for a batch of structures.
        
        Parameters
        ----------
        start : int
            Index offset in the global refcode list corresponding to this batch.
        batch : List[str]
            Refcode identifiers for this batch.
        dims : Dict[str, int]
            Maximum-dimension dict from scan_max_dimensions.
        h5_in : h5py.File
            Open HDF5 file handle for raw input data.
        h5_out : h5py.File
            Open HDF5 file handle for processed output data.
        
        Returns
        -------
        None
        """
        # 1) Read the raw data from input data file 
        (
            crystal_parameters,
            atom_parameters,
            bond_parameters,
            inter_cc_parameters,
            inter_hb_parameters,
            intra_cc_parameters,
            intra_hb_parameters
            ) = self._read_raw_data(batch, dims)
        
        # 2) Convert strings variables to torch-ready tensors
        self._convert_symmetry(inter_cc_parameters, inter_hb_parameters)
        
        # 3) Unpack usefull input parameters
        crystal, atom, bond, inter_cc, inter_hb = self._unpack_parameters(
            crystal_parameters, 
            atom_parameters, 
            bond_parameters,
            inter_cc_parameters, 
            inter_hb_parameters
            )
        
        # 4) Initialize dictionaries for computed parameters
        crystal_parameters_c  = {}
        atom_parameters_c     = {}
        bond_parameters_c     = {}
        molecule_parameters_c = {}
        inter_cc_parameters_c = {}
        inter_hb_parameters_c = {}
        fragment_parameters_c = {}

        
        # 5) Compute crystal level properties
        crystal_parameters_c.update(self._compute_crystal_properties(
            crystal.cell_lengths, 
            crystal.cell_angles
            ))

        # 6) Compute atom level properties
        atom_parameters_c.update(self._compute_atom_properties(
            atom.frac_coords, 
            atom.mask))
        
        # 7) Compute bond level properties
        bond_parameters_c.update(self._compute_bond_properties(
            atom.frac_coords,
            bond.atom1_idx,
            bond.atom2_idx,
            bond.mask,
            atom.symbols,
            bond.bond_type,
            bond.is_cyclic,
            bond.is_rotatable_raw
            ))
        
        # 8) Compute molecule level properties
        molecule_parameters_c.update(self._compute_molecule_properties(
            atom.labels,
            atom.coords,
            atom.mask,
            bond.atom1_idx,
            bond.atom2_idx,
            bond.mask
            ))
        
        # 9) Expand intermolecular close contacts
        inter_cc_parameters_c.update(self._expand_inter_contacts(
            inter_cc,
            crystal_parameters_c['cell_matrix']
            ))
        
        # 10) Expand intermolecular hydrogen bonds
        inter_hb_parameters_c.update(self._expand_inter_hbonds(
            inter_hb,
            crystal_parameters_c['cell_matrix']
            ))
        
        # 11) Determine in a close contact is actualy part of a hydrogen bond
        inter_cc_parameters_c['inter_cc_is_hbond'] = self._flag_hbond_contacts(
            inter_cc_parameters_c['inter_cc_central_atom_idx'],
            inter_cc_parameters_c['inter_cc_contact_atom_idx'],
            inter_cc_parameters_c['inter_cc_mask'],
            inter_hb_parameters_c['inter_hb_central_atom_idx'],
            inter_hb_parameters_c['inter_hb_hydrogen_atom_idx'],
            inter_hb_parameters_c['inter_hb_contact_atom_idx'],
            inter_hb_parameters_c['inter_hb_mask']
            )
        
        # 12) Identify rigid fragments
        atom_parameters_c['atom_fragment_id'] = self._compute_rigid_fragments(
            atom.mask,
            bond.atom1_idx,
            bond.atom2_idx,
            bond_parameters_c['bond_is_rotatable']
            )
        
        # 13) Prepare fragments for further processing
        fragment_parameters_c.update(self._prepare_fragments(
            atom_parameters_c['atom_fragment_id'],
            atom.coords,
            atom.frac_coords,
            atom.weights,
            atom.charges,
            atom.mask,
            atom.symbols
            ))
        
        # 14) Compute the center of mass and centroid for each fragment
        fragment_parameters_c.update(self._compute_fragment_com_centroid(
            fragment_parameters_c['fragment_atom_coords'],
            fragment_parameters_c['fragment_atom_frac_coords'],
            fragment_parameters_c['fragment_atom_weight'],
            fragment_parameters_c['fragment_atom_mask']
            ))
        
        # 15) Compute inertia and quadrupole tensors, eigenvectors and eigenvalues
        fragment_parameters_c.update(self._compute_fragment_tensors(
            fragment_parameters_c['fragment_atom_coords'],
            fragment_parameters_c['fragment_atom_weight'],
            fragment_parameters_c['fragment_atom_charge'],
            fragment_parameters_c['fragment_atom_mask'],
            fragment_parameters_c['fragment_com_coords']
            ))
        
        # 16) Compute the corresponding quaternions for the inertia/quadrupole rotation matrices
        fragment_parameters_c.update(self._compute_fragment_quaternions(
            fragment_parameters_c['fragment_inertia_eigvecs'],
            fragment_parameters_c['fragment_quadrupole_eigvecs']
            ))
        
        # 17) Compute the distances of the atoms for each fragment to the center of mass of the fragments
        fragment_parameters_c.update(self._compute_fragment_atom_vectors_to_com(
            fragment_parameters_c['fragment_atom_coords'],
            fragment_parameters_c['fragment_atom_frac_coords'],
            fragment_parameters_c['fragment_atom_mask'],
            fragment_parameters_c['fragment_com_coords'],
            fragment_parameters_c['fragment_com_frac_coords']
            ))
        
        # 18) Compute global Steinhardt Q_l for each fragment in Cartesian coords (excludes H atoms)
        fragment_parameters_c['fragment_Ql'] = self._compute_fragment_Ql(
            fragment_parameters_c['fragment_atom_to_com_vec'],
            fragment_parameters_c['fragment_atom_mask'],
            fragment_parameters_c['fragment_atom_weight']
            )
        
        # 19) Compute planarity metrics for the fragments (excludes H atoms)
        fragment_parameters_c.update(self._compute_fragment_planarity(
            fragment_parameters_c['fragment_atom_coords'],
            fragment_parameters_c['fragment_atom_heavy_mask']
            ))
        
        # 20) Compute pairwise distances for the atoms in each fragment (exclude H atoms)
        fragment_parameters_c.update(compute_fragment_pairwise_vectors_and_distances_batch(
            fragment_parameters_c['fragment_atom_frac_coords'],
            fragment_parameters_c['fragment_atom_mask'],
            fragment_parameters_c['fragment_atom_heavy_mask'],
            device=self.device,
            ))
        
        # 21) Identify which contacts are related to each fragment
        inter_cc_parameters_c.update(self._identify_contact_fragments(
            inter_cc_parameters_c['inter_cc_central_atom_idx'],
            inter_cc_parameters_c['inter_cc_contact_atom_idx'],
            atom_parameters_c['atom_fragment_id']
            ))
        
        # 22) Conpute vectors and distances from contact atoms to the center of mass of the central fragment
        inter_cc_parameters_c.update(self._compute_contact_com_vectors(
            inter_cc_parameters_c['inter_cc_central_atom_coords'],
            inter_cc_parameters_c['inter_cc_central_atom_frac_coords'],
            inter_cc_parameters_c['inter_cc_central_atom_fragment_idx'],
            fragment_parameters_c['fragment_com_coords'],
            fragment_parameters_c['fragment_com_frac_coords'],
            fragment_parameters_c['fragment_structure_id'],
            fragment_parameters_c['fragment_local_id']
            ))
        
        # 22) Write raw and computed data to output file
        self.raw_writer.write_raw_crystal_data(start, crystal_parameters)
        self.raw_writer.write_raw_atom_data(start, atom_parameters)
        self.raw_writer.write_raw_bond_data(start, bond_parameters)
        self.raw_writer.write_raw_intramolecular_contact_data(start, intra_cc_parameters)
        self.raw_writer.write_raw_intramolecular_hbond_data(start, intra_hb_parameters)
        
        self.computed_writer.write_computed_crystal_data(start, crystal_parameters_c)
        self.computed_writer.write_computed_atom_data(start, atom_parameters_c)
        self.computed_writer.write_computed_bond_data(start, bond_parameters_c)
        self.computed_writer.write_computed_molecule_data(start, molecule_parameters_c)
        self.computed_writer.write_computed_intermolecular_contact_data(start, inter_cc_parameters_c)
        self.computed_writer.write_computed_intermolecular_hbond_data(start, inter_hb_parameters_c)
        self.computed_writer.write_computed_fragment_data(start, fragment_parameters_c)
        
        
            
        
            
        
        
            