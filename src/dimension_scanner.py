"""
Module: dimension_scanner.py

Scan a raw HDF5 file to determine maximum ragged-array dimensions required
for atoms, bonds, inter-/intra-molecular contacts, and inter-/intra-molecular H-bonds.

Dependencies
------------
h5py
"""
import h5py
from typing import List, Dict

def scan_max_dimensions(h5_in: h5py.File, refcodes: List[str]) -> Dict[str, int]:
    """
    Compute the maximum sizes needed to pad ragged arrays across all structures.

    Parameters
    ----------
    h5_in : h5py.File
        Open HDF5 file containing `/structures/<refcode>` groups produced by the raw-data extractor.
    refcodes : List[str]
        List of all structure refcodes to scan.

    Returns
    -------
    Dict[str, int]
        Dictionary mapping:
        - 'atoms'          : maximum number of atoms across all structures
        - 'bonds'          : maximum number of bonds
        - 'contacts_inter' : maximum number of intermolecular contacts
        - 'contacts_intra' : maximum number of intramolecular contacts
        - 'hbonds_inter'   : maximum number of intermolecular H-bonds
        - 'hbonds_intra'   : maximum number of intramolecular H-bonds
        - 'fragments'      : recommended maximum fragments (equal to 'atoms')

    Raises
    ------
    KeyError
        If an expected subgroup or dataset is missing under `/structures/<refcode>`.
    """
    max_atoms = max_bonds = max_contacts_inter = max_contacts_intra = max_hbonds_inter = max_hbonds_intra = 0
    for ref in refcodes:
        grp = h5_in['structures'][ref]
        # Atoms
        nat = grp['atom_label'].shape[0]
        max_atoms = max(max_atoms, nat)
        # Bonds
        if 'bond_atom1_idx' in grp:
            nb = grp['bond_atom1_idx'].shape[0]
            max_bonds = max(max_bonds, nb)
        # Contacts
        if 'inter_cc_id' in grp:
            nc_inter = grp['inter_cc_id'].shape[0]
            max_contacts_inter = max(max_contacts_inter, nc_inter)
        if 'intra_cc_id' in grp:
            nc_intra = grp['intra_cc_id'].shape[0]
            max_contacts_intra = max(max_contacts_intra, nc_intra)
        # H-bonds
        if 'inter_hb_id' in grp:
            nh_inter = grp['inter_hb_id'].shape[0]
            max_hbonds_inter = max(max_hbonds_inter, nh_inter)
        if 'intra_hb_id' in grp:
            nh_intra = grp['intra_hb_id'].shape[0]
            max_hbonds_intra = max(max_hbonds_intra, nh_intra)
    return {
        'atoms': max_atoms,
        'bonds': max_bonds,
        'contacts_inter': max_contacts_inter,
        'contacts_intra': max_contacts_intra,
        'hbonds_inter': max_hbonds_inter,
        'hbonds_intra': max_hbonds_intra,
        'fragments': max_atoms
    }