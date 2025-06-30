"""
Module: structure_data_extractor.py

Parallel extraction of raw CSD data into an HDF5 container.

This module defines:
- `_extract_one`: Retrieve and package raw crystal, molecule, contact, and hydrogen-bond data for a single refcode.
- `StructureDataExtractor`: Manage parallel batch extraction of multiple refcodes into an HDF5 file.

Dependencies
------------
numpy
h5py
ccdc
hdf5_utils
"""
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Union, Tuple

import numpy as np
import h5py
from ccdc import io
from concurrent.futures import ProcessPoolExecutor, as_completed

from hdf5_utils import initialize_hdf5_file

logger = logging.getLogger(__name__)

def _extract_one(
        refcode: str,
        filters: Dict[str, Any],
        center_molecule: bool
        ) -> Tuple[str, Dict[str, Any]]:
    """
    Retrieve raw fields for one CSD refcode.

    Parameters
    ----------
    refcode : str
        CSD identifier (e.g., "ABCDEF12").
    filters : Dict[str, Any]
        ExtractionConfig.filters may include:
        - 'target_z_prime_values'
        - 'target_space_groups'
        - 'center_molecule'
    center_molecule : bool
        If True, recenter the crystal before extracting coordinates.

    Returns
    -------
    Tuple[str, Dict[str, Any]]
        refcode : str
            Same as input.
        raw_data : Dict[str, Any]
            Contains:
            - 'crystal_data': dict of crystal properties (identifier, space_group,
              z_value, z_prime, cell_lengths, cell_angles, cell_volume,
              cell_density, packing_coefficient, symmetry_operators).
            - 'molecule_data': dict with:
                - 'atoms': mapping label → dict of atomic_symbol, coordinates,
                  fractional_coordinates, atomic_weight, atomic_number,
                  partial_charge, neighbour_labels, sybyl_type.
                - 'bonds': mapping id → dict of atom1, atom2, bond_type,
                  is_cyclic, is_rotatable, length.
            - 'intermolecular_contacts', 'intramolecular_contacts',
              'intermolecular_hbonds', 'intramolecular_hbonds': dicts of
              corresponding data (if any).

    Raises
    ------
    Exception
        If any CCDC call fails (entry not found, missing coordinates, etc.).
    """
    mode = filters.get("structure_list", ["csd-unique"])[0].lower()
    if mode == "cif":
        # refcode here is actually the filename (e.g. "foo.cif")
        cif_dir = Path(filters["structure_list"][1])
        cif_path = cif_dir / refcode
        # read crystal + molecule from CIF
        crystal = io.CrystalReader(str(cif_path))[0]
        molecule = io.MoleculeReader(str(cif_path))[0]
    else:
        # default CSD behavior
        reader   = io.EntryReader("CSD")
        entry    = reader.entry(refcode)
        crystal  = entry.crystal
        molecule = entry.molecule
    
    if center_molecule:
        crystal.centre_molecule()
        
    molecule.assign_bond_types()
    molecule.add_hydrogens(mode='missing', add_sites=True)
    molecule.assign_partial_charges()
    
    if not molecule.all_atoms_have_sites:
        return None

    # --- Crystal-level data ---
    crystal_data = {
        "identifier": crystal.identifier,
        "space_group": crystal.spacegroup_symbol,
        "z_value": crystal.z_value,
        "z_prime": crystal.z_prime,
        "cell_lengths": list(crystal.cell_lengths),
        "cell_angles": list(crystal.cell_angles),
        "cell_volume": crystal.cell_volume,
        "cell_density": crystal.calculated_density,
        "packing_coefficient": crystal.packing_coefficient,
        "symmetry_operators": crystal.symmetry_operators
        }

    # --- Molecule-level data ---
    atoms_dict: Dict[str, Any] = {}
    for atom in molecule.atoms:
        atoms_dict[atom.label] = {
            "atomic_symbol": atom.atomic_symbol,
            "coordinates": [atom.coordinates.x, atom.coordinates.y, atom.coordinates.z],
            "fractional_coordinates": [
                atom.fractional_coordinates.x,
                atom.fractional_coordinates.y,
                atom.fractional_coordinates.z
                ],
            "atomic_weight": atom.atomic_weight,
            "atomic_number": atom.atomic_number,
            "partial_charge": atom.partial_charge,
            "neighbour_labels": [nbr.label for nbr in atom.neighbours],
            "sybyl_type": atom.sybyl_type
            }

    bonds_dict: Dict[str, Any] = {}
    for bond in molecule.bonds:
        key = f"{bond.atoms[0].label}-{bond.atoms[1].label}"
        bonds_dict[key] = {
            "atom1": bond.atoms[0].label,
            "atom2": bond.atoms[1].label,
            "bond_type": str(bond.bond_type),
            "is_cyclic": bond.is_cyclic,
            "is_rotatable": bond.is_rotatable,
            "length": bond.length
            }

    molecule_data = {
        "atoms": atoms_dict,
        "bonds": bonds_dict
        }

    # --- Contacts ---
    inter_contact_data: Dict[str, Any] = {}
    intra_contact_data: Dict[str, Any] = {}
    for contact in crystal.contacts(intermolecular="Any", distance_range=(-3.0, 0.6)):
        cid = f"{contact.atoms[0].label}-{contact.atoms[1].label}"
        c_dict = {
            "central_atom": contact.atoms[0].label,
            "central_atom_coordinates": [
                contact.atoms[0].coordinates.x,
                contact.atoms[0].coordinates.y,
                contact.atoms[0].coordinates.z
                ],
            "central_atom_fractional_coordinates": [
                contact.atoms[0].fractional_coordinates.x,
                contact.atoms[0].fractional_coordinates.y,
                contact.atoms[0].fractional_coordinates.z
                ],
            "contact_atom": contact.atoms[1].label,
            "contact_atom_coordinates": [
                contact.atoms[1].coordinates.x,
                contact.atoms[1].coordinates.y,
                contact.atoms[1].coordinates.z
                ],
            "contact_atom_fractional_coordinates": [
                contact.atoms[1].fractional_coordinates.x,
                contact.atoms[1].fractional_coordinates.y,
                contact.atoms[1].fractional_coordinates.z
                ],
            "length": contact.length,
            "strength": contact.strength,
            "is_intermolecular": contact.intermolecular,
            "is_in_line_of_sight": contact.is_in_line_of_sight,
            "symmetry_operator": contact.symmetry_operators[1]
            }
        if contact.intermolecular:
            inter_contact_data[cid] = c_dict
        else:
            intra_contact_data[cid] = c_dict

    # --- Hydrogen bonds ---
    inter_hbond_data: Dict[str, Any] = {}
    intra_hbond_data: Dict[str, Any] = {}
    for hbond in crystal.hbonds(intermolecular="Any", distance_range=(-3.0, 0.6)):
        hid = f"{hbond.atoms[0].label}-{hbond.atoms[1].label}-{hbond.atoms[2].label}"
        hb_dict = {
            "central_atom": hbond.atoms[0].label,
            "central_atom_coordinates": [
                hbond.atoms[0].coordinates.x,
                hbond.atoms[0].coordinates.y,
                hbond.atoms[0].coordinates.z
                ],
            "central_atom_fractional_coordinates": [
                hbond.atoms[0].fractional_coordinates.x,
                hbond.atoms[0].fractional_coordinates.y,
                hbond.atoms[0].fractional_coordinates.z
                ],
            "hydrogen_atom": hbond.atoms[1].label,
            "hydrogen_atom_coordinates": [
                hbond.atoms[1].coordinates.x,
                hbond.atoms[1].coordinates.y,
                hbond.atoms[1].coordinates.z
                ],
            "hydrogen_atom_fractional_coordinates": [
                hbond.atoms[1].fractional_coordinates.x,
                hbond.atoms[1].fractional_coordinates.y,
                hbond.atoms[1].fractional_coordinates.z
                ],
            "contact_atom": hbond.atoms[2].label,
            "contact_atom_coordinates": [
                hbond.atoms[2].coordinates.x,
                hbond.atoms[2].coordinates.y,
                hbond.atoms[2].coordinates.z
                ],
            "contact_atom_fractional_coordinates": [
                hbond.atoms[2].fractional_coordinates.x,
                hbond.atoms[2].fractional_coordinates.y,
                hbond.atoms[2].fractional_coordinates.z
                ],
            "length": hbond.da_distance,
            "angle": hbond.angle,
            "is_intermolecular": hbond.intermolecular,
            "is_in_line_of_sight": hbond.is_in_line_of_sight,
            "symmetry_operator": hbond.symmetry_operators[2]
            }
        if hbond.intermolecular:
            inter_hbond_data[hid] = hb_dict
        else:
            intra_hbond_data[hid] = hb_dict

    raw_data: Dict[str, Any] = {
        "crystal_data": crystal_data,
        "molecule_data": molecule_data,
        "intermolecular_contacts": inter_contact_data,
        "intermolecular_hbonds": inter_hbond_data,
        "intramolecular_contacts": intra_contact_data,
        "intramolecular_hbonds": intra_hbond_data
        }
    
    return refcode, raw_data

class StructureDataExtractor:
    """
    Manage parallel extraction of raw CSD data into an HDF5 container.

    This class:
    - Loads a list of refcodes from CSV (clustered or unique).
    - Initializes or overwrites an HDF5 file with a '/structures' group.
    - Batches refcodes and invokes `_extract_one()` in parallel.
    - Writes each structure's raw fields into its own HDF5 subgroup.

    Attributes
    ----------
    hdf5_path : Path
        File path for the output HDF5 container.
    filters : Dict[str, Any]
        ExtractionConfig.filters dictionary (e.g., data_directory, data_prefix,
        center_molecule, etc.).
    batch_size : int
        Number of refcodes to process concurrently per batch.
    reader : io.EntryReader
        CCDC EntryReader instance used by `_extract_one`.

    Methods
    -------
    run : () -> None
        Execute the full extraction pipeline: overwrite HDF5, initialize,
        load refcode list, and process each batch.
    _load_refcodes : () -> List[str]
        Read the CSV of refcodes to extract.
    _process_batch : (batch: List[str], h5: h5py.File) -> None
        Extract raw data for each batch of refcodes and write to HDF5.
    """

    def __init__(
            self,
            hdf5_path: Union[str, Path],
            filters: Dict[str, Any],
            batch_size: int
            ):
        """
        Initialize a StructureDataExtractor.

        Parameters
        ----------
        hdf5_path : Union[str, Path]
            Path for the HDF5 file to create or overwrite.
        filters : Dict[str, Any]
            ExtractionConfig.filters, containing:
            - 'data_directory'
            - 'data_prefix'
            - 'center_molecule'
            and other keys.
        batch_size : int
            Number of structures to extract concurrently per batch.
        """
        self.hdf5_path = Path(hdf5_path)
        self.filters = filters
        self.batch_size = batch_size
        self.reader = io.EntryReader("CSD")

    def run(self) -> None:
        """
        Perform the full raw-data extraction for all refcodes into HDF5.

        This method:
        - Deletes the existing HDF5 file at `hdf5_path` if it exists.
        - Calls `initialize_hdf5_file` to create the '/structures' group.
        - Loads all refcodes via `_load_refcodes()` and writes the '/refcode_list' dataset.
        - Processes refcodes in batches of size `batch_size`:
          - Submits each refcode to `ProcessPoolExecutor` running `_extract_one()`.
          - Collects `(refcode, raw_data)` tuples.
          - Writes raw fields under `/structures/<refcode>/...` as typed datasets.
        - Closes the HDF5 file.

        Raises
        ------
        FileNotFoundError
            If the refcode list CSV is missing.
        Exception
            If any CCDC call or HDF5 write operation fails.
        """
        # Overwrite existing file
        if self.hdf5_path.exists():
            logger.info(f"Overwriting existing HDF5 file: {self.hdf5_path}")
            self.hdf5_path.unlink()

        # Initialize file and group
        h5 = initialize_hdf5_file(
            self.hdf5_path, compression="gzip", chunk_size=self.batch_size
            )

        # Load and store refcode list
        refcodes = self._load_refcodes()
        dt = h5py.string_dtype(encoding="utf-8")
        h5.create_dataset(
            "refcode_list",
            data=np.array(refcodes, dtype=object),
            dtype=dt
            )

        total = len(refcodes)
        logger.info(f"{total} structures to extract (batch size {self.batch_size})")

        # Process in batches
        for start in range(0, total, self.batch_size):
            batch = refcodes[start : start + self.batch_size]
            logger.info(
                f"Extracting batch {start//self.batch_size + 1} (size {len(batch)})"
                )
            self._process_batch(batch, h5)

        h5.close()
        logger.info("Raw data extraction complete; HDF5 file closed.")

    def _load_refcodes(self) -> List[str]:
        """
        Read the list of refcodes to extract from a CSV file.

        The CSV filename is chosen based on `filters['structure_list'][0]`:
        - "csd-unique" → "{data_prefix}_refcode_families_unique.csv"
        - otherwise → "{data_prefix}_refcode_families_clustered.csv"
        
        If filters["structure_list"][0] == "cif", returns all .cif filenames
        from the directory in filters["structure_list"][1].

        Returns
        -------
        List[str]
            Refcode strings to process.

        Raises
        ------
        FileNotFoundError
            If the CSV file does not exist.
        """
        mode = self.filters.get("structure_list", ["csd-unique"])[0].lower()
        if mode == "cif":
            cif_dir = Path(self.filters["structure_list"][1])
            if not cif_dir.is_dir():
                raise FileNotFoundError(f"CIF directory {cif_dir} not found")
            # return filenames (with .cif) so refcode==filename
            return [p.name for p in sorted(cif_dir.glob("*.cif"))]

        base   = Path(self.filters["data_directory"])
        prefix = self.filters["data_prefix"]
        mode   = self.filters.get("structure_list", ["csd-unique"])[0]

        fname = (
            f"{prefix}_refcode_families_unique.csv"
            if mode == "csd-unique"
            else f"{prefix}_refcode_families_clustered.csv"
        )
        path = base / fname
        if not path.is_file():
            raise FileNotFoundError(f"Refcode list {path} not found")

        import pandas as pd
        df = pd.read_csv(path)
        return df["refcode"].astype(str).tolist()

    def _process_batch(self, batch: List[str], h5: h5py.File) -> None:
        """
        Extract raw data for a batch of refcodes and write to HDF5.

        This method:
        - Submits each refcode in `batch` to `ProcessPoolExecutor` calling
          `_extract_one(refcode, filters, center_flag)`.
        - Collects `(refcode, raw_data)` tuples for successful extractions.
        - For each tuple, writes datasets under `/structures/<refcode>/`:
          - Crystal-level data: identifier, space_group, z_value, etc.
          - Atom data: atom_label, atom_coords, atom_frac_coords, etc.
          - Bond data: bond_id, bond_atom1, bond_atom2, etc.
          - Contact and hydrogen-bond datasets for intermolecular and intramolecular interactions.

        Parameters
        ----------
        batch : List[str]
            Sub-list of refcodes to process.
        h5 : h5py.File
            Open HDF5 file returned by `initialize_hdf5_file()`.

        Raises
        ------
        Exception
            If any error occurs during extraction or dataset writing.
        """
        # 1) Parallel raw extraction
        center_flag = self.filters.get('center_molecule', False)
        max_workers = min(len(batch), (os.cpu_count() or 1) - 1)
        results: List[Tuple[str, Dict[str, Any]]] = []
    
        with ProcessPoolExecutor(max_workers=max_workers) as exe:
            futures = {
                exe.submit(_extract_one, ref, self.filters, center_flag): ref
                for ref in batch
                }
            for fut in as_completed(futures):
                out = fut.result()
                if out:
                    results.append(out)
    
        if not results:
            return
    
        # 2) Write each structure’s raw fields as typed datasets
        for refcode, raw in results:
            grp = h5["structures"].require_group(refcode)
    
            # --- Crystal-level data ---
            cd = raw["crystal_data"]
            grp.create_dataset("identifier",          data=cd["identifier"], dtype=h5py.string_dtype(encoding="utf-8"))
            grp.create_dataset("space_group",         data=cd["space_group"], dtype=h5py.string_dtype(encoding="utf-8"))
            grp.create_dataset("z_value",             data=cd["z_value"])
            grp.create_dataset("z_prime",             data=cd["z_prime"])
            grp.create_dataset("cell_volume",         data=cd["cell_volume"])
            grp.create_dataset("cell_density",        data=cd["cell_density"])
            grp.create_dataset("packing_coefficient", data=cd["packing_coefficient"])
            grp.create_dataset("cell_lengths",        data=np.array(cd["cell_lengths"], dtype=np.float32))
            grp.create_dataset("cell_angles",         data=np.array(cd["cell_angles"], dtype=np.float32))
            grp.create_dataset("symmetry_operators",  data=np.array(cd["symmetry_operators"], dtype=object), dtype=h5py.string_dtype(encoding="utf-8"))
    
            # --- Molecule atoms ---
            atoms       = raw["molecule_data"]["atoms"]
            labels      = list(atoms.keys())
            N           = len(labels)
            coords      = np.zeros((N, 3), dtype=np.float32)
            frac_coords = np.zeros((N, 3), dtype=np.float32)
            weights     = np.zeros((N,), dtype=np.float32)
            numbers     = np.zeros((N,), dtype=np.int32)
            charges     = np.zeros((N,), dtype=np.float32)
            symbols     = []
            sybyl       = []
            neighbours  = []
            for i, lbl in enumerate(labels):
                p = atoms[lbl]
                coords[i]      = p["coordinates"]
                frac_coords[i] = p["fractional_coordinates"]
                weights[i]     = p["atomic_weight"]
                numbers[i]     = p["atomic_number"]
                charges[i]     = p["partial_charge"]
                symbols.append(p["atomic_symbol"])
                sybyl.append(p["sybyl_type"])
                neighbours.append(",".join(p["neighbour_labels"]))
    
            grp.create_dataset("atom_label",           data=np.array(labels, dtype=object), dtype=h5py.string_dtype(encoding="utf-8"))
            grp.create_dataset("atom_symbol",          data=np.array(symbols, dtype=object), dtype=h5py.string_dtype(encoding="utf-8"))
            grp.create_dataset("atom_coords",          data=coords)
            grp.create_dataset("atom_frac_coords",     data=frac_coords)
            grp.create_dataset("atom_weight",          data=weights)
            grp.create_dataset("atom_number",          data=numbers)
            grp.create_dataset("atom_charge",          data=charges)
            grp.create_dataset("atom_sybyl_type",      data=np.array(sybyl, dtype=object), dtype=h5py.string_dtype(encoding="utf-8"))
            grp.create_dataset("atom_neighbour_list",  data=np.array(neighbours, dtype=object), dtype=h5py.string_dtype(encoding="utf-8"))
    
            # --- Molecule bonds ---
            bonds     = raw["molecule_data"]["bonds"]

            bids = list(bonds.keys())
            M         = len(bonds)
            atom1_idx = np.zeros((M,), dtype=np.int32)
            atom2_idx = np.zeros((M,), dtype=np.int32)
            cyclic    = np.zeros((M,), dtype=bool)
            rotat     = np.zeros((M,), dtype=bool)
            lengths   = np.zeros((M,), dtype=np.float32)
            atom1     = []
            atom2     = []
            types     = []
            for j, (bid, bp) in enumerate(bonds.items()):
                atom1_idx[j] = labels.index(bp["atom1"])
                atom2_idx[j] = labels.index(bp["atom2"])
                cyclic[j]    = bp["is_cyclic"]
                rotat[j]     = bp["is_rotatable"]
                lengths[j]   = bp["length"]
                atom1.append(bp["atom1"])
                atom2.append(bp["atom2"])
                types.append(bp["bond_type"])
    
            grp.create_dataset("bond_id",           data=np.array(bids, dtype=object), dtype=h5py.string_dtype(encoding="utf-8"))
            grp.create_dataset("bond_atom1",        data=np.array(atom1, dtype=object), dtype=h5py.string_dtype(encoding="utf-8"))
            grp.create_dataset("bond_atom2",        data=np.array(atom2, dtype=object), dtype=h5py.string_dtype(encoding="utf-8"))
            grp.create_dataset("bond_atom1_idx",    data=atom1_idx)
            grp.create_dataset("bond_atom2_idx",    data=atom2_idx)
            grp.create_dataset("bond_type",         data=np.array(types, dtype=object), dtype=h5py.string_dtype(encoding="utf-8"))
            grp.create_dataset("bond_is_cyclic",    data=cyclic)
            grp.create_dataset("bond_is_rotatable", data=rotat)
            grp.create_dataset("bond_length",       data=lengths)
    
            # --- Contacts ---
            contacts_inter = raw.get("intermolecular_contacts", {})
            contacts_intra = raw.get("intramolecular_contacts", {})
            
            # Intermolecular contacts
            cids = list(contacts_inter.keys())
            if len(cids) == 0:
                # no contacts → empty 1D arrays for labels/indices and (0,3) for coords
                central             = np.empty((0,),   dtype=object)
                contact             = np.empty((0,),   dtype=object)
                central_idx         = np.empty((0,),   dtype=np.int32)
                contact_idx         = np.empty((0,),   dtype=np.int32)
                lengths             = np.empty((0,),   dtype=np.float32)
                strengths           = np.empty((0,),   dtype=np.float32)
                in_line             = np.empty((0,),   dtype=bool)
                sym                 = np.empty((0,),   dtype=object)
                central_coords      = np.zeros((0, 3), dtype=np.float32)
                contact_coords      = np.zeros((0, 3), dtype=np.float32)
                central_frac_coords = np.zeros((0, 3), dtype=np.float32)
                contact_frac_coords = np.zeros((0, 3), dtype=np.float32)
            else:
                central             = np.array([contacts_inter[c]["central_atom"] for c in cids], dtype=object)
                contact             = np.array([contacts_inter[c]["contact_atom"] for c in cids], dtype=object )
                central_idx         = np.array([labels.index(contacts_inter[c]["central_atom"]) for c in cids], dtype=np.int32)
                contact_idx         = np.array([labels.index(contacts_inter[c]["contact_atom"]) for c in cids], dtype=np.int32)
                lengths             = np.array([contacts_inter[c]["length"] for c in cids], dtype=np.float32)
                strengths           = np.array([contacts_inter[c]["strength"] for c in cids], dtype=np.float32)
                in_line             = np.array([contacts_inter[c]["is_in_line_of_sight"] for c in cids], dtype=bool)
                sym                 = np.array([contacts_inter[c]["symmetry_operator"] for c in cids], dtype=object)
                central_coords      = np.array([contacts_inter[c]["central_atom_coordinates"] for c in cids], dtype=np.float32)
                contact_coords      = np.array([contacts_inter[c]["contact_atom_coordinates"] for c in cids], dtype=np.float32)
                central_frac_coords = np.array([contacts_inter[c]["central_atom_fractional_coordinates"] for c in cids], dtype=np.float32)
                contact_frac_coords = np.array([contacts_inter[c]["contact_atom_fractional_coordinates"] for c in cids], dtype=np.float32)
            
            grp.create_dataset("inter_cc_id",                       data=np.array(cids, dtype=object), dtype=h5py.string_dtype(encoding="utf-8"))
            grp.create_dataset("inter_cc_central_atom",             data=central, dtype=h5py.string_dtype(encoding="utf-8"))
            grp.create_dataset("inter_cc_contact_atom",             data=contact, dtype=h5py.string_dtype(encoding="utf-8"))
            grp.create_dataset("inter_cc_central_atom_idx",         data=central_idx)
            grp.create_dataset("inter_cc_contact_atom_idx",         data=contact_idx)
            grp.create_dataset("inter_cc_length",                   data=lengths)
            grp.create_dataset("inter_cc_strength",                 data=strengths)
            grp.create_dataset("inter_cc_in_los",                   data=in_line)
            grp.create_dataset("inter_cc_symmetry",                 data=sym, dtype=h5py.string_dtype(encoding="utf-8"))
            grp.create_dataset("inter_cc_central_atom_coords",      data=central_coords)
            grp.create_dataset("inter_cc_contact_atom_coords",      data=contact_coords)
            grp.create_dataset("inter_cc_central_atom_frac_coords", data=central_frac_coords)
            grp.create_dataset("inter_cc_contact_atom_frac_coords", data=contact_frac_coords)
            
            # Intramolecular contacts
            cids = list(contacts_intra.keys())
            if len(cids) == 0:
                # no contacts → empty 1D arrays for labels/indices and (0,3) for coords
                central             = np.empty((0,),   dtype=object)
                contact             = np.empty((0,),   dtype=object)
                central_idx         = np.empty((0,),   dtype=np.int32)
                contact_idx         = np.empty((0,),   dtype=np.int32)
                lengths             = np.empty((0,),   dtype=np.float32)
                strengths           = np.empty((0,),   dtype=np.float32)
                in_line             = np.empty((0,),   dtype=bool)
                central_coords      = np.zeros((0, 3), dtype=np.float32)
                contact_coords      = np.zeros((0, 3), dtype=np.float32)
                central_frac_coords = np.zeros((0, 3), dtype=np.float32)
                contact_frac_coords = np.zeros((0, 3), dtype=np.float32)
            else:
                central             = np.array([contacts_intra[c]["central_atom"] for c in cids], dtype=object)
                contact             = np.array([contacts_intra[c]["contact_atom"] for c in cids], dtype=object )
                central_idx         = np.array([labels.index(contacts_intra[c]["central_atom"]) for c in cids], dtype=np.int32)
                contact_idx         = np.array([labels.index(contacts_intra[c]["contact_atom"]) for c in cids], dtype=np.int32)
                lengths             = np.array([contacts_intra[c]["length"] for c in cids], dtype=np.float32)
                strengths           = np.array([contacts_intra[c]["strength"] for c in cids], dtype=np.float32)
                in_line             = np.array([contacts_intra[c]["is_in_line_of_sight"] for c in cids], dtype=bool)
                central_coords      = np.array([contacts_intra[c]["central_atom_coordinates"] for c in cids], dtype=np.float32)
                contact_coords      = np.array([contacts_intra[c]["contact_atom_coordinates"] for c in cids], dtype=np.float32)
                central_frac_coords = np.array([contacts_intra[c]["central_atom_fractional_coordinates"] for c in cids], dtype=np.float32)
                contact_frac_coords = np.array([contacts_intra[c]["contact_atom_fractional_coordinates"] for c in cids], dtype=np.float32)
        
            grp.create_dataset("intra_cc_id",                       data=np.array(cids, dtype=object), dtype=h5py.string_dtype(encoding="utf-8"))
            grp.create_dataset("intra_cc_central_atom",             data=central, dtype=h5py.string_dtype(encoding="utf-8"))
            grp.create_dataset("intra_cc_contact_atom",             data=contact, dtype=h5py.string_dtype(encoding="utf-8"))
            grp.create_dataset("intra_cc_central_atom_idx",         data=central_idx)
            grp.create_dataset("intra_cc_contact_atom_idx",         data=contact_idx)
            grp.create_dataset("intra_cc_length",                   data=lengths)
            grp.create_dataset("intra_cc_strength",                 data=strengths)
            grp.create_dataset("intra_cc_in_los",                   data=in_line)
            grp.create_dataset("intra_cc_central_atom_coords",      data=central_coords)
            grp.create_dataset("intra_cc_contact_atom_coords",      data=contact_coords)
            grp.create_dataset("intra_cc_central_atom_frac_coords", data=central_frac_coords)
            grp.create_dataset("intra_cc_contact_atom_frac_coords", data=contact_frac_coords)
            
            # --- Hydrogen bonds ---
            hbonds_inter = raw.get("intermolecular_hbonds", {})
            hbonds_intra = raw.get("intramolecular_hbonds", {})
            
            # Intermolecular hydrogen bonds
            hids = list(hbonds_inter.keys())
            if len(hids) == 0:
                # no contacts → empty 1D arrays for labels/indices and (0,3) for coords
                central              = np.empty((0,),   dtype=object)
                hydrogen             = np.empty((0,),   dtype=object)
                contact              = np.empty((0,),   dtype=object)
                central_idx          = np.empty((0,),   dtype=np.int32)
                hydrogen_idx         = np.empty((0,),   dtype=np.int32)
                contact_idx          = np.empty((0,),   dtype=np.int32)
                lengths              = np.empty((0,),   dtype=np.float32)
                angles               = np.empty((0,),   dtype=np.float32)
                in_line              = np.empty((0,),   dtype=bool)
                sym                  = np.empty((0,),   dtype=object)
                central_coords       = np.zeros((0, 3), dtype=np.float32)
                hydrogen_coords      = np.zeros((0, 3), dtype=np.float32)
                contact_coords       = np.zeros((0, 3), dtype=np.float32)
                central_frac_coords  = np.zeros((0, 3), dtype=np.float32)
                hydrogen_frac_coords = np.zeros((0, 3), dtype=np.float32)
                contact_frac_coords  = np.zeros((0, 3), dtype=np.float32)
            else:
                central              = np.array([hbonds_inter[h]["central_atom"]  for h in hids], dtype=object)
                hydrogen             = np.array([hbonds_inter[h]["hydrogen_atom"] for h in hids], dtype=object)
                contact              = np.array([hbonds_inter[h]["contact_atom"]  for h in hids], dtype=object) 
                central_idx          = np.array([labels.index(hbonds_inter[h]["central_atom"])  for h in hids], dtype=np.int32)
                hydrogen_idx         = np.array([labels.index(hbonds_inter[h]["hydrogen_atom"]) for h in hids], dtype=np.int32)
                contact_idx          = np.array([labels.index(hbonds_inter[h]["contact_atom"])  for h in hids], dtype=np.int32)
                lengths              = np.array([hbonds_inter[h]["length"] for h in hids], dtype=np.float32)
                angles               = np.array([hbonds_inter[h]["angle"] for h in hids], dtype=np.float32)
                in_line              = np.array([hbonds_inter[h]["is_in_line_of_sight"] for h in hids], dtype=bool)
                sym                  = np.array([hbonds_inter[h]["symmetry_operator"] for h in hids], dtype=object)
                central_coords       = np.array([hbonds_inter[h]["central_atom_coordinates"]  for h in hids], dtype=np.float32)
                hydrogen_coords      = np.array([hbonds_inter[h]["hydrogen_atom_coordinates"] for h in hids], dtype=np.float32)
                contact_coords       = np.array([hbonds_inter[h]["contact_atom_coordinates"]  for h in hids], dtype=np.float32)
                central_frac_coords  = np.array([hbonds_inter[h]["central_atom_fractional_coordinates"]  for h in hids], dtype=np.float32)
                hydrogen_frac_coords = np.array([hbonds_inter[h]["hydrogen_atom_fractional_coordinates"] for h in hids], dtype=np.float32)
                contact_frac_coords  = np.array([hbonds_inter[h]["contact_atom_fractional_coordinates"]  for h in hids], dtype=np.float32)
        
            grp.create_dataset("inter_hb_id",                        data=np.array(hids, dtype=object), dtype=h5py.string_dtype(encoding="utf-8"))
            grp.create_dataset("inter_hb_central_atom",              data=central, dtype=h5py.string_dtype(encoding="utf-8"))
            grp.create_dataset("inter_hb_hydrogen_atom",             data=hydrogen, dtype=h5py.string_dtype(encoding="utf-8"))
            grp.create_dataset("inter_hb_contact_atom",              data=contact, dtype=h5py.string_dtype(encoding="utf-8"))
            grp.create_dataset("inter_hb_central_atom_idx",          data=central_idx)
            grp.create_dataset("inter_hb_hydrogen_atom_idx",         data=hydrogen_idx)
            grp.create_dataset("inter_hb_contact_atom_idx",          data=contact_idx)
            grp.create_dataset("inter_hb_length",                    data=lengths)
            grp.create_dataset("inter_hb_angle",                     data=angles)
            grp.create_dataset("inter_hb_in_los",                    data=in_line)
            grp.create_dataset("inter_hb_symmetry",                  data=sym, dtype=h5py.string_dtype(encoding="utf-8"))
            grp.create_dataset("inter_hb_central_atom_coords",       data=central_coords)
            grp.create_dataset("inter_hb_hydrogen_atom_coords",      data=hydrogen_coords)
            grp.create_dataset("inter_hb_contact_atom_coords",       data=contact_coords)
            grp.create_dataset("inter_hb_central_atom_frac_coords",  data=central_frac_coords)
            grp.create_dataset("inter_hb_hydrogen_atom_frac_coords", data=hydrogen_frac_coords)
            grp.create_dataset("inter_hb_contact_atom_frac_coords",  data=contact_frac_coords)
        
            # Intramolecular hydrogen bonds
            hids = list(hbonds_intra.keys())
            if len(hids) == 0:
                # no contacts → empty 1D arrays for labels/indices and (0,3) for coords
                central              = np.empty((0,),   dtype=object)
                hydrogen             = np.empty((0,),   dtype=object)
                contact              = np.empty((0,),   dtype=object)
                central_idx          = np.empty((0,),   dtype=np.int32)
                hydrogen_idx         = np.empty((0,),   dtype=np.int32)
                contact_idx          = np.empty((0,),   dtype=np.int32)
                lengths              = np.empty((0,),   dtype=np.float32)
                angles               = np.empty((0,),   dtype=np.float32)
                in_line              = np.empty((0,),   dtype=bool)
                central_coords       = np.zeros((0, 3), dtype=np.float32)
                hydrogen_coords      = np.zeros((0, 3), dtype=np.float32)
                contact_coords       = np.zeros((0, 3), dtype=np.float32)
                central_frac_coords  = np.zeros((0, 3), dtype=np.float32)
                hydrogen_frac_coords = np.zeros((0, 3), dtype=np.float32)
                contact_frac_coords  = np.zeros((0, 3), dtype=np.float32)
            else:
                central              = np.array([hbonds_intra[h]["central_atom"]  for h in hids], dtype=object)
                hydrogen             = np.array([hbonds_intra[h]["hydrogen_atom"] for h in hids], dtype=object)
                contact              = np.array([hbonds_intra[h]["contact_atom"]  for h in hids], dtype=object) 
                central_idx          = np.array([labels.index(hbonds_intra[h]["central_atom"])  for h in hids], dtype=np.int32)
                hydrogen_idx         = np.array([labels.index(hbonds_intra[h]["hydrogen_atom"]) for h in hids], dtype=np.int32)
                contact_idx          = np.array([labels.index(hbonds_intra[h]["contact_atom"])  for h in hids], dtype=np.int32)
                lengths              = np.array([hbonds_intra[h]["length"] for h in hids], dtype=np.float32)
                angles               = np.array([hbonds_intra[h]["angle"] for h in hids], dtype=np.float32)
                in_line              = np.array([hbonds_intra[h]["is_in_line_of_sight"] for h in hids], dtype=bool)
                central_coords       = np.array([hbonds_intra[h]["central_atom_coordinates"]  for h in hids], dtype=np.float32)
                hydrogen_coords      = np.array([hbonds_intra[h]["hydrogen_atom_coordinates"] for h in hids], dtype=np.float32)
                contact_coords       = np.array([hbonds_intra[h]["contact_atom_coordinates"]  for h in hids], dtype=np.float32)
                central_frac_coords  = np.array([hbonds_intra[h]["central_atom_fractional_coordinates"]  for h in hids], dtype=np.float32)
                hydrogen_frac_coords = np.array([hbonds_intra[h]["hydrogen_atom_fractional_coordinates"] for h in hids], dtype=np.float32)
                contact_frac_coords  = np.array([hbonds_intra[h]["contact_atom_fractional_coordinates"]  for h in hids], dtype=np.float32)
        
            grp.create_dataset("intra_hb_id",                        data=np.array(hids, dtype=object), dtype=h5py.string_dtype(encoding="utf-8"))
            grp.create_dataset("intra_hb_central_atom",              data=central, dtype=h5py.string_dtype(encoding="utf-8"))
            grp.create_dataset("intra_hb_hydrogen_atom",             data=hydrogen, dtype=h5py.string_dtype(encoding="utf-8"))
            grp.create_dataset("intra_hb_contact_atom",              data=contact, dtype=h5py.string_dtype(encoding="utf-8"))
            grp.create_dataset("intra_hb_central_atom_idx",          data=central_idx)
            grp.create_dataset("intra_hb_hydrogen_atom_idx",         data=hydrogen_idx)
            grp.create_dataset("intra_hb_contact_atom_idx",          data=contact_idx)
            grp.create_dataset("intra_hb_length",                    data=lengths)
            grp.create_dataset("intra_hb_angle",                     data=angles)
            grp.create_dataset("intra_hb_in_los",                    data=in_line)
            grp.create_dataset("intra_hb_central_atom_coords",       data=central_coords)
            grp.create_dataset("intra_hb_hydrogen_atom_coords",      data=hydrogen_coords)
            grp.create_dataset("intra_hb_contact_atom_coords",       data=contact_coords)
            grp.create_dataset("intra_hb_central_atom_frac_coords",  data=central_frac_coords)
            grp.create_dataset("intra_hb_hydrogen_atom_frac_coords", data=hydrogen_frac_coords)
            grp.create_dataset("intra_hb_contact_atom_frac_coords",  data=contact_frac_coords)



