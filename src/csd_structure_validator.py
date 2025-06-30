"""
Module: structure_validator.py

High-level validation of CSD crystal structures and molecules against filter criteria.

This module defines:
- StructureValidationResult: Container for validation outcomes.
- StructureValidator: Applies a sequence of checks to CCDC Crystal and Molecule objects.

Dependencies
------------
ccdc
"""

from dataclasses import dataclass
from typing import Optional, Set
from ccdc.molecule import Molecule
from ccdc.crystal import Crystal

@dataclass
class StructureValidationResult:
    """
    Result of validating a CSD Crystal and Molecule pair.

    Attributes
    ----------
    is_valid : bool
        True if all checks pass; False otherwise.
    failure_reason : Optional[str]
        Reason for failure when `is_valid` is False.
    """
    is_valid: bool
    failure_reason: Optional[str] = None

class StructureValidator:
    """
    Validate a CSD Crystal and Molecule against specified criteria.

    This class executes a pipeline of validation steps:
    - Zʹ value check
    - Space group check
    - Molecule preprocessing (bonds, hydrogens, charges)
    - Atom existence and coordinate validation
    - Crystal type determination
    - Molecular properties validation (charge, weight)
    - Species inclusion check

    Attributes
    ----------
    filters : dict
        Mapping of filter criteria. Supported keys:
        - target_z_prime_values : List[int]
        - target_space_groups : List[str]
        - crystal_type : List[str]
        - molecule_formal_charges : List[int]
        - molecule_weight_limit : float
        - target_species : List[str]
    """
    
    def __init__(self, filters: dict):
        self.filters = filters
        
    def validate(self, crystal: Crystal, molecule: Molecule) -> StructureValidationResult:
        """
        Run full validation pipeline on a crystal and molecule pair.

        Parameters
        ----------
        crystal : Crystal
            CCDC Crystal object to validate.
        molecule : Molecule
            CCDC Molecule object to validate.

        Returns
        -------
        StructureValidationResult
            Outcome of the validation, with `failure_reason` set when invalid.
        """
        # Z prime check
        if crystal.z_prime not in self.filters['target_z_prime_values']:
            return StructureValidationResult(False, "Invalid Z prime value")
            
        # Space group check
        if (self.filters['target_space_groups'] and 
            crystal.spacegroup_symbol not in self.filters['target_space_groups']):
            return StructureValidationResult(False, "Invalid space group")
            
        # Process molecule
        if not self._process_molecule(molecule):
            return StructureValidationResult(False, "Failed to process molecule")
            
        # Validate atoms
        atoms_result = self._validate_atoms(molecule)
        if not atoms_result.is_valid:
            return atoms_result
            
        # Crystal type validation
        crystal_type = self._determine_crystal_type(molecule)
        if crystal_type not in self.filters['crystal_type']:
            return StructureValidationResult(False, f"Invalid crystal type: {crystal_type}")
            
        # Molecular properties validation
        mol_result = self._validate_molecular_properties(molecule, crystal_type)
        if not mol_result.is_valid:
            return mol_result
            
        # Species validation
        if not self._validate_species(crystal):
            return StructureValidationResult(False, "Invalid chemical species")
            
        return StructureValidationResult(True)
        
    def _process_molecule(self, molecule: Molecule) -> bool:
        """
        Assign bond types, add hydrogens, and compute partial charges.

        Parameters
        ----------
        molecule : Molecule
            CCDC Molecule to process.

        Returns
        -------
        bool
            True if processing succeeds; False otherwise.
        """
        try:
            molecule.assign_bond_types()
            molecule.add_hydrogens(mode='missing')
            molecule.assign_partial_charges()
            return True
        except Exception:
            return False
            
    def _validate_atoms(self, molecule: Molecule) -> StructureValidationResult:
        """
        Ensure the molecule has atoms and each atom has coordinates.

        Parameters
        ----------
        molecule : Molecule
            CCDC Molecule to check.

        Returns
        -------
        StructureValidationResult
            is_valid=False if no atoms or missing coordinates; otherwise True.
        """
        try:
            atoms = molecule.atoms
            if not atoms:
                return StructureValidationResult(False, "No atoms found")
                
            if any(at.coordinates is None for at in atoms):
                return StructureValidationResult(False, "Missing atomic coordinates")
                
            return StructureValidationResult(True)
        except Exception:
            return StructureValidationResult(False, "Failed to access atomic data")
            
    def _determine_crystal_type(self, molecule: Molecule) -> str:
        """
        Classify a molecule as 'homomolecular', 'hydrate', or 'co-crystal'.

        Parameters
        ----------
        molecule : Molecule
            CCDC Molecule containing one or more components.

        Returns
        -------
        str
            One of {'homomolecular', 'hydrate', 'co-crystal'}.

        Notes
        -----
        - 'homomolecular' if all component formulas are identical.
        - 'hydrate' if any component formula equals 'H2 O1'.
        - 'co-crystal' otherwise.
        """
        components = [c.formula for c in molecule.components]
        
        if all(item == components[0] for item in components):
            return 'homomolecular'
            
        return 'hydrate' if 'H2 O1' in components else 'co-crystal'
        
    def _validate_molecular_properties(self, molecule: Molecule, crystal_type: str) -> StructureValidationResult:
        """
        Enforce formal charge and molecular weight limits for each component.

        Parameters
        ----------
        molecule : Molecule
            CCDC Molecule containing components.
        crystal_type : str
            Classification from `_determine_crystal_type`.

        Returns
        -------
        StructureValidationResult
            is_valid=False if any component violates charge or weight limits; otherwise True.
        """
        # Formal charge check for homomolecular crystals
        if crystal_type == 'homomolecular':
            if any(c.formal_charge not in self.filters['molecule_formal_charges'] 
                  for c in molecule.components):
                return StructureValidationResult(False, "Invalid formal charge")
                
        # Molecular weight check
        if any(c.molecular_weight > self.filters['molecule_weight_limit'] 
              for c in molecule.components):
            return StructureValidationResult(False, "Molecular weight exceeds limit")
            
        return StructureValidationResult(True)
        
    def _validate_species(self, crystal: Crystal) -> bool:
        """
        Check that all unique atomic species in the crystal formula are allowed.

        Parameters
        ----------
        crystal : Crystal
            CCDC Crystal object to check.

        Returns
        -------
        bool
            True if all species are in `filters['target_species']` or if that list is empty.
        """
        if not self.filters['target_species']:
            return True
            
        species = self._get_unique_species(crystal.formula)
        return all(s in self.filters['target_species'] for s in species)
        
    @staticmethod
    def _get_unique_species(formula: str) -> Set[str]:
        """
        Parse a chemical formula into unique element symbols.

        Parameters
        ----------
        formula : str
            Chemical formula string (e.g., "C6H12O6").

        Returns
        -------
        Set[str]
            Unique element symbols extracted from the formula.

        Notes
        -----
        - Element symbols start with an uppercase letter, optionally followed by lowercase letters.
        - Numeric characters are ignored.
        """
        species = set()
        current = ''
        
        for char in formula:
            if char.isupper():
                if current:
                    species.add(current)
                current = char
            elif char.islower():
                current += char
            elif current:
                species.add(current)
                current = ''
                
        if current:
            species.add(current)
            
        return species