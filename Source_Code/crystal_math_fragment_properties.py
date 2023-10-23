import ast 
import ccdc.search

def fragment_list(fragments_input_file):
    """ 
    Defines and returns the fragment list for the search 
    
    Parameters:
        -
        
    Returns:
        fragment_list (list): A list of the fragments for the substructure search
    """
    # Load fragment list
    with open(fragments_input_file) as f:
        data = f.read()
        
    fragment_list = ast.literal_eval(data)
    
    return fragment_list

def structure_fragments(fragment_list,atom_labels,atom_species,molecule):
    """ 
    Identify and returns the fragments in a molecule 
    
    Parameters:
        fragment_list (list): A list of the fragments for the substructure search
        atom_labels (list): A list of the labels of the atoms in the molecule
        atom_species (list): A list of the species of the atoms in the molecule
        
    Returns:
        fragments (dict): A dictionary with the identified fragments in the molecule
    """
    fragments = {}
    i_hit = 0
    for fragment in fragment_list:
        csd_fragment = ccdc.search.SMARTSSubstructure(fragment_list[fragment]["smarts"])
        fragmentSearch = ccdc.search.SubstructureSearch()
        fragmentID = fragmentSearch.add_substructure(csd_fragment)
        hits = fragmentSearch.search(molecule)
        for hit in hits:
            i_hit += 1
            key = "F" + str(i_hit).zfill(2) + "." + fragment
            hit_atoms = []
            hit_atoms_species = []
            hit_atoms_labels = []
            for at in hit.match_atoms():
                hit_atoms.append(list(atom_labels).index(at.label))
                hit_atoms_species.append(at.atomic_symbol)
                hit_atoms_labels.append(at.label)
            fragments[key] = {"smarts": fragment_list[fragment]["smarts"], 
                              "atoms": hit_atoms, 
                              "atom_species": hit_atoms_species, 
                              "atom_labels": hit_atoms_labels,
                              "ref_pos": fragment_list[fragment]["pos"],
                              "atoms_to_align": fragment_list[fragment]["atoms_to_align"]}
     
    # Remove subsets (sub-fragments)
    entries_to_remove = set()

    # Compare all pairs of keys
    for key1 in fragments:
        for key2 in fragments:
            if key1 != key2 and key1 not in entries_to_remove and key2 not in entries_to_remove:
                if fragments[key1]["smarts"] == fragments[key2]["smarts"]:
                    continue
                
                atoms1 = set(fragments[key1]['atom_labels'])
                atoms2 = set(fragments[key2]['atom_labels'])
                
                # Check if atoms of entry1 are subset of entry2
                if atoms1.issubset(atoms2):
                    entries_to_remove.add(key1)
                elif atoms2.issubset(atoms1):
                    entries_to_remove.add(key2)
    
    # Remove identified keys from the dictionary
    for key in entries_to_remove:
        del fragments[key]
        
    # Add a fragment number ID
    str_fragments = {}
    for i, key in enumerate(fragments):
        new_key = "F" + str(i + 1).zfill(2) + "." + key[4:]
        str_fragments[new_key] = fragments[key]
        
    return str_fragments