def get_structures_list(input_parameters,reference_structures):
    '''
    Returns the structure list for the analysis
    
    Parameters
    ----------
    inpu_parameters : dict
        A dictionary with the user defined input parameters.
    reference_structures : dict
        The unique reference structures calculated based on the user defined
        criteria
        
    Returns
    -------
    structures_list : dict
        A dictionary with the structures to analyze
    '''
    # Create the structures list        
    structures_list = {}
    if input_parameters["structure_list"][0] in ["csd-all","csd-unique"]:
        if input_parameters["structure_list"][0] == "csd-all": 
            if input_parameters["structure_list"][1] == "all":
                for family in reference_structures:
                    for group in reference_structures[family]:
                        for structure in group:
                                structures_list[structure] = {}
            else:
                target_families = [families[0] for families in input_parameters["structure_list"][1]]
                target_families_structures = [families[1] for families in input_parameters["structure_list"][1]]
                for target_family, target_structures in zip(target_families, target_families_structures):
                    if target_structures == "all":
                        for group in reference_structures[target_family]:
                            for structure in group: 
                                structures_list[structure] = {}
                                
                    else:
                        structure_indices = [str(target_structure).zfill(2) if target_structure != 0 else '' for target_structure in target_structures ]
                        for index in structure_indices:
                            if target_family + index not in [structure for target_family in reference_structures for group in reference_structures[target_family] for structure in group]:
                                print(f'Structure {target_family + index} is not found in reference structures and will be excluded from the data extraction process.')
                                continue
                            structures_list[target_family + index] = {}
                            
        if input_parameters["structure_list"][0] == "csd-unique": 
            if input_parameters["structure_list"][1] == "all":
                for family in reference_structures:
                    for structure in reference_structures[family]:
                            structures_list[structure] = {}
            else:
                target_families = [families[0] for families in input_parameters["structure_list"][1]]
                target_families_structures = [families[1] for families in input_parameters["structure_list"][1]]
                for target_family, target_structures in zip(target_families, target_families_structures):
                    if target_structures == "all":
                        for structure in reference_structures[target_family]:
                            structures_list[structure] = {}
                            
                    else:
                        structure_indices = [str(target_structure).zfill(2) if target_structure != 0 else '' for target_structure in target_structures ]
                        for index in structure_indices:
                            if target_family + index not in reference_structures[target_family]:
                                print(f'Structure {target_family + index} is not found in reference structures and will be excluded from the data extraction process.')
                                continue
                            structures_list[target_family + index] = {}
                            
    elif input_parameters["structure_list"][0] == "cif":
        for family in reference_structures:
            for structure in reference_structures[family]:
                structures_list[structure] = {}
                
    return structures_list