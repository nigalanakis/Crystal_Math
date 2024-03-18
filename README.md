# CrystalMath 

## Tools for systematic exploration of the molecular structures in the CSD databasetowards a topological based CSP

### 1. Installation

Anaconda is highly recommended for its ease of package management and environment handling. 
It comes with numerous scientific computing packages pre-installed, aiding in a smoother setup process. 
Visit the [https://www.anaconda.com/products/distribution](URL) to download and install the distribution. Please download the version that includes Python 3.9 or higher.

The software requires the following Python packages:
* ast
* datetime
* itertools
* json
* matplotlib
* networkx
* numpy
* os
* scipy
* re
* time

The current version also requires the installation of the CSD Python API.
Due to its specific installation instructions and licensing, refer to the official [https://downloads.ccdc.cam.ac.uk/documentation/API/installation_notes.html](URL) page for detailed guidance. 
Please, adhere strictly to their guidelines to ensure full functionality within the CSP algorithm environment. 
The API is crucial for the statistical analysis phase and for retrieving molecular structure data.

### 2. Features

1. Analysis of Existing Structures within the CSD: This feature represents the algorithm's investigative aspect, wherein it meticulously explores the repository of the CSD to extract and analyze structural data. By employing a sophisticated fragment-based approach to assess molecular geometry, the algorithm can discern subtle nuances and patterns within the crystal structures. This process is not merely a data retrieval mechanism. It involves the calculation of crucial geometrical and topological properties, such as relative orientation, plane intersections with unit cell vertices, close contacts, hydrogen bonds, and void analysis in the unit cell. These computations are invaluable, forming the bedrock of the dataset that the subsequent predictive stage will utilize. What makes this feature particularly compelling is its adaptability; researchers can set specific criteria, enabling the algorithm to target structures that bear direct relevance to their studies, thereby ensuring a customized, relevant, and rich analytical output.

2. Prediction of Molecular Crystal Structures: (To be added) Building upon the robust foundation laid by the analytical phase, the prediction feature marks the algorithm’s leap into the realm of prospective crystallography. This innovative function does not merely extrapolate from existing data but employs a rigorous mathematical, geometrical, and topological framework to envision and predict feasible crystal structures. Bypassing traditional methods that rely heavily on force fields and energy calculations, this feature stands out due to its unique approach, essentially rewriting the rules of crystal structure prediction. By utilizing the detailed insights gleaned from the analysis of existing CSD structures, the algorithm assesses countless possibilities and predicts structures that are not just theoretically plausible but ripe for synthesis and experimental verification.

#### 2.1 Analysis of Existing CSD Structures

The algorithm delves into the CSD, applying user-defined criteria to identify and analyze structures pertinent to your research. 

#### Fragment-Based Analysis

The script communicates with the CSD database, seeking structures that align with specific user-defined rules.  These criteria could range from the atomic species present in the crystal to more complex attributes such as space group, Z' value, and molecular weight for the components within the asymmetric unit.
	
Upon identifying the relevant structures, the algorithm proceeds to extract critical data, focusing particularly on geometric and topological properties that inform the subsequent prediction phase.

A pivotal aspect of the CSP Algorithm's analytical prowess hinges on its geometric interpretation of intermolecular forces, by extracting properties for the close contacts and hydrogen bonding within crystal structures. These interactions are not merely physical constraints but are insightful topological and energetic indicators that guide the strategic assembly of molecular crystals.

#### Geometrical and Topological Properties Analysis

The extracted data encompasses several key molecular aspects, with calculations and analyses including, but not limited to:

* Orientation Relative to Inertia Frame: Assessing molecular and fragmentary orientation within the unit cell, referenced against their inertia frames. This analysis goes beyond simple spatial representation; it is a profound exploration of the positional relationship between molecular fragments and the encompassing lattice geometry. The algorithm calculates the orientations by establishing a molecule's inertia frame, a defined coordinate system based on the molecule's moment of inertia. This frame serves as a reference point, allowing for a standardized comparison of molecular orientations. With this approach, the algorithm can systematically analyze how different fragments within a molecule orient themselves relative to each other and their collective orientation within the unit cell.
	
* Relative positions of principal planes of inertia: The algorithm computes the distances of certain points in a unit cell to the pnincipal planes of inertia (planes perpendicular to the principal axes of inertia, passing through the center of mass of each fragment). This calculation is instrumental in understanding the molecule's spatial orientation and placement.

* Inter-Fragment Correlations: By observing the relative orientations of fragments within a molecule, the algorithm unveils potential correlations in geometric conformations. These insights are crucial for understanding the molecule's structural dynamics, offering clues about its stability, reactivity, or interactions with neighboring entities.
		
* Molecule-Unit Cell Interplay: Expand the analysis to explore how the molecule fits and orients itself within the unit cell. This exploration can reveal critical insights into whether the molecule's orientation is influenced by the unit cell's geometric constraints, contributing to a deeper understanding of the crystal packing phenomena.
		
* Predictive Insights for New Structures: By identifying trends and correlations between molecular orientation and unit cell geometry, the algorithm can hypothesize about probable orientations for molecules in novel crystal structures, providing a reliable foundation for anticipating the behavior of molecules in uncharted configurations.

In essence, the orientation analysis relative to the inertia frame is not a mere calculation but a holistic examination of the molecule's spatial narrative. It provides contextual insights that are indispensable for predicting how new molecular assemblies might accommodate themselves within various lattice frameworks, essentially influencing the design strategy for new materials with desired properties.
	
* Close Contacts: Traditional analysis of close contacts often stops at identifying distances shorter than the sum of van der Waals radii. However, the CSP Algorithm delves deeper, recognizing that the strength of these contacts is an extremely important topological property, intimately tied to the interaction energy's minimum. By examining a comprehensive matrix of atomic species pairs and their distribution across various space groups, the algorithm calculates the optimal strength of close contacts. In addition, it analyzes the spatial distribution of the contacts in respect to the center of mass for each fragment. This analysis  provides a benchmark for constructing molecular crystals with judicious interatomic interactions, ensuring structural stability without compromising the lattice's integrity. These calculated parameters are instrumental during the prediction phase, where the algorithm utilizes this statistical backbone to forecast interaction energies, guiding the assembly of molecules within the crystal lattice in a manner that's energetically favorable.
	
* Hydrogen Bonds: The analysis of the hydrogen bonds within the crystal matrix, provide insights into their geometric configuration which is tied to their energetic profile. This understanding is crucial because hydrogen bonds impart significant directional character to molecular arrangements in crystal lattices, influencing both structure and properties. The CSP Algorithm evaluates the geometry of potential hydrogen bond, ensuring not only geometric precision but also the right balance of strength and directionality in these interactions. This information is vital for constructing viable hydrogen-bonded networks, especially in complex molecular crystals where these interactions dictate structural feasibility and stability.
	
* Voids in Unit Cell: Analyzing the van der Waals free volume and solvent-accessible surface within the crystal lattice provides insights into the potential for molecular movement, stability under pressure, or where guest molecules might reside.

#### Procedure

The source code can be executed by simply copying the source code in a parent directory (for example crystal_math). The first step is to create the necessary directories within the main working directory (crystal_math):
crystal_math

	├── csd_db_analysis
	│ └── db_data
	├── source_code
	│ └── input_files
	└── source_data

All the *.py code files provided should be placed in the Source Code directory. The input files "input_data_extraction.txt", "input_data_analysis.txt" are placed in the input files directory and the user generated "fragment_list.json" is placed in the source_data directory.

The first step is to modify the input_data_extraction.txt file based on the required criteria. The general format of the file is as follows:

	{"save_directory": "../csd_db_analysis/db_data/",
	 "get_refcode_families": True,
	 "cluster_refcode_families": True,
	 "get_unique_structures": True,
	 "get_structure_data": True,
	 "structure_list": ["csd-unique","all"],
	 "data_prefix": "homomolecular_crystals",
	 "unique_structures_clustering_method": "energy",
	 "target_species": ["C","H","N","O","F","Cl","Br","I","P","S"],
	 "target_space_groups": ["P1","P-1","P21","C2","Pc","Cc","P21/m","C2/m","P2/c","P21/c","P21/n","C2/c","P21212","P212121","Pca21","Pna21","Pbcn","Pbca","Pnma","R-3","I41/a"],
	 "target_z_prime_values": [1,2,3,4,5],
	 "molecule_weight_limit": 500.0,
	 "crystal_type": "homomolecular_crystal",
	 "molecule_formal_charges": [0],
	 "structures_to_exclude": ["ZEPDAZ04","BALDUP"],
	 "center_molecule": True,
	 "add_full_component": True,
	 "fragments_to_check_alignment": [],
	 "proposed_vectors_n_max": 5
	 }

* "save_directory": The directory to save data. It is recommended to use the default option
* "get_refcode_families": Set to True to extract all the refcode families from the csd. This option will create a file "csd_refcode_families.json" within the directory "../csd_db_analysis/db_data/".
* "cluster_refcode_families": Set to True to cluster the structures for each refcode family from the csd. This option will create a file "csd_refcode_families_clustered.json" within the directory "../csd_db_analysis/db_data/".
* "get_unique_structures": Set to True if to get the unique structures for each cluster of structures for each refcode family from the csd. This option will create a file "csd_refcode_families_unique_structures.json" within the directory "../csd_db_analysis/db_data/".
* "structure_list": [option_1,option_2].
	* option_1: Available values:
 		* "csd-all": Analyze all structures in the csd matching the criteria (this will analyze all the structures in the "csd_refcode_families.json").
 		* "csd-unique": Analyze all the unique structures in the csd matching the criteria (this will analyze all the structures in the "csd_refcode_families_unique_structures.json"). 
 		* "cif": Analyze user provided *.cif files.
   	* option_2: Available values:
   		* "all" or a list of structures if option_1 = "csd-all", "csd-unique". A list of structures may have the following formats: [[refcode_family, [refcode_index_1, refcode_index_2, ...]], [refcode_family, "all"]]. The refcode_family is a family of structures in the CSD database (for example "ACSALA" for the aspirin structures).  The list [refcode_index_1, refcode_index_2, ...] contains the indices of the structures in family to be analyzed (e.g. 0 for "ACSALA", 1 for "ACSALA01" etc). To analyze all structures in the family use  the list [refcode_family, "all"].
   	 	* A list of *.cif structures (complete path) to be analyzed if option_2 = "cif".
* "data_prefix": A user defined prefix that is placed in the front of the output files.
* "unique_structures_clustering_method": The method used to select the unique structures when clustering similar structures in the CSD. Currently the only available method is "energy", which selects the structure with the least intermolecular lattice energy calculated using the Gavezzotti-Filippini potentials implemented in the CSD Python API.
* target_species": A list of the allowed atomic species. Any structure with atomic species not in this list will be discarded.
* "target_space_groups": A list of the allowed space groups. Any structure with a space group not in this list will be discarded. The default option contains the 2 most common space groups.
* "target_z_prime_values":  A list of the allowed Z' values. Any structure with Z' value not in this list will be discarded.
* "molecule_weight_limit": The maximum allowed molecular weight for each component in the asymmetric unit.
* "crystal_type": List of structure types (eg. ["homomlecular"]). Available values:
	* "homomolecular"
 	* "co-crystal"
  	* "hydrate"
* "molecule_formal_charges": A list of the allowed molecular charges. Set to [0] to analyze neutral structures.
* "structures_to_exclude": For an unknown reason, there are a few structures in the CSD that can not be analyzed, as they produce a kernel error which causes the program to crash. Once such a structure is identified, it should be added to this list to avoid the crash. Unless a solution is found, it is strongly recommended to not modify this field.
* "center_molecule": Set to True if it is required to move the reference molecule in the referece unit cell (recommended).
* "add_full_component": Set to True to analyze the complete components in the asymmetric unit cell along with the fragments (This will account for the hydrogen atoms too).
* "proposed_vectors_n_max": A positive integer number represpenting the maximum value for each component of a crystallographic vector from the set n_max (recommended value: 5). 
