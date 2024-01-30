# CrystalMath 

## Tools for systematic exploration of the molecular structures in the CSD databasetowards a topological based CSP

### 1. Installation

Anaconda is highly recommended for its ease of package management and environment handling. 
It comes with numerous scientific computing packages pre-installed, aiding in a smoother setup process. 
Visit the [https://www.anaconda.com/products/distribution](URL) to download and install the distribution. Please download the version that includes Python 3.9 or higher.

The software requires the following Python packages:
* numpy
* scipy
* matplotlib
* json
* itertools
* re
* ast
* os
* datetime
* time

The current version also requires the installation of the CSD Python API.
Due to its specific installation instructions and licensing, refer to the official [https://downloads.ccdc.cam.ac.uk/documentation/API/installation_notes.html](URL) page for detailed guidance. 
Please, adhere strictly to their guidelines to ensure full functionality within the CSP algorithm environment. 
The API is crucial for the statistical analysis phase and for retrieving molecular structure data.

### 2. Features

1. Analysis of Existing Structures within the CSD: This feature represents the algorithm's investigative aspect, wherein it meticulously explores the vast repository of the CSD to extract and analyze structural data. By employing a sophisticated fragment-based approach to assess molecular geometry, the algorithm can discern subtle nuances and patterns within the crystal structures. This process is not merely a data retrieval mechanism. It involves the calculation of crucial geometrical and topological properties, such as relative orientation, plane intersections with unit cell vertices, close contacts, hydrogen bonds, and void analysis in the unit cell. These computations are invaluable, forming the bedrock of the dataset that the subsequent predictive stage will utilize. What makes this feature particularly compelling is its adaptability; researchers can set specific criteria, enabling the algorithm to target structures that bear direct relevance to their studies, thereby ensuring a customized, relevant, and rich analytical output.

2. Prediction of Molecular Crystal Structures: (To be added) Building upon the robust foundation laid by the analytical phase, the prediction feature marks the algorithm’s leap into the realm of prospective crystallography. This innovative function does not merely extrapolate from existing data but employs a rigorous mathematical, geometrical, and topological framework to envision and predict feasible crystal structures. Bypassing traditional methods that rely heavily on force fields and energy calculations, this feature stands out due to its unique approach, essentially rewriting the rules of crystal structure prediction. By utilizing the detailed insights gleaned from the analysis of existing CSD structures, the algorithm assesses countless possibilities and predicts structures that are not just theoretically plausible but ripe for synthesis and experimental verification.

#### 2.1 Analysis of Existing CSD Structures

The algorithm delves into the CSD, applying user-defined criteria to identify and analyze structures pertinent to your research. 

#### Fragment-Based Analysis

The script communicates with the CSD database, seeking structures that align with specific user-defined rules.  These criteria could range from the atomic species present in the crystal to more complex attributes like space group, Z' value, and molecular weight components within the asymmetric unit.
	
Upon identifying the relevant structures, the algorithm proceeds to extract critical data, focusing particularly on geometric and topological properties that inform the subsequent prediction phase.

A pivotal aspect of the CSP Algorithm's analytical prowess hinges on its geometric interpretation of intermolecular forces, by extracting properties for the close contacts and hydrogen bonding within crystal structures. These interactions are not merely physical constraints but are insightful topological and energetic indicators that guide the strategic assembly of molecular crystals.

#### Geometrical and Topological Properties Analysis

The extracted data encompasses several key molecular aspects, with calculations and analyses including, but not limited to:

* Orientation Relative to Inertia Frame: Assessing molecular and fragmentary orientation within the unit cell, referenced against their inertia frames. This analysis goes beyond simple spatial representation; it is a profound exploration of the positional relationship between molecular fragments and the encompassing lattice geometry. The algorithm calculates the orientations by establishing a molecule's inertia frame, a defined coordinate system based on the molecule's moment of inertia. This frame serves as a reference point, allowing for a standardized comparison of molecular orientations. With this approach, the algorithm can systematically analyze how different fragments within a molecule orient themselves relative to each other and their collective orientation within the unit cell.
	
* Inter-Fragment Correlations: By observing the relative orientations of fragments within a molecule, the algorithm unveils potential correlations in geometric conformations. These insights are crucial for understanding the molecule's structural dynamics, offering clues about its stability, reactivity, or interactions with neighboring entities.
		
* Molecule-Unit Cell Interplay: Expand the analysis to explore how the molecule fits and orients itself within the unit cell. This exploration can reveal critical insights into whether the molecule's orientation is influenced by the unit cell's geometric constraints, contributing to a deeper understanding of the crystal packing phenomena.
		
* Predictive Insights for New Structures: By identifying trends and correlations between molecular orientation and unit cell geometry, the algorithm can hypothesize about probable orientations for molecules in novel crystal structures, providing a reliable foundation for anticipating the behavior of molecules in uncharted configurations.

In essence, the orientation analysis relative to the inertia frame is not a mere calculation but a holistic examination of the molecule's spatial narrative. It provides contextual insights that are indispensable for predicting how new molecular assemblies might accommodate themselves within various lattice frameworks, essentially influencing the design strategy for new materials with desired properties.
	
* Intersections with Unit Cell Vertices: The algorithm computes intersections of planes, perpendicular to the principal axes of inertia, passing through the center of mass of each fragment, with the vertices of the unit cell. This calculation is instrumental in understanding the molecule's spatial orientation and placement.
	
* Close Contacts: Traditional analysis of close contacts often stops at identifying distances shorter than the sum of van der Waals radii. However, the CSP Algorithm delves deeper, recognizing that the strength of these contacts is an extremely important topological property, intimately tied to the interaction energy's minimum. By examining a comprehensive matrix of atomic species pairs and their distribution across various space groups, the algorithm calculates the optimal strength of close contacts. This analysis  provides a benchmark for constructing molecular crystals with judicious interatomic interactions, ensuring structural stability without compromising the lattice's integrity. These calculated parameters are instrumental during the prediction phase, where the algorithm utilizes this statistical backbone to forecast interaction energies, guiding the assembly of molecules within the crystal lattice in a manner that's energetically favorable.
	
* Hydrogen Bonds: The analysis of the hydrogen bonds within the crystal matrix, provide insights into their geometric configuration which is tied to their energetic profile. This understanding is crucial because hydrogen bonds impart significant directional character to molecular arrangements in crystal lattices, influencing both structure and properties. The CSP Algorithm evaluates the geometry of potential hydrogen bond, ensuring not only geometric precision but also the right balance of strength and directionality in these interactions. This information is vital for constructing viable hydrogen-bonded networks, especially in complex molecular crystals where these interactions dictate structural feasibility and stability.
	
* Voids in Unit Cell: Analyzing the van der Waals free volume and solvent-accessible surface within the crystal lattice provides insights into the potential for molecular movement, stability under pressure, or where guest molecules might reside.

#### Procedure

The process starts by configuring a JSON input file to specify the details of the analysis. It is required to set:
* save_directory: Specifies the path where all output data and results from the analysis will be stored, ensuring organized record-keeping and easy accessibility for subsequent review and usage.
* fragments_input_file: This key requires the path to a file containing reference data essential for the algorithm's fragment-based analysis. The file includes detailed information on the molecular fragments that the algorithm will consider during its data extraction and analysis phases. These predefined fragments serve as a benchmark for the algorithm, enabling it to recognize corresponding structures within the crystallographic data it processes. The correct definition of these fragments is paramount as they form the foundation upon which the algorithm identifies, aligns, and evaluates complex crystal structures.
The fragments_input_file is generated by crystal_math_reference_fragments.py script, specifically designed to standardize and prepare molecular fragment references for subsequent analytical procedures. This algorithm interfaces with a user-provided dictionary file, fragments_list.txt. Each entry within this dictionary delineates a molecular fragment, capturing its essence through SMARTS notation, atomic positions, masses, and a directive concerning atom alignment, as shown in the exemplary "benzene" entry below:
	
	"benzene": {
		"smarts": "c1ccccc1",
		"pos": [
		[ 1.3750, 0.0000, 0.0000],
		...
		],
		"mass": [12.0107, 12.0107, ...],
		"atoms_to_align": "all"
	}
	
A crucial aspect of these entries is the "atoms_to_align" key. This instruction designates specific atoms within the fragment, employed to synchronize the orientation of the reference fragment with a congruent fragment identified within a crystal structure. It accepts an "all" value, directing the algorithm to utilize all available atoms for alignment procedures. Alternatively, it accommodates a list of integers, signifying atom indexes, essential for instances where fragments exhibit mirror symmetries. This nuanced approach addresses scenarios where traditional SMARTS representation falls short, particularly for fragments bearing indistinguishable, mirror-image formations, such as the ambiguity in oxygens in a structure like "[\#6]S(=O)(=O)[NH2]".
	
Upon successful processing, crystal_math_reference_fragments.py generates the a refined dictionary saved in the file reference_fragment_list.txt. Each fragment entry within this product bears a resemblance to its precursor, with atomic positions "pos" recalibrated to the inertia frame, ensuring consistency and facilitating precise, error-resistant comparisons during subsequent stages of analysis. An illustrative excerpt of an entry is as follows:

	"benzene": {
		"smarts": "c1ccccc1",
		"pos": [
		[0.0, 1.375, 0.0],
		...
		],
		"atoms_to_align": "all"
  	}
 
* data_file: These entries designate the output files that the algorithm generates during the data extraction phase. Each file is tailored to capture extensive details on various facets of the analyzed crystal structures:
* structure_data_file: Accumulates general data on the explored structures, serving as a comprehensive record of the crystallographic investigations conducted.
* contacts_data_file: The close contact interactions identified between neighbouring molecules.
* h-bonds_data_file: Details the hydrogen bond contacts within the crystal lattice.
* plane_intersection_data_file: Records the intricate details of the principal inertia planes of molecular fragments intersect with the vertices of the crystal's unit cell, a geometric exploration that might reveal insights into the spatial orientation and constraints of molecules within the lattice.
* fragments_geometry_data_file: Gathers geometric data specific to the molecular fragments within the structures analyzed, offering insights into their shape and orientation.

* structures_list: A list of structures (a single structure, list of structures or the complete CSD database) to be queried during the analysis, allowing for a broad, comprehensive exploration of available crystallographic data.
	
* target_species, target_space_groups, target_z_prime_values, molecule_weight_limit: These parameters allow users to narrow down the scope of the analysis by defining specific chemical elements, space groups, Z' values, and molecular weight limits. This targeted approach ensures the algorithm's focus on structures that are most relevant to the user's research objectives.
	
* add_full_component, center_molecule: Boolean values that dictate certain aspects of the algorithm's operation, such as whether to consider the full component in the analysis and whether to geometrically center the molecule in the crystal lattice during investigation, affecting the computational representation of the structures.
	
* fragments_to_check_alignment, alignment_tolerance: These settings pertain to the investigation of fragment alignments within the crystal structure, specifying which fragments to analyze and the degree of deviation (tolerance) from ideal alignment the user considers acceptable.
	
* visualize_eigenvectors, proposed_vectors_n_max, write_data: Operational parameters that govern the algorithm's additional functionalities—like eigenvector visualization for understanding molecular orientation, the number of proposed vectors for consideration, and whether to write the resultant data onto the system for further analysis.

This configuration file essentially serves as the instruction manual for the algorithm, guiding its operation by specifying what, where, and how to analyze, thereby customizing the functionality to the user's specific research needs and objectives.

Begin by setting your criteria within the \texttt{crystal\_math\_csd\_data\_extraction.py} script, specifying the desired atomic species, space group, $Z^{\prime}$ value, and molecular weight restrictions for the components in the asymmetric unit.
	
Execute the script. It will initiate communication with the CSD, apply the set filters, and begin data extraction on the structures that fit the defined criteria.

Monitor the process. Depending on the complexity of your criteria and the number of applicable structures, this may take some time. The script is designed for robust data handling, ensuring comprehensive extraction without loss of information.
	
Upon completion, the script will generate an output (commonly in the form of structured data files) encompassing all the calculated properties and relevant analyses. It’s advisable to review this data to ensure consistency and completeness in preparation for Part 2 of the operational procedure.

#### Post-Extraction Analysis

After running crystal_math_csd_data_extraction.py, it is advisable to perform a thorough review of the extracted data. Look for trends, anomalies, or unique characteristics that could influence the predictive phase. This post-extraction analysis lays robust groundwork for Part 2, where the algorithm will use this dataset to generate and predict new feasible crystal structures.


