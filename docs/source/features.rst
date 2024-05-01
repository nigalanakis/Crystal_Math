Features
========

CrystalMath provides a comprehensive statistical analysis of molecular crystal structures from the CSD database and custom structures in the *.cif format. It offers deep insights into molecular packing trends, intermolecular interactions, and the topological nuances that dictate these patterns.

The algorithm begins with a systematic exploration of the CSD, extracting and analyzing topological and geometrical data. This method integrates a fundamental understanding that molecular crystals conform to specific geometrical constraints and topological patterns. Through statistical analysis, CrystalMath derives logical rules and predictive models that enhance our understanding of molecular structures.

This section outlines the main features of the Crystal Math software, which include analysis of existing structures within the CSD and predictions of molecular crystal structures.

Analysis of Existing Structures within the CSD
----------------------------------------------
This feature represents the algorithm's investigative aspect, wherein it meticulously explores the repository of the CSD to extract and analyze structural data. The process employs a sophisticated fragment-based approach to assess molecular geometry, allowing it to discern subtle nuances and patterns within the crystal structures.

This computational process is not merely a data retrieval mechanism. It involves the calculation of crucial geometrical and topological properties, including:

- **Relative orientation**
- **Plane intersections with unit cell vertices**
- **Close contacts**
- **Hydrogen bonds**
- **Void analysis in the unit cell**

These computations are invaluable, forming the bedrock of the dataset that the subsequent predictive stage will utilize. The adaptability of this feature allows researchers to set specific criteria, enabling the algorithm to target structures that bear direct relevance to their studies, thereby ensuring a customized, relevant, and rich analytical output.

Prediction of Molecular Crystal Structures
------------------------------------------
Building upon the robust foundation laid by the analytical phase, the prediction feature marks the algorithm’s leap into the realm of prospective crystallography. This innovative function does not merely extrapolate from existing data but employs a rigorous mathematical, geometrical, and topological framework to envision and predict feasible crystal structures.

Bypassing traditional methods that rely heavily on force fields and energy calculations, this feature stands out due to its unique approach, essentially rewriting the rules of crystal structure prediction. By utilizing the detailed insights gleaned from the analysis of existing CSD structures, the algorithm assesses countless possibilities and predicts structures that are not just theoretically plausible but ripe for synthesis and experimental verification.

Detailed Analysis of Existing CSD Structures
--------------------------------------------
The algorithm delves into the CSD, applying user-defined criteria to identify and analyze structures pertinent to your research. These criteria could range from the atomic species present in the crystal to more complex attributes such as:

- Space group
- :math:`Z^{\prime}` value
- Molecular weight for the components within the asymmetric unit

Fragment-Based Analysis
^^^^^^^^^^^^^^^^^^^^^^^
The script communicates with the CSD database, seeking structures that align with specific user-defined rules. Upon identifying the relevant structures, the algorithm proceeds to extract critical data, focusing particularly on geometric and topological properties that inform the subsequent prediction phase.

A pivotal aspect of the CSP Algorithm's analytical prowess hinges on its geometric interpretation of intermolecular forces, by extracting properties for the close contacts and hydrogen bonding within crystal structures. These interactions are not merely physical constraints but are insightful topological and energetic indicators that guide the strategic assembly of molecular crystals.

Geometrical and Topological Properties Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The extracted data encompasses several key molecular aspects, with calculations and analyses including, but not limited to:

- **Orientation Relative to Inertia Frame**: Assessing molecular and fragmentary orientation within the unit cell, referenced against their inertia frames. This analysis goes beyond simple spatial representation; it is a profound exploration of the positional relationship between molecular fragments and the encompassing lattice geometry. The algorithm calculates the orientations by establishing a molecule's inertia frame, a defined coordinate system based on the molecule's moment of inertia. This frame serves as a reference point, allowing for a standardized comparison of molecular orientations. With this approach, the algorithm can systematically analyze how different fragments within a molecule orient themselves relative to each other and their collective orientation within the unit cell.
- **Relative positions of principal planes of inertia**: The algorithm computes the distances of certain points in a unit cell to the pnincipal planes of inertia (planes perpendicular to the principal axes of inertia, passing through the center of mass of each fragment). This calculation is instrumental in understanding the molecule's spatial orientation and placement.
- **Inter-Fragment Correlations**: By observing the relative orientations of fragments within a molecule, the algorithm unveils potential correlations in geometric conformations. These insights are crucial for understanding the molecule's structural dynamics, offering clues about its stability, reactivity, or interactions with neighboring entities.
- **Molecule-Unit Cell Interplay**: Expand the analysis to explore how the molecule fits and orients itself within the unit cell. This exploration can reveal critical insights into whether the molecule's orientation is influenced by the unit cell's geometric constraints, contributing to a deeper understanding of the crystal packing phenomena.
- **Predictive Insights for New Structures**: By identifying trends and correlations between molecular orientation and unit cell geometry, the algorithm can hypothesize about probable orientations for molecules in novel crystal structures, providing a reliable foundation for anticipating the behavior of molecules in uncharted configurations.

In essence, the orientation analysis relative to the inertia frame is not a mere calculation but a holistic examination of the molecule's spatial narrative. It provides contextual insights that are indispensable for predicting how new molecular assemblies might accommodate themselves within various lattice frameworks, essentially influencing the design strategy for new materials with desired properties.

- **Close Contacts**: Traditional analysis of close contacts often stops at identifying distances shorter than the sum of van der Waals radii. However, the CSP Algorithm delves deeper, recognizing that the strength of these contacts is an extremely important topological property, intimately tied to the interaction energy's minimum. By examining a comprehensive matrix of atomic species pairs and their distribution across various space groups, the algorithm calculates the optimal strength of close contacts. In addition, it analyzes the spatial distribution of the contacts in respect to the center of mass for each fragment. This analysis provides a benchmark for constructing molecular crystals with judicious interatomic interactions, ensuring structural stability without compromising the lattice's integrity. These calculated parameters are instrumental during the prediction phase, where the algorithm utilizes this statistical backbone to forecast interaction energies, guiding the assembly of molecules within the crystal lattice in a manner that's energetically favorable.
- **Hydrogen Bonds**: The analysis of the hydrogen bonds within the crystal matrix, provide insights into their geometric configuration which is tied to their energetic profile. This understanding is crucial because hydrogen bonds impart significant directional character to molecular arrangements in crystal lattices, influencing both structure and properties. The CSP Algorithm evaluates the geometry of potential hydrogen bond, ensuring not only geometric precision but also the right balance of strength and directionality in these interactions. This information is vital for constructing viable hydrogen-bonded networks, especially in complex molecular crystals where these interactions dictate structural feasibility and stability.
- **Voids in Unit Cell**: Analyzing the van der Waals free volume and solvent-accessible surface within the crystal lattice provides insights into the potential for molecular movement, stability under pressure, or where guest molecules might reside.
