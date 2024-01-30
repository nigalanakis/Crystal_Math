# CrystalMath 

## Tools for systematic exploration of the molecular structures in the CSD database and for a tpological based CSP

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

- Analysis of Existing Structures within the CSD: This feature represents the algorithm's investigative aspect, wherein it meticulously explores the vast repository of the CSD to extract and analyze structural data. By employing a sophisticated fragment-based approach to assess molecular geometry, the algorithm can discern subtle nuances and patterns within the crystal structures. This process is not merely a data retrieval mechanism. It involves the calculation of crucial geometrical and topological properties, such as relative orientation, plane intersections with unit cell vertices, close contacts, hydrogen bonds, and void analysis in the unit cell. These computations are invaluable, forming the bedrock of the dataset that the subsequent predictive stage will utilize. What makes this feature particularly compelling is its adaptability; researchers can set specific criteria, enabling the algorithm to target structures that bear direct relevance to their studies, thereby ensuring a customized, relevant, and rich analytical output.

- Prediction of Molecular Crystal Structures: (To be added) Building upon the robust foundation laid by the analytical phase, the prediction feature marks the algorithm’s leap into the realm of prospective crystallography. This innovative function does not merely extrapolate from existing data but employs a rigorous mathematical, geometrical, and topological framework to envision and predict feasible crystal structures. Bypassing traditional methods that rely heavily on force fields and energy calculations, this feature stands out due to its unique approach, essentially rewriting the rules of crystal structure prediction. By utilizing the detailed insights gleaned from the analysis of existing CSD structures, the algorithm assesses countless possibilities and predicts structures that are not just theoretically plausible but ripe for synthesis and experimental verification. %This feature is revolutionary, propelling forward the boundaries of predictive science and offering new avenues for the discovery and creation of novel molecular materials.

