
# GMD-Biharmonic-cSGS-Surface

This repository contains the official Python source code and data assets for the numerical simulation and visualization workflows presented in the manuscript:

 "Mathematical Modelling of Sediment Thickness and Bedrock Topography Using the Biharmonic Equation and Sequential Gaussian Simulation"

As a complement to the article, we provide the Python source code covering the computational steps of "Algorithm 1: Constrained sequential Gaussian simulation (cSGS)", alongside the complete datasets and pre-computed simulation matrices required to reproduce the visualization and validation workflows.



## Requirements & Dependencies

To run these codes and visualization scripts, you need a standard Python installation (Python 3.8 or higher recommended) with the following scientific computing packages:

1. "NumPy" (>= 1.20.0)  For array manipulation and file handling ('.npy').
2. "SciPy" (>= 1.7.0)  For spatial data structures ('cKDTree'), grid interpolation ('exposed region griddata'), and matrix operations.
3. "Pandas"  For loading the well database Excel files ('.xlsx').
4. "Matplotlib"  For generating the final model visualization plots.



You can install all dependencies at once using pip:
bash
pip install numpy scipy



Project Data Repository: constrained sequential Gaussian simulation on raw data and Biharmonic Trend Analysis

This repository contains the datasets and pre-computed simulation matrices required to reproduce the visualization and validation workflow for conditional sequential Gaussian simulation (cSGS) and biharmonic trend analysis. 

To maintain a clean structural lineage between raw inputs and processed outputs, the assets are divided into two main directories: "Data/" and "Data_simulation/".



"Data/" (Observed and Raw Data)
This directory contains the foundational spatial data, raw wells, and regional boundary definitions used to construct the study area and ground-truth the models.

"GrunnvannsBorehull.xlsx": Raw Norwegian well database containing structural metadata.
    - Sheet "GrunnvannBrnn": Water borehole extraction data.
    - Sheet "EnergiBrnn": Energy borehole extraction data.
"exposed_bedrock.txt": 2D array grid representing the presence of exposed bedrock in the study domain.
"xcoord.txt" & "ycoord.txt": Spatial coordinate vectors defining the grid for the exposed bedrock area.
"local_NADAG.txt": Independent spatial validation dataset (NADAG) used for testing model accuracy against real-world observations.



"Data_simulation/" (Simulation Assets)
This directory contains the compiled '.npy' matrices generated from the computationally intensive simulation runs. These pre-computed assets allow users to quickly reproduce the visualizations without needing to re-run the entire geostatistical model.


Trend and Grid Definitions
	 - "X_locBdB.npy": Coordinates for the boundary locations within the regular grid.
	 - "moved_observations.npy": Array of observed well coordinates mapped/snapped to the nearest computational grid points.
	 - "IndAllDis.npy": Discretization indices used to map the trend profiles.
	 - "points4biharmonicsol.npy": Spatial coordinate points specifically designated for evaluating the biharmonic trend.

Biharmonic Model Outputs
    - "biharmonicSol.npy": The solved biharmonic trend values over the study domain.
    - "Ub_std.npy": The calculated standard deviation of the biharmonic trend.

cSGS Simulation Ensembles
    - "100cSGS_S_R.npy": Ensemble of 100 simulated fields generated via conditional sequential Gaussian simulation (cSGS) directly on the raw data. Each realization is stored as an '(N, 3)' array, where the first two columns represent the sorted 2D grid or nodal coordinate points, and the third column contains the corresponding simulated sediment thickness. 
    - "100cSGS_S_T.npy": Ensemble of 100 simulated fields generated via cSGS on the residual data, which are then added back to the biharmonic trend. Each realization is stored as an '(N, 3)' array, where the first two columns represent the sorted 2D grid or nodal coordinate points, and the third column contains the corresponding simulated sediment thickness.

Validation and Control Matrices
	 - "A_matOk_P2.npy": Ordinary Kriging validation matrix extracted at test locations.
	 - "A_matCV.npy": Cross-validation matrix for the cSGS model.
	 - "A_matCV_atX_locNew.npy": Extracted cSGS simulation values specifically at inequality bound locations.
	 - "A_matOk_inequality.npy": Extracted OK baseline simulation values at inequality bound locations for comparative bound checking.



Path Configuration Notes

  If you are running the main visualization script ("cSGS_model_visualization.py"), ensure your relative paths point to these directories exactly as named. The script utilizes the `pathlib` library to handle system-agnostic routing:

python
from pathlib import Path

base = Path("Data")
base1 = Path("Data_simulation")

Future Updates
Additional documentation, detailed setup instructions, and workflow refinements may be added in future repository updates as needed.


