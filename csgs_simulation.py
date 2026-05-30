"""
Code Title:       Constrained Sequential Gaussian Simulation (cSGS) Source Code
Paper Title:      Mathematical Modelling of Sediment Thickness and Bedrock Topography 
                  Using the Biharmonic Equation and Constrained Sequential Gaussian Simulation

What this code does:
    This is the main Python code for the cSGS method (Algorithm 1) from our paper. 
    It creates 2D maps of sediment thickness. It makes sure the final 
    maps always respect physical rules like making sure sediment thickness never 
    drops below zero, and matching the observed values found at real well locations.

    We designed this new source code to handle two different mapping jobs automatically:
    1. The Trend Map: It models variations (residuals) and adds them on 
       top of a smooth regional trend line calculated using the Biharmonic Equation.
    2. The Raw Data Map: It models observed sediment thickness values directly from 
       the data. 

Author:         Maregnesh-Mechal Wolde
Email:          maregm@hu.edu.et
Date:           May 2026
University:     Hawassa University (HU)
Target Journal: Geoscientific Model Development (GMD)

What you need to run it:
    - numpy (version 1.20.0 or higher)
    - scipy (version 1.7.0 or higher)

Functions in this file:
    - cSGS_Simulation_SourceCode: 
        The main calculator. It builds the random path across the map, searches 
        for neighboring wells, solves the Kriging math formulas safely, and double-checks 
        every guess to ensure it fits our physical boundaries.
    - Run_M_cSGS_on_residual_with_SedimentTrend (cSGS_S_T): 
        A loop function that lets us run the "Trend Map" calculation as many times 
        as you want to create multiple map options.
    - Run_M_cSGS_on_rawSediment (cSGS_S_R): 
        A loop function that lets you run the "Raw Data Map" calculation as many times 
        as you want to create multiple map options.
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from scipy.spatial import distance
from numpy.random import permutation

# =======================================================================================================================
# 1. Defnition of constrained sequential Gaussian simulation on raw data as well as on residual of trend then added to trend
# ===================================================================================================================

def cSGS_Simulation_SourceCode(data_points, Z_observation, grid_points, poinsBh1, U_bdis1, Ub_std1, Variogram_model,
                           Z_observationAll, Z_nan, Nugget, Sill, Range, alpha, dx, dy,
                           X_expAllInGigDom, use_trend=True):
   
    n_points = grid_points.shape[0]

    # ============================================================
    # 2. IDENTIFY MASKED / EXCLUDED POINTS
    # ============================================================
    tree_mask = cKDTree(grid_points)
    _, excluded_indices = tree_mask.query(X_expAllInGigDom)
    excluded_set = set(excluded_indices)

    # ============================================================
    # 3. RANDOM PATH GENERATION (EXCLUDING BEDROCK)
    # ============================================================
    r_filtered = [idx for idx in permutation(n_points) if idx not in excluded_set]

    # ============================================================
    # 4. TREND MATRIX INITIALIZATION (DYNAMIC TOGGLE)
    # ============================================================
    if use_trend and poinsBh1 is not None and U_bdis1 is not None:
        # Scenario A: Residual with Biharmonic Sediment Trend Framework
        Zmodel = griddata(poinsBh1, U_bdis1, grid_points, method='cubic', fill_value=0)
        Ubstd = np.abs(griddata(poinsBh1, Ub_std1, grid_points, method='cubic', fill_value=0))
        Ubstd = np.nan_to_num(Ubstd, nan=0.0)
    else:
        # Scenario B: Raw Sediment Data (Trend profiles are absolute zero)
        Zmodel = np.zeros(n_points)
        Ubstd = np.zeros(n_points)

    # ============================================================
    # 5. INITIALIZE RESULT CONTAINERS
    # ============================================================
    Zest = np.zeros(n_points)
    Zsim = np.zeros(n_points)
    Zest[list(excluded_set)] = 0.0
    Zsim[list(excluded_set)] = 0.0
    
    # Initialize conditioning data arrays
    conditioning_points = np.array(data_points, dtype=np.float64)
    conditioning_values = np.array(Z_observation, dtype=np.float64)
    
    # Anchor excluded bedrock/water zones to zero physical thickness
    for idx in excluded_set:
        conditioning_points = np.vstack([conditioning_points, grid_points[idx]])
        conditioning_values = np.hstack([conditioning_values, 0.0 - Zmodel[idx]])

    # ============================================================
    # 6. INITIALIZE SEARCH TREES
    # ============================================================
    kdtree = cKDTree(conditioning_points) 
    tree_obs = cKDTree(data_points)          
    
    max_neighbors = 100 

    # ============================================================
    # 7. MAIN SEQUENTIAL GAUSSIAN SIMULATION (SGS) LOOP
    # ============================================================
    for count, i in enumerate(r_filtered):
        x_star = grid_points[i]

        # --- LOCAL KRIGING NEIGHBOR SEARCH ---
        dist_neigh, idx_neigh = kdtree.query(x_star, k=max_neighbors, distance_upper_bound=Range)
        
        valid_mask = idx_neigh < len(conditioning_points)
        idx_neigh = idx_neigh[valid_mask]
        dist_neigh = dist_neigh[valid_mask]

        if len(idx_neigh) > 2: 
            local_points = conditioning_points[idx_neigh]
            local_vals = conditioning_values[idx_neigh]
            
            # Compute raw covariance steps
            cov_to_u0 = Sill - Variogram_model(dist_neigh, Nugget, Sill, Range, alpha)
            cov_to_u0[dist_neigh < 1e-12] = Sill - Nugget
            
            D_local = distance.cdist(local_points, local_points)
            covariance_matrix = Sill - Variogram_model(D_local, Nugget, Sill, Range, alpha)

            # --- SOLVE STRUCTURAL KRIGING SYSTEM ---
            N = len(idx_neigh)
            C_aug = np.zeros((N + 1, N + 1))
            C_aug[:N, :N] = covariance_matrix
            
            # Main Diagonal Stabilizer: Prevents matrix ill-conditioning near data clusters
            np.fill_diagonal(C_aug[:N, :N], Sill + 1e-9) 
            
            # Map structural Ordinary Kriging row/col coordinates
            C_aug[N, :N] = 1.0
            C_aug[:N, N] = 1.0
            C_aug[N, N] = 0.0  # Safe explicit tracking anchor
            
            rhs = np.zeros(N + 1)
            rhs[:N] = cov_to_u0
            rhs[N] = 1.0

            try:
                weights_aug = np.linalg.solve(C_aug, rhs)
                weights = weights_aug[:-1]
                la = weights[-1]
                
                calculated_mean = np.dot(weights, local_vals)
                calculated_var = Sill - np.dot(weights, cov_to_u0) - la
                
                # Extreme Weight Safeguard Intercept
                if calculated_mean < -200.0 or calculated_mean > (np.max(local_vals) * 3.0):
                    mean = np.mean(local_vals)
                    Zvar = Sill
                else:
                    mean = calculated_mean
                    Zvar = max(calculated_var, 1e-10)
                    
            except np.linalg.LinAlgError:
                mean = np.mean(local_vals) if len(local_vals) > 0 else 0.0
                Zvar = Sill 
        else:
            mean = 0.0 if use_trend else (np.mean(conditioning_values) if len(conditioning_values) > 0 else 0.0)
            Zvar = Sill

        Zstd1 = np.sqrt(Zvar)

        # ========================================================
        # 8. TWO-TIER PHYSICAL CONSTRAINT LOGIC
        # ========================================================
        dist_to_obs, idx_obs = tree_obs.query(x_star, k=1)
        is_near_obs = dist_to_obs < (max(dx, dy) / 2)
        
        z_floor = 0.0  # Baseline physical floor constraint
        is_exact = False

        if is_near_obs:
            z_floor = Z_observationAll[idx_obs]
            if not np.isnan(Z_nan[idx_obs]) and Z_nan[idx_obs] != 0:
                is_exact = True

        # VARIANCE ADJUSTMENT MULTIPLIER 
        
        var_multiplier = 2 #if use_trend else 3
        
        if is_exact:
            std_total = max(0.05*np.abs(Zstd1), 1e-6)
            current_mean = mean
        elif is_near_obs and np.isnan(Z_nan[idx_obs]) and z_floor > 0:
            std_total = max(var_multiplier * (np.abs(Ubstd[i]) + np.abs(Zstd1)), 1e-6)
            current_mean = max(mean, z_floor - Zmodel[i])
        else:
            std_total = max(var_multiplier * (np.abs(Ubstd[i]) + np.abs(Zstd1)), 1e-6)
            current_mean = mean

        # ========================================================
        # 9. REJECTION SAMPLING (ENFORCES SIMULATION BOUNDS)
        # ========================================================
        count_iter = 0
        success = False
        
        # Restored fully to 500 iterations matching your original raw script limits
        while count_iter < 500:
            curr_std = std_total * 0.1 if (is_exact and count_iter == 0) else std_total
            shift = (count_iter / 30.0) * std_total
            
            temp_res = np.random.normal(current_mean + shift, curr_std)
            
            # Reconstruction Formula: Total Thickness = Residual Candidate + Trend Component
            if (temp_res + Zmodel[i]) >= z_floor:
                Zest[i] = temp_res
                Zsim[i] = temp_res + Zmodel[i]
                success = True
                break
            count_iter += 1

        # Fallback safety clamp
        if not success:
            Zsim[i] = z_floor + 1e-4
            Zest[i] = Zsim[i] - Zmodel[i]

        # ========================================================
        # 10. UPDATE DATA MATRIX FOR CONDITIONING PATH
        # ========================================================
        conditioning_points = np.vstack([conditioning_points, x_star])
        conditioning_values = np.hstack([conditioning_values, Zest[i]])
        
        if count % 20 == 0:
            kdtree = cKDTree(conditioning_points)

    # ============================================================
    # 11. FINAL DATA MATRIX RETRIEVAL
    # ============================================================
    final_zsim = Zsim
    final_zsim[list(excluded_set)] = 0.0  # Force exposed bedrock cells strictly to zero
    
    return [np.column_stack((grid_points[:, 0], grid_points[:, 1], final_zsim)).tolist()]


def Run_M_cSGS_on_rawSediment(data_points, Z_observation, grid_points, Variogram_model, Z_observationAll, Z_nan, Nugget, Sill,
                              Range, alpha, dx, dy, X_expAllInGigDom, n_realizations):
    """
    Helper for running multiple realizations using RAW sediment data constraints.
    Automatically forces the engine's internal trend parameters to absolute zero.
    """
    simulated_fields = []
    for i in range(n_realizations):
        # Calls the source code with trend mean and trend variance suppressed (None)
        Zsim_vec3 = cSGS_Simulation_SourceCode(
            data_points=data_points, Z_observation=Z_observation, grid_points=grid_points, 
            poinsBh1=None, U_bdis1=None, Ub_std1=None, Variogram_model=Variogram_model,
            Z_observationAll=Z_observationAll, Z_nan=Z_nan, Nugget=Nugget, Sill=Sill, 
            Range=Range, alpha=alpha, dx=dx, dy=dy, X_expAllInGigDom=X_expAllInGigDom,
            use_trend=False  # Tells source code to bypass griddata interpolation entirely
        )
        simulated_fields.append(Zsim_vec3[0])
       # np.save('100cSGS_S_R1.npy', simulated_fields)
        
    return simulated_fields


def Run_M_cSGS_on_residual_with_SedimentTrend(data_points, Z_observation, grid_points, poinsBh1, U_bdis1, Ub_std1, Variogram_model,
                                              Z_observationAll, Z_nan, Nugget, Sill, Range, alpha, dx, dy,
                                              X_expAllInGigDom, n_realizations):
    """
    Helper for running multiple realizations using RESIDUAL values superimposed on a Biharmonic Trend.
    Activates the engine's internal cubic interpolation calculations.
    """
    simulated_fields = []
    for i in range(n_realizations):
        # Calls the source with all spatial biharmonic matrices active
        Zsim2 = cSGS_Simulation_SourceCode(
            data_points=data_points, Z_observation=Z_observation, grid_points=grid_points, 
            poinsBh1=poinsBh1, U_bdis1=U_bdis1, Ub_std1=Ub_std1, Variogram_model=Variogram_model,
            Z_observationAll=Z_observationAll, Z_nan=Z_nan, Nugget=Nugget, Sill=Sill, 
            Range=Range, alpha=alpha, dx=dx, dy=dy, X_expAllInGigDom=X_expAllInGigDom,
            use_trend=True  # Tells source code to calculate sediment as: Residual + Trend
        )
        simulated_fields.append(Zsim2[0])
        #np.save('100cSGS_S_T1.npy', simulated_fields)
        
    return simulated_fields
