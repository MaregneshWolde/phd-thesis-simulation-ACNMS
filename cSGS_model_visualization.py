"""
Visualization and validation workflow for conditional sequential Gaussian simulation (cSGS)
and biharmonic trend analysis.

Author: Maregnesh Wolde
Contributors: Samson S. B., Andreas K., Christian F., Nils-Otto K.
Date: May 2026

This script reproduces the visualization and validation results presented in:
[Paper 2 citation or DOI placeholder]

Requirements:
    numpy
    pandas
    scipy
    matplotlib
    openpyxl
"""

#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

# ============================================================
# 1. GLOBAL CONFIGURATIONS & PLOTTING STYLES
# ============================================================
mpl.rc('image', cmap='viridis')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'figure.dpi': 150
})

# Coordinate limits for the study area
xmin, xmax = 212990, 215400
ymin, ymax = 6.63968e6, 6.64255e6
base = Path("Data")
base1 = Path("Data_simulation")

# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================
def normalize_coordinates(well_data, scale=None):
    """Normalize x and y coordinates to a unit-scaled domain."""
    x_min = well_data['x_coord'].min()
    x_max = well_data['x_coord'].max()
    scale = scale or (x_max - x_min)
    well_data['x_normal'] = (well_data['x_coord'] - x_min) / scale
    
    y_min = well_data['y_coord'].min()
    well_data['y_normal'] = (well_data['y_coord'] - y_min) / scale
    return scale

def bilinear_interpolation_and_derivatives(x_coords, y_coords, f_values, points_of_interest):
    """Computes bilinear interpolation and structural spatial derivatives."""
    n_pts = len(points_of_interest)
    interpolated_values = np.zeros(n_pts)
    derivatives_x = np.zeros(n_pts)
    derivatives_y = np.zeros(n_pts)

    for idx, (x, y) in enumerate(points_of_interest):
        x1_idx = max(0, np.searchsorted(x_coords, x) - 1)
        x2_idx = min(len(x_coords) - 1, x1_idx + 1)
        y1_idx = max(0, np.searchsorted(y_coords, y) - 1)
        y2_idx = min(len(y_coords) - 1, y1_idx + 1)

        x1, x2 = x_coords[x1_idx], x_coords[x2_idx]
        y1, y2 = y_coords[y1_idx], y_coords[y2_idx]

        f11 = f_values[y1_idx, x1_idx]
        f21 = f_values[y1_idx, x2_idx]
        f12 = f_values[y2_idx, x1_idx]
        f22 = f_values[y2_idx, x2_idx]

        t = (x - x1) / (x2 - x1) if x2 != x1 else 0
        u = (y - y1) / (y2 - y1) if y2 != y1 else 0

        f_interp = (f11 * (1 - t) * (1 - u) + f21 * t * (1 - u) +
                    f12 * (1 - t) * u + f22 * t * u)

        df_dx = ((f21 - f11) * (1 - u) + (f22 - f12) * u) / (x2 - x1) if x2 != x1 else 0
        df_dy = ((f12 - f11) * (1 - t) + (f22 - f21) * t) / (y2 - y1) if y2 != y1 else 0

        interpolated_values[idx] = f_interp
        derivatives_x[idx] = df_dx
        derivatives_y[idx] = df_dy

    return interpolated_values, derivatives_x, derivatives_y

def access_value_at_nadag_loc_optimized(simulated_fields, X_target, tolerance=1e-3):
    """Extracts grid values at validation target points across realizations using a KDTree."""
    n_sims = len(simulated_fields)
    n_targets = len(X_target)
    A = np.zeros((n_sims, n_targets))
    
    ref_field = np.array(simulated_fields[0])
    s_idx = np.lexsort((ref_field[:, 0], ref_field[:, 1]))
    grid_coords = ref_field[s_idx, :2]
    
    tree = cKDTree(grid_coords)
    distances, indices = tree.query(X_target, distance_upper_bound=tolerance)
    
    for j, Ueach in enumerate(simulated_fields):
        sorted_values = np.array(Ueach)[s_idx, 2]
        for i in range(n_targets):
            idx = indices[i]
            if idx < len(grid_coords) and not np.isinf(distances[i]):
                A[j, i] = sorted_values[idx]
            else:
                tx, ty = X_target[i]
                solution = griddata(grid_coords, sorted_values, (tx, ty), method='cubic')
                if np.isnan(solution):
                    solution = griddata(grid_coords, sorted_values, (tx, ty), method='nearest')
                A[j, i] = solution
    return A

def plot_scatter(points, values, title, cbar_label='Values'):
    """Plotting companion helper for structural asset evaluation fields."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(points[:, 0], points[:, 1], c=values, cmap='viridis', s=100, edgecolors='black')
    cbar = plt.colorbar(sc)
    cbar.set_label(cbar_label)
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_title(title)
    plt.show()

# ============================================================
# 3. RAW BOREHOLE DATA EXTRACTION & FILTERING
# ============================================================
print("Extracting raw water and energy borehole dataset entries...")

Waterwell_orig = pd.read_excel(base / 'GrunnvannsBorehull.xlsx', sheet_name='GrunnvannBrønn')
Energywell_orig = pd.read_excel(base / 'GrunnvannsBorehull.xlsx', sheet_name='EnergiBrønn')

columndict = {
    'objekttype': 'Type', 
    'brønnNr': 'WellId', 
    'geolMedium': 'Medium', 
    'boretLengde': 'BoreholeLength', 
    'boretLengdeTilBerg': 'depth',  
    'boretHelningsgrad': 'Angle', 
    'boretAzimuth': 'Azimuth', 
    'lengdeForingsrør': 'CasingLength',
    'x_koordinat': 'x_coord', 
    'y_koordinat': 'y_coord'
}

Waterwell_orig.rename(columns=columndict, inplace=True)
Energywell_orig.rename(columns=columndict, inplace=True)

Waterwell = Waterwell_orig.loc[:, columndict.values()].copy()
Energywell = Energywell_orig.loc[:, columndict.values()].copy()

# Fix translations safely without SettingWithCopy Warnings
Waterwell['Medium'] = Waterwell['Medium'].replace({'Fjell': 'Rock', 'Løsmasse': 'Soil'})
Energywell['Medium'] = Energywell['Medium'].replace({'Fjell': 'Rock', 'Løsmasse': 'Soil'})

All_well = pd.concat([Energywell, Waterwell], ignore_index=True)
scale = normalize_coordinates(All_well)

# Load Exposed Bedrock Arrays
exposed = np.loadtxt(base / 'exposed_bedrock.txt')
x = np.loadtxt(base / 'xcoord.txt')
y = np.loadtxt(base / 'ycoord.txt')

# Sanity Plot 1: Full Exposed Bedrock Map
fig, ax = plt.subplots(figsize=(8,8))
ax.contour(x, y, exposed)
ax.set_title("Exposed Bedrock (Full Region)")
ax.set_aspect('equal')
plt.show()

# Crop Exposed Bedrock to bounding limits
mask_spatial = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
exposed_small = np.where(mask_spatial, exposed, np.nan)

# Sanity Plot 2: Cropped Exposed Bedrock Area
fig, ax = plt.subplots(figsize=(8,8))
co = ax.contourf(x, y, exposed_small, cmap='terrain')
plt.colorbar(co, ax=ax)
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_aspect('equal')
plt.show()

# Filter wells inside the bounding study region
All_small2 = All_well[
    (All_well['x_coord'] < xmax) & (All_well['x_coord'] > xmin) & 
    (All_well['y_coord'] < ymax) & (All_well['y_coord'] > ymin)
].copy()

x1 = All_small2['x_coord'].to_numpy()
y1 = All_small2['y_coord'].to_numpy()
Ddepth = All_small2['depth'].to_numpy()
DBlength = All_small2['BoreholeLength'].to_numpy()

# Vectorized rule to compute real sediment thickness
D_obs = np.where(Ddepth == 0, DBlength, Ddepth)
X_locNew = np.column_stack((x1, y1))

# Calculation of Constraints Indices
IndSed = np.where(Ddepth != 0)[0]
IndWell = np.where(Ddepth == 0)[0]

# Export clean filtered reference copy to disk for safety backup
All_small2.to_excel("Data1.xlsx", index=True, header=True)

# Sanity Plot 3: Sample Field Data Points Over Local Bedrock
fig, ax = plt.subplots(figsize=(12,10))
ax.contour(x, y, exposed_small)
plt.scatter(X_locNew[:, 0], X_locNew[:, 1], c=D_obs, cmap='viridis', s=100)
plt.colorbar(label='D_obs')
plt.title('Sample Data Points over Target Domain Area')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
ax.set_aspect('equal')
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
plt.show()

# ============================================================
# 4. REFERENCE / NADAG SPATIAL VALIDATION DATA PREPROCESSING
# ============================================================
print("Processing independent NADAG validation sets")
dataNadag = pd.read_csv(base / 'local_NADAG.txt', header=0)

mask_nadag = (
    (dataNadag['UTM33_X'] < xmax) & (dataNadag['UTM33_X'] > xmin) & 
    (dataNadag['UTM33_Y'] < ymax) & (dataNadag['UTM33_Y'] > ymin)
)
All_smallNadag = dataNadag[mask_nadag].copy()

X_Nadag = All_smallNadag[['UTM33_X', 'UTM33_Y']].to_numpy()
D_Nadag = All_smallNadag['distance_to_bedrock'].to_numpy()
W_Nadag = All_smallNadag['drilled_length'].to_numpy()

# Impute missing values
valid_nadag = ~np.isnan(D_Nadag)
D_obsNadag = np.where(valid_nadag, D_Nadag, W_Nadag)

# Apply manual indexing exclusion masks to discard points touching exposed bedrock boundaries
keep_idx = np.array([0, 1, 2, 4, 5, 6, 7, 8, 11, 12, 15, 16, 18, 19])
X_nadag1 = X_Nadag[keep_idx]
D_nadag1 = D_obsNadag[keep_idx]

remove_idx = np.array([2, 12])
mask_keep = np.ones(len(X_nadag1), dtype=bool)
mask_keep[remove_idx] = False

# Final dynamic output designations for verification plots
X_nadag = X_nadag1[mask_keep]
D_nadag = D_nadag1[mask_keep]

# ============================================================
# 5. LOADING PRE-COMPUTED SIMULATION ASSETS (.NPY FILES)
# ============================================================




print("Load model grid asset matrices")
try:
    X_locBdInReg = np.load(base1 / 'X_locBdB.npy')
    moved_observations = np.load(base1 / 'moved_observations.npy', allow_pickle=True)
    IndAllDis = np.load(base1 / 'IndAllDis.npy')
    U_b = np.load(base1 / 'biharmonicSol.npy')
    Ubstd = np.load(base1 / 'Ub_std.npy')
    poinsBh1 = np.load(base1 / 'points4biharmonicsol.npy')
    simulated_fieldsR = np.load(base1 / '100cSGS_S_T.npy')  # Residual fields
    simulated_fields = np.load(base1 / '100cSGS_S_R.npy')   # Raw fields
    A_matOk = np.load(base1 / 'A_matOk_P2.npy')
    A_matCV_atX_locNew = np.load(base1 / 'A_matCV_atX_locNew.npy')
    A_matCV = np.load(base1 / 'A_matCV.npy')
    A_matOk_inequality = np.load(base1 / 'A_matOk_inequality.npy')
except FileNotFoundError as e:
    raise FileNotFoundError(
        f"Missing tracking file asset: {e.filename}. Ensure execution matches local data directory paths."
    )
# Map Trend Profiles
U_trend = U_b[IndAllDis]
Ub_std1 = Ubstd[IndAllDis]

X_sed = X_locNew[IndSed]
D_sed = D_obs[IndSed]
Xwells = X_locNew[IndWell]
Dwells = D_obs[IndWell]

locationsbdInReg = np.vstack([X_sed, np.array(X_locBdInReg)]) 
D_obsbdInReg = np.hstack([D_sed, np.zeros(len(X_locBdInReg))])

# ============================================================
# 6. INTEGRATED PRODUCTION PLOTTING SUITE
# ============================================================

# Plot 4: Equality wells and boundary points spatial sayout
plt.figure(figsize=(10, 8))
sc = plt.scatter(locationsbdInReg[:, 0], locationsbdInReg[:, 1], c=D_obsbdInReg, cmap='viridis', s=50)
plt.scatter(X_sed[:, 0], X_sed[:, 1], color='red', s=100, label='Equality wells')
plt.colorbar(sc, label='Observed Data (D_obs)')
plt.title('Equality Data Spatial Distribution')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# Structural shape identifiers 
n_realizations = len(simulated_fieldsR)
ref_layout = simulated_fieldsR[0]
sort_idx = np.lexsort((ref_layout[:, 0], ref_layout[:, 1]))
sorted_ref = ref_layout[sort_idx]

unique_x = np.unique(sorted_ref[:, 0])
unique_y = np.unique(sorted_ref[:, 1])
n, m = len(unique_y), len(unique_x)

Xm = sorted_ref[:, 0].reshape((n, m))
Ym = sorted_ref[:, 1].reshape((n, m))
ZmnRa = sorted_ref[:, 2].reshape((n, m))

minX, maxX = np.min(Xm), np.max(Xm)
minY, maxY = np.min(Ym), np.max(Ym)

# Plot 5: Single Simulation of constrained sequential Gaussian simulationon residuals and added back to 
#the biharmonic trend (cSGS_S_T)
plt.figure(figsize=(10, 7))
plt.imshow(ZmnRa, origin='lower', extent=[minX, maxX, minY, maxY], cmap='viridis')
plt.colorbar(label='Sediment thickness [m]')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.title('Single Realization Layout cSGS_S_T')
plt.tight_layout()
plt.show()

# Compute ensemble mean of cSGS_S_T realizations 
ZsumR = np.zeros((n, m))
for field in simulated_fieldsR:
    ZsumR += field[sort_idx, 2].reshape((n, m))
ZsumR /= n_realizations

# Plot 6: Ensemble mean 
plt.figure(figsize=(10, 7))
plt.imshow(ZsumR, origin='lower', extent=[minX, maxX, minY, maxY], cmap='viridis')
plt.colorbar(label='Sediment thickness [m]')
plt.title("Ensemble Mean Field (100 Realizations of cSGS_S_T)")
plt.tight_layout()
plt.show()


#Plot 7: constrained sequential Gaussian simulation on raw data (cSGS_S_R )
n_realizations_R = len(simulated_fields)
Zsum_ok = np.zeros((n, m))

for i in range(n_realizations_R):
    field_sorted = simulated_fields[i][np.lexsort((simulated_fields[i][:, 0], simulated_fields[i][:, 1]))]
    Zsum_ok += field_sorted[:, 2].reshape((n, m))
Zsum_ok /= n_realizations_R

plt.figure(figsize=(12, 8))
plt.imshow(Zsum_ok, origin='lower', extent=[minX, maxX, minY, maxY], cmap='viridis')
plt.colorbar(label='Sediment thickness [m]')
plt.title("Ensemble Mean Field (100 Realizations of cSGS_S_R)")
plt.show()


# Plot 8: Check the simulation at observation points (observation points moved to grid points)
moved = np.array([obs['moved'] for obs in moved_observations])
solR = bilinear_interpolation_and_derivatives(np.unique(Xm[0, :]), np.unique(Ym[:, 0]), ZmnRa, moved)

plt.figure(figsize=(12, 6))
plt.scatter(moved[:, 1], solR[0], c='r', marker='s', label='Observed wells mapped to grid points')

for idx, i in enumerate(IndSed):
    plt.scatter(moved[:, 1][i], D_obs[i], marker='*', c='b', s=40,
                label='cSGS sediment estimation (Equality)' if idx == 0 else "")
for idx, j in enumerate(IndWell):
    plt.scatter(moved[:, 1][j], D_obs[j], marker='d', c='g', s=40,
                label='cSGS sediment estimation (Inequality)' if idx == 0 else "")
for l in range(len(moved)):
    plt.text(moved[:, 1][l], D_obs[l], str(l), color='k', ha='center', va='bottom', fontsize=9)

plt.xlabel('X-coordinate of observation locations')
plt.ylabel('Sediment thickness [m]')
plt.title('cSGS Residual and Trend Model Reconstruction')
plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
plt.tight_layout()
plt.show()

# Plot 10: Validation Cumulative Distribution Profiles (cSGS_S_R vs cSGS_S_T) 
fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=150)
colors = plt.cm.viridis(np.linspace(0, 1, len(D_nadag)))

for idx, (ax, matrix, title_letter, label_suffix) in enumerate([
    (axes[0], A_matOk, "(a)", "raw data"), 
    (axes[1], A_matCV, "(b)", "residual")
]):
    for i in range(len(D_nadag)):
        Y_1_sorted = np.sort(matrix[:, i])
        if idx == 0:
            Y_1_sorted[Y_1_sorted < 0] = 0  

        closest_index = np.argmin(np.abs(Y_1_sorted - D_nadag[i]))
        median_index = np.argmin(np.abs(Y_1_sorted - np.median(Y_1_sorted)))
        Y_probs = np.linspace(0, 1, len(Y_1_sorted))

        ax.plot(Y_1_sorted, Y_probs, color=colors[i], alpha=0.6)
        ax.scatter(Y_1_sorted[closest_index], Y_probs[closest_index],
                   color="blue", marker='o', edgecolor='black', zorder=4,
                   label="Nadag observed" if i == 0 else "")

    median_y = Y_probs[median_index]
    ax.axhline(y=median_y, color="m", linestyle="--", label=f"Median realizations on {label_suffix}")
    ax.text(0.05, 0.92, title_letter, fontsize=22, transform=ax.transAxes, fontweight='bold')
    ax.set_xlabel("Simulated sediment thickness [m]")
    ax.set_ylabel("Cumulative probability")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.4)
    if idx == 1:
        ax.set_xlim(0, 25)

plt.tight_layout()
plt.show()

# Plot 11: Inequality bound distribution checking (cSGS_S_T boxplot) 
num_wells = A_matCV_atX_locNew.shape[1]
simulated_data = [A_matCV_atX_locNew[:, i] for i in range(num_wells)]
means_per_well = np.mean(A_matCV_atX_locNew, axis=0)

fig, ax = plt.subplots(figsize=(14, 6))
xpos = np.arange(num_wells)
bp = ax.boxplot(simulated_data, positions=xpos, patch_artist=True, widths=0.5,
                boxprops=dict(facecolor='#A2C8F2', color='blue'),
                medianprops=dict(color='red', linewidth=1.5))
bp['fliers'][0].set_label('Outliers')

ax.scatter(xpos, Dwells, color='red', marker='*', s=120, zorder=5, label='Minimum thickness Bound (Inequality)')
ax.scatter(xpos, means_per_well, color='black', marker='o', s=40, zorder=5, label='Ensemble Mean')
ax.set_ylabel('Sediment thickness [m]')
ax.set_xlabel('Index of inequality wells')
ax.set_xticks(xpos)
ax.set_xticklabels(np.arange(1, num_wells + 1))
ax.legend(loc='lower left')
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.show()


# Plot 12: Inequality control checking (baseline cSGS_S_R boxplot) 
num_wells_ok = len(Dwells)
simulated_data_ok = [A_matOk_inequality[:, i] for i in range(num_wells_ok)]
means_per_well_ok = np.mean(A_matOk_inequality[:, :num_wells_ok], axis=0)

fig, ax = plt.subplots(figsize=(14, 6))
bp_ok = ax.boxplot(simulated_data_ok, positions=xpos, patch_artist=True, widths=0.5,
                   boxprops=dict(facecolor='#A2C8F2', color='blue'),
                   medianprops=dict(color='red', linewidth=1.5))
bp_ok['fliers'][0].set_label('Outliers')

ax.scatter(xpos, Dwells, color='red', marker='*', s=120, zorder=5, label='Minimum thickness Bound')
ax.scatter(xpos, means_per_well_ok, color='black', marker='o', s=40, zorder=5, label='Ensemble Mean')
ax.set_ylabel('Sediment thickness [m]')
ax.set_xlabel('Index of inequality wells')
ax.set_xticks(xpos)
ax.set_xticklabels(np.arange(1, num_wells_ok + 1))
ax.legend(loc='lower left')
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.show()


# Plots 13 & 14: Trend and standard deviation visualization scatter fields
plot_scatter(poinsBh1, U_trend, 'Biharmonic Trend Values', cbar_label='trend sediment estimation')
plot_scatter(poinsBh1, Ub_std1, 'Biharmonic Standard Deviation', cbar_label='standard deviation')

# Continuous field mesh surface conversions
Z_reg = griddata((poinsBh1[:, 0], poinsBh1[:, 1]), U_trend, (Xm, Ym), method='cubic')
Ubstd_grid = griddata((poinsBh1[:, 0], poinsBh1[:, 1]), Ub_std1, (Xm, Ym), method='cubic')

for z_field, title, label in [(Z_reg, "Interpolation to Regular Grid", "Values"), 
                             (Ubstd_grid, "Standard Deviation of Biharmonic", "Std Dev")]:
    plt.figure(figsize=(10, 8))
    plt.contourf(Xm, Ym, z_field, levels=50, cmap='viridis')
    plt.colorbar(label=label)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.show()




