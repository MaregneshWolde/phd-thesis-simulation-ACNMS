#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Bedrock Surface Reconstruction Using Constrained Sequential Gaussian
Simulation (cSGS).

This program reconstructs bedrock topography from sparse observations
using a constrained geostatistical simulation framework. The workflow
combines trend surface modeling, residual simulation, and Radial Basis
Function (RBF) interpolation to generate continuous bedrock surfaces
that satisfy both equality and inequality constraints.

Main Features
-------------
- Loads and preprocesses spatial data, including:
    * grid point coordinates,
    * exposed bedrock nodal points,
    * equality-constrained well measurements,
    * inequality well locations and maximum bedrock elevations.

- Computes residuals between the trend surface and:
    * terrain elevations at grid points,
    * maximum bedrock elevations at inequality wells.

- Performs constrained Sequential Gaussian Simulation (cSGS) of the
  residual field conditioned on known observations.

- Enforces inequality constraints to ensure the reconstructed bedrock
  surface remains below the terrain or specified maximum elevations.

- Uses Radial Basis Function (RBF) interpolation to construct smooth
  continuous surfaces from the simulated realizations.

- Evaluates uncertainty through multiple stochastic realizations and
  associated variance estimates.

Outputs
-------
The program produces:
- multiple constrained bedrock realizations,
- interpolated continuous bedrock surfaces,
- random simulation paths,
- uncertainty/variance estimates,
- visualization of simulated and interpolated surfaces.

Notes
-----
The methodology is intended for inverse reconstruction problems with
sparse and partially constrained observations, particularly in
geostatistical and geological modeling applications.
"""


# In[2]:


## IMPORT REQUIRED MODULES 

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import ticker
import random
import matplotlib
matplotlib.rc( 'image', cmap='jet' )
from scipy.interpolate import Rbf



# In[3]:


def plot_cSGS(z, ax=None):
    """
    Plot a continuous surface using Radial Basis Function (RBF) interpolation.

    Parameters
    ----------
    x : array-like
        Array containing the x-coordinates of the grid points.

    y : array-like
        Array containing the y-coordinates of the grid points.

    z : array-like
        Array containing the estimated elevation values at the
        corresponding grid points.

    Returns
    -------
    None
        Displays a continuous interpolated surface that passes through
        the provided grid point values.

    Notes
    -----
    The surface is constructed using Radial Basis Function (RBF)
    interpolation based on the input coordinate and elevation data.
    """
    # build interpolator
    rbf = Rbf(x, y, z, function='multiquadric', smooth=0.1)

    # create grid
    xi = np.linspace(x.min(), x.max(), 200)
    yi = np.linspace(y.min(), y.max(), 200)
    XI, YI = np.meshgrid(xi, yi)

    # evaluate surface
    ZI = rbf(XI, YI)

    # use provided axis or create one
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    cf = ax.contourf(XI, YI, ZI, levels=50)
    plt.colorbar(cf, ax=ax, label="Estimated surface")

    return ax


# In[4]:


"""
Here loading of data:  the grid points, locations and measurement values at known locations(i.e nodal points in the exposed regions 
and equality GRANADA wells). LOcation and maximum bederock elevation at the inequality wells are also accessed.  Residual, which is devation 
of the trend surface from the terain (for the grid points) and from the maximum value (for inequality wells), which will be used in the code 
to insure the estimated surface lie below the terrain or maximum bedrock elevation.  The variance due to the Leave out random surface
(i.e. \sigma^2_u) at the grid points and inequality wells.
"""

"""
Load and preprocess the input data required for the simulations.

The following data are accessed:

- Grid point coordinates
- Locations and measurement values at known locations
  (i.e., nodal points in the exposed regions and equality GRANADA wells)
- Locations and maximum bedrock elevations at the inequality wells

In addition, the residual values are computed, representing the deviation
between:
- the trend surface and terrain elevation at the grid points, and
- the trend surface and maximum bedrock elevation at the inequality wells.

These residuals are used to ensure that the estimated surface remains
below the terrain or maximum bedrock elevation constraints.

The variance associated with the leave-out random surface (i.e., :math:`\sigma_u^2`) is also evaluated at:
- the grid points, and
- the inequality wells.
"""


## Coordinates of grid points
polyGrid_coordinates = np.loadtxt( "polyGrid_coordinates.txt")
# coordinates
points = polyGrid_coordinates 
x = points[:, 0]
y = points[:, 1]




# Locations and measurement values at known locations
known_locations4 = np.loadtxt("known_locations4.txt")
known_values4 = np.loadtxt("known_values4.txt")


## Location and maximum bedrock elevation at inequality wells
Bed_rock_inequality_locations = np.loadtxt(  "Bed_rock_inequality_locations.txt")
Bedrock_inequality_elevation = np.loadtxt( "Bedrock_inequality_elevation.txt")

## Trend surface value at grid points
Trend_values = np.loadtxt( "Trend_values.txt")

### Residuals
residual_at_inequality_well = np.loadtxt( "residual_at_inequality_well.txt")
residual_at_unknown_locations_0 = np.loadtxt( "Residual_at_all_grid_pts.txt")
residual_at_unknown_locations = np.hstack((residual_at_unknown_locations_0,residual_at_inequality_well))

# Load variances due to LOR surfaces
Variance_Leave_Many_out = np.loadtxt( "Variance_Leave_Many_out.txt")
Variance_Leave_many_out_at_wells = np.loadtxt("Variance_Leave_many_out_at_wells.txt")


# In[5]:


def Model_variogram2(h, c0, c1, h_R):
    beta = -3  
    alpha = 1        # 1: exponential, 2: Gaussian
    return c0 + c1 * (1 - np.exp(beta * (h / h_R)**alpha))


def sgs_on_residue_data_test2(No_of_realization ,known_locations, known_values, unknown_locations):
    """
Perform constrained Sequential Gaussian Simulation (cSGS).

Parameters
----------
No_of_realization : int
    Number of realizations to generate.

known_locations : array-like
    Coordinates of sampled locations where measurement values are known.

known_values : array-like
    Observed values corresponding to the sampled locations.

unknown_locations : array-like
    Coordinates of locations where the simulations are to be performed.

Returns
-------
realization_matrix : ndarray
    Matrix containing all simulated realizations at the unknown locations.
    Each row corresponds to one realization.

random_path_indices : ndarray
    Matrix containing the indices representing the random simulation path
    used for each realization.

Notes
-----
The function performs constrained Sequential Gaussian Simulation (cSGS)
conditioned on the known sampled data.
"""
    data = known_values


    c0, c1, h_R = 1.68,15.66,168   # variogram parameters


    def Variogram_Model(h):
        return Model_variogram2(h, c0, c1, h_R)




    np.random.seed(12)

    Random_paths = np.array([ np.random.permutation(len(unknown_locations)) for _ in range(No_of_realization)])

    Realizations_Matrix = np.zeros((No_of_realization, len(unknown_locations) ))

    for l in range(No_of_realization):

        Updated_locations = known_locations.copy()
        Updated_values = data.copy()



        Estimated_values = np.zeros(len(unknown_locations))
        # === Generate random SGS path once ===
        random_path =   Random_paths[l,:]              

        n_0 = len(Updated_locations)
        variogram_matrix = np.zeros((n_0 + 1, n_0 + 1))
        for i in range(n_0):
            for j in range(i, n_0):
                dist = np.linalg.norm(Updated_locations[i] - Updated_locations[j])
                variogram_matrix[i, j] = c0 + c1 - Variogram_Model(dist)
                variogram_matrix[j, i] = variogram_matrix[i, j]
        variogram_matrix[-1, :-1] = 1
        variogram_matrix[:-1, -1] = 1



        v3 = np.array([[0, 1], [1, 0]])


        should_update_variogram_matrix = False


        ## === Loop through SGS random path ===
        for index_original in random_path:

            selected_point = unknown_locations[index_original]

            if should_update_variogram_matrix:
                v1 = c0 + c1 - Variogram_Model(np.linalg.norm(Updated_locations[:-1] - Updated_locations[-1], axis=1))
                v2 = np.ones(len(v1))
                Variogram_matrix_updater = np.vstack([v1, v2]).T
                new_variogram_matrix = np.block([
                    [variogram_matrix[:-1, :-1], Variogram_matrix_updater],
                    [Variogram_matrix_updater.T, v3]
                ])
                variogram_matrix = new_variogram_matrix

            n = len(Updated_locations)
            variogram_vector = np.zeros(n + 1)
            for i in range(n):
                dist = np.linalg.norm(Updated_locations[i] - selected_point)
                if dist == 0:
                    variogram_vector[i] = c0+c1
                else:
                    variogram_vector[i] = c0 + c1 - Variogram_Model(dist)
            variogram_vector[-1] = 1


            np.fill_diagonal(variogram_matrix[:-1, :-1],  c0+c1)




            kriging_weights = np.linalg.solve(variogram_matrix, variogram_vector)


            estimated_value = np.dot(kriging_weights[:-1], Updated_values)
            variance = c0 + c1 - np.dot(kriging_weights, variogram_vector)

            mean = estimated_value

            std = 0 if variance < 0 else 3 * np.sqrt(variance)

            residual_at_selected_pt = residual_at_unknown_locations[index_original]



            # These are trend uncertainity \sigma^2_u at the unknown locations: With LOR surfaces

            if selected_point in polyGrid_coordinates[np.where(np.array(Variance_Leave_Many_out) > 2)[0]]:
                points = polyGrid_coordinates
                Index_sel_pt_in_grid = np.where(np.all(np.isclose(points, selected_point, atol=1e-6), axis=1))[0][0]

                std = std +  3*np.sqrt(Variance_Leave_Many_out[Index_sel_pt_in_grid])

            elif selected_point in Bed_rock_inequality_locations:
                points = Bed_rock_inequality_locations
                Index_sel_pt_in_wells = np.where(np.all(np.isclose(points, selected_point, atol=1e-6), axis=1))[0][0]

                std = std +  3*np.sqrt(Variance_Leave_many_out_at_wells[Index_sel_pt_in_wells])


            # For simulation at conditioning point: no measurement error so hard conditioning

            if (np.all(np.trunc(known_locations).astype(int) == np.trunc(selected_point).astype(int), axis=1)).any():

                simulated_value = mean

                Estimated_values[index_original] = simulated_value
                should_update_variogram_matrix = False

                continue

            # === For simulation at non conditioning point: impose bounds/truncation ===

            if (np.all(np.trunc(Bed_rock_inequality_locations).astype(int) == np.trunc(selected_point).astype(int), axis=1)).any():

                index_selected = np.where((np.trunc(Bed_rock_inequality_locations).astype(int) == np.trunc(selected_point).astype(int)).all(axis=1))[0][0]

                if mean - std < 0:
                        simulated_value = random.uniform(mean - std, min(0, mean + std))

                else:
                        simulated_value = random.uniform(-3, -1)

            else:

                if residual_at_selected_pt > 0:
                    if mean + std < residual_at_selected_pt:
                        simulated_value = random.uniform(mean - std, mean + std)
                    elif mean - std < residual_at_selected_pt < mean + std:
                        simulated_value = random.uniform(mean - std, residual_at_selected_pt)
                    else:
                        simulated_value = random.uniform(residual_at_selected_pt - std, residual_at_selected_pt)
                else:
                    if mean - std < residual_at_selected_pt:
                        simulated_value = random.uniform(mean - std, min(residual_at_selected_pt, mean + std))
                    else:
                        simulated_value = random.uniform(residual_at_selected_pt - std, residual_at_selected_pt)

            Updated_locations = np.array(list(Updated_locations) + [list(selected_point)])
            Updated_values = np.hstack([Updated_values, np.array([simulated_value])])

            Estimated_values[index_original] = simulated_value

            should_update_variogram_matrix = True

        Realizations_Matrix[l,:]  =    np.array(Estimated_values) + np.hstack((Trend_values, Bedrock_inequality_elevation ))
    return Realizations_Matrix, Random_paths #Estimated_values


# In[ ]:





# In[6]:


# Unsampled locations: Locations where simulations are performed using cSGS
Unsampled_locations = np.vstack(( polyGrid_coordinates, Bed_rock_inequality_locations))


# In[7]:


# Specify number of realization you need and call the function for cSGS to perform simulation
N0_realization  = 3
Matr,Random_paths = sgs_on_residue_data_test2(N0_realization,known_locations4,known_values4,Unsampled_locations)



# In[8]:


#Access simulation at grid points and inequality wells extracted
Realizations_at_grid_pts = Matr[:,:len(polyGrid_coordinates)] 
Realizations_at_inequality_wells = Matr[:,-18:]


# In[9]:


# Plotting of a single(chosen) realization and mena of all realizations

z_single_realization = Realizations_at_grid_pts[1,:]
z_mean_of_realizations = np.mean(Realizations_at_grid_pts,axis = 0)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

plot_cSGS(z_single_realization, ax=ax[0])
plot_cSGS(z_mean_of_realizations, ax=ax[1])

ax[0].set_title("Single realization")
ax[1].set_title("Mean of realizations")

plt.tight_layout()
plt.show()


# In[10]:


# Plot of the maximum bedrock topography at inequality wells and simulated elevations at those locations 

plt.figure(figsize = (10,5))
plt.plot(Bedrock_inequality_elevation, label='Maximum elevation')

for i in range(N0_realization):
    if i == 0:
        plt.plot(Realizations_at_inequality_wells[i, :], '.', color ='red',label='Estimates')
    else:
        plt.plot(Realizations_at_inequality_wells[i, :], '.', color ='red')

    plt.plot(Realizations_at_inequality_wells[i, :], '-.')

plt.legend()
plt.show()


# In[ ]:




