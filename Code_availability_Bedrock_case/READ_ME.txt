This folder contains all input and precomputed files required for the implementation of the cSGS (constrained Sequential Gaussian Simulation) algorithm for the bedrock topography case.

1. Grid Data
		polyGrid_coordinates.txt
	Coordinates of grid points used in the simulation domain.
2. Known Data (Equality Constraints)
		known_locations4.txt
	Locations of known measurements (nodal points in exposed regions and equality GRANADA wells).
		known_values4.txt
	Measured values at the corresponding known locations.
3. Inequality Constraints (Wells)
		Bed_rock_inequality_locations.txt
	Locations of inequality (constraint) wells.
		Bedrock_inequality_elevation.txt
	Maximum bedrock elevation values at inequality wells.
4. Trend Surface
		Trend_values.txt
	Trend surface values evaluated at all grid points.
5. Residuals

	Residuals represent deviations between observed values and the trend surface:

		Residual_at_all_grid_pts.txt
	Difference between terrain elevation and trend surface at grid points.
		residual_at_inequality_well.txt
	Difference between maximum bedrock elevation and trend surface at inequality wells.

	These residuals are used to enforce that the estimated surface respects terrain and maximum bedrock constraints.

6. Variance of Random Field

	Variance of the leave-one-out random surface $\sigma^2_u$:

		Variance_Leave_Many_out.txt
	Variance at grid points.
		Variance_Leave_many_out_at_wells.txt
	Variance at inequality wells.
