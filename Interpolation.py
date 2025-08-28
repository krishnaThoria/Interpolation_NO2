# Ensure these are installed:
# !pip install netCDF4 h5netcdf h5py xarray numpy matplotlib scipy pykrige pandas cartopy

import matplotlib # Import matplotlib first
matplotlib.use('Agg') # <<< SET BACKEND TO AGG BEFORE IMPORTING PYPLOT
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import xarray as xr
import numpy as np
import os
import glob
import time
import warnings
import traceback
from scipy.interpolate import Rbf
from scipy.interpolate import griddata as scipy_griddata
from pykrige.ok import OrdinaryKriging
from scipy.spatial import cKDTree
import pandas as pd
import h5py
from datetime import datetime
import re
import shutil

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')

# --- Configuration ---
omi_input_dir = r'/content/drive/MyDrive/Dataset/Satellite_data/Satellite_data_2024/'
omi_plot_output_dir = r'/content/drive/MyDrive/Interpolated_0.05deg_v13_OMI_Plots_TropomiScaled/' # Changed output dir slightly

OMI_NO2_VAR_NAME_INTERNAL = 'ColumnAmountNO2' # Keep as is, used for xarray coord names
OMI_LAT_VAR_NAME_INTERNAL = 'Latitude'       # Keep as is
OMI_LON_VAR_NAME_INTERNAL = 'Longitude'      # Keep as is

tropomi_truth_dir = r'/content/drive/MyDrive/Final_Filtered_Tropomi_2024/'
TROPOMI_NO2_H5_PATH = 'PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/nitrogendioxide_summed_total_column'
TROPOMI_LAT_H5_PATH = 'PRODUCT/latitude'
TROPOMI_LON_H5_PATH = 'PRODUCT/longitude'
TROPOMI_QA_H5_PATH = 'PRODUCT/qa_value'

lat_min_delhi, lat_max_delhi = 28.0, 29.5
lon_min_delhi, lon_max_delhi = 76.5, 78.0
target_resolution_deg = 0.05
TROPOMI_QA_THRESHOLD = 0.75
# This factor converts TROPOMI from mol/m^2 to molecules/cm^2, assuming OMI is already in molecules/cm^2
TROPOMI_SCALING_FACTOR_UNIT_CONVERSION = 6.02214E19

validation_output_dir = omi_plot_output_dir
output_stats_csv = os.path.join(validation_output_dir, 'omi_vs_tropomi_validation_metrics_v13_tropomiscaled_fixedOMI.csv')

# --- Path Verification ---
drive_path_ok = False
try:
    if not os.path.exists(omi_plot_output_dir): os.makedirs(omi_plot_output_dir, exist_ok=True)
    if os.path.isdir(omi_plot_output_dir): drive_path_ok = True
    else: print(f"FATAL: Path '{omi_plot_output_dir}' exists but is NOT A DIRECTORY.")
except Exception as e: print(f"FATAL: Error Drive plot output dir: {e}")
if not drive_path_ok: print("Halting: Drive output directory not accessible."); exit()

local_temp_plot_dir = "/tmp/my_omi_tropomi_plots_v13_critical_fixedOMI" # Changed temp dir
try:
    if os.path.exists(local_temp_plot_dir): shutil.rmtree(local_temp_plot_dir)
    os.makedirs(local_temp_plot_dir, exist_ok=True)
    print(f"Local temporary plot directory '{local_temp_plot_dir}' created/cleaned.")
except Exception as e: print(f"FATAL: Error local temp plot dir: {e}"); exit()


def improved_idw_interpolation(x_orig, y_orig, z_orig, x_target_coords, y_target_coords,
                               power=1.5, initial_radius=0.3, max_radius=1.0,
                               min_points=4, k_max=100):
    target_grid_lon, target_grid_lat = np.meshgrid(x_target_coords, y_target_coords)
    z_target_shape = target_grid_lon.shape
    z_target_flat = np.full(target_grid_lon.size, np.nan)
    target_points_flat = np.column_stack((target_grid_lon.ravel(), target_grid_lat.ravel()))
    source_points = np.column_stack((x_orig, y_orig))
    if len(source_points) == 0 or len(z_orig) == 0: return z_target_flat.reshape(z_target_shape)
    if not isinstance(z_orig, np.ndarray) or z_orig.ndim != 1:
        if hasattr(z_orig, 'flatten') and len(source_points) == z_orig.size: z_orig = z_orig.flatten()
        else: return z_target_flat.reshape(z_target_shape)
    tree = cKDTree(source_points)
    result_values = []
    for pt_idx, pt in enumerate(target_points_flat):
        radius, found, interpolated_val, expansions = initial_radius, False, np.nan, 0
        while not found and radius <= max_radius and expansions < 5:
            current_k = min(k_max, len(source_points))
            if current_k == 0: break
            distances, indices = tree.query(pt, k=current_k, distance_upper_bound=radius)
            valid = (indices < len(source_points)) & np.isfinite(distances)
            if np.sum(valid) >= min_points:
                valid_dists, valid_indices = distances[valid], indices[valid]
                valid_values = z_orig[valid_indices]
                if np.any(valid_dists < 1e-10): interpolated_val = valid_values[valid_dists < 1e-10][0]
                else:
                    weights = 1.0 / (valid_dists ** power)
                    if np.sum(weights) > 1e-9 : interpolated_val = np.sum(weights * valid_values) / np.sum(weights)
                found = True
            else: radius *= 1.5; expansions +=1
        result_values.append(interpolated_val)
    return np.array(result_values).reshape(z_target_shape)

def calculate_validation_stats(obs_truth, pred_interp):
    if obs_truth is None or pred_interp is None or len(obs_truth) == 0 or len(pred_interp) == 0 or len(obs_truth) != len(pred_interp):
        return (np.nan,) * 8
    valid_mask = np.isfinite(obs_truth) & np.isfinite(pred_interp)
    obs_f, pred_f = obs_truth[valid_mask], pred_interp[valid_mask]
    n_pairs = len(obs_f)
    if n_pairs < 2: # Need at least 2 pairs for correlation, and meaningful stats
        return np.nan, np.nan, np.nan, np.nan, n_pairs, np.nan, np.nan, np.nan

    mae = np.mean(np.abs(pred_f - obs_f)); rmse = np.sqrt(np.mean((pred_f - obs_f)**2)); bias = np.mean(pred_f - obs_f)
    correlation = np.nan
    # Ensure standard deviations are not zero for correlation
    if np.std(obs_f) > 1e-9 and np.std(pred_f) > 1e-9:
        correlation = np.corrcoef(pred_f, obs_f)[0, 1]

    mean_obs_truth = np.mean(obs_f)
    # Check for zero or very small mean_obs_truth to avoid division by zero or huge normalized errors
    if abs(mean_obs_truth) > 1e-9:
        nmae = mae / mean_obs_truth
        nrmse = rmse / mean_obs_truth
        nbias = bias / mean_obs_truth
    else:
        nmae, nrmse, nbias = np.nan, np.nan, np.nan

    return mae, rmse, bias, correlation, n_pairs, nmae, nrmse, nbias


def extract_date_from_filename(filename_str):
    base = os.path.basename(filename_str)
    if re.fullmatch(r"\d{8}", base):
        try: return datetime.strptime(base, "%Y%m%d").date()
        except ValueError: pass
    match_tropomi = re.search(r"(\d{4})(\d{2})(\d{2})T\d{6}", base)
    if match_tropomi: return datetime(int(match_tropomi.group(1)), int(match_tropomi.group(2)), int(match_tropomi.group(3))).date()
    match_omi1 = re.search(r"_(\d{4})m(\d{2})(\d{2})t", base)
    if match_omi1: return datetime(int(match_omi1.group(1)), int(match_omi1.group(2)), int(match_omi1.group(3))).date()
    match_omi2 = re.match(r"(\d{4})(\d{2})(\d{2})\.(nc|he5|nc4)", base, re.IGNORECASE)
    if match_omi2: return datetime(int(match_omi2.group(1)), int(match_omi2.group(2)), int(match_omi2.group(3))).date()
    match_generic = re.search(r"(\d{4})(\d{2})(\d{2})", base)
    if match_generic:
        try: return datetime(int(match_generic.group(1)), int(match_generic.group(2)), int(match_generic.group(3))).date()
        except ValueError: pass
    print(f"    Warning: Could not extract date from filename: {base}")
    return None


def interpolate_omi_and_validate_with_tropomi():
    print("--- Interpolate OMI & Validate with TROPOMI Script (TROPOMI Unit-Converted, OMI Cleaned) ---")
    all_items_in_omi_dir = []
    try: all_items_in_omi_dir = os.listdir(omi_input_dir)
    except FileNotFoundError: print(f"\n*** ERROR: OMI Input Directory not found: '{omi_input_dir}'. Exiting."); return
    omi_files_temp = []
    for item_name in all_items_in_omi_dir:
        full_item_path = os.path.join(omi_input_dir, item_name)
        if os.path.isfile(full_item_path) and not item_name.startswith('.') and re.fullmatch(r"\d{8}", item_name):
            omi_files_temp.append(full_item_path)
    omi_files = sorted(omi_files_temp)
    if not omi_files: print(f"\n*** ERROR: No OMI data files (YYYYMMDD format) found in '{omi_input_dir}'. Exiting."); return
    print(f"Found {len(omi_files)} OMI files (YYYYMMDD) to process.")

    print(f"Scanning TROPOMI directory: {tropomi_truth_dir} for date matching...")
    tropomi_file_map = {}
    tropomi_raw_files = sorted(glob.glob(os.path.join(tropomi_truth_dir, '**', '*.nc*'), recursive=True))
    tropomi_raw_files = [f for f in tropomi_raw_files if os.path.isfile(f) and not os.path.basename(f).startswith('.')]
    for trop_f in tropomi_raw_files:
        date_obj = extract_date_from_filename(trop_f)
        if date_obj: tropomi_file_map.setdefault(date_obj, []).append(trop_f)
    print(f"Found TROPOMI files for {len(tropomi_file_map)} unique dates.")

    all_validation_results = []
    omi_processed_count = 0
    omi_successfully_processed_plots_on_drive = 0
    omi_error_count, omi_skipped_nan_slice, tropomi_not_found_count = 0,0,0
    plot_local_save_failures, plot_drive_copy_failures, plots_skipped_no_data = 0,0,0

    POTENTIAL_OMI_NO2_NAMES = ['ColumnAmountNO2', '/HDFEOS/SWATHS/OMI Column Amount NO2/Data Fields/ColumnAmountNO2']
    POTENTIAL_OMI_LAT_NAMES = ['Latitude', '/HDFEOS/SWATHS/OMI Column Amount NO2/Geolocation Fields/Latitude']
    POTENTIAL_OMI_LON_NAMES = ['Longitude', '/HDFEOS/SWATHS/OMI Column Amount NO2/Geolocation Fields/Longitude']

    for omi_idx, omi_file_to_process in enumerate(omi_files):
        print(f"\n--- Processing OMI File {omi_idx+1}/{len(omi_files)}: {os.path.basename(omi_file_to_process)} ---")
        omi_interpolation_results = {}
        fig_omi_plots = None
        omi_x_orig_valid, omi_y_orig_valid, omi_z_orig_valid = None, None, None
        omi_lons_2d_for_plot_raw, omi_lats_2d_for_plot_raw, omi_no2_2d_for_plot_raw_cleaned = None, None, None # Renamed for clarity
        plot_figure_created = False

        try:
            print(f"  Opening OMI file: {omi_file_to_process}...")
            actual_omi_no2_name_found, actual_omi_lat_name_found, actual_omi_lon_name_found = None, None, None
            with h5py.File(omi_file_to_process, 'r') as f_h5_omi:
                def find_h5_var(h5_obj, var_list):
                    for var_name in var_list:
                        if var_name in h5_obj: return var_name
                    return None
                actual_omi_no2_name_found = find_h5_var(f_h5_omi, POTENTIAL_OMI_NO2_NAMES)
                actual_omi_lat_name_found = find_h5_var(f_h5_omi, POTENTIAL_OMI_LAT_NAMES)
                actual_omi_lon_name_found = find_h5_var(f_h5_omi, POTENTIAL_OMI_LON_NAMES)

                if not (actual_omi_no2_name_found and actual_omi_lat_name_found and actual_omi_lon_name_found):
                    missing = [v for v,f in zip(["NO2","Lat","Lon"],[actual_omi_no2_name_found,actual_omi_lat_name_found,actual_omi_lon_name_found]) if not f]
                    raise KeyError(f"OMI vars missing: {','.join(missing)} in {omi_file_to_process}.")

                omi_no2_h5_var = f_h5_omi[actual_omi_no2_name_found]
                _no2_raw = omi_no2_h5_var[:]

                # --- START OMI FILL VALUE AND NEGATIVE VALUE HANDLING ---
                if '_FillValue' in omi_no2_h5_var.attrs:
                    fill_value = omi_no2_h5_var.attrs['_FillValue']
                    print(f"    OMI File: {os.path.basename(omi_file_to_process)}, NO2 Var: {actual_omi_no2_name_found}, Detected _FillValue: {fill_value}")
                    _no2_raw = np.where(_no2_raw == fill_value, np.nan, _no2_raw)
                else:
                    print(f"    OMI File: {os.path.basename(omi_file_to_process)}, NO2 Var: {actual_omi_no2_name_found}, No _FillValue attribute found. Assuming NaNs are used or data is clean.")

                # Set negative values to NaN (as they are unphysical)
                _no2_raw = np.where(_no2_raw < 0, np.nan, _no2_raw)
                print(f"    Applied negative value filter to OMI NO2.")
                # --- END OMI FILL VALUE AND NEGATIVE VALUE HANDLING ---

                _lat_raw = f_h5_omi[actual_omi_lat_name_found][:]
                _lon_raw = f_h5_omi[actual_omi_lon_name_found][:]

                # Squeeze singleton dimensions if present
                if _no2_raw.ndim == 3 and _no2_raw.shape[0] == 1: _no2_raw = _no2_raw[0,:,:]
                if _lat_raw.ndim == 3 and _lat_raw.shape[0] == 1: _lat_raw = _lat_raw[0,:,:]
                if _lon_raw.ndim == 3 and _lon_raw.shape[0] == 1: _lon_raw = _lon_raw[0,:,:]

            omi_no2_np_processed = _no2_raw; omi_lats_processed = _lat_raw; omi_lons_processed = _lon_raw

            if omi_lats_processed.ndim == 1 and omi_lons_processed.ndim == 1:
                if omi_no2_np_processed.shape == (omi_lats_processed.shape[0], omi_lons_processed.shape[0]):
                    omi_lons_2d_for_plot_raw, omi_lats_2d_for_plot_raw = np.meshgrid(omi_lons_processed, omi_lats_processed); omi_no2_2d_for_plot_raw_cleaned = omi_no2_np_processed
                elif omi_no2_np_processed.shape == (omi_lons_processed.shape[0], omi_lats_processed.shape[0]):
                    omi_lons_2d_for_plot_raw, omi_lats_2d_for_plot_raw = np.meshgrid(omi_lons_processed, omi_lats_processed); omi_no2_2d_for_plot_raw_cleaned = omi_no2_np_processed.T
                else: raise ValueError(f"OMI NO2 shape {omi_no2_np_processed.shape} incompatible for 1D Lat/Lon meshgrid.")
            elif omi_lats_processed.ndim == 2 and omi_lons_processed.ndim == 2:
                if omi_no2_np_processed.shape == omi_lats_processed.shape and omi_no2_np_processed.shape == omi_lons_processed.shape:
                    omi_lats_2d_for_plot_raw, omi_lons_2d_for_plot_raw, omi_no2_2d_for_plot_raw_cleaned = omi_lats_processed, omi_lons_processed, omi_no2_np_processed
                else: raise ValueError(f"OMI NO2 shape {omi_no2_np_processed.shape} mismatch with 2D Lat/Lon.")
            else: raise ValueError(f"Unsupported OMI Lat/Lon dimensions.")

            delhi_mask_omi_2d_raw = ((omi_lats_2d_for_plot_raw >= lat_min_delhi) & (omi_lats_2d_for_plot_raw <= lat_max_delhi) & (omi_lons_2d_for_plot_raw >= lon_min_delhi) & (omi_lons_2d_for_plot_raw <= lon_max_delhi))

            # omi_no2_2d_for_plot_raw_cleaned already has fill values and negatives as NaN
            omi_no2_delhi_masked_for_plot = np.where(delhi_mask_omi_2d_raw, omi_no2_2d_for_plot_raw_cleaned, np.nan)

            finite_mask_omi_2d = np.isfinite(omi_no2_2d_for_plot_raw_cleaned) # Now this is more robust
            valid_mask_for_interp = delhi_mask_omi_2d_raw & finite_mask_omi_2d # Already includes non-negative check due to previous step

            omi_x_orig_valid = omi_lons_2d_for_plot_raw[valid_mask_for_interp].flatten()
            omi_y_orig_valid = omi_lats_2d_for_plot_raw[valid_mask_for_interp].flatten()
            omi_z_orig_valid = omi_no2_2d_for_plot_raw_cleaned[valid_mask_for_interp].flatten() # Use the cleaned data

            if len(omi_z_orig_valid) == 0: print("  --> No FINITE, NON-NEGATIVE OMI data in Delhi. Skipping."); omi_skipped_nan_slice += 1; continue
            print(f"  --> Found {len(omi_z_orig_valid)} valid (finite, non-negative) OMI points in Delhi for interpolation.")

            epsilon = target_resolution_deg/2; omi_target_lat_coords=np.arange(lat_min_delhi,lat_max_delhi+epsilon,target_resolution_deg); omi_target_lon_coords=np.arange(lon_min_delhi,lon_max_delhi+epsilon,target_resolution_deg); omi_target_lon_grid,omi_target_lat_grid=np.meshgrid(omi_target_lon_coords,omi_target_lat_coords)

            print("  Starting OMI Interpolations...");
            omi_source_pts_for_griddata = np.vstack((omi_x_orig_valid, omi_y_orig_valid)).T

            try: # Bilinear
                if len(omi_source_pts_for_griddata) >=3: # Need at least 3 points for triangulation
                    omi_bilinear_vals = scipy_griddata(omi_source_pts_for_griddata, omi_z_orig_valid, (omi_target_lon_grid, omi_target_lat_grid), method='linear', fill_value=np.nan)
                    # Additional mask for Delhi boundaries if scipy_griddata extrapolates slightly
                    delhi_boundary_mask_target = (omi_target_lat_grid >= lat_min_delhi) & (omi_target_lat_grid <= lat_max_delhi) & \
                                                 (omi_target_lon_grid >= lon_min_delhi) & (omi_target_lon_grid <= lon_max_delhi)
                    omi_bilinear_vals = np.where(delhi_boundary_mask_target, omi_bilinear_vals, np.nan)

                    omi_interpolation_results['Bilinear'] = xr.DataArray(omi_bilinear_vals, coords={OMI_LAT_VAR_NAME_INTERNAL: omi_target_lat_coords, OMI_LON_VAR_NAME_INTERNAL: omi_target_lon_coords}, dims=[OMI_LAT_VAR_NAME_INTERNAL, OMI_LON_VAR_NAME_INTERNAL])
                else:
                    print("    Skipping Bilinear: Less than 3 valid OMI source points.")
                    omi_interpolation_results['Bilinear'] = None
            except Exception as e:
                print(f"    Error Bilinear: {e}");
                omi_interpolation_results['Bilinear'] = None

            if len(omi_z_orig_valid) >= 3 : # Min points for other point-based methods
                try: # Kriging
                    OK_omi = OrdinaryKriging(omi_x_orig_valid, omi_y_orig_valid, omi_z_orig_valid, variogram_model='spherical', verbose=False, enable_plotting=False, nlags=max(2,min(6, int(len(omi_z_orig_valid)/3)+1 )))
                    omi_krig_vals, _ = OK_omi.execute('grid', omi_target_lon_coords, omi_target_lat_coords)
                    omi_interpolation_results['Kriging'] = xr.DataArray(omi_krig_vals, coords={OMI_LAT_VAR_NAME_INTERNAL: omi_target_lat_coords, OMI_LON_VAR_NAME_INTERNAL: omi_target_lon_coords}, dims=[OMI_LAT_VAR_NAME_INTERNAL, OMI_LON_VAR_NAME_INTERNAL])
                except ValueError as ve_krig: print(f"    Error Kriging (ValueError): {ve_krig}"); omi_interpolation_results['Kriging'] = None
                except Exception as e: print(f"    Error Kriging (General): {e}"); omi_interpolation_results['Kriging'] = None
                try: # RBF
                    rbf_omi = Rbf(omi_x_orig_valid, omi_y_orig_valid, omi_z_orig_valid, function='gaussian', smooth=0.1)
                    omi_rbf_vals = rbf_omi(omi_target_lon_grid, omi_target_lat_grid)
                    omi_interpolation_results['RBF'] = xr.DataArray(omi_rbf_vals, coords={OMI_LAT_VAR_NAME_INTERNAL: omi_target_lat_coords, OMI_LON_VAR_NAME_INTERNAL: omi_target_lon_coords}, dims=[OMI_LAT_VAR_NAME_INTERNAL, OMI_LON_VAR_NAME_INTERNAL])
                except Exception as e: print(f"    Error RBF: {e}"); omi_interpolation_results['RBF'] = None
                try: # IDW
                    omi_idw_vals = improved_idw_interpolation(omi_x_orig_valid, omi_y_orig_valid, omi_z_orig_valid, omi_target_lon_coords, omi_target_lat_coords)
                    omi_interpolation_results['IDW'] = xr.DataArray(omi_idw_vals, coords={OMI_LAT_VAR_NAME_INTERNAL: omi_target_lat_coords, OMI_LON_VAR_NAME_INTERNAL: omi_target_lon_coords}, dims=[OMI_LAT_VAR_NAME_INTERNAL, OMI_LON_VAR_NAME_INTERNAL])
                except Exception as e: print(f"    Error IDW: {e}"); omi_interpolation_results['IDW'] = None
            else:
                print(f"    Skipping Kriging, RBF, IDW: Only {len(omi_z_orig_valid)} valid OMI source points.")
                omi_interpolation_results['Kriging'] = omi_interpolation_results['RBF'] = omi_interpolation_results['IDW'] = None


            # --- TROPOMI Data Loading ---
            omi_date = extract_date_from_filename(omi_file_to_process)
            x_tropomi_truth, y_tropomi_truth, z_tropomi_truth_molec_cm2 = None,None,None # Renamed for clarity
            tropomi_lons_2d_for_plot_raw, tropomi_lats_2d_for_plot_raw, tropomi_no2_delhi_masked_for_plot_molec_cm2 = None,None,None # Renamed
            tropomi_file_for_validation_name="N/A"
            if omi_date:
                matching_tropomi_files = tropomi_file_map.get(omi_date, [])
                if matching_tropomi_files:
                    tropomi_file_for_validation=matching_tropomi_files[0]; tropomi_file_for_validation_name=os.path.basename(tropomi_file_for_validation)
                    try:
                        with h5py.File(tropomi_file_for_validation,'r') as f_tropomi_h5:
                            _trop_no2_original_units = f_tropomi_h5[TROPOMI_NO2_H5_PATH][:]
                            _trop_lat = f_tropomi_h5[TROPOMI_LAT_H5_PATH][:]
                            _trop_lon = f_tropomi_h5[TROPOMI_LON_H5_PATH][:]
                            _trop_qa  = f_tropomi_h5[TROPOMI_QA_H5_PATH][:] if TROPOMI_QA_H5_PATH in f_tropomi_h5 else np.full_like(_trop_no2_original_units, 1.0)

                        if _trop_no2_original_units.ndim==3: _trop_no2_original_units=_trop_no2_original_units[0]
                        if _trop_lat.ndim==3: _trop_lat=_trop_lat[0]
                        if _trop_lon.ndim==3: _trop_lon=_trop_lon[0]
                        if _trop_qa.ndim ==3: _trop_qa =_trop_qa[0]

                        # Convert TROPOMI from mol/m^2 to molecules/cm^2
                        _trop_no2_molec_cm2 = _trop_no2_original_units * TROPOMI_SCALING_FACTOR_UNIT_CONVERSION
                        print(f"    Converted TROPOMI units using factor: {TROPOMI_SCALING_FACTOR_UNIT_CONVERSION}")

                        tropomi_lons_2d_for_plot_raw, tropomi_lats_2d_for_plot_raw = _trop_lon, _trop_lat
                        _trop_no2_qa_filtered_molec_cm2 = np.where(_trop_qa >= TROPOMI_QA_THRESHOLD, _trop_no2_molec_cm2, np.nan)

                        _trop_delhi_mask_2d = (_trop_lat >= lat_min_delhi)&(_trop_lat <= lat_max_delhi)&(_trop_lon >= lon_min_delhi)&(_trop_lon <= lon_max_delhi)
                        tropomi_no2_delhi_masked_for_plot_molec_cm2 = np.where(_trop_delhi_mask_2d, _trop_no2_qa_filtered_molec_cm2, np.nan)

                        _trop_valid_mask_for_truth = _trop_delhi_mask_2d & np.isfinite(_trop_no2_qa_filtered_molec_cm2)
                        x_tropomi_truth=_trop_lon[_trop_valid_mask_for_truth].flatten()
                        y_tropomi_truth=_trop_lat[_trop_valid_mask_for_truth].flatten()
                        z_tropomi_truth_molec_cm2 =_trop_no2_qa_filtered_molec_cm2[_trop_valid_mask_for_truth].flatten()

                        if len(z_tropomi_truth_molec_cm2)==0:print(f"No valid TROPOMI points for {omi_date}."); x_tropomi_truth,y_tropomi_truth,z_tropomi_truth_molec_cm2=None,None,None; tropomi_no2_delhi_masked_for_plot_molec_cm2=None
                        else: print(f"Found {len(z_tropomi_truth_molec_cm2)} TROPOMI truth points (molecules/cm^2) for {omi_date}.")
                    except Exception as e_trop: print(f"Error TROPOMI {tropomi_file_for_validation_name}: {e_trop}"); x_tropomi_truth,y_tropomi_truth,z_tropomi_truth_molec_cm2=None,None,None; tropomi_no2_delhi_masked_for_plot_molec_cm2=None
                else: tropomi_not_found_count+=1; print(f"No TROPOMI file for {omi_date}")
            else: print("No date for OMI file.")


            # --- Plotting ---
            print("  Preparing plots...")
            plot_figure_created = True
            fig_omi_plots, omi_axes = plt.subplots(2, 3, figsize=(18, 10))
            omi_axes = omi_axes.flatten()
            any_subplot_has_data = False
            all_plot_data_values = []
            # Use the cleaned OMI data for plotting range determination
            if omi_no2_delhi_masked_for_plot is not None and np.any(np.isfinite(omi_no2_delhi_masked_for_plot)):
                all_plot_data_values.append(omi_no2_delhi_masked_for_plot[np.isfinite(omi_no2_delhi_masked_for_plot)])
            # Use the unit-converted TROPOMI data for plotting range
            if tropomi_no2_delhi_masked_for_plot_molec_cm2 is not None and np.any(np.isfinite(tropomi_no2_delhi_masked_for_plot_molec_cm2)):
                 all_plot_data_values.append(tropomi_no2_delhi_masked_for_plot_molec_cm2[np.isfinite(tropomi_no2_delhi_masked_for_plot_molec_cm2)])
            for da_method_name_key in ['Bilinear', 'Kriging', 'RBF', 'IDW']:
                da = omi_interpolation_results.get(da_method_name_key)
                if da is not None and hasattr(da, 'data') and np.any(np.isfinite(da.data)):
                    all_plot_data_values.append(da.data[np.isfinite(da.data)])

            master_vmin, master_vmax = None,None
            if all_plot_data_values:
                concatenated_values = np.concatenate([arr.flatten() for arr in all_plot_data_values if hasattr(arr, 'size') and arr.size > 0])
                if concatenated_values.size > 0:
                    concatenated_values = concatenated_values[np.isfinite(concatenated_values)]
                    if concatenated_values.size > 0:
                        # Use percentiles for robust scaling, but ensure vmin < vmax
                        p_low, p_high = np.percentile(concatenated_values, [2, 98])
                        if p_low < p_high:
                            master_vmin, master_vmax = p_low, p_high
                        else: # Fallback if percentiles are too close or inverted
                            master_vmin, master_vmax = np.nanmin(concatenated_values), np.nanmax(concatenated_values)

                        if master_vmin >= master_vmax :
                            master_vmax=master_vmin + max(1e-9, abs(master_vmin*0.1) if master_vmin!=0 else 0.1) # Ensure vmax > vmin
            current_norm = Normalize(vmin=master_vmin, vmax=master_vmax) if master_vmin is not None and master_vmax is not None and master_vmin < master_vmax else None
            print(f"    Plot Scale (molecules/cm^2): vmin={master_vmin}, vmax={master_vmax}")

            ax = omi_axes[0] # Raw OMI (cleaned)
            if omi_lons_2d_for_plot_raw is not None and omi_lats_2d_for_plot_raw is not None and omi_no2_delhi_masked_for_plot is not None and np.any(np.isfinite(omi_no2_delhi_masked_for_plot)):
                try: ax.pcolormesh(omi_lons_2d_for_plot_raw, omi_lats_2d_for_plot_raw, omi_no2_delhi_masked_for_plot, cmap='jet', norm=current_norm, shading='auto'); any_subplot_has_data=True
                except Exception as e: print(f"Err pcm OMI: {e}")
            ax.set_title(f"Raw OMI (25 km)", fontsize=9); ax.set_xlim(lon_min_delhi,lon_max_delhi); ax.set_ylim(lat_min_delhi,lat_max_delhi); ax.set_xlabel("Lon"); ax.set_ylabel("Lat"); ax.tick_params(labelsize=7)

            ax = omi_axes[1]; main_im_for_cbar = None # Raw TROPOMI (unit-converted)
            if tropomi_lons_2d_for_plot_raw is not None and tropomi_lats_2d_for_plot_raw is not None and tropomi_no2_delhi_masked_for_plot_molec_cm2 is not None and np.any(np.isfinite(tropomi_no2_delhi_masked_for_plot_molec_cm2)):
                try:
                    im_trop = ax.pcolormesh(tropomi_lons_2d_for_plot_raw, tropomi_lats_2d_for_plot_raw, tropomi_no2_delhi_masked_for_plot_molec_cm2, cmap='jet', norm=current_norm, shading='auto')
                    if main_im_for_cbar is None: main_im_for_cbar = im_trop
                    any_subplot_has_data=True
                except Exception as e: print(f"Err pcm TROPOMI: {e}")
            ax.set_title(f"Raw TROPOMI (5.5 km x 3.5 km)", fontsize=9); ax.set_xlim(lon_min_delhi,lon_max_delhi); ax.set_ylim(lat_min_delhi,lat_max_delhi); ax.set_xlabel("Lon"); ax.set_ylabel("Lat"); ax.tick_params(labelsize=7)

            omi_plot_order_interp = ['Bilinear', 'Kriging', 'RBF', 'IDW']
            for i, method in enumerate(omi_plot_order_interp):
                ax = omi_axes[i+2]; da = omi_interpolation_results.get(method)
                if da is not None and hasattr(da, 'data') and np.any(np.isfinite(da.data)):
                    lon_coords_plot = da[OMI_LON_VAR_NAME_INTERNAL].data; lat_coords_plot = da[OMI_LAT_VAR_NAME_INTERNAL].data
                    lon_edges = np.append(lon_coords_plot - target_resolution_deg/2, lon_coords_plot[-1] + target_resolution_deg/2)
                    lat_edges = np.append(lat_coords_plot - target_resolution_deg/2, lat_coords_plot[-1] + target_resolution_deg/2)
                    X_plot, Y_plot = np.meshgrid(lon_edges, lat_edges)
                    im = ax.pcolormesh(X_plot, Y_plot, da.data, cmap='jet', norm=current_norm, shading='flat')
                    if main_im_for_cbar is None: main_im_for_cbar = im
                    any_subplot_has_data=True
                ax.set_title(f"OMI {method} (5 km)", fontsize=9); ax.set_xlim(lon_min_delhi,lon_max_delhi); ax.set_ylim(lat_min_delhi,lat_max_delhi); ax.set_xlabel("Lon"); ax.set_ylabel("Lat"); ax.tick_params(labelsize=7)

            if not any_subplot_has_data: plots_skipped_no_data += 1; print("    No data to plot.")
            else:
                fig_omi_plots.tight_layout(rect=[0,0.03,0.90,0.93]);
                if main_im_for_cbar and current_norm:
                    cbar_ax=fig_omi_plots.add_axes([0.92,0.15,0.02,0.7]);
                    try: cb=fig_omi_plots.colorbar(main_im_for_cbar,cax=cbar_ax,norm=current_norm);cb.set_label(f"NO2 Column (molecules/cm$^2$)",fontsize=9);cb.ax.tick_params(labelsize=7) # Updated label
                    except Exception as e_cb: print(f"Warn cbar: {e_cb}")
                title_date_str=omi_date.strftime('%Y-%m-%d')if omi_date else "UnkDate";fig_omi_plots.suptitle(f"OMI & TROPOMI Comparison - Delhi ({title_date_str})\nOMI:{os.path.basename(omi_file_to_process)}|TROPOMI:{tropomi_file_for_validation_name}",fontsize=10)
                plot_file_leaf=f"plot_{os.path.basename(omi_file_to_process)}.png";local_full_path=os.path.join(local_temp_plot_dir,plot_file_leaf);drive_full_path=os.path.join(omi_plot_output_dir,plot_file_leaf);local_save_ok=False
                try:
                    fig_omi_plots.savefig(local_full_path,dpi=150,bbox_inches='tight')
                    if os.path.exists(local_full_path) and os.path.getsize(local_full_path)>500:local_save_ok=True;print(f"  Saved: {local_full_path}")
                    else: print(f"  Fail/Empty(Local):{local_full_path}");plot_local_save_failures+=1
                except Exception as e_sl:print(f"  ERR savefig:{e_sl}");plot_local_save_failures+=1
                if local_save_ok:
                    try:
                        if not os.path.isdir(omi_plot_output_dir):print(f"  ERR:Drive dir miss {omi_plot_output_dir}");plot_drive_copy_failures+=1
                        else: shutil.copy2(local_full_path,drive_full_path);time.sleep(0.5);
                        if os.path.exists(drive_full_path) and os.path.getsize(drive_full_path)>500:print(f"  Copied to Drive:{drive_full_path}");omi_successfully_processed_plots_on_drive+=1
                        else:print(f"  Fail/Empty(Drive):{drive_full_path}");plot_drive_copy_failures+=1
                    except Exception as e_cd:print(f"  ERR copy Drive:{e_cd}");plot_drive_copy_failures+=1

            omi_processed_count +=1

            # --- Validation ---
            # Ensure OMI is in molecules/cm^2 (should be by default, now cleaned)
            # Ensure TROPOMI (z_tropomi_truth_molec_cm2) is also in molecules/cm^2
            if x_tropomi_truth is not None and len(x_tropomi_truth) > 0 and z_tropomi_truth_molec_cm2 is not None:
                for method_name, omi_interpolated_da in omi_interpolation_results.items():
                    if omi_interpolated_da is None: err_msg, stats_tuple = "OMI Interp Failed", (np.nan,) * 8
                    else:
                        try:
                            # Sampled OMI values are already in molecules/cm^2 from omi_z_orig_valid
                            sampled_omi_values = omi_interpolated_da.interp(
                                {OMI_LON_VAR_NAME_INTERNAL: xr.DataArray(x_tropomi_truth, dims="point"),
                                 OMI_LAT_VAR_NAME_INTERNAL: xr.DataArray(y_tropomi_truth, dims="point")},
                                method="nearest", kwargs={"fill_value": np.nan}).data

                            # Both z_tropomi_truth_molec_cm2 and sampled_omi_values should now be in molecules/cm^2
                            stats_tuple = calculate_validation_stats(z_tropomi_truth_molec_cm2, sampled_omi_values)
                            err_msg = None if stats_tuple[4] > 0 else "No valid pairs"
                        except Exception as e_val_interp: stats_tuple, err_msg = (np.nan,) * 8, f"Validation Interp Error: {e_val_interp}"
                    all_validation_results.append({'omi_file':os.path.basename(omi_file_to_process),'tropomi_file':tropomi_file_for_validation_name,'omi_date':omi_date,'method':f"OMI_{method_name}",
                                                   'mae':stats_tuple[0],'rmse':stats_tuple[1],'bias':stats_tuple[2],'correlation':stats_tuple[3],'N_pairs':stats_tuple[4],
                                                   'nmae':stats_tuple[5],'nrmse':stats_tuple[6],'nbias':stats_tuple[7],'error_message':err_msg})
            else:
                for method_name_key in ['Bilinear', 'Kriging', 'RBF', 'IDW']:
                    all_validation_results.append({'omi_file':os.path.basename(omi_file_to_process),'tropomi_file':tropomi_file_for_validation_name,'omi_date':omi_date,'method':f"OMI_{method_name_key}",
                                                   'mae':np.nan,'rmse':np.nan,'bias':np.nan,'correlation':np.nan,'N_pairs':0,'nmae':np.nan,'nrmse':np.nan,'nbias':np.nan,'error_message':"No TROPOMI truth data"})

        except (KeyError, FileNotFoundError, ValueError, MemoryError, TypeError) as main_err: print(f"*** MAJOR ERROR OMI file {os.path.basename(omi_file_to_process)}: {main_err} ({type(main_err).__name__}) ***"); traceback.print_exc(limit=1); omi_error_count += 1
        except Exception as general_err: print(f"*** UNEXPECTED GENERAL ERROR OMI file {os.path.basename(omi_file_to_process)}: {general_err} ({type(general_err).__name__}) ***"); traceback.print_exc(); omi_error_count += 1
        finally:
            if plot_figure_created and fig_omi_plots is not None and plt.fignum_exists(fig_omi_plots.number): plt.close(fig_omi_plots)
            vars_to_del=['omi_x_orig_valid','omi_y_orig_valid','omi_z_orig_valid','omi_lons_2d_for_plot_raw','omi_lats_2d_for_plot_raw','omi_no2_2d_for_plot_raw_cleaned','omi_no2_delhi_masked_for_plot','x_tropomi_truth','y_tropomi_truth','z_tropomi_truth_molec_cm2','tropomi_lons_2d_for_plot_raw','tropomi_lats_2d_for_plot_raw','tropomi_no2_delhi_masked_for_plot_molec_cm2','omi_interpolation_results']
            for v_name in vars_to_del:
                if v_name in locals() and locals()[v_name] is not None: # Check for None before deleting
                    try:
                        del locals()[v_name]
                    except KeyError: # Should not happen if v_name in locals() but good practice
                        pass
            import gc;gc.collect()

    print(f"\n--- Processing Complete ---")
    print(f"Total OMI files found: {len(omi_files)}")
    print(f"OMI files successfully processed (plots/validation attempted): {omi_processed_count}")
    print(f"OMI files with plots successfully saved to Drive: {omi_successfully_processed_plots_on_drive}")
    print(f"OMI files skipped (No FINITE, NON-NEGATIVE OMI data in Delhi): {omi_skipped_nan_slice}")
    print(f"Plots skipped (No data in any subplot): {plots_skipped_no_data}")
    print(f"Plot local save failures: {plot_local_save_failures}")
    print(f"Plot Drive copy/verify failures: {plot_drive_copy_failures}")
    print(f"OMI files with other major processing errors: {omi_error_count}")
    print(f"OMI files for which TROPOMI truth was not found/loaded: {tropomi_not_found_count}")

    if all_validation_results:
        stats_df = pd.DataFrame(all_validation_results)
        try:
            stats_df.to_csv(output_stats_csv, index=False)
            print(f"Validation metrics saved to: {output_stats_csv}")
            # Filter for rows where N_pairs is a valid number and greater than 1 for meaningful stats
            valid_stats_summary = stats_df[(stats_df['N_pairs'].notna()) & (stats_df['N_pairs'] > 1)].copy()
            if not valid_stats_summary.empty:
                summary_cols = ['mae','rmse','bias','correlation','nmae','nrmse','nbias','N_pairs']
                agg_dict = {col: 'mean' for col in summary_cols if col != 'N_pairs'}
                agg_dict['N_pairs'] = 'sum' # Sum of pairs
                agg_dict['num_files_contributed'] = ('omi_file', 'nunique') # Count unique files

                # Calculate mean but drop NaNs for numeric columns before aggregation to avoid warnings/errors
                for col in ['mae','rmse','bias','correlation','nmae','nrmse','nbias']:
                    valid_stats_summary[col] = pd.to_numeric(valid_stats_summary[col], errors='coerce')

                summary = valid_stats_summary.groupby('method').agg(agg_dict).reset_index()
                print("\nSummary of Mean Metrics (N_pairs > 1):\n", summary.to_string())
            else: print("No data pairs with N_pairs > 1 for summary stats.")
        except Exception as e_csv: print(f"*** ERROR saving CSV/summary: {e_csv} ***"); traceback.print_exc(limit=1)
    else: print("No validation results were generated.")
    try:
        if os.path.exists(local_temp_plot_dir): shutil.rmtree(local_temp_plot_dir); print(f"Cleaned up local temp plot directory: {local_temp_plot_dir}")
    except Exception as e_clean: print(f"Error cleaning up local temp dir '{local_temp_plot_dir}': {e_clean}")


if __name__ == '__main__':
    interpolate_omi_and_validate_with_tropomi()