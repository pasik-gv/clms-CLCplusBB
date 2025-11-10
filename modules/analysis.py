from glob import glob
import numpy as np
import rasterio
import os

from modules.utils import read_image
from modules.regions_dict import regions_dict


def get_boundary_length(
    rasters_dir,
    chosen_region,
    dataset_label='CLCplusBB',
    target_projection='4326',
    band_num=0,
    pixel_size=None,
    input_nodata=None,
    return_density=True,
    class_info=None
):
    """Calculate the total boundary length (km) and optionally edge density (m/ha) for a region raster.

    The raster filename is derived automatically using the pattern:
        ``{dataset_label}_{<region_label>}_{target_projection}.tif``

    Parameters
    ----------
    rasters_dir : str
        Directory containing region rasters.
    chosen_region : str
        Chosen region slug used to construct filepaths.
    dataset_label : str, default 'CLCplusBB'
        Dataset label used in raster filenames.
    target_projection : str, default '4326'
        Projection code used in filename.
    band_num : int, default 0
        Zero-based band index to read (converted internally to 1-based for rasterio).
    pixel_size : float, optional
        Pixel size in meters; if None it's inferred from the raster transform.
    input_nodata : int or list, optional
        Value(s) to treat as nodata (reassigned to sentinel 255). Class 0 also treated as nodata.
    return_density : bool, default True
        If True, return tuple (boundary_length_km, edge_density_m_per_ha); otherwise (boundary_length_km, None).
    class_info : dict, optional
        Reserved for future use (kept for signature consistency with pairwise function); ignored.

    Returns
    -------
    tuple
        (boundary_length_km, edge_density_m_per_ha or None)
    """

    # Wildcard pattern allows extra suffix after projection
    pattern = os.path.join(
        rasters_dir,
        f"{dataset_label}_{chosen_region}_{target_projection}*.tif"
    )
    matches = glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No raster matched pattern: {pattern}")
    if len(matches) > 1:
        # Pick first or raise; here we pick first and warn
        print(f"Warning: multiple rasters matched; using first: {matches[0]}")
    raster_path = matches[0]
    
    with rasterio.open(raster_path) as src:
        data = src.read(band_num + 1).copy()  # rasterio uses 1-based band indexing
        if pixel_size is None:
            pixel_size = abs(src.transform.a)

    # Apply nodata
    if input_nodata is not None:
        mask = np.isin(data, input_nodata) if isinstance(input_nodata, list) else (data == input_nodata)
        data[mask] = 255
    data[data == 0] = 255  # treat class 0 as nodata

    # Count boundaries (4-neighborhood: horizontal + vertical adjacencies)
    boundary_count = 0
    boundary_count += np.sum((data[:, :-1] != data[:, 1:]) & (data[:, :-1] != 255) & (data[:, 1:] != 255))
    boundary_count += np.sum((data[:-1, :] != data[1:, :]) & (data[:-1, :] != 255) & (data[1:, :] != 255))

    boundary_length_m = boundary_count * pixel_size
    boundary_length_km = boundary_length_m / 1000.0

    valid_pixels = np.sum(data != 255)
    pixel_area = pixel_size ** 2
    total_area_m2 = valid_pixels * pixel_area

    if total_area_m2 > 0:
        edge_density_m_per_ha = boundary_length_m / (total_area_m2 / 10000.0)
    else:
        edge_density_m_per_ha = 0.0

    if return_density:
        return round(boundary_length_km, 3), round(edge_density_m_per_ha, 3)
    else:
        return round(boundary_length_km, 3), None

def get_boundary_length_per_class_pair(
    rasters_dir,
    chosen_region,
    dataset_label='CLCplusBB',
    target_projection='4326',
    band_num=0,
    pixel_size=None,
    input_nodata=None,
    return_density=True,
    class_info=None
):
    """Calculate pairwise boundary lengths (km) and optionally edge densities (m/ha) between land cover classes.

    The raster file is derived automatically from the directory and region name using the naming pattern:
        ``{dataset_label}_{<region_label>}_{target_projection}.tif``

    Parameters
    ----------
    rasters_dir : str
        Directory containing the raster files.
    chosen_region : str
        Either the human-readable region name (key in ``regions_dict``) OR directly the slug used in filenames.
    dataset_label : str, default 'CLCplusBB'
        Dataset label used in the file naming pattern.
    target_projection : str, default '4326'
        Projection code appended in the filename.
    band_num : int, default 0
        Zero-based band index (converted internally to rasterio 1-based indexing when reading).
    pixel_size : float, optional
        Pixel size in meters. If None, inferred from the raster transform.
    input_nodata : int or list, optional
        Value(s) to treat as nodata; class 0 is also treated as nodata.
    return_density : bool, default True
        If True, include edge density (m/ha) in the output dictionaries.
    class_info : dict, optional
        Mapping of class ids to metadata; if provided, result keys use class names.

    Returns
    -------
    dict
        {(class_a, class_b): {'length_km': float, 'edge_density_m_per_ha': float}} when return_density is True.
        {(class_a, class_b): {'length_km': float}} when return_density is False.
        Class labels replaced by names if class_info provided. Boundary length is always reported FIRST as
        'length_km'; edge density SECOND as 'edge_density_m_per_ha'.
    """

    if chosen_region in regions_dict:
        region_label = regions_dict[chosen_region][2]
    else:
        # Treat chosen_region as already being the slug
        region_label = chosen_region
    raster_path = os.path.join(
        rasters_dir,
        f"{dataset_label}_{region_label}_{target_projection}.tif"
    )

    if not os.path.exists(raster_path):
        raise FileNotFoundError(f"Raster file does not exist: {raster_path}")

    with rasterio.open(raster_path) as src:
        data = src.read(band_num + 1).copy()
        if pixel_size is None:
            pixel_size = abs(src.transform.a)

    # Apply nodata handling
    if input_nodata is not None:
        mask = np.isin(data, input_nodata) if isinstance(input_nodata, list) else (data == input_nodata)
        data[mask] = 255
    data[data == 0] = 255  # treat class 0 as nodata sentinel

    boundary_lengths = {}

    # Horizontal boundaries (right neighbor comparisons)
    right_a = data[:, :-1]
    right_b = data[:, 1:]
    right_mask = (right_a != 255) & (right_b != 255) & (right_a != right_b)
    if np.any(right_mask):
        pairs_right = np.stack([right_a[right_mask], right_b[right_mask]], axis=1)
        for a, b in pairs_right:
            key = tuple(sorted((int(a), int(b))))
            boundary_lengths[key] = boundary_lengths.get(key, 0) + 1

    # Vertical boundaries (bottom neighbor comparisons)
    bottom_a = data[:-1, :]
    bottom_b = data[1:, :]
    bottom_mask = (bottom_a != 255) & (bottom_b != 255) & (bottom_a != bottom_b)
    if np.any(bottom_mask):
        pairs_bottom = np.stack([bottom_a[bottom_mask], bottom_b[bottom_mask]], axis=1)
        for a, b in pairs_bottom:
            key = tuple(sorted((int(a), int(b))))
            boundary_lengths[key] = boundary_lengths.get(key, 0) + 1

    valid_pixels = np.sum(data != 255)
    pixel_area = pixel_size ** 2
    total_area = valid_pixels * pixel_area

    # Convert counts to meters
    for key in boundary_lengths:
        boundary_lengths[key] *= pixel_size

    # Replace class ids with names if requested
    if class_info is not None:
        named_lengths = {}
        for (class_a, class_b), length in boundary_lengths.items():
            name_a = class_info.get(class_a, {}).get('name', f'Class {class_a}')
            name_b = class_info.get(class_b, {}).get('name', f'Class {class_b}')
            key_named = tuple(sorted((name_a, name_b)))
            named_lengths[key_named] = length
        boundary_lengths = named_lengths

    results = {}
    if return_density:
        for key, length_m in boundary_lengths.items():
            length_km = float(length_m / 1000.0)
            edge_density_m_per_ha = float(length_m / (total_area / 10000.0)) if total_area > 0 else 0.0
            results[key] = {
                'length_km': round(length_km, 3),
                'edge_density_m_per_ha': round(edge_density_m_per_ha, 3)
            }
    else:
        for key, length_m in boundary_lengths.items():
            length_km = float(length_m / 1000.0)
            results[key] = {
                'length_km': round(length_km, 3)
            }

    return results

def calculate_class_areas(rasters_dir, chosen_region, dataset_label='CLCplusBB', 
                         target_projection='3035', class_info=None, nodata_value=0):
    """
    Calculate the area of each CLC class and the overall area for a specified region.
    
    Parameters
    ----------
    rasters_dir : str
        Path to the directory containing raster files.
    chosen_region : str
        Name of the region to analyze.
    dataset_label : str, optional
        Label of the dataset (default 'CLCplusBB').
    target_projection : str, optional
        Target projection for the raster (default '4326').
    class_info : dict, optional
        Dictionary mapping class IDs to class information.
        If None, uses class numbers instead of class names.
    nodata_value : int, optional
        Value to treat as nodata (default 0).
    
    Returns
    -------
    dict
        Dictionary with the following keys:
        - 'class_areas': dict mapping class_id to area in square meters
        - 'class_areas_ha': dict mapping class_id to area in hectares  
        - 'class_percentages': dict mapping class_id to percentage of total area
        - 'total_area_m2': total area in square meters
        - 'total_area_ha': total area in hectares
        - 'pixel_size_m': pixel size in meters
        - 'class_names': dict mapping class_id to class name (if class_info provided)
    """
    
  
    # Read the raster data
    dataset_dict = read_image(rasters_dir, chosen_region, dataset_label, 
                             target_projection=target_projection)
    arr = dataset_dict['array']
    
    # Calculate pixel area based on projection
    if target_projection == '4326':
        # For geographic coordinates (lat/lon), we need to calculate area differently
        # Use a rough approximation: 1 degree = ~111 km at equator
        bounds = dataset_dict['bounds']
        # bounds format: [[bottom, left], [top, right]]
        width_deg = abs(bounds[1][1] - bounds[0][1])  # right - left in degrees
        height_deg = abs(bounds[1][0] - bounds[0][0])  # top - bottom in degrees
        
        # Convert degrees to meters (rough approximation)
        # At latitude ~52.6° (Poland), 1 degree longitude ≈ 67 km
        lat_center = (bounds[0][0] + bounds[1][0]) / 2
        m_per_deg_lat = 111000  # meters per degree latitude (constant)
        m_per_deg_lon = 111000 * np.cos(np.radians(lat_center))  # meters per degree longitude
        
        width_m = width_deg * m_per_deg_lon
        height_m = height_deg * m_per_deg_lat
        
        pixel_width_m = width_m / arr.shape[1]
        pixel_height_m = height_m / arr.shape[0]
        pixel_area_m2 = pixel_width_m * pixel_height_m
        pixel_size_m = (pixel_width_m + pixel_height_m) / 2
        
    else:
        # For projected coordinates (already in meters)
        bounds = dataset_dict['bounds']
        width_m = abs(bounds[1][1] - bounds[0][1])
        height_m = abs(bounds[1][0] - bounds[0][0])
        
        pixel_width_m = width_m / arr.shape[1]
        pixel_height_m = height_m / arr.shape[0]
        pixel_area_m2 = pixel_width_m * pixel_height_m
        pixel_size_m = (pixel_width_m + pixel_height_m) / 2
    
    # Get unique classes, excluding NaN values and nodata values
    valid_mask = ~np.isnan(arr) & (arr != nodata_value)
    unique_classes = np.unique(arr[valid_mask])
    
    # Calculate areas for each class
    class_areas_m2 = {}
    class_areas_ha = {}
    class_percentages = {}
    class_names = {}
    
    # Total valid (non-NaN and non-nodata) pixels
    valid_pixels = np.sum(valid_mask)
    total_area_m2 = valid_pixels * pixel_area_m2
    total_area_ha = total_area_m2 / 10000  # Convert to hectares
    
    for class_id in unique_classes:
        if not np.isnan(class_id):
            class_id_int = int(class_id)
            
            # Count pixels for this class
            class_pixels = np.sum(arr == class_id)
            
            # Calculate areas
            area_m2 = class_pixels * pixel_area_m2
            area_ha = area_m2 / 10000  # Convert to hectares
            percentage = (class_pixels / valid_pixels) * 100 if valid_pixels > 0 else 0
            
            # Store results
            class_areas_m2[class_id_int] = area_m2
            class_areas_ha[class_id_int] = area_ha
            class_percentages[class_id_int] = percentage
            
            # Get class name if available
            if class_info is not None and class_id_int in class_info:
                class_names[class_id_int] = class_info[class_id_int].get('name', f'Class {class_id_int}')
            else:
                class_names[class_id_int] = f'Class {class_id_int}'
    
    # Prepare output dictionary
    result = {
        'class_areas': class_areas_m2,
        'class_areas_ha': class_areas_ha,
        'class_percentages': class_percentages,
        'total_area_m2': total_area_m2,
        'total_area_ha': total_area_ha,
        'pixel_size_m': pixel_size_m,
        'class_names': class_names,
        'region': chosen_region,
        'dataset': dataset_label
    }
    
    return result

def print_class_areas(area_results, header=None):
    """
    Print a formatted summary of class area calculations.

    Parameters
    ----------
    area_results : dict
        Results dictionary returned by calculate_class_areas().
    header : str, optional
        If provided, printed as a title above the table.
    """
    print("Area Statistics")
    print("-" * 15)

    if header:
        print(f"Region: {header}")
    else:
        print(f"Region: {area_results['region']}")
    print(f"Total area: {area_results['total_area_ha']:.2f} ha")
    print()

    print(f"{'Class ID':<8} {'Class Name':<45} {'Area (ha)':<12} {'Area (%)':<10}")
    print("-" * 70)

    sorted_classes = sorted(
        area_results['class_areas_ha'].items(),
        key=lambda x: x[1],
        reverse=True
    )

    for class_id, area_ha in sorted_classes:
        class_name = area_results['class_names'][class_id]
        percentage = area_results['class_percentages'][class_id]
        print(f"{class_id:<8} {class_name:<45} {area_ha:<12.2f} {percentage:<10.1f}")

    print("-" * 70)
    print(f"{'Total':<54} {area_results['total_area_ha']:<12.2f} {'100.0':<10}")

