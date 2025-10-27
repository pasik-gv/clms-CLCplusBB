import numpy as np
import rasterio

from modules.utils import read_image


def get_boundary_length(raster_path, band_num=0, pixel_size=None, input_nodata=None, return_density=True):
    """
    Calculate the total boundary length (in kilometers) between all classes in a raster.
    Optionally returns edge density (meters per square meter).

    Parameters
    ----------
    raster_path : str
        Path to the raster file.
    band_num : int
        The band number to use (default 0).
    pixel_size : float, optional
        Length of one pixel edge in meters. If None, calculated from raster transform.
    input_nodata : int or list, optional
        Value(s) to treat as nodata in input raster.
    return_density : bool, optional
        If True, also returns edge density (default True).

    Returns
    -------
    float or tuple
        Total boundary length in kilometers, or (boundary_length_km, edge_density) if return_density is True (default).
    """
    with rasterio.open(raster_path) as src:
        data = src.read(band_num + 1).copy()  # rasterio uses 1-based indexing
        
        if pixel_size is None:
            pixel_size = abs(src.transform.a)
            print("pixel size:", pixel_size)
    if input_nodata is not None:
        mask = np.isin(data, input_nodata) if isinstance(input_nodata, list) else (data == input_nodata)
        data[mask] = 255  # set nodata to a unique value
    
    # Also treat class 0 as nodata
    data[data == 0] = 255

    boundary_count = 0
    boundary_count += np.sum((data[:, :-1] != data[:, 1:]) & (data[:, :-1] != 255) & (data[:, 1:] != 255))
    boundary_count += np.sum((data[:-1, :] != data[1:, :]) & (data[:-1, :] != 255) & (data[1:, :] != 255))

    boundary_length_m = boundary_count * pixel_size
    boundary_length_km = boundary_length_m / 1000  # Convert meters to kilometers

    valid_pixels = np.sum(data != 255)
    pixel_area = pixel_size ** 2
    total_area = valid_pixels * pixel_area
    edge_density = boundary_length_m / total_area if total_area > 0 else 0.0

    if return_density:
        return round(boundary_length_km, 3), round(edge_density, 6)
    else:
        return round(boundary_length_km, 3), None

def get_boundary_length_per_class_pair(raster_path, band_num=0, pixel_size=None, input_nodata=None, return_density=True, class_info=None):
    """
    Calculate the boundary length (in kilometers) for each pair of classes in a categorical raster.
    Optionally returns edge density per class pair.

    Parameters
    ----------
    raster_path : str
        Path to the raster file.
    band_num : int
        The band number to use (default 0).
    pixel_size : float, optional
        Length of one pixel edge in meters. If None, calculated from raster transform.
    input_nodata : int or list, optional
        Value(s) to treat as nodata in input raster.
    return_density : bool, optional
        If True, also returns edge density per class pair (default True).
    class_info : dict, optional
        Dictionary mapping class IDs to {"name": str, "color": str}.
        If provided, returns results using class names instead of numbers.

    Returns
    -------
    dict
        {(class_a, class_b): boundary_length_km, ...} if class_info is None
        {(class_name_a, class_name_b): boundary_length_km, ...} if class_info is provided
        If return_density is True: values are tuples (boundary_length_km, edge_density)
    """
    with rasterio.open(raster_path) as src:
        data = src.read(band_num + 1).copy()  # rasterio uses 1-based indexing
        
        if pixel_size is None:
            pixel_size = abs(src.transform.a)
        
    if input_nodata is not None:
        mask = np.isin(data, input_nodata) if isinstance(input_nodata, list) else (data == input_nodata)
        data[mask] = 255  # set nodata to a unique value
    
    # Also treat class 0 as nodata
    data[data == 0] = 255

    boundary_lengths = {}

    # Right neighbor
    right_a = data[:, :-1]
    right_b = data[:, 1:]
    right_mask = (right_a != 255) & (right_b != 255) & (right_a != right_b)
    pairs_right = np.stack([right_a[right_mask], right_b[right_mask]], axis=1)
    for a, b in pairs_right:
        key = tuple(sorted((int(a), int(b))))
        boundary_lengths[key] = boundary_lengths.get(key, 0) + 1

    # Bottom neighbor
    bottom_a = data[:-1, :]
    bottom_b = data[1:, :]
    bottom_mask = (bottom_a != 255) & (bottom_b != 255) & (bottom_a != bottom_b)
    pairs_bottom = np.stack([bottom_a[bottom_mask], bottom_b[bottom_mask]], axis=1)
    for a, b in pairs_bottom:
        key = tuple(sorted((int(a), int(b))))
        boundary_lengths[key] = boundary_lengths.get(key, 0) + 1

    valid_pixels = np.sum(data != 255)
    pixel_area = pixel_size ** 2
    total_area = valid_pixels * pixel_area

    # Convert to actual lengths and round to 2 decimal places
    for key in boundary_lengths:
        boundary_lengths[key] *= pixel_size

    # If class_info is provided, convert class numbers to names
    if class_info is not None:
        boundary_lengths_named = {}
        for (class_a, class_b), length in boundary_lengths.items():
            name_a = class_info.get(class_a, {}).get('name', f'Class {class_a}')
            name_b = class_info.get(class_b, {}).get('name', f'Class {class_b}')
            key_named = tuple(sorted((name_a, name_b)))
            boundary_lengths_named[key_named] = length
        boundary_lengths = boundary_lengths_named

    if return_density:
        for key in boundary_lengths:
            length_m = boundary_lengths[key]
            length_km = length_m / 1000  # Convert meters to kilometers
            edge_density = length_m / total_area if total_area > 0 else 0.0
            boundary_lengths[key] = (round(length_km, 3), round(edge_density, 6))
            # print(f"Class pair {key}: Boundary length = {length_km:.3f} km, Edge density = {edge_density:.6f} m/m²")
    else:
        for key in boundary_lengths:
            length_km = boundary_lengths[key] / 1000  # Convert meters to kilometers
            boundary_lengths[key] = round(length_km, 3)
            # print(f"Class pair {key}: Boundary length = {length_km:.3f} km")

    return boundary_lengths

def calculate_class_areas(rasters_dir, chosen_region, dataset_label='CLCplusBB', 
                         target_projection='4326', class_info=None, nodata_value=0):
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

def print_class_areas(area_results):
    """
    Print a formatted summary of class area calculations.
    
    Parameters
    ----------
    area_results : dict
        Results dictionary from calculate_class_areas function.
    """
    # Print area statistics
    print(f"Land Cover Area Analysis")
    print("-" * 35)
    print(f"Region: {area_results['region']}")
    print(f"Total area: {area_results['total_area_ha']:.2f} ha")
    print()

    print(f"{'Class ID':<8} {'Class Name':<45} {'Area (ha)':<12} {'Area (%)':<10}")
    print("-" * 70)

    # Sort by area (largest first)
    sorted_classes = sorted(area_results['class_areas_ha'].items(), 
                           key=lambda x: x[1], reverse=True)

    for class_id, area_ha in sorted_classes:
        class_name = area_results['class_names'][class_id]
        percentage = area_results['class_percentages'][class_id]
        print(f"{class_id:<8} {class_name:<45} {area_ha:<12.2f} {percentage:<10.1f}")

    print("-" * 70)
    print(f"{'Total':<54} {area_results['total_area_ha']:<12.2f} {'100.0':<10}")
