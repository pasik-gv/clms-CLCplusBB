import os
import numpy as np
import pandas as pd
import rasterio
import ipywidgets as widgets
from IPython.display import display

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from modules.regions_dict import regions_dict
from PIL import Image
from scipy.ndimage import zoom

# CLC+Backbone class mapping with names and colors
CLC_CLASS_INFO = {
    1: {"name": "Sealed", "color": "#ff0000"},
    2: {"name": "Woody-needle leaved trees", "color": "#228b22"},
    3: {"name": "Woody-Broadleaved deciduous trees", "color": "#80ff00"},
    4: {"name": "Woody-Broadleaved evergreen trees", "color": "#00ff08"},
    5: {"name": "Low-growing woody plants (bushes, shrubs)", "color": "#804000"},
    6: {"name": "Permanent herbaceous", "color": "#ccf24d"},
    7: {"name": "Periodically herbaceous", "color": "#ffff80"},
    8: {"name": "Lichens and mosses", "color": "#ff80ff"},
    9: {"name": "Non- and sparsely-vegetated", "color": "#bfbfbf"},
    10: {"name": "Water", "color": "#0080ff"},
    11: {"name": "Snow and ice", "color": "#00ffff"},
    253: {"name": "Coastal seawater buffer", "color": "#bfdfff"},
    254: {"name": "Outside area", "color": "#e6e6e6"}
}

def visualize_class_pair_boundaries(class_pair_results, class_info=None, use_names=True, metric='length', figsize=(12, 10)):
    """
    Visualize class pair boundary results as a confusion matrix-style heatmap.
    
    Parameters
    ----------
    class_pair_results : dict
        Results from get_boundary_length_per_class_pair()
    class_info : dict
        Dictionary mapping class IDs to {"name": str, "color": str}.
    use_names : bool
        If True, use descriptive class names; if False, use class IDs (default True)
    metric : str
        Which metric to display: 'length', 'density', or 'both' (default 'length')
        If 'both', shows length and density side by side
    figsize : tuple
        Figure size for the plot (default (12, 10) for single, automatically adjusted for 'both')
        
    Returns
    -------
    matplotlib.figure.Figure, matplotlib.axes.Axes
        The figure and axes objects (axes will be array if metric='both')
    """
    
    # Extract all unique classes
    all_classes = set()
    for (class_a, class_b) in class_pair_results.keys():
        all_classes.add(class_a)
        all_classes.add(class_b)
    
    all_classes = sorted(list(all_classes))
    
    # Create labels with text wrapping
    def wrap_text(text, max_chars_per_line=15):
        """Wrap text to multiple lines for better display"""
        import textwrap
        if len(text) <= max_chars_per_line:
            return text
        # Use textwrap to break at word boundaries
        wrapped = textwrap.fill(text, width=max_chars_per_line)
        return wrapped
    
    if use_names and class_info is not None:
        labels = [class_info.get(class_id, {}).get('name', f'Class {class_id}') for class_id in all_classes]
        # Wrap long names for better display
        labels = [wrap_text(label) for label in labels]
    else:
        labels = [str(class_id) for class_id in all_classes]
    
    # Determine if showing both metrics
    show_both = metric == 'both'
    
    if show_both:
        # Adjust figsize for side-by-side plots
        figsize = (figsize[0] * 2, figsize[1])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        axes = [ax1, ax2]
        metrics = ['length', 'density']
    else:
        fig, ax = plt.subplots(figsize=figsize)
        axes = [ax]
        metrics = [metric]
    
    for i, current_metric in enumerate(metrics):
        # Create matrix for current metric
        matrix = np.zeros((len(all_classes), len(all_classes)))
        
        # Fill matrix with boundary data
        for (class_a, class_b), result in class_pair_results.items():
            if isinstance(result, tuple):
                # Has both length and density
                value = result[0] if current_metric == 'length' else result[1]
            else:
                # Only has length
                value = result
                
            idx_a = all_classes.index(class_a)
            idx_b = all_classes.index(class_b)
            
            # Fill both symmetric positions
            matrix[idx_a, idx_b] = value
            matrix[idx_b, idx_a] = value
        
        # Create DataFrame for easier plotting
        df = pd.DataFrame(matrix, index=labels, columns=labels)
        
        # Create heatmap with viridis colormap
        current_ax = axes[i] if show_both else ax
        
        # Create heatmap with viridis colormap
        current_ax = axes[i] if show_both else ax
        sns.heatmap(df, 
                    annot=True, 
                    fmt='.3f' if current_metric == 'length' else '.6f',
                    cmap='viridis',
                    cbar_kws={'label': 'Boundary Length (km)' if current_metric == 'length' else 'Edge Density (m/m²)'},
                    ax=current_ax)
        
        # Set labels and title with units
        if current_metric == 'length':
            title = "Class Pair Boundary Lengths [km]"
        else:
            title = "Class Pair Edge Densities [m/m²]"
        current_ax.set_title(title, fontsize=16, pad=20)
        
        # Rotate labels for better readability with proper alignment for wrapped text
        current_ax.tick_params(axis='x', rotation=45, labelrotation=45)
        current_ax.tick_params(axis='y', rotation=0)
    
    plt.tight_layout()
    plt.show()

def choose_region(regions_dict):
    '''
    Generate a dropdown menu for selecting a region.
    
    Input:
    - `regions_dict (dict)`: dictionary with region names as keys and corresponding descriptions as values.
    
    Output:
    - chosen region (list with one element)
    '''

    # define a dropdown menu for selecting a region
    region_dropdown = widgets.Dropdown(
        options=list(regions_dict.keys()),
        value=list(regions_dict.keys())[0],
        description='Region:'
    )

    # define a variable to store the chosen value
    chosen_region = [region_dropdown.value]

    # define a callback function to update the variable when the dropdown value changes
    def on_region_change(change):
        # global chosen_region
        chosen_region[0] = change['new']

    # attach the callback function to the dropdown menu
    region_dropdown.observe(on_region_change, names='value')

    # display the dropdown menu
    display(region_dropdown)
    
    return chosen_region

def list_filepaths(dir, patterns_in, patterns_out, include_all_patterns=True, print_warning=True):






    '''
    List of filepaths in a dir that contain patterns in patterns_in, and do not contain patterns in patterns_out.
    
    Input:
    - `dir (str)`: path to the directory to be searched.
    - `patterns_in (list)`: list of patterns that the filenames should contain.
    - `patterns_out (list)`: list of patterns that the filenames should not contain.
    - `include_all_patterns (bool)`: if True, all patterns in patterns_in are required in a single filename.
            If False, any of the patterns in patterns_in is sufficient. Default is True.
    - `print_warning (bool)`: if True, print a warning if no paths are found for the specified patterns. Default is True.
    
    Output:
    - list of filepaths that satisfy the specified conditions.
    '''
    
    if include_all_patterns:
        def patterns_in_bool(i):
            return all(pattern in i for pattern in patterns_in)
    else:
        def patterns_in_bool(i):
            return any(pattern in i for pattern in patterns_in)
        
    def patterns_out_bool(i):
        return all(pattern not in i for pattern in patterns_out)

    out = [os.path.join(dir,i) for i in os.listdir(dir) if patterns_in_bool(i) and patterns_out_bool(i)]
    
    if not out and print_warning:
        print(f'Warning! No paths found for the specified patterns ({patterns_in}) in {dir} (returning an empty list). ')
        
    return out

def read_image(rasters_dir, chosen_region, dataset_label, target_projection = '4326', mask_below=None):    
    '''
    Read raster into an array.
    
    Input:
    - `rasters_dir (str)`: path to the directory with rasters
    - `chosen_region (str)`: name of the region to be displayed.
            Used to identify the right region images in `rasters_dir`. 
    - `dataset_label (str)`: label of the dataset to be displayed 
            (`'IMD'` for imperviousness, `'LSM'` for land surface temperature).
            Used to identify the dataset in `rasters_dir`.
    - `mask_below (float)`: mask values below this threshold. 
            If `None`, no masking is applied. Default is `None`.
    - `target_projection (str)`: target projection of the raster. Default is `'4326'`.
        
    Output:
    - dictionary with the following keys:
        - `'array' (numpy array)`: array with the raster values
        - `'bounds' (list)`: bounds of the raster, format: `[[bottom, left], [top, right]]`
        - `'min_value' (float)`: minimum value of the raster
        - `'max_value' (float)`: maximum value of the raster
        - `'crs' (str)`: coordinate reference system of the raster
        - `'mask' (numpy array)`: array with the mask of the values below the threshold (if `mask_below` is not `None`)
    '''

    # get the region label from the regions_dict
    region_label = regions_dict[chosen_region][2]
    
    # get the path to the right raster
    path_to_dataset = list_filepaths(rasters_dir, 
        [dataset_label, region_label,  '.tif', target_projection], ['.aux'])[0]

    # read the dataset into array
    with rasterio.open(path_to_dataset) as src:
        arr = src.read(1).astype(np.float32)
        
        # mask nodata values
        arr[arr == src.nodata] = np.nan
        
        # get CRS, bounds, min and max values
        src_crs = src.crs.to_string().upper()
        bounds = [[src.bounds.bottom, src.bounds.left], [src.bounds.top, src.bounds.right]]
        arr_min = np.nanmin(arr)
        arr_max = np.nanmax(arr)
        
        # mask values below the threshold if requested
        if mask_below is not None:
            mask = np.where(arr<mask_below)
            arr[mask] = np.nan
        
    # return the dictionary with the raster properties
    output_dict = {
        'array': arr, 
        'bounds': bounds, 
        'min_value': arr_min, 
        'max_value': arr_max, 
        'crs': src_crs
    }
    
    if mask_below is not None:
        output_dict['mask'] = mask
    
    return output_dict

def save_as_png(arr, path, color_code='viridis', clim=None, reverse=False):
    '''
    Save an array as a colored PNG image.
    
    Input:
    - `arr (numpy array)`: array to be saved
    - `path (str)`: path to the output PNG file
    - `color_code (str)`: color code to be used for the dataset. Default is `'viridis'`.
    - `clim (tuple)`: min and max values for the color scale. Default is `None`.
    - `reverse (bool)`: whether to reverse the color code. Default is `False`. 
         
    Output: none
    '''
    
    # normalize array data to the range [0, 1]
    if not clim:
        norm = Normalize(vmin=np.nanmin(arr), vmax=np.nanmax(arr))
    else:
        norm = Normalize(vmin=clim[0], vmax=clim[1])
        
    arr_norm = norm(arr)

    # apply colormap
    colormap = plt.get_cmap(color_code)
    arr_colored = colormap(arr_norm)

    # convert the image to uint8 format
    arr_uint8 = (arr_colored[:, :, :3] * 255).astype(np.uint8)

    # set nan values to be transparent
    arr_uint8_with_alpha = np.dstack((arr_uint8, (~np.isnan(arr) * 255).astype(np.uint8)))

    # save image
    image = Image.fromarray(arr_uint8_with_alpha, mode='RGBA')
    image.save(path)

def plot_discrete_histogram(rasters_dir, chosen_region, dataset_label='CLCplusBB', 
                           class_info=None, target_projection='4326', 
                           figure_size=(10, 6), title=None, nodata_value=0):
    '''
    Plot a histogram for discrete/categorical values with bars colored according to class colors.
    
    Input:
    - `rasters_dir (str)`: path to the directory with rasters
    - `chosen_region (str)`: name of the region to be analyzed
    - `dataset_label (str)`: label of the dataset (default 'CLCplusBB')
    - `class_info (dict)`: Dictionary mapping class IDs to {"name": str, "color": str}.
                          If None, uses class numbers with default gray colors.
    - `target_projection (str)`: target projection (default '4326')
    - `figure_size (tuple)`: size of the figure (default (10, 6))
    - `title (str)`: custom title for the plot (optional)
    - `nodata_value (int)`: value to treat as nodata (default 0)
    
    Output: matplotlib figure showing discrete histogram with class colors
    '''
    
    # class_info is optional - if None, will use class numbers instead
    
    # Read the raster data
    dataset_dict = read_image(rasters_dir, chosen_region, dataset_label, 
                             target_projection=target_projection)
    arr = dataset_dict['array']
    
    # Get unique classes (excluding NaN values and nodata values)
    valid_mask = ~np.isnan(arr) & (arr != nodata_value)
    unique_classes = np.unique(arr[valid_mask])
    unique_classes = sorted([int(cls) for cls in unique_classes if not np.isnan(cls)])
    
    # Count occurrences of each class
    class_counts = {}
    total_pixels = np.sum(valid_mask)
    
    for class_id in unique_classes:
        count = np.sum(arr == class_id)
        class_counts[class_id] = count
    
    # Prepare data for plotting
    class_ids = list(class_counts.keys())
    counts = list(class_counts.values())
    colors = []
    labels = []
    
    for class_id in class_ids:
        # Get color and name from class_info if provided
        if class_info is not None and class_id in class_info:
            color = class_info[class_id].get('color', '#808080')
            name = class_info[class_id].get('name', f'Class {class_id}')
        else:
            color = '#808080'  # Gray for unknown classes or when no class_info provided
            name = f'Class {class_id}'
        
        colors.append(color)
        labels.append(f'{name} ({class_id})')
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figure_size)
    
    # Create bars with class colors
    bars = ax.bar(range(len(class_ids)), counts, color=colors, 
                  edgecolor='black', linewidth=0.8, alpha=0.8)
    
    # Set labels and formatting
    ax.set_xlabel('Land Cover Classes')
    ax.set_ylabel('Number of Pixels')
    ax.set_xticks(range(len(class_ids)))
    ax.set_xticklabels([f'{class_id}' for class_id in class_ids], rotation=45, ha='right')
    
    # Add percentage labels on top of bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        percentage = (count / total_pixels) * 100
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{percentage:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Set title
    if title is None:
        title = f'{chosen_region} \nLand Cover Class Distribution'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add grid
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    
    # Create legend with class names and colors
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, edgecolor='black', alpha=0.8, label=label) 
                      for color, label in zip(colors, labels)]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', 
              title='Land Cover Classes', title_fontsize=12)
    
    plt.tight_layout()
    
    plt.show()
    plt.close()