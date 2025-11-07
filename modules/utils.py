import os
import numpy as np
import pandas as pd
import rasterio
import ipywidgets as widgets
from IPython.display import display, HTML

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from modules.regions_dict import regions_dict, parks_info
from PIL import Image

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
    """Visualize class pair boundary results as a confusion matrix-style heatmap.

    Accepts output from `get_boundary_length_per_class_pair` after its refactor to
    return dictionaries per pair: {"length_km": float, "edge_density_m_per_ha": float}.
    Backward compatibility: still supports older tuple outputs (length_km, edge_density_m_per_ha)
    and legacy scalar-only length values.

    Parameters
    ----------
    class_pair_results : dict
        Mapping of (class_a, class_b) -> result where result is one of:
          - dict with keys 'length_km' and 'edge_density_m_per_ha'
          - tuple(length_km, edge_density_m_per_ha)
          - single numeric (interpreted as length_km only)
    class_info : dict, optional
        Mapping class_id -> {"name": str, "color": str} for axis labeling.
    use_names : bool, default True
        Use descriptive names (from class_info) instead of numeric IDs for axes.
    metric : str, default 'length'
        One of 'length', 'density', or 'both'. If 'both', shows length and density side-by-side.
    figsize : tuple, default (12, 10)
        Base figure size; doubled horizontally if metric == 'both'.

    Returns
    -------
    (Figure, Axes or list[Axes])
        Matplotlib Figure and the Axes object(s). If metric == 'both', returns list of two Axes.
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
        labels = [class_info.get(class_id, {}).get('name', f' {class_id}') for class_id in all_classes]
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

        # Fill matrix with boundary data using flexible result parsing
        for (class_a, class_b), result in class_pair_results.items():
            if isinstance(result, dict):
                if current_metric == 'length':
                    value = result.get('length_km', np.nan)
                else:
                    # Support both new key and possible legacy key naming
                    value = result.get('edge_density_m_per_ha', result.get('edge_density', np.nan))
            elif isinstance(result, tuple) and len(result) >= 2:
                value = result[0] if current_metric == 'length' else result[1]
            else:
                # Scalar fallback (assume length only)
                value = result if current_metric == 'length' else np.nan

            idx_a = all_classes.index(class_a)
            idx_b = all_classes.index(class_b)

            # Fill both symmetric positions
            matrix[idx_a, idx_b] = value
            matrix[idx_b, idx_a] = value

        # Create DataFrame for easier plotting
        df = pd.DataFrame(matrix, index=labels, columns=labels)

        # Choose formatting precision based on metric
        fmt = '.3f' if current_metric == 'length' else '.6f'

        # Create heatmap
        current_ax = axes[i] if show_both else ax
        sns.heatmap(
            df,
            annot=True,
            fmt=fmt,
            cmap='viridis',
            cbar_kws={'label': 'Boundary Length (km)' if current_metric == 'length' else 'Edge Density (m/ha)'},
            ax=current_ax
        )

        # Set labels and title with units
        title = "Class Pair Boundary Lengths [km]" if current_metric == 'length' else "Class Pair Edge Densities [m/ha]"
        current_ax.set_title(title, fontsize=16, pad=20)

        # Rotate labels for readability
        current_ax.tick_params(axis='x', rotation=45, labelrotation=45)
        current_ax.tick_params(axis='y', rotation=0)
    
    plt.tight_layout()
    plt.show()

def choose_region(regions_dict):
    """Display a region selection dropdown and return the widget itself.

    This avoids capturing only the initial default value. Read the current
    selection via `dropdown.value` in subsequent cells after user interaction.

    Parameters
    ----------
    regions_dict : dict
        Dictionary whose keys are human-readable region names.

    Returns
    -------
    ipywidgets.Dropdown
        The dropdown widget; use `widget.value` to access the selected region.
    """
    dropdown = widgets.Dropdown(
        options=list(regions_dict.keys()),
        value=list(regions_dict.keys())[0],
        description='Region:'
    )
    display(dropdown)
    return dropdown

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

def read_image(rasters_dir, chosen_region, dataset_label, target_projection='4326', mask_below=None):
    """
    Read a raster into an array for the specified region and dataset.

    Accepts either a human-readable region name (e.g. "Swietokrzyski National Park")
    or a slug (e.g. "Swietokrzyski_National_Park"). Tries both slug and human forms
    when locating the raster file.

    Parameters
    ----------
    rasters_dir : str
        Directory containing raster files.
    chosen_region : str
        Region name (with spaces) or slug (underscores).
    dataset_label : str
        Dataset identifier included in filename (e.g. 'CLCplusBB').
    target_projection : str, default '4326'
        Projection string expected in the raster filename.
    mask_below : float, optional
        If provided, values below this threshold are set to NaN.

    Returns
    -------
    dict
        {
          'array': np.ndarray (float32, nodata -> NaN, optional masking applied),
          'bounds': [[bottom, left], [top, right]],
          'min_value': float,
          'max_value': float,
          'crs': str,
          'mask': np.ndarray (indices masked)  # only if mask_below provided
        }

    Raises
    ------
    FileNotFoundError
        If no matching raster file is found after all attempts.
    """
    # Determine slug/human variants
    if '_' in chosen_region and ' ' not in chosen_region:
        slug = chosen_region.strip()
        human = slug.replace('_', ' ')
    else:
        human = chosen_region.strip()
        slug = human.replace(' ', '_')

    # Order of filename fragments to try
    region_labels_to_try = [slug, human]

    path_to_dataset = None
    for region_label in region_labels_to_try:
        try:
            paths = list_filepaths(
                rasters_dir,
                [dataset_label, region_label, '.tif', target_projection],
                ['.aux'],
                include_all_patterns=True,
                print_warning=False
            )
            if paths:
                path_to_dataset = paths[0]
                break
        except Exception:
            continue

    if path_to_dataset is None:
        raise FileNotFoundError(
            f"No raster found in '{rasters_dir}' for region '{chosen_region}' "
            f"(tried: {region_labels_to_try}) with dataset '{dataset_label}' and projection '{target_projection}'."
        )

    with rasterio.open(path_to_dataset) as src:
        arr = src.read(1).astype(np.float32)
        arr[arr == src.nodata] = np.nan
        src_crs = src.crs.to_string().upper()
        bounds = [[src.bounds.bottom, src.bounds.left], [src.bounds.top, src.bounds.right]]
        arr_min = np.nanmin(arr)
        arr_max = np.nanmax(arr)

        if mask_below is not None:
            mask = np.where(arr < mask_below)
            arr[mask] = np.nan

    out = {
        'array': arr,
        'bounds': bounds,
        'min_value': arr_min,
        'max_value': arr_max,
        'crs': src_crs
    }
    if mask_below is not None:
        out['mask'] = mask
    return out

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

def plot_dual_histograms(rasters_dir1, rasters_dir2, chosen_region, dataset_label,
                        class_info=CLC_CLASS_INFO, target_projection='4326',
                        title1="Area 1", title2="Area 2", figure_size=(16, 8),
                        wrap_width=16, headroom_factor=1.25):
    """
    Plot two histograms side by side with a common percentage y-axis and shared legend.
    
    Parameters
    ----------
    rasters_dir1 : str
        Path to the first raster directory
    rasters_dir2 : str  
        Path to the second raster directory
    chosen_region : str
        Name of the chosen region
    dataset_label : str
        Label for the dataset (e.g., 'CLCplusBB')
    class_info : dict
        Dictionary mapping class IDs to {"name": str, "color": str}
    target_projection : str
        Target projection for reprojection (default '4326')
    title1 : str
        Title for the first histogram
    title2 : str
        Title for the second histogram
    figure_size : tuple
        Figure size (width, height)
    """
    from modules.analysis import calculate_class_areas
    
    # Calculate class areas for both datasets
    area_results1 = calculate_class_areas(rasters_dir1, chosen_region, dataset_label, 
                                         target_projection, class_info)
    area_results2 = calculate_class_areas(rasters_dir2, chosen_region, dataset_label, 
                                         target_projection, class_info)
    
    # Extract data for plotting
    def extract_plot_data(area_results):
        class_areas = area_results['class_areas']
        class_percentages = area_results['class_percentages']
        pixel_size_m = area_results['pixel_size_m']
        total_area_m2 = area_results['total_area_m2']
        
        # Calculate pixel counts from areas
        pixel_area_m2 = pixel_size_m ** 2
        total_pixels = int(total_area_m2 / pixel_area_m2)
        
        class_ids = list(class_areas.keys())
        counts = [int(area_m2 / pixel_area_m2) for area_m2 in class_areas.values()]
        percentages = list(class_percentages.values())
        
        colors = [class_info[class_id]["color"] for class_id in class_ids]
        labels = [class_info[class_id]["name"] for class_id in class_ids]
        
        return class_ids, counts, percentages, colors, labels, total_pixels
    
    data1 = extract_plot_data(area_results1)
    data2 = extract_plot_data(area_results2)
    
    # Get all unique classes from both datasets for consistent scaling
    all_classes = sorted(list(set(data1[0] + data2[0])))
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figure_size, sharey=True)
    
    # Function to create histogram
    def create_histogram(ax, data, title):
        import textwrap
        class_ids, counts, percentages, colors, labels, total_pixels = data

        # Wrapped labels (insert '\n' to break long names)
        wrapped_labels = [textwrap.fill(label, wrap_width) for label in labels]

        # Use percentages for bar heights
        bars = ax.bar(range(len(class_ids)), percentages, color=colors,
                      edgecolor='black', linewidth=0.8, alpha=0.85)

        # Axis labels
        ax.set_xlabel('Land Cover Classes', fontsize=12)
        if ax is ax1:  # y-axis only on left subplot
            ax.set_ylabel('Percentage of Pixels (%)', fontsize=12)

        # X ticks: wrapped class names
        ax.set_xticks(range(len(class_ids)))
        # Rotate wrapped labels to reduce horizontal overlap while keeping multi-line wrapping
        ax.set_xticklabels(wrapped_labels, rotation=40, ha='right', fontsize=9)

        # Percentage annotations above bars
        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            # Slightly larger offset to avoid touching top border when scaled tightly
            ax.text(bar.get_x() + bar.get_width()/2., height + max(height*0.02, 0.7),
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)

        # Title
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Grid styling
        ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)

        # Dynamic y-limit based on data (headroom factor)
        max_pct = max(percentages) if percentages else 0
        def round_up(val, base=5):
            return base * int(np.ceil(val / base)) if val > 0 else base
        upper = round_up(max_pct * headroom_factor)
        if upper < 5:
            upper = 5
        if upper > 100 and max_pct <= 100:
            upper = 100
        if upper - max_pct < 3:
            upper = round_up(upper + 3)
            if upper > 100 and max_pct <= 100:
                upper = 100
        ax.set_ylim(0, upper)

        return bars, colors, wrapped_labels
    
    # Create both histograms (now using class names on x-axis; legend removed)
    bars1, colors1, labels1 = create_histogram(ax1, data1, title1)
    bars2, colors2, labels2 = create_histogram(ax2, data2, title2)

    # Tight layout without legend space adjustments
    plt.tight_layout()
    
    plt.show()
    plt.close()

def display_park_info(region_name_or_slug, image_width=160, max_width=1100):
    """Display park info panel with logo + location map and descriptive text.

    Accepts either a human-readable park name (key in parks_info) OR a slug
    (underscored). If a slug is provided, it attempts to convert it back to
    the human-readable form for description lookup while retaining the slug
    for asset file paths.

    Parameters
    ----------
    region_name_or_slug : str
        Human-readable park name (e.g. "Swietokrzyski National Park") or slug (e.g. "Swietokrzyski_National_Park").
    image_width : int, default 160
        Width (px) for the logo and location map images.
    max_width : int, default 1100
        Max overall panel width.

    Returns
    -------
    IPython.display.HTML
        Rendered HTML panel.
    """
    import html as _html

    # Detect slug vs human-readable name
    if region_name_or_slug in parks_info:
        human_name = region_name_or_slug
        slug = region_name_or_slug.replace(" ", "_")
    else:
        # Treat input as slug; attempt to recover human name by replacing underscores
        slug = region_name_or_slug
        candidate = slug.replace("_", " ")
        print("candidate:   ", candidate)
        human_name = candidate if candidate in parks_info else slug  # fallback to slug

    # Description text (fallback if not found)
    park_text = parks_info.get(human_name, ["No descriptive information available for this park."])[0]

    # Asset paths based on slug
    logo_path = f"images/logos/{slug}_logo.png"
    location_path = f"images/maps/{slug}_location_3035.png"

    # Build <img> tags only if files exist
    logo_tag = (
        f"<img src='{logo_path}' alt='{_html.escape(human_name)} logo' style='width:{image_width}px; height:auto; display:block;'>"
        if os.path.exists(logo_path) else ""
    )
    location_tag = (
        f"<img src='{location_path}' alt='{_html.escape(human_name)} location map' style='width:{image_width}px; height:auto; display:block;'>"
        if os.path.exists(location_path) else ""
    )

    escaped_body = _html.escape(park_text).replace("\n", "<br>")
    escaped_title = _html.escape(human_name)

    html_block = f"""
<div style='display:flex; align-items:flex-start; gap:16px; max-width:{max_width}px;'>
  <div style='flex:0 0 auto; display:flex; flex-direction:column; gap:10px;'>
    {logo_tag}
    {location_tag}
  </div>
  <div style='flex:1 1 auto; line-height:1.5; font-size:15px;'>
    <h3 style='margin:0 0 12px 0; font-size:22px; font-weight:600;'>{escaped_title}</h3>
    <div style='text-align:justify; text-justify:inter-word;'>
      <p style='margin:0;'>{escaped_body}</p>
    </div>
  </div>
</div>
"""
    return HTML(html_block)