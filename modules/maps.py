import os
import numpy as np

import folium
import branca.colormap as cm

from modules.regions_dict import regions_dict
from modules.utils import read_image, save_as_png

def prepare_layer_to_map_discrete(rasters_dir, chosen_region, layer_dict, class_info,
                                  target_projection='4326', figure_size=(14, 14), dpi=300):
    '''
    Helper function to process a discrete/categorical raster dataset, convert it to a high-resolution PNG file,
    and return properties for displaying on a Folium map.
    
    Input:
    - `rasters_dir (str)`: directory containing raster datasets.
    - `chosen_region (str)`: selected region name.
    - `layer_dict (dict)`: dictionary containing properties of the dataset.
        It should contain the following keys:
        - `label (str)`: label of the dataset
        - `layer_name (str)`: name to be displayed on the map
        - `opacity (float)`: opacity of the layer
    - `class_info (dict)`: Dictionary mapping class IDs to {"name": str, "color": str}.
    - `target_projection (str)`: target projection of the raster. Default is '4326'.
    - `figure_size (tuple)`: matplotlib figure size used when rendering the PNG (default (14,14) for higher detail).
    - `dpi (int)`: dots per inch for PNG export (default 300 for crisper display when zooming).

    Output:
    - `path_to_png (str)`: path to the saved PNG file.
    - `bounds (list)`: bounding box of the dataset.
    - `class_colors (list)`: list of colors for each class present in the data.
    - `class_names (list)`: list of class names for each class present in the data.
    '''
    dataset_label = layer_dict['label']

    # read the dataset into an array
    dataset_dict = read_image(rasters_dir, chosen_region, dataset_label, target_projection=target_projection)
    arr, bounds = dataset_dict['array'], dataset_dict['bounds']
    
    # get unique values in the raster
    unique_vals = np.unique(arr)
    # remove nodata values if they exist
    if 'nodata' in dataset_dict and dataset_dict['nodata'] is not None:
        unique_vals = unique_vals[unique_vals != dataset_dict['nodata']]
    # also treat 0 as nodata value
    unique_vals = unique_vals[unique_vals != 0]
    
    # create discrete colormap using class_info
    colors = []
    class_names = []
    
    for val in unique_vals:
        if not np.isnan(val):
            val_int = int(val)
            class_data = class_info.get(val_int, {})
            color = class_data.get('color', '#808080')  # Gray for unknown classes
            name = class_data.get('name', f'Class {val_int}')
            colors.append(color)
            class_names.append(name)
    
    # create discrete PNG using matplotlib
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import matplotlib.pyplot as plt
    
    # create discrete colormap
    cmap = ListedColormap(colors)
    bounds_norm = np.arange(len(colors) + 1) - 0.5
    norm = BoundaryNorm(bounds_norm, cmap.N)
    
    # map values to indices for visualization
    arr_indexed = np.full_like(arr, -1, dtype=float)
    for idx, val in enumerate(unique_vals):
        if not np.isnan(val) and idx < len(colors):
            arr_indexed[arr == val] = idx
    
    # mask unmapped values and 0 values (treated as nodata)
    arr_indexed = np.ma.masked_where((arr_indexed == -1) | (arr == 0), arr_indexed)
    
    # save as PNG
    os.makedirs('tmp', exist_ok=True)
    path_to_png = f'tmp/tmp_{dataset_label}_discrete.png'
    
    fig_temp, ax_temp = plt.subplots(figsize=figure_size)
    ax_temp.imshow(arr_indexed, cmap=cmap, norm=norm)
    ax_temp.set_xticks([])
    ax_temp.set_yticks([])
    ax_temp.axis('off')
    plt.savefig(path_to_png, bbox_inches='tight', pad_inches=0, dpi=dpi, transparent=True)
    plt.close()
    
    return path_to_png, bounds, colors, class_names

def display_map_discrete(rasters_dir, chosen_region, base_map, layer_dict, class_info,
                         map_size=(650, 400), target_projection='4326',
                         overlay_figure_size=(14,14), overlay_dpi=300):
    '''
    Create a folium map of the chosen region showing a single discrete/categorical layer.
    Designed for discrete datasets like land cover classifications.
    
    Input:
    - `rasters_dir (str)`: path to the directory with rasters to be displayed
    - `chosen_region (str)`: name of the region to be displayed. Used to identify: 
        (1) the right images in rasters_dir, (2) the coordinates of the region and, 
        (3) zoom level for the map. 
    - `base_map (str)`: name of the base map to be used (e.g. 'OpenStreetMap')
    - `layer_dict (dict)`: dictionary with properties of the layer to be displayed.
        The dictionary should contain the following keys:
        - `label (str)`: label of the dataset
        - `layer_name (str)`: name to be displayed on the map
        - `opacity (float)`: opacity of the layer
    - `class_info (dict)`: Dictionary mapping class IDs to {"name": str, "color": str}.
    - `map_size (tuple)`: size of the map in pixels (width, height). Default is (650, 400).
    - `target_projection (str)`: target projection of the raster. Default is '4326'.
    - `overlay_figure_size (tuple)`: figure size used to render the categorical PNG overlay (default (14,14)).
    - `overlay_dpi (int)`: dpi used to render the overlay PNG (default 300).
                    
    Output: folium map of the chosen region with the discrete layer and legend
    '''
    
    # create a folium map centered around the chosen region
    coordinates = regions_dict[chosen_region][0]
    figure = folium.Figure(width=map_size[0], height=map_size[1])
    map = folium.Map(coordinates, zoom_start=regions_dict[chosen_region][1], tiles=base_map).add_to(figure)    

    # process the discrete layer using the discrete-specific function
    path_to_png, bounds, colors, class_names = prepare_layer_to_map_discrete(
        rasters_dir, chosen_region, layer_dict, class_info,
        target_projection=target_projection,
        figure_size=overlay_figure_size,
        dpi=overlay_dpi
    )

    # add ImageOverlay to the map
    folium.raster_layers.ImageOverlay(
        image=path_to_png,
        name=layer_dict['layer_name'],
        bounds=bounds,  # bounds is already in [[bottom, left], [top, right]] format
        opacity=layer_dict['opacity'],
        interactive=False,
        cross_origin=False,
        zindex=1,
        alt=layer_dict['layer_name']
    ).add_to(map)
    
    # create custom legend for discrete classes
    legend_html = f"""
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 200px; height: auto; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <b>{layer_dict['layer_name']}</b><br>
    """
    
    for color, name in zip(colors, class_names):
        legend_html += f"""
        <i style="background:{color}; width:15px; height:15px; float:left; margin-right:8px; margin-top:2px;"></i>
        {name}<br style="clear:both">
        """
    
    legend_html += "</div>"
    map.get_root().html.add_child(folium.Element(legend_html))
    
    # HTML custom title placement
    map_title = f'Selected region: {chosen_region}'
    title_html = f"""<h1 style="position:absolute;z-index:1000;top:12px;left:55px;font-size:16px;">{map_title}</h1>"""
    map.get_root().html.add_child(folium.Element(title_html))
    
    return map
