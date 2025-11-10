"""
Reproject the specific raster from EPSG:3035 to EPSG:4326.
Hardcoded input path; output written alongside with 3035 -> 4326 in filename.
Usage: python reproject_fixed.py
"""

from pathlib import Path
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

INPUT_PATH = Path(r"C:\Users\pasik\code\clms-CLCplusBB\aoi_rasters\buffered_1km\CLCplusBB_Babiogorski_National_Park_3035_BUFFERED_1km.tif")

def derive_output_path(src_path: Path) -> Path:
    name = src_path.name
    if "_3035" not in name:
        raise ValueError("Expected '_3035' in filename to replace.")
    new_name = name.replace("_3035", "_4326", 1)
    return src_path.parent / new_name

def reproject_to_4326(src_path: Path, dst_path: Path):
    with rasterio.open(src_path) as src:
        if str(src.crs).upper() not in ("EPSG:3035", "EPSG:ETRS89 / LAEA EUROPE"):
            print(f"Warning: source CRS reported as {src.crs}")
        dst_crs = "EPSG:4326"

        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )

        profile = src.profile.copy()
        profile.update({
            "crs": dst_crs,
            "transform": transform,
            "width": width,
            "height": height,
            "compress": profile.get("compress", "LZW"),
            "tiled": profile.get("tiled", True)
        })

        band_index = 1
        nodata = src.nodata
        try:
            colormap = src.colormap(band_index)
        except Exception:
            colormap = None

        import numpy as np
        dest = np.zeros((height, width), dtype=profile["dtype"])

        reproject(
            source=rasterio.band(src, band_index),
            destination=dest,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
            src_nodata=nodata,
            dst_nodata=nodata
        )

        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(dest, band_index)
            if nodata is not None:
                dst.nodata = nodata
            if colormap:
                dst.write_colormap(band_index, colormap)
            tags = src.tags()
            if tags:
                dst.update_tags(**tags)

def main():
    if not INPUT_PATH.exists():
        print(f"Input not found: {INPUT_PATH}")
        return
    output_path = derive_output_path(INPUT_PATH)
    print(f"Source: {INPUT_PATH}")
    print(f"Target: {output_path}")
    reproject_to_4326(INPUT_PATH, output_path)
    print("Done.")

if __name__ == "__main__":
    main()