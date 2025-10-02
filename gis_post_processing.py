import os
import cv2
import numpy as np
import rasterio
import fiona
from shapely.geometry import shape, mapping
from rasterio.features import shapes
from rasterio.transform import from_origin
from fiona.crs import from_epsg
import geopandas as gpd
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

# Helper Functions

def get_georeferencing_info(tif_path):
    """Extracts GeoTIFF transform and CRS."""
    with rasterio.open(tif_path) as dataset:
        transform = dataset.transform
        crs = dataset.crs
    return transform, crs

def geo_reference_mask(mask_path, original_tif_path, output_tif_path):
    """Georeferences a predicted PNG mask using metadata from the original TIFF."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not load mask from: {mask_path}")

    transform, crs = get_georeferencing_info(original_tif_path)

    with rasterio.open(
        output_tif_path, "w", driver="GTiff",
        height=mask.shape[0], width=mask.shape[1], count=1,
        dtype=mask.dtype, crs=crs, transform=transform
    ) as dst:
        dst.write(mask, 1)
    
    print(f"Georeferenced mask saved to: {output_tif_path}")
    return output_tif_path

def raster_to_geojson(georeferenced_mask_path, output_geojson_path, threshold_value=128):
    """Converts a georeferenced raster mask to a vector GeoJSON file."""
    with rasterio.open(georeferenced_mask_path) as src:
        raster = src.read(1)
        transform = src.transform
        raster_crs = src.crs

    binary_raster = (raster > threshold_value).astype(np.uint8)
    shapes_generator = shapes(binary_raster, transform=transform)

    polygons = []
    for geom, value in shapes_generator:
        if value == 1: # Only export shapes where the pixel value is '1' (or > threshold)
            polygons.append(shape(geom))

    # Define schema and CRS for the output GeoJSON
    schema = {'geometry': 'Polygon', 'properties': {'id': 'int'}}
    crs = from_epsg(4326)

    with fiona.open(output_geojson_path, 'w', driver='GeoJSON', crs=crs, schema=schema) as output:
        for i, poly in enumerate(polygons):
            output.write({'geometry': mapping(poly), 'properties': {'id': i}})

    print(f"Vector GeoJSON saved to: {output_geojson_path}")

def calculate_area(geojson_path):
    """Loads GeoJSON, calculates area in m², and saves back."""
    gdf = gpd.read_file(geojson_path)
    gdf = gdf.to_crs(epsg=3857)  # Convert to a metric CRS for accurate area calculation (meters)
    gdf["area_m2"] = gdf.geometry.area
    gdf = gdf.to_crs(epsg=4326)  # Convert back to WGS84 for GeoJSON

    gdf.to_file(geojson_path, driver="GeoJSON")
    print("Final GeoJSON updated with accurate 'area_m2' column.")


# Mask Refinement/Smoothing

def smooth_mask_for_gis(mask_path, save_path, downscale_factor=2, max_peaks=1000):
    """Applies morphological ops, distance transform, and rectangular approximation."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not load mask from: {mask_path}")

    original_shape = mask.shape
    if downscale_factor > 1:
        mask = cv2.resize(mask, (mask.shape[1] // downscale_factor, mask.shape[0] // downscale_factor))

    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    # Distance transform and Watershed setup
    dist_transform = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5).astype(np.float32)
    dist_norm = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)

    local_max = peak_local_max(dist_norm, min_distance=15, threshold_abs=0.2, labels=cleaned, num_peaks=max_peaks)
    markers = np.zeros_like(dist_transform, dtype=np.int32)
    for i, (y, x) in enumerate(local_max):
        markers[y, x] = i + 1

    labels = watershed(-dist_transform, markers, mask=cleaned)

    # Rectangular approximation
    rect_mask = np.zeros_like(binary_mask)
    for label_id in np.unique(labels):
        if label_id == 0: continue
        rooftop_mask = (labels == label_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(rooftop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 50: continue
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            cv2.fillPoly(rect_mask, [box], 255)

    # Upscale back to original size
    if downscale_factor > 1:
        rect_mask = cv2.resize(rect_mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(save_path, rect_mask)
    print(f"Smoothed and refined mask saved to: {save_path}")
    return save_path


def full_vectorization_pipeline(input_mask_path, original_tif_path, output_geojson_path):
    """Runs the full post-processing, georeferencing, and vectorization."""
    # 1. Refine Mask (using your existing logic)
    smoothed_mask_path = smooth_mask_for_gis(
        mask_path=input_mask_path, 
        save_path=input_mask_path.replace('.png', '_smoothed.png'),
        # Use appropriate parameters, these are from your code:
        downscale_factor=2, max_peaks=1000
    )
    
    # 2. Georeference the Mask
    geo_tif_path = geo_reference_mask(
        mask_path=smoothed_mask_path, 
        original_tif_path=original_tif_path, 
        output_tif_path=output_geojson_path.replace('.geojson', '.tif')
    )
    
    # 3. Raster to Vector (GeoJSON)
    raster_to_geojson(geo_tif_path, output_geojson_path)
    
    # 4. Calculate Area
    calculate_area(output_geojson_path)


if __name__ == '__main__':
    # Example Configuration
    # NOTE: These paths reference your Colab setup, update them locally!
    # INPUT_MERGED_MASK = ' ' # Output from inference_and_merge.py
    # ORIGINAL_TIF_PATH = ' '
    # FINAL_GEOJSON_PATH = 'final_rooftop_polygons.geojson'
    
    # full_vectorization_pipeline(INPUT_MERGED_MASK, ORIGINAL_TIF_PATH, FINAL_GEOJSON_PATH)
    print("Script meant for GIS post-processing. Define and update paths to run.")