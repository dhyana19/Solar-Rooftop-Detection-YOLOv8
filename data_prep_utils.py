import os
import numpy as np
import cv2
import rasterio
from rasterio.windows import Window
import tifffile
from PIL import Image
from tqdm import tqdm

# 1. Tiling without overlap (JPEG output, based on first function)

def tile_satellite_tif_no_overlap(image_path, output_dir, tile_size=512):
    """
    Tiles a large satellite GeoTIFF image into smaller, non-overlapping JPEG tiles.
    Uses the first 3 bands as RGB.
    """
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    
    with rasterio.open(image_path) as src:
        width, height = src.width, src.height

        for y in tqdm(range(0, height, tile_size), desc="Tiling rows"):
            for x in range(0, width, tile_size):
                if y + tile_size <= height and x + tile_size <= width:
                    window = Window(x, y, tile_size, tile_size)
                    tile = src.read(window=window)

                    if tile.shape[0] >= 3:
                        # Stack first 3 bands for RGB
                        rgb_tile = np.stack([tile[0], tile[1], tile[2]], axis=-1)
                        # Normalize and convert to uint8
                        rgb_tile = np.clip(rgb_tile, 0, 255).astype(np.uint8)

                        tile_path = os.path.join(output_dir, f"tile_{count}.jpg")
                        # Convert RGB to BGR for OpenCV and save
                        cv2.imwrite(tile_path, cv2.cvtColor(rgb_tile, cv2.COLOR_RGB2BGR))
                        count += 1
                    # Note: No 'else' block for partial tiles, only full tiles are saved
    
    print(f"Done! Tiled and saved {count} full, non-overlapping images to: {output_dir}")
    return output_dir, count


# 2. Tiling with overlap (PNG output, based on third function)

def split_tiff_to_tiles_with_overlap(tiff_path, output_folder, tile_size=2048, overlap=512):
    """
    Splits a TIFF image into tiles of tile_size x tile_size, with specified overlap.
    Tiles at the edge are adjusted to fit fully within the image bounds.
    """
    os.makedirs(output_folder, exist_ok=True)

    with tifffile.TiffFile(tiff_path) as tif:
        image = tif.asarray()
        img = Image.fromarray(image)
        
    width, height = img.size
    tile_num = 0
    stride = tile_size - overlap

    for top in range(0, height, stride):
        for left in range(0, width, stride):
            right = left + tile_size
            bottom = top + tile_size
            
            # Edge handling
            if right > width:
                left = width - tile_size
                right = width
            if bottom > height:
                top = height - tile_size
                bottom = height
            
            # Skip if the tile calculation resulted in an invalid area
            if left < 0 or top < 0:
                continue

            box = (left, top, right, bottom)
            tile = img.crop(box)
            tile_path = os.path.join(output_folder, f"tile_{tile_num:04}.png")
            tile.save(tile_path)
            tile_num += 1

    print(f"Saved {tile_num} overlapping tiles to '{output_folder}'")
    return output_folder, tile_num

# 3. Tiling with padding (PNG output, based on fourth function)

def split_tiff_to_tiles_include_all(tiff_path, output_folder, tile_size=2048):
    """
    Splits a TIFF image into tiles, padding edge tiles to maintain consistent size.
    Returns a map of tile coordinates for reconstruction.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    with tifffile.TiffFile(tiff_path) as tif:
        image = tif.asarray()
        img = Image.fromarray(image)

    width, height = img.size
    tile_num = 0
    tile_map = []

    for top in range(0, height, tile_size):
        for left in range(0, width, tile_size):
            right = min(left + tile_size, width)
            bottom = min(top + tile_size, height)

            box = (left, top, right, bottom)
            tile = img.crop(box)

            # Pad the tile if it's smaller than tile_size
            if tile.size != (tile_size, tile_size):
                # Assuming RGB image, fill with black/zeros
                padded_tile = Image.new("RGB", (tile_size, tile_size)) 
                padded_tile.paste(tile, (0, 0))
                tile = padded_tile

            tile_path = os.path.join(output_folder, f"tile_{tile_num:04}.png")
            tile.save(tile_path)
            tile_map.append((left, top, right, bottom))
            tile_num += 1

    print(f"Saved {tile_num} padded tiles to '{output_folder}'")
    return tile_map, width, height

if __name__ == "__main__":
    print("This file contains utility functions and should be imported by the main scripts.")