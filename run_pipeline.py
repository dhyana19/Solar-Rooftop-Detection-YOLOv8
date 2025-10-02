import os
import inference_and_merge
import gis_post_processing
import data_prep_utils 

# Configuration
# NOTE: Update all paths to point to your local machine's data/model locations
CONFIG = {
    # Data paths (Update these!)
    "ORIGINAL_TIF_PATH": "data/marhara_satellite_image.tif", # The full image
    "MODEL_WEIGHTS": "weights/best.pt", # Your trained model weights
    
    # Output paths (Will be created)
    "OUTPUT_DIR": "outputs",
    "MERGED_MASK_PATH": "outputs/marhara_merged_mask.png",
    "FINAL_GEOJSON_PATH": "outputs/rooftop_polygons_final.geojson",

    # Tiling/Inference Parameters
    "TILE_SIZE": 1024,
    "OVERLAP": 256 # 25% overlap
}


def create_directories(config):
    """Ensure all necessary output directories exist."""
    os.makedirs(config["OUTPUT_DIR"], exist_ok=True)
    print(f"Output directory created/exists at: {config['OUTPUT_DIR']}")


if __name__ == '__main__':
    print("Starting Solar Rooftop Detection Pipeline")
    
    create_directories(CONFIG)
    
    # Optional: Run Tiling (Only if you need to create training tiles)
    # print("\n--- OPTIONAL: Running Tiling for Training Data ---")
    # TILE_OUTPUT_DIR = os.path.join(CONFIG["OUTPUT_DIR"], "training_tiles")
    # data_prep_utils.split_tiff_to_tiles_with_overlap(
    #     CONFIG["ORIGINAL_TIF_PATH"], TILE_OUTPUT_DIR, 
    #     tile_size=CONFIG["TILE_SIZE"], overlap=CONFIG["OVERLAP"]
    # )
    
    # 1. Run Tiled Inference and Merge
    print("\n STAGE 1: Running Tiled Inference and Merging")
    final_mask_path = inference_and_merge.run_tiled_inference(
        model_path=CONFIG["MODEL_WEIGHTS"], 
        tiff_path=CONFIG["ORIGINAL_TIF_PATH"], 
        save_path=CONFIG["MERGED_MASK_PATH"],
        chunk_size=CONFIG["TILE_SIZE"],
        overlap_ratio=CONFIG["OVERLAP"] / CONFIG["TILE_SIZE"]
    )
    
    # 2. Run GIS Post-Processing and Vectorization
    print("\n STAGE 2: Post-Processing, Georeferencing, and Vectorization")
    gis_post_processing.full_vectorization_pipeline(
        input_mask_path=final_mask_path,
        original_tif_path=CONFIG["ORIGINAL_TIF_PATH"],
        output_geojson_path=CONFIG["FINAL_GEOJSON_PATH"]
    )
    
    print("\n Pipeline Complete!")
    print(f"Final GeoJSON polygons saved to: {CONFIG['FINAL_GEOJSON_PATH']}")