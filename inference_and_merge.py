import numpy as np
import cv2
from skimage import io
from ultralytics import YOLO
import gc
import torch

# Utility Functions 

def load_and_prepare_image(tiff_path):
    """Loads TIFF, converts to 3-channel RGB (if needed), and normalizes to uint8."""
    image = io.imread(tiff_path)

    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image

def predict_chunk(model, chunk_rgb):
    """Runs prediction on a single image chunk and returns the combined mask."""
    # Run prediction
    result = model.predict(chunk_rgb, verbose=False)[0]

    pred_mask = np.zeros((chunk_rgb.shape[0], chunk_rgb.shape[1]), dtype=np.float32)
    if result.masks is not None:
        for i in range(result.masks.data.shape[0]):
            mask_data = result.masks.data[i].cpu().numpy()
            # Resize mask to chunk size
            mask_resized = cv2.resize(mask_data, (chunk_rgb.shape[1], chunk_rgb.shape[0]))
            pred_mask = np.logical_or(pred_mask, mask_resized).astype(np.float32)
    
    # Clean up GPU memory for large model pipelines
    del result
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return pred_mask


# Main Inference Pipeline

def run_tiled_inference(model_path, tiff_path, save_path, chunk_size=1024, overlap_ratio=0.25):
    """
    Performs tiled inference on a large TIFF image and merges the results.
    """
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    # Use half precision on GPU to save memory if available
    if torch.cuda.is_available():
        model.model.half()

    image = load_and_prepare_image(tiff_path)
    height, width = image.shape[:2]
    
    pad = int(chunk_size * overlap_ratio)
    step = chunk_size 

    merged_mask = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    
    total_chunks = ((height + step - 1) // step) * ((width + step - 1) // step)
    chunk_count = 0

    print(f"Starting inference on {total_chunks} chunks...")

    for y in range(0, height, step):
        for x in range(0, width, step):
            chunk_count += 1
            print(f"Processing chunk {chunk_count}/{total_chunks} at ({x}, {y})", end='\r')

            y0, y1 = max(0, y - pad), min(height, y + chunk_size + pad)
            x0, x1 = max(0, x - pad), min(width, x + chunk_size + pad)
            chunk = image[y0:y1, x0:x1]

            chunk_rgb = cv2.cvtColor(chunk, cv2.COLOR_BGR2RGB)
            pred_mask = predict_chunk(model, chunk_rgb)

            # Merge Logic (extract center and place in merged_mask)
            y_start = pad if y - pad >= 0 else 0
            x_start = pad if x - pad >= 0 else 0
            y_end = y_start + chunk_size
            x_end = x_start + chunk_size
            mask_center = pred_mask[y_start:y_end, x_start:x_end]

            y_dst_start, y_dst_end = y, y + chunk_size
            x_dst_start, x_dst_end = x, x + chunk_size

            merged_mask[y_dst_start:y_dst_end, x_dst_start:x_dst_end] += mask_center
            weight_sum[y_dst_start:y_dst_end, x_dst_start:x_dst_end] += 1.0

            del chunk, chunk_rgb, pred_mask, mask_center
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Final Processing
    merged_mask = np.divide(merged_mask, weight_sum, where=weight_sum != 0)
    merged_mask = np.nan_to_num(merged_mask)
    final_mask = (merged_mask >= 0.5).astype(np.uint8) * 255

    # Clean up mask (Morphological Opening)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)

    cv2.imwrite(save_path, cleaned_mask)
    print(f"\nInference complete. Merged mask saved to: {save_path}")
    return save_path

if __name__ == '__main__':
    # Example Call (Update paths for your environment)
    # MODEL_PATH = '/content/runs/segment/train/weights/best.pt'
    # TIFF_PATH = 'original_satellite_image_path'
    # OUTPUT_MASK_PATH = 'marhara_merged_mask.png'
    # run_tiled_inference(MODEL_PATH, TIFF_PATH, OUTPUT_MASK_PATH)
    print("Script meant for inference. Define and update the run_tiled_inference parameters.")