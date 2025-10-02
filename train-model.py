from ultralytics import YOLO

# --- Setup Steps (Normally done in the environment/terminal) ---
# !pip install -U ultralytics
# from google.colab import drive
# drive.mount('/content/drive')
# !unzip "path_to_your_image_chunk_files"
# ----------------------------------------------------------------

def train_yolov8_model(data_yaml_path, epochs, img_size, batch_size, model_type='yolov8n-seg.pt'):
    """
    Trains a YOLOv8 segmentation model.

    Args:
        data_yaml_path (str): Path to the dataset YAML file.
        epochs (int): Number of training epochs.
        img_size (int): Image size for training.
        batch_size (int): Batch size for training.
        model_type (str): Initial model weight (e.g., 'yolov8n-seg.pt').
    """
    print(f"Starting training on {model_type}...")
    model = YOLO(model_type)
    
    # Train the model
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size
    )
    print("Training completed. Weights saved in 'runs/segment/train...'")
    return results

if __name__ == '__main__':
    # Define parameters (adjust paths/values as needed for your local setup)
    DATA_YAML = 'path/to/your/data.yaml'
    EPOCHS = 100
    IMG_SIZE = 1024
    BATCH_SIZE = 4
    MODEL_TYPE = 'yolov8n-seg.pt' # or 'yolov8l-seg.pt' for larger models
    
    # NOTE: You must ensure data is unzipped and mounted before running this
    # Example call (Run this in your environment once data is ready)
    # train_yolov8_model(DATA_YAML, EPOCHS, IMG_SIZE, BATCH_SIZE, MODEL_TYPE)
    print("Script meant for training. Set up data paths and uncomment call to run.")