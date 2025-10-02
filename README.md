# Solar Rooftop Detection and GIS Vectorization (YOLOv8 Segmentation)

## Project Focus

This project focuses on automated rooftop detection from satellite imagery using state-of-the-art deep learning models, with a comparative study of augmentation frameworks (Ultralytics vs. Roboflow). The goal is to build a robust, end-to-end pipeline for solar rooftop mapping, useful for renewable energy planning, urban studies, and geospatial analysis.

***

## Key Features

- **Advanced Preprocessing:** Handled high-resolution imagery with **tiling + overlap** to effectively manage memory and prevent object cut-offs at tile boundaries.
- **Data Pipeline:** Integrated **Roboflow** for data annotation and robust augmentation generation.
- **Model Training:** Trained and customized **YOLOv8 segmentation models** using the native Ultralytics pipeline in Google Colab.
- **Post-Processing:** Implemented sophisticated logic for mask stitching, morphological filtering, and refinement of predicted rooftop boundaries.
- **GIS Integration:** Achieved full **georeferencing** of outputs and final export to **GeoJSON / Shapefile** for seamless use in GIS applications.
- **Evaluation:** Performed quantitative evaluation using **IoU** comparison against ground truth shapefiles.

***

## Detailed Project Workflow

### Data Acquisition
- Downloaded Marehra, Uttar Pradesh satellite imagery (`.tif`) using QGIS QuickMapServices.

### Preprocessing
- Split large images into 2048×2048 tiles with 512px overlap using a Python script.

### Annotation & Augmentation
- Annotated rooftops in Roboflow.
- Applied augmentations (flip, rotate, stretch, etc.).

### Model Training
- Trained **YOLOv8n segmentation model** using Ultralytics.
- Compared results with Roboflow-trained YOLOv8 models.

### Post-Processing
- Predicted masks stitched back into original size.
- Applied morphological filtering to refine rooftop edges.

### GIS Integration
- Georeferenced stitched outputs with **Rasterio**.
- Converted to **Shapefile** for rooftop mapping in QGIS.

### Evaluation
- Model performance was measured using Intersection over Union (IoU) between predicted rooftop masks and manually annotated ground truth shapefiles.
- Predicted shapefiles were imported into QGIS, and IoU was computed using the Vector Analysis → Intersection tool.
- The best IoU achieved was 91.5% with Ultralytics YOLOv8n model.

***

## Tech Stack

| Domain | Key Tools & Libraries |
| :--- | :--- |
| **Deep Learning** | **YOLOv8 (Ultralytics)**, PyTorch |
| **Preprocessing** | Python, NumPy, OpenCV |
| **Annotation & Augmentation** | Roboflow |
| **Geospatial Processing** | **QGIS**, **Rasterio**, **Shapely**, GeoPandas |
| **Visualization** | Matplotlib, GeoPandas |

***

