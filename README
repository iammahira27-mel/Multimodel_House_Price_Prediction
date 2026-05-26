# Multimodal House Price Prediction Using Images and Tabular Data

## Objective

This project aims to predict housing prices using two complementary sources of information:

- **Structured/tabular data** such as bedrooms, bathrooms, square footage, city, and street-related features
- **House images** that provide visual context about the property

The objective is to build a **multimodal regression model** that learns from both data types and evaluates performance using **MAE** and **RMSE**.

## Methodology / Approach

### 1. Data preparation
- Loaded the housing dataset from CSV.
- Matched each house record with its corresponding image using `image_id`.
- Cleaned the street field by removing the house number and keeping only the street name.
- Added a `street_freq` feature using frequency encoding to keep street information compact.
- Applied **one-hot encoding** to the city column.
- Standardized numeric features using `StandardScaler`.
- Transformed the target variable using `log1p(price)` to reduce skew and improve regression stability.

### 2. Image preprocessing
- Resized all house images to `224 x 224`.
- Applied ImageNet-style normalization.
- Used light augmentation on the training set, including:
  - horizontal flip
  - slight rotation
  - mild color jitter

### 3. Multimodal model design
The final architecture uses three parts:

- **Image branch:** pretrained **ResNet18** used as a feature extractor
- **Tabular branch:** a compact MLP for structured features
- **Fusion head:** concatenates image and tabular embeddings and predicts the final price

### 4. Training strategy
To improve stability and reduce overfitting, training was done in two stages:

- **Stage 1:** froze the ResNet18 backbone and trained only the tabular branch and fusion head
- **Stage 2:** unfroze only the last ResNet block (`layer4`) and fine-tuned with a much smaller learning rate

Additional techniques used:
- `SmoothL1Loss` for robust regression training
- `AdamW` optimizer with weight decay
- `ReduceLROnPlateau` scheduler
- gradient clipping
- early stopping behavior through controlled stage-wise training

### 5. Evaluation
- Predictions were converted back from log scale to the original price scale using `expm1`.
- Final performance was measured with:
  - **MAE**
  - **RMSE**
- Scatter plots and residual plots were used to inspect prediction quality.

## Key Results / Observations

- By using **pretrained ResNet18 backbone** made the image branch efficient and reliable.
- The compact fusion head reduced model complexity.
- Log-transforming the target improved learning stability for a skewed price distribution.
- The final pipeline was better suited for multimodal learning and more appropriate for the available dataset size.

## Model Architecture

### Image Branch
- **ResNet18 pretrained on ImageNet**
- Final classification layer removed
- Output: **512-dimensional image embedding**

### Tabular Branch
- Dense layer: `input_dim -> 128`
- Batch normalization
- ReLU activation
- Dropout
- Dense layer: `128 -> 64`
- Output: **64-dimensional tabular embedding**

### Fusion Head
- Concatenates image and tabular embeddings
- Dense layer: `576 -> 128`
- Dense layer: `128 -> 64`
- Final regression output: `64 -> 1`

## Notes

This project demonstrates a practical multimodal learning workflow for house price prediction and highlights the importance of:
- proper preprocessing
- controlled fine-tuning
- compact model design
- evaluation on the original target scale
