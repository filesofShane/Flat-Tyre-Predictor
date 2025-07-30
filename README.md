# Flat Tire Classification using KNN

This project classifies images of tires into one of three categories: **flat**, **full**, or **no tire** using image data and a **K-Nearest Neighbors (KNN)** model. The dataset is loaded using TensorFlow, preprocessed using `NumPy` and `sklearn`, and evaluated for performance.

---

## Dataset Structure

Images are from kaggle(https://www.kaggle.com/datasets/rhammell/full-vs-flat-tire-images) and are organized in the following folder structure:
tire-dataset/
  flat.class/
  full.class/
  no-tire.class/


Each category contains **300 images**, giving a total of **900 images**.

---

## Project Workflow

### 1. Loading Images

- Used `tensorflow.keras.preprocessing.image.load_img` and `img_to_array` to load and convert images into NumPy arrays.
- All images were resized to `(240, 240)`.

### 2. Visualizing the Data

- Sample images from each class were displayed.
- Grids were added to images to better observe structure and visual patterns.

### 3. Preprocessing

- Each category was assigned a numeric label:
  - `flat.class`: 0
  - `full.class`: 1
  - `no-tire.class`: 2
- All pixel values were normalized by dividing by `255.0`.
- Image arrays were flattened to feed into the KNN model:
  - Original shape: `(240, 240, 3)`
  - Flattened shape: `(172800,)`

### 4. Train-Test Split

- 80% training / 20% testing using `train_test_split` with stratification.
- Final training shape: `(720, 172800)`
- Final testing shape: `(180, 172800)`

### 5. Model Training

- Trained a **K-Nearest Neighbors** classifier with:
  - `n_neighbors = 7`
  - Model fitted on flattened and normalized image data.

### 6. Evaluation

Model performance based on `classification_report`:

          precision    recall  f1-score   support

       0       0.89      0.83      0.86        60
       1       0.84      0.90      0.87        60
       2       1.00      1.00      1.00        60

accuracy                           0.91       180

**Accuracy:** ~91% on test data.

---

## Libraries Used

- `tensorflow`
- `matplotlib`
- `seaborn`
- `numpy`
- `sklearn`

---

## How to Run

1. Ensure your dataset is organized in the specified folder structure.
2. Install required libraries:
   ```bash
   pip install tensorflow numpy matplotlib seaborn scikit-learn
