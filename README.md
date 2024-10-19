
# Breast Cancer Detection Using Convolutional Neural Networks (CNN)

This project focuses on detecting breast cancer from histopathological images using a Convolutional Neural Network (CNN) model. The dataset is from Kaggle, and the model aims to classify images of cancerous and non-cancerous tissue.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Training the Model](#training-the-model)
- [Predicting New Images](#predicting-new-images)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview
This project leverages deep learning techniques to classify breast cancer histopathological images. The images are processed and fed into a CNN model, which is trained to distinguish between cancerous and non-cancerous tissue samples.

## Dataset
The dataset used in this project is the **Breast Cancer Histopathological Image Dataset** available on Kaggle. It consists of labeled images of breast tissue samples.

- [Kaggle Dataset Link](https://www.kaggle.com/paultimothymooney/breast-histopathology-images)

The dataset contains:
- `Class 0`: Non-cancerous tissue
- `Class 1`: Cancerous tissue

Each image is labeled and the CNN is trained to classify new samples into these categories.

## Installation
Follow these steps to set up the project locally:

### Prerequisites
- Python 3.x
- TensorFlow
- Keras
- OpenCV
- Numpy
- Pandas

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/breast-cancer-detection-cnn.git
   cd breast-cancer-detection-cnn
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter notebook or the Python script:
   ```bash
   jupyter notebook BCD.ipynb
   ```
   Or execute the script:
   ```bash
   python breast_cancer_detection.py
   ```

## Training the Model
The model used in this project is a **Convolutional Neural Network (CNN)** built with Keras and TensorFlow. The network consists of convolutional layers, max-pooling layers, and fully connected layers to classify images.

Training includes:
- Loading and preprocessing images
- Data augmentation (if necessary)
- Training with a batch size of 32 and 10 epochs

Example code for model training:
```python
model.fit(train_data, validation_data=(X_val, y_val), epochs=10, batch_size=32)
model.save('my_model.keras')
```

## Predicting New Images
After training, you can use the model to predict whether a new image is cancerous or not. Ensure the image is resized and preprocessed correctly before prediction.

Example code for prediction:
```python
# Load and preprocess image
img = cv2.imread(image_path)
img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img_array = np.array(img_resized).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
img_array = img_array / 255.0  # Normalize

# Make prediction
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction, axis=1)
print(f"Predicted class: {predicted_class}")
```

## Results
The CNN achieved good classification accuracy, distinguishing between cancerous and non-cancerous images effectively. Details on model accuracy, loss, and other performance metrics can be found in the notebook.

## Contributing
Feel free to contribute by forking the repository and submitting a pull request. You can add new features, improve the model, or clean up the code.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
