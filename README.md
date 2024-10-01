# Breast Cancer Detection Using Deep Learning

A deep learning project for detecting invasive ductal carcinoma (IDC) in breast histopathology images using Convolutional Neural Networks (CNNs). The dataset is sourced from Kaggle's IDC breast histopathology dataset, and the project is implemented in Python using TensorFlow, Keras, and OpenCV.

## Introduction
Breast cancer is one of the most common types of cancer affecting women globally. Early detection is crucial for effective treatment. This project uses deep learning to classify histopathology images into two categories:
- **0**: Non-IDC (Benign)
- **1**: IDC (Malignant)

A Convolutional Neural Network (CNN) is trained to differentiate between these two categories by analyzing the histopathology images.

## Dataset
The dataset used in this project is the **[Breast Histopathology Images Dataset](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images)** available on Kaggle. It contains labeled images of breast cancer histopathology.

To use this dataset, download it from Kaggle using the Kaggle API:

```bash
kaggle datasets download -d paultimothymooney/breast-histopathology-images
```

Once downloaded, organize the dataset as follows:

```bash
IDC_regular_ps50_idx5/
├── 10253
│   ├── 0
│   ├── 1
├── 10254
│   ├── 0
│   ├── 1
├── ...
```

## Model Architecture
The model is a Convolutional Neural Network (CNN) consisting of multiple convolutional and pooling layers. The architecture is designed to extract features from the images and classify them as IDC or non-IDC.

Here is a summary of the architecture:

- Convolutional Layer (3x3, 32 filters)
- MaxPooling Layer (2x2)
- Convolutional Layer (3x3, 64 filters)
- MaxPooling Layer (2x2)
- Fully Connected Dense Layer (128 units, ReLU)
- Output Layer (2 units, Softmax)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/tejaspavanb/Breast-Cancer-Detection.git
   cd Breast-Cancer-Detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset from Kaggle using the Kaggle API and place it in the project directory.

## Usage
1. Run the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Load the `breast_cancer_detection.ipynb` notebook and execute the cells to train the model.

3. The model can be trained and evaluated using the preprocessed data. You can adjust hyperparameters like `epochs` and `batch_size` in the notebook.

4. After training, you can make predictions on new histopathology images using the following code snippet:

   ```python
   img = cv2.imread('path_to_new_image.jpg')
   img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
   img_array = np.array(img_resized).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
   img_array = img_array / 255.0  # Normalize pixel values
   prediction = model.predict(img_array)
   print("Prediction:", np.argmax(prediction))
   ```

## Results
- **Training Accuracy**: xx%
- **Validation Accuracy**: xx%
- **Test Accuracy**: xx%

Graphs for training and validation accuracy/loss can be found in the notebook.

## Contributing
Feel free to contribute to this project by submitting a pull request or reporting issues.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


