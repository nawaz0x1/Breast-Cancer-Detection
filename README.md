# Breast Cancer Detection using Convolutional Neural Networks

This project implements a Convolutional Neural Network (CNN) for detecting breast cancer in histopathological images. The model classifies images as either benign or malignant.

## Dataset

The dataset used in this project is the IDC_regular dataset, which contains histopathological images of breast cancer. 

- Original dataset: [IDC_regular_ps50_idx5.zip](http://gleason.case.edu/webdata/jpi-dl-tutorial/IDC_regular_ps50_idx5.zip)
- Citation: [PubMed](https://www.ncbi.nlm.nih.gov/pubmed/27563488) and [SPIE Digital Library](http://spie.org/Publications/Proceedings/Paper/10.1117/12.2043872)

## Requirements

- Python 3.x
- TensorFlow 2.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Model Architecture

The model is a Convolutional Neural Network with the following key features:
- Input shape: (128, 50, 50, 3)
- Multiple convolutional layers with ReLU activation
- Dropout layers for regularization
- Dense layers with softmax activation for classification

## Training

The model is trained with the following parameters:
- Batch size: 128
- Number of epochs: 15
- Optimizer: Adam
- Loss function: Sparse Categorical Crossentropy

## Performance

The model achieves the following performance on the test set:
- Accuracy: 86.79%
- F1 Score: 0.8722
- Recall: 0.8998
- Precision: 0.8463

## Usage

1. Clone the repository
2. Install the required dependencies
3. Run the Jupyter notebook to train the model and evaluate its performance

## Files

- `breast_cancer_detection.ipynb`: Main Jupyter notebook containing the code
- `model.keras`: Saved model file

## Future Work

- Experiment with different model architectures
- Try data augmentation techniques to improve performance
- Implement cross-validation for more robust evaluation

## Sample
![Breast Cancer Detection Results](https://i.postimg.cc/tgJKS50p/download.png)
