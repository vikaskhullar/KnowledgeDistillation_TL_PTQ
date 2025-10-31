Optimizing Lightweight Medical AI for Chest CT Classification: A Distillation and Quantization Approach

Project Overview
This project implements knowledge distillation techniques for classifying chest CT scan images. The system uses various pre-trained teacher models (VGG16, VGG19, ResNet50, DenseNet121, EfficientNetB3, NASNetMobile, Xception) to train a compact student model, enabling efficient deployment while maintaining high accuracy.

Table of Contents
Dataset
Project Structure
Installation
Usage
Methodology
Results
Model Architectures
Configuration
Output Files
License

Dataset
The project uses chest CT scan images from Kaggle:

Dataset: Chest CT-Scan Images

Source: Kaggle Dataset, https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images

Classes: Multiple chest-related conditions

Preprocessing: Images are resized to 32x32 pixels with 3 channels

Data Preparation
The dataset should be preprocessed and saved as numpy arrays:

FData/X_train.npy, FData/y_train.npy

FData/X_test.npy, FData/y_test.npy

FData/X_val.npy, FData/y_val.npy

Project Structure
text
knowledge-distillation-ctscan/
├── src/
│   ├── KnowledgeDistillation_TL_PTQ.py                    # Main training script
├── Results/                       # Output directory
│   ├── *teachermodel.csv         # Teacher training logs
│   ├── *Distiller_history.csv    # Distillation training history
│   ├── *classification_report.csv # Classification metrics
│   ├── *confusion_matrix.png     # Visualization
│   └── *Student.h5               # Trained student models
│   └── *Plots
├── FData/                        # Preprocessed dataset
└── Documentation
Installation
Prerequisites
Python 3.7+

TensorFlow 2.x

CUDA-enabled GPU (recommended)

Dependencies
bash
pip install tensorflow==2.10.0
pip install numpy pandas matplotlib seaborn scikit-learn
pip install tensorflow-model-optimization
Or install from requirements.txt:

bash
pip install -r requirements.txt
Usage
Basic Training
Run the complete knowledge distillation pipeline:

python
python main.py
Custom Training
To train with specific teacher models:

python
# Modify the teacher_models list in the code
teacher_models = ['vgg16', 'resnet', 'efficientnet']
Hyperparameter Configuration
Modify the hyperparameters at the top of the script:

python
# Hyperparameters
TEMPERATURE = 2      # Softmax temperature for distillation
ALPHA = 0.5          # Weight for teacher vs student loss
BATCH_SIZE = 256     # Training batch size
EPOCHS = 300         # Number of training epochs
size = 32            # Input image size
channel = 3          # Number of color channels
Methodology
Knowledge Distillation Process
Teacher Training: Pre-trained models are fine-tuned on the CT scan dataset

Knowledge Transfer: Student model learns from teacher's soft labels

Loss Calculation: Combined loss using:

Student loss (hard labels)

Distillation loss (soft labels from teacher)

Custom Distiller Class
python
class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student
Loss Function
The total loss is computed as:

text
loss = ALPHA * student_loss + (1-ALPHA) * distillation_loss
Model Architectures
Student Model
Simple CNN with 3 convolutional layers

Batch normalization and max pooling

Dense layers with dropout

Output layer with softmax activation

Teacher Models
VGG16/VGG19: Deep convolutional networks

ResNet50: Residual connections

DenseNet121: Dense connectivity patterns

EfficientNetB3: Compound scaling

NASNetMobile: Neural architecture search

Xception: Depthwise separable convolutions

Configuration
Input Specifications
Image Size: 32×32 pixels

Channels: 3 (RGB)

Input Shape: (32, 32, 3)

Training Parameters
Optimizer: Adam

Loss Function: Categorical Crossentropy + KL Divergence

Metrics: Accuracy, Precision, Recall

Callbacks: CSVLogger for training history

Output Files
The training process generates multiple output files:

Training Logs
{model_name}_teachermodel.csv - Teacher training metrics

{model_name}_Distiller_history.csv - Distillation training history

Evaluation Results
{model_name}_student_classification_report.csv - Detailed classification metrics

{model_name}_student_Confusion.csv - Confusion matrix data

{model_name}_student_confusion_matrix.png - Visual confusion matrix

{model_name}_predictions.csv - Prediction probabilities

Model Files
{model_name}_Student.h5 - Trained student model

TinyEvalRes_Quant.csv - Quantized model evaluation results

Quantization Results
The project includes model quantization for deployment:

Full precision model size comparison

Quantized TensorFlow Lite model evaluation

Memory footprint analysis

Results Interpretation
Key Metrics
Accuracy: Overall classification correctness

Precision: True positives / (True positives + False positives)

Recall: True positives / (True positives + False negatives)

Model Size: Memory requirements for deployment

Performance Analysis
Compare teacher vs student model performance

Analyze trade-offs between accuracy and model size

Evaluate quantization impact on performance

License
This project is intended for research and educational purposes. Please ensure proper attribution and compliance with the dataset's original license terms.

Contributing
Contributions are welcome! Please feel free to submit pull requests or open issues for improvements and bug fixes.
