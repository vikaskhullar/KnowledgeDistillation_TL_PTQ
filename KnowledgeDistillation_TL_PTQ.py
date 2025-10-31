# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 12:59:12 2025

@author: vikas
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, applications
from tensorflow.keras.datasets import cifar10
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from keras.callbacks import CSVLogger
import tempfile
from sklearn.metrics import precision_score, recall_score, accuracy_score
import os
import tensorflow_model_optimization as tfmot


# Hyperparameters
TEMPERATURE = 2  # Softmax temperature for distillation
ALPHA = 0.5     # Weight for teacher loss (1-ALPHA for student loss)
BATCH_SIZE = 256
EPOCHS = 300
size=32
channel = 3
INPUT_SHAPE = (size, size, channel)

# Load and preprocess CIFAR-10 dataset

(x_train, y_train), (x_test, y_test), (x_val, y_val) = (np.load("FData/X_train.npy"),np.load("FData/y_train.npy")),(np.load("FData/X_test.npy"),np.load("FData/y_test.npy")),(np.load("FData/X_val.npy"),np.load("FData/y_val.npy"))
NUM_CLASSES = y_train.shape[1]
x_train.shape

# Function to create a simple student model
def create_student_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    
    # Simple CNN architecture
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs, outputs)

# Function to create teacher model based on architecture name
def create_teacher_model(model_name, input_shape, num_classes):
    # Resize input for models that expect larger input sizes
    if model_name == 'vgg16':
        base_model = applications.VGG16(weights='imagenet', include_top=False, input_tensor=x)
    elif model_name == 'vgg19':
        base_model = applications.VGG19(weights='imagenet', include_top=False, input_tensor=x)
    elif model_name == 'resnet':
        base_model = applications.ResNet50(weights='imagenet', include_top=False, input_tensor=x)
    elif model_name == 'densenet':
        base_model = applications.DenseNet121(weights='imagenet', include_top=False, input_tensor=x)
    elif model_name == 'efficientnet':
        base_model = applications.EfficientNetB3(weights='imagenet', include_top=False, input_tensor=x)
    elif model_name == 'nasnetmobile':
        base_model = applications.NASNetMobile(weights='imagenet', include_top=False, input_tensor=x)
    elif model_name == 'xception':
        base_model = applications.Xception(weights='imagenet', include_top=False, input_tensor=x)
    else:
        raise ValueError("Unknown model name")
    
    # Add custom top layers
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(base_model.input, outputs)
    
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    return model

# Custom distillation loss function
class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student

    def compile(self, optimizer, metrics, student_loss_fn, distillation_loss_fn):
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn

    def train_step(self, data):
        x, y = data
        
        # Forward pass of teacher (inference mode)
        teacher_predictions = self.teacher(x, training=False)
        
        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)
            
            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)
            
            # Compute distillation loss with temperature
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / TEMPERATURE, axis=1),
                tf.nn.softmax(student_predictions / TEMPERATURE, axis=1)
            )
            
            loss = 0.1
        
        # Compute gradients and update student weights
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update the metrics
        self.compiled_metrics.update_state(y, student_predictions)
        
        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({
            "student_loss": student_loss,
            "distillation_loss": distillation_loss,
        })
        return results

    def test_step(self, data):
        x, y = data
                
        # Calculate the loss
        student_loss = self.student_loss_fn(y, student_predictions)
        
        # Update the metrics
        self.compiled_metrics.update_state(y, student_predictions)
        
        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results

# Function to train and evaluate knowledge distillation
def train_knowledge_distillation(teacher_model_name):
    print(f"\nTraining with {teacher_model_name} as teacher")
    
    # Create teacher and student models
    teacher = create_teacher_model(teacher_model_name, INPUT_SHAPE, NUM_CLASSES)
    student = create_student_model(INPUT_SHAPE, NUM_CLASSES)
    
    # Compile teacher (only top layers are trainable)
    teacher.compile(
        optimizer=optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'Precision', 'Recall']
    )
    csv_logger = CSVLogger(f'Results//{teacher_model_name}_teachermodel.csv')
    # Train teacher on the data (only top layers)
    print("Training teacher model...")
    teacher.fit(
        x_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_test, y_test),
        callbacks=[csv_logger]
    )
    
    teacher.evaluate(x_val, y_val, verbose=2)
    y_predT = student.predict(x_val)
    
    # Initialize and compile distiller
    csv_logger = CSVLogger(f'Results//{teacher_model_name}_teachermodel.csv')
    
    
    #h5_file = tf.keras.models.save_model(teacher, f'Results//{model_name}_Teacher.h5', save_format='h5')
    '''
    #teacher.save(f'Results//{model_name}_Teacher.h5')
    teachermemory = os.path.getsize(f'Results//{model_name}_Teacher.h5')*1024/ float(2**20)
    os.remove(f'Results//{model_name}_Teacher.h5')
    '''
    teachermemory = 0
    
    distiller = Distiller(student=student, teacher=teacher)
    distiller.compile(
        optimizer=optimizers.Adam(),
        metrics=['accuracy', 'Precision', 'Recall'],
        student_loss_fn=keras.losses.CategoricalCrossentropy(),
        distillation_loss_fn=keras.losses.KLDivergence(),
    )
    
    # Distill teacher to student
    print("Distilling knowledge to student model...")
    #csv_logger = CSVLogger(f'Results//{teacher_model_name}_Distillermodel.csv')
    history = distiller.fit(
        x_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_test, y_test),
        #callbacks=[csv_logger]
    )
    
        
    # Evaluate student standalone
    print("Evaluating student model...")
    student.compile(
        optimizer=optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'Precision', 'Recall']
    )
    
    student.evaluate(x_val, y_val, verbose=2)
    y_pred = student.predict(x_val)
    return student, history, y_pred, y_predT, teachermemory


def evaluate_model(interpreter,test_images,test_labels):
    test_images = test_images.reshape(test_images.shape[0],test_images.shape[1],
                                      test_images.shape[2],3)
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    
    # Run predictions on every image in the "test" dataset.
    prediction_digits = []
    test_labels = np.argmax(test_labels, axis=1)
    
    for i, test_image in enumerate(test_images):
        test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
        interpreter.set_tensor(input_index, test_image)
        interpreter.invoke()
        output = interpreter.tensor(output_index)
        digit = np.argmax(output()[0])
        prediction_digits.append(digit)

    print('\n')
    # Compare prediction results with ground truth labels to calculate accuracy.
    prediction_digits = np.array(prediction_digits)
    accuracy = accuracy_score(test_labels, prediction_digits)
    precision = precision_score(test_labels, prediction_digits, average='micro')
    recall = recall_score(test_labels, prediction_digits, average='micro')
    cm = confusion_matrix(test_labels, prediction_digits)
    return accuracy, precision, recall, cm


# Train with different teacher models
teacher_models = ['vgg16', 'vgg19', 'resnet', 'densenet', 'efficientnet',
                  'nasnetmobile','xception']
for model_name in teacher_models:
    #try:
        student_model, history, y_pred, y_predT, teachermemory = train_knowledge_distillation(model_name)
        # Save the student model
        # 4. Save Training History to CSV
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(f'Results//{model_name}_Distiller_history.csv', index=False)
        # 6. Evaluate and Save Final Metrics
        
        # Classification report
        y_pred_classes = np.argmax(y_pred, axis=1)
        report = classification_report(np.argmax(y_val, axis=1), y_pred_classes, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(f'Results//{model_name}_student_classification_report.csv')
        
        # Confusion matrix
        cm = confusion_matrix(np.argmax(y_val, axis=1), y_pred_classes)
        pd.DataFrame(cm).to_csv(f"Results//{model_name}_student_Confusion.csv")

        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(f'Results//{model_name}_student_confusion_matrix.png')
        
        # 7. Save Predictions
        predictions_df = pd.DataFrame({
            'True_Labels': np.argmax(y_val, axis=1),
            'Predicted_Labels': y_pred_classes,
            'Probability_Class_0': y_pred[:, 0],
            'Probability_Class_1': y_pred[:, 1]
        })
        predictions_df.to_csv(f'Results//{model_name}_predictions.csv', index=False)
        
        print("All results saved successfully!")
                        
        converter = tf.lite.TFLiteConverter.from_keras_model(student_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        quantized_tflite_model = converter.convert()
        interpreter = tf.lite.Interpreter(model_content=quantized_tflite_model)
        interpreter.allocate_tensors()
        
        with open(quant_file, 'wb') as f:
            f.write(quantized_tflite_model)
        quantizedmemory = os.path.getsize(quant_file)*1024/ float(2**20)
        
        accuracy, precision, recall, cm = evaluate_model(interpreter,x_val,y_val)    
        f2 = open("Results//TinyEvalRes_Quant.csv", "a+")
        f2.write(model_name+","+str(accuracy)+","+str(recall)+","+str(precision)+","+str(teachermemory)+","+str(studentmemory)+","+str(quantizedmemory)+"\n")
        f2.close()
        dfNor3 = pd.DataFrame(cm)
        dfNor3.to_csv(f"Results//{model_name}_ConfusionTiny.csv",mode='a')


    
    #except Exception as e:
        #print(f"Error with {model_name}: {str(e)}")
    