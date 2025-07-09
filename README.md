# solid-waste-segregation-model

Waste Classification Using CNN

A Convolutional Neural Network (CNN)-based image classification model that classifies waste into five categories to enhance the accuracy of existing manual waste segregation methods in low-infrastructure environments. This project is designed to support scalable deployment and potential collaborations with NGOs or local bodies working in waste management.

Project Overview

Manual waste sorting in India is often inaccurate and limited due to a shortage of Material Recovery Facilities (MRFs). This model addresses that by:

- Classifying waste into 5 categories using images
- Achieving over 80% validation accuracy
- Leveraging TensorFlow/Keras for efficient model design and training
- Supporting future integration with on-ground NGOs for deployment

Model Architecture

- Input: 256 × 256 × 3 RGB images
- Preprocessing: Rescaling, batching, and dataset splitting using `image_dataset_from_directory`
- CNN Layers:
  - Conv2D → ReLU → MaxPooling (×3)
  - Flatten → Dense (128) → Dense (5, softmax)
- Loss Function: `sparse_categorical_crossentropy`
- Optimizer: `Adam`
- Evaluation Metrics: `Accuracy`

Directory Structure

project/
├── data/
│   ├── dataset-resized/
│   ├── one-indexed-files-notrash_train.txt
│   └── one-indexed-files-notrash_val.txt
├── train_waste_model.py
└── waste_classification_model.h5

Results

- Training Accuracy: ~95%
- Validation Accuracy: 90%+
- Trained over 15 epochs with real-time performance visualization (accuracy & loss curves)

How to Run
1. Install Dependencies

    pip install tensorflow matplotlib

2. Set Your Dataset Path
   Update the `dataset_dir` path in `train_waste_model.py` to match your local directory structure.

3. Train the Model

    python train_waste_model.py

4. Saved Model
   After training, the model is saved as `waste_classification_model.h5` for future deployment or inference.

Impact & Collaboration

This model can support NGO-led waste segregation efforts by:

- Reducing dependency on manual handpicking
- Increasing sorting accuracy in under-resourced areas
- Offering a scalable and lightweight solution for smart waste systems

Author
Shashwat Vinayak Sinha

