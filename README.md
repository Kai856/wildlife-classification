# Animal Classification with ResNet152V2

Deep learning model for classifying 22 different animal species using transfer learning with ResNet152V2.

## Model Architecture
- **Base Model**: ResNet152V2 (pre-trained on ImageNet)
- **Custom Layers**: Global Average Pooling + Dense output layer
- **Total Parameters**: 174.8M (222.1M trainable)
- **Input Size**: 224x224 RGB images

## Dataset
Uses the UCF Flying Wildlife Classification Through Detection (FWCTD) dataset with 104,495 images across 22 animal classes.

**Data Split:**
- Training: 70% (73,147 images)
- Validation: 20% (20,899 images)
- Testing: 10% (10,449 images)

## Performance
- **Test Accuracy**: 96.02%
- **Test Loss**: 0.1575
- Training completed in 10 epochs

## Requirements
```
tensorflow
numpy
opencv-python
matplotlib
```

## Training Configuration
- **Optimizer**: Adam (learning rate: 0.001)
- **Loss Function**: Categorical Crossentropy
- **Batch Size**: 64
- **Epochs**: 10

## Usage
```python
from tensorflow import keras

# Load the trained model
model = keras.models.load_model('animal_classification.keras')

# Make predictions
predictions = model.predict(your_image_batch)
```

## Results
The model achieves high accuracy with consistent performance across training, validation, and test sets, demonstrating effective transfer learning from ImageNet features.
