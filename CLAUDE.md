# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a machine learning/AI project containing multiple Jupyter notebooks exploring different deep learning and ML techniques using TensorFlow/Keras and scikit-learn. The project focuses on food image classification (Food-101 dataset), digit classification (MNIST), heart disease prediction, and student performance prediction.

## Project Structure

### Main Notebooks

1. **nutri-v3-15clases.ipynb** - Food-101 Classification (15 Classes Experiment)
   - Deep CNN model for food image classification with 15 classes (vs 10 in v2.1)
   - Tests model capacity with more classes and aggressive augmentation
   - **Architecture**: 5 convolutional blocks (32→64→128→256→512) + 2-layer classifier (512→256)
   - **Regularization**: Progressive dropout (0.25→0.35→0.4), L2 regularization 0.0003
   - **Data Augmentation**: Agresive (Flip + Rotation + Zoom + Contrast + Brightness) + Mixup/CutMix (60%)
   - **Training**: Batch size 32, LR 0.0003, 100 epochs, patience 25, ReduceLROnPlateau
   - **Expected**: 50-65% accuracy on 15 classes (from scratch, no transfer learning)
   - Best model checkpoint: `best_model_v3_15classes.keras`
   - Includes detailed analysis: top-10 class confusions, per-class accuracy, confidence analysis

2. **nutri-v2.1.ipynb** - Food-101 Classification (10 Classes Baseline)
   - Deep CNN model for food image classification
   - Currently working on achieving 80% accuracy on 10 food classes
   - Uses TensorFlow/Keras with advanced techniques: Mixup, CutMix, BatchNorm, LR scheduling
   - Best model checkpoint: `best_model_v2.3.keras`

3. **mnist_v2.ipynb** - MNIST Digit Recognition
   - CNN classifier for 28x28 handwritten digits
   - Achieves ~98.85% test accuracy (well-tuned)
   - Generates confusion matrices for evaluation

4. **RN-Corazon.ipynb** - Heart Disease Prediction
   - Feed-forward neural network for binary classification (healthy/disease)
   - Trained on medical feature data (11 features)
   - Currently achieving ~48% test accuracy (needs improvement)
   - Data split between `heartE.csv` (training) and `heartV.csv` (validation)

5. **m_soporte_vec.ipynb** - Student Performance Prediction
   - Support Vector Machine (SVM) classifier
   - Predicts student academic performance (binary: high/low)
   - Uses `student25Todo_2.csv` dataset
   - Currently achieving ~64% accuracy (needs tuning)

### Other Notebooks
- **nutri.ipynb, nutri-v2.ipynb** - Earlier iterations of food classification
- **regresion_lineal.ipynb, regresion_polinomial.ipynb** - Regression examples
- **mnist.ipynb** - Earlier MNIST iteration

## Key Technologies & Dependencies

- **TensorFlow/Keras** - Deep learning framework
- **NumPy, Pandas** - Data manipulation
- **scikit-learn** - ML algorithms and metrics (train_test_split, SVM, confusion_matrix, classification_report)
- **Matplotlib** - Data visualization
- **tensorflow_datasets** - Pre-built datasets (Food-101, MNIST)

## Common Commands

### Running Notebooks
```bash
jupyter notebook                    # Launch Jupyter interface
jupyter notebook nutri-v2.1.ipynb   # Run specific notebook
```

### Data & Environment
The project expects CSV files in the root directory:
- `heartE.csv` - Heart disease training data
- `heartV.csv` - Heart disease validation data
- `student25Todo_2.csv` - Student performance data

## Important Architecture Notes

### Food-101 Model v3 (nutri-v3-15clases) - For 15 Classes
- **Input**: 224×224 RGB images
- **Architecture**: 5 convolutional blocks (32→64→128→256→512 filters) + 2-layer classifier (512→256→output)
- **Regularization Strategy**: Progressive dropout (0.25→0.3→0.35→0.4) + L2 regularization 0.0003
- **Data Augmentation**: Aggressive (Flip + Rotation±15° + Zoom±15% + Contrast±20% + Brightness±20%) + Mixup (30%) + CutMix (30%) + Normal (40%)
- **Training**: Adam optimizer (LR 0.0003), CategoricalCrossentropy loss
- **LR Scheduling**: Warmup (epochs 1-8: 0→0.0003) + Cosine decay (epochs 9-100: 0.0003→0.000005) + ReduceLROnPlateau
- **Batch Size**: 32 (reduced for noisier gradients)
- **Early Stopping**: patience=25 on validation loss (more patient for 15 classes)
- **Expected**: 50-65% accuracy on 15 food classes from scratch

### Food-101 Model v2.1 (nutri-v2.1) - For 10 Classes
- **Input**: 224×224 RGB images
- **Architecture**: 4 convolutional blocks (32→64→128→256 filters) with BatchNorm
- **Regularization Strategy**: Dropout applied selectively (0.2 in conv blocks, 0.3 in dense layers)
- **Data Augmentation**: RandomFlip + Mixup (25%) + CutMix (25%) + Normal (50%)
- **Training**: Adam optimizer (LR 0.0005), CategoricalCrossentropy loss
- **LR Scheduling**: Warmup (epochs 1-5: 0→0.0005) + Cosine decay (epochs 6-60: 0.0005→0.00001)
- **Batch Size**: 64
- **Early Stopping**: patience=15 on validation loss
- **Target**: 80% accuracy on 10 food classes

### Critical Design Decisions
1. **Mixed precision disabled** - Causes numerical instability for this architecture
2. **Label smoothing removed** - Mixup/CutMix already smooth labels naturally
3. **Aggressive regularization tuned** - Dropout reduced to balance learning vs overfitting
4. **Learning rate carefully calibrated** - 0.0005 prevents training collapse seen in v2.2
5. **Batch size 64** - Balances memory and gradient quality for 224×224 images

### Debugging Tips for Food Model
- Check train/val gap in `history.history` to diagnose overfitting
- Confusion matrix shows which food classes are most confused
- If accuracy plateaus early: reduce dropout, increase LR, or add more augmentation
- If model collapses: verify mixed precision is disabled, check learning rate schedule

## Model Checkpoints

- **best_model_v3_15classes.keras** - Food classification model with 15 classes (v3)
  - Architecture: 5 conv blocks + 2-layer classifier
  - Load with: `tf.keras.models.load_model('best_model_v3_15classes.keras')`

- **best_model_v2.3.keras** - Food classification model with 10 classes (v2.1 baseline)
  - Architecture: 4 conv blocks + 1-layer classifier
  - Load with: `tf.keras.models.load_model('best_model_v2.3.keras')`

## Git Info

Recent commits show progression of model iterations:
- 8a56825: good acc (best performance)
- 8760643: pre-modelo (pre-model)
- 61a9992: nutri-v1
- abf6602: mnist_v2
- dfea3e4: SVM

## Development Notes

- Notebooks are educational/exploratory - individual files, not a unified application
- Each notebook can run independently with proper CSV files in place

### Food Classification Progression
- **v2.1 (10 classes)**: Baseline model, goal 80% accuracy
- **v3 (15 classes)**: Experimental scaling test with deeper architecture and aggressive augmentation
  - Demonstrates how model scales to more classes
  - Identifies visually similar food classes that confuse the model
  - Uses more conservative training (smaller batch size, lower LR, more patience)
  - Includes detailed per-class analysis to guide future improvements

### Next Steps for Food Classification
1. Try transfer learning (ResNet, EfficientNet) on v3 to see improvement potential
2. Apply successful v3 techniques to v2.1 (15 classes with transfer learning)
3. Analyze top-confused class pairs and consider collecting more data for those categories
4. Test ensemble approaches combining v2.1 and v3 models

### Other Projects
- Heart disease and student prediction models underperform and may need architecture changes
- Confusion matrices and classification reports are standard evaluation outputs
