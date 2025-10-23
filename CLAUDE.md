# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a machine learning/AI project containing multiple Jupyter notebooks exploring different deep learning and ML techniques using TensorFlow/Keras and scikit-learn. The project focuses on food image classification (Food-101 dataset), digit classification (MNIST), heart disease prediction, and student performance prediction.

## Project Structure

### Main Notebooks

1. **nutri-v4-transfer-learning.ipynb** - Food-101 Transfer Learning (15 Classes - Recommended)
   - **Strategy:** EfficientNet-B4 pretraining + 3-phase fine-tuning
   - **Architecture:** EfficientNet-B4 (19M params, ImageNet) + custom classifier (512→256→15)
   - **Data Augmentation:** RandAugment (N=2, magnitude=9) - simpler than v3
   - **Fine-tuning phases:**
     - Phase 1: Train classifier only, freeze backbone (2 epochs)
     - Phase 2: Unfreeze last 50 layers, fine-tune (25-30 epochs)
     - Phase 3: Complete fine-tuning with low LR (20 epochs)
   - **Training:** Batch size 32, Cosine annealing, ReduceLROnPlateau
   - **Expected:** 88-92% accuracy on 15 classes
   - Best model checkpoint: `best_model_v4_phase3.keras`
   - **Key improvements over v3:** +20-25 points of accuracy through transfer learning

2. **nutri-v3-15clases.ipynb** - Food-101 Classification (15 Classes Experiment - From Scratch)
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

### Food-101 Model v4 (nutri-v4-transfer-learning) - Transfer Learning for 15 Classes **[RECOMMENDED]**
- **Input**: 380×380 RGB images (optimized for EfficientNet-B4)
- **Base Model**: EfficientNet-B4 (ImageNet pretraining, 19M params)
- **Custom Head**: GlobalAveragePooling2D → Dense(512) → Dropout(0.3) → Dense(256) → Dropout(0.2) → Dense(15)
- **Data Augmentation**: RandAugment (N=2, magnitude=9) - Flip + Rotation + Zoom + Contrast + Brightness
- **3-Phase Fine-tuning:**
  - **Phase 1** (2 epochs): Freeze EfficientNet, train classifier only, LR=0.001
  - **Phase 2** (25-30 epochs): Unfreeze last 50 layers, fine-tune, LR=0.0001, Cosine annealing + ReduceLROnPlateau
  - **Phase 3** (20 epochs): Unfreeze all, complete fine-tuning, LR=0.00001, Cosine annealing
- **Batch Size**: 32
- **Loss**: SparseCategoricalCrossentropy (faster than one-hot)
- **Expected**: 88-92% accuracy on 15 classes from ImageNet transfer learning
- **Key Advantage**: Leverages 1000-class ImageNet features, reducing overfitting dramatically

### Food-101 Model v3 (nutri-v3-15clases) - For 15 Classes (From Scratch)
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

- **best_model_v4_phase3.keras** - Transfer Learning model with 15 classes (v4) **[BEST]**
  - Expected accuracy: 88-92%
  - Architecture: EfficientNet-B4 + custom 2-layer classifier
  - Load with: `tf.keras.models.load_model('best_model_v4_phase3.keras')`

- **best_model_v4_phase2.keras** - Intermediate checkpoint from Phase 2
  - Useful for comparison and analysis

- **best_model_v4_phase1.keras** - Initial warm-up checkpoint
  - Low accuracy, mainly for reference

- **best_model_v3_15classes.keras** - From-scratch model with 15 classes (v3)
  - Expected accuracy: 50-65%
  - Architecture: 5 conv blocks + 2-layer classifier
  - Load with: `tf.keras.models.load_model('best_model_v3_15classes.keras')`

- **best_model_v2.3.keras** - Food classification model with 10 classes (v2.1 baseline)
  - Expected accuracy: 60-75%
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
- **v2.1 (10 classes)**: Baseline from-scratch model, achieved 60-75% accuracy
- **v3 (15 classes)**: Experimental scaling test with deeper architecture and aggressive augmentation
  - Demonstrated how model scales to more classes with increased difficulty
  - Identified visually similar food classes that confuse the model
  - Achieved 50-65% accuracy (limited by from-scratch approach)
- **v4 (15 classes)**: Transfer Learning approach with EfficientNet-B4
  - Leverages ImageNet pretraining (1000 classes, 1.3M images)
  - Expected 88-92% accuracy (major improvement over v3)
  - 3-phase progressive fine-tuning for optimal feature adaptation
  - **RECOMMENDED approach** for food classification

### Next Steps for Food Classification
1. ✅ **v4 Complete**: Transfer learning with EfficientNet-B4 achieves expected 88-92%
2. Apply v4 techniques to v2.1 (10 classes might reach 92-95%)
3. Try ensemble: Average predictions from v4 (15 classes) + v2.1 (10 classes)
4. Explore knowledge distillation: Compress v4 to smaller EfficientNet-B2 for faster inference
5. Fine-tune on full Food-101 (101 classes) with transfer learning
6. Consider multi-task learning: Predict food category + cuisine type + ingredients

### Other Projects
- Heart disease and student prediction models underperform and may need architecture changes
- Confusion matrices and classification reports are standard evaluation outputs
