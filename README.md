# RISCVNeuralNetwork

Deploying memory-efficient ML and DL models on the VSDSquadron PRO RISC-V development board, featuring the SiFive FE310-G002 RISC-V SoC.  

---
# Overview

This project is the outcome of an intensive 10-day workshop on Edge-AI deployment on VSDSuadron Pro board containing a RISC V processor by SiFive. It contains Colab notebooks, datasets and header files that can be directly implemented on the VSD Squadron Pro board using Freedom Studio. The primary challenge of this project is being able to implement ML models on highly resource constrained embedded systems. Learning the quantization and optimization methods necessary to do so is the primary goal of this project.

---
## Project Structure
```RISCVNeuralNetwork
├── Datasets
│ ├── 50_Startups.csv
│ ├── Social_Network_Ads.csv
│ └── studentscores.csv
├── gradientdescent.ipynb
├── linear_regression_A.gif
├── mnist_training.ipynb
├── scaler.h
├── svm_model_mnist.h
├── svm_model_startup.h
├── svmtrainingset.png
└── test_images.h
```
---

## Datasets

**50_Startups.csv** – Regression dataset predicting profit based on R&D, marketing, and administration costs.

**Social_Network_Ads.csv** – Classification dataset predicting purchase decision based on user data.

**studentscores.csv** – Simple regression dataset mapping study hours to exam scores.

**MNIST dataset** (used in mnist_training.ipynb) – Digit recognition.

## Tech Stack

**Python** (NumPy, Pandas, Matplotlib, Scikit-learn, TensorFlow/PyTorch for MNIST)

**C/C++** (for RISC-V deployment, .h model files)

**Google Colab** (for model training and visualization)


