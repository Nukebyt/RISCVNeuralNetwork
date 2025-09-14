# RISCVNeuralNetwork

Deploying memory-efficient ML and DL models on the VSDSquadron PRO RISC-V development board, featuring the SiFive FE310-G002 RISC-V SoC.  

---
# Overview

This project is the outcome of an intensive 10-day workshop on Edge-AI deployment on VSDSuadron Pro board containing a RISC V processor by SiFive. It contains Colab notebooks, datasets and header files that can be directly implemented on the VSD Squadron Pro board using Freedom Studio. The primary challenge of this project is being able to implement ML models on highly resource constrained embedded systems. Learning the quantization and optimization methods necessary to do so is the primary goal of this project.

---

## About VSDSquadronPro board

<img width="740" height="493" alt="image" src="https://github.com/user-attachments/assets/99b30f52-45f1-42b5-8b67-1a4e0d775cf8" />

<img width="647" height="365" alt="image" src="https://github.com/user-attachments/assets/2b952403-1e98-496d-b098-d56b5b61d2f7" />

The VSDSquadron PRO RISC-V development boards features a RISC-V SoC with the following capabilities:

1. 48-lead 6x6 QFN package

2. On-board 16MHz crystal

3. 19 Digital IO pins and 9 PWM pins

4. 2 UART and 1 I2C

5. Dedicated quad-SPI (QSPI) flash interface

6. 32 Mbit Off-Chip (ISSI SPI Flash)
   
7. USB-C type for Program, Debug, and Serial Communication

---

## About SiFive FE310-G002 RISC-V SoC

The FE310-G002 is the second revision of the General Purpose Freedom E300 family.
The FE310-G002 is built around the **SiFive E31 Core Complex** instantiated in the Freedom E300 plaform and fabricated in the TSMC CL018G 180nm process. 
The E31 core has a(n) 2-way set-associative **16 KiB L1 instruction cache and a(n) 16 KiB L1 DTIM.**
<img width="913" height="857" alt="image" src="https://github.com/user-attachments/assets/f7aac486-40f7-4f6d-b640-9f42559d8b92" />

---

## Challenges of Implementing ML on FE310

The FE310-G002’s extremely limited SRAM (16 KB) and lack of DSP/vector hardware make storing and running even modest ML models (like MNIST CNNs) very difficult without extreme quantization and external memory streaming.

Even a tiny MNIST neural net requires hundreds of KBs to MBs of weight storage. However FE310-G002 has only 16 KB instruction cache + 16 KB data SRAM.
We would need to store model weights in external QSPI Flash and stream them, which is slow and introduces latency. Larger models (e.g., CNNs like LeNet) will simply not fit in RAM. 

Moreover, the FE310 has a 32-bit scalar core (RV32IMAC) but no DSP, SIMD, or vector instructions, and no floating-point unit.
Neural nets need lots of multiply-accumulate (MAC) operations. On FE310, every multiply/add runs sequentially in software, which is much slower than on MCUs with DSP extensions (e.g., ARM Cortex-M4/M7).
In this project, we aim to overcome these challenges by developing a heavily quantized model (<20kb) that successfully runs on the SoC.

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
├── knnandsvmheader.ipynb
└── test_images.h
```
---
## Workshop Flow

We start by a hands-on demonstration of linear regression with gradient descent.
In the notebook [gradientdescent](gradientdescent.ipynb), I demonstrate the implementation of Linear Regression using Gradient Descent from scratch in Python. It uses a dataset (studentscores.csv) that records the number of study hours and corresponding exam scores, with the objective of predicting scores based on hours studied. The relationship between the two variables is first visualized through scatter plots, highlighting the linear correlation. **A custom Model class is then built to perform regression, featuring methods for training (fit), prediction (predict), and parameter updates (update_weights) using gradient descent optimization**. During training, the slope and intercept are iteratively adjusted according to the specified learning rate and number of iterations, and the resulting regression line is compared with the actual data points. **The model outputs both the predicted scores and the learned parameters, and the results are displayed graphically by overlaying the fitted line on the scatter plot.** Additionally, the notebook extends the work with another LinearRegression class that uses animation tools to visualize how gradient descent gradually converges to the best-fit line.(linear_regression_A.gif).

Next, in the notebook [knnandsvmheader.ipynb](knnandsvmheader.ipynb) we focus on implementing and comparing machine learning models, specifically **K-Nearest Neighbors (KNN) and Support Vector Machines (SVM)**, for classification tasks. The models are trained using both linear and radial basis function (RBF) kernels to analyze performance differences. 

We see that **for Edge AI applications, SVM is superior to KNN because it requires less memory, has faster inference, and provides better generalization**. 

Once trained, an SVM only needs to store the support vectors (a subset of the training data) and the learned weights. This usually results in a compact model.
KNN needs to store all training samples since classification requires comparing a new input against every stored sample. This quickly becomes infeasible for edge devices with limited memory.

After training, **the important model parameters—such as weights and biases—are extracted and saved into a .header file format**. This file can then be integrated into Freedom Studio, enabling deployment of the trained models on embedded RISC-V systems. Then we deploy it in Freedom Studio using the code snippet:

```

```

Here I learnt how machine learning models can be developed in Python, processed into a format suitable for low-level embedded platforms, and ultimately applied in real-world hardware implementations.


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

## Future Work

As I currently did not have access to the VSDSquadronPro board, I tried simulating it in QEMU. I plan to implement it in Hardware in further iterations of this project and implement more complex ML models and CNNs.




