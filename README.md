
# Skin Cancer Classification using Robust and Explainable Vision Models

This project aims to build a robust vision model capable of accurate and reliable classification of skin lesions, with a focus on model explainability. The model should have good specificity, recall, and F1 scores compared to the initial version, as false negatives in model classification are not acceptable in a medical diagnosis model. To achieve this goal, we employ techniques from active research areas like GRAD-CAM and ResNet.

## Table of Contents

- [Model Engineering](#model-engineering)
- [Data Engineering](#data-engineering)
- [Pre-built Architectures (Res Net)](#pre-built-architectures-res-net)
- [Explainability in Vision Models](#explainability-in-vision-models)
- [Solution Design and Plan](#solution-design-and-plan)
    - [Solution Stages](#solution-stages)
    - [High-level Solution Architecture](#high-level-solution-architecture)
- [Data](#data)
    - [Data Samples](#data-samples)
- [Solution & Experiments](#solution--experiments)
    - [Model](#model)
    - [Model Tuning](#model-tuning)
    - [Receptive Field & Batch Normalization](#receptive-field--batch-normalization)
    - [Data Augmentation & Regularization](#data-augmentation--regularization)

- [Explainability](#explainability)




## Model Engineering

Designing models requires discipline, with every step taken having a clear purpose for achieving the final goal. There is often a trade-off while tuning the model, for example, adding/removing layers or resizing input data.

Various components build up the ideal model for the use case at hand. A few that really helped improve and optimize the model performance and make the model robust are 1x1 convolution and GAP (Global Average Pooling Layer) layers. BatchNorm layer helped large number of layers to be trained more quickly and demonstrate improved generalization when the distribution of activations is maintained normalized throughout the BackProp process. Lr schedulers also a played a crucial part in converging faster and not be stuck on platue during learning process.


## Data Engineering

The core thought process behind data engineering for the models is "Garbage in, garbage out." Feeding the model good representative data is crucial. Even the best vision architectures are wasted and will perform poorly if the input data is of low quality or unrepresentative of real-world data. For vision models, one can even use domain knowledge to augment data to fill in for unseen variations in train data that would most likely occur in real data. Image normalization, batch normalization, and data augmentations such as rotation, scaling, saturation, and random noise are a few techniques that help distort the train data to create new, noisier data that also helps the model learn more variations and generalize better.

## Pre-built Architectures (Res Net)

ResNet is a pre-built architecture that we utilize to build the solution, with ResNet 34 and 18 being the ones used. The training of parameters will happen from scratch. The core idea behind ResNet architecture is introducing "identity shortcut connection or Skip connections" that skips layers and adds the output down the road to a different layer. A residual block is displayed in the following fig. ResNet can be considered an ensemble of smaller networks.



## Explainability in Vision Models

The core idea behind this project is model explainability. Understanding what the model is looking at while making a particular prediction is crucial for model maintenance, tuning, and re-training. In vision models, this can be achieved either by gradients or by understanding feature importance with a change in output.

### GRAD-CAM

GRAD-CAM (Gradient-weighted Class Activation Mapping) is one of the model explainability algorithms used in this project to bring out model explainability. It utilizes the gradients of a particular class of interest, such as 'dog', that flow into the last convolutional layer and does a weighted average of the activation maps to generate a heatmap. This heat map indicates significant areas in the image that aided the model in predicting the target class. A few other approaches were also utilized and will be further touched upon in the [Solution & Experiments](#solution--experiments) section.

## Solution Design and Plan

In this solution, we plan to build a model that is not only robust and lightweight but also explainable and locally interpretable. We will implement and showcase explain



### Solution Stages
- Business process flow
- Research
- Data Collection
- Experimentation
- Implementation
- Simple UI for Demos (Streamlit or Grad io)
- Dockerization
- Amazon ECS deployment

### High-level Solution Architecture
If the final solution is to be divided into layers from the inside out, I would consist of the following layers:

- Deep Model - Model Layer
- Streamlit/ grad IO - UI Layer
- Docker - Docker Layer
- AWS ECS - Deployment layer



## The Data
Data Collection and Understanding
The dermatoscopic data used in this project has been acquired through Harvard Dataverse & Kaggle. The dataset comprises of 10015 dermatoscopic images, representing all significant pigmented lesion diagnostic subtypes.

### Data Samples
The cases comprise of good representative collection of all significant pigmented lesion diagnostic subtypes:

- Actinic keratoses and intraepithelial carcinoma / Bowen's disease (AKIEC)
- Basal cell carcinoma (BCC)
- Benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus-like keratoses, BKL)
- Dermatofibroma (DF)
- Melanoma (MEL)
- Melanocytic nevi (NV)
- Vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage, VASC)


## The Solution & Experiments

### Model
Several models were experimented with in this project, from building a custom model from scratch to utilizing pre-built architectures like resnet-18 and resnet-34. Multiple data augmentation techniques, optimizers, learning rate schedulers, and other regularizing techniques were also experimented with. The model building started with a base CNN model built from scratch using the idea of a receptive field.

- Custom Model Architecture

| Layer (type)   | Output Shape      | Param #   |
| -------------- | ---------------- | ---------|
| Conv2d-1       | [-1, 8, 224, 224] | 224      |
| ReLU-2         | [-1, 8, 224, 224] | 0        |
| BatchNorm2d-3  | [-1, 8, 224, 224] | 16       |
| Dropout-4      | [-1, 8, 224, 224] | 0        |
| Conv2d-5       | [-1, 16, 224, 224]| 1,168    |
| ReLU-6         | [-1, 16, 224, 224]| 0        |
| BatchNorm2d-7  | [-1, 16, 224, 224]| 32       |
| Dropout-8      | [-1, 16, 224, 224]| 0        |
| Conv2d-9       | [-1, 8, 224, 224] | 136      |
| ReLU-10        | [-1, 8, 224, 224] | 0        |
| BatchNorm2d-11 | [-1, 8, 224, 224] | 16       |
| Dropout-12     | [-1, 8, 224, 224] | 0        |
| MaxPool2d-13   | [-1, 8, 112, 112] | 0        |
| Conv2d-14      | [-1, 16, 112, 112]| 1,168    |
| ReLU-15        | [-1, 16, 112, 112]| 0        |
| BatchNorm2d-16 | [-1, 16, 112, 112]| 32       |
| Dropout-17     | [-1, 16, 112, 112]| 0        |
| Conv2d-18      | [-1, 16, 112, 112]| 2,320    |
| ReLU-19        | [-1, 16, 112, 112]| 0        |
| BatchNorm2d-20 | [-1, 16, 112, 112]| 32       |
| Dropout-21     | [-1, 16, 112, 112]| 0        |
| Conv2d-22      | [-1, 32, 112, 112]| 4,640    |
| ReLU-23        | [-1, 32, 112, 112]| 0        |
| BatchNorm2d-24 | [-1, 32, 112, 112]| 64       |
| Dropout-25     | [-1, 32, 112, 112]| 0        |
| Conv2d-26      | [-1, 8, 112, 112] | 264      |
| ReLU-27        | [-1, 8, 112, 112] | 0        |
| BatchNorm2d-28 | [-1, 8, 112, 112] | 16       |
| Dropout-29     | [-1, 8, 112, 112] | 0        |
| MaxPool2d-30   | [-1, 8, 56, 56]   | 0        |
| Conv2d-31      | [-1, 8, 56, 56]  |     584 |
| ReLU-32        | [-1, 8, 56, 56]  |       0 |
| BatchNorm2d-33 | [-1, 8, 56, 56]  |      16 |
| Dropout-34     | [-1, 8, 56, 56]  |       0 |
| Conv2d-35      | [-1, 16, 56, 56] |   1,168 |
| ReLU-36        | [-1, 16, 56, 56] |       0 |
| BatchNorm2d-37 | [-1, 16, 56, 56] |      32 |
| Dropout-38     | [-1, 16, 56, 56] |       0 |
| Conv2d-39      | [-1, 32, 56, 56] |   4,640 |
| ReLU-40        | [-1, 32, 56, 56] |       0 |
| BatchNorm2d-41 | [-1, 32, 56, 56] |      64 |
| Dropout-42     | [-1, 32, 56, 56] |       0 |
| Conv2d-43      | [-1, 8, 56, 56]  |     264 |
| ReLU-44        | [-1, 8, 56, 56]  |       0 |
| BatchNorm2d-45 | [-1, 8, 56, 56]  |      16 |
| Dropout-46     | [-1, 8, 56, 56]  |       0 |
| MaxPool2d-47   | [-1, 8, 28, 28]  |       0 |
| Conv2d-48      | [-1, 16, 28, 28] |   1,168 |
| ReLU-49        | [-1, 16, 28, 28] |       0 |
| BatchNorm2d-50 | [-1, 16, 28, 28] |      32 |
| Dropout-51     | [-1, 16, 28, 28] |       0 |
| Conv2d-52      | [-1, 32, 28, 28] |   4,640 |
| ReLU-53        | [-1, 32, 28, 28] |       0 |
| BatchNorm2d-54 | [-1, 32, 28, 28] |      64 |
| Dropout-55     | [-1, 32, 28, 28] |       0 |
| MaxPool2d-56   | [-1, 32, 14, 14] |       0 |
| Conv2d-57      | [-1, 32, 14, 14] |   9,248 |
| ReLU-58        | [-1, 32, 14, 14] |       0 |
| BatchNorm2d-59 | [-1, 32, 14, 14] |      64 |
| BatchNorm2d-59 | [-1, 32, 14, 14]  | 64       |
| Dropout-60     | [-1, 32, 14, 14]  | 0        |
| Conv2d-61      | [-1, 64, 14, 14]  | 18,496   |
| ReLU-62        | [-1, 64, 14, 14]  | 0        |
| BatchNorm2d-63 | [-1, 64, 14, 14]  | 128      |
| Dropout-64     | [-1, 64, 14, 14]  | 0        |
| MaxPool2d-65   | [-1, 64, 7, 7]    | 0        |
| Conv2d-66      | [-1, 7, 7, 7]     | 455      |
| ReLU-67        | [-1, 7, 7, 7]     | 0        |
| BatchNorm2d-68 | [-1, 7, 7, 7]     | 14       |
| Dropout-69     | [-1, 7, 7, 7]     | 0        |
| AdaptiveAvgPool2d-70 | [-1, 7, 1, 1] | 0        |

Total params: 51,221
Trainable params: 51,221
Non-trainable params: 0



- Res 18 Architecture


| Layer (type) | Output Shape         | Param #    |
|--------------|----------------------|------------|
| Conv2d       | [-1, 64, 64, 64]     | 1,728      |
| BatchNorm2d  | [-1, 64, 64, 64]     | 128        |
| 2 x BasicBlock (2 Conv2d + 2 BatchNorm2d each) | [-1, 64, 64, 64] | 148,096  |
| BasicBlock (1 Conv2d + 1 BatchNorm2d + 1 Conv2d + 1 BatchNorm2d + 1 Conv2d + 1 BatchNorm2d) | [-1, 128, 32, 32] | 230,144  |
| 2 x BasicBlock (2 Conv2d + 2 BatchNorm2d each) | [-1, 128, 32, 32] | 592,128  |
| BasicBlock (1 Conv2d + 1 BatchNorm2d + 1 Conv2d + 1 BatchNorm2d + 1 Conv2d + 1 BatchNorm2d) | [-1, 256, 16, 16] | 920,576  |
| 2 x BasicBlock (2 Conv2d + 2 BatchNorm2d each) | [-1, 256, 16, 16] | 2,361,856  |
| BasicBlock (1 Conv2d + 1 BatchNorm2d + 1 Conv2d + 1 BatchNorm2d + 1 Conv2d + 1 BatchNorm2d) | [-1, 512, 8, 8] | 4,722,688  |
| 2 x BasicBlock (2 Conv2d + 2 BatchNorm2d each) | [-1, 512, 8, 8] | 9,439,232  |
| Linear       | [-1, 7]              | 3,591      |
| Total        |                      | 11,172,423 |


Total params: 11,172,423
Trainable params: 11,172,423
Non-trainable params: 0

Resnet 18 was performing the best out of the models that I have tested.

Training phase graphs for accuracy and loss
![alt text](https://github.com/SainadhAmul/explainable_cnn_sc/blob/main/Other%20Plots/Res%2018%20Acc.png?raw=true)
![alt text](https://github.com/SainadhAmul/explainable_cnn_sc/blob/main/Other%20Plots/Res%2018%20Loss.png?raw=true)

Comapritive -Test Accuracies across models with a particular set of tuning

| Model       | Base Model | With Batch Normalization | BN + With Select Data Augmentations + Tuning |
| ----------- | ---------- | ------------------------ | -------------------------------------------- |
| Custom CNN  | 0.678      | 0.723                    | 0.739                                        |
| Resnet-18   | 0.82       | NA                       | 0.889                                        |
| Resnet-34   | 0.8321     | NA                       | 0.8742                                       |


Comapritive -Test Accuracies with different optimizers

| Optimizers                | SGD with Momentum | RMSprop | Adam |
| -------------------------| ----------------- | ------- | ---- |
| Resnet-18 Base (20 Epoch) | 0.82              | 0.734   | 0.76 |


Final Model specificity measures
| class | sensitivity | specificity |
| ----- | -----------| ------------|
| akiec | 0.999171   | 0.994845    |
| bcc   | 0.996647   | 0.990338    |
| bkl   | 0.988468   | 0.924731    |
| df    | 0.999155   | 1.000000    |
| mel   | 0.990909   | 0.957895    |
| nv    | 0.992487   | 0.920792    |
| vasc  | 0.999163   | 1.000000    |

Final Model results
| class     | precision | recall    | f1-score | support   |
| --------- | --------- | --------- | -------- | --------- |
| akiec     | 0.994845  | 0.994845  | 0.994845 | 194.000000|
| bcc       | 0.980861  | 0.990338  | 0.985577 | 207.000000|
| bkl       | 0.924731  | 0.924731  | 0.924731 | 186.000000|
| df        | 0.995392  | 1.000000  | 0.997691 | 216.000000|
| mel       | 0.943005  | 0.957895  | 0.950392 | 190.000000|
| nv        | 0.953846  | 0.920792  | 0.937028 | 202.000000|
| vasc      | 0.995146  | 1.000000  | 0.997567 | 205.000000|
| accuracy  | 0.970714  | 0.970714  | 0.970714 | 0.970714  |
| macro avg | 0.969689  | 0.969800  | 0.969690 | 1400.000000|
| weighted avg | 0.970640 | 0.970714 | 0.970622 | 1400.000000|


### Model Tuning
The base model was tuned by adding more layers and complexities like 1x1 convolution to reduce the number of channels, batch normalization to deal with exploding & vanishing gradients, data augmentations, and regularization to keep the model generalized, global average pooling instead of huge linear layers to reduce the computations and parameters and even yield better accuracy. Tested the models with multiple optmizers to see which one is converging faster and which is more compute efficient.

![alt text](https://github.com/SainadhAmul/explainable_cnn_sc/blob/main/Other%20Plots/opt.png?raw=true)

### Receptive Field & Batch Normalization
To build the base model that would look at the entire image before making the classification, the idea of receptive fields was crucial. This concept helped add enough layers to reach the final receptive field equal to the size of the image. And to counteract the problems like exploding gradient with deeper networks, batch normalization was used.


### Data Augmentation & Regularization
Select data augmentation techniques played a huge role in improving the overall accuracy of the model by generating more varied train data for the model to learn. Using data augmentation techniques, a better accuracy/ generalized model with a smaller initial training set was achieved. Regularization methods like dropout with a probability of 0.01, l1& l2 regularizations were also used.


![alt text](https://github.com/SainadhAmul/explainable_cnn_sc/blob/main/Other%20Plots/DA.png?raw=true)

![alt text](https://github.com/SainadhAmul/explainable_cnn_sc/blob/main/Explainable%20%26%20Debuggable%20Outputs/j8csJNB.gif?raw=true)


To dive deeper into the project and experiments please refer to the final report



