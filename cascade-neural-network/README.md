# Ear Landmark Detection with Convolutional Neural Network (CNN)

## Overview

This repository presents a Convolutional Neural Network (CNN) model for the detection of 55 landmarks on the human ear. These landmarks play a crucial role in person identification and authentication. The model is trained on a dataset comprising 500 training images, 105 test images, and their corresponding landmarks. It primarily focuses on the right ear but has been modified to handle various forms, such as left ears and rotated images. By running the `CreateDataSet.py` script, you can generate an extended dataset with 3000 training images and 630 test images.

Before running the dataset creation script, ensure that you download the required data folder from the following link: [Data Folder](https://www.dropbox.com/sh/c8hizptl60lfogh/AADQN-kkuzkiP3ZcREQRxERsa?dl=0). The files and folder names in the downloaded data should be compatible with the functions provided in this repository.

The original data used for this project can be found at [source](https://ibug.doc.ic.ac.uk/resources/ibug-ears/) (check out Collection A). We've reorganized and renamed the content to simplify preprocessing for the model.

## Model Overview

- **Input:** The model takes 224x224 pixel images of ears as input.

- **Output:** It predicts the landmark points on the ear.

### Example Input and Output

#### Input
![right ear](/images/test_11.png)

#### Output
![right ear w/landmarks](/images/result_11.png)

#### Input
![left ear](/images/test_198.png)

#### Output
![left ear w/landmarks](/images/result_198.png)

## Model Architecture

![Model Architecture](/images/modelarch.jpg)

## Getting Started

To get started with this project, follow these steps:

1. Clone the repository to your local machine.

2. Download the required data folder from the link provided above.

3. Run the `CreateDataSet.py` script to generate an extended dataset.

4. Train and evaluate the CNN model using the provided code.

## Contributions and Issues

Contributions and feedback are welcome. Feel free to submit issues if you encounter any problems or have suggestions for improvements.

---

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fkbulutozler%2Fear-landmark-detection-with-CNN&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

---

*This repository is maintained by [Your Name].*
