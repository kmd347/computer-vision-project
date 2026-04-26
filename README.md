# Grading Coffee Bean Quality: A Computer Vision Approach

# Coffee Quality Classification (Check-In 1)

## 1. Problem Framing + Scope

My project aims to build a supervised classification model that predicts whether a coffee sample is high quality based on measureable chemical and sensory attributes. 

The aim is to create a binary classification task.

Input: Numerical features describing coffee properties (acidity, aroma, body sweetness, moisture, etc.)

Output: 
- 1 = High-quality
- 2 = Not high-quality

A coffee sample is classified as high quality if its quality score is greater than or equal to 85. 

Success will be evaluated using accuracy, macro F1 score and a confusion matrix.

Modality: Structured tabular data

The scope of my project is to develop a strong baseline model using logistic regression before exploring more complex models later in the semester.


## 2. Dataset Access + Documentation

Dataset: Coffee Quality Data
Source: UCI Machine Learning Repository

The dataset contains expert-evaluated coffee samples with numerical measurements describing sensory + chemical characteristics.

Key features include:
- Acidity
- Aroma
- Body
- Sweetness
- Moisture
- Uniformity
- Aftertaste
- Total cup points (quality score)

The numeric quality score will be converted into a binary classification label.

Access instructions:
1. The dataset is loaded directly from a public GitHub repository using a URL.
2. No manual download or local CSV file is required.
3. Run the notebook (eda.ipynb) and the dataset will automatically load using pandas.

## 3. Data Audit (Summary)
EDA includes:

- Reviewing the dataset shape (1311 rows and 44 columns)
- Looking at summary statistics for the numerical features
- Visualizing the distribution of total cup quality scores
- Checking class balance after creating the binary target
- Exploring correlations between key numerical features

From the initial analysis there is clear variation across the sensory attributes and a wide range of quality scores. After converting the quality score into a binary variable (quality ≥ 85), the dataset is highly imbalanced with 1205 samples labeled as "not high quality" and 106 labeled as "high quality."

The correlation heatmap shows strong positive relationships among sensory attributes such as Aroma, Flavor, Aftertaste, and Acidity, and these features are also strongly related to Total Cup Points.

Before modeling, feature scaling and handling missing values will most likely be necessary.

Full exploratory analysis is provided in eda.ipynb.

## 4. Evaluation Plan
Dataset split:
- 70% training
- 15% validation
- 15% testing

Evaluation metrics:
- Accuracy
- Macro F1 score
- Confusion matrix

Macro F1 is included to make sure there that performance is balanced across both of the classes.

## 5. Initial Baseline
The initial baseline model will use logistic regression.

Steps:
1. Handle missing values.
2. Standardize numerical features.
3. Train logistic regression on the training set.
4. Evaluate on validation and test sets.

Future improvements may include tree-based models such as random forests or gradient boosting.





# Coffee Quality Classification (Check-In 2)
### Note on Project Update
While the initial project was framed as a binary classification task (high quality vs. not high quality), further exploration of the dataset revealed that a multi-class classification setup provides a more informative and realistic representation of coffee quality. 

As a result, the project scope was updated in Check-In 2 to predict multiple quality levels instead of a binary outcome. This allows for a more nuanced evaluation of model performance and better reflects the structure of the underlying data.

### Classical Baseline 
A logistic regression model was used as the classical baseline. The model was trained on a selected subset of meaningful numerical features (e.g., Aroma, Flavor, Aftertaste, Acidity, and related sensory attributes), after removing features that directly encoded overall quality scores to avoid data leakage. 

All features were standardized prior to training. The model was evaluated using accuracy, macro F1, and a confusion matrix to assess performance across multiple quality classes. 

### Deep Learning Baseline
A multilayer perceptron (MLP) was used as the deep learning baseline. This model is capable of capturing nonlinear relationships between the selected sensory features and coffee quality. The same preprocessing steps and evaluation metrics were applied to ensure a fair comparison with the logistic regression model. 

### Model Comparison 
Both logistic regression and the neural network achieve strong overall performance, with accuracies around 88%. The neural network slighly outperforms logistic regression in macro F1 score, indicating improved balance across classes. 

However, both models exhibit similar prediction patterns, suggesting that much of the underlying signal in the data is relatively linear. While the neural network captures some additional complexity, it does not dramatically outperform the simpler logistic regression baseline. 

### Failure Analysis 
Despite strong overall performance, both models struggle to accurately predict the highest quality class (class 2). The confusion matrix shows that many of these samples are misclassified as the middle quality class (class 1).

The issue is likely due to class imbalance, as the highest quality class has significantly fewer examples compared to the other classes. As a result, the models are biased toward predicting the more common classes and have difficulty learning distinct patterns for the rare class. 



# Coffee Quality Classification (Check-In 3)
In check-in 3, I wanted to extend my project from more structured, tabular data into a more computer vision-based approach using image data of coffee beans. 

Instead of relying strictly on numerical features, the model now takes an image of coffee beans as input and predicts the quality grade directly from visual characteristics. 

This extension uses transfer learning via a pretrained CNN which serves as the "advanced component" for the third check-in.

### Methodology 
I used a pretrained ResNet18 CNN as a feature extractor. 
- The final classification layer of ResNet was removed 
- The model outputs high-level feature vectors representing each image 
- These features are then used as input into a logistic regression classifier

As a result I was able to create a two-stage pipeline: 
Image --> ResNet --> Logistic Regression --> Prediction

### Dataset 
The dataset I used, called the CBD Coffee Bean Dataset obtained from the Amrita Vishwa Vidyapeetham School of Arts and Sciences in India, contains labeled images of coffee beans grouped by quality grade (ex. A, AA, PB-II, Bulk, etc). The dataset was originally created to support automated coffee grading where visual features like size, uniformity and defects can be used to assess quality. This is similar to how experts evaluate beans in real-world production.

In order to improve efficiency, I made sure a subset of images from each class was used during training. 

The dataset is not included in the repository due to size. It can be downloaded from:
https://data.mendeley.com/datasets/52877z55vr/1 

Place the dataset folder in the root directory as:
CBD_Coffee Bean Dataset/

### Results 
The image-based model achieved an accuracy of approx. 80% on the test set. While this is slightly lower than the tabular models it shows that visual features alone contain some meaningful signals for predicting coffee quality.

### Comparison to Previous Models 
Compared to the earlier tabular models:
- The tabular models performed better (approx. 88% accuracy)
- The image model introduces a new modality (vision)
- It captures visual characteristics that are not present in structured data 

My new extension shows how different data modalities can impact model performance and highlights the tradeoffs between structured / unstructured data.

### Demo Application 
To demonstrate my model, I built a Streamlit app where users can upload an image of coffee beans and receive a predicted quality grade. 

This gives an interactive way to visualize model behavior and test predictions on new images as well. 

### Ablation
To test the effect of dataset size I changed the number of images per class (10,20,30) while keeping the remainder of the pipeline fixed. The results showed inconsistent performance the more data that was added which suggests that the model finds it tough to effectively use additional / more variable images. 

### Failure Analysis
The model performs worse on images where beans overlap or lighting is more inconsistent in nature. These conditions undoubtedly made it tougher for the model to identify clear visual patterns. This therefore demonstrates a limitation in the feature extraction approach. 

### Future Work 
In future work, I would replace the current fixed feature approach with a fine-tuned CNN and increase the dataset size as well. This would probably improve the model's ability to capture more meaningful visual features + generalize better. 

## Requirements
This project uses the following Python libraries:

- torch
- torchvision
- scikit-learn
- streamlit
- numpy
- pillow
- matplotlib 

Run the strealit app with streamlit run app.py