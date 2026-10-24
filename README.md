## Overview

This repository contains an end-to-end machine learning and deep learning pipeline built in Python/PyTorch to analyze the **USA Real Estate Dataset** (over 2.2 million records). The project implements data preprocessing, Principal Component Analysis (PCA) for dimensionality reduction, target price classification, and a PyTorch feedforward classification neural network.

---

## Key Pipeline Steps

1. **Data Ingestion & Preprocessing**:
* Loads `realtor-data.zip.csv` via KaggleHub.


* Cleans and formats numerical and categorical features (such as `status`, `city`, `state`) using label encoding and median imputation.


* Converts timestamp columns (`prev_sold_date`) into numeric representations.




2. **Dimensionality Reduction (Part I)**:


* Applies `PCA` on standardized features to analyze variance preservation.


* Selects $m^* = 8$ principal components to capture $\ge 80\%$ cumulative variance ($\approx 84.35\%$).


* Reconstructs feature space and evaluates reconstruction MSE.




3. **Price Class Engineering (Part II)**:


* Bins continuous house prices into 4 balanced quantiles (`Low`, `Medium`, `High`, `Very High`) to frame the problem as a multi-class classification task.




4. **Neural Network Modeling (Part III)**:


* Splits data into stratified training and testing sets.


* Implements a PyTorch feedforward network (`SimpleNet` / classification architectures) utilizing `CrossEntropyLoss` and the Adam optimizer.





---

## Quick Start / Installation

Run the following commands in your environment or open directly in Google Colab using the badge below:

```bash
pip install numpy scikit-learn matplotlib pandas kagglehub seaborn torch --quiet

```
