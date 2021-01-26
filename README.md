# Bug Severity Predictor for Mozilla

This repository contains code and associated files for deploying a bug severity predictor for Mozilla projects using Heroku.

## Project Overview

In this project, I'll build a severity predictor for the [Mozilla project](https://www.mozilla.org/en-US/) that uses the description of a bug report stored a in [Bugzilla Tracking System](https://bugzilla.mozilla.org/home) to predict its severity. 

The severity in the Mozilla project indicates how severe the problem is – from blocker ("application unusable") to trivial ("minor cosmetic issue"). Also, this field can be used to indicate whether a bug is an enhancement request. In my project, I have considered five severity levels: **trivial**, **minor**, **major**, **critical**, and **blocker**. I have ignored the default severity level (often **"normal"**) because this level is considered as a choice made by users when they are not sure about the correct severity level. 

This project will be broken down into three main notebooks:

**Notebook 1: Data Preparation**
* Load bug reports data from a single CSV file stored at [Mendey Data](https://data.mendeley.com/datasets/v446tfssgj/2).
* Download the necessary data from [Mendeley Data (https://data.mendeley.com/datasets/v446tfssgj/2) and extract the files into the folder **data/raw**.
* Explore basicaly the existing data features and the data distribution.
* Clean and convert data to suitable format for next steps in workflow machine learning.
* Notebook file path [Here].(1-data-preparation/prepare-data.ipynb)

**Notebook 2: Feature Engineering**

* Clean and pre-process the text data.
* Define features for comparing the similarity of an answer text and a source text, and extract similarity features.
* Select "good" features, by analyzing the correlations between different features.
* Create train/test `.csv` files that hold the relevant features and class labels for train/test data points.

**Notebook 3: Train and Deploy Your Model in SageMaker**

* Upload your train/test feature data to S3.
* Define a binary classification model and a training script.
* Train your model and deploy it using SageMaker.
* Evaluate your deployed classifier.

---

Please see the [README](https://github.com/udacity/ML_SageMaker_Studies/tree/master/README.md) in the root directory for instructions on setting up a SageMaker notebook and downloading the project files (as well as the other notebooks).

