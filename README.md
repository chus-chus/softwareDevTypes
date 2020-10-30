[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


# An extraction and analysis of software development profiles from project quality data.

### **Description:**
The aim of this analysis is to find distinct groups of developer profiles based on distinct characteristics based on software development quality data. Moreover, we want to identify the different types of bugs that each developer group tends to introduce.

The results of this project could be used by department leaders and general managers to help their developers be more efficient by reducing the number of bugs they produce providing them with training options, for example; or directly by developers if they want to improve their self-awareness. Bearing in mind that developers want to improve their abilities and managers are (or should be) interested in dealing with their projects more efficiently, we see the potential results as very favourable for any applicable entity.

### **Data:**
In this project we have used The Technical Debt Dataset, a set of project measurement data from 33 Java projects. This data is obtained using different tools to analyse each commit and can be downloaded from this [link](https://github.com/clowee/The-Technical-Debt-Dataset).

### **Usage:**
All of the steps taken to get our results (data description, pre-processing, modelling, and evaluation) can be reproduced exactly. Most, if not all, of the codebase is based on jupyter notebooks, which allow a nice interactiveness. So, following the structure indicated in this README file one will be able to replicate our results. Moreover, the required packages and their corresponding versions are indicated in the `requirements.txt` file. Note that the results that relied on random components, such as initialisation, were seeded. Here is the explicit order one would follow as per the names of the folders:

1. **DataDescription:** optional, does not affect results. There are 8 jupyter notebooks with the description analysis of each table from the original dataset.

2. **DataPreparation:** in here there are five things that need to be done:
        
    a. **SelectData:** select the relevant attributes.
In this case here are 7 notebooks. Once we execute them, they will read the raw dataset and return new csv files with just the selected attributes.
        
    b. **CleanData:** deal with the missing values.
There are also 7 notebooks. Once we execute them, they will read the datasets obtained in the previous step (*Select data*) and return new csv files with the data cleaned (without missing values). In this case, we have to take in mind that some notebooks need to be executed before some others because they use the csv obtained after executing them. This is the case of the `SONAR_ISSUES.ipynb` notebook that has to be executed after the notebook `GIT_COMMITS.ipynb` of the same folder.
        
    c. **ConstructData:** create new attributes.
In this case there are 9 notebooks. Each one reads the csv files (from the ones obtained in the previous steps, *Clean data* and *Select data*) that needs to create new attributes and returns new csv files.
        
    d. **Integrate data:** join the different tables.
It is just one notebook (`integration.ipynb`) that reads all the csv files from the previous steps that needs and joins them into a unique dataset.
        
    e. **Format data:** check the type of all the attributes.
Also just one notebook (`format.ipynb`). Once we execute it, it reads the previous dataset and returns a new one with the correct  names for the attributes (to make them more understandable) and the correct types for each attribut. This final dataset is save as a csv file into the folder `data/processed/`.
        
3. **Modelling:** a full step by step modelling guide, constituted by just one notebook that reads the csv files obtained from *DataPreparation* and returns 3 csv files: one with the mean values of each attribute for the 5 clusters obtained using tSVD for the dimensionality reduction, one with the mean values of each attribute for the 5 clusters obtained using PCA for the dimensionality reduction and one with the mean values of each attribute for the 3 clusters obtained using tSVD for the dimensionality reduction. These new files are saved into the folder `data/interim/Modelling/`.

4. **Evaluation:** where we evaluate the different results obtained in the previous step. Contains 2 notebooks, one to analyze the clusters obtained with PCA and one to analyze the clusters obtained using tSVD to make the dimensionality reduction. They read the dataset obtained after *DataPreparation* and also the .csv files with the mean values of the attributs of each cluster obtained from *Modelling*.

An alternative for **2. DataPreparation**, **3. Modelling** and **4. Evaluation** is to run the `.py` scripts in the `src` folder:

 1. **src/features/build_features.py** Runs all of the data preparation steps, building the dataset that contains developers and their features and saving each interim data frame.
 2. **src/models/modelling.py** Performs the modelling and outputs the different mean developer profiles from each model.
 3. **src/models/evaluation_*.py** Evaluate different clustering models.
 
Before performing these two steps, the raw csv files from the TechDebt Dataset must be in `/data/raw/`.

We have also included in the replication package (in the folder `models`) the final selected model in a joblib object: using tSVD for dimensionality reduction and KMeans with 5 clusters. This model could in fact be put in production, for example, as an Amazon Lambda function and simple web app, but we do not believe it would be of great use: the conclusions stated and the recommendations to the business (and the actions taken by it) are the important part of this study. In the future, one could re-perform our analysis to see if the clusters have changed in a good way (i.e. more clean code, less issues induced).

### Project structure
```
.
├── LICENSE
├── README.md
├── data
│   ├── interim
│   │   ├── DataPreparation
│   │   │   ├── CleanData
│   │   │   ├── ConstructData
│   │   │   └── SelectData
│   │   └── Modelling
│   │       ├── 3clusterProfilesTSVD.csv
│   │       ├── clusterProfilesPCA.csv
│   │       └── clusterProfilesTSVD.csv
│   ├── processed
│   │   └── DEVELOPERS_DATA.csv
│   └── raw
├── models
│   └── KMeans_tSVD_5_clusters.joblib
├── notebooks
│   ├── 1-DataUnderstanding
│   │   ├── 1.\ DataDescription
│   │   │   ├── 1-DB-projects.ipynb
│   │   │   ├── 2-DB-jira-issues.ipynb
│   │   │   ├── 3-DB-git-sonar-measures.ipynb
│   │   │   ├── 4-DB-sonar-issues.ipynb
│   │   │   ├── 5-DB-git-commits.ipynb
│   │   │   ├── 6-DB-szz-fault-inducing-commits.ipynb
│   │   │   ├── 7-DB-git-commits-changes.ipynb
│   │   │   └── 8-DB-refactoring-miner.ipynb
│   │   ├── 2-DB-data-exploration.ipynb
│   │   └── 3-DB-data-quality.ipynb
│   ├── 2-DataPreparation
│   │   ├── 1-DB-integration.ipynb
│   │   ├── 1-SelectData
│   │   │   ├── 1-DB-JIRA-ISSUES.ipynb
│   │   │   ├── 2-DB-SONAR-MEASURES.ipynb
│   │   │   ├── 3-DB-SONAR-ISSUES.ipynb
│   │   │   ├── 4-DB-GIT-COMMITS.ipynb
│   │   │   ├── 5-DB-SZZ-FAULT-INDUCING-COMMITS.ipynb
│   │   │   ├── 6-DB-GIT-COMMITS-CHANGES.ipynb
│   │   │   └── 7-DB-REFACTORING-MINER.ipynb
│   │   ├── 2-CleanData
│   │   │   ├── 1-DB-JIRA-ISSUES.ipynb
│   │   │   ├── 2-DB-SONAR-MEASURES.ipynb
│   │   │   ├── 3-DB-SONAR-ISSUES.ipynb
│   │   │   ├── 4-DB-GIT-COMMITS.ipynb
│   │   │   ├── 5-DB-SZZ-FAULT-INDUCING-COMMITS.ipynb
│   │   │   ├── 6-DB-GIT-COMMITS-CHANGES.ipynb
│   │   │   └── 7-DB-REFACTORING-MINER.ipynb
│   │   ├── 2-DB-format.ipynb
│   │   └── 3-ConstructData
│   │       ├── 1-DB-TIME-IN-EACH-PROJECT.ipynb
│   │       ├── 2-DB-NUMBER_COMMITS.ipynb
│   │       ├── 3-DB-FIXED-ISSUES.ipynb
│   │       ├── 4-DB-INDUCED-ISSUES.ipynb
│   │       ├── 5-DB-REFACTORING-MINER-bug.ipynb
│   │       ├── 6-DB-JIRA-ISSUES-time.ipynb
│   │       ├── 7-DB-SONAR-ISSUES-time.ipynb
│   │       └── 8-DB-SONAR-MEASURES-difference.ipynb
│   ├── 3-Modelling
│   │   └── 1-DB-modelling.ipynb
│   └── 4-Evaluation
│       ├── 1-DB-Evaluation-TSVD.ipynb
│       └── 2-DB-Evaluation-PCA.ipynb
├── reports
│   ├── Report.docx
│   └── Report.pdf
├── requirements.txt
└── src
    ├── evaluation
    │   ├── evaluation_PCA.py
    │   └── evaluation_TSVD.py
    ├── features
    │   └── build_features.py
    └── models
        └── modelling.py
```
