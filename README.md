# Jaya-Based Feature Selection for Transport Layer DDoS Detection using XGBoost and SHAP

## Project Overview

Distributed Denial of Service (DDoS) attacks are one of the most common cyber threats that target network availability by overwhelming systems with malicious traffic. Detecting these attacks accurately while reducing computational complexity is an important challenge.

This project proposes a feature selection framework using the **Jaya Optimization Algorithm** to identify the most relevant transport layer features from the **CIC-DDoS2019** dataset. The selected features will later be used to train an **XGBoost** classifier, and **SHAP (SHapley Additive Explanations)** will be employed to interpret the model's predictions.

---

## Objectives

- Perform preprocessing on the CIC-DDoS2019 dataset.
- Remove unnecessary and redundant features.
- Implement the Jaya Optimization Algorithm for feature selection.
- Select an optimal subset of features.
- Train an XGBoost classifier using the selected features. *(Upcoming)*
- Explain model predictions using SHAP. *(Upcoming)*

---

## Dataset

**Dataset:** CIC-DDoS2019

The dataset contains network traffic flows consisting of both benign traffic and multiple DDoS attack types.

### Dataset Statistics

- Total Records: 2500
- Original Features: 87
- Features after preprocessing: 67
- Number of Classes: 14

Classes include:

- BENIGN
- DrDoS_DNS
- DrDoS_LDAP
- DrDoS_MSSQL
- DrDoS_NTP
- DrDoS_NetBIOS
- DrDoS_SNMP
- DrDoS_SSDP
- Portmap
- Syn
- TFTP
- UDP
- UDP-lag
- WebDDoS

---

# Project Workflow

```
CIC-DDoS2019 Dataset
        │
        ▼
Data Preprocessing
        │
        ▼
Jaya Feature Selection
        │
        ▼
Selected Features
        │
        ▼
XGBoost Classification (Upcoming)
        │
        ▼
Performance Evaluation (Upcoming)
        │
        ▼
SHAP Explainability (Upcoming)
```

---

# Data Preprocessing

The preprocessing module performs the following operations:

- Dataset loading
- Removal of identifier columns
- Missing value checking
- Duplicate removal
- Infinity value handling
- Label Encoding
- Constant feature removal
- Train-Test Split
- Feature Scaling using StandardScaler

Identifier columns removed:

- Unnamed: 0
- Flow ID
- Source IP
- Destination IP
- Timestamp
- SimillarHTTP

---

# Jaya Feature Selection

The project implements the **Standard Jaya Optimization Algorithm** for feature selection.

### Workflow

1. Initialize binary population
2. Generate candidate feature subsets
3. Train XGBoost on each subset
4. Compute fitness score
5. Identify best and worst solutions
6. Update population
7. Repeat for specified iterations
8. Select optimal feature subset

### Fitness Function

```
Fitness = (0.9 × Weighted F1 Score)
          - (0.1 × Feature Ratio)
```

where

Feature Ratio =

```
Number of Selected Features
---------------------------
Total Features
```

---

# Current Results

## Features after preprocessing

67

## Features selected using Jaya

20

### Selected Features

- Source Port
- Destination Port
- Protocol
- Total Length of Fwd Packets
- Fwd Packet Length Max
- Fwd Packet Length Min
- Fwd Packet Length Std
- Bwd Packet Length Max
- Flow Bytes/s
- Flow IAT Min
- Bwd IAT Mean
- Bwd IAT Std
- Bwd Packets/s
- Min Packet Length
- Packet Length Mean
- Packet Length Variance
- SYN Flag Count
- Init_Win_bytes_forward
- Init_Win_bytes_backward
- Active Max

### Best Fitness Score

```
0.8497
```

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost

---

# Project Structure

```
Final_Year_Project/
│
├── dataset/
├── preprocessing.py
├── jaya.py
├── main.py
├── cleaned_dataset.csv
├── scaled_dataset.csv
├── selected_features.csv
├── requirements.txt
└── README.md
```

---

# Current Progress

- Dataset preprocessing completed
- Jaya Feature Selection completed
- Selected feature subset generated

### Upcoming Work

- Train XGBoost classifier
- Evaluate model performance
- Generate confusion matrix
- Apply SHAP explainability
- Compare results with baseline model

