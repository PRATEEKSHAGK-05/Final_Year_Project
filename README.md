## Jaya-Based Feature Selection for Transport Layer DDoS Detection using XGBoost and SHAP

## Project Overview

Distributed Denial of Service (DDoS) attacks are one of the most common cyber threats that target network availability by overwhelming systems with malicious traffic. Detecting these attacks accurately while reducing computational complexity is an important challenge.

This project proposes a feature selection framework using the **Standard Jaya Optimization Algorithm** to identify a compact subset of relevant network traffic features from the **CIC-DDoS2019** dataset. The selected features are then used to train an **XGBoost multiclass classifier**, and **SHAP (SHapley Additive exPlanations)** is used to interpret the model predictions.

The current implementation performs:

- Data preprocessing
- Feature reduction
- Standard Jaya-based feature selection
- XGBoost multiclass classification
- Performance evaluation
- Confusion matrix generation
- SHAP-based model explainability
- Global feature importance analysis

---

# Objectives

- Perform preprocessing on the CIC-DDoS2019 dataset.
- Remove unnecessary, identifier, and constant features.
- Implement the Standard Jaya Optimization Algorithm for feature selection.
- Select an optimal subset of network traffic features.
- Reduce the feature dimensionality while retaining useful classification information.
- Train an XGBoost multiclass classifier using the selected features.
- Evaluate DDoS traffic classification performance.
- Generate a confusion matrix for class-wise performance analysis.
- Apply SHAP explainability to understand model predictions.
- Identify the most influential features using global SHAP importance.
- Compare the Jaya-selected feature model with a baseline model in future work.

---

# Dataset

**Dataset:** CIC-DDoS2019

The CIC-DDoS2019 dataset contains network traffic flows representing benign traffic and multiple types of DDoS attacks.

For this project, a subset containing **2500 records** and **87 original features** is used.

## Dataset Statistics

| Property | Value |
|---|---:|
| Total Records | 2500 |
| Original Features | 87 |
| Features after Preprocessing | 67 |
| Features Selected by Jaya | 20 |
| Training Samples | 2000 |
| Testing Samples | 500 |
| Number of Classes | 14 |

## Classes

The dataset contains the following 14 classes:

1. BENIGN
2. DrDoS_DNS
3. DrDoS_LDAP
4. DrDoS_MSSQL
5. DrDoS_NTP
6. DrDoS_NetBIOS
7. DrDoS_SNMP
8. DrDoS_SSDP
9. Portmap
10. Syn
11. TFTP
12. UDP
13. UDP-lag
14. WebDDoS

---

# Project Workflow

```text
                    CIC-DDoS2019 Dataset
                              │
                              ▼
                    Data Preprocessing
                              │
              ┌───────────────┴───────────────┐
              │                               │
       Remove Identifier               Handle Missing /
          Features                    Infinite Values
              │                               │
              └───────────────┬───────────────┘
                              ▼
                       Label Encoding
                              │
                              ▼
                   Remove Constant Features
                              │
                              ▼
                     Train-Test Split
                              │
                              ▼
                     Feature Scaling
                              │
                              ▼
                67 Preprocessed Features
                              │
                              ▼
              Standard Jaya Feature Selection
                              │
                              ▼
                  20 Selected Features
                              │
                              ▼
                XGBoost Multiclass Classifier
                              │
                              ▼
                  Performance Evaluation
                              │
                ┌─────────────┼─────────────┐
                │             │             │
             Accuracy     Precision      Recall
                │             │             │
                └─────────────┼─────────────┘
                              ▼
                       Weighted F1 Score
                              │
                              ▼
                       Confusion Matrix
                              │
                              ▼
                    SHAP Explainability
                              │
                              ▼
               Global Feature Importance
