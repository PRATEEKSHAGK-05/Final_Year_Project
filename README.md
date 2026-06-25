# Adaptive Chaotic Jaya and XGBoost-Based DDoS Detection for HAPS Networks

Final Year B.E. Computer Science and Engineering Project

**Domain:** Cybersecurity | Machine Learning | Network Intrusion Detection | Explainable AI

---

## Project Overview

Distributed Denial-of-Service (DDoS) attacks are one of the most critical cybersecurity threats, capable of disrupting communication services by flooding networks with malicious traffic. Existing detection methods often suffer from high computational complexity, limited interpretability, and inefficient feature selection.

This project proposes an intelligent DDoS detection framework using **Adaptive Chaotic Jaya Feature Selection**, **XGBoost**, and **SHAP Explainable AI**. The model is trained using the **CIC-DDoS2019** benchmark dataset and is designed as a lightweight and explainable intrusion detection framework for future **High-Altitude Platform Station (HAPS)** communication networks.

---

## Objectives

- Detect DDoS attacks using Machine Learning.
- Reduce redundant network features using Adaptive Chaotic Jaya.
- Perform binary classification (DDoS / Non-DDoS) using XGBoost.
- Improve model interpretability using SHAP Explainable AI.
- Develop a lightweight framework suitable for future HAPS communication networks.

---

## Problem Statement

- DDoS attacks overload network resources and disrupt legitimate services.
- Existing detection systems have high computational complexity.
- Large numbers of network traffic features increase processing time.
- Most existing models lack explainability.
- Future HAPS communication systems require efficient cybersecurity solutions.

---

## Proposed Workflow

```
CIC-DDoS2019 Dataset
        │
        ▼
Data Cleaning
        │
        ▼
Train-Test Split
        │
        ▼
Feature Scaling
        │
        ▼
Adaptive Chaotic Jaya Feature Selection
        │
        ▼
Optimal Feature Selection
        │
        ▼
XGBoost Classifier
        │
        ▼
DDoS / Non-DDoS Classification
        │
        ▼
SHAP Explainability
        │
        ▼
Performance Evaluation
```

---

## Dataset

**Dataset:** CIC-DDoS2019

The project uses the publicly available CIC-DDoS2019 benchmark dataset containing both benign and DDoS network traffic.

Binary Classification:

- **Benign → Non-DDoS**
- **All Attack Classes → DDoS**

---

## Base Papers

### Paper 1

**Memory-Efficient Multi-Island Jaya Feature Selection for Large-Scale DDoS Attack Detection (IEEE 2026)**

**Contribution**
- CIC-DDoS2019 dataset
- Jaya-based feature selection
- H2O AutoML classification

**Limitations**
- High implementation complexity
- AutoML increases computational overhead
- No Explainable AI support

---

### Paper 2

**Cybersecurity of High-Altitude Platform Stations: Threat Taxonomy, Attacks and Defenses With Standards Mapping—DDoS Attack Use Case (IEEE 2026)**

**Contribution**
- HAPS cybersecurity analysis
- DDoS threat identification
- Security standards and defense mechanisms

**Limitations**
- Survey-based work
- No Machine Learning framework
- No attack detection model

---

### Paper 3

**Performance Analysis of Machine Learning Techniques for Server Health Monitoring Using Time Series Data Against DDoS Attacks (IEEE 2025)**

**Contribution**
- Machine Learning-based DDoS detection
- Server health monitoring
- Boosting algorithms

**Limitations**
- Detects impact after server degradation
- No feature optimization
- No Explainable AI support

---

## Research Gap

- Existing methods focus mainly on improving classification accuracy.
- Limited use of Explainable AI in DDoS detection.
- High-dimensional traffic features increase computational complexity.
- Lack of optimized Machine Learning frameworks for HAPS security.
- Need for lightweight and interpretable DDoS detection systems.

---

## Proposed Solution

The proposed framework combines **Adaptive Chaotic Jaya Feature Selection**, **XGBoost**, and **SHAP Explainable AI** to develop an efficient DDoS detection model.

The system aims to:

- Select the most relevant network features.
- Reduce computational complexity.
- Improve classification accuracy.
- Provide interpretable predictions.
- Support future deployment in HAPS communication networks.

---

## Novelty

- Adaptive Chaotic Jaya using Logistic, Tent, and Sine chaos maps.
- Optimized feature selection with reduced computational complexity.
- Binary DDoS detection using XGBoost.
- SHAP-based Explainable AI for transparent predictions.
- Framework designed for future HAPS communication networks.

---

## Technologies

| Category | Tools |
|----------|-------|
| Programming Language | Python |
| Dataset | CIC-DDoS2019 |
| Machine Learning | XGBoost |
| Feature Selection | Adaptive Chaotic Jaya |
| Explainability | SHAP |
| Libraries | Pandas, NumPy, Scikit-learn, Matplotlib |

---

## Current Repository Status

### Completed

- Literature Survey
- Base Paper Analysis
- Problem Definition
- Domain Study
- Feasibility Analysis
- Workflow Design
- Research Gap Identification
- Zeroth Review Presentation

### Upcoming

- Dataset Preprocessing
- Feature Engineering
- Adaptive Chaotic Jaya Implementation
- XGBoost Model Development
- SHAP Explainability
- Model Evaluation
- Performance Comparison
- Final Project Documentation

---

## Repository Contents

- Research Papers
- Zeroth Review Presentation (PPT)
- Project Documentation
- README

Implementation notebooks, source code, and experimental results will be added in future updates.

---

## Future Work

- Implement Adaptive Chaotic Jaya Feature Selection.
- Train and evaluate the XGBoost classifier.
- Generate SHAP explanations.
- Compare results with baseline machine learning models.
- Validate the proposed framework for DDoS detection.

---

## Team

**Project Title:** Adaptive Chaotic Jaya and XGBoost-Based DDoS Detection for HAPS Networks

**Institution:** Kongu Engineering College

**Department:** Computer Science and Engineering

**Academic Year:** 2026–2027

---

## License

This repository is created for academic and educational purposes as part of the Final Year B.E. Project.
