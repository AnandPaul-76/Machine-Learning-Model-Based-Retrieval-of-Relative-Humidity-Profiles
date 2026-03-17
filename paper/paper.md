---
title: "Machine Learning Model-Based Retrieval of Relative Humidity Profiles Measured by SAPHIR onboard Megha-Tropiques Satellite"

authors:
- name: Nouduri Anand Paul
  affiliation: 1

- name: Dr. Kandula V Subrahmanyam
  affiliation: 2

affiliations:
- name: MLR Institute of Technology, Hyderabad, India
  index: 1

- name: National Remote Sensing Centre (NRSC), Indian Space Research Organisation (ISRO), Hyderabad, India
  index: 2

date: 2026
bibliography: paper.bib
---
# Summary

Megha-Tropiques AI Humidity Retrieval is a machine learning–based software system for estimating atmospheric relative humidity from satellite observations. The software processes brightness temperature measurements from the SAPHIR instrument onboard the Megha-Tropiques satellite mission and predicts humidity across multiple atmospheric layers.

The system integrates satellite preprocessing, machine learning training, evaluation, and visualization tools into a unified workflow. Gradient boosted regression models implemented using LightGBM are trained to learn relationships between microwave radiometric observations and atmospheric humidity profiles.

# Statement of Need

Atmospheric humidity retrieval from satellite instruments is an important component of meteorological research and climate monitoring. Instruments such as SAPHIR measure microwave brightness temperatures that contain information about atmospheric water vapor.

Traditional retrieval techniques often rely on physical inversion algorithms that can be computationally expensive and complex to implement. Machine learning methods offer an alternative approach by learning nonlinear relationships between radiometric observations and atmospheric variables directly from data.

This software provides a reproducible machine learning pipeline for humidity retrieval using satellite observations. It enables researchers to preprocess satellite data, train humidity retrieval models, evaluate prediction performance, and visualize atmospheric humidity fields using an integrated Python workflow.

# Software Architecture

The software consists of four primary components:

1. Satellite data preprocessing pipeline for HDF5 datasets  
2. Machine learning training framework using LightGBM  
3. Model evaluation and statistical analysis tools  
4. Interactive visualization interface using Streamlit and Plotly

Satellite brightness temperature measurements are first extracted and preprocessed from SAPHIR HDF5 files. These observations are spatially matched with humidity measurements from corresponding atmospheric layers. The resulting dataset is used to train gradient boosted regression models that estimate humidity values across multiple layers.

The visualization interface allows users to explore prediction outputs through geospatial humidity maps, correlation plots, and error distributions.

# References

# AI Usage Disclosure

Generative AI tools (Google Gemini and OpenAI ChatGPT) were used during the development of this project. These tools assisted with code structuring, documentation drafting, formatting of the software paper, and general development guidance.

All AI-assisted outputs were reviewed, edited, and validated by the authors. The core scientific design, machine learning methodology, software architecture, and validation of results were performed by the human authors.