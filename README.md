# RadonFAN - Intelligent Real-Time Radon Mitigation through IoT, Rule-Based Logic, and AI Forecasting

This repository includes the code, model checkpoints, and data used for the development of the paper **RadonFAN: Intelligent Real-Time Radon Mitigation through IoT, Rule-Based Logic, and AI Forecasting**.  

---

## Abstract

Radon (Rn-222) is a major indoor air pollutant with significant health risks.  
This work presents **RadonFAN**, a low-cost IoT system deployed in two galleries at the Institute of Physical and Information Technologies (ITEFI-CSIC, Madrid), integrating distributed sensors, microcontrollers, cloud analytics, and automated fan control to maintain radon concentrations below recommended limits.  

Initially, ventilation relied on a reactive, rule-based mechanism triggered when thresholds were exceeded. To improve preventive control, two end-to-end deep learning models based on **regression-to-classification (R2C)** and **direct classification (DC)** are developed.  

A quantitative analysis of predictive performance and computational efficiency is reported. While the R2C model is challenged by time-series persistence and error accumulation, the DC model achieves high classification performance (recall > 0.975) with low computational cost (<4 million parameters, 7 million FLOPs).  

Modifications to the DC model are studied to identify potential performance bottlenecks and the most relevant components, showing that most limitations arise from feature richness and time series behavior. When evaluated against the existing rule-based ventilation system, the DC model reduces both unsafe radon exposure events and energy consumption, demonstrating its effectiveness for preventive radon mitigation.

---

## Models defined

![Comparison of DC and R2C models](architecture.png)  

---
## Estructure of this repository
```
RadonFAN/
├── data/        # Input datasets
├── models/      # Saved trained checkpoints and preprocessing scalers for deployment
├── notebooks/   # Exploratory analysis, figures, and tables
└── sc/         # Source code for preprocessing, training, and evaluation
```
---
## How to use this repository

### 1. Clone the repository
```bash
git clone https://github.com/lidiaabad/RadonFAN-Intelligent-Real-Time-Radon-Mitigation-through-IoT-Rule-Based-Logic-and-AI-Forecasting.git
cd RadonFAN-Intelligent-Real-Time-Radon-Mitigation-through-IoT-Rule-Based-Logic-and-AI-Forecasting
```
### 2. Install requirements
```bash
pip install -r requirements.txt
```
### 3. Run the scripts

Shrun.sh is a bash file that calls freezetraintest.py of freezetest.py files to train and test or only test the models. 

```bash
nohup ./sc/shrun.sh > ./logname.log
```

The main files (freezetraintest.py and freezetest.py) use other files of the sc folder including: 
- the arguments file to parse arguments, 
- the datapreprocesing file to preprocess data, augment them, and create dataloaders on the fly, 
- the archs file with the architecture modules,
- losscustom with several losses implemented, 
- helpers to include functionalities into the script,
- reporter to define the reported that produces the .json file with the training and testing information of each model. 


