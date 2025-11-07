# Physical activity data correlate with fluctuations in estradiol

This repository contains code and data necessary for replicating the results from our paper "Physical activity data correlate with fluctuations in estradiol". 

## Data
Data included in this repository are as follows:
* Energy expenditure estimates from a data-driven model trained on shank IMU data and ground-truth respirometry
* Subject metadata (height, weight, age, date of menstruation onset, date of menstruation offset, and date of ovulation)
* Menstrual cycle phase labels in the order of study participation
* Estrogen and Progesterone reference values from prior literature

## Code
* Run main.ipynb to produce the figures presented in our paper
* model_functions.py contains functions for validating the shank-only XGBoost model for female participants
* data_functions.py contains functions for formatting the shank-only XGBoost model energy expenditure estimates
* plotting_functions.py contains functions for creating correlation plots used to compare energy expenditure with hormone reference values 

