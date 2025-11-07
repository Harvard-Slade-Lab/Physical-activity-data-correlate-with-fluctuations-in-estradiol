# Physical activity data correlate with fluctuations in estradiol

This repository contains code and data necessary for replicating the results from our paper "Physical activity data correlate with fluctuations in estradiol." 

## Code
* Run main.ipynb to produce the figures presented in our paper
* model_functions.py contains functions for validating the shank-only XGBoost model for female participants
* data_functions.py contains functions for formatting the shank-only XGBoost model energy expenditure estimates
* plotting_functions.py contains functions for creating correlation plots used to compare energy expenditure with hormone reference values
  
## Data
Data included in this repository are as follows:
* **energy_estimates.pkl.zip:** Energy expenditure estimates from a data-driven model trained on shank IMU data and ground-truth respirometry
* **metadata.csv:** Subject height, weight, age, date of menstruation onset, date of menstruation offset, and date of ovulation
* **phasetable.csv:** Menstrual cycle phase labels in the order of study participation
* **hormone_reference_values.csv:** Estrogen and Progesterone reference values collected from similar cohorts (citations for reference values are below)
  * Estradiol values:
    * Dighe, A. S. et al. "High-resolution reference ranges for estradiol, luteinizing hormone, and follicle-stimulating hormone in men and women using the AxSYM assay system." Clin. Biochem. 38, 175–179 (2005).
    * Verdonk, S. J. E. et al. "Estradiol reference intervals in women during the menstrual cycle, postmenopausal women and men using an LC-MS/MS method." Clin. Chim. Acta 495, 198–204 (2019).
  * Estradiol and Progesterone values:
    * Stricker, R. et al. Establishment of detailed reference values for luteinizing hormone, follicle stimulating hormone, estradiol, and progesterone during different phases of the menstrual cycle on the Abbott ARCHITECT analyzer. Clin. Chem. Lab. Med. 44, 883–887 (2006).
 * **model_validation**: directory containing each subject's shank IMU (x.csv) and ground-truth respirometry (y.csv)
   * This dataset is from Slade, P. et al. "Sensing leg movement enhances wearable monitoring of energy expenditure." Nature Communications, 12.1 (2021)

