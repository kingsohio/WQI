# WQI
Predictive modeling that forecast Water Quality Index for the city of Greensboro Division of Water Resources

## Load libraries

- Header names are called for later use in the program. Load and frame dataset along with default header.

### Run Descriptive Statistics of raw data for data insight and understanding for further operations (such as cleaning and dimensioning)
- summarize distribution of each attributes
- correlation
  
  ### Unimodal Data Visualizations 
- histograms
- density
- box and whisker plots
  
  ### Multimodal Data Visualizations
- scatter plot matrix
- correlation matrix

### Clean dataset by removing empty cells for better visualization
- Peek into clean data

### Normalize dataset about a mean point for good data population distribution
- Peek into data

### Split-out Training and Testing Datasets in ratio 4:1, Training:Testing at random seed value of 7 to initiate the program
- Peek into X and Y Training datasets 

### Data pre-processing 
  
  ### Features scaling
Scale data features to standardize the independent features present in the data in a fixed range in order to handle highly varying magnitudes features units 

### identify and remove outliers in the training dataset
- Peek into data

### Apply PCA to reduce the numbers of features from 23 to the important components that affect WQI prediction in accordance with model
- Peek into new data dimension

## Carry out Baseline Algorithm Evaluation within 10-folds iteration while using a negative scoring for model accuracy check

### Preparing algorithms to be tested among six models (3 non-linear and 3 linear models for unbiase prediction)

### Evaluate each model in turn and check results

### Compare Algorithms

### Standardize the dataset within each model pipelines and compare algorithms again

### Use KNN Algorithm tuning to improve power of model

### Display Statistics

## Carry out hybrid method known as Ensembles methods to compare model outcome with the Baseline Algorithm method
- Print stats    

### Compare Algorithms

### Tune scaled GBM to improve model power

### Display Statistics    

### Finalize model -  prepare the model for prediction

### Transform the validation dataset
