# Marine Power Binary Classifier
This repositary contains scripts to run a binary classification on marine power data.

## Scripts
The repositary consists of four main folders:
data - where the original data is kept.
notebooks - notebooks for EDA and running the models.
outputs - fitted models and performance metrics are saved.
src - python scripts containing functions for the analysis.

/src
---- data_cleaning.py- contains functions for cleaning the data. This includes removing outliers, removing and imputing missing values, converting to the correct data types and one hot encoding.
---- evaluation.py- contains functions which evaluate the performance of the models.
---- modelling.py - contains functions for each model, which fit and measure model performance.

## To Run
To create an environment which contains the required packages run the following:
    conda env create --name environment_name -f environment.yml
Once this has completed, run:
    conda activate environment_name

The initial EDA can be ran by running the eda.ipynb file within the notebooks folder.
The analysis can be ran by running the pipeline.ipynb file within the notebooks folder.
