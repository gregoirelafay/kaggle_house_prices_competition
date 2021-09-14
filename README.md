# House Prices - Advanced Regression Techniques

Predict sales prices and practice feature engineering, RFs, and gradient boosting

## Metric

Submissions are evaluated on Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted value and the logarithm of the observed sales price. (Taking logs means that errors in predicting expensive houses and cheap houses will affect the result equally.)

## How to use the trainer package

### Download the Data

You can use the makefile to download the data from Kaggle into the raw_data folder.

```bash
make download_files
```

### Import the trainer

First you need to import the trainer.

```py
# Import
from houses_trainer.trainer import Trainer
```

### Initialize the Trainer

You have to initialize the trainer by choosing a predefined model.

```py
# Instanciate trainer
trainer_ridge = Trainer(model="ridge")
```

### Load Data and build pipeline

Then you can load the data and build the pipeline.

```py
# Load data
trainer_ridge.load_data()
# Build Pipeline
trainer_ridge.build_pipeline(feature_cutoff_percentage=75)
```

### Cross validate and predict

Once the pipeline is built, you can cross validate your model, and make a prediction using the test dataset.
The prediction is saved in a csv file located in the submission folder.

```py
# Cross Validate
trainer_ridge.cross_validate(cv=5)
# Prediction
trainer_ridge.predict()
```

### Submit your results to kaggle

Finally you may submit your results to the Kaggle competition

```py
# Submit results
trainer_ridge.submit()
```
