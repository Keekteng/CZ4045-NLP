# CZ4045-NLP

1. Helper functions in part_1_utils.py are used by Part_1.ipynb only

2. Part_1.ipynb TLDR 
    - Organised according to question format in Assignment docs
    - For Qns 1.1 and 1.2 Simply run the notebook in sequence to replicate the results
    - Qns 1.3
        - First Section is for Defining Model Architecture --> CNN-BLSTM
        - NOTE that "Tuning" section of notebook takes a few days to run
            - Provided search space for selected hyperparameters
            - Used Tree Parzen Estimator(optimization algorithm) and Hyperband Pruning (pruning method)
            - Made use of optuna visualisation package to further guide selection of best model
        - Training Best Model Section
            - Apply the best hyperparameter settings based on findings from Tuning section
            - Apply early stopping mechanism to stop training once f1 score for dev set stops increasing for certain number of epoch
            - Save the best model weights (highest f1 score dev) in the directory /model_weights
        - Results section
            - Load the Weight of the best model and Predict on Test set
            - Total Run Time section shows the total training time --> calculated by summing up time taken for each epoch till early stopping
            - Plotted the dev f1 score against epoch until early stopping
