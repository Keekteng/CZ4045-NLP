# CZ4045-NLP

Note: Helper functions required in Part_1.ipynb are in part_1_utils.py only while functions required in Part_2.ipynb are in part_2_utils.py only.

PART 1:
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

3. model.png is an image of the model architecture for Qns 1 generated from Part_1.ipynb

PART 2:
This part implements the sentence-level categorisation using question classification. 

Following are the instructions and explanations of sample output after running the different sections of the code:
    1) Reading dataset, re-classifying classes and splitting to train and dev set
        - Read the train and test dataset using pandas library
        - Drop the unnecessary columns and combine 2 classes randomly to form one new label
        - Output is two new dataframes, each for train and class dataset
        - Then, split the train dataset into train and dev set (size=500)
    2) Neural Network implementation
        - Firstly, process the input sentences by splitting them into words, which the output will be a list of list of words
        - Next, find the longest sentence from the train, dev and test dataset to determine length of padding
        - Create a labelset and wordset to store all unique words and labels, and use word2vec to convert them into embeddings
        - Output is a dictionary called 'word2Idx' that maps each word to a unique index
        - Padding and unknown words are then handled, before converting the final wordEmbeddings into an array (output)
        - Next, create a matrix to convert the list of sentences into a matrix format (using output of wordList and word2Idx), apply padding and convert it into PyTorch tensors
        - Output is the subsequent tensors for trainSentences, devSentences and testSentences
        - Convert the labels from the train, dev and test set to PyTorch tensors as well
        - Finally, create a dataloader for train and dev set to be used for model training using PyTorch library
        - Output will be two dataloaders: train_dataloader and dev_dataloader
    2.1) Bi-LSTM implementation
        - The BiLSTModel class is defined, with the functions to initialise it, and implement the forward pass function
        - Important parameters that are considered when running the code are: hidden_size, num_classes, aggregation method (max vs avg) and dropout_rate
        - The softmax outputs are returned as the model's predictions in the forward function
        - Back function is not implemented as PyTorch will automatically compute the gradient of the loss and update the model parameters
    2.2) Model training and initialisation
        - Create a function for training and evaluating the model by calculating and storing the training and validation accuracy at each epoch
        - Output returns the model used, the training and validation accuracy and best model 
        - Avg and max pooling aggregation methods are implemented for the Bi-LSTM model, with pre-defined loss function (CrossEntropyLoss), optimizer (Adam) and model parameters (hidden_size, learning rate, no.of epochs, etc.)
        - Output is the training and dev accuracy at each epoch, as well as the plot of the accuracies against epochs
    2.3) Model evaluation
        - Create test_dataloader for evaluation of the trained model
        - Output is the test accuracy based on the best_max_pool_lstm model
    2.4) Saving best model weights
        - Save the best model as a python object using pickle library
        - The model is also loaded into loaded_model which outputs the details of the BiLSTM model used
        - Due to size of the weights of the model, it is not uploaded into the zip file containing the source code