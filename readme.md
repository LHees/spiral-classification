## Readme
The program expects a folder "train_data" and a folder "test_data" that both contain spiral drawing images named in the format of "control_dynamic12.tiff".

To run five-fold cross validation on the training data, run "main.py". This includes a call to the data augmentation code, which can also be run directly from "augmentation.py". To train and evaluate a model on the test data, run "final_train.py".