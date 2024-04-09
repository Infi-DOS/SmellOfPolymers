import gc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve,
    roc_auc_score, precision_score, recall_score, accuracy_score
)
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.metrics import Recall, Precision, AUC, Metric
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from tqdm.notebook import tqdm
from kerastuner import HyperModel, Hyperband


######################################################################
######################################################################
######################################################################
# HIGH LEVEL FUSION
######################################################################
######################################################################
######################################################################

def confusion_matrices(clfs, test_data, thresholds):
    """
    Compute confusion matrices for given classifiers, test data, and classification thresholds.
    
    Parameters:
        clfs (list): A list of classifiers (models) to evaluate.
        test_data (list): A list of tuples, where each tuple contains test features and true labels.
        thresholds (list): A list of thresholds for classifying a data point as normal or anomaly.

    Returns:
        list: A list of confusion matrices, one for each classifier.
    """
    confusion_matrices = []
    for index, clf in enumerate(clfs):
        reconstructed = clf.predict(test_data[index][0])
        reconstruction_error = np.mean(np.power(test_data[index][0] - reconstructed, 2), axis=1)
        predictions = [1 if e > thresholds[index] else 0 for e in reconstruction_error]
        cm = confusion_matrix(test_data[index][1], predictions)
        confusion_matrices.append(cm)
    return confusion_matrices


def bayesian_fusion_predict(clfs, datapoints, conf_matrices, thresholds):
    """
    Predict the final class of each data point using Bayesian decision fusion.

    Parameters:
        clfs (list): A list of classifiers (autoencoders) used for prediction.
        datapoints (list): A list of DataFrames containing data for prediction.
        conf_matrices (list): A list of confusion matrices for each classifier.
        thresholds (list): A list of reconstruction error thresholds for each classifier.

    Returns:
        np.array: An array of final predictions (0 or 1) for each data point.
    """
    final_predictions = []

    data_batch_1 = datapoints[0].to_numpy()
    data_batch_2 = datapoints[1].to_numpy()

    reconstructed_1 = clfs[0].predict(data_batch_1)
    reconstructed_2 = clfs[1].predict(data_batch_2)

    for n_datapoint in range(datapoints[0].shape[0]):
        predictions = []
        for index, reconstructed in enumerate([reconstructed_1, reconstructed_2]):
            error = np.mean(np.power(datapoints[index].iloc[n_datapoint,:].to_numpy() - reconstructed[n_datapoint], 2))
            prediction = 1 if error > thresholds[index] else 0
            predictions.append(prediction)

        p_h0 = 0.5
        p_h1 = 0.5

        k = 0
        for conf_mat in conf_matrices:

            p_0_h0 = conf_mat[0][0] / (conf_mat[0][0] + conf_mat[0][1])
            p_1_h0 = 1 - p_0_h0
            p_1_h1 = conf_mat[1][1] / (conf_mat[1][0] + conf_mat[1][1])
            p_0_h1 = 1 - p_1_h1

            if predictions[k] == 0:

                p_h0_0 = (p_0_h0 * p_h0) / (p_0_h0 * p_h0 + p_0_h1 * p_h1)
                p_h1_0 = 1 - p_h0_0

                p_h0 = p_h0_0
                p_h1 = p_h1_0

            else:

                p_h0_1 = (p_1_h0 * p_h0) / (p_1_h0 * p_h0 + p_1_h1 * p_h1)
                p_h1_1 = 1 - p_h0_1

                p_h0 = p_h0_1
                p_h1 = p_h1_1

            k += 1

        if p_h0 > p_h1:
            final_predictions.append(0)
        else:
            final_predictions.append(1)

    return np.array(final_predictions)


def weighted_voting_fusion_autoencoder(clfs, data, thresholds):
    """
    Apply weighted voting fusion for predictions from multiple autoencoders.

    Parameters:
        clfs (list): A list of autoencoder models.
        data (list): A list of DataFrames containing data for prediction.
        thresholds (list): A list of reconstruction error thresholds for each autoencoder.

    Returns:
        np.array: An array of final predictions (0 or 1) for each data point based on weighted voting.
    """
    data_batch_1 = data[0].to_numpy()
    data_batch_2 = data[1].to_numpy()

    reconstructed_batch_1 = clfs[0].predict(data_batch_1)
    reconstructed_batch_2 = clfs[1].predict(data_batch_2)

    errors_1 = np.mean(np.power(data_batch_1 - reconstructed_batch_1, 2), axis=1)
    errors_2 = np.mean(np.power(data_batch_2 - reconstructed_batch_2, 2), axis=1)

    votes_1 = (errors_1 > thresholds[0]).astype(int)
    votes_2 = (errors_2 > thresholds[1]).astype(int)

    weighted_voting = []

    for v1, v2 in zip(votes_1, votes_2):
        final_vote = 1 if v1 + v2 > len(clfs) / 2 else 0
        weighted_voting.append(final_vote)

    return np.array(weighted_voting)



def conf_matrix_aucroc_and_classif_report(true, pred):
    """
    Print the classification report, confusion matrix, and plot ROC curve for the true labels and predictions.

    Parameters:
        true (array-like): The true labels.
        pred (array-like): The predicted labels.

    Returns:
        None: The function prints the classification report, confusion matrix, and plots the ROC curve.
    """
    print('Classification Report:')
    print(classification_report(true, pred))

    print('Confusion Matrix:')
    conf_matrix = confusion_matrix(true, pred)
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Calculate the ROC AUC score
    roc_auc = roc_auc_score(true, pred)
    print(f'ROC AUC Score: {roc_auc:.4f}')
    
    # Calculate the ROC curve
    fpr, tpr, thresholds = roc_curve(true, pred)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

def predict_anomalies(model, data, threshold):
    """
    Predict anomalies based on the reconstruction error of an autoencoder model.

    Parameters:
        model (Model): The trained autoencoder model.
        data (DataFrame or np.array): The data to predict anomalies for.
        threshold (float): The threshold for classifying a data point as an anomaly based on reconstruction error.

    Returns:
        np.array: An array of predictions (0 for normal, 1 for anomaly).
    """
    reconstructed = model.predict(data)
    mse = np.mean(np.power(data - reconstructed, 2), axis=1)
    pred_labels = mse > threshold
    return pred_labels.astype(int)

def conf_matrix_and_classif_report(true,pred):
        print('Classification Report:')
        print(classification_report(true, pred))

        print('Confusion Matrix:')
        conf_matrix = confusion_matrix(true, pred)
        sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion Matrix')
        plt.show()
        
def modelizer(df, classification):
    """
    Train an autoencoder model for anomaly detection on the given dataset.

    Parameters:
        df (pd.DataFrame): The dataset containing features and labels.
        classification (str): The type of classification ('occ', 'bcc', etc.).

    Returns:
        tuple: A tuple containing the trained autoencoder model, feature data, labels, and the calculated threshold for anomaly detection.
    """

    X = df.drop(['SMILES', 'SENTENCE', 'binary', 'multi'], axis=1)
    y = df['binary']

    def prepare_for_train(X,y, mode='occ',split=0.2):
        if mode == 'occ' or mode == 'vae':
            X_normal = X[y == 0]
            X_train, X_val = train_test_split(X_normal, test_size=split, random_state=42)
            return X_normal, X_train, X_val
        if mode == 'bcc' or mode == 'od':
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=split, random_state=42)
            return X_train, X_val, y_train, y_val

    def plot_roc_curve(true_labels, predicted_scores, model_name):
        fpr, tpr, _ = roc_curve(true_labels, predicted_scores)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve for {model_name} (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic - ' + model_name)
        plt.legend(loc="lower right")
        plt.show()

    def conf_matrix_and_classif_report(true,pred):
        print('Classification Report:')
        print(classification_report(true, pred))

        print('Confusion Matrix:')
        conf_matrix = confusion_matrix(true, pred)
        sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion Matrix')
        plt.show()

    print('#################################')
    print('# AutoEncoder - Anomaly Detection')
    print('#################################')
    X_normal, X_train, X_val = prepare_for_train(X,y)
    input_dim = X_train.shape[1]
    learning_rate = 0.001
    batch_size = 64
    epochs = 50

    input_layer = Input(shape=(input_dim,))
    encoder = Dense(512, activation="relu", activity_regularizer=l1(10e-5))(input_layer)
    encoder = BatchNormalization()(encoder)
    encoder = Dense(256, activation="relu")(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = Dense(128, activation="relu")(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = Dense(64, activation="relu")(encoder)
    encoder = BatchNormalization()(encoder)
    decoder = Dense(128, activation="relu")(encoder)
    decoder = BatchNormalization()(decoder)
    decoder = Dense(256, activation='relu')(decoder)
    decoder = BatchNormalization()(decoder)
    decoder = Dense(512, activation='relu')(decoder)
    decoder = BatchNormalization()(decoder)
    decoder = Dense(input_dim, activation='sigmoid')(decoder)

    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = autoencoder.fit(X_train, X_train,
                                epochs=epochs,
                                batch_size=batch_size,
                                shuffle=True,
                                validation_data=(X_val, X_val),
                                callbacks=[early_stopping])

    reconstruction_error = np.mean(np.power(X_train - autoencoder.predict(X_train), 2), axis=1)
    threshold = np.percentile(reconstruction_error, 90)
    # After calculating reconstruction error and threshold
    print("Reconstruction error:", reconstruction_error)
    print("Threshold:", threshold)


    test_reconstruction_error = np.mean(np.power(X - autoencoder.predict(X), 2), axis=1)
    y_pred = [1 if e > threshold else 0 for e in test_reconstruction_error]
    conf_matrix = confusion_matrix(y, y_pred)
    precision, recall, _ = precision_recall_curve(y, test_reconstruction_error)
    conf_matrix_and_classif_report(y, y_pred)
    plot_roc_curve(y, test_reconstruction_error, 'AutoEncoder - Anomaly Detection')
    return autoencoder, X, y, threshold

######################################################################
######################################################################
######################################################################
# KERAS TUNER
######################################################################
######################################################################
######################################################################
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.model_selection import train_test_split
from kerastuner.tuners import Hyperband
from kerastuner import HyperModel
import numpy as np
import os
import datetime
import random
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)
# # class AutoencoderHyperModel(HyperModel):
# #     def __init__(self, input_dim):
# #         self.input_dim = input_dim

# #     def build(self, hp):
# #         input_layer = Input(shape=(self.input_dim,))
# #         x = input_layer

# #         # Encoder
# #         for i in range(hp.Int('num_layers_encoder', 1, 5)):
# #             x = Dense(
# #                 units=hp.Int('units_enc_' + str(i), min_value=32, max_value=512, step=32),
# #                 activation=hp.Choice('act_enc_' + str(i), ['relu', 'tanh', 'sigmoid']),
# #                 activity_regularizer=l1(hp.Float('l1_reg_enc_' + str(i), 1e-5, 1e-2, sampling='LOG'))
# #             )(x)
# #             if hp.Boolean('dropout_enc_' + str(i)):
# #                 x = Dropout(rate=hp.Float('dropout_rate_enc_' + str(i), 0.1, 0.5))(x)

# #         # Decoder
# #         for i in range(hp.Int('num_layers_decoder', 1, 5)):
# #             x = Dense(
# #                 units=hp.Int('units_dec_' + str(i), min_value=32, max_value=512, step=32),
# #                 activation=hp.Choice('act_dec_' + str(i), ['relu', 'tanh', 'sigmoid'])
# #             )(x)
# #             if hp.Boolean('dropout_dec_' + str(i)):
# #                 x = Dropout(rate=hp.Float('dropout_rate_dec_' + str(i), 0.1, 0.5))(x)

# #         output_layer = Dense(self.input_dim, activation='sigmoid')(x)
# #         model = Model(inputs=input_layer, outputs=output_layer)
# #         model.compile(optimizer=Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG')),
# #                       loss='mean_squared_error')
# #         return model

# class AutoencoderHyperModel(HyperModel):
#     def __init__(self, input_dim):
#         self.input_dim = input_dim

#     def build(self, hp):
#         input_layer = Input(shape=(self.input_dim,))
#         x = input_layer

#         # Encoder
#         encoder_layers = hp.Int('num_layers', 1, 6)
#         encoder_units = [hp.Int(f'encoder_units_{i}', min_value=16, max_value=self.input_dim // 2, step=16) for i in range(encoder_layers)]

#         for units in encoder_units:
#             x = Dense(units=units, activation=hp.Choice('encoder_activation', ['relu', 'tanh', 'sigmoid']),
#                       activity_regularizer=l1(hp.Float('l1_reg', 1e-5, 1e-2, sampling='LOG')))(x)
#             if hp.Boolean('encoder_dropout'):
#                 x = Dropout(rate=hp.Float('dropout_rate', 0.1, 0.5))(x)

#         # Decoder
#         for units in reversed(encoder_units):
#             x = Dense(units=units, activation=hp.Choice('decoder_activation', ['relu', 'tanh', 'sigmoid']))(x)
#             if hp.Boolean('decoder_dropout'):
#                 x = Dropout(rate=hp.Float('dropout_rate', 0.1, 0.5))(x)

#         output_layer = Dense(self.input_dim, activation='sigmoid')(x)
#         model = Model(inputs=input_layer, outputs=output_layer)
#         model.compile(optimizer=Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG')),
#                       loss='mean_squared_error')
#         return model
from keras import Input, Model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.regularizers import l1
from keras_tuner import HyperModel

class AutoencoderHyperModel(HyperModel):
    def __init__(self, input_dim):
        self.input_dim = input_dim

    def build(self, hp):
        input_layer = Input(shape=(self.input_dim,))
        x = input_layer

        # Define constraints based on input_dim
        if self.input_dim <= 256:
            layer_constraints = [
                {'min': 64, 'max': 128, 'step': 64},
                {'min': 32, 'max': 64, 'step': 32},
                {'min': 16, 'max': 32, 'step': 16},
                {'min': 8, 'max': 16, 'step': 8}
            ]
        else:  # For larger input_dim
            layer_constraints = [
                {'min': 512, 'max': min(2048, self.input_dim // 2), 'step': 512},
                {'min': 256, 'max': min(1024, self.input_dim // 4), 'step': 256},
                {'min': 128, 'max': min(512, self.input_dim // 8), 'step': 128},
                {'min': 64, 'max': min(256, self.input_dim // 16), 'step': 64},
                {'min': 32, 'max': min(128, self.input_dim // 32), 'step': 32}
            ]

        # Encoder
        encoder_layers = hp.Int('num_layers', 1, len(layer_constraints))
        encoder_units = []  # List to store the units for each layer of the encoder
        for i in range(encoder_layers):
            constraint = layer_constraints[i]
            units = hp.Int(f'units_encoder_{i}', min_value=constraint['min'], max_value=constraint['max'], step=constraint['step'])
            encoder_units.append(units)  # Store the units for the encoder layer
            x = Dense(units=units, activation=hp.Choice('encoder_activation', ['relu', 'tanh', 'sigmoid']),
                      activity_regularizer=l1(hp.Float('l1_reg', 1e-5, 1e-2, sampling='LOG')))(x)
            if hp.Boolean('encoder_dropout'):
                x = Dropout(rate=hp.Float('dropout_rate', 0.1, 0.5))(x)

        # Decoder (mirrored structure of the encoder)
        for units in reversed(encoder_units):
            x = Dense(units=units, activation=hp.Choice('decoder_activation', ['relu', 'tanh', 'sigmoid']))(x)
            if hp.Boolean('decoder_dropout'):
                x = Dropout(rate=hp.Float('dropout_rate', 0.1, 0.5))(x)

        output_layer = Dense(self.input_dim, activation='sigmoid')(x)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG')),
                      loss='mean_squared_error')
        return model



def prepare_for_train(X, y, split=0.2, seed=42):
    X_normal = X[y == 0]
    y_normal = y[y == 0]
    return train_test_split(X_normal, y_normal, test_size=split, random_state=seed)

def calculate_threshold(model, X_train, percentile=95):
    reconstructed = model.predict(X_train)
    mse = np.mean(np.power(X_train - reconstructed, 2), axis=1)
    return np.percentile(mse, percentile)

def run_keras_tuner(df, epochs=10, split=0.2, patience=5, seed=42):
    X = df.drop(['SMILES', 'SENTENCE', 'binary', 'multi'], axis=1)
    y = df['binary']
    X_train, X_val, _, _ = prepare_for_train(X, y, split=split, seed=seed)

    input_dim = X_train.shape[1]
    hypermodel = AutoencoderHyperModel(input_dim=input_dim)

    tuner = Hyperband(
        hypermodel,
        objective='val_loss',
        max_epochs=epochs,
        factor=3,
        directory='keras_tuner_dir',
        project_name='autoencoder_tuning',
        overwrite=True
    )
    # Define your base log path
    base_log_path = '/content/drive/My Drive/Group14/Models/OCC/Logs/'

    # Create a directory for TensorBoard logs with a timestamp
    # log_dir = os.path.join(base_log_path, 'fit', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

    # Check if the log directory exists, if not, create it
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir)

    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    # tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    tuner.search(
        X_train, X_train,
        epochs=epochs,
        validation_data=(X_val, X_val),
        callbacks=[early_stopping],# tensorboard_callback],
        verbose=1
    )

    best_model = tuner.get_best_models(num_models=1)[0]
    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
    threshold = calculate_threshold(best_model, X_train)

    return best_model, best_hyperparameters, X, y, threshold