from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, OneClassSVM
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve,
    precision_recall_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Recall, Precision, AUC
from tensorflow.keras.regularizers import l1
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


##################################################################################
##################################################################################
##################################################################################

def bcc_occ_models(df, classification):
    """
    Train and evaluate models for one-class classification (OCC) and binary class classification (BCC).

    Parameters:
        df (pd.DataFrame): The dataset containing features, labels, and additional columns.
        classification (str): The type of classification to perform. Can be 'occ' for one-class classification or 'bcc' for binary class classification.
    
    Returns:
        None: The function prints the evaluation results and plots relevant metrics and visualizations.

    This function performs the following operations:
    - For OCC:
        1. Trains an OC-SVM model for anomaly detection.
        2. Trains an Autoencoder for anomaly detection.
        3. Evaluates the models and displays the confusion matrix, classification report, and ROC curve.
    - For BCC:
        1. Trains an SVC model for binary classification.
        2. Trains a custom Deep Learning model for binary classification.
        3. Evaluates the models and displays the confusion matrix, classification report, and ROC curve.

    Note: The function contains commented-out sections for PyOD and Variational Autoencoder (VAE) models. These sections can be uncommented and used if needed.
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

    if classification=='occ':
      X_normal, X_train, X_val = prepare_for_train(X,y)

      print('############################')
      print('# OC-SVM - Anomaly Detection')
      print('############################')
      oc_svm = OneClassSVM(kernel='rbf', gamma='auto')
      oc_svm.fit(X_train)
      y_pred = oc_svm.predict(X)
      y_pred = np.where(y_pred == 1, 0, 1)
      conf_matrix_and_classif_report(y, y_pred)
      decision_scores = oc_svm.decision_function(X)
      plot_roc_curve(y, decision_scores, 'OC-SVM - Anomaly Detection')




      print('#################################')
      print('# AutoEncoder - Anomaly Detection')
      print('#################################')
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

      test_reconstruction_error = np.mean(np.power(X - autoencoder.predict(X), 2), axis=1)
      y_pred = [1 if e > threshold else 0 for e in test_reconstruction_error]
      conf_matrix = confusion_matrix(y, y_pred)
      precision, recall, _ = precision_recall_curve(y, test_reconstruction_error)
      conf_matrix_and_classif_report(y, y_pred)
      plot_roc_curve(y, test_reconstruction_error, 'AutoEncoder - Anomaly Detection')

    if classification=='bcc':

      print('#############################')
      print('# SVC - Binary Classification')
      print('#############################')
      X_train, X_val, y_train, y_val = prepare_for_train(X, y, mode=classification)

      svc = SVC(kernel='rbf', gamma='auto', class_weight='balanced')
      svc.fit(X_train, y_train)

      y_pred = svc.predict(X_val)
      conf_matrix_and_classif_report(y_val, y_pred)
      decision_scores = svc.decision_function(X_val)
      plot_roc_curve(y_val, decision_scores, 'SVC - Binary Classification')


      print('###############################################')
      print('# Custom Binary DL Model- Binary Classification')
      print('###############################################')

      X_train, X_temp, y_train, y_temp = prepare_for_train(X, y, mode=classification)
      X_val, X_test, y_val, y_test = prepare_for_train(X_temp, y_temp, mode=classification,split=0.5)

      input_shape = X_train.shape[1]
      learning_rate = 0.001
      dropout_rate = 0.3
      batch_size = 64
      epochs = 50

      model = Sequential([
          Dense(256, activation='relu', input_shape=(input_shape,)),# kernel_regularizer=tf.keras.regularizers.l2(0.01)),
          BatchNormalization(),
          Dropout(dropout_rate),
          Dense(128, activation='relu'),# kernel_regularizer=tf.keras.regularizers.l2(0.01)),
          BatchNormalization(),
          Dropout(dropout_rate),
          Dense(64, activation='relu'),# kernel_regularizer=tf.keras.regularizers.l2(0.01)),
          BatchNormalization(),
          Dropout(dropout_rate),
          Dense(1, activation='sigmoid')
      ])

      optimizer = Adam(learning_rate=learning_rate)
      model.compile(optimizer=optimizer,
                    loss='binary_crossentropy',
                    metrics=[
                        Recall(),
                        Precision(),
                        AUC()
                    ])

      early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

      class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
      class_weight_dict = dict(enumerate(class_weights))

      history = model.fit(
          X_train, y_train,
          epochs=epochs,
          batch_size=batch_size,
          validation_data=(X_val, y_val),
          class_weight=class_weight_dict,
          callbacks=[early_stopping]
      )

      loss, recall, precision, auca = model.evaluate(X_test, y_test)
      print(f"Test Recall: {recall}, Precision: {precision}, AUC: {auca}")

      y_pred = model.predict(X_test)
      y_pred = (y_pred > 0.5).astype(int)

      conf_matrix_and_classif_report(y_test, y_pred)

    
# if classification=='od':
#   print('############################')
#   print('# PyOD - Outlier Detection')
#   print('############################')

#   contamination = 0.3
#   epochs = 30
#   clf = AutoEncoder(epochs=epochs, contamination=contamination)#, hidden_neurons=hn)

#   X_train, X_val, y_train, y_val = prepare_for_train(X,y,mode=classification)

#   clf.fit(X_train)

#   y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
#   y_train_scores = clf.decision_scores_  # raw outlier scores

#   # get the prediction on the test data
#   y_test_pred = clf.predict(X_val)  # outlier labels (0 or 1)
#   y_test_scores = clf.decision_function(X_val)  # outlier scores

#   print("\nOn Training Data:")
#   evaluate_print(clf, y_train, y_train_scores)
#   print("\nOn Test Data:")
#   evaluate_print(clf, y_val, y_test_scores)
    
# if classification=='vae':
#     print('#############################################')
#     print('# Variational AutoEncoder - Anomaly Detection')
#     print('#############################################')

#     X_normal, X_train, X_val = prepare_for_train(X,y)
#     def sampling(args):
#         z_mean, z_log_var = args
#         batch = K.shape(z_mean)[0]
#         dim = K.int_shape(z_mean)[1]
#         epsilon = K.random_normal(shape=(batch, dim))
#         return z_mean + K.exp(0.5 * z_log_var) * epsilon

#     # Network parameters
#     input_shape = (X_train.shape[1], )
#     intermediate_dim = 1024  # Adjusted to a higher dimension
#     latent_dim = 10  # Adjusted latent dimension
#     epochs = 10  # Increased number of epochs
#     batch_size = 16
#     learning_rate = 0.001  # You can experiment with this value

#     # Encoder
#     inputs = Input(shape=input_shape, name='encoder_input')
#     x = Dense(intermediate_dim, activation=LeakyReLU())(inputs)
#     x = BatchNormalization()(x)
#     x = Dropout(0.3)(x)  # Increased dropout rate
#     x = Dense(intermediate_dim // 2, activation=LeakyReLU())(x)
#     z_mean = Dense(latent_dim, name='z_mean')(x)
#     z_log_var = Dense(latent_dim, name='z_log_var')(x)
#     z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
#     encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

#     # Decoder
#     latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
#     x = Dense(intermediate_dim // 2, activation=LeakyReLU())(latent_inputs)
#     x = BatchNormalization()(x)
#     x = Dropout(0.3)(x)
#     x = Dense(intermediate_dim, activation=LeakyReLU())(x)
#     outputs = Dense(input_shape[0], activation='sigmoid')(x)
#     decoder = Model(latent_inputs, outputs, name='decoder')

#     # VAE Model
#     outputs = decoder(encoder(inputs)[2])
#     vae = Model(inputs, outputs, name='vae_mlp')

#     # Loss function
#     reconstruction_loss = mse(inputs, outputs)
#     reconstruction_loss *= input_shape[0]
#     kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
#     kl_loss = K.sum(kl_loss, axis=-1)
#     kl_loss *= -0.5
#     vae_loss = K.mean(reconstruction_loss + kl_loss)
#     vae.add_loss(vae_loss)
#     vae.compile(optimizer=Adam(learning_rate=learning_rate))

#     # Train the model
#     vae.fit(X_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, None))

#     # Calculate reconstruction error on the normal dataset
#     reconstruction_error_normal = mse(X_normal, vae.predict(X_normal)).numpy()

#     # Determine a threshold for anomaly detection
#     threshold = np.percentile(reconstruction_error_normal, 95) # Adjust as needed

#     # Assuming X is your complete dataset
#     reconstruction_error_X = mse(X, vae.predict(X)).numpy()

#     # Detect anomalies in the entire dataset
#     anomalies_X = reconstruction_error_X > threshold

#     print(confusion_matrix(y, anomalies_X))
#     print(classification_report(y, anomalies_X))

#     # ROC Curve and AUC
#     fpr, tpr, thresholds = roc_curve(y, reconstruction_error_X)
#     roc_auc = auc(fpr, tpr)

#     plt.figure()
#     plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
#     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic')
#     plt.legend(loc="lower right")
#     plt.show()

#     # Precision-Recall Curve
#     precision, recall, _ = precision_recall_curve(y, reconstruction_error_X)

#     plt.figure()
#     plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.title('Precision-Recall Curve')
#     plt.legend(loc="lower left")
#     plt.show()

#     # Plotting Training and Validation Loss
#     plt.figure(figsize=(8, 4))
#     plt.plot(history.history['loss'], label='Training Loss')
#     plt.plot(history.history['val_loss'], label='Validation Loss')
#     plt.title('Training and Validation Loss')
#     plt.ylabel('Loss')
#     plt.xlabel('Epoch')
#     plt.legend()
#     plt.show()

#     # Plotting the Histogram of Reconstruction Error
#     plt.figure(figsize=(8, 4))
#     plt.hist(test_reconstruction_error, bins=50)
#     plt.axvline(threshold, color='r', linestyle='dashed', linewidth=2)
#     plt.title('Reconstruction Error Histogram')
#     plt.xlabel('Reconstruction Error')
#     plt.ylabel('Frequency')
#     plt.show()
