import os
import datetime
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

# model functions
def load_model(model, checkpoint_path):
    """ Load model from previous work"""
    model.load_weights(checkpoint_path)
    return model
  
  
def model_checkpoint_constructor(checkpoint_name): 
    """ Define the callback to save the model weights """
    checkpoint_path = os.path.join(r"..\models\checkpoints", checkpoint_name)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor="accuracy",
        save_best_only=True,
        verbose=1
    )
    
    return checkpoint_callback

def get_optimizer(learning_rate = 0.001, decay_steps = 1000, decay_rate = 1, staircase_flag = False):
  """ Learning rate schedule"""
  lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  learning_rate,
  decay_steps=decay_steps,
  decay_rate=decay_rate,
  staircase=staircase_flag)

  return tf.keras.optimizers.Adam(lr_schedule)

def compile_fit_model(model): 
    """_summary_

    Args:
        model (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy',
            tf.keras.metrics.Precision(), 
            tf.keras.metrics.TruePositives(thresholds=0),
            tf.keras.metrics.TrueNegatives(thresholds=0),
            tf.keras.metrics.FalseNegatives(thresholds=0),
            tf.keras.metrics.FalsePositives(thresholds=0)])

    hist = model.fit(
        train_ds,
        verbose = 1,
        validation_data=val_ds,
        epochs=3
    )
    
    return model, hist

# Setting the learning rate to reduce gradually over the training period
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    0.001,
    decay_steps=100,
    decay_rate=1,
    staircase=False)


def get_optimizer():
    return tf.keras.optimizers.Adam(lr_schedule)
  
# Model evaluation functions

def create_tf_board(log_name): 
    """ Create logs for tensorboard """
    
    path = f'../../models/logs/{log_name}/'
    
    # if os.path.exists(path):
    #     # Clear any logs from previous runs
    #     shutil.rmtree(path)
    
    log_dir = path + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    return tensorboard_callback
  

def save_weights(model, model_name): 
    """ saving weights of the model to the models folder """
    model.save_weights(f'./models/model_weights/{model_name}/{model_name}_weights', save_format='tf')
    
def save_model(model, model_name): 
    """ Save the model for use later"""
    saved_model_path = f"/models/saved_{model_name}"
    tf.keras.models.save_model(model, saved_model_path)

# Confusion matrix for binary classification
def plot_cm(labels, predictions, p=0.5):
  cm = confusion_matrix(labels, predictions > p)
  plt.figure(figsize=(5,5))
  sns.heatmap(cm, annot=True, fmt="d")
  plt.title('Confusion matrix @{:.2f}'.format(p))
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')

  print('True Negatives: ', cm[0][0])
  print('False Positives: ', cm[0][1])
  print('False Negatives: ', cm[1][0])
  print('True Positives: ', cm[1][1])
  print('Total Malignant Cases: ', np.sum(cm[1]))
  
  
def multi_label_cm(labels, predictions):
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    results = {
        'True Negatives': [cm[0][0]],
        'False Positives':[cm[0][1]],
        'False Negatives': [cm[1][0]],
        'True Positives': [cm[1][1]],
        'Total Correct Cases': [cm[0][0] + cm[1][1]],
        'Total Cases': [np.sum(cm)]
    }

    df = pd.DataFrame.from_dict(results)
    
    display(df)
    return df
  
def plot_hist(hist): 
    """ View into model history. 
    hist = model.fit() results
    
    """
    
    # get history
    hist_dict = hist.history
    hist_df = pd.DataFrame.from_dict(hist_dict)

    # Use index to get Epochs
    hist_df.reset_index(inplace = True)
    hist_df.rename({'index': 'Epoch'}, axis = 1, inplace = True)
    # hist_df = pd.melt(hist_df, id_vars = 'index')

    # Create Subsets
    accuracy = hist_df.loc[:, ['Epoch','accuracy', 'val_accuracy']]
    loss = hist_df.loc[:, ['Epoch', 'loss', 'val_loss']]

    # Flatten dfs
    accuracy = pd.melt(accuracy, id_vars = 'Epoch')
    loss = pd.melt(loss, id_vars = 'Epoch')

    # flatten df to plot
    # hist_melt = pd.melt(hist_df, id_vars = 'index').rename({'index': 'Epoch'}, axis = 1)

    # call regplot on each axes
    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=False)

    sns.lineplot(data = accuracy, ax = ax1, x = 'Epoch', y = 'value', hue = 'variable').set_title('Accuracy')
    sns.lineplot(data = loss, ax = ax2, x = 'Epoch', y = 'value', hue = 'variable').set_title('Loss')




