#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Chapter-7/ex-amr_mimo_adversarial_retraining.py

Adapted for Chapter 7: Adaptive Modulation Recognition in MIMO Systems
Using dataset from local files in ./Nt4Nr2/
Implements Adversarial Retraining instead of Defensive Distillation
"""

import os
import numpy as np
import pandas as pd
import scipy.io as scio
import tensorflow as tf
from tensorflow.keras import layers, models, Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import logging
from datetime import datetime
import random
from tqdm.keras import TqdmCallback
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.basic_iterative_method import basic_iterative_method
from cleverhans.tf2.attacks.momentum_iterative_method import momentum_iterative_method
import uuid

# Set up logging
def setup_logging(chapter_folder):
    log_dir = os.path.join(chapter_folder, "Logs_AdvRetrain")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"execution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("Logging setup complete. Log file: %s", log_file)
    return logger

# Constants
CHAPTER_FOLDER = './Chapter-7'
DATA_FOLDER = os.path.join(CHAPTER_FOLDER, "Nt4Nr2")
MODEL_FOLDER = os.path.join(CHAPTER_FOLDER, "Saved_Models_AdvRetrain")
PLOTS_FOLDER = os.path.join(CHAPTER_FOLDER, "Plots_AdvRetrain")
TABLES_FOLDER = os.path.join(CHAPTER_FOLDER, "Tables_AdvRetrain")
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(PLOTS_FOLDER, exist_ok=True)
os.makedirs(TABLES_FOLDER, exist_ok=True)

UNDEFENDED_MODEL_FILE = os.path.join(MODEL_FOLDER, "lstm_model_undefended.keras")
DEFENDED_MODEL_FILE = os.path.join(MODEL_FOLDER, "lstm_model_defended.keras")
UNDEFENDED_WEIGHTS_FILE = os.path.join(MODEL_FOLDER, "weights_undefended.keras")
DEFENDED_WEIGHTS_FILE = os.path.join(MODEL_FOLDER, "weights_defended.keras")

# Hyperparameters
INPUT_SHAPE = [128, 2]
CLASSES = 6
BATCH_SIZE = 2048
EPOCHS = 30000
PATIENCE = 50
DROPOUT_RATE = 0.2
CLASSES_LIST = ['2PSK', 'QPSK', '8PSK', '16QAM', '64QAM', '128QAM']
SNR_RANGE = range(-10, 21)
L = 500
SNR_NUM = 31
NUM_MOD = 6
ADV_EPSILON = 0.01  # Epsilon for adversarial training

# Initialize logging
logger = setup_logging(CHAPTER_FOLDER)

# Set environment variables
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

###########################################
# Adversarial Retraining Helper Functions #
###########################################

def generate_adversarial_batch(model, X_batch, Y_batch, epsilon=ADV_EPSILON):
    """Generate adversarial examples for a batch using FGSM."""
    X_batch = tf.convert_to_tensor(X_batch, dtype=tf.float32)
    X_adv = fast_gradient_method(
        model, X_batch, eps=epsilon, norm=np.inf,
        clip_min=X_batch.numpy().min(), clip_max=X_batch.numpy().max(), targeted=False
    )
    return X_adv.numpy(), Y_batch

def train_defended_model(X_train, Y_train, X_val, Y_val, attack='FGSM'):
    """Train a defended model using adversarial retraining."""
    logger.info("Training defended model with adversarial retraining (%s)...", attack)
    model = build_lstm(weights=None)
    
    callbacks = [
        ModelCheckpoint(DEFENDED_WEIGHTS_FILE, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=0, patience=5, min_lr=1e-7),
        EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=1, mode='auto', restore_best_weights=True),
        TqdmCallback(verbose=1, desc="Defended Model Training")
    ]
    
    # Custom training loop to include adversarial examples
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    
    # Prepare datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(10000).batch(BATCH_SIZE)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(BATCH_SIZE)
    
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

    # Load undefended model weights if available
    if os.path.exists(UNDEFENDED_WEIGHTS_FILE):
        model.load_weights(UNDEFENDED_WEIGHTS_FILE)
        logger.info("Loaded undefended model weights from %s", UNDEFENDED_WEIGHTS_FILE)
    
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        epoch_acc = 0.0
        total_batches = 0
        
        for X_batch, Y_batch in tqdm(train_dataset, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            # Train on clean examples
            with tf.GradientTape() as tape:
                preds = model(X_batch, training=True)
                loss = loss_fn(Y_batch, preds)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            # Train on adversarial examples
            X_adv, Y_adv = generate_adversarial_batch(model, X_batch.numpy(), Y_batch.numpy())
            with tf.GradientTape() as tape:
                preds_adv = model(X_adv, training=True)
                loss_adv = loss_fn(Y_adv, preds_adv)
            gradients = tape.gradient(loss_adv, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            # Update metrics
            epoch_loss += (loss.numpy() + loss_adv.numpy()) / 2
            epoch_acc += tf.keras.metrics.categorical_accuracy(Y_batch, preds).numpy().mean()
            total_batches += 1
        
        # Validation
        val_loss = 0.0
        val_acc = 0.0
        val_batches = 0
        for X_val_batch, Y_val_batch in val_dataset:
            val_preds = model(X_val_batch, training=False)
            val_loss += loss_fn(Y_val_batch, val_preds).numpy()
            val_acc += tf.keras.metrics.categorical_accuracy(Y_val_batch, val_preds).numpy().mean()
            val_batches += 1
        
        # Log epoch metrics
        epoch_loss /= total_batches
        epoch_acc /= total_batches
        val_loss /= val_batches
        val_acc /= val_batches
        
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        
        logger.info("Epoch %d: loss=%.4f, acc=%.4f, val_loss=%.4f, val_acc=%.4f",
                    epoch+1, epoch_loss, epoch_acc, val_loss, val_acc)
        
        # Check early stopping
        if len(history['val_loss']) > PATIENCE and min(history['val_loss'][:-PATIENCE]) < min(history['val_loss'][-PATIENCE:]):
            logger.info("Early stopping at epoch %d", epoch+1)
            model.load_weights(DEFENDED_WEIGHTS_FILE)
            break
        
        # Save best weights
        if len(history['val_loss']) == 1 or val_loss < min(history['val_loss'][:-1]):
            model.save_weights(DEFENDED_WEIGHTS_FILE)
    
    model.save(DEFENDED_MODEL_FILE)
    logger.info("Defended model saved to %s", DEFENDED_MODEL_FILE)
    return model, history

###########################################
# Model and Data Functions                #
###########################################

def build_lstm(weights=None, input_shape=INPUT_SHAPE, classes=CLASSES):
    logger.info("Building LSTM model...")
    if weights is not None and not os.path.exists(weights):
        raise ValueError(f"Weights file {weights} does not exist.")
    
    input_layer = layers.Input(input_shape, name='input1')
    x = layers.LSTM(units=128, return_sequences=True)(input_layer)
    x = layers.LSTM(units=128)(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    output = layers.Dense(classes, activation='softmax', name='softmax')(x)
    
    model = Model(inputs=input_layer, outputs=output)
    
    if weights is not None:
        model.load_weights(weights)
        logger.info("Loaded weights from %s", weights)
    
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    return model

def load_data():
    logger.info("Loading and concatenating dataset...")
    data_files = [
        'data2psk.mat', 'dataqpsk.mat', 'data8psk.mat', 
        'data16qam.mat', 'data64qam.mat', 'data128qam.mat'
    ]
    label_files = [
        'label2psk.mat', 'labelqpsk.mat', 'label8psk.mat', 
        'label16qam.mat', 'label64qam.mat', 'label128qam.mat'
    ]
    snr_files = [
        'snr2psk.mat', 'snrqpsk.mat', 'snr8psk.mat', 
        'snr16qam.mat', 'snr64qam.mat', 'snr128qam.mat'
    ]
    
    dataset = []
    labels = []
    snrs = []
    
    for d_file, l_file, s_file in zip(data_files, label_files, snr_files):
        data = scio.loadmat(os.path.join(DATA_FOLDER, d_file))['data_save']
        label = scio.loadmat(os.path.join(DATA_FOLDER, l_file))['label_save']
        snr = scio.loadmat(os.path.join(DATA_FOLDER, s_file))['snr_save']
        dataset.append(data)
        labels.append(label)
        snrs.append(snr)
    
    dataset = np.concatenate(dataset)
    labels = np.concatenate(labels)
    snrs = np.concatenate(snrs)
    
    logger.info("Dataset shape: %s, Labels shape: %s, SNRs shape: %s", dataset.shape, labels.shape, snrs.shape)
    return dataset, labels, snrs

def split_data(dataset, labels, snrs):
    logger.info("Splitting dataset into train and test sets...")
    train_idx = []
    test_idx = []
    a = 0
    
    for j in range(NUM_MOD):
        for i in range(SNR_NUM):
            indices = range(a * L, (a + 1) * L)
            train_samples = np.random.choice(indices, size=400, replace=False)
            test_samples = list(set(indices) - set(train_samples))
            train_idx.extend(train_samples)
            test_idx.extend(test_samples)
            a += 1
    
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)
    
    X_train = dataset[train_idx]
    X_test = dataset[test_idx]
    
    X_train = np.swapaxes(X_train, 1, 2)
    X_test = np.swapaxes(X_test, 1, 2)
    
    Y_train = labels[train_idx]
    Y_test = labels[test_idx]
    snrs_test = snrs[test_idx]
    
    logger.info("Train Shape: %s, Test Shape: %s", X_train.shape, X_test.shape)
    return X_train, X_test, Y_train, Y_test, snrs_test

###########################################
# Plotting and Evaluation Functions       #
###########################################

def plot_history(history, model_type='undefended'):
    logger.info("Plotting %s training history...", model_type)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(history['loss'] if isinstance(history, dict) else history.epoch, 
             history['loss'] if isinstance(history, dict) else history.history['loss'], 
             label='Train Loss')
    ax1.plot(history['val_loss'] if isinstance(history, dict) else history.epoch, 
             history['val_loss'] if isinstance(history, dict) else history.history['val_loss'], 
             label='Val Loss')
    ax1.set_title(f'{model_type.capitalize()} Training Loss Performance')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history['accuracy'] if isinstance(history, dict) else history.epoch, 
             history['accuracy'] if isinstance(history, dict) else history.history['accuracy'], 
             label='Train Accuracy')
    ax2.plot(history['val_accuracy'] if isinstance(history, dict) else history.epoch, 
             history['val_accuracy'] if isinstance(history, dict) else history.history['val_accuracy'], 
             label='Val Accuracy')
    ax2.set_title(f'{model_type.capitalize()} Training Accuracy Performance')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_FOLDER, f"training_history_{model_type}.pdf"), bbox_inches='tight')
    plt.close()
    logger.info("%s training history plot saved", model_type.capitalize())

def calculate_confusion_matrix(Y, Y_hat, classes):
    n_classes = classes
    conf = np.zeros([n_classes, n_classes])
    confnorm = np.zeros([n_classes, n_classes])
    
    for k in range(Y.shape[0]):
        i = np.argmax(Y[k, :])
        j = np.argmax(Y_hat[k, :])
        conf[i, j] += 1
    
    for i in range(n_classes):
        confnorm[i, :] = conf[i, :] / np.sum(conf[i, :]) if np.sum(conf[i, :]) > 0 else 0
    
    right = np.sum(np.diag(conf))
    wrong = np.sum(conf) - right
    return confnorm, right, wrong

def plot_confusion_matrix_sklearn(y_true, y_pred, classes, title='Confusion Matrix', save_filename=None):
    y_true_idx = np.argmax(y_true, axis=1)
    y_pred_idx = np.argmax(y_pred, axis=1)
    
    cm = confusion_matrix(y_true_idx, y_pred_idx)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_filename:
        plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    plt.close()

def train_undefended_model(X_train, X_val, Y_train, Y_val):
    logger.info("Training undefended LSTM model...")
    model = build_lstm()
    
    callbacks = [
        ModelCheckpoint(UNDEFENDED_WEIGHTS_FILE, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=0, patience=5, min_lr=1e-7),
        EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=1, mode='auto', restore_best_weights=True),
        TqdmCallback(verbose=1, desc="Undefended Model Training")
    ]
    
    history = model.fit(
        X_train, Y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=0,
        validation_data=(X_val, Y_val),
        callbacks=callbacks
    )
    
    model.save(UNDEFENDED_MODEL_FILE)
    logger.info("Undefended model saved to %s", UNDEFENDED_MODEL_FILE)
    
    plot_history(history, 'undefended')
    return model, history

def evaluate_and_predict(model, X_test, Y_test, snrs_test, model_type='undefended'):
    logger.info("Evaluating %s model on test set...", model_type)
    score = model.evaluate(X_test, Y_test, verbose=0, batch_size=BATCH_SIZE)
    logger.info("%s Test Loss: %.4f, Test Accuracy: %.4f", model_type.capitalize(), score[0], score[1])
    
    logger.info("Generating predictions on clean test examples for %s model...", model_type)
    weights_file = UNDEFENDED_WEIGHTS_FILE if model_type == 'undefended' else DEFENDED_WEIGHTS_FILE
    model.load_weights(weights_file)
    test_Y_hat = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    
    plot_confusion_matrix_sklearn(Y_test, test_Y_hat, classes=CLASSES_LIST, 
                                 title=f"Total Confusion Matrix ({model_type.capitalize()})",
                                 save_filename=os.path.join(PLOTS_FOLDER, f"total_confusion_{model_type}.pdf"))
    
    acc = {}
    acc_mod_snr = np.zeros((CLASSES, SNR_NUM))
    snr_idx = 0
    for snr in tqdm(SNR_RANGE, desc=f"Evaluating {model_type} per SNR", leave=False):
        test_SNRs = snrs_test.reshape(-1)
        test_X_i = X_test[np.where(test_SNRs == snr)]
        test_Y_i = Y_test[np.where(test_SNRs == snr)]
        if len(test_X_i) > 0:
            test_Y_i_hat = model.predict(test_X_i, verbose=0)
            confnorm_i, cor, ncor = calculate_confusion_matrix(test_Y_i, test_Y_i_hat, CLASSES)
            acc[snr] = 1.0 * cor / (cor + ncor) if (cor + ncor) > 0 else 0
            with open(os.path.join(TABLES_FOLDER, f'accuracy_per_snr_{model_type}.csv'), 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([snr, acc[snr]])
            plot_confusion_matrix_sklearn(test_Y_i, test_Y_i_hat, classes=CLASSES_LIST, 
                                         title=f"Confusion Matrix ({model_type.capitalize()}, SNR={snr})",
                                         save_filename=os.path.join(PLOTS_FOLDER, f"confusion_snr_{snr}_{model_type}.pdf"))
            acc_mod_snr[:, snr_idx] = np.round(np.diag(confnorm_i) / np.sum(confnorm_i, axis=1), 3)
            snr_idx += 1
    
    df_acc_mod_snr = pd.DataFrame(acc_mod_snr, index=CLASSES_LIST, columns=[f"SNR={snr}" for snr in SNR_RANGE])
    with open(os.path.join(TABLES_FOLDER, f"acc_mod_snr_{model_type}.tex"), 'w') as f:
        f.write(df_acc_mod_snr.to_latex(float_format="%.3f"))
    logger.info("%s accuracy per modulation and SNR saved as LaTeX", model_type.capitalize())
    
    return acc, acc_mod_snr

def generate_adversarial_examples(model, X_test, eps_val, attack):
    test_input = X_test.astype(np.float32)
    if attack == 'FGSM':
        mal_input = fast_gradient_method(model, test_input, eps=eps_val/1.0, norm=np.inf,
                                         clip_min=test_input.min(), clip_max=test_input.max(), targeted=False)
    elif attack == 'BIM':
        mal_input = basic_iterative_method(model, test_input, eps=eps_val, eps_iter=eps_val/50.0, nb_iter=1000,
                                           norm=np.inf, targeted=False)
    elif attack == 'MIM':
        mal_input = momentum_iterative_method(model, test_input, eps=eps_val, eps_iter=eps_val/50.0, nb_iter=1000,
                                              norm=np.inf, clip_min=test_input.min(), clip_max=test_input.max(), targeted=False)
    elif attack == 'PGD':
        mal_input = projected_gradient_descent(model, test_input, eps=eps_val, eps_iter=eps_val/50.0, nb_iter=1000,
                                               norm=np.inf, clip_min=test_input.min(), clip_max=test_input.max(), targeted=False)
    else:
        raise ValueError(f"Unsupported attack type: {attack}")
    return mal_input.numpy()

def evaluate_adversarial_attacks(model, X_test, Y_test, epsilons, attack='FGSM', model_type='undefended'):
    num_examples = 5
    idx = np.random.choice(X_test.shape[0], num_examples, replace=False)
    X_test_sample = X_test[idx]
    Y_test_sample = Y_test[idx]
    
    accuracies = {}
    for eps in tqdm(epsilons, desc=f"Evaluating {attack} adversarial attacks ({model_type})", leave=False):
        X_adv = generate_adversarial_examples(model, X_test_sample, eps, attack)
        score = model.evaluate(X_adv, Y_test_sample, verbose=0, batch_size=BATCH_SIZE)
        accuracies[eps] = score[1]
    
    df_adv = pd.DataFrame(list(accuracies.items()), columns=["Epsilon", "Accuracy"])
    table_path = os.path.join(TABLES_FOLDER, f"adv_accuracy_{attack}_{model_type}.tex")
    with open(table_path, "w") as f:
        f.write(df_adv.to_latex(index=False, float_format="%.4f"))
    logger.info("%s adversarial accuracy table saved to %s", model_type.capitalize(), table_path)
    
    return accuracies, df_adv

def aggregate_adv_results(adv_results_dict, attack_methods, model_type='undefended'):
    summary_df = pd.DataFrame()
    for attack in attack_methods:
        _, df_adv = adv_results_dict[attack]
        df_adv = df_adv.set_index("Epsilon")
        summary_df[attack] = df_adv["Accuracy"]
    summary_table_path = os.path.join(TABLES_FOLDER, f"aggregate_adv_accuracy_{model_type}.tex")
    with open(summary_table_path, "w") as f:
        f.write(summary_df.to_latex(float_format="%.4f"))
    logger.info("%s aggregate adversarial accuracy table saved to %s", model_type.capitalize(), summary_table_path)
    return summary_df

def plot_aggregate_comparison(agg_undefended_df, agg_defended_df):
    plt.figure(figsize=(10, 6))
    for attack in agg_undefended_df.columns:
        plt.plot(agg_undefended_df.index, agg_undefended_df[attack], marker='o', linestyle='--', 
                 label=f"Undefended {attack}")
        plt.plot(agg_defended_df.index, agg_defended_df[attack], marker='s', linestyle='-', 
                 label=f"Defended {attack}")
    plt.xlabel("Epsilon")
    plt.ylabel("Adversarial Accuracy")
    plt.title("Aggregate Adversarial Accuracy Comparison")
    plt.legend()
    plt.grid(True)
    agg_plot_path = os.path.join(PLOTS_FOLDER, "aggregate_adv_comparison.pdf")
    plt.savefig(agg_plot_path, bbox_inches="tight", dpi=300)
    plt.close()
    logger.info("Aggregate adversarial comparison plot saved to %s", agg_plot_path)

def calculate_mse_difference(model, X_clean, X_adv, Y_true):
    """Calculate the MSE difference between adversarial and clean predictions relative to ground truth."""
    # Get predictions
    Y_pred_clean = model.predict(X_clean, batch_size=2048, verbose=0)
    Y_pred_adv = model.predict(X_adv, batch_size=2048, verbose=0)
    
    # Calculate MSE for clean and adversarial predictions
    mse_clean = np.mean((Y_true - Y_pred_clean) ** 2, axis=1)
    mse_adv = np.mean((Y_true - Y_pred_adv) ** 2, axis=1)
    
    # Compute the difference
    mse_diff = np.abs(mse_adv - mse_clean)
    return np.mean(mse_diff)  # Return average MSE difference

def plot_attack_success_ratio(model, X_test, Y_test, attack_methods, epsilons, model_type='undefended'):
    """Plot MSE-based attack success ratio for each attack method across epsilon values."""
    logger.info("Plotting MSE-based attack success ratio for %s model...", model_type)
    
    # Ensure attack_methods is a flat list of strings
    if isinstance(attack_methods, (list, np.ndarray)):
        # Flatten the list if it contains nested lists
        attack_methods = [item[0] if isinstance(item, (list, np.ndarray)) and len(item) == 1 else item for item in attack_methods]
        # Convert to list if it's a NumPy array
        if isinstance(attack_methods, np.ndarray):
            attack_methods = attack_methods.tolist()
    
    # Verify that all elements are strings
    if not all(isinstance(attack, str) for attack in attack_methods):
        raise ValueError(f"attack_methods must contain only strings, got: {attack_methods}")
    
    # Prepare data
    mse_diffs = {attack: [] for attack in attack_methods}
    num_examples = 50
    idx = np.random.choice(X_test.shape[0], num_examples, replace=False)
    X_test_sample = X_test[idx]
    Y_test_sample = Y_test[idx]
    
    for attack in tqdm(attack_methods, desc=f"Plotting {model_type} attack success ratio", leave=False): # Loop over attack_methods:
        for eps in tqdm(epsilons, desc=f"Epsilon ({model_type}, for the attack: {attack})", leave=False): # Loop over epsilons:
            X_adv = generate_adversarial_examples(model, X_test_sample, eps, attack)
            mse_diff = calculate_mse_difference(model, X_test_sample, X_adv, Y_test_sample)
            mse_diffs[attack].append(mse_diff)
    
    # Plot settings
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.2
    indices = np.arange(len(attack_methods))
    colors = sns.color_palette("viridis", len(epsilons)) if model_type == 'undefended' else sns.color_palette("magma", len(epsilons))
    hatch = '' if model_type == 'undefended' else '//'
    
    # Plot bars
    for i, eps in enumerate(epsilons):
        ratios = [mse_diffs[attack][i] for attack in attack_methods]
        ax.bar(indices + i * bar_width, ratios, bar_width, label=f'ε={eps}', color=colors[i], hatch=hatch)
    
    # Customize plot
    ax.set_xlabel('Attack Method', fontsize=16)
    # set ylabel fontsize 16
    ax.tick_params(axis='y', labelsize=16)
    ax.set_ylabel('MSE Difference (Adv - Clean)', fontsize=16)
    # ax.set_title(f'Attack Success Ratio (MSE Difference) by Attack Method ({model_type.capitalize()} Model)')
    ax.set_xticks(indices + bar_width * (len(epsilons) - 1) / 2)
    ax.set_xticklabels(attack_methods)
    ax.legend(title='Epsilon', fontsize=14)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    # set ylim to 0.0 to 0.85
    #ax.set_ylim(0.0, 0.1)
    
    # Save plot
    plot_path = os.path.join(PLOTS_FOLDER, f"mse_attack_success_ratio_{model_type}.pdf")
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    logger.info("%s MSE attack success ratio plot saved to %s", model_type.capitalize(), plot_path)

def create_adv_performance_table_by_snr(model, X_test, Y_test, snrs_test, epsilons, attack='FGSM', model_type='undefended'):
    rows = []
    snrs_test_1d = snrs_test.reshape(-1)
    for snr in tqdm(sorted(np.unique(snrs_test_1d)), desc=f"SNR ({model_type}), Attack: {attack}", leave=False):
        indices = np.where(snrs_test_1d == snr)[0]
        if len(indices) == 0:
            continue
        X_snr = X_test[indices]
        Y_snr = Y_test[indices]
        clean_score = model.evaluate(X_snr, Y_snr, verbose=0, batch_size=BATCH_SIZE)[1]
        row = {"SNR": snr, "Clean": clean_score}
        for eps in tqdm(epsilons, desc=f"Epsilon ({model_type})", leave=False):
            X_adv = generate_adversarial_examples(model, X_snr, eps, attack)
            adv_score = model.evaluate(X_adv, Y_snr, verbose=0, batch_size=BATCH_SIZE)[1]
            row[f"Adv_eps_{eps}"] = adv_score
        rows.append(row)
    df = pd.DataFrame(rows).set_index("SNR")
    table_path = os.path.join(TABLES_FOLDER, f"{model_type}_adv_performance_{attack}.tex")
    with open(table_path, "w") as f:
        f.write(df.to_latex(float_format="%.4f"))
    logger.info("%s adversarial performance table saved for %s to %s", model_type.capitalize(), attack, table_path)
    return df

###########################################
# Main Execution                          #
###########################################

def main(train_undefended=True, train_defended=True):
    logger.info("Starting Chapter 7 adversarial retraining experiment...")
    
    # Load and split data
    dataset, labels, snrs = load_data()
    X_train, X_test, Y_train, Y_test, snrs_test = split_data(dataset, labels, snrs)
    
    # Split test set for validation
    split_point = X_test.shape[0] // 2
    X_val, X_test = X_test[:split_point], X_test[split_point:]
    Y_val, Y_test = Y_test[:split_point], Y_test[split_point:]
    snrs_val, snrs_test = snrs_test[:split_point], snrs_test[split_point:]
    
    # Train or load undefended model
    if train_undefended:
        undefended_model, undefended_history = train_undefended_model(X_train, X_val, Y_train, Y_val)
    else:
        undefended_model = tf.keras.models.load_model(UNDEFENDED_MODEL_FILE)
        logger.info("Undefended model loaded from %s", UNDEFENDED_MODEL_FILE)
    
    # Train or load defended model
    if train_defended:
        defended_model, defended_history = train_defended_model(X_train, Y_train, X_val, Y_val, attack='FGSM')
    else:
        defended_model = tf.keras.models.load_model(DEFENDED_MODEL_FILE)
        logger.info("Defended model loaded from %s", DEFENDED_MODEL_FILE)
    
    # Evaluate adversarial attacks
    epsilons = [0.005, 0.01, 0.02, 0.05, 0.1]
    attack_methods = ['FGSM', 'BIM', 'MIM', 'PGD']

    plot_attack_success_ratio(undefended_model, X_test, Y_test, attack_methods, epsilons, model_type='undefended')
    plot_attack_success_ratio(defended_model, X_test, Y_test, attack_methods, epsilons, model_type='defended')
    
    # Undefended model adversarial evaluation
    undefended_adv_results = {}
    for attack in tqdm(attack_methods, desc="Evaluating undefended adversarial attacks"):
        adv_acc, df_adv = evaluate_adversarial_attacks(
            undefended_model, X_test, Y_test, epsilons, attack=attack, model_type='undefended')
        undefended_adv_results[attack] = (adv_acc, df_adv)
    
    # Defended model adversarial evaluation
    defended_adv_results = {}
    for attack in tqdm(attack_methods, desc="Evaluating defended adversarial attacks"):
        adv_acc, df_adv = evaluate_adversarial_attacks(
            defended_model, X_test, Y_test, epsilons, attack=attack, model_type='defended')
        defended_adv_results[attack] = (adv_acc, df_adv)

    
    # Create performance tables by SNR
    undefended_adv_tables = {}
    defended_adv_tables = {}
    for attack in tqdm(attack_methods, desc="Creating adversarial performance tables"):
        df_undefended_adv = create_adv_performance_table_by_snr(
            undefended_model, X_test, Y_test, snrs_test, epsilons, attack=attack, model_type='undefended')
        undefended_adv_tables[attack] = df_undefended_adv
        
        df_defended_adv = create_adv_performance_table_by_snr(
            defended_model, X_test, Y_test, snrs_test, epsilons, attack=attack, model_type='defended')
        defended_adv_tables[attack] = df_defended_adv
    
    logger.info("Chapter 7 adversarial retraining experiment completed successfully")

if __name__ == "__main__":
    main(train_undefended=False, train_defended=False)