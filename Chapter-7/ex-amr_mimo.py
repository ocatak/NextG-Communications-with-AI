#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Chapter-7/ex-amr_mimo.py

Adapted for Chapter 7: Adaptive Modulation Recognition in MIMO Systems
Using dataset from local files in ./Nt4Nr2/
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

# Set up logging
def setup_logging(chapter_folder):
    log_dir = os.path.join(chapter_folder, "Logs")
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
MODEL_FOLDER = os.path.join(CHAPTER_FOLDER, "Saved_Models")
PLOTS_FOLDER = os.path.join(CHAPTER_FOLDER, "Plots")
TABLES_FOLDER = os.path.join(CHAPTER_FOLDER, "Tables")
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(PLOTS_FOLDER, exist_ok=True)
os.makedirs(TABLES_FOLDER, exist_ok=True)

MODEL_FILE = os.path.join(MODEL_FOLDER, "lstm_model.keras")
WEIGHTS_FILE = os.path.join(MODEL_FOLDER, "weights.keras")

# Hyperparameters
INPUT_SHAPE = [128, 2]
CLASSES = 6
BATCH_SIZE = 2048
EPOCHS = 10000
PATIENCE = 50
DROPOUT_RATE = 0.2
CLASSES_LIST = ['2PSK', 'QPSK', '8PSK', '16QAM', '64QAM', '128QAM']
SNR_RANGE = range(-10, 21)      # 31 SNR levels
L = 500  # Samples per SNR per modulation
SNR_NUM = 31  # Number of SNR levels
NUM_MOD = 6   # Number of modulation types

# Initialize logging
logger = setup_logging(CHAPTER_FOLDER)

# Set environment variables
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

###########################################
# Defensive Distillation Helper Functions #
###########################################

def get_soft_labels(preds, temperature):
    """
    Given teacher predictions (softmax outputs), returns softened probabilities
    using temperature scaling: p_soft = p^(1/T) / sum(p^(1/T)).
    """
    preds = np.clip(preds, 1e-7, 1.0)
    preds_power = np.power(preds, 1.0 / temperature)
    return preds_power / np.sum(preds_power, axis=1, keepdims=True)

def train_distilled_model(X_train, Y_train, X_val, Y_val, teacher_model, temperature=20):
    """
    Trains a distilled (defended) model using soft labels computed from the teacher model.
    The student model is built with the same architecture as the teacher.
    """
    logger.info("Computing teacher predictions for distillation...")
    teacher_preds_train = teacher_model.predict(X_train, batch_size=BATCH_SIZE, verbose=0)
    teacher_preds_val = teacher_model.predict(X_val, batch_size=BATCH_SIZE, verbose=0)
    soft_labels_train = get_soft_labels(teacher_preds_train, temperature)
    soft_labels_val = get_soft_labels(teacher_preds_val, temperature)
    
    logger.info("Training distilled (student) model with temperature=%.1f...", temperature)
    student_model = build_lstm()  # same architecture as teacher
    # Compile and train student on soft targets
    student_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = student_model.fit(
        X_train, soft_labels_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,  # Using fewer epochs for distillation
        validation_data=(X_val, soft_labels_val),
        callbacks=[EarlyStopping(monitor='val_loss', patience=int(PATIENCE*4), verbose=1, restore_best_weights=True), TqdmCallback(verbose=1, desc="Distillation Training")],
        verbose=0
    )
    
    distilled_model_path = os.path.join(MODEL_FOLDER, "lstm_model_distilled.keras")
    student_model.save(distilled_model_path)
    logger.info("Distilled model saved to %s", distilled_model_path)
    return student_model, history

###########################################
# Model and Data Functions                #
###########################################

# LSTM Model Definition
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

# Load and concatenate dataset
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

# Split dataset (train/test only)
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

def plot_history(history):
    logger.info("Plotting training history...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(history.epoch, history.history['loss'], label='Train Loss')
    ax1.plot(history.epoch, history.history['val_loss'], label='Val Loss')
    ax1.set_title('Training Loss Performance')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history.epoch, history.history['accuracy'], label='Train Accuracy')
    ax2.plot(history.epoch, history.history['val_accuracy'], label='Val Accuracy')
    ax2.set_title('Training Accuracy Performance')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_FOLDER, "training_history.pdf"), bbox_inches='tight')
    plt.close()
    logger.info("Training history plot saved")

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
    """
    Computes and plots the normalized confusion matrix using scikit‑learn's confusion_matrix.
    y_true and y_pred are expected to be one‑hot encoded.
    """
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

def train_model(X_train, X_val, Y_train, Y_val):
    logger.info("Training LSTM model...")
    model = build_lstm()
    
    callbacks = [
        ModelCheckpoint(WEIGHTS_FILE, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=0, patience=5, min_lr=1e-7),
        EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=1, mode='auto', restore_best_weights=True),
        TqdmCallback(verbose=1, desc="Model Training")
    ]
    
    history = model.fit(
        X_train, Y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=0,
        validation_data=(X_val, Y_val),
        callbacks=callbacks
    )
    
    model.save(MODEL_FILE)
    logger.info("Teacher model saved to %s", MODEL_FILE)
    
    plot_history(history)
    return model, history

def evaluate_and_predict(model, X_test, Y_test, snrs_test):
    logger.info("Evaluating model on test set...")
    score = model.evaluate(X_test, Y_test, verbose=0, batch_size=BATCH_SIZE)
    logger.info("Test Loss: %.4f, Test Accuracy: %.4f", score[0], score[1])
    
    logger.info("Generating predictions on clean test examples...")
    model.load_weights(WEIGHTS_FILE)
    test_Y_hat = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    
    plot_confusion_matrix_sklearn(Y_test, test_Y_hat, classes=CLASSES_LIST, title="Total Confusion Matrix",
                                  save_filename=os.path.join(PLOTS_FOLDER, "total_confusion.pdf"))
    
    acc = {}
    acc_mod_snr = np.zeros((CLASSES, SNR_NUM))
    snr_idx = 0
    for snr in tqdm(SNR_RANGE, desc="Evaluating per SNR", leave=False):
        test_SNRs = snrs_test.reshape(-1)
        test_X_i = X_test[np.where(test_SNRs == snr)]
        test_Y_i = Y_test[np.where(test_SNRs == snr)]
        if len(test_X_i) > 0:
            test_Y_i_hat = model.predict(test_X_i, verbose=0)
            confnorm_i, cor, ncor = calculate_confusion_matrix(test_Y_i, test_Y_i_hat, CLASSES)
            acc[snr] = 1.0 * cor / (cor + ncor) if (cor + ncor) > 0 else 0
            # logger.info("SNR %d: Accuracy = %.4f", snr, acc[snr])
            with open(os.path.join(TABLES_FOLDER, 'accuracy_per_snr.csv'), 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([snr, acc[snr]])
            plot_confusion_matrix_sklearn(test_Y_i, test_Y_i_hat, classes=CLASSES_LIST, 
                                          title=f"Confusion Matrix (SNR={snr})",
                                          save_filename=os.path.join(PLOTS_FOLDER, f"confusion_snr_{snr}.pdf"))
            acc_mod_snr[:, snr_idx] = np.round(np.diag(confnorm_i) / np.sum(confnorm_i, axis=1), 3)
            snr_idx += 1
    
    df_acc_mod_snr = pd.DataFrame(acc_mod_snr, index=CLASSES_LIST, columns=[f"SNR={snr}" for snr in SNR_RANGE])
    with open(os.path.join(TABLES_FOLDER, "acc_mod_snr.tex"), 'w') as f:
        f.write(df_acc_mod_snr.to_latex(float_format="%.3f"))
    logger.info("Accuracy per modulation and SNR saved as LaTeX")
    
    return acc, acc_mod_snr

def generate_adversarial_examples(model, X_test, eps_val, attack):
    test_input = X_test.astype(np.float32)
    if attack == 'FGSM':
        mal_input = fast_gradient_method(model, test_input, eps=eps_val/20.0, norm=np.inf,
                                         clip_min=test_input.min(), clip_max=test_input.max(), targeted=False)
    elif attack == 'BIM':
        mal_input = basic_iterative_method(model, test_input, eps=eps_val, eps_iter=0.001, nb_iter=200,
                                           norm=np.inf, targeted=False)
    elif attack == 'MIM':
        mal_input = momentum_iterative_method(model, test_input, eps=eps_val, eps_iter=0.001, nb_iter=200,
                                              norm=np.inf, clip_min=test_input.min(), clip_max=test_input.max(), targeted=False)
    elif attack == 'PGD':
        mal_input = projected_gradient_descent(model, test_input, eps=eps_val, eps_iter=0.001, nb_iter=200,
                                               norm=np.inf, clip_min=test_input.min(), clip_max=test_input.max(), targeted=False)
    else:
        raise ValueError(f"Unsupported attack type: {attack}")
    return mal_input.numpy()

def evaluate_adversarial_attacks(model, X_test, Y_test, epsilons, attack='FGSM'):
    """
    Evaluates model performance on adversarial examples generated by a selected attack
    for a range of epsilon values using a random sample of 50 test examples.
    Generates a LaTeX table and a single line plot for adversarial accuracy.
    """
    num_examples = 50
    idx = np.random.choice(X_test.shape[0], num_examples, replace=False)
    X_test_sample = X_test[idx]
    Y_test_sample = Y_test[idx]
    
    accuracies = {}
    for eps in tqdm(epsilons, desc=f"Evaluating {attack} adversarial attacks", leave=False):
        X_adv = generate_adversarial_examples(model, X_test_sample, eps, attack)
        score = model.evaluate(X_adv, Y_test_sample, verbose=0, batch_size=BATCH_SIZE)
        accuracies[eps] = score[1]
        #logger.info("Adversarial Attack (%s) with epsilon=%.4f: Accuracy = %.4f", attack, eps, score[1])
    
    df_adv = pd.DataFrame(list(accuracies.items()), columns=["Epsilon", "Accuracy"])
    table_path = os.path.join(TABLES_FOLDER, f"adv_accuracy_{attack}.tex")
    with open(table_path, "w") as f:
        f.write(df_adv.to_latex(index=False, float_format="%.4f"))
    logger.info("Adversarial accuracy table saved to %s", table_path)
    
    plt.figure(figsize=(6,4))
    plt.plot(df_adv["Epsilon"], df_adv["Accuracy"], marker="o", linestyle='-')
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.title(f"Adversarial Accuracy vs. Epsilon ({attack})")
    plt.grid(True)
    plot_path = os.path.join(PLOTS_FOLDER, f"adv_accuracy_plot_{attack}.pdf")
    plt.savefig(plot_path, bbox_inches="tight", dpi=300)
    plt.close()
    logger.info("Adversarial accuracy plot saved to %s", plot_path)
    
    return accuracies, df_adv


def aggregate_adv_results(adv_results_dict, attack_methods):
    # Create a DataFrame with epsilon as index and columns for each attack's accuracy
    summary_df = pd.DataFrame()
    for attack in attack_methods:
        _, df_adv = adv_results_dict[attack]
        df_adv = df_adv.set_index("Epsilon")
        summary_df[attack] = df_adv["Accuracy"]
    summary_table_path = os.path.join(TABLES_FOLDER, "aggregate_adv_accuracy.tex")
    with open(summary_table_path, "w") as f:
        f.write(summary_df.to_latex(float_format="%.4f"))
    logger.info("Aggregate adversarial accuracy table saved to %s", summary_table_path)
    return summary_df

def aggregate_adv_results(adv_results_dict, attack_methods):
    """
    Aggregates adversarial accuracy DataFrames from different attack methods into one summary table.
    """
    summary_df = pd.DataFrame()
    for attack in attack_methods:
        _, df_adv = adv_results_dict[attack]
        df_adv = df_adv.set_index("Epsilon")
        summary_df[attack] = df_adv["Accuracy"]
    summary_table_path = os.path.join(TABLES_FOLDER, "aggregate_adv_accuracy.tex")
    with open(summary_table_path, "w") as f:
        f.write(summary_df.to_latex(float_format="%.4f"))
    logger.info("Aggregate adversarial accuracy table saved to %s", summary_table_path)
    return summary_df

def plot_aggregate_comparison(agg_teacher_df, agg_distilled_df):
    """
    Generates a single plot that compares the teacher (undefended) and distilled (defended)
    models across attack methods (columns) and epsilon values (index).
    """
    plt.figure(figsize=(10, 6))
    for attack in agg_teacher_df.columns:
        plt.plot(agg_teacher_df.index, agg_teacher_df[attack], marker='o', linestyle='--', 
                 label=f"Teacher {attack}")
        plt.plot(agg_distilled_df.index, agg_distilled_df[attack], marker='s', linestyle='-', 
                 label=f"Distilled {attack}")
    plt.xlabel("Epsilon")
    plt.ylabel("Adversarial Accuracy")
    plt.title("Aggregate Adversarial Accuracy Comparison")
    plt.legend()
    plt.grid(True)
    agg_plot_path = os.path.join(PLOTS_FOLDER, "aggregate_adv_comparison.pdf")
    plt.savefig(agg_plot_path, bbox_inches="tight", dpi=300)
    plt.close()
    logger.info("Aggregate adversarial comparison plot saved to %s", agg_plot_path)

def create_adv_performance_table_by_snr(model, X_test, Y_test, snrs_test, epsilons, attack='FGSM'):
    """
    For a given model and attack method, this function creates a table (DataFrame) 
    that for each SNR value shows the clean accuracy (without attack) and the adversarial accuracy
    (with attack) for each epsilon value.
    
    Returns a DataFrame with SNR as the index and columns:
      - 'Clean' for clean accuracy,
      - 'Adv_eps_{eps}' for the adversarial accuracy at each epsilon.
    """
    rows = []
    snrs_test_1d = snrs_test.reshape(-1)
    for snr in tqdm(sorted(np.unique(snrs_test_1d)), desc="SNR", leave=False):
        # Get indices for the current SNR
        indices = np.where(snrs_test_1d == snr)[0]
        if len(indices) == 0:
            continue
        X_snr = X_test[indices]
        Y_snr = Y_test[indices]
        # Compute clean accuracy on this subset
        clean_score = model.evaluate(X_snr, Y_snr, verbose=0, batch_size=BATCH_SIZE)[1]
        row = {"SNR": snr, "Clean": clean_score}
        # For each epsilon value, compute adversarial accuracy
        for eps in tqdm(epsilons, desc="Epsilon", leave=False):
            # You can either use the full SNR subset or a random sample:
            # Here we use all available examples for a given SNR.
            X_adv = generate_adversarial_examples(model, X_snr, eps, attack)
            adv_score = model.evaluate(X_adv, Y_snr, verbose=0, batch_size=BATCH_SIZE)[1]
            row[f"Adv_eps_{eps}"] = adv_score
        rows.append(row)
    df = pd.DataFrame(rows).set_index("SNR")
    return df


###########################################
# Main Execution                          #
###########################################

def main(train_model_param=True, train_distillation=True):
    logger.info("Starting Chapter 7 experiment...")
    
    # Load and split data
    dataset, labels, snrs = load_data()
    X_train, X_test, Y_train, Y_test, snrs_test = split_data(dataset, labels, snrs)
    
    # Train or load the teacher (undefended) model
    if train_model_param:
        teacher_model, history = train_model(X_train, X_test, Y_train, Y_test)
    else:
        teacher_model = tf.keras.models.load_model(MODEL_FILE)
        logger.info("Teacher model loaded from %s", MODEL_FILE)
    
    # Evaluate teacher model on clean test data.
    acc_clean, acc_mod_snr = evaluate_and_predict(teacher_model, X_test, Y_test, snrs_test)
    
    # Evaluate adversarial attacks for teacher model (per attack method)
    epsilons = [0.005, 0.01, 0.02, 0.05, 0.1]
    attack_methods = ['FGSM', 'BIM', 'MIM', 'PGD']
    teacher_adv_results = {}
    for attack in tqdm(attack_methods, desc="Evaluating teacher adversarial attacks"):
        adv_acc, df_adv = evaluate_adversarial_attacks(teacher_model, X_test, Y_test, epsilons, attack=attack)
        teacher_adv_results[attack] = (adv_acc, df_adv)
    
    # Train or load the distilled (defended) model.
    if train_distillation:
        # For distillation, use half of the test set as a stand-in validation set.
        split_point = X_test.shape[0] // 2
        X_train_distill, X_val_distill = X_test[:split_point], X_test[split_point:]
        Y_train_distill, Y_val_distill = Y_test[:split_point], Y_test[split_point:]
        distilled_model, distill_history = train_distilled_model(X_train_distill, Y_train_distill, 
                                                                 X_val_distill, Y_val_distill,
                                                                 teacher_model, temperature=10)
    else:
        distilled_model = tf.keras.models.load_model(os.path.join(MODEL_FOLDER, "lstm_model_distilled.keras"))
        logger.info("Distilled model loaded.")
    
    # Evaluate adversarial attacks for distilled model.
    distilled_adv_results = {}
    for attack in tqdm(attack_methods, desc="Evaluating distilled adversarial attacks"):
        adv_acc_dist, df_adv_dist = evaluate_adversarial_attacks(distilled_model, X_test, Y_test, epsilons, attack=attack)
        distilled_adv_results[attack] = (adv_acc_dist, df_adv_dist)
    
    # Aggregate the results and generate summary tables/plots.
    agg_teacher_df = aggregate_adv_results(teacher_adv_results, attack_methods)
    agg_distilled_df = aggregate_adv_results(distilled_adv_results, attack_methods)
    plot_aggregate_comparison(agg_teacher_df, agg_distilled_df)

    # Specify epsilon values and attack methods
    epsilons = [0.005, 0.01, 0.02, 0.05, 0.1]
    attack_methods = ['FGSM', 'BIM', 'MIM', 'PGD']

    # For teacher (undefended) model, create a table for each attack and save as LaTeX.
    teacher_adv_tables = {}
    for attack in tqdm(attack_methods, desc="Creating teacher adversarial performance tables"): #attack_methods:
        df_teacher_adv = create_adv_performance_table_by_snr(teacher_model, X_train, Y_train, snrs_test, epsilons, attack=attack)
        teacher_adv_tables[attack] = df_teacher_adv
        table_path = os.path.join(TABLES_FOLDER, f"teacher_adv_performance_{attack}.tex")
        with open(table_path, "w") as f:
            f.write(df_teacher_adv.to_latex(float_format="%.4f"))
        # logger.info("Teacher adversarial performance table saved for %s to %s", attack, table_path)

    # For distilled (defended) model, similarly:
    distilled_adv_tables = {}
    for attack in attack_methods:
        df_distilled_adv = create_adv_performance_table_by_snr(distilled_model, X_train, Y_train, snrs_test, epsilons, attack=attack)
        distilled_adv_tables[attack] = df_distilled_adv
        table_path = os.path.join(TABLES_FOLDER, f"distilled_adv_performance_{attack}.tex")
        with open(table_path, "w") as f:
            f.write(df_distilled_adv.to_latex(float_format="%.4f"))
        logger.info("Distilled adversarial performance table saved for %s to %s", attack, table_path)

    
    logger.info("Chapter 7 experiment completed successfully")


if __name__ == "__main__":
    main(train_model_param=False, train_distillation=False)
