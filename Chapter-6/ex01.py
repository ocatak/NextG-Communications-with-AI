#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Chapter-6/ex01.py

Adapted for Chapter 6: [Your Chapter Title Here]
Using dataset from [Your Dataset Source Here]
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input  # Add this to imports if not already present
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.linalg import norm
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.keras import TqdmCallback
from tqdm import tqdm
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.basic_iterative_method import basic_iterative_method
from cleverhans.tf2.attacks.momentum_iterative_method import momentum_iterative_method
from itertools import product
from random import shuffle
import logging
from datetime import datetime

# Custom import for defensive distillation (assuming util_defdistill.py is available)
try:
    from util_defdistill import Distiller
except ImportError:
    raise ImportError("Please ensure util_defdistill.py is available in the working directory.")

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
CHAPTER_FOLDER = './Chapter-6'
PROJECT_FOLDER = CHAPTER_FOLDER  # Default local folder
GOOGLE_COLAB = False  # Set to True if running on Google Colab
HYPARAMETER_TUNING = False
TRAIN_MODEL = False

# Directory setup
MODEL_FOLDER = os.path.join(CHAPTER_FOLDER, "Saved_Models")
PLOTS_FOLDER = os.path.join(CHAPTER_FOLDER, "Plots")
TABLES_FOLDER = os.path.join(CHAPTER_FOLDER, "Tables")
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(PLOTS_FOLDER, exist_ok=True)
os.makedirs(TABLES_FOLDER, exist_ok=True)

TEACHER_MODEL_FILE = os.path.join(MODEL_FOLDER, "teacher_model.keras")
STUDENT_MODEL_FILE = os.path.join(MODEL_FOLDER, "student_model.keras")
UNDEFENDED_MODEL_FILE = os.path.join(MODEL_FOLDER, "undefended_model.keras")

# ----- New constants added near the other global constants -----
CLASSIFICATION_MODEL_FILE = os.path.join(MODEL_FOLDER, "classification_model.keras")
DISTILLED_CLASSIFICATION_MODEL_FILE = os.path.join(MODEL_FOLDER, "distilled_classification_model.keras")


# Hyperparameters
EPS_VALUES = [0.01, 0.1, 0.3, 0.5, 0.7, 0.8]
ATTACKS = ['FGSM', 'BIM', 'MIM', 'PGD']
BATCH_SIZE = 4096
EPOCHS = 5000
PATIENCE = 20
TEMPERATURE = 20
STUDENT_MODEL_MULTIPLICATION = 0.5
TEACHER_MODEL_MULTIPLICATION = 1.0
NUM_SAMPLES = 150  # Number of test samples to attack

# Initialize logging
logger = setup_logging(CHAPTER_FOLDER)

# Load dataset
def load_data():
    logger.info("Loading dataset...")
    X = pd.read_csv(os.path.join(PROJECT_FOLDER, 'xtrain_32_10000.csv'), header=None).values
    Y = pd.read_csv(os.path.join(PROJECT_FOLDER, 'ytrain_32_10000.csv'), header=None).values
    M = X.shape[1]
    logger.info("Dataset loaded. X shape: %s, Y shape: %s, M: %d", X.shape, Y.shape, M)
    return X, Y, M

# Split dataset
def split_data(X, Y):
    logger.info("Splitting dataset into train and test sets (67/33 ratio)")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    logger.info("Train Shape: %s, Test Shape: %s", X_train.shape, X_test.shape)
    return X_train, X_test, y_train, y_test

# Build model
def get_model(mult_factor, M, output_dim):
    model = Sequential([
        Dense(int(M * mult_factor), input_dim=M, activation='relu'),
        Dense(int(2 * M * mult_factor), activation='relu'),
        Dense(int(4 * M * mult_factor), activation='relu'),
        Dense(int(4 * M * mult_factor), activation='relu'),
        Dense(output_dim, activation='relu')
    ])
    model.compile(loss='mse', optimizer='sgd', metrics=['mse'])
    return model

# Train undefended model
def train_undefended_model(X_train, y_train, M, output_dim):
    logger.info("Training Undefended Model...")
    model = get_model(1.0, M, output_dim)
    # Explicitly build the model with the input shape
    model.build(input_shape=(None, M))  # None for batch size, M for feature dimension
    early_stopping = EarlyStopping(monitor='val_mse', patience=PATIENCE, verbose=1, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,  # Consider increasing this for better training
        batch_size=BATCH_SIZE,
        verbose=0,
        validation_split=0.2,
        callbacks=[TqdmCallback(verbose=1, desc="Undefended Training"), early_stopping]
    )
    model.save(UNDEFENDED_MODEL_FILE)
    logger.info("Undefended Model saved: %s", UNDEFENDED_MODEL_FILE)
    
    # Plot and save training history
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['mse'], label='Training MSE')
    plt.plot(history.history['val_mse'], label='Validation MSE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    # plt.title('Undefended Model Training History')
    plt.legend()
    # grid minor and major ticks
    plt.minorticks_on()
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.savefig(os.path.join(PLOTS_FOLDER, "undefended_training_history.pdf"), bbox_inches='tight')
    plt.close()
    logger.info("Undefended training history plot saved")
    
    return model

# Convert model for CleverHans compatibility
def convert_to_classification_model(model, X_train, output_dim):
    logger.info("Converting model to CleverHans-compatible classification model using Functional API...")
    
    inputs = tf.keras.Input(shape=(X_train.shape[1],))
    x = Dense(int(X_train.shape[1] * 1.0), activation='relu')(inputs)
    x = Dense(int(2 * X_train.shape[1] * 1.0), activation='relu')(x)
    x = Dense(int(4 * X_train.shape[1] * 1.0), activation='relu')(x)
    x = Dense(int(4 * X_train.shape[1] * 1.0), activation='relu')(x)
    x = Dense(output_dim, activation='relu')(x)
    outputs = Dense(2, activation='softmax')(x)
    
    model_copy = tf.keras.Model(inputs=inputs, outputs=outputs)
    model_copy.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model on dummy labels to initialize it as done previously.
    y_dummy = np.zeros((X_train.shape[0], 2))
    mid_range = int(X_train.shape[0] / 2)
    y_dummy[:mid_range, 0] = 1.0
    
    
    history = model_copy.fit(
        X_train, y_dummy,
        epochs=1,
        batch_size=256,
        verbose=0,
        validation_split=0.33,
        shuffle=True,
        callbacks=[TqdmCallback(verbose=1, desc="Classification Conversion")]
    )

    logger.info("Classification model saved: %s", CLASSIFICATION_MODEL_FILE)
    
    # Save training history plot as before...
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    # plt.title('Classification Model Conversion History')
    plt.legend()
    plt.minorticks_on()
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.savefig(os.path.join(PLOTS_FOLDER, "classification_conversion_history.pdf"), bbox_inches='tight')
    plt.close()
    
    logger.info("Classification conversion history plot saved")
    return model_copy

# Generate adversarial examples
def generate_adversarial_examples(model, X_test, eps_val, attack):
    # logger.info("Generating adversarial examples with %s (epsilon=%f)", attack, eps_val)
    test_input = X_test.astype(np.float32)
    
    if attack == 'FGSM':
        mal_input = fast_gradient_method(model, test_input, eps=eps_val/20.0, norm=np.inf,
                                         clip_min=test_input.min(), clip_max=test_input.max(), targeted=False)
    elif attack == 'BIM':
        mal_input = basic_iterative_method(model, test_input, eps=eps_val, eps_iter=0.01, nb_iter=200,
                                           norm=np.inf, targeted=False)
    elif attack == 'MIM':
        mal_input = momentum_iterative_method(model, test_input, eps=eps_val, eps_iter=0.01, nb_iter=200,norm=np.inf,
                                              clip_min=test_input.min(), clip_max=test_input.max(), targeted=False)
    elif attack == 'PGD':
        mal_input = projected_gradient_descent(model, test_input, eps=eps_val, eps_iter=0.01, nb_iter=200,norm=np.inf,
                                               clip_min=test_input.min(), clip_max=test_input.max(), targeted=False)
    else:
        raise ValueError(f"Unsupported attack type: {attack}")
    
    return mal_input.numpy()

# Evaluate adversarial attacks
def evaluate_attacks(classification_model, undefended_model, X_test, y_test, attacks, eps_vals):
    logger.info("Evaluating adversarial attacks on undefended model...")
    logger.info("Evaluating adversarial attacks on undefended model...")
    # Ensure the model is built by calling it with a sample input
    dummy_input = X_test[:1]  # Use a single sample to define the input shape
    classification_model(dummy_input)  # This builds the model if it hasn't been built
    _ = classification_model.predict(X_test[:1], verbose=0)  # Optional: predict to ensure initialization
    logits_model = tf.keras.Model(classification_model.input, classification_model.output)

    params_list = list(product(eps_vals, attacks))
    shuffle(params_list)
    
    mal_diff_distance_list = []
    real_mse_list = []
    mal_mse_list = []
    mal_predicted_diff_list = []
    attack_name_list = []
    eps_val_list = []
    
    for _ in tqdm(range(NUM_SAMPLES), desc="Processing Test Samples"):
        i = np.random.randint(0, X_test.shape[0])
        test_input = X_test[i:i+1, :]
        real_output = y_test[i:i+1, :]
        
        for eps_val, attack in tqdm(params_list, desc="Evaluating Attacks", leave=False):
            mal_input = generate_adversarial_examples(logits_model, test_input, eps_val, attack)
            
            mal_diff = mal_input - test_input
            mal_diff_distance = norm(mal_diff, ord=np.inf)
            mal_diff_distance_list.append(mal_diff_distance)
            
            test_output = undefended_model.predict(test_input, verbose=0)
            real_mse = mean_squared_error(real_output, test_output)
            real_mse_list.append(real_mse)
            
            mal_output = undefended_model.predict(mal_input, verbose=0)
            mal_mse = mean_squared_error(real_output, mal_output)
            mal_mse_list.append(mal_mse)
            
            mal_predicted_diff = norm(mal_output - test_output, ord=np.inf)
            mal_predicted_diff_list.append(mal_predicted_diff)
            
            attack_name_list.append(attack)
            eps_val_list.append(eps_val)
    
    df_result = pd.DataFrame({
        'Malicious_Distance': mal_diff_distance_list,
        'Real_Predicted_MSE': real_mse_list,
        'Malicious_Predicted_MSE': mal_mse_list,
        'MalOut_RealOut_Diff': mal_predicted_diff_list,
        'Attack': attack_name_list,
        'eps': eps_val_list
    })
    
    # Save table as LaTeX
    with open(os.path.join(TABLES_FOLDER, "undefended_attack_results.tex"), 'w') as f:
        f.write(df_result.to_latex(index=False, float_format="%.4f"))
    logger.info("Undefended attack results saved as LaTeX")
    
    return df_result

# Train defensively distilled models
def train_distilled_models(X_train, X_test, y_train, y_test, M, output_dim):
    logger.info("Training Defensive Distillation Models...")
    
    # Teacher Model
    teacher_model = get_model(TEACHER_MODEL_MULTIPLICATION, M, output_dim)
    early_stopping_teacher = EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=1, restore_best_weights=True, mode='min')
    hist_teacher = teacher_model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=[early_stopping_teacher, TqdmCallback(verbose=1, desc="Teacher Training")],
        validation_data=(X_test, y_test)
    )
    teacher_model.save(TEACHER_MODEL_FILE)
    logger.info("Teacher Model saved: %s", TEACHER_MODEL_FILE)
    
    plt.figure(figsize=(8, 5))
    plt.plot(hist_teacher.history['loss'], label='Training Loss')
    plt.plot(hist_teacher.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.title('Teacher Model Training History')
    plt.legend()
    plt.minorticks_on()
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.savefig(os.path.join(PLOTS_FOLDER, "teacher_training_history.pdf"), bbox_inches='tight')
    plt.close()
    logger.info("Teacher training history plot saved")
    
    # Student Model with Distillation
    student_model = get_model(STUDENT_MODEL_MULTIPLICATION, M, output_dim)
    distiller = Distiller(student=student_model, teacher=teacher_model)
    distiller.compile(
        optimizer='adam',
        metrics=['mse'],
        student_loss_fn=tf.keras.losses.MeanSquaredError(),
        distillation_loss_fn=tf.keras.losses.KLDivergence(),
        alpha=0.1,
        temperature=TEMPERATURE
    )
    early_stopping_distill = EarlyStopping(monitor='val_student_loss', patience=20, verbose=1, restore_best_weights=True, mode='min')
    hist_distill = distiller.fit(
        X_train, y_train,
        epochs=EPOCHS,
        verbose=0,
        callbacks=[early_stopping_distill, TqdmCallback(verbose=1, desc="Distillation Training")],
        validation_data=(X_test, y_test)
    )
    distiller.student.save(STUDENT_MODEL_FILE)
    logger.info("Student Model saved: %s", STUDENT_MODEL_FILE)
    
    plt.figure(figsize=(8, 5))
    plt.plot(hist_distill.history['mse'], label='Training MSE')
    plt.plot(hist_distill.history['val_mse'], label='Validation MSE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    # plt.title('Student Model Distillation History')
    plt.legend()
    plt.minorticks_on()
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.savefig(os.path.join(PLOTS_FOLDER, "distillation_training_history.pdf"), bbox_inches='tight')
    plt.close()
    logger.info("Distillation training history plot saved")
    
    return distiller.student

# Convert distilled student model for CleverHans
def convert_distilled_to_classification(distiller_student, X_train):
    logger.info("Converting distilled student model to classification model...")
    student_copy = tf.keras.models.clone_model(distiller_student)
    for layer in student_copy.layers:
        layer.trainable = False
    student_copy.add(Dense(2, activation='softmax'))
    student_copy.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    y_dummy = np.zeros((X_train.shape[0], 2))
    mid_range = int(X_train.shape[0] / 2.0)
    y_dummy[0:mid_range, 0] = 1.0
    
    history = student_copy.fit(
        X_train, y_dummy,
        epochs=1,
        batch_size=128,
        verbose=0,
        validation_split=0.33,
        shuffle=True,
        callbacks=[TqdmCallback(verbose=1, desc="Distilled Classification Conversion")]
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    # plt.title('Distilled Classification Model Conversion History')
    plt.legend()
    plt.minorticks_on()
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.savefig(os.path.join(PLOTS_FOLDER, "distilled_classification_conversion_history.pdf"), bbox_inches='tight')
    plt.close()
    logger.info("Distilled classification conversion history plot saved")
    
    # Build the model by calling it with sample input to define the input tensor.
    dummy_input = X_train[:1]
    _ = student_copy(dummy_input)
    
    return student_copy



# Evaluate attacks on distilled model
def evaluate_distilled_attacks(distiller_student, distilled_classification_model, X_test, y_test, attacks, eps_vals):
    logger.info("Evaluating adversarial attacks on distilled model...")
    # Instead of wrapping the model, use it directly:
    logits_model_student = distilled_classification_model

    params_list = list(product(eps_vals, attacks))
    shuffle(params_list)
    
    distil_mal_diff_distance_list = []
    distil_real_mse_list = []
    distil_mal_mse_list = []
    distil_mal_predicted_diff_list = []
    distil_attack_name_list = []
    distil_eps_val_list = []
    
    for _ in tqdm(range(NUM_SAMPLES), desc="Processing Test Samples (Distilled)"):
        i = np.random.randint(0, X_test.shape[0])
        test_input = X_test[i:i+1, :]
        real_output = y_test[i:i+1, :]
        
        for eps_val, attack in tqdm(params_list, desc="Evaluating Attacks", leave=False): #params_list:
            mal_input = generate_adversarial_examples(logits_model_student, test_input, eps_val, attack)
            
            mal_diff = mal_input - test_input
            mal_diff_distance = norm(mal_diff, ord=np.inf)
            distil_mal_diff_distance_list.append(mal_diff_distance)
            
            test_output = distiller_student.predict(test_input, verbose=0)
            real_mse = mean_squared_error(real_output, test_output)
            distil_real_mse_list.append(real_mse)
            
            mal_output = distiller_student.predict(mal_input, verbose=0)
            mal_mse = mean_squared_error(real_output, mal_output)
            distil_mal_mse_list.append(mal_mse)
            
            mal_predicted_diff = norm(mal_output - test_output, ord=np.inf)
            distil_mal_predicted_diff_list.append(mal_predicted_diff)
            
            distil_attack_name_list.append(attack)
            distil_eps_val_list.append(eps_val)
    
    df_result_distill = pd.DataFrame({
        'Malicious_Distance': distil_mal_diff_distance_list,
        'Real_Predicted_MSE': distil_real_mse_list,
        'Malicious_Predicted_MSE': distil_mal_mse_list,
        'MalOut_RealOut_Diff': distil_mal_predicted_diff_list,
        'Attack': distil_attack_name_list,
        'eps': distil_eps_val_list
    })
    
    with open(os.path.join(TABLES_FOLDER, "distilled_attack_results.tex"), 'w') as f:
        f.write(df_result_distill.to_latex(index=False, float_format="%.4f"))
    logger.info("Distilled attack results saved as LaTeX")
    
    return df_result_distill


# Plot comparison
def plot_comparison(df_result, df_result_distill):
    logger.info("Plotting MSE comparison for undefended and distilled models...")
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(10, 5))
    
    sns.lineplot(x='eps', y='Malicious_Predicted_MSE', hue='Attack', ci='sd', estimator="median",
                 data=df_result, ax=ax[0])
    ax[0].set_title("Undefended Model")
    ax[0].set_xlabel("Epsilon (ε)")
    ax[0].set_ylabel("Malicious Predicted MSE")
    
    sns.lineplot(x='eps', y='Malicious_Predicted_MSE', hue='Attack', ci='sd', estimator="median",
                 data=df_result_distill, ax=ax[1])
    # ax[1].set_title("Defensively Distilled Model")
    ax[1].set_xlabel("Epsilon ($\epsilon$)")
    ax[1].set_ylabel("Malicious Predicted MSE")
    
    plt.tight_layout()
    ax[0].minorticks_on()
    ax[1].minorticks_on()
    ax[0].grid(which='both', linestyle='--', linewidth=0.5)
    ax[1].grid(which='both', linestyle='--', linewidth=0.5)

    plt.savefig(os.path.join(PLOTS_FOLDER, "mse_comparison.pdf"), bbox_inches='tight')
    plt.close()
    logger.info("MSE comparison plot saved")

def plot_mse_distribution_for_attack(df_undefended, df_defended, attack='FGSM'):
    """
    Plots two side-by-side histograms of MSE values for a given attack:
    - Left: Undefended model
    - Right: Defended (distilled) model
    """
    # Filter rows for the chosen attack
    df_un = df_undefended[df_undefended['Attack'] == attack]
    df_def = df_defended[df_defended['Attack'] == attack]
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

    # Undefended histogram
    sns.histplot(data=df_un, x='Malicious_Predicted_MSE', bins=30, stat="percent", ax=axes[0])
    axes[0].set_title("(a) Undefended")
    axes[0].set_xlabel("MSE")
    axes[0].set_ylabel("Percent")

    # Defended histogram
    sns.histplot(data=df_def, x='Malicious_Predicted_MSE', bins=30, stat="percent", ax=axes[1])
    axes[1].set_title("(b) Defensive Distillation")
    axes[1].set_xlabel("MSE")
    axes[1].set_ylabel("Percent")

    fig.suptitle(f"Distribution of MSE values for {attack} attack")


    # set grids for both axes and enable minor ticks
    axes[0].minorticks_on()
    axes[1].minorticks_on()
    axes[0].grid(which='both', linestyle='--', linewidth=0.5)
    axes[1].grid(which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    # Save the figure
    plt.savefig(os.path.join(PLOTS_FOLDER, f"mse_distribution_{attack}.pdf"), bbox_inches='tight')
    plt.close()
    logger.info(f"MSE distribution plot for {attack} saved")

def create_eps_comparison_table(df_undefended, df_defended, eps_values, attacks, table_filename="eps_comparison_table.tex"):
    """
    Create a LaTeX table comparing MSE values for undefended vs. defended models
    across multiple attacks (ATTACKS) and epsilon (EPS_VALUES).
    The table is grouped by (Attack, Model) rows and epsilon columns.
    
    :param df_undefended: DataFrame of attack results for the undefended model
    :param df_defended: DataFrame of attack results for the defended/distilled model
    :param eps_values: List of epsilon values, e.g. [0.01, 0.1, 0.3, 0.5, 0.7, 0.8]
    :param attacks: List of attack names, e.g. ['FGSM', 'BIM', 'MIM', 'PGD']
    :param table_filename: Name of the output LaTeX file
    :return: A pandas DataFrame representation of the table (also writes LaTeX to file)
    """
    # We’ll create a multi-level row index: (Attack, Model) => e.g. (FGSM, Undef.)
    row_tuples = []
    for atk in attacks:
        row_tuples.append((atk, "Undef."))
        row_tuples.append((atk, "Distill."))

    row_index = pd.MultiIndex.from_tuples(row_tuples, names=["Attack", "Model"])

    # Columns are simply the eps_values
    cols = eps_values

    # Create an empty DataFrame with the multi-level rows and epsilon columns
    table_df = pd.DataFrame(index=row_index, columns=cols)

    # Fill in each cell with the median MSE from the respective DataFrame
    for atk in attacks:
        for eps in eps_values:
            # 1) Undefended
            mask_un = ((df_undefended['Attack'] == atk) & (df_undefended['eps'] == eps))
            if not df_undefended[mask_un].empty:
                val_undef = df_undefended.loc[mask_un, 'Malicious_Predicted_MSE'].median()
            else:
                val_undef = np.nan
            table_df.loc[(atk, "Undef."), eps] = val_undef

            # 2) Defended
            mask_def = ((df_defended['Attack'] == atk) & (df_defended['eps'] == eps))
            if not df_defended[mask_def].empty:
                val_def = df_defended.loc[mask_def, 'Malicious_Predicted_MSE'].median()
            else:
                val_def = np.nan
            table_df.loc[(atk, "Distill."), eps] = val_def

    # Convert to LaTeX
    latex_str = table_df.to_latex(
        float_format="%.4f",
        multirow=True,
        index=True,
        header=True
    )

    # Save LaTeX to file
    with open(os.path.join(TABLES_FOLDER, table_filename), 'w') as f:
        f.write(latex_str)

    logger.info(f"Epsilon comparison table saved as {table_filename}")
    return table_df

# Main execution
def main(train_model=True):
    logger.info("Starting Chapter 6 experiment...")
    
    # Load and split data
    X, Y, M = load_data()
    X_train, X_test, y_train, y_test = split_data(X, Y)

    if train_model:
        # Train undefended model and create classification conversion
        undefended_model = train_undefended_model(X_train, y_train, M, Y.shape[1])
        classification_model = convert_to_classification_model(undefended_model, X_train, Y.shape[1])
        # Save the newly converted classification model
        classification_model.save(CLASSIFICATION_MODEL_FILE)
        
        # df_result = evaluate_attacks(classification_model, undefended_model, X_test, y_test, ATTACKS, EPS_VALUES)
        
        # Train defensive distillation models and convert student to classification model
        distiller_student = train_distilled_models(X_train, X_test, y_train, y_test, M, Y.shape[1])
        distilled_classification_model = convert_distilled_to_classification(distiller_student, X_train)
        # Save the distilled classification model
        distilled_classification_model.save(DISTILLED_CLASSIFICATION_MODEL_FILE)
        
        # df_result_distill = evaluate_distilled_attacks(distiller_student, distilled_classification_model, X_test, y_test, ATTACKS, EPS_VALUES)
    else:
        # Load pre-trained models from disk
        undefended_model = tf.keras.models.load_model(UNDEFENDED_MODEL_FILE)
        classification_model = tf.keras.models.load_model(CLASSIFICATION_MODEL_FILE)
        distiller_student = tf.keras.models.load_model(STUDENT_MODEL_FILE)
        distilled_classification_model = tf.keras.models.load_model(DISTILLED_CLASSIFICATION_MODEL_FILE)
        
        df_result = evaluate_attacks(classification_model, undefended_model, X_test, y_test, ATTACKS, EPS_VALUES)
        df_result_distill = evaluate_distilled_attacks(distiller_student, distilled_classification_model, X_test, y_test, ATTACKS, EPS_VALUES)
    
    plot_comparison(df_result, df_result_distill)

    # Plot MSE distribution for each attack
    for attack in tqdm(ATTACKS, desc="Processing Attacks for mse distribution plot"):
        plot_mse_distribution_for_attack(df_result, df_result_distill, attack=attack)

    create_eps_comparison_table(
        df_undefended=df_result,
        df_defended=df_result_distill,
        eps_values=EPS_VALUES,
        attacks=ATTACKS,
        table_filename="eps_comparison_table.tex"
    )
    
    
    logger.info("Chapter 6 experiment completed successfully")


if __name__ == "__main__":
    main(train_model=False)
