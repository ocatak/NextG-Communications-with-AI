# -*- coding: utf-8 -*-
"""SpectrumSensingAdversarialDefense.ipynb

Adapted for Chapter 5: Strengthening AI-Driven Spectrum Sensing with Robust Adversarial Defenses
Using dataset from https://github.com/ocatak/RadarSpectrumSensing-FL-AML/raw/main/converted_dataset.zip
"""

import os
import scipy.io
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from lapixdl.evaluation.evaluate import evaluate_segmentation
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib.colors import ListedColormap
from tqdm.keras import TqdmCallback
from tqdm import tqdm
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.basic_iterative_method import basic_iterative_method
from tensorflow.keras import backend as K
import logging
from datetime import datetime
from collections import Counter
import json
import random
from keras.callbacks import ModelCheckpoint

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
CHAPTER_FOLDER = './Chapter-5'
DATA_URL = "https://github.com/ocatak/RadarSpectrumSensing-FL-AML/raw/main/converted_dataset.zip"
DATA_FILE = os.path.join(CHAPTER_FOLDER, "converted_dataset.zip")
EXTRACTED_DATA_PATH = os.path.join(CHAPTER_FOLDER, "data/convertedFolder")
MODEL_FOLDER = os.path.join(CHAPTER_FOLDER, "Saved_Models")
RESULTS_FOLDER = os.path.join(CHAPTER_FOLDER, "Results")
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(os.path.join(CHAPTER_FOLDER, "Plots"), exist_ok=True)

TEACHER_MODEL_FILE = os.path.join(MODEL_FOLDER, "teacher_model.keras")
STUDENT_MODEL_FILE = os.path.join(MODEL_FOLDER, "student_model.keras")
UNDEFENDED_MODEL_FILE = os.path.join(MODEL_FOLDER, "undefended_model.keras")

# Hyperparameters
EPS_VALUES = [13, 26, 39, 52, 64, 77, 90, 103, 115, 128]
ATTACKS = ['BIM', 'PGD', 'FGSM']
BATCH_SIZE = 10
EPOCHS = 2000
PATIENCE = 50
TEMPERATURE = 10
NUM_CLASSES = 3
CLASSES = ["LTE", "5G", "Noise"]
NUM_RUNS = 5

# Initialize logging
logger = setup_logging(CHAPTER_FOLDER)

# Download and unzip dataset
if not os.path.exists(DATA_FILE):
    logger.info("Downloading dataset from %s", DATA_URL)
    os.system(f"wget -q -O {DATA_FILE} {DATA_URL}")
    logger.info("Dataset downloaded successfully")
else:
    logger.info("Dataset file already exists, skipping download: %s", DATA_FILE)

if not os.path.exists(EXTRACTED_DATA_PATH) or not os.path.exists(os.path.join(EXTRACTED_DATA_PATH, "rcvdSpectrogram_1.mat")):
    logger.info("Unzipping dataset file: %s", DATA_FILE)
    os.system(f"unzip -q {DATA_FILE} -d {CHAPTER_FOLDER}")
    logger.info("Dataset unzipped successfully")
else:
    logger.info("Dataset already unzipped, skipping unzip: %s", EXTRACTED_DATA_PATH)

# Custom U-Net Model
def conv_block(input_tensor, num_filters):
    x = layers.Conv2D(num_filters, 3, padding="same")(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(num_filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x

def encoder_block(input_tensor, num_filters):
    x = conv_block(input_tensor, num_filters)
    p = layers.MaxPooling2D((2, 2))(x)
    return x, p

def decoder_block(input_tensor, concat_tensor, num_filters):
    x = layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input_tensor)
    x = layers.concatenate([x, concat_tensor], axis=-1)
    x = conv_block(x, num_filters)
    return x

def build_unet(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x1, p1 = encoder_block(inputs, 16)
    x2, p2 = encoder_block(p1, 32)
    x3, p3 = encoder_block(p2, 64)
    x4, p4 = encoder_block(p3, 128)
    b = conv_block(p4, 256)
    d1 = decoder_block(b, x4, 128)
    d2 = decoder_block(d1, x3, 64)
    d3 = decoder_block(d2, x2, 32)
    d4 = decoder_block(d3, x1, 16)
    outputs = layers.Conv2D(num_classes, (1, 1), activation="softmax")(d4)
    model = models.Model(inputs, outputs, name="U-Net")
    return model

# Load and preprocess dataset
def convert_to_int(filename):
    conv_dict = {'Noise': 2, 'LTE': 0, 'NR': 1, 'Radar': 2}
    with open(filename, 'r') as f, open('tmp.txt', 'w') as f_tmp:
        for line in f:
            line_tmp = line
            for key, value in conv_dict.items():
                line_tmp = line_tmp.replace(key, str(value))
            f_tmp.write(line_tmp)
    val = pd.read_csv('tmp.txt', sep=',', header=None).values
    return val

def load_and_preprocess_data(random_seed=42):
    logger.info("Loading and preprocessing dataset...")
    X = []
    Y = []
    for idx in tqdm(range(1, 500), desc="Loading Dataset"):
        inp = scipy.io.loadmat(os.path.join(CHAPTER_FOLDER, f'data/convertedFolder/rcvdSpectrogram_{idx}.mat'))['rcvdSpectrogram']
        real = convert_to_int(os.path.join(CHAPTER_FOLDER, f'data/convertedFolder/trueLabels_{idx}.csv'))
        X.append(inp)
        Y.append(real)
    
    flattened_labels = np.array(Y).flatten()
    class_counts = Counter(flattened_labels)
    logger.info("Class distribution in dataset: %s", class_counts)
    
    if len(X) == 0 or len(Y) == 0:
        logger.error("Dataset is empty after preprocessing. Please check the dataset and class mapping.")
        raise ValueError("Dataset is empty after preprocessing.")
    
    logger.info("Splitting dataset into train and validation sets (90/10 ratio)")
    x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.1, random_state=random_seed)
    
    x_train = np.array(x_train).astype(float)
    x_val = np.array(x_val).astype(float)
    y_train = np.array(y_train).astype(np.int32)
    y_val = np.array(y_val).astype(np.int32)
    
    logger.info("Normalizing input data to range [0, 1]")
    x_train = x_train / 255.0
    x_val = x_val / 255.0
    
    logger.info("Converting labels to one-hot encoding")
    y_train_onehot = tf.one_hot(y_train, depth=NUM_CLASSES)
    y_val_onehot = tf.one_hot(y_val, depth=NUM_CLASSES)
    
    logger.info("Dataset loaded. Train Shape: %s, Test Shape: %s", x_train.shape, x_val.shape)
    return x_train, x_val, y_train, y_val, y_train_onehot, y_val_onehot

def compute_sample_weights(y, class_weight_dict):
    sample_weights = np.zeros_like(y, dtype=np.float32)
    for class_idx, weight in class_weight_dict.items():
        sample_weights[y == class_idx] = weight
    return sample_weights

def apply_temperature(logits, temperature=5.0):
    # Convert to tensor if it's a NumPy array
    logits = tf.convert_to_tensor(logits, dtype=tf.float32)
    return K.softmax(logits / temperature)

def train_models(x_train, y_train_onehot, x_val, y_val_onehot, y_train, y_val, create_models=False):
    input_shape = (256, 256, 3)
    
    flat_labels = y_train.flatten()
    class_weights = compute_class_weight('balanced', classes=np.unique(flat_labels), y=flat_labels)
    class_weight_dict = dict(enumerate(class_weights))
    logger.info("Class weights: %s", class_weight_dict)
    
    logger.info("Computing per-pixel sample weights for training and validation...")
    train_sample_weights = compute_sample_weights(y_train, class_weight_dict)
    val_sample_weights = compute_sample_weights(y_val, class_weight_dict)
    
    if create_models:
        logger.info("Training Teacher Model...")
        teacher_model = build_unet(input_shape, NUM_CLASSES)
        teacher_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, min_delta=0.001)
        # Uncomment to train the teacher model if not already trained
        teacher_model.fit(
             x_train, y_train_onehot,
             batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=0,
             validation_data=(x_val, y_val_onehot),
             sample_weight=train_sample_weights,
             callbacks=[TqdmCallback(verbose=1, desc="Teacher Training"), early_stopping]
         )
        teacher_model.save(TEACHER_MODEL_FILE)
        logger.info("Teacher Model saved: %s", TEACHER_MODEL_FILE)
        
        # Load the pre-trained teacher model (as per your modification)
        teacher_model = tf.keras.models.load_model(TEACHER_MODEL_FILE)
        logger.info("Teacher Model loaded: %s", TEACHER_MODEL_FILE)
        
        logger.info("Generating teacher predictions for distillation with temperature=%d", TEMPERATURE)
        # Extract logits from the final Conv2D layer before softmax
        # The last layer is softmax, so we take the second-to-last layer's output (logits)
        teacher_logits_model = models.Model(inputs=teacher_model.input, outputs=teacher_model.layers[-1].output)
        # Remove softmax by creating a new model without activation
        logits_layer = layers.Conv2D(NUM_CLASSES, (1, 1), name="logits")(teacher_model.layers[-2].output)
        teacher_logits_model = models.Model(inputs=teacher_model.input, outputs=logits_layer)
        
        teacher_logits_train = teacher_logits_model.predict(x_train, verbose=0, callbacks=[TqdmCallback(verbose=1, desc="Teacher Logits (Train)")])
        teacher_predictions = apply_temperature(teacher_logits_train, temperature=TEMPERATURE).numpy()
        
        teacher_logits_val = teacher_logits_model.predict(x_val, verbose=0, callbacks=[TqdmCallback(verbose=1, desc="Teacher Logits (Val)")])
        val_teacher_predictions = apply_temperature(teacher_logits_val, temperature=TEMPERATURE).numpy()
        
        logger.info("Training Student Model with Defensive Distillation...")
        student_model = build_unet(input_shape, NUM_CLASSES)
        student_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        
        early_stopping = EarlyStopping(monitor='loss', patience=PATIENCE, restore_best_weights=True)
        # add model checkpoint to save the best model
        model_checkpoint = ModelCheckpoint(STUDENT_MODEL_FILE, monitor='loss', save_best_only=True)
        # if student model exists, load it
        if os.path.exists(STUDENT_MODEL_FILE):
            student_model = tf.keras.models.load_model(STUDENT_MODEL_FILE)
            logger.info("Student Model loaded: %s", STUDENT_MODEL_FILE)
        
        student_model.fit(
            x_train, teacher_predictions,
            batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=0,
            validation_data=(x_val, y_val_onehot),
            sample_weight=train_sample_weights,
            callbacks=[TqdmCallback(verbose=1, desc="Student Training"), early_stopping, model_checkpoint]
        )
        student_model.save(STUDENT_MODEL_FILE)
        logger.info("Student Model saved: %s", STUDENT_MODEL_FILE)
    else:
        logger.info("Loading pre-trained models...")
        teacher_model = tf.keras.models.load_model(TEACHER_MODEL_FILE)
        student_model = tf.keras.models.load_model(STUDENT_MODEL_FILE)
        logger.info("Models loaded successfully")
    
    return teacher_model, student_model, teacher_model

def generate_adversarial_examples(model, x, y, epsilon, attack_type, num_iterations=2000):
    logger.info("Generating adversarial examples with %s (epsilon=%d)", attack_type, epsilon)
    
    def loss_fn(labels, logits):
        labels_onehot = tf.one_hot(labels, depth=NUM_CLASSES)
        return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(labels_onehot, logits))
    
    clip_min = 0.0
    clip_max = 1.0
    
    if attack_type == 'FGSM':
        adv_x = fast_gradient_method(
            model_fn=model,
            x=x,
            eps=epsilon/255.0,
            norm=np.inf,
            y=y,
            loss_fn=loss_fn
        )
    elif attack_type == 'BIM':
        adv_x = basic_iterative_method(
            model_fn=model,
            x=x,
            y=y,
            eps=epsilon,
            eps_iter=(epsilon)/num_iterations,
            nb_iter=num_iterations,
            norm=np.inf,
            clip_min=clip_min,
            clip_max=clip_max
        )
    elif attack_type == 'PGD':
        adv_x = projected_gradient_descent(
            model_fn=model,
            x=x,
            y=y,
            eps=epsilon,
            eps_iter=(epsilon)/num_iterations,
            nb_iter=num_iterations,
            norm=np.inf,
            clip_min=clip_min,
            clip_max=clip_max
        )
    else:
        raise ValueError(f"Unsupported attack type: {attack_type}")
    
    logger.info("Adversarial examples generated")
    return adv_x.numpy()

def compute_metrics(y_true, y_pred, classes):
    logger.info("Computing performance metrics...")
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=range(len(classes)))
    
    metrics = {
        'Accuracy': [],
        'Recall': [],
        'Precision': [],
        'Specificity': [],
        'F-Score': [],
        'FPR': [],
        'IoU': []
    }
    
    for i in tqdm(range(len(classes)), desc="Computing Metrics per Class"):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)
        
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        f_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
        iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
        
        metrics['Accuracy'].append(accuracy)
        metrics['Recall'].append(recall)
        metrics['Precision'].append(precision)
        metrics['Specificity'].append(specificity)
        metrics['F-Score'].append(f_score)
        metrics['FPR'].append(fpr)
        metrics['IoU'].append(iou)
    
    avg_metrics = {key: np.mean(values) for key, values in metrics.items()}
    logger.info("Metrics computation completed")
    return metrics, avg_metrics

def plot_iou_vs_epsilon(iou_results_all, attacks, epsilons, chapter_folder):
    logger.info("Plotting IoU vs. Epsilon with error bars")
    plt.figure(figsize=(10, 6))
    
    for model_type in ["Undefended", "Defensive Distillation"]:
        for attack in attacks:
            iou_values = np.array([[iou_results_all[run][model_type][attack][eps] for run in range(NUM_RUNS)] for eps in epsilons])
            iou_mean = np.mean(iou_values, axis=1)
            iou_std = np.std(iou_values, axis=1)
            plt.errorbar(epsilons, iou_mean, yerr=iou_std, fmt='-o', label=f"{model_type} - {attack}", capsize=5, capthick=2)
    
    plt.xlabel("Epsilon (ε)", fontsize=12)
    plt.ylabel("IoU", fontsize=12)
    plt.title("IoU vs. Epsilon for Adversarial Attacks (Mean ± Std)", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(chapter_folder, "Plots/iou_vs_epsilon_with_error_bars.pdf"), bbox_inches='tight')
    plt.show()
    logger.info("IoU vs. Epsilon plot with error bars saved")

def plot_iou_boxplots(iou_results_all, attacks, chapter_folder):
    for attack in tqdm(attacks, desc="Plotting IoU Boxplots"):
        logger.info("Plotting IoU boxplot for %s", attack)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        
        data_undefended = []
        for eps in EPS_VALUES:
            iou_values = [iou_results_all[run]["Undefended"][attack][eps] for run in range(NUM_RUNS)]
            data_undefended.append(iou_values)
        axes[0].boxplot(data_undefended, labels=EPS_VALUES)
        axes[0].set_title(f"Undefended - {attack}", fontsize=12)
        axes[0].set_xlabel("Epsilon (ε)", fontsize=10)
        axes[0].set_ylabel("IoU", fontsize=10)
        axes[0].grid(True, linestyle='--', alpha=0.7)
        
        data_defended = []
        for eps in EPS_VALUES:
            iou_values = [iou_results_all[run]["Defensive Distillation"][attack][eps] for run in range(NUM_RUNS)]
            data_defended.append(iou_values)
        axes[1].boxplot(data_defended, labels=EPS_VALUES)
        axes[1].set_title(f"Defensive Distillation - {attack}", fontsize=12)
        axes[1].set_xlabel("Epsilon (ε)", fontsize=10)
        axes[1].grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(chapter_folder, f"Plots/iou_boxplot_{attack}.pdf"), bbox_inches='tight')
        plt.show()
        logger.info("IoU boxplot for %s saved", attack)

def plot_initial_performance_boxplot(initial_metrics_all, chapter_folder):
    logger.info("Plotting initial performance boxplot")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    
    metrics_to_plot = ['IoU', 'Accuracy', 'F-Score']
    
    data_undefended = []
    for metric in metrics_to_plot:
        values = [initial_metrics_all[run]["Undefended"][metric] for run in range(NUM_RUNS)]
        data_undefended.append(values)
    axes[0].boxplot(data_undefended, labels=metrics_to_plot)
    axes[0].set_title("Undefended Model", fontsize=12)
    axes[0].set_ylabel("Metric Value", fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    data_defended = []
    for metric in metrics_to_plot:
        values = [initial_metrics_all[run]["Defensive Distillation"][metric] for run in range(NUM_RUNS)]
        data_defended.append(values)
    axes[1].boxplot(data_defended, labels=metrics_to_plot)
    axes[1].set_title("Defensive Distillation Model", fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(chapter_folder, "Plots/initial_performance_boxplot.pdf"), bbox_inches='tight')
    plt.show()
    logger.info("Initial performance boxplot saved")

def save_results(initial_metrics_all, performance_results_all, iou_results_all, chapter_folder):
    logger.info("Saving experimental results...")
    results = {
        "initial_metrics": initial_metrics_all,
        "performance_results": performance_results_all,
        "iou_results": iou_results_all
    }
    results_file = os.path.join(chapter_folder, "Results/experimental_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info("Results saved to %s", results_file)

def run_experiment(random_seed, create_models=True):
    logger.info("Starting experiment with random seed: %d", random_seed)
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    x_train, x_val, y_train, y_val, y_train_onehot, y_val_onehot = load_and_preprocess_data(random_seed)
    
    teacher_model, student_model, undefended_model = train_models(x_train, y_train_onehot, x_val, y_val_onehot, y_train, y_val, create_models)
    
    logger.info("Evaluating initial performance (no attack)...")
    y_pred_undefended = np.argmax(undefended_model.predict(x_train, verbose=0, callbacks=[TqdmCallback(verbose=1, desc="Undefended Prediction")]), axis=-1)
    y_pred_defended = np.argmax(student_model.predict(x_train, verbose=0, callbacks=[TqdmCallback(verbose=1, desc="Defended Prediction")]), axis=-1)
    
    metrics_undefended, avg_metrics_undefended = compute_metrics(y_train, y_pred_undefended, CLASSES)
    metrics_defended, avg_metrics_defended = compute_metrics(y_train, y_pred_defended, CLASSES)
    
    initial_metrics = {
        "Undefended": avg_metrics_undefended,
        "Defensive Distillation": avg_metrics_defended
    }
    
    logger.info("Initial Performance (No Attack):")
    logger.info("Undefended Model:")
    for metric, values in metrics_undefended.items():
        logger.info("%s: Average=%.4f, 5G=%.4f, LTE=%.4f, Noise=%.4f", metric, avg_metrics_undefended[metric], values[1], values[0], values[2])
    logger.info("Defended Model:")
    for metric, values in metrics_defended.items():
        logger.info("%s: Average=%.4f, 5G=%.4f, LTE=%.4f, Noise=%.4f", metric, avg_metrics_defended[metric], values[1], values[0], values[2])
    
    logger.info("Evaluating under adversarial attacks...")
    models = {"Undefended": undefended_model, "Defensive Distillation": student_model}
    performance_results = {"Undefended": {}, "Defensive Distillation": {}}
    iou_results = {"Undefended": {attack: {} for attack in ATTACKS}, "Defensive Distillation": {attack: {} for attack in ATTACKS}}
    
    for attack in tqdm(ATTACKS, desc="Processing Attacks"):
        performance_results["Undefended"][attack] = {}
        performance_results["Defensive Distillation"][attack] = {}
        for epsilon in tqdm(EPS_VALUES, desc=f"Processing {attack} with Epsilon", leave=False):
            logger.info("Processing %s with epsilon=%d", attack, epsilon)
            adv_x_undefended = generate_adversarial_examples(undefended_model, x_train[:20], y_train[:20], epsilon, attack)
            adv_x_defended = generate_adversarial_examples(student_model, x_train[:20], y_train[:20], epsilon, attack)
            
            y_pred_adv_undefended = np.argmax(undefended_model.predict(adv_x_undefended, verbose=0, callbacks=[TqdmCallback(verbose=1, desc="Undefended Adv Prediction")]), axis=-1)
            y_pred_adv_defended = np.argmax(student_model.predict(adv_x_defended, verbose=0, callbacks=[TqdmCallback(verbose=1, desc="Defended Adv Prediction")]), axis=-1)
            
            metrics_adv_undefended, avg_metrics_adv_undefended = compute_metrics(y_train[:20], y_pred_adv_undefended, CLASSES)
            metrics_adv_defended, avg_metrics_adv_defended = compute_metrics(y_train[:20], y_pred_adv_defended, CLASSES)
            
            if epsilon in [13, 64, 128]:
                performance_results["Undefended"][attack][epsilon] = avg_metrics_adv_undefended
                performance_results["Defensive Distillation"][attack][epsilon] = avg_metrics_adv_defended
            
            iou_results["Undefended"][attack][epsilon] = avg_metrics_adv_undefended['IoU']
            iou_results["Defensive Distillation"][attack][epsilon] = avg_metrics_adv_defended['IoU']
    
    logger.info("Undefended Model Performance Under Attack:")
    for epsilon in [13, 64, 128]:
        logger.info("Epsilon = %d:", epsilon)
        for attack in ATTACKS:
            metrics = performance_results["Undefended"][attack][epsilon]
            logger.info("%s: %s", attack, ", ".join([f"{key}={value:.4f}" for key, value in metrics.items()]))
    
    logger.info("Defended Model Performance Under Attack:")
    for epsilon in [13, 64, 128]:
        logger.info("Epsilon = %d:", epsilon)
        for attack in ATTACKS:
            metrics = performance_results["Defensive Distillation"][attack][epsilon]
            logger.info("%s: %s", attack, ", ".join([f"{key}={value:.4f}" for key, value in metrics.items()]))
    
    logger.info("Experiment with random seed %d completed", random_seed)
    return initial_metrics, performance_results, iou_results

def main_multiple_runs(create_models=False):
    initial_metrics_all = []
    performance_results_all = []
    iou_results_all = []
    
    for run in range(NUM_RUNS):
        logger.info("Starting run %d/%d", run + 1, NUM_RUNS)
        random_seed = 42 + run
        if run == 0:
            initial_metrics, performance_results, iou_results = run_experiment(random_seed, create_models=create_models)
        else:
            initial_metrics, performance_results, iou_results = run_experiment(random_seed, create_models=False)
        
        initial_metrics_all.append(initial_metrics)
        performance_results_all.append(performance_results)
        iou_results_all.append(iou_results)
    
    save_results(initial_metrics_all, performance_results_all, iou_results_all, CHAPTER_FOLDER)
    plot_initial_performance_boxplot(initial_metrics_all, CHAPTER_FOLDER)
    plot_iou_vs_epsilon(iou_results_all, ATTACKS, EPS_VALUES, CHAPTER_FOLDER)
    plot_iou_boxplots(iou_results_all, ATTACKS, CHAPTER_FOLDER)
    logger.info("All experiments completed successfully")

if __name__ == "__main__":
    main_multiple_runs(create_models=True)