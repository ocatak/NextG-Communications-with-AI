from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent,fast_gradient_method
from cleverhans.tf2.attacks.basic_iterative_method import basic_iterative_method

import os
import numpy as np
from scipy.io import loadmat
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import tensorflow as tf
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.keras import TqdmCallback

# Constants
CHAPTER_FOLDER = './Chapter-3'
DATASET_FOLDER = './Chapter-3/Dataset'
INPUT_FILE = os.path.join(DATASET_FOLDER, 'DLCB_input.mat')
OUTPUT_FILE = os.path.join(DATASET_FOLDER, 'DLCB_output.mat')
MODEL_FOLDER = './Chapter-3/Saved_Models'
ADVERSARIAL_MODEL_FOLDER = './Chapter-3/Saved_Models_Adversarial'
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(ADVERSARIAL_MODEL_FOLDER, exist_ok=True)

# Parameters
DL_SIZE_RATIO = 0.8
NUM_TOTAL_TX = 4
NUM_BEAMS = 512
NB_EPOCH = 5000
BATCH_SIZE = 5000
LOSS_FUNCTION = 'mean_squared_error'
ACTIVATION_FUNCTION = 'relu'
TRAIN_MODE = True  # Set to False to load saved models
EPSILONS = [0.01, 0.05, 0.1, 0.15, 0.2]  # Epsilon values for adversarial attacks

# Load Dataset
def load_dataset(input_file, output_file):
    """Loads input and output datasets from MAT files."""
    input_data = loadmat(input_file)['DL_input']
    output_data = loadmat(output_file)['DL_output']

    # Extract real and imaginary parts
    input_data = np.hstack([input_data.real, input_data.imag])
    output_data = np.hstack([output_data.real, output_data.imag])
    return input_data, output_data

# Split Dataset
def split_dataset(input_data, output_data, split_ratio=0.8, random_seed=2016):
    """Splits the dataset into training and testing subsets."""
    np.random.seed(random_seed)
    num_users = input_data.shape[0]
    train_size = int(num_users * split_ratio)
    test_size = num_users - train_size

    train_indices = np.random.choice(num_users, size=train_size, replace=False)
    remaining_indices = list(set(range(num_users)) - set(train_indices))
    test_indices = np.random.choice(remaining_indices, size=test_size, replace=False)

    input_train, input_test = input_data[train_indices], input_data[test_indices]
    output_train, output_test = output_data[train_indices], output_data[test_indices]

    return input_train, input_test, output_train, output_test, test_indices

# Generate Adversarial Examples
def generate_adversarial_examples(model, X, epsilon, attack_type):
    """Generates adversarial examples for a given model and dataset."""
   # print(f"[{attack_type}]: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if attack_type == 'fgsm':
        return fast_gradient_method(model, X, epsilon, norm=np.inf)
    elif attack_type == 'pgd':
        return projected_gradient_descent(model, X, epsilon, eps_iter=0.01, nb_iter=100, norm=np.inf, targeted=False)
    elif attack_type == 'bim':
        return basic_iterative_method(model, X, epsilon, eps_iter=0.01, nb_iter=100, norm=np.inf, targeted=False)
    else:
        raise ValueError(f"Unsupported attack type: {attack_type}")

# Build and Train Models
def train_models_old(input_train, output_train, input_test, output_test,
                 num_hidden_layers, nodes_per_layer, num_tx, num_beams,
                 nb_epoch, batch_size, loss_function, train_mode, adversarial=False):
    """
    Trains a separate model for each transmitter or loads pre-trained models.
    """
    input_shape = list(input_train.shape[1:])
    models = []
    folder = ADVERSARIAL_MODEL_FOLDER if adversarial else MODEL_FOLDER

    for tx_idx in tqdm(range(0, num_tx * num_beams, num_beams), desc='Training Models', position=0, leave=True):
        model_path = os.path.join(folder, f'TX{tx_idx // num_beams + 1}.keras')
        if train_mode:
            model = Sequential([
                Dense(nodes_per_layer, activation=ACTIVATION_FUNCTION, input_dim=input_shape[0]),
                *[Dense(nodes_per_layer, activation=ACTIVATION_FUNCTION) for _ in range(num_hidden_layers)],
                Dense(num_beams, activation=ACTIVATION_FUNCTION)
            ])
            model.compile(optimizer='rmsprop', loss=loss_function, metrics=['mean_squared_error'])

            if adversarial:
                adversarial_data = input_train + 0.1 * np.sign(np.random.normal(size=input_train.shape))
                model.fit(
                    adversarial_data,
                    output_train[:, tx_idx:tx_idx + num_beams],
                    batch_size=batch_size,
                    epochs=nb_epoch,
                    verbose=2,
                    validation_data=(input_test, output_test[:, tx_idx:tx_idx + num_beams])
                )
            else:
                model.fit(
                    input_train,
                    output_train[:, tx_idx:tx_idx + num_beams],
                    batch_size=batch_size,
                    epochs=nb_epoch,
                    verbose=2,
                    validation_data=(input_test, output_test[:, tx_idx:tx_idx + num_beams])
                )

            # Save the model
            model.save(model_path)
        else:
            print(f"Loading model from {model_path}")
            # Load the model
            model = load_model(model_path)

        models.append(model)

    return models

def train_models(input_train, output_train, input_test, output_test,
                 num_hidden_layers, nodes_per_layer, num_tx, num_beams,
                 nb_epoch, batch_size, loss_function, train_mode, distillation=False, temperature=10):
    """Trains models with an option for defensive distillation."""
    input_shape = list(input_train.shape[1:])
    models = []
    
    for tx_idx in tqdm(range(0, num_tx * num_beams, num_beams), desc='Training Models', position=0, leave=True):
        tx_id = tx_idx // num_beams + 1
        
        if distillation:
            # Train Teacher Model
            teacher_model = Sequential([
                Dense(int(nodes_per_layer * 2), activation='relu', input_dim=input_shape[0]),
                *[Dense(int(nodes_per_layer * 2), activation='relu') for _ in range(num_hidden_layers)],
                Dense(num_beams, activation='relu'),
                # Activation(lambda x: x / temperature)  # Softened softmax
            ])

            teacher_model.compile(optimizer='adam', loss=loss_function, metrics=['mean_squared_error'])
            # add early stopping
            early_stopping = EarlyStopping(monitor='val_loss', patience=200, restore_best_weights=True)

            teacher_model.fit(input_train, output_train[:, tx_idx:tx_idx + num_beams], batch_size=batch_size, epochs=nb_epoch, verbose=0,
                              validation_data=(input_test, output_test[:, tx_idx:tx_idx + num_beams]),
                              callbacks=[TqdmCallback(verbose=1), early_stopping])
            teacher_model.save(f"{MODEL_FOLDER}/TX{tx_id}_teacher.keras")
            teacher_predictions = teacher_model.predict(input_train, verbose = 0)
            
            # Train Student Model using Teacher Predictions
            student_model = Sequential([
                Dense(nodes_per_layer, activation='relu', input_dim=input_shape[0]),
                *[Dense(nodes_per_layer, activation='relu') for _ in range(num_hidden_layers)],
                Dense(num_beams, activation = 'relu'),
                Activation('relu')
            ])
            student_model.compile(optimizer='adam', loss=loss_function, metrics=['mean_squared_error'])
            # add early stopping
            early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)

            student_model.fit(input_train, teacher_predictions, batch_size=batch_size, epochs=nb_epoch, verbose=0,
                              validation_data=(input_test, output_test[:, tx_idx:tx_idx + num_beams]),
                              callbacks=[TqdmCallback(verbose=1), early_stopping])
            student_model.save(f"{MODEL_FOLDER}/TX{tx_id}_student.keras")
            models.append(student_model)
        else:
            # Standard Model Training
            model = Sequential([
                Dense(nodes_per_layer, activation='relu', input_dim=input_shape[0]),
                *[Dense(nodes_per_layer, activation='relu') for _ in range(num_hidden_layers)],
                Dense(num_beams),
                Activation('relu')
            ])
            model.compile(optimizer='adam', loss=loss_function, metrics=['mean_squared_error'])
            
            # Custom callback to print validation loss
            class ValidationLossCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    if logs is not None and epoch % 10 == 0:  # Print every 100 epochs
                        print(f"=== Epoch {epoch}: val_loss = {logs.get('val_loss'):.10f}")
            val_loss_callback = ValidationLossCallback()

            early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, mode = "min")
            
            model.fit(input_train, output_train[:, tx_idx:tx_idx + num_beams], batch_size=batch_size, epochs=nb_epoch, verbose=0,
                      validation_data=(input_test, output_test[:, tx_idx:tx_idx + num_beams]),
                      callbacks=[TqdmCallback(verbose=1), early_stopping])
            model.save(f"{MODEL_FOLDER}/TX{tx_id}_undefended.keras")
            models.append(model)
    
    return models


# Evaluate Models
def evaluate_models(models, input_test, output_test, num_tx, num_beams, epsilons, adversarial=False):
    """
    Evaluates models on clean and adversarial data.
    """
    results = []
    for tx_idx in tqdm(range(num_tx), desc='Evaluating Models', position=0, leave=False):
        model = models[tx_idx]
        original_predictions = model.predict(input_test, batch_size=10, verbose=0)
        original_mse = mean_squared_error(output_test[:, tx_idx * num_beams:(tx_idx + 1) * num_beams], original_predictions)

        for epsilon in tqdm(epsilons, desc='Generating Adversarial Examples', position=1, leave=False):
            for attack_type in ['fgsm', 'pgd', 'bim']:
                adversarial_data = generate_adversarial_examples(model, input_test, epsilon, attack_type)
                adversarial_predictions = model.predict(adversarial_data, verbose=0)
                adversarial_mse = mean_squared_error(output_test[:, tx_idx * num_beams:(tx_idx + 1) * num_beams], adversarial_predictions)

                results.append({
                    'TX': tx_idx + 1,
                    'Epsilon': epsilon,
                    'Attack': attack_type,
                    'Original MSE': original_mse,
                    'Adversarial MSE': adversarial_mse,
                    'Adversarial Training': adversarial
                })

    return pd.DataFrame(results)

def plot_mse_vs_epsilon(results_df):
    """Plots MSE vs. epsilon with separate subplots for adversarial training settings."""
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    
    for i, adversarial_training in enumerate([False, True]):
        subset = results_df[results_df['Adversarial Training'] == adversarial_training]
        sns.lineplot(ax=axes[i], data=subset, x='Epsilon', y='Adversarial MSE', hue='Attack', style='Attack')
        axes[i].set_title(f"Defensive Distillation: {adversarial_training}")
        axes[i].set_xlabel("$\epsilon$ values")
        axes[i].set_ylabel("MSE")
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(CHAPTER_FOLDER, 'mse_vs_epsilon.pdf'), bbox_inches='tight')
    plt.show()

def create_pearson_correlation_table(models, input_test, epsilon=0.1):
    """
    Computes the Pearson correlation coefficient between the original (clean)
    predictions and adversarial predictions for each attack, averaged over
    all models (TX). Returns a single-row DataFrame with columns = [FGSM, PGD, BIM, MIM].
    
    Parameters
    ----------
    models : list
        List of trained Keras models, one for each transmitter.
    input_test : np.ndarray
        Clean (non-adversarial) input samples for testing.
    epsilon : float
        Epsilon value used to generate adversarial examples.

    Returns
    -------
    df_corr : pd.DataFrame
        A one-row DataFrame with columns for each attack type (FGSM, PGD, BIM, MIM).
        The row contains the average Pearson correlation coefficients across all models.
    """
    # We'll use the same generate_adversarial_examples(...) function you have above
    # in your code to produce adversarial data.
    
    # Define the attacks we want to evaluate
    attacks = ['fgsm', 'pgd', 'bim']
    
    # Dictionary to hold average correlation per attack
    correlation_values = {}
    
    for attack_type in attacks:
        # Collect correlation coefficients for each transmitter, then average
        corrs_for_attack = []
        
        for model in models:
            # 1) Predict on clean data
            original_predictions = model.predict(input_test, verbose=0)
            
            # 2) Generate adversarial data
            adv_data = generate_adversarial_examples(model, input_test, epsilon, attack_type)
            
            # 3) Predict on adversarial data
            adv_predictions = model.predict(adv_data, verbose=0)
            
            # 4) Compute Pearson correlation between clean and adversarial predictions
            corr_value, _ = pearsonr(original_predictions.ravel(), adv_predictions.ravel())
            corrs_for_attack.append(corr_value)
        
        # Average correlation across all models (TX)
        correlation_values[attack_type.upper()] = np.mean(corrs_for_attack)
    
    # Create a single-row DataFrame with columns = attacks
    df_corr = pd.DataFrame([correlation_values])
    return df_corr


def plot_mse_distribution(standard_models, student_models, input_test, output_test, epsilons, num_beams=512, save_path="./plots"):
    """
    Plots KDE-filled histograms of MSE distributions for different epsilon values.
    Each plot includes only 2 distributions: one for the standard model and one for the student model.
    MSE values are computed as the average across all attacks (FGSM, BIM, PGD).
    
    Saves each plot as a separate PDF file.

    Parameters:
    -----------
    standard_models : list
        List of trained Keras models (standard models).
    student_models : list
        List of trained Keras models (student models).
    input_test : np.ndarray
        Test input data.
    output_test : np.ndarray
        Ground truth output data (full set covering all transmitters).
    epsilons : list
        List of epsilon values to use for adversarial attacks.
    num_beams : int
        Number of output neurons per model (512 in this case).
    save_path : str
        Directory where plots will be saved.

    Returns:
    --------
    None (saves plots as PDFs)
    """
    
    sns.set(style="whitegrid")
    os.makedirs(save_path, exist_ok=True)

    attacks = ['fgsm', 'bim', 'pgd']

    for epsilon in epsilons:
        mse_standard = []
        mse_student = []

        for tx_idx, (std_model, stu_model) in enumerate(zip(standard_models, student_models)):
            # Predictions for clean test data
            pred_standard = std_model.predict(input_test, verbose=0)
            pred_student = stu_model.predict(input_test, verbose=0)

            # Extract corresponding part of `output_test`
            output_test_subset = output_test[:, tx_idx * num_beams : (tx_idx + 1) * num_beams]

            # Initialize lists to store MSEs from different attacks
            mse_standard_attacks = []
            mse_student_attacks = []

            # Generate adversarial examples and compute MSE for each attack
            for attack in attacks:
                adv_data_standard = generate_adversarial_examples(std_model, input_test, epsilon, attack)
                pred_adv_standard = std_model.predict(adv_data_standard, verbose=0)

                adv_data_student = generate_adversarial_examples(stu_model, input_test, epsilon, attack)
                pred_adv_student = stu_model.predict(adv_data_student, verbose=0)

                # Compute MSE for this attack
                mse_standard_attacks.append(np.mean((pred_adv_standard - output_test_subset) ** 2, axis=1))
                mse_student_attacks.append(np.mean((pred_adv_student - output_test_subset) ** 2, axis=1))

            # Average MSE across attacks for each sample
            mse_standard.extend(np.mean(mse_standard_attacks, axis=0))
            mse_student.extend(np.mean(mse_student_attacks, axis=0))

        # Convert to numpy arrays
        mse_standard = np.array(mse_standard)
        mse_student = np.array(mse_student)

        # Create the plot
        plt.figure(figsize=(8, 5))
        sns.histplot(mse_standard, bins=50, color="red", label="Standard Model", log_scale=True, stat="count", element="step", fill=True, alpha=0.5)
        sns.histplot(mse_student, bins=50, color="blue", label="Student Model", log_scale=True, stat="count", element="step", fill=True, alpha=0.5)


        # Formatting
        plt.xlabel("MSE (log scale)")
        plt.ylabel("Count")
        plt.title(f"MSE Distribution (ε = {epsilon})")
        plt.legend()

        # Save as PDF
        filename = f"{save_path}/mse_distribution_epsilon_{epsilon}.pdf"
        plt.savefig(filename, bbox_inches="tight")
        plt.close()

        print(f"[*] Saved plot for epsilon {epsilon} as {filename}")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_mse_comparison(standard_models, student_models, input_test, output_test, epsilons, attacks=['fgsm', 'pgd', 'bim'], num_beams=512, save_path="./Chapter-3/Plots/mse_comparison.pdf"):
    """
    Generates adversarial examples using different epsilon values and compares 
    MSE for standard (undefended) and student (defended) models across attacks.
    
    Parameters:
    -----------
    standard_models : list
        List of trained Keras models (undefended models).
    student_models : list
        List of trained Keras models (defended models with distillation).
    input_test : np.ndarray
        Test input data.
    output_test : np.ndarray
        Ground truth output data (full set covering all transmitters).
    epsilons : list
        List of epsilon values to use for adversarial attacks.
    attacks : list
        List of attack types (default: ['fgsm', 'pgd', 'bim']).
    num_beams : int
        Number of output neurons per model (512 in this case).
    save_path : str
        File path to save the figure.

    Returns:
    --------
    None (saves the figure)
    """
    
    sns.set(style="whitegrid")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Storage for results
    results = []

    # Iterate through models (assumes one model per transmitter)
    for tx_idx, (std_model, stu_model) in enumerate(zip(standard_models, student_models)):
        # Extract corresponding part of `output_test`
        output_test_subset = output_test[:, tx_idx * num_beams : (tx_idx + 1) * num_beams]

        for epsilon in tqdm(epsilons, position=0, leave=True) :
            for attack in tqdm(attacks,position=1, leave=False):
                # Generate adversarial examples for each attack
                adv_data_standard = generate_adversarial_examples(std_model, input_test, epsilon, attack)
                pred_adv_standard = std_model.predict(adv_data_standard, verbose=0)

                adv_data_student = generate_adversarial_examples(stu_model, input_test, epsilon, attack)
                pred_adv_student = stu_model.predict(adv_data_student, verbose=0)

                # Compute MSE for standard and student models
                mse_standard = np.mean((pred_adv_standard - output_test_subset) ** 2)
                mse_student = np.mean((pred_adv_student - output_test_subset) ** 2)

                # Store results
                results.append(["Undefended", epsilon, attack, mse_standard])
                results.append(["Defensive Distillation", epsilon, attack, mse_student])

    # Convert to DataFrame
    results_df = pd.DataFrame(results, columns=["Model Type", "Epsilon", "Attack", "MSE"])

    # Define color palette
    num_epsilons = len(epsilons)
    epsilon_colors = sns.color_palette("muted", num_epsilons)

    # Create the plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    model_types = ["Undefended", "Defensive Distillation"]

    for ax, model in zip(axes, model_types):
        subset = results_df[results_df["Model Type"] == model]

        sns.barplot(
            data=subset,
            x="Attack",
            y="MSE",
            hue="Epsilon",
            palette=epsilon_colors,
            ax=ax
        )

        ax.set_title(model)
        ax.set_xlabel("Attacks")

    axes[0].set_ylabel("MSE")
    # fig.legend(title="Epsilon ($\epsilon$)", loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=num_epsilons)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    print(f"[*] Saved plot as {save_path}")

def generate_mse_table(standard_models, student_models, input_test, output_test, epsilons, attacks=['fgsm', 'pgd', 'bim'], num_beams=512, save_path="./Chapter-3/Plots/mse_table.tex"):
    """
    Generates a LaTeX table summarizing MSE results for adversarial attacks 
    on Undefended and Defensive Distillation models, including mean and std dev.

    Parameters:
    -----------
    standard_models : list
        List of trained Keras models (undefended models).
    student_models : list
        List of trained Keras models (defended models with distillation).
    input_test : np.ndarray
        Test input data.
    output_test : np.ndarray
        Ground truth output data (full set covering all transmitters).
    epsilons : list
        List of epsilon values to use for adversarial attacks.
    attacks : list
        List of attack types (default: ['fgsm', 'pgd', 'bim', 'mim']).
    num_beams : int
        Number of output neurons per model (512 in this case).
    save_path : str
        File path to save the LaTeX table.

    Returns:
    --------
    None (saves LaTeX table)
    """

    # Storage for results
    results = []

    # Iterate through models (assumes one model per transmitter)
    for tx_idx, (std_model, stu_model) in enumerate(zip(standard_models, student_models)):
        # Extract corresponding part of `output_test`
        output_test_subset = output_test[:, tx_idx * num_beams : (tx_idx + 1) * num_beams]

        for epsilon in epsilons:
            for attack in attacks:
                # Generate adversarial examples for each attack
                adv_data_standard = generate_adversarial_examples(std_model, input_test, epsilon, attack)
                pred_adv_standard = std_model.predict(adv_data_standard, verbose=0)

                adv_data_student = generate_adversarial_examples(stu_model, input_test, epsilon, attack)
                pred_adv_student = stu_model.predict(adv_data_student, verbose=0)

                # Compute MSE and standard deviation
                mse_standard = np.mean((pred_adv_standard - output_test_subset) ** 2)
                mse_student = np.mean((pred_adv_student - output_test_subset) ** 2)
                std_standard = np.std((pred_adv_standard - output_test_subset) ** 2)
                std_student = np.std((pred_adv_student - output_test_subset) ** 2)

                # Store results
                results.append([attack.upper(), epsilon, mse_standard, std_standard, mse_student, std_student])

    # Convert to DataFrame
    results_df = pd.DataFrame(results, columns=["Attack", "Epsilon", "Undefended MSE", "Undefended STD", "Defended MSE", "Defended STD"])

    # Convert to LaTeX format
    latex_code = results_df.to_latex(index=False, column_format="lcccccc", float_format="%.4f", caption="MSE and Standard Deviation for Undefended and Defensive Distillation Models.", label="tab:mse_results")

    # Save LaTeX table
    with open(save_path, "w") as f:
        f.write(latex_code)

    print(f"[*] LaTeX table saved to {save_path}")
# Main
def main(create_models = False):
    print("[*] Loading dataset...")
    input_data, output_data = load_dataset(INPUT_FILE, OUTPUT_FILE)

    print("[*] Splitting dataset...")
    input_train, input_test, output_train, output_test, test_indices = split_dataset(input_data, output_data)

    # min max values of the training data
    print(f"[*] Min values of the training data: {input_train.min()}")
    print(f"[*] Max values of the training data: {input_train.max()}")
    # print input_train shape
    print(f"[*] Shape of the training data: {input_train.shape}")

    nodes_per_layer = int(input_train.shape[1] / 10.0)

    if create_models:
        # print a seperator
        print("="*50)
        print("[*] Training standard models...")
        standard_models = train_models(
            input_train, output_train, input_test, output_test,
            num_hidden_layers=4,
            nodes_per_layer=nodes_per_layer,
            num_tx=NUM_TOTAL_TX,
            num_beams=NUM_BEAMS,
            nb_epoch=NB_EPOCH,
            batch_size=BATCH_SIZE,
            loss_function=LOSS_FUNCTION,
            train_mode=TRAIN_MODE,
            distillation=False
        )

        print("[*] Evaluating standard models...")
        standard_results = evaluate_models(standard_models, input_train, output_train, NUM_TOTAL_TX, NUM_BEAMS, EPSILONS, adversarial=False)

        print("[*] Computing Pearson correlation coefficients...")
        df_pearson = create_pearson_correlation_table(standard_models, input_test, epsilon=0.1)
        print(df_pearson)

        print("="*50)
        print("[*] Training teacher and student models...")
        student_models = train_models(
            input_train, output_train, input_test, output_test,
            num_hidden_layers=4,
            nodes_per_layer=nodes_per_layer,
            num_tx=NUM_TOTAL_TX,
            num_beams=NUM_BEAMS,
            nb_epoch=NB_EPOCH,
            batch_size=BATCH_SIZE,
            loss_function=LOSS_FUNCTION,
            train_mode=TRAIN_MODE, #True,
            distillation=True,
            temperature=50
        )
        

        print("[*] Evaluating adversarial models...")
        print("[*] Evaluating student models...")
        student_results = evaluate_models(student_models, input_train, output_train, NUM_TOTAL_TX, NUM_BEAMS, EPSILONS, adversarial=True)

        # Combine and save results
        combined_results = pd.concat([standard_results, student_results], ignore_index=True)
        # save into csv file
        combined_results.to_csv('combined_results.csv', index=False)

    print("[*] Loading standard models from disk...")
    standard_models = []
    for tx_idx in range(1, NUM_TOTAL_TX + 1):
        model_path = os.path.join(MODEL_FOLDER, f'TX{tx_idx}_undefended.keras')
        print(f"    Loading {model_path}...")
        loaded_model = load_model(model_path)
        standard_models.append(loaded_model)

    print("[*] Loading student models from disk...")
    student_models = []
    for tx_idx in range(1, NUM_TOTAL_TX + 1):
        model_path = os.path.join(MODEL_FOLDER, f'TX{tx_idx}_student.keras')
        print(f"    Loading {model_path}...")
        loaded_model = load_model(model_path)
        student_models.append(loaded_model)

    # Compute Pearson correlation table for standard models
    print("[*] Computing Pearson correlation table for standard models...")
    df_pearson_standard = create_pearson_correlation_table(standard_models, input_test, epsilon=EPSILONS[-1])
    print(df_pearson_standard)
    print("    Standard model results saved to pearson_correlation_results_standard.csv")

    # Compute Pearson correlation table for student models
    print("[*] Computing Pearson correlation table for student models...")
    df_pearson_student = create_pearson_correlation_table(student_models, input_test, epsilon=EPSILONS[-1])
    print(df_pearson_student)
    print("    Student model results saved to pearson_correlation_results_student.csv")

    generate_mse_table(standard_models, student_models, input_test, output_test, EPSILONS, save_path="./Chapter-3/Plots/mse_table.tex")

    plot_mse_comparison(standard_models, student_models, input_test, output_test, EPSILONS, save_path="./Chapter-3/Plots/mse_comparison.pdf")

    plot_mse_distribution(standard_models, student_models, input_test, output_test, epsilons=[0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5], save_path="./Chapter-3/Plots")

    # load combined_results.csv
    combined_results = pd.read_csv('combined_results.csv')

    # print random 10 samples
    print(combined_results.sample(10))
    # Assuming `combined_results` contains a 'Model Type' column
    plot_mse_vs_epsilon(combined_results)

    print(f"[*] finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
if __name__ == "__main__":
    main(create_models=False)
