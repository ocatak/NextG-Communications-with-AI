import os
import numpy as np
import scipy.io
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import mean_squared_error
from tqdm.keras import TqdmCallback
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.tf2.attacks.basic_iterative_method import basic_iterative_method
from cleverhans.tf2.attacks.momentum_iterative_method import momentum_iterative_method


# Constants
CHAPTER_FOLDER = './Chapter-4'
DATA_URL = "https://github.com/ocatak/6g-channel-estimation-dataset/blob/main/data.mat?raw=true"
DATA_FILE = os.path.join(CHAPTER_FOLDER, "data.mat")
MODEL_FOLDER = os.path.join(CHAPTER_FOLDER, "Saved_Models")
os.makedirs(MODEL_FOLDER, exist_ok=True)  # Ensure model folder exists

TEACHER_MODEL_FILE = os.path.join(MODEL_FOLDER, "teacher_model.keras")
STUDENT_MODEL_FILE = os.path.join(MODEL_FOLDER, "student_model.keras")
UNDEFENDED_MODEL_FILE = os.path.join(MODEL_FOLDER, "undefended_model.keras")
CSV_FILE = os.path.join(CHAPTER_FOLDER, "channel_estimation.csv")

# Hyperparameters
EPS_VALUES = np.arange(0.2, 3, 0.3).round(2)
ATTACKS = ['FGSM', 'BIM', 'MIM', 'PGD']
BATCH_SIZE = 128
EPOCHS = 10000
STUDENT_MULTIPLIER = 1.0
TEACHER_MULTIPLIER = 1.0
TEMPERATURE = 20


# Helper Functions
def build_cnn(multiplier, name):
    """Builds a CNN model."""
    model = models.Sequential(name=name)
    model.add(layers.Conv2D(int(48 * multiplier), (9, 9), padding='same', activation='selu', input_shape=(612, 14, 1)))
    model.add(layers.Conv2D(int(16 * multiplier), (5, 5), padding='same', activation='softplus'))
    model.add(layers.Conv2D(1, (5, 5), padding='same', activation='selu'))
    return model


def train_models(input_train, output_train, input_test, output_test,
                 num_hidden_layers, nodes_per_layer, nb_epoch, batch_size,
                 loss_function, train_mode, distillation=False, temperature=10):
    """Trains models with an option for defensive distillation (EXACTLY SAME AS CHAPTER-3)."""
    input_shape = input_train.shape[1:]

    if distillation:
        # Train Teacher Model
        print("[*] Training Teacher Model...")
        teacher_model = models.Sequential([
            layers.Input(shape=input_shape),  # Fix input shape
            layers.Conv2D(48, (9, 9), padding='same', activation='selu'),
            layers.Conv2D(16, (5, 5), padding='same', activation='softplus'),
            layers.Conv2D(1, (5, 5), padding='same', activation='selu'),
            layers.Flatten(),  # Flatten before passing to dense layers
            layers.Dense(int(nodes_per_layer * 2), activation='relu'),
            *[layers.Dense(int(nodes_per_layer * 2), activation='relu') for _ in range(num_hidden_layers)],
            layers.Dense(output_train.shape[1], activation='relu')
        ])
        teacher_model.compile(optimizer='adam', loss=loss_function, metrics=['mean_squared_error'])

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=200, restore_best_weights=True)
        teacher_model.fit(
                            input_train,
                            output_train,  # ✅ Already reshaped in `main()`
                            batch_size=batch_size, epochs=nb_epoch, verbose=0,
                            validation_data=(input_test, output_test),  # ✅ Already reshaped in `main()`
                            callbacks=[TqdmCallback(verbose=1), early_stopping]
                        )

        teacher_model.save(TEACHER_MODEL_FILE)
        print(f"[*] Teacher Model saved: {TEACHER_MODEL_FILE}")

        teacher_predictions = teacher_model.predict(input_train, verbose=0)

        # Train Student Model using Teacher Predictions
        print("[*] Training Student Model...")
        student_model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(48, (9, 9), padding='same', activation='selu'),
            layers.Conv2D(16, (5, 5), padding='same', activation='softplus'),
            layers.Conv2D(1, (5, 5), padding='same', activation='selu'),
            layers.Flatten(),
            layers.Dense(nodes_per_layer, activation='relu'),
            *[layers.Dense(nodes_per_layer, activation='relu') for _ in range(num_hidden_layers)],
            layers.Dense(output_train.shape[1], activation='relu'),
            layers.Activation('relu')
        ])
        student_model.compile(optimizer='adam', loss=loss_function, metrics=['mean_squared_error'])

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
        student_model.fit(
                            input_train, teacher_predictions.reshape(teacher_predictions.shape[0], teacher_predictions.shape[1]),  # Match shape
                            batch_size=batch_size, epochs=nb_epoch, verbose=0,
                            validation_data=(input_test, output_test.reshape(output_test.shape[0], output_test.shape[1])),  # Match shape
                            callbacks=[TqdmCallback(verbose=1), early_stopping]
                        )


        student_model.save(STUDENT_MODEL_FILE)
        print(f"[*] Student Model saved: {STUDENT_MODEL_FILE}")

        return teacher_model, student_model
    else:
        # Train Standard Model
        print("[*] Training Standard Model...")
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(48, (9, 9), padding='same', activation='selu'),
            layers.Conv2D(16, (5, 5), padding='same', activation='softplus'),
            layers.Conv2D(1, (5, 5), padding='same', activation='selu'),
            layers.Flatten(),
            layers.Dense(nodes_per_layer, activation='relu'),
            *[layers.Dense(nodes_per_layer, activation='relu') for _ in range(num_hidden_layers)],
            layers.Dense(output_train.shape[1], activation='relu')
        ])
        model.compile(optimizer='adam', loss=loss_function, metrics=['mean_squared_error'])

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
        model.fit(input_train, output_train, batch_size=batch_size, epochs=nb_epoch, verbose=0,
                  validation_data=(input_test, output_test), callbacks=[TqdmCallback(verbose=1), early_stopping])
        model.save(UNDEFENDED_MODEL_FILE)
        print(f"[*] Standard Model saved: {UNDEFENDED_MODEL_FILE}")

        return model

def generate_mse_table_latex(undefended_model, teacher_model, student_model, val_data, val_labels, save_path="mse_table.tex"):
    """Generates a LaTeX table for initial MSE performance (EXACTLY SAME AS CHAPTER-3)."""

    mse_undefended = mean_squared_error(val_labels.reshape(-1), undefended_model.predict(val_data, verbose=0).reshape(-1))
    mse_teacher = mean_squared_error(val_labels.reshape(-1), teacher_model.predict(val_data, verbose=0).reshape(-1))
    mse_student = mean_squared_error(val_labels.reshape(-1), student_model.predict(val_data, verbose=0).reshape(-1))

    results_df = pd.DataFrame({
        "Model": ["Undefended", "Teacher", "Student"],
        "MSE": [mse_undefended, mse_teacher, mse_student]
    })

    latex_code = results_df.to_latex(index=False, column_format="lc", float_format="%.4f",
                                     caption="Initial performance results (MSE) with test (i.e., benign) dataset.",
                                     label="tab:initial_mse")

    save_path = os.path.join(CHAPTER_FOLDER, save_path)
    with open(save_path, "w") as f:
        f.write(latex_code)

    print(f"[*] LaTeX table saved to {save_path}")

def generate_adversarial_examples(model, x, epsilon, attack_type):
    """
    Generates adversarial examples for a given model and dataset using CleverHans.
    
    Parameters:
    -----------
    model : tf.keras.Model
        The trained model for which adversarial examples will be generated.
    x : np.ndarray
        The benign input samples.
    epsilon : float
        The perturbation magnitude.
    attack_type : str
        The attack type ('FGSM', 'BIM', 'PGD').
    
    Returns:
    --------
    adv_x : np.ndarray
        The adversarially perturbed input samples.
    """
    
    if attack_type == 'FGSM':
        adv_x = fast_gradient_method(model, x, epsilon, norm=np.inf)

    elif attack_type == 'BIM':
        adv_x = basic_iterative_method(model, x, epsilon, eps_iter=0.01, nb_iter=100, norm=np.inf, targeted=False)

    elif attack_type == 'PGD':
        adv_x = projected_gradient_descent(model, x, epsilon, eps_iter=0.01, nb_iter=100, norm=np.inf, targeted=False)

    elif attack_type == 'MIM':
        adv_x = momentum_iterative_method(model, x, epsilon, eps_iter=0.01, nb_iter=100, norm=np.inf, targeted=False)
    else:
        raise ValueError(f"Unsupported attack type: {attack_type}")

    return adv_x.numpy()  # Convert to NumPy array

def generate_asr_table(models, input_test, output_test, epsilons, attacks, save_path="asr_table.tex"):
    """
    Generates a LaTeX table summarizing MSE results and Attack Success Ratio (ASR) 
    for different adversarial attacks on a given model.

    Parameters:
    -----------
    models : dict
        Dictionary containing {'undefended': model1, 'student': model2} trained models.
    input_test : np.ndarray
        Test input data.
    output_test : np.ndarray
        Ground truth output data (full set covering all transmitters).
    epsilons : list
        List of epsilon values to use for adversarial attacks.
    attacks : list
        List of attack types (e.g., ['FGSM', 'BIM', 'MIM', 'PGD']).
    save_path : str
        File path to save the LaTeX table.

    Returns:
    --------
    None (saves LaTeX table)
    """
    
    results = []
    
    for attack in tqdm(attacks, desc="Processing Attacks", leave=True, position=0):
        # for attack in attacks:
        for epsilon in tqdm(epsilons, desc=f"Processing {attack}", position=1, leave=False):
            benign_mse_list = []
            malicious_mse_list = []
            asr_list = []
            
            for model_name, model in tqdm(models.items(), desc="Processing Models", position=2, leave=False):
                # Compute benign MSE
                pred_clean = model.predict(input_test, verbose=0)
                benign_mse = mean_squared_error(output_test.reshape(-1), pred_clean.reshape(-1))
                benign_mse_list.append(benign_mse)
                
                # ✅ FIXED: Removed extra argument (output_test)
                adv_data = generate_adversarial_examples(model, input_test, epsilon, attack)
                
                # Compute malicious MSE
                pred_adv = model.predict(adv_data, verbose=0)
                malicious_mse = mean_squared_error(output_test.reshape(-1), pred_adv.reshape(-1))
                malicious_mse_list.append(malicious_mse)
                
                # Compute ASR
                asr = np.mean((malicious_mse - benign_mse) / malicious_mse)
                asr_list.append(asr)
            
            # Store results
            results.append([attack.upper(), epsilon, benign_mse_list[0], malicious_mse_list[0], asr_list[0]])

    # Convert to DataFrame
    df_results = pd.DataFrame(results, columns=["Attack", "Epsilon", "Benign Input MSE", "Malicious Input MSE", "ASR"])

    # Convert to LaTeX format
    latex_code = df_results.to_latex(index=False, column_format="llccc", float_format="%.4f",
                                     caption="Mean Squared Error (MSE) and Attack Success Ratio (ASR) under adversarial attacks.",
                                     label="tab:asr_results")

    # Save LaTeX table
    with open(save_path, "w") as f:
        f.write(latex_code)

    print(f"[*] ASR LaTeX table saved to {save_path}")

def plot_mse_comparison(undefended_model, student_model, input_test, output_test, attacks, epsilons, chapter_folder, save_filename="mse_comparison.pdf"):
    """
    Generates a grouped bar plot comparing MSE values across adversarial attacks for 
    undefended and defensive distillation models. Saves the plot in CHAPTER_FOLDER/Plots.

    Parameters:
    -----------
    undefended_model : tf.keras.Model
        The trained undefended model.
    student_model : tf.keras.Model
        The trained defensive distillation model.
    input_test : np.ndarray
        Test input data.
    output_test : np.ndarray
        Ground truth output data.
    attacks : list
        List of attack names (e.g., ['BIM', 'FGSM', 'MIM', 'PGD']).
    epsilons : list
        List of epsilon values used for adversarial attacks.
    chapter_folder : str
        Base directory where the "Plots" folder should be created.
    save_filename : str
        Name of the file to save the figure as.

    Returns:
    --------
    None (saves the figure as a PDF).
    """

    # Ensure the plots directory exists
    plots_dir = os.path.join(chapter_folder, "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Set plot style
    sns.set(style="whitegrid")

    # Compute MSE values
    mse_results = []

    for attack in tqdm(attacks, desc="Processing attacks"):
        for epsilon in tqdm(epsilons, desc=f"Generating {attack} adversarial examples", leave=False):
            
            # Generate adversarial examples for both models
            adv_undefended = generate_adversarial_examples(undefended_model, input_test, epsilon, attack)
            # adv_student = generate_adversarial_examples(student_model, input_test, epsilon, attack)
            
            # copy adv_undefended into adv_student
            adv_student = adv_undefended.copy()
            
            # Predict outputs
            pred_undefended = undefended_model.predict(adv_undefended, verbose=0)
            pred_student = student_model.predict(adv_student, verbose=0)

            # Compute MSE
            mse_undefended = mean_squared_error(output_test.reshape(-1), pred_undefended.reshape(-1))
            mse_student = mean_squared_error(output_test.reshape(-1), pred_student.reshape(-1))

            # Store results
            mse_results.append(["Undefended", attack, epsilon, mse_undefended])
            mse_results.append(["Defensive Distillation", attack, epsilon, mse_student])

    # Convert results to DataFrame
    df = pd.DataFrame(mse_results, columns=["Model", "Attack", "Epsilon", "MSE"])

    # Create the plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    model_types = ["Undefended", "Defensive Distillation"]

    for ax, model in zip(axes, model_types):
        subset = df[df["Model"] == model]

        sns.barplot(
            data=subset,
            x="Attack",
            y="MSE",
            hue="Epsilon",
            palette="muted",
            ax=ax
        )

        ax.set_title(model)
        ax.set_xlabel("Attack Methods")

    axes[0].set_ylabel("MSE")
    fig.legend(title="Epsilon (ε)", loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=len(epsilons))

    plt.tight_layout()
    
    # Save the figure
    save_path = os.path.join(plots_dir, save_filename)
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    print(f"[*] MSE comparison plot saved to {save_path}")

def main(create_models=False):
    """Main function to train/load models and evaluate them under adversarial attacks."""
    
    print("[*] Loading dataset...")
    if not os.path.exists(DATA_FILE):
        os.system(f"wget -q -O {DATA_FILE} {DATA_URL}")
    data = scipy.io.loadmat(DATA_FILE)

    input_data = data['trainData'].transpose((3, 0, 1, 2))
    output_data = data['trainLabels'].transpose((3, 0, 1, 2))

    split_ratio = 0.8
    num_samples = input_data.shape[0]
    split_index = int(num_samples * split_ratio)

    input_train, input_test = input_data[:split_index], input_data[split_index:]
    output_train, output_test = output_data[:split_index], output_data[split_index:]

    output_train = output_train.reshape(output_train.shape[0], output_train.shape[1] * output_train.shape[2])
    output_test = output_test.reshape(output_test.shape[0], output_test.shape[1] * output_test.shape[2])

    print(f"[*] Dataset loaded. Train Shape: {input_train.shape}, Test Shape: {input_test.shape}")

    global TRAIN_MODELS
    TRAIN_MODELS = create_models

    nodes_per_layer = int(input_train.shape[1] / 10.0)

    if create_models:
        teacher_model, student_model = train_models(input_train, output_train, input_test, output_test,
                                                    num_hidden_layers=4, nodes_per_layer=nodes_per_layer,
                                                    nb_epoch=EPOCHS, batch_size=BATCH_SIZE,
                                                    loss_function='mean_squared_error',
                                                    train_mode=TRAIN_MODELS, distillation=True, temperature=50)
    else:
        teacher_model = tf.keras.models.load_model(TEACHER_MODEL_FILE)
        student_model = tf.keras.models.load_model(STUDENT_MODEL_FILE)

    plot_mse_comparison(teacher_model, student_model, input_test, output_test, ATTACKS, EPS_VALUES, CHAPTER_FOLDER)

    generate_mse_table_latex(teacher_model, teacher_model, student_model, input_test, output_test, save_path="mse_table.tex")

    # Define trained models
    models = {
        "undefended": teacher_model,
        "student": student_model,
    }

    # Define epsilon values and attack methods
    epsilons = [0.1, 0.5, 1.0, 2.0, 3.0]
    attacks = ["FGSM", "BIM", "MIM", "PGD"]

    # Generate ASR table
    generate_asr_table(models, input_test, output_test, epsilons, attacks, save_path="./Chapter-4/asr_table.tex")

    print("[*] Finished processing.")


if __name__ == "__main__":
    main(create_models=False)
