import os
import scipy.io
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.metrics import confusion_matrix
from tqdm.keras import TqdmCallback
from tqdm import tqdm
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.tf2.attacks.basic_iterative_method import basic_iterative_method
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
import logging
from datetime import datetime
import random
import json

# --- Setup Logging -------------------------------------------------------------

def setup_logging(chapter_folder):
    log_dir = os.path.join(chapter_folder, "Logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(
        log_dir,
        f"execution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    logger.info("Logging setup complete. Log file: %s", log_file)
    return logger

# --- Constants ----------------------------------------------------------------
CHAPTER_FOLDER        = './Chapter-5'
DATA_URL              = (
    "https://github.com/ocatak/RadarSpectrumSensing-FL-AML/raw/main/converted_dataset.zip"
)
DATA_FILE             = os.path.join(CHAPTER_FOLDER, "converted_dataset.zip")
EXTRACTED_DATA_PATH   = os.path.join(CHAPTER_FOLDER, "data/convertedFolder")
MODEL_FOLDER          = os.path.join(CHAPTER_FOLDER, "Saved_Models")
RESULTS_FOLDER        = os.path.join(CHAPTER_FOLDER, "Results")
PLOTS_FOLDER          = os.path.join(CHAPTER_FOLDER, "Plots")
UNDEFENDED_MODEL_FILE = os.path.join(MODEL_FOLDER, "undefended_model.keras")
DEFENDED_MODEL_FILE   = os.path.join(MODEL_FOLDER, "robust_model.keras")

os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(PLOTS_FOLDER, exist_ok=True)

# Hyperparameters

EPS_VALUES  = np.array([13, 26, 39, 52, 64, 77, 90, 103, 115, 128]) / 255.0
# round 2 EPS_VALUES
EPS_VALUES = np.round(EPS_VALUES,2)
ATTACKS     = ['FGSM', 'BIM', 'PGD']
BATCH_SIZE  = 10
EPOCHS      = 2000000
PATIENCE    = 150
NUM_CLASSES = 3
CLASSES     = ["LTE", "5G", "Noise"]
NUM_RUNS    = 2

# Initialize logger
tf.random.set_seed(42)
logger = setup_logging(CHAPTER_FOLDER)

# --- Data Download & Extraction ----------------------------------------------
if not os.path.exists(DATA_FILE):
    logger.info("Downloading dataset...")
    os.system(f"wget -q -O {DATA_FILE} {DATA_URL}")
    logger.info("Downloaded: %s", DATA_FILE)
else:
    logger.info("Dataset exists: %s", DATA_FILE)

if not os.path.exists(EXTRACTED_DATA_PATH) or \
   not os.path.exists(os.path.join(EXTRACTED_DATA_PATH, "rcvdSpectrogram_1.mat")):
    logger.info("Extracting dataset...")
    os.system(f"unzip -q {DATA_FILE} -d {CHAPTER_FOLDER}")
    logger.info("Extraction complete.")
else:
    logger.info("Data already extracted.")

# --- U-Net Model Definition --------------------------------------------------

def conv_block(inp, filters):
    x = layers.Conv2D(filters, 3, padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def encoder_block(inp, filters):
    x = conv_block(inp, filters)
    return x, layers.MaxPooling2D()(x)


def decoder_block(inp, skip, filters):
    x = layers.Conv2DTranspose(filters, 2, strides=2, padding="same")(inp)
    x = layers.Concatenate()([x, skip])
    return conv_block(x, filters)


def build_unet(input_shape, num_classes):
    inp = layers.Input(shape=input_shape)
    x1, p1 = encoder_block(inp, 16)
    x2, p2 = encoder_block(p1, 32)
    x3, p3 = encoder_block(p2, 64)
    x4, p4 = encoder_block(p3, 128)
    b    = conv_block(p4, 256)
    d1   = decoder_block(b,  x4, 128)
    d2   = decoder_block(d1, x3,  64)
    d3   = decoder_block(d2, x2,  32)
    d4   = decoder_block(d3, x1,  16)
    out  = layers.Conv2D(num_classes, 1, activation="softmax")(d4)
    return models.Model(inp, out, name="U-Net")

# --- Data Loading & Preprocessing -------------------------------------------

def convert_to_int(label_file):
    mapping = {'Noise':2,'LTE':0,'NR':1,'Radar':2}
    with open(label_file) as f, open('tmp_labels.txt','w') as out:
        for line in f:
            for k,v in mapping.items():
                line = line.replace(k, str(v))
            out.write(line)
    return pd.read_csv('tmp_labels.txt', header=None).values


def load_and_preprocess_data(seed=42):
    logger.info("Loading data...")
    X, Y = [], []
    for i in tqdm(range(1, 500), desc="Files"):
        mat = scipy.io.loadmat(
            os.path.join(EXTRACTED_DATA_PATH, f"rcvdSpectrogram_{i}.mat")
        )['rcvdSpectrogram']
        lbl = convert_to_int(
            os.path.join(EXTRACTED_DATA_PATH, f"trueLabels_{i}.csv")
        )
        X.append(mat); Y.append(lbl)

    X = np.array(X, dtype=np.float32) / 255.0
    Y = np.array(Y, dtype=np.int32)
    x_tr,x_va,y_tr,y_va = train_test_split(
        X, Y, test_size=0.1, random_state=seed
    )
    y_tr_oh = tf.one_hot(y_tr, depth=NUM_CLASSES)
    y_va_oh = tf.one_hot(y_va, depth=NUM_CLASSES)
    logger.info("Data shapes: %s train, %s val", x_tr.shape, x_va.shape)
    # min max values printing
    logger.info("X min: %.4f, max: %.4f", x_tr.min(), x_tr.max())
    logger.info("Y min: %d, max: %d", y_tr.min(), y_tr.max())

    return x_tr, x_va, y_tr, y_va, y_tr_oh, y_va_oh

# --- Adversarial Example Generation -----------------------------------------

def generate_adversarial_examples(model, x, y, epsilon, attack_type, num_iterations=10):
    """
    Generate adversarial examples using FGSM, BIM, or PGD.
    x : numpy or tf.Tensor of shape (batch, H, W, C)
    y : numpy or tf.Tensor of shape (batch, H, W) or (batch,) for labels
    epsilon : float (perturbation magnitude in [0,1] scale)
    attack_type : one of 'FGSM', 'BIM', 'PGD'
    num_iterations : number of iterations for BIM/PGD
    """
    # logger.info("Generating adversarial examples with %s (epsilon=%.5f)", attack_type, epsilon)
    # Ensure tensors
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.int32)

    batch_size = tf.shape(x)[0]
    n = tf.minimum(200, batch_size)
    idx = tf.random.shuffle(tf.range(batch_size))[:n]
    x = tf.gather(x, idx)
    y = tf.gather(y, idx)

    def loss_fn(labels, logits):
        # labels are integer class values per pixel
        one_hot = tf.one_hot(labels, depth=NUM_CLASSES)
        return tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(one_hot, logits)
        )

    clip_min, clip_max = 0.0, 1.0
    eps = float(epsilon) / 1.0

    if attack_type == 'FGSM':
        adv = fast_gradient_method(
            model_fn=model,
            x=x,
            eps=eps,
            norm=2, #np.inf,
            y=y,
            loss_fn=loss_fn,
            targeted=False,
            clip_min=clip_min,
            clip_max=clip_max
        )

    elif attack_type in ['PGD', 'BIM']:
        adv = projected_gradient_descent(
            model_fn=model,
            x=x,
            eps=eps,
            eps_iter=eps / num_iterations,
            nb_iter=num_iterations,
            norm=2, #np.inf,
            y=y,
            loss_fn=loss_fn,
            targeted=False,
            clip_min=clip_min,
            clip_max=clip_max
        )
    else:
        raise ValueError(f"Unsupported attack type: {attack_type}")

    adv_x = adv.numpy() if isinstance(adv, tf.Tensor) else adv
    # print out the each distance between the original and adversarial examples 
    #for i in range(adv_x.shape[0]):
    #    dist = np.linalg.norm(adv_x[i] - x[i])
    #    logger.info("Distance between original and adversarial example %d: %.4f", i, dist)
    #    logger.info("Adversarial example %d: min: %.4f, max: %.4f", i, adv_x[i].min(), adv_x[i].max())
    #    logger.info("Original example %d: min: %.4f, max: %.4f", i, x[i].numpy().min(), x[i].numpy().max())
    #    # print epsilon value
    #    logger.info("Epsilon value: %.4f", eps)
    # print out the min and max values of the adversarial examples
    #logger.info("Adversarial examples min: %.4f, max: %.4f", adv_x.min(), adv_x.max())


    # logger.info("Adversarial examples generated")
    return adv_x

# --- Adversarial Training ---------------------------------------------------

def train_model_adversarial(x_tr, y_tr_oh, x_va, y_va_oh, y_tr_lbl):
    # Set up TensorBoard log directories
    # tb-env/bin/tensorboard --logdir=Chapter-5/Logs/tensorboard --port=6006

    tb_base = os.path.join(CHAPTER_FOLDER, 'Logs', 'tensorboard')
    os.makedirs(tb_base, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    tb_log_dir_undef = os.path.join(tb_base, f"undef_{ts}")
    tb_log_dir_def = os.path.join(tb_base, f"def_{ts}")

    model_undefended = build_unet(x_tr.shape[1:], NUM_CLASSES)
    model_undefended.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(1e-4),
        loss='categorical_crossentropy', metrics=['accuracy']
    )

    cbks_undefended = [
        TensorBoard(log_dir=tb_log_dir_undef, histogram_freq=1),
        TqdmCallback(verbose=1, desc="Undefended-Training"),
        ModelCheckpoint(UNDEFENDED_MODEL_FILE, monitor='loss', save_best_only=True),
        EarlyStopping(monitor='loss', patience=PATIENCE, restore_best_weights=True)
    ]

    # if model exists, load weights
    if os.path.exists(UNDEFENDED_MODEL_FILE):
        logger.info("Loading existing model weights...")
        model_undefended.load_weights(UNDEFENDED_MODEL_FILE)

    # Train on augmented data
    #model_undefended.fit(
    #    x_tr, y_tr_oh,
    #    batch_size=BATCH_SIZE, epochs=EPOCHS,
    #    validation_data=(x_va, y_va_oh),
    #    callbacks=cbks_undefended, verbose=0
    #)

    #model_undefended.save(UNDEFENDED_MODEL_FILE)
    logger.info("Undefended model saved: %s", UNDEFENDED_MODEL_FILE)
    # show the model metrics
    logger.info("Undefended model metrics: %s", model_undefended.evaluate(x_va, y_va_oh, verbose=0))

    ## defended model

        # if adversarial_examples.npz exists, load the data into x_adv_list and y_adv_lbl_list
    adv_data_file = os.path.join(CHAPTER_FOLDER, 'adversarial_examples.npz')
    if os.path.exists(adv_data_file):
        logger.info("Loading existing adversarial examples...")
        adv_data = np.load(adv_data_file)
        x_adv_list = [adv_data['x']]
        y_adv_lbl_list = [adv_data['y']]
    else:
        logger.info("Generating adversarial examples...")        
        x_adv_list, y_adv_lbl_list = [], []
        subset_size = min(50, x_tr.shape[0])
        # generate once across all attacks and eps
        for attack in ATTACKS:
            for eps in EPS_VALUES:
                logger.info("Generating adversarials: %s @ eps=%.3f on %d samples", attack, eps, subset_size)
                idx = random.sample(range(x_tr.shape[0]), subset_size)
                x_sub = x_tr[idx]
                y_sub_lbl = y_tr_lbl[idx]
                # Generate adversarial on subset
                x_adv = generate_adversarial_examples(
                    model_undefended, x_sub, y_sub_lbl, eps, attack
                )
                x_adv_list.append(x_adv)
                y_adv_lbl_list.append(y_sub_lbl)
        # save adversarials and labels
        np.savez(
            adv_data_file,
            x=np.concatenate(x_adv_list, axis=0),
            y=np.concatenate(y_adv_lbl_list, axis=0)
        )
        # Merge clean + adversarial
    x_aug = np.concatenate([x_tr] + x_adv_list, axis=0)
    y_aug = tf.concat([y_tr_oh] + [tf.one_hot(lbl, depth=NUM_CLASSES) for lbl in y_adv_lbl_list], axis=0)

    # Shuffle
    idx = np.arange(len(x_aug))
    np.random.shuffle(idx)
    x_aug, y_aug = x_aug[idx], tf.gather(y_aug, idx)

    # Callbacks

    idx = np.arange(len(x_aug))
    np.random.shuffle(idx)
    x_aug, y_aug = x_aug[idx], tf.gather(y_aug, idx)

    # Callbacks
    cbks = [
        TensorBoard(log_dir=tb_log_dir_def, histogram_freq=1),
        TqdmCallback(verbose=1, desc="Adv-Training"),
        ModelCheckpoint(DEFENDED_MODEL_FILE, monitor='loss', save_best_only=True),
        EarlyStopping(monitor='loss', patience=PATIENCE, restore_best_weights=True)
    ]

    model = build_unet(x_tr.shape[1:], NUM_CLASSES)

    # if the undefendedmodel exists, load weights to the defended model
    model.load_weights(UNDEFENDED_MODEL_FILE)
    logger.info("Defended model loaded from: %s", UNDEFENDED_MODEL_FILE)

    # if the defended model exists, load weights to the defended model
    if os.path.exists(DEFENDED_MODEL_FILE):
        logger.info("Loading existing model weights...")
        model.load_weights(DEFENDED_MODEL_FILE)

    # Get all layers in the model including nested ones
    flat_layers = []
    for layer in model.layers:
        if hasattr(layer, 'layers'):
            flat_layers.extend(layer.layers)
        else:
            flat_layers.append(layer)
    
    # Freeze first 1/3 of layers (mostly encoder)
    freeze_count = len(flat_layers) // 3
    for i, layer in enumerate(flat_layers):
        if i < freeze_count:
            if hasattr(layer, 'trainable'):
                layer.trainable = False
                logger.info(f"Freezing layer {i}: {layer.name}")


    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(1e-4),
        loss='categorical_crossentropy', metrics=['accuracy']
    )

    

    # Train on augmented data
    #model.fit(
    #    x_aug, y_aug,
    #    batch_size=BATCH_SIZE, epochs=EPOCHS,
    #    validation_data=(x_va, y_va_oh),
    #    callbacks=cbks, verbose=0
    #)

    #model.save(DEFENDED_MODEL_FILE)
    logger.info("Adversarially-trained model saved: %s", DEFENDED_MODEL_FILE)
    # show the model metrics
    logger.info("Defended model metrics: %s", model.evaluate(x_va, y_va_oh, verbose=0))
    return model, model_undefended

# --- Metrics & Plotting -----------------------------------------------------

def compute_metrics(y_true, y_pred, classes):
    y_t, y_p = y_true.flatten(), y_pred.flatten()
    cm = confusion_matrix(y_t, y_p, labels=range(len(classes)))
    # plot this confusion matrix with the percentages in the plots
    #cm_percetage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #cm_percetage = np.round(cm_percetage, 2)
    #print(f"Confusion Matrix:\n{cm_percetage}")
    #print(f"original:\n{cm}")
    #import matplotlib.pyplot as plt
    #import seaborn as sns
    #plt.figure(figsize=(8, 6))
    #sns.heatmap(cm_percetage, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes)
    #plt.xlabel('Predicted')
    #plt.ylabel('True')
    #plt.title('Confusion Matrix')
    #plt.savefig(os.path.join(PLOTS_FOLDER, 'confusion_matrix.png'), bbox_inches='tight')
    #plt.show()

    mets = {m: [] for m in ['Accuracy','Recall','Precision','Specificity','F-Score','FPR','IoU']}
    for i in range(len(classes)):
        TP = cm[i,i]; FP = cm[:,i].sum()-TP
        FN = cm[i,:].sum()-TP; TN = cm.sum()-(TP+FP+FN)
        acc = (TP+TN)/(TP+TN+FP+FN)
        rec = TP/(TP+FN) if TP+FN else 0
        prec= TP/(TP+FP) if TP+FP else 0
        spec= TN/(TN+FP) if TN+FP else 0
        fsc = 2*(prec*rec)/(prec+rec) if prec+rec else 0
        fpr = FP/(FP+TN) if FP+TN else 0
        iou = TP/(TP+FP+FN) if TP+FP+FN else 0
        for k,v in zip(mets.keys(), [acc,rec,prec,spec,fsc,fpr,iou]): mets[k].append(v)
    return mets, {k: np.mean(v) for k,v in mets.items()}

# --- Experiment Runner ------------------------------------------------------

def save_results(init, perf, iou, folder):
    with open(os.path.join(folder,'Results','results.json'),'w') as f:
        json.dump({'initial':init,'performance':perf,'iou':iou}, f, indent=2)
    logger.info("Results saved.")


def run_experiment(seed, create_models=True):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 1) Load data
    x_tr, x_va, y_tr, y_va, y_tr_oh, y_va_oh = load_and_preprocess_data(seed)

    # 2) Train or load both models
    if create_models:
        defended_model, undefended_model = train_model_adversarial(
            x_tr, y_tr_oh, x_va, y_va_oh, y_tr
        )
    else:
        logger.info("Loading saved models…")
        defended_model   = tf.keras.models.load_model(DEFENDED_MODEL_FILE)
        undefended_model = tf.keras.models.load_model(UNDEFENDED_MODEL_FILE)

    # 3) Clean (initial) evaluation
    y_pred_undef = np.argmax(
        undefended_model.predict(x_tr, callbacks=[TqdmCallback(desc="Undef-Eval")], verbose=0), axis=-1
    )

    _, init_undef = compute_metrics(y_tr, y_pred_undef, CLASSES)

    y_pred_def = np.argmax(
        defended_model.predict(x_tr, callbacks=[TqdmCallback(desc="Def-Eval")], verbose=0), axis=-1
    )
    _, init_def = compute_metrics(y_tr, y_pred_def, CLASSES)

    initial_metrics = {
        "Undefended": init_undef,
        "Defended":   init_def
    }

    # 4) Adversarial evaluation
    performance = {"Undefended": {}, "Defended": {}}
    iou_results = {"Undefended": {}, "Defended": {}}

    # Load precomputed adversarial examples
    adv_path = os.path.join(CHAPTER_FOLDER, 'adversarial_examples.npz')
    if not os.path.exists(adv_path):
        raise FileNotFoundError(f"Missing adversarial file: {adv_path}")
    adv_data = np.load(adv_path)
    x_adv_all = adv_data['x']  # shape (N_total, H, W, C)
    y_adv_all = adv_data['y']  # shape (N_total, ...) matching y labels

    # Compute per-batch size: equal chunks for each attack & epsilon
    n_attacks = len(ATTACKS)
    n_eps = len(EPS_VALUES)
    total = x_adv_all.shape[0]
    chunk = total // (n_attacks * n_eps)

    idx = 0
    for name, model in [("Undefended", undefended_model), ("Defended", defended_model)]:
        performance[name] = {atk: {} for atk in ATTACKS}
        iou_results[name] = {atk: {} for atk in ATTACKS}
        idx = 0
        for atk in ATTACKS:
            for eps in EPS_VALUES:
                logger.info("Evaluating %s model with %s attack at eps=%.3f", name, atk, eps)
                x_adv = x_adv_all[idx: idx + chunk]
                y_adv_lbl = y_adv_all[idx: idx + chunk]
                idx += chunk

                # Predict and evaluate
                y_pred = np.argmax(
                    model.predict(x_adv, verbose=0), axis=-1
                )
                _, avg_adv = compute_metrics(y_adv_lbl, y_pred, CLASSES)

                performance[name][atk][eps] = avg_adv
                iou_results[name][atk][eps] = avg_adv['IoU']


    return initial_metrics, performance, iou_results


def visualize_results(results_file, output_folder):
    """
    Load saved JSON results and generate summary tables and plots for both Undefended and Defended models.
    """
    import os
    import json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # Load results
    with open(results_file, 'r') as f:
        data = json.load(f)
    initial_list   = data['initial']       # list of {Undefended: {...}, Defended: {...}}
    performance_list = data['performance'] # list of {Undefended: {attack: {eps: {...}}}, Defended: {...}}
    iou_list         = data['iou']         # list of {Undefended: {attack: {eps: iou}}, Defended: {...}}

    # 1) Initial Performance Table (one row per run per model)
    init_records = []
    for run_idx, init in enumerate(initial_list):
        for model_name, metrics in init.items():
            rec = {'run': run_idx, 'model': model_name}
            rec.update(metrics)
            init_records.append(rec)
    df_init = pd.DataFrame(init_records)
    # Save CSV
    init_csv = os.path.join(output_folder, 'initial_performance.csv')
    df_init.to_csv(init_csv, index=False)
    logger.info("Saved initial performance table to %s", init_csv)
    # Boxplot
    plt.figure()
    df_init.boxplot(column=['Accuracy','F-Score','IoU'], by='model')
    plt.title('Initial Performance by Model')
    plt.suptitle('')
    plt.savefig(os.path.join(output_folder, 'initial_performance.pdf'), bbox_inches='tight')
    plt.close()

    # 2) Performance Summary (eps=13,64,128)
    perf_records = []
    for run_idx, perf in enumerate(performance_list):
        for model_name, atk_map in perf.items():
            for attack, eps_map in atk_map.items():
                for eps_str, metrics in eps_map.items():
                    eps = float(eps_str)
                    # record normalized key epsilons
                    rec = {'run': run_idx, 'model': model_name,
                            'attack': attack, 'epsilon': eps}
                    rec.update(metrics)
                    perf_records.append(rec)
    df_perf = pd.DataFrame(perf_records)
    perf_csv = os.path.join(output_folder, 'performance_summary.csv')
    df_perf.to_csv(perf_csv, index=False)
    logger.info("Saved performance summary table to %s", perf_csv)
    # Stats
    summary = df_perf.groupby(['model','attack','epsilon']).agg({
        'Accuracy': ['mean','std'],
        'F-Score': ['mean','std'],
        'IoU': ['mean','std']
    }).reset_index()
    # flatten columns
    summary.columns = ['model','attack','epsilon',
                       'Accuracy_mean','Accuracy_std',
                       'F-Score_mean','F-Score_std',
                       'IoU_mean','IoU_std']
    stats_csv = os.path.join(output_folder, 'performance_summary_stats.csv')
    summary.to_csv(stats_csv, index=False)
    logger.info("Saved performance stats to %s", stats_csv)

        # 3) IoU Raw and Summary
    iou_records = []
    for run_idx, iou_data in enumerate(iou_list):
        for model_name, atk_map in iou_data.items():
            for attack, eps_map in atk_map.items():
                for eps_str, val in eps_map.items():
                    eps = float(eps_str)
                    iou_records.append({
                        'run': run_idx,
                        'model': model_name,
                        'attack': attack,
                        'epsilon': eps,
                        'IoU': val
                    })
    df_iou = pd.DataFrame(iou_records)
    iou_csv = os.path.join(output_folder, 'iou_values.csv')
    df_iou.to_csv(iou_csv, index=False)
    logger.info("Saved IoU values to %s", iou_csv)
    # summary
    iou_summary = df_iou.groupby(['model','attack','epsilon'])['IoU'].agg(['mean','std']).reset_index()
    iou_summary.columns = ['model','attack','epsilon','IoU_mean','IoU_std']
    iou_stats_csv = os.path.join(output_folder, 'iou_summary.csv')
    iou_summary.to_csv(iou_stats_csv, index=False)
    logger.info("Saved IoU summary to %s", iou_stats_csv)
    # Plot by model
    for model_name in df_iou['model'].unique():
        plt.figure()
        sub = iou_summary[iou_summary['model']==model_name]
        for attack in sub['attack'].unique():
            s2 = sub[sub['attack']==attack]
            plt.errorbar(s2['epsilon'], s2['IoU_mean'], yerr=s2['IoU_std'],
                         fmt='-o', capsize=4, label=attack)
        plt.xlabel('Epsilon')
        plt.ylabel('IoU')
        plt.title(f'IoU vs Epsilon ({model_name})')
        plt.legend()
        plt.grid(True)
        plot_file = os.path.join(output_folder, f'iou_vs_epsilon_{model_name}.pdf')
        plt.savefig(plot_file, bbox_inches='tight')
        plt.close()
        logger.info("Saved IoU plot for %s to %s", model_name, plot_file)

    # 4) Line plots: Clean vs Adversarial performance
    import seaborn as sns
    # Melt initial performance
    df_clean = df_init.melt(
        id_vars=['run','model'],
        value_vars=['Accuracy','F-Score','IoU'],
        var_name='metric', value_name='value'
    )
    df_clean['attack'] = 'Clean'
    df_clean['epsilon'] = 0.0
    # Melt adversarial performance for eps=13,64,128
    df_adv = df_perf.melt(
        id_vars=['run','model','attack','epsilon'],
        value_vars=['Accuracy','F-Score','IoU'],
        var_name='metric', value_name='value'
    )
    # Combine
    df_comb = pd.concat([df_clean, df_adv], axis=0)
    df_comb.sort_values(['metric','epsilon'], inplace=True)

    for metric in ['Accuracy','F-Score','IoU']:
        plt.figure()
        sns.lineplot(
            data=df_comb[df_comb['metric']==metric],
            x='epsilon', y='value', hue='attack', style='model', markers=True, dashes=False
        )
        plt.title(f'{metric} vs Epsilon (Clean & Adversarial)')
        plt.xlabel('Epsilon')
        plt.ylabel(metric)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Attack / Model')
        plt.tight_layout()
        plot_fp = os.path.join(output_folder, f'{metric}_vs_epsilon_lineplot.pdf')
        plt.savefig(plot_fp, bbox_inches='tight')
        plt.close()
        logger.info("Saved %s lineplot to %s", metric, plot_fp)


    # Plot by model
    for model_name in df_iou['model'].unique():
        plt.figure()
        sub = iou_summary[iou_summary['model']==model_name]
        for attack in sub['attack'].unique():
            s2 = sub[sub['attack']==attack]
            plt.errorbar(s2['epsilon'], s2['IoU_mean'], yerr=s2['IoU_std'],
                         fmt='-o', capsize=4, label=attack)
        plt.xlabel('Epsilon')
        plt.ylabel('IoU')
        plt.title(f'IoU vs Epsilon ({model_name})')
        plt.legend()
        plt.grid(True)
        plot_file = os.path.join(output_folder, f'iou_vs_epsilon_{model_name}.pdf')
        plt.savefig(plot_file, bbox_inches='tight')
        plt.close()
        logger.info("Saved IoU plot for %s to %s", model_name, plot_file)

    perf_csv = os.path.join(PLOTS_FOLDER, 'performance_summary.csv')
    if os.path.exists(perf_csv):
        plot_performance_from_csv(perf_csv, PLOTS_FOLDER)

def plot_performance_from_csv(perf_csv, output_folder):
    """
    Read performance_summary.csv and generate publication-quality PDF violin+swarm plots and lineplots for Accuracy and F-Score.
    """
    import os
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator

    # Styling for publication
    sns.set_context('paper', font_scale=1.5)
    sns.set_style('whitegrid')
    plt.rcParams['axes.grid'] = True

    # Load data
    df = pd.read_csv(perf_csv)
    df['epsilon'] = pd.to_numeric(df['epsilon'], errors='coerce')

    figsize = (6, 4)
    dpi = 300

    # 1) Violin + swarm plots for each metric and attack
    for metric in ['Accuracy', 'F-Score']:
        for attack in df['attack'].unique():
            df_att = df[df['attack'] == attack]
            plt.figure(figsize=figsize)
            ax = sns.violinplot(
                x='epsilon', y=metric, hue='model',
                data=df_att, split=True, inner='quartile', palette='Set2'
            )
            sns.swarmplot(
                x='epsilon', y=metric, hue='model',
                data=df_att, dodge=True, color='k', alpha=0.6, size=3
            )
            # Keep single legend
            handles, labels = ax.get_legend_handles_labels()
            n = len(df_att['model'].unique())
            ax.legend(handles[:n], labels[:n], title='Model')

            # Major & minor grid
            ax.grid(which='major', linestyle='--', linewidth=0.7)
            ax.grid(which='minor', linestyle=':', linewidth=0.5)
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())

            ax.set_title(f'{metric} under {attack} Attack', fontsize=16, pad=12)
            ax.set_xlabel('Epsilon', fontsize=14)
            ax.set_ylabel(metric, fontsize=14)
            ax.tick_params(axis='both', which='major', labelsize=12)

            plt.tight_layout()
            out_file = os.path.join(
                output_folder,
                f'{attack.lower()}_{metric.lower()}_violin.pdf'
            )
            plt.savefig(out_file, dpi=dpi)
            plt.close()
            logger.info("Saved publication-quality violin plot for %s under %s to %s", metric, attack, out_file)

    # 2) Lineplots: mean metric vs epsilon
    df_mean = df.groupby(['model', 'attack', 'epsilon'])[['Accuracy', 'F-Score']].mean().reset_index()
    for metric in ['Accuracy', 'F-Score']:
        for model_name in df_mean['model'].unique():
            plt.figure(figsize=figsize)
            ax = sns.lineplot(
                data=df_mean[df_mean['model'] == model_name],
                x='epsilon', y=metric,
                hue='attack', marker='o', dashes=False, palette='Set1'
            )
            ax.set_title(f'{metric} vs Epsilon ({model_name})', fontsize=16, pad=12)
            ax.set_xlabel('Epsilon', fontsize=14)
            ax.set_ylabel(metric, fontsize=14)
            ax.tick_params(axis='both', which='major', labelsize=12)

            # Grids
            ax.grid(which='major', linestyle='--', linewidth=0.7)
            ax.grid(which='minor', linestyle=':', linewidth=0.5)
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())

            ax.legend(title='Attack', fontsize=12, title_fontsize=13, loc='best')
            plt.tight_layout()
            out_file = os.path.join(
                output_folder,
                f'{model_name.lower()}_{metric.lower()}_line.pdf'
            )
            plt.savefig(out_file, dpi=dpi)
            plt.close()
            logger.info("Saved publication-quality line plot for %s of model %s to %s", metric, model_name, out_file)

def main_multiple_runs(create_models=False):
    all_init, all_perf, all_iou = [], [], []
    for run in range(NUM_RUNS):
        init, perf, iou = run_experiment(42+run, create_models if run==0 else False)
        all_init.append(init); all_perf.append(perf); all_iou.append(iou)
    save_results(all_init, all_perf, all_iou, CHAPTER_FOLDER)

if __name__ == "__main__":
    # main_multiple_runs(create_models=True)
    visualize_results(os.path.join(CHAPTER_FOLDER, 'Results', 'results.json'), PLOTS_FOLDER)
