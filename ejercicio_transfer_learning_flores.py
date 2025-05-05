import os
import sys
import math
import time
import cv2
import datetime
import zoneinfo
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import keras
import pathlib
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Flatten, Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
import keras_tuner as kt
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from PIL import Image
import io
from tqdm import tqdm
import glob

os.makedirs("figures", exist_ok=True)

# Configuraciones para la Busqueda de Hiperparametros
TUNING_MAX_EPOCHS = 10
OBJECTIVE = "val_accuracy"
FACTOR = 3
EPT = 1
N_EPOCHS = 25
BATCH_SIZE = 256

# Configuraciones para el Procesamiento de Imagenes
IMG_RESIZE_W_H = 224
TFREC_ROOT = "/content"
OUTPUT_ROOT = "/content/flowers_by_name"

# Listado de Clases (Fuente: Kaggle)
CLASSES = ["pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea", "wild geranium",
           "tiger lily", "moon orchid", "bird of paradise", "monkshood", "globe thistle",
           "snapdragon", "colt's foot", "king protea", "spear thistle", "yellow iris",
           "globe-flower", "purple coneflower", "peruvian lily", "balloon flower", "giant white arum lily",
           "fire lily", "pincushion flower", "fritillary", "red ginger", "grape hyacinth",
           "corn poppy", "prince of wales feathers", "stemless gentian", "artichoke", "sweet william",
           "carnation", "garden phlox", "love in the mist", "cosmos", "alpine sea holly",
           "ruby-lipped cattleya", "cape flower", "great masterwort", "siam tulip", "lenten rose",
           "barberton daisy", "daffodil", "sword lily", "poinsettia", "bolero deep blue",
           "wallflower", "marigold", "buttercup", "daisy", "common dandelion",
           "petunia", "wild pansy", "primula", "sunflower", "lilac hibiscus",
           "bishop of llandaff", "gaura", "geranium", "orange dahlia", "pink-yellow dahlia",
           "cautleya spicata", "japanese anemone", "black-eyed susan", "silverbush", "californian poppy",
           "osteospermum", "spring crocus", "iris", "windflower", "tree poppy",
           "gazania", "azalea", "water lily", "rose", "thorn apple",
           "morning glory", "passion flower", "lotus", "toad lily", "anthurium",
           "frangipani", "clematis", "hibiscus", "columbine", "desert-rose",
           "tree mallow", "magnolia", "cyclamen ", "watercress", "canna lily",
           "hippeastrum ", "bee balm", "pink quill", "foxglove", "bougainvillea",
           "camellia", "mallow", "mexican petunia", "bromelia", "blanket flower",
           "trumpet creeper", "blackberry lily", "common tulip", "wild rose"]


feature_description_with_class = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "class": tf.io.FixedLenFeature([], tf.int64),
    "id": tf.io.FixedLenFeature([], tf.string),
}

feature_description_without_class = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "id": tf.io.FixedLenFeature([], tf.string),
}

def _parse_function(example_proto, has_class=True):
    feature_description = feature_description_with_class if has_class else feature_description_without_class
    return tf.io.parse_single_example(example_proto, feature_description)

def convert_split(split):
    tfrecord_files = glob.glob(os.path.join(TFREC_ROOT, split, "*.tfrec"))
    split_output_dir = os.path.join(OUTPUT_ROOT, split)
    os.makedirs(split_output_dir, exist_ok=True)

    has_class = split in ["train", "val"]

    for tfrec_file in tfrecord_files:
        raw_dataset = tf.data.TFRecordDataset(tfrec_file)
        parsed_dataset = raw_dataset.map(lambda x: _parse_function(x, has_class))

        for example in tqdm(parsed_dataset, desc=f"[{split}] {os.path.basename(tfrec_file)}"):
            image_bytes = example["image"].numpy()
            image_id = example["id"].numpy().decode()

            if has_class:
                class_id = example["class"].numpy()
                class_name = CLASSES[class_id].strip().replace(" ", "_").replace("/", "_")
            else:
                class_name = "unknown"

            class_dir = os.path.join(split_output_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

            image = tf.image.decode_jpeg(image_bytes, channels=3)
            image = tf.image.resize(image, (IMG_RESIZE_W_H, IMG_RESIZE_W_H))
            image = tf.cast(image, tf.uint8).numpy()

            img = Image.fromarray(image)
            img.save(os.path.join(class_dir, f"{image_id}.jpg"))

for split in ["train", "val", "test"]:
    convert_split(split)

AUTOTUNE = tf.data.AUTOTUNE

train_dataset_dir = pathlib.Path(OUTPUT_ROOT + "/train")
val_dataset_dir = pathlib.Path(OUTPUT_ROOT + "/val")
test_dataset_dir = pathlib.Path(OUTPUT_ROOT + "/test")

train_dataset = keras.utils.image_dataset_from_directory(
    train_dataset_dir,
    image_size=(IMG_RESIZE_W_H, IMG_RESIZE_W_H),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

val_dataset = keras.utils.image_dataset_from_directory(
    val_dataset_dir,
    validation_split=0.5,
    subset="training",
    seed=42,
    image_size=(IMG_RESIZE_W_H, IMG_RESIZE_W_H),
    batch_size=BATCH_SIZE
)

test_dataset = keras.utils.image_dataset_from_directory(
    val_dataset_dir,
    validation_split=0.5,
    subset="validation",
    seed=42,
    image_size=(IMG_RESIZE_W_H, IMG_RESIZE_W_H),
    batch_size=BATCH_SIZE
)

data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.2),
    keras.layers.RandomZoom(0.2),
    keras.layers.RandomContrast(0.2)
])

def preprocess_train(image, label):
    image = data_augmentation(image)
    image = preprocess_input(image)
    return image, label

def preprocess_eval(image, label):
    image = preprocess_input(image)
    return image, label

train_ds = train_dataset.map(preprocess_train, num_parallel_calls=AUTOTUNE)
val_ds = val_dataset.map(preprocess_eval, num_parallel_calls=AUTOTUNE)
test_ds = test_dataset.map(preprocess_eval, num_parallel_calls=AUTOTUNE)

train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)
test_ds = test_ds.cache().prefetch(AUTOTUNE)

print(f"Lotes del Conjunto de Entrenamiento: {tf.data.experimental.cardinality(train_ds).numpy()}")
print(f"Lotes del Conjunto de Validacion: {tf.data.experimental.cardinality(val_ds).numpy()}")
print(f"Lotes del Conjunto de Prueba: {tf.data.experimental.cardinality(test_ds).numpy()}")

def show_sample_images(dataset, title, n=5):
    plt.figure(figsize=(12, 3))
    for images, labels in dataset.take(1):
        for i in range(n):
            ax = plt.subplot(1, n, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(CLASSES[labels[i]])
            plt.axis("off")
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
plt.savefig(os.path.join("figures", "output_figure_1.png"))

show_sample_images(train_ds, "Muestras del Conjunto de Entrenamiento (con Aumentacion)")
show_sample_images(val_ds, "Muestras del Conjunto de Validacion (sin Aumentacion)")
show_sample_images(test_ds, "Muestras del Conjunto de Prueba (sin Aumentacion)")

class_counts = Counter(np.concatenate([y.numpy() for _, y in train_ds], axis=0))
total_count = sum(class_counts.values())

labels = sorted(class_counts.keys())
counts = [class_counts[label] for label in labels]
class_labels = [CLASSES[label] for label in labels]

fig1, ax1 = plt.subplots(figsize=(24, 6))
sns.barplot(x=class_labels, y=counts, ax=ax1, palette="rocket")

ax1.set_title("Distribucion de Clases")
ax1.set_xlabel("Clase (Tipo de Flor)")
ax1.set_ylabel("Cantidad")

plt.xticks(rotation=90, fontsize=8)

for i, count in enumerate(counts):
    percentage = (count / total_count) * 100
    ax1.text(i, count + 5, f"{percentage:.1f}%", ha="center", va="bottom", fontsize=6)

plt.tight_layout()
plt.savefig(os.path.join("figures", "output_figure_2.png"))

early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)

def tl_model(hp):

    inputs = Input(shape=(IMG_RESIZE_W_H, IMG_RESIZE_W_H, 3))

    backbone = EfficientNetV2B0(include_top=False, input_tensor=inputs, weights="imagenet")
    backbone.trainable = False

    x = backbone.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(hp.Int("units_1", min_value=32, max_value=48, step=16), activation="relu", name="dense_1", kernel_regularizer=l2(0.001))(x)
    x = Dropout(hp.Float("dropout_1", min_value=0.05, max_value=0.1, step=0.05))(x)
    outputs = Dense(104, activation="softmax", name="output")(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model

tl_tuner = kt.Hyperband(
    tl_model,
    objective=OBJECTIVE,
    max_epochs=TUNING_MAX_EPOCHS,
    factor=FACTOR,
    directory="tl_tuner",
    executions_per_trial = EPT,
    project_name="transfer_learning_flower_classifier",
    overwrite = True
)

tl_tuner.search(train_ds, epochs=N_EPOCHS, validation_data=val_ds, batch_size=BATCH_SIZE, callbacks=[early_stopping])
best_tl_model = tl_tuner.get_best_models(num_models=1)[0]

best_tl_model.summary()

print(f"Dropout en la Capa Densa al Final: {best_tl_model.layers[-2].name} :", best_tl_model.layers[-2].rate)

metrics_history_tl = best_tl_model.fit(train_ds, epochs=N_EPOCHS, validation_data=val_ds, verbose=1, batch_size=BATCH_SIZE)

metrics_history_list = [metrics_history_tl]
best_models_labels = ["Transfer Learning (EfficientNetV2B0)"]

fig, axs = plt.subplots(2, 1, figsize=(16, 7))
axs_pairs = [[0,0]]
epochs_range = range(N_EPOCHS)

for ax_pair, metrics_history, method in zip(axs_pairs, metrics_history_list, best_models_labels):

    accuracy = metrics_history.history["accuracy"]
    val_accuracy = metrics_history.history["val_accuracy"]
    loss = metrics_history.history["loss"]
    val_loss = metrics_history.history["val_loss"]

    axs[0].plot(epochs_range, accuracy, label=f"Entrenamiento {method}")
    axs[0].plot(epochs_range, val_accuracy, label=f"Validacion {method}")
    axs[0].legend(loc="upper left")
    axs[0].set_title(f"Precision de Entrenamiento y Validacion (Historial {method})", fontsize=8)

    axs[1].plot(epochs_range, loss, label=f"Entrenamiento {method}")
    axs[1].plot(epochs_range, val_loss, label=f"Validacion {method}")
    axs[1].legend(loc="upper left")
    axs[1].set_title(f"Perdida de Entrenamiento y Validacion (History {method})", fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join("figures", "output_figure_3.png"))

best_models = [best_tl_model]
test_sets = [test_ds]

best_models_acc = []

for model, method_label, test_set in zip(best_models, best_models_labels, test_sets):
  test_loss, test_accuracy = model.evaluate(test_set, verbose=2)
  best_models_acc.append(test_accuracy)
  print(f"Test accuracy of the best model for the {method_label} method: {test_accuracy:.4f}\n")

y_pred_probs = best_tl_model.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=1)
y_pred

y_true = []

for _, labels in test_ds:
    y_true.extend(labels.numpy())

y_true = np.array(y_true)
y_true

print("Tamaños de arreglos de Predicciones y Etiquetas Reales de Prueba:", y_pred.shape, y_true.shape)

cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))

cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)

used_labels = sorted(list(set(y_true) | set(y_pred)))
used_names = [CLASSES[i] for i in used_labels]
cm_display = cm_normalized[np.ix_(used_labels, used_labels)]

plt.figure(figsize=(16, 14))
sns.heatmap(cm_display, xticklabels=used_names, yticklabels=used_names, cmap="YlGnBu", annot=False, fmt=".2f", cbar_kws={"label": "Porcentaje"})

plt.xlabel("Etiqueta Predecida")
plt.ylabel("Etiqueta Real (Validacion)")
plt.title("Matriz de Confusion")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join("figures", "output_figure_4.png"))

best_tl_model.export("tl_flower_classifier_lfmm")