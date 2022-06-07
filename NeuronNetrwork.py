import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras

class_names = ['Adposhel', 'Agent', 'Allaple', 'Amonetize', 'Androm',
               'Autorun', 'BrowseFox', 'Dinwod', 'Elex', 'Expiro', 'Fasong',
               'HackKMS', 'Hlux', 'Injector', 'InstallCore',
               'MultiPlug', 'Neoreklami', 'Neshta', 'Other',
               'Regrun', 'Sality', 'Snarasite', 'Stantinko',
               'VBA', 'VBKrypt', 'Vilsel']

train_directory = 'malevis_train_val_224x224/train/'
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_directory,
    labels="inferred",
    label_mode="int",
    class_names=class_names,
    color_mode="rgb",
    batch_size=32,
    image_size=(224, 224),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    smart_resize=False,
)

val_directory = 'malevis_train_val_224x224/val/'
val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    val_directory,
    labels="inferred",
    label_mode="int",
    class_names=class_names,
    color_mode="rgb",
    batch_size=32,
    image_size=(224, 224),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    smart_resize=False,
)
print(train_dataset.take(1).__class__)

normalization_layer = keras.layers.preprocessing.Rescaling(1. / 255, input_shape=(224, 224, 3))

train_ds = train_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
model = tf.keras.Sequential([
    normalization_layer,
    keras.layers.Flatten(input_shape=(224, 224, 3)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(len(class_names))
])

model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

history = model.fit(
    train_ds,
    validation_data=val_dataset,
    epochs=50
)

model.save("trained")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(5)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Точность при тренировке')
plt.plot(epochs_range, val_acc, label='Точность проверочной выборки')
plt.legend(loc='lower right')
plt.title('Точность тренировки и валидации')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Неточность при тренировке')
plt.plot(epochs_range, val_loss, label='Неточность проверки')
plt.legend(loc='upper right')
plt.title('Тренировочная и проверочная неточноость')
plt.show()