import tkinter
import tkinter as tk
from tkinter.ttk import *
from tkinter import filedialog

import numpy as np

import tensorflow as tf
from keras_preprocessing.image import img_to_array
from keras_preprocessing.image import load_img
from tensorflow import keras

class_names = ['Adposhel', 'Agent', 'Allaple', 'Amonetize', 'Androm',
               'Autorun', 'BrowseFox', 'Dinwod', 'Elex', 'Expiro', 'Fasong',
               'HackKMS', 'Hlux', 'Injector', 'InstallCore',
               'MultiPlug', 'Neoreklami', 'Neshta', 'Other',
               'Regrun', 'Sality', 'Snarasite', 'Stantinko',
               'VBA', 'VBKrypt', 'Vilsel']



def printScore(predictions):
    for k in range(len(class_names)):
        # txt_arr[k].delete(0, tk.END)
        # txt_arr[k].insert(0, predictions[k])
        txt_arr[k].config(value=predictions[k])


def open():
    file = filedialog.askopenfilename()
    img = load_img(file, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = reconstructed_model.predict(img_array)
    score = tf.nn.softmax(predictions)
    printScore(predictions)
    if 100 * np.max(score) > 85:
        txt = "Файл пренадлежит к {} с {:.2f} вероятностью.".format(class_names[np.argmax(score)], 100 * np.max(score))
        Label(window, text=txt, font=("Robot", 14), bg="white").grid(column=1, row=10, columnspan=4)
    else:
        txt = "Файл пренадлежит к обычному ПО."
        Label(window, text=txt, font=("Roboto", 14), bg="white").grid(column=1, row=10, columnspan=4)


reconstructed_model = keras.models.load_model('trained')

window = tkinter.Tk()
window.title("NN")
window.config(bg="white")

count = 0

txt_arr = [tk.Entry, tk.Entry, tk.Entry, tk.Entry, tk.Entry,
           tk.Entry, tk.Entry, tk.Entry, tk.Entry, tk.Entry,
           tk.Entry, tk.Entry, tk.Entry, tk.Entry, tk.Entry,
           tk.Entry, tk.Entry, tk.Entry, tk.Entry, tk.Entry,
           tk.Entry, tk.Entry, tk.Entry, tk.Entry, tk.Entry,
           tk.Entry]

for i in range(9):
    for j in range(3):
        if i == 8 and j == 2:
            btn = tk.Button(window, text="Проверить", command=open, font=("Roboto", 14), relief=tk.GROOVE,
                            borderwidth=1,
                            bg="white", fg="black", )
            btn.grid(row=i, column=j * 2)
            continue
        tk.Label(window, text=class_names[count], font=("Roboto", 14), width=10, bg="white").grid(row=i, column=j * 2)
        txt_arr[count] = Progressbar(window, orient="horizontal", length=100, mode="determinate")
        txt_arr[count].grid(row=i, column=j * 2 + 1, pady=5, padx=5)

        # txt_arr[count] = tk.Entry(window, width=10, font=("Roboto", 14), relief=FLAT, borderwidth=1, bg="white",
        #                        fg="black")
        # txt_arr[count].grid(row=i, column=j * 2 + 1, padx=5, pady=5)

        count += 1

window.mainloop()