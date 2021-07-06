import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

learn_rate = 0.00001
epoch = 20
batch_size = 32

DIR = r"C:\Users\Amol Gupta\Documents\face mask tf\face\dataset"
CATEGORIES = ["with_mask", "without_mask"]

data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DIR, category)
    for img in os.listdir(path):
    	img_path = os.path.join(path, img)
    	image = load_img(img_path, target_size=(224, 224))
    	image = img_to_array(image)
    	image = preprocess_input(image)

    	data.append(image)
    	labels.append(category)

LB = LabelBinarizer()
labels = LB.fit_transform(labels)
labels = to_categorical(labels)
labels = np.array(labels)
data = np.array(data, dtype="float32")


(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=36)
#augmentation
Aug_ = ImageDataGenerator(rotation_range=20,zoom_range=0.15,width_shift_range=0.2,
		height_shift_range=0.2,shear_range=0.15,horizontal_flip=True,fill_mode="nearest")

base_model = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

main_model = base_model.output
main_model = AveragePooling2D(pool_size=(7, 7))(main_model)
main_model = Flatten(name="flatten")(main_model)
main_model = Dense(128, activation="relu")(main_model)
main_model = Dropout(0.5)(main_model)
main_model = Dense(2, activation="softmax")(main_model)

model = Model(inputs=base_model.input, outputs=main_model)

for l in base_model.layers:
	l.trainable = False
# optimisation
output = Adam(lr=learn_rate, decay=learn_rate / epoch)
model.compile(loss="binary_crossentropy", optimizer =output, metrics=["accuracy"])
# training
H = model.fit(Aug_.flow(trainX, trainY, batch_size=batch_size), steps_per_epoch=len(trainX) // batch_size, 
		validation_data=(testX, testY), validation_steps=len(testX) // batch_size, epochs=epoch)
# testing/prediction
pred_indexs = model.predict(testX, batch_size=batch_size)

pred_indexs = np.argmax(pred_indexs, axis=1)

print(classification_report(testY.argmax(axis=1), pred_indexs, target_names=LB.classes_))

model.save("detect_mask.model", save_format="h5")

N = epoch
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="value loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Accuracy graph")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("graph.png")
