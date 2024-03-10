
from tensorflow.keras import datasets, models, layers, backend as K
import matplotlib.pyplot as plt
import numpy as np


# Load and preprocess the data
(train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

"""
model = models.Sequential([
    layers.Flatten(input_shape=(28,28)),
    layers.Dense(128,activation='relu', input_shape=(784,)),
    layers.Dense(10)
    ])
"""

model = models.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu',input_shape=(28,28,1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dropout(0.2),
    layers.Dense(10,activation = 'softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, epochs=1, batch_size=32, validation_data=(test_images, test_labels))
test_loss, test_acc = model.evaluate(test_images, test_labels)




# Extract the weights of the first layer
first_layer_weights = model.layers[0].get_weights()[0]
# Normalize the weights
first_layer_weights = (first_layer_weights - first_layer_weights.min()) / (first_layer_weights.max() - first_layer_weights.min())
# Get the number of filters
num_filters = first_layer_weights.shape[-1]
# Set the number of columns for the subplot
num_columns = 6
num_rows = num_filters // num_columns + 1
# Create a figure for the filters
plt.figure(figsize=(num_columns*2, num_rows*2))
for i in range(num_filters):
    # Get the filter
    f = first_layer_weights[:, :, :, i]  
    # If it's a grayscale filter, you can just plot it
    if f.shape[2] == 1:
        f = np.squeeze(f, -1)  # Remove the last dimension
    # Add a subplot for each filter
    plt.subplot(num_rows, num_columns, i+1)
    plt.imshow(f, cmap='gray')  # You can experiment with different color maps (cmap)
    plt.axis('off')
# Show the plot
plt.show()




#Plot history
history_dict = history.history
acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()





#Make predictions
probability_model = models.Sequential([model, layers.Softmax()])
predictions = probability_model.predict(test_images)
predictions = np.argmax(predictions,axis=1)


"""
#Example model for regression task
import pandas as pd

#Get data
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(url, names=column_names,na_values='?', comment='\t',sep=' ', skipinitialspace=True)
dataset = raw_dataset.copy()
dataset = dataset.dropna()

dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'}) #change to one-hot encoding
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_features = train_dataset.copy().astype(np.float32)
test_features = test_dataset.copy().astype(np.float32)

train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')

normalizer = layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))

model = models.Sequential([
    normalizer,
    layers.Dense(1)
])

model.compile(optimizer='adam',loss='mean_absolute_error',metrics=['mean_absolute_error'])


history = model.fit(train_features,train_labels, epochs=100, validation_split = 0.2)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Error')

assert(0)
"""
