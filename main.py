import numpy as np
import sys
from utils.data_loader import loadData
from utils.one_hot_encoding import oneHotEncoder
from utils.model_creator import make_or_restore_model
from utils.train import train
from utils.visualization import show_plot

# Encoding variables
input_shape_2D = (224, 224)
input_shape_3D = (224, 224, 3)
seed = 1
batch_size = 32
epochs = 63
X = []
y = []

# Loading Image Data as GrayScale/RGB
data = loadData(input_shape_2D, seed)
class_names = data.class_names

# Data Processing
for images, labels in data:
    X.append(images.numpy())
    y.append(labels.numpy())
# convert X , y list into numpy array
X = np.concatenate(X, axis=0)
y = np.concatenate(y, axis=0)

# Normalising Data
X = X.astype('float32') / 260

# Split data into train and test / validation
X_train, X_test = X[:100], X[100:]
y_train, y_test = y[:100], y[100:]

# importing One Hot Encoding
y_train, y_test = oneHotEncoder(class_names, y_train, y_test)

# Importing model # Build or restore model
model = make_or_restore_model(input_shape_3D)

# Asking the user if they want to train the model or not.
print("====================================================================================================")
train_model = str(input("Do you want to train the model? (Y/N): "))
print("====================================================================================================")
if ((train_model == "Y") or (train_model == "y")):
    train(model, X_train, y_train, batch_size, epochs, X_test, y_test)
    model = make_or_restore_model(input_shape_3D)
elif ((train_model == "N") or (train_model == "n")):
    # Build or restore model
    model = make_or_restore_model(input_shape_3D)
    print("====================================================================================================")
    print("Model is loaded successfully")
    print("====================================================================================================")
else:
    print("Invalid Input. Run the program again.")
    sys.exit(1)

# Accuracy
test_Accuracy = model.evaluate(X_test, y_test)
print(f"Model's Accuracy: {test_Accuracy[1] * 100}")

# Show Image
y_prediction = model.predict(X_test)
leaf_class = ['Bacterial leaf blight', 'Brown spot', 'Leaf smut']
show_plot(X_test, y_prediction, y_test, leaf_class)