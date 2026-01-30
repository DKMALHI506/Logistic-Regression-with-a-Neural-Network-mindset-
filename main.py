# main.py
# Main script to train, test and predict

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from src.model import model, predict
from lr_utils import load_dataset  # Your dataset loader

# Load dataset
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# Flatten and normalize images
m_train = train_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]
train_set_x = train_set_x_orig.reshape(m_train, -1).T / 255.
test_set_x = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T / 255.

# Train the model
logistic_model = model(train_set_x, train_set_y, test_set_x, test_set_y,
                       num_iterations=2000, learning_rate=0.005, print_cost=True)

# Plot learning curve
plt.plot(np.squeeze(logistic_model['costs']))
plt.ylabel('Cost')
plt.xlabel('Iterations (per hundreds)')
plt.title(f"Learning rate = {logistic_model['learning_rate']}")
plt.show()

# Test on custom image
my_image = "my_image.jpg"  # Change to your image
fname = "images/" + my_image
image = np.array(Image.open(fname).resize((num_px, num_px)))
plt.imshow(image)
image = image / 255.
image = image.reshape((1, num_px * num_px * 3)).T
my_predicted_image = predict(logistic_model["w"], logistic_model["b"], image)
print("y = {}, your algorithm predicts a '{}' picture.".format(int(np.squeeze(my_predicted_image)), 
      classes[int(np.squeeze(my_predicted_image))].decode("utf-8")))
