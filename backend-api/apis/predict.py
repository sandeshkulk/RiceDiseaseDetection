from utils.model_creator import make_or_restore_model
from flask import request, jsonify
import os
import numpy as np
from utils.preprocess_image import preprocess_image

model = make_or_restore_model()
leaf_class = ['Bacterial leaf blight', 'Brown spot', 'Leaf smut']

def predict_api(uploaded_image):
    """
    Predict the type of disease and accuracy for an uploaded image.

    Args:
        uploaded_image (werkzeug.datastructures.FileStorage):
            The uploaded image file.

    Returns:
        jsonify: JSON response containing the accuracy and type of disease predicted by the model.
            If an error occurs during prediction, an error message is included in the response.
    """
    try:
        if 'image' not in uploaded_image:
            return jsonify({'error': 'No image provided'})

        image = uploaded_image['image']

        image.save('temp_image.jpg')

        processed_image = preprocess_image('temp_image.jpg')

        predictions = model.predict(processed_image)

        accuracy = str((np.max(predictions))*100)
        disease_type = leaf_class[np.argmax(predictions)]

        os.remove('temp_image.jpg')

        return jsonify({'Accuracy': accuracy, 'Type of Disease': disease_type})
    except Exception as e:
        return jsonify({'error': str(e)})