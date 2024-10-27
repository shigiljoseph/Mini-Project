import os
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image, ImageFilter

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('modelb.keras')  

# Define allowed image file types
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Define function to preprocess image (resize and sharpen)
from PIL import ImageOps

def prepare_image(image_path):
    # Open the image using PIL
    img = Image.open(image_path)
    
    # If the image is grayscale, convert it to RGB
    if img.mode != 'RGB':
        img = ImageOps.grayscale(img)  
        img = img.convert('RGB') 
    
    # Resize the image to the expected input size of the model
    img = img.resize((260, 260))
    
    # Apply a sharpening filter
    img = img.filter(ImageFilter.SHARPEN)
    
    # Convert the image to an array
    img = img_to_array(img)
    
    # Expand dimensions to match model input
    img = np.expand_dims(img, axis=0)  
    
    # Normalize the pixel values to [0, 1]
    img /= 255.0
    
    return img

# Define function to predict image class
def predict_image(image_path):
    prepared_image = prepare_image(image_path)
    prediction = model.predict(prepared_image)[0][0]  

    # Print the prediction probability
    print(f"Prediction probability: {prediction:.2f}")
    if prediction==0:
        return f"Fractured (Accuracy: {prediction:.2f})"
    # If the prediction probability is less than 0.5, classify as "Fractured"
    elif prediction < 0.5:
        return f"Fractured (Accuracy: {1-prediction*100:.2f})"
    # Otherwise, classify as "Non Fractured"
    else:
        return f"Non Fractured (Accuracy: {prediction*100:.2f})"

# Define Flask route for homepage
@app.route('/')
def upload_form():
    return render_template('index.html')

# Define route for handling image upload
@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join('uploads', filename)
        file.save(filepath) 
        prediction = predict_image(filepath) 
        
        # Render the result page and pass the prediction and image URL
        return render_template('result.html', prediction=prediction, image_url=filename)

    return redirect(request.url)

# Define route to serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

# Run the Flask app
if __name__ == "__main__":
    # Ensure the 'uploads' directory exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
