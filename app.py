from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import torch
from torchvision.models import resnet50
from torchvision.transforms import transforms
from PIL import Image
from flask import render_template

app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load your pretrained model
model = resnet50(pretrained=False)  # Set pretrained=False if using custom weights
model.load_state_dict(torch.load('resnet50_model.pth'))  # Update with your model path
model.eval()

# Define image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# Load class labels from the file
with open('imagenet_classes.txt', 'r') as f:
    class_labels = [line.strip() for line in f]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Save and preprocess the image
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        image = Image.open(filepath).convert('RGB')
        input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

        # Predict using the model
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = outputs.max(1)  # Get the class index

        # Map the prediction to a label
        predicted_label = class_labels[predicted.item()]

        # Cleanup
        os.remove(filepath)

        return jsonify({'prediction': predicted_label})
    except Exception as e:
        return jsonify({'error': str(e)})
    
if __name__ == '__main__':
    app.run(debug=True)