from flask import Flask, request, jsonify, render_template
from PIL import Image
import base64
import io
from fastcore.all import *
from fastai.vision.all import *

app = Flask(__name__)

# Load the model and create DataLoaders once
path = Path('bass_or_not')
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path, bs=32)
learn_loaded = vision_learner(dls, resnet18, metrics=error_rate)
learn_loaded.load("my_model")

def classify_image(img):
    is_bass, _, probs = learn_loaded.predict(PILImage.create(img))
    return is_bass

def preprocess_image(image):
    # Convert the image to RGB format
    preprocessed_image = image.convert('RGB')
    # Resize the image to the expected input size (for example, 224x224)
    preprocessed_image = preprocessed_image.resize((224, 224))
    return preprocessed_image


@app.route('/classify', methods=['POST'])
def classify():
    encoded_image = request.form['image']
    image_data = base64.b64decode(encoded_image)
    image = Image.open(io.BytesIO(image_data))
    preprocessed_image = preprocess_image(image)
    result = classify_image(preprocessed_image)
    return jsonify({'result': result})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
