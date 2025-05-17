from flask import Flask, render_template, request
from ultralytics import YOLO
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load the YOLOv8 model (ensure 'best.pt' is in the same directory)
model = YOLO('best.pt')

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    image_path = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            # Save uploaded image
            filename = file.filename
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)

            # Run YOLOv8 prediction
            results = model(image_path)

            # Save image with bounding boxes
            results[0].save(filename=os.path.join(app.config['UPLOAD_FOLDER'], 'result.jpg'))

            # Get detected class indices and names
            names = results[0].names  # Class names dictionary
            classes = results[0].boxes.cls.tolist()  # Detected class indices

            if classes:
                # Get unique class names
                unique_classes = set(int(cls) for cls in classes)
                result = ", ".join([names[cls] for cls in unique_classes])
            else:
                result = "Healthy"

            # Show the result image
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.jpg')

    return render_template('index.html', result=result, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
