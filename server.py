from flask import Flask, render_template, request
import numpy as np
import base64
import io
from PIL import Image
from tensorflow.keras.models import load_model
from prepare_data import normalize
import json

# 1. Initialize the Flask application
app = Flask(__name__)

# 2. Load your trained "brain"
try:
    # Make sure this filename matches exactly what you saved in train.py
    conv = load_model("my_model.h5") 
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Make sure 'my_model.h5' is in the same folder as this script.")

FRUITS = {
    0: "Apple", 1: "Banana", 2: "Grape", 3: "Pineapple",
    4: "Watermelon", 5: "Strawberry", 6: "Pear", 7: "Blackberry",
    8: "Blueberry", 9: "Broccoli", 10: "Mushroom", 11: "Carrot",
    12: "Peas", 13: "Potato", 14: "Asparagus"
}

# 3. Define the routes
@app.route("/", methods=["GET", "POST"])
def ready():
    if request.method == "GET":
        return render_template("index1.html")
    
    if request.method == "POST":
        # Get the drawing data and the selected network from the browser
        payload = request.form["payload"].split(",")[1]
        net = request.form.get("net", "ConvNet") # Defaults to ConvNet if not found
        
        # Decode and process the image
        img_data = base64.b64decode(payload)
        img = Image.open(io.BytesIO(img_data)).convert('L') 
        
        # Resize to 28x28 (what the AI expects)
        img = img.resize((28, 28))
        x = np.array(img)
        
        # Check if canvas is empty (all or mostly white)
        # If pixel values are high (close to 255), canvas is empty
        empty_pixels = np.sum(x > 200)
        total_pixels = x.shape[0] * x.shape[1]
        if empty_pixels / total_pixels > 0.95:  # If >95% of pixels are white
            return render_template("index1.html", 
                                   chart=True,
                                   putback=request.form["payload"],
                                   net=net,
                                   error_message="Please draw something first!",
                                   empty_canvas=True)

        # Invert and Brighten (AI models usually learn white-on-black)
        x = np.invert(x)
        x = x.astype('float32') 
        x[x > 50] = np.minimum(255, x[x > 50] * 1.6)

        # Shape for ConvNet: (Batch, Height, Width, Channels)
        x = x.reshape(1, 28, 28, 1)

        # Normalize pixels to -1 to 1 range
        x = normalize(x)
        
        # Ask the AI for a prediction
        val = conv.predict(x)
        max_confidence = float(np.max(val))
        pred_index = np.argmax(val)
        pred = FRUITS[pred_index]
        
        # Debug: Print predictions to console
        print(f"\n=== PREDICTION DEBUG ===")
        print(f"Top prediction: {pred} ({max_confidence:.2%} confidence)")
        for i, fruit in FRUITS.items():
            print(f"  {fruit}: {val[0][i]:.2%}")
        print("=======================\n")
        
        # Check if prediction confidence is too low (unknown object)
        confidence_threshold = 0.85  # Strict threshold - requires 85% confidence for valid prediction
        if max_confidence < confidence_threshold:
            return render_template("index1.html",
                                   chart=True,
                                   putback=request.form["payload"],
                                   net=net,
                                   error_message=f"I dont recognize this drawing. Please draw one of my known items: {', '.join(FRUITS.values())}",
                                   unknown_object=True)
        
        # Data for the chart
        classes = [FRUITS[i] for i in range(len(FRUITS))]
        clean_preds = [float(p) for p in val[0]]
        
        return render_template("index1.html", 
                               preds=clean_preds, 
                               classes=json.dumps(classes), 
                               chart=True, 
                               putback=request.form["payload"], 
                               net=net)

# 4. Start the server
if __name__ == "__main__":
    app.run(debug=True)