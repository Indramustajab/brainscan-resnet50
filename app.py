from flask import Flask, render_template, request
from predict import predict_image
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    img_path = None
    scroll_to_result = False  # Trigger untuk otomatis scroll ke hasil setelah upload

    if request.method == "POST":
        if "image" in request.files:
            file = request.files["image"]
            if file.filename != "":
                path = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(path)
                
                # Memanggil fungsi AI prediksimu
                result, confidence = predict_image(path)
                img_path = path
                scroll_to_result = True

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        img_path=img_path,
        scroll_to_result=scroll_to_result
    )

if __name__ == "__main__":
    app.run()