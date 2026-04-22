from flask import Flask, render_template, request, jsonify
from predict import load_model
from PIL import Image
import os
import base64
import io

app = Flask(__name__)

# Load model 1 lần
model = load_model("model")

# Mapping tên bệnh sang tiếng Việt
disease_map = {
    "Early_blight": "Bệnh mốc sớm",
    "Late_blight": "Bệnh mốc muộn",
    "Leaf_Mold": "Bệnh mốc lá",
    "Septoria_leaf_spot": "Đốm lá Septoria",
    "Spider_mites": "Nhện đỏ",
    "Target_Spot": "Đốm mục tiêu",
    "Yellow_Leaf_Curl_Virus": "Virus xoăn lá vàng",
    "Mosaic_virus": "Virus khảm",
    "Healthy": "Khỏe mạnh"
}

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Dự đoán từ file upload"""
    try:
        if "file" not in request.files:
            return jsonify({"error": "Không có file"}), 400
        
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Tên file rỗng"}), 400
        
        # Đọc file từ memory
        image = Image.open(file.stream)
        result = model.predict(image)
        
        # Đổi tên sang tiếng Việt
        result['predicted_class_vi'] = disease_map.get(
            result['predicted_class'], 
            result['predicted_class']
        )
        
        # Top 3
        sorted_probs = sorted(
            result['probabilities'].items(), 
            key=lambda x: -x[1]
        )[:3]
        result['top3'] = [
            (disease_map.get(cls, cls), prob) 
            for cls, prob in sorted_probs
        ]
        
        result['confidence'] = float(result['confidence']) * 100
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict_canvas", methods=["POST"])
def predict_canvas():
    """Dự đoán từ canvas (capture từ camera)"""
    try:
        data = request.json
        if 'image' not in data:
            return jsonify({"error": "Không có dữ liệu ảnh"}), 400
        
        # Decode base64 image
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Dự đoán
        result = model.predict(image)
        result['predicted_class_vi'] = disease_map.get(
            result['predicted_class'], 
            result['predicted_class']
        )
        
        # Top 3
        sorted_probs = sorted(
            result['probabilities'].items(), 
            key=lambda x: -x[1]
        )[:3]
        result['top3'] = [
            (disease_map.get(cls, cls), prob) 
            for cls, prob in sorted_probs
        ]
        
        result['confidence'] = float(result['confidence']) * 100
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    import os
    # Lấy port từ môi trường của server, nếu không có thì dùng 5000
    port = int(os.environ.get("PORT", 5000))
    # host='0.0.0.0' giúp server có thể nhận request từ bên ngoài
    app.run(host='0.0.0.0', port=port, debug=False)
