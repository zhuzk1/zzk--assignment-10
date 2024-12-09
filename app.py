from flask import Flask, render_template, request, jsonify, send_from_directory
from image_search import image_to_image_query, text_to_image_query, hybrid_query
import os

app = Flask(__name__)


PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Move up one directory
COCO_IMAGES_DIR = os.path.join(PARENT_DIR, "coco_images_resized")  # Path to coco_images_resized


@app.route('/')
def index():
    """Serve the main HTML page."""
    return render_template('index.html')

@app.route('/search_image', methods=['POST'])
def search_image():
    image_file = request.files.get('image')
    if not image_file:
        return jsonify({"error": "No image file provided"}), 400

    # Save query image
    query_image_path = os.path.join(COCO_IMAGES_DIR, "query_image.jpg")
    image_file.save(query_image_path)

    # Get top 5 results
    results = image_to_image_query(query_image_path)

    return jsonify(results)


@app.route('/search_text', methods=['POST'])
def search_text():
    query_text = request.json.get('query')
    if not query_text:
        return jsonify({"error": "No text query provided"}), 400

    # Get top 5 results
    results = text_to_image_query(query_text)
    print(results)

    return jsonify(results)  # 返回 JSON 格式的多结果



@app.route('/search_hybrid', methods=['POST'])
def search_hybrid():
    image_file = request.files.get('image')
    query_text = request.form.get('query')
    lambda_weight = float(request.form.get('weight', 0.5))

    if not image_file or not query_text:
        return jsonify({"error": "Both image and text queries are required"}), 400

    # Save query image
    query_image_path = os.path.join(COCO_IMAGES_DIR, "query_image.jpg")
    image_file.save(query_image_path)

    # Get top 5 results
    results = hybrid_query(query_image_path, query_text, lambda_weight)

    return jsonify(results)


@app.route('/results/<filename>')
def serve_result_image(filename):
    """Serve result images."""
    try:
        return send_from_directory(COCO_IMAGES_DIR, filename)
    except FileNotFoundError:
        return jsonify({"error": f"File '{filename}' not found in {COCO_IMAGES_DIR}"}), 404


if __name__ == '__main__':
    app.run(debug=True)
