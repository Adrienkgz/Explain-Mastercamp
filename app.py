from flask import Flask, render_template, request, jsonify
from data_processing import get_text_from_html_doc
from new_predict import PatentPredicterAI
from listes_labels import list_label_level_0
import logging

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('aPropos.html')

@app.route('/faq')
def faq():
    return render_template('faq.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

try:
    predictor = PatentPredicterAI(level=0)
    logging.debug("Predictor initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize the predictor: {str(e)}")

@app.route('/classify', methods=['POST'])
def classify():
    try:
        data = request.get_json()
        raw_text = data['input_data']
        clean_text = get_text_from_html_doc(raw_text)

        prediction_indices = predictor.predict(input=clean_text, method='0.8*max')
        predictions_labels = [list_label_level_0[idx] for idx in prediction_indices] 

        return jsonify({'prediction': predictions_labels}), 200
    except Exception as e:
        logging.error(f"Error during classification: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)