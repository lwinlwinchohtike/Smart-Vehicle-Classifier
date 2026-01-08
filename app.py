import os
import numpy as np
from PIL import Image
import time
from fpdf import FPDF
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename

# absolute path relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CUSTOM_CNN_MODEL_PATH = os.path.join(BASE_DIR, "trained-models", "cnn_vehicle_classifier_model.keras")
MOBILENET_V2_MODEL_PATH = os.path.join(BASE_DIR, "trained-models", "mobilenetv2_vehicle_classifier_model.keras")
IMAGE_FOLDER = 'static/images'

app = Flask(__name__)
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER

CLASS_NAMES = ['Auto Rickshaw', 'Bike', 'Car', 'Motorcycle', 'Plane', 'Ship', 'Train']
TARGET_SIZE = (224, 224)

MODELS = {}
PREPROCESS_FUNCTIONS = {}

plt.switch_backend('Agg') 

# Define constants for layout 
CHART_Y_OFFSET = 50 
ROW_HEIGHT_CHART = 65  
IMG_WIDTH_HALF = 90    
IMG_HEIGHT_HALF = 50   
IMG_WIDTH_CENTER = 60  
ROW_HEIGHT = 20      
CHART_WIDTH = 100
CHART_HEIGHT = 50
SUMMARY_X_POS = 120

CONFIDENCE_THRESHOLDS = {
    'mobilenet_v2': 0.80, 
    'custom_cnn': 0.75    
}

class PDF(FPDF):
    def header(self):
        # Position the header elements
        self.set_y(10) 
        
        if self.page_no() == 1:
            self.set_font('helvetica', 'B', 15)
            self.cell(80)
            self.cell(30, 10, 'Smart Vehicle Classifier Batch Report', 0, 0, 'C')
        
        self.ln(20)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')

def preprocess_image_to_array(image_path, target_size=TARGET_SIZE):
    try:
        img = keras_image.load_img(image_path, target_size=target_size)
        img_array = keras_image.img_to_array(img)
        return img_array
        
    except Exception as e:
        print(f"Error loading image array: {e}")
        raise

def custom_cnn_preprocess(image_path):
    img_array = preprocess_image_to_array(image_path, TARGET_SIZE)
    scaled_array = img_array.astype('float32') / 255.0
    return scaled_array

def mobilenet_wrapper_preprocess(image_path):
    img_array = preprocess_image_to_array(image_path, TARGET_SIZE)
    normalized_array = mobilenet_v2_preprocess(img_array)
    return normalized_array

# def generate_prediction_chart(results, pdf_path):

#     counts = {}
#     successful_results = [r for r in results if 'error' not in r]

#     for r in successful_results:
#         # Tally the top prediction for the chart
#         if 'top_prediction' in r:
#             counts[r['top_prediction']] = counts.get(r['top_prediction'], 0) + 1

#     if not counts:
#         return None 
        
#     classes = list(counts.keys())
#     values = list(counts.values())
    
#     # Create the chart
#     fig, ax = plt.subplots(figsize=(6, 3)) 
#     ax.bar(classes, values, color='#007bff') 
    
#     # Add labels and title
#     ax.set_title('Batch Prediction Distribution', fontsize=12)
#     ax.set_ylabel('Count', fontsize=10)
#     plt.xticks(rotation=45, ha='right', fontsize=8) 
    
#     # Add count labels on top of bars
#     for i, v in enumerate(values):
#         ax.text(i, v + 0.5, str(v), ha='center', va='bottom', fontsize=8)
    
#     plt.tight_layout()
    
#     # Save the chart to a temporary file
#     chart_dir = os.path.dirname(pdf_path)
#     chart_path = os.path.join(chart_dir, f'batch_chart_temp_{os.getpid()}.png') # Use PID for safety
    
#     plt.savefig(chart_path, dpi=300) 
#     plt.close(fig) 
    
#     return chart_path
    
def get_prediction_status(confidence_ratio, model_id):
    
    threshold = CONFIDENCE_THRESHOLDS.get(model_id, 0.75) 

    if confidence_ratio >= threshold:
        return "CONFIRMED"
    elif confidence_ratio >= 0.50:
        return "AMBIGUOUS"
    else:
        return "UNRELIABLE"
    
def generate_visualizations(detailed_results, pdf_path):
    
    chart_paths = {}
    successful_results = [r for r in detailed_results if 'error' not in r]
    if not successful_results:
        return chart_paths
        
    base_dir = os.path.dirname(pdf_path)
    
    # CHART 1: Prediction Distribution (Bar Chart)
    counts = {}
    for r in successful_results:
        counts[r['top_prediction']] = counts.get(r['top_prediction'], 0) + 1

    classes = list(counts.keys())
    values = list(counts.values())
    
    # Use Matplotlib for plotting
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(classes, values, color='#007bff')
    ax.set_title('Top Prediction Distribution', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=8) 
    plt.tight_layout()
    chart_paths['dist'] = os.path.join(base_dir, 'chart_dist.png')
    plt.savefig(chart_paths['dist'], dpi=200)
    plt.close(fig)

    # CHART 2: Confidence vs. Inference Time (Scatter Plot)
    confidences = [r['top_confidence'] for r in successful_results]
    inference_times = [r['inference_time_ms'] for r in successful_results]

    colors = []
    # Determine model ID for threshold check 
    model_id_for_chart = successful_results[0]['model_used'] 
    
    for r in successful_results:
        confidence_ratio = r['top_confidence'] / 100.0
        status = get_prediction_status(confidence_ratio, model_id_for_chart)
        
        if status == "CONFIRMED":
            colors.append('green')
        elif status == "AMBIGUOUS":
            colors.append('orange')
        else:
            colors.append('red')

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.scatter(inference_times, confidences, c=colors, alpha=0.6, s=20) 

    ax.set_title('Confidence vs. Inference Time', fontsize=12)
    ax.set_xlabel('Inference Time (ms)', fontsize=10)
    ax.set_ylabel('Top Confidence (%)', fontsize=10)
    
    max_threshold = CONFIDENCE_THRESHOLDS.get(model_id_for_chart, 0.75) * 100 
    ax.axhline(y=max_threshold, color='gray', linestyle='--', linewidth=1) # Plot threshold line

    plt.tight_layout()
    chart_paths['time_conf_scatter'] = os.path.join(base_dir, 'chart_time_conf_scatter.png')
    plt.savefig(chart_paths['time_conf_scatter'], dpi=200)
    plt.close(fig)

    # CHART 3: Failure Breakdown (Pie Chart) - 3 categories
    confirmed = sum(1 for r in successful_results if get_prediction_status(r['top_confidence']/100, r['model_used']) == 'CONFIRMED')
    warning = sum(1 for r in successful_results if get_prediction_status(r['top_confidence']/100, r['model_used']) == 'AMBIGUOUS')
    unreliable = sum(1 for r in successful_results if get_prediction_status(r['top_confidence']/100, r['model_used']) == 'UNRELIABLE')
    
    labels = ['Confirmed', 'Ambiguous', 'Unreliable']
    sizes = [confirmed, warning, unreliable]
    colors = ['lightgreen', 'gold', 'lightcoral']
    
    labels_filtered = [l for i, l in enumerate(labels) if sizes[i] > 0]
    sizes_filtered = [s for s in sizes if s > 0]

    fig, ax = plt.subplots(figsize=(3, 3))
    if sizes_filtered:
        ax.pie(sizes_filtered, labels=labels_filtered, autopct='%1.1f%%', startangle=90, colors=colors)
    ax.axis('equal') 
    ax.set_title('Batch Success Breakdown', fontsize=12)
    plt.tight_layout()
    chart_paths['breakdown_pie'] = os.path.join(base_dir, 'chart_breakdown_pie.png')
    plt.savefig(chart_paths['breakdown_pie'], dpi=200)
    plt.close(fig)
    
    return chart_paths

def create_pdf_report(summary, detailed_results, model_name, pdf_path):
    
    try:
        pdf = PDF()
        pdf.alias_nb_pages()
        pdf.add_page()

        validated_results = []
        for row in detailed_results:
            row['model_used'] = model_name
            validated_results.append(row)
            
        if model_name == "mobilenet_v2":
            display_model_name = "MobileNetV2"
        elif model_name == "custom_cnn":
            display_model_name = "Custom CNN"
        else:
            display_model_name = model_name
        
        # Summary Block
        pdf.set_font('helvetica', 'B', 12)
        pdf.cell(0, 7, f'Model Used: {display_model_name}', 0, new_x='LMARGIN', new_y='NEXT', align='L') 
        
        total_adherent = summary.get('total_successful', 0) 
        total_samples = summary.get('total_processed', 0)
        total_non_adherent = summary.get('total_failures', 0)
        
        pdf.set_font('helvetica', '', 9)
        pdf.multi_cell(0, 4, 
            f"Total Processed: {total_samples} \n"
            f"Successful (Meets Threshold): {total_adherent} | Failures (Below Threshold): {total_non_adherent}\n"
            f"Avg Confidence: {summary.get('avg_confidence', 0.0):.2f}% | "
            f"Avg Time: {summary.get('avg_inference_time_ms', 0.0):.2f} ms"
        )
        pdf.ln(5)
        
        # Visualizations
        current_y = pdf.get_y()
        chart_paths = generate_visualizations(validated_results, pdf_path)
        
        if 'dist' in chart_paths:
            pdf.image(chart_paths['dist'], x=10, y=current_y, w=IMG_WIDTH_HALF, h=IMG_HEIGHT_HALF)
        
        if 'time_conf_scatter' in chart_paths: 
            pdf.image(chart_paths['time_conf_scatter'], x=110, y=current_y, w=IMG_WIDTH_HALF, h=IMG_HEIGHT_HALF)
        
        pdf.set_y(current_y + IMG_HEIGHT_HALF + 5) 
        
        if 'breakdown_pie' in chart_paths:
            center_x = (210 - IMG_WIDTH_CENTER) / 2
            pdf.image(chart_paths['breakdown_pie'], x=center_x, y=pdf.get_y(), w=IMG_WIDTH_CENTER, h=IMG_WIDTH_CENTER) 

        pdf.set_y(pdf.get_y() + IMG_WIDTH_CENTER + 10) 
        
        # Detailed Results Table 
        pdf.set_font('helvetica', 'B', 12)
        pdf.cell(0, 10, 'Detailed Results Per Image', 0, 1)
        
        col_widths = [25, 30, 15, 20, 25, 20, 20] 
        headers = ['Image', 'Top Pred.', 'Conf. %', 'Time (ms)', 'Dimensions', 'Size (KB)', 'Status']
        
        # Function to draw table header row
        def draw_header(pdf, col_widths, headers):
            pdf.set_font('helvetica', 'B', 7) 
            for w, header in zip(col_widths, headers):
                pdf.cell(w, 7, header, 1, 0, 'C')
            pdf.ln()
            pdf.set_font('helvetica', '', 7) 

        draw_header(pdf, col_widths, headers) 
        
        # 5. Populate Detailed Results Table
        pdf.set_font('helvetica', '', 7)
        pdf.set_fill_color(255, 255, 255)
        
        for row in validated_results:
            
            # --- Proactive Page Break Check ---
            if pdf.get_y() + ROW_HEIGHT > pdf.page_break_trigger:
                pdf.add_page()
                draw_header(pdf, col_widths, headers) 
                
            confidence_ratio = row.get('top_confidence', 0.0) / 100.0
            status_string = get_prediction_status(confidence_ratio, row['model_used'])
            
            # fill color based on status
            if status_string == "CONFIRMED":
                 pdf.set_fill_color(200, 255, 200)
            elif status_string == "AMBIGUOUS":
                 pdf.set_fill_color(255, 255, 200) 
            else:
                 pdf.set_fill_color(255, 200, 200)
            
            start_x_row = pdf.get_x() 
            start_y_row = pdf.get_y() 

            pdf.cell(col_widths[0], ROW_HEIGHT, '', 1, 0, 'C', fill=False) 
            
            if 'file_path' in row and os.path.exists(row['file_path']):
                image_display_size = ROW_HEIGHT - 2 
                pdf.image(
                    row['file_path'], 
                    x=start_x_row + 1, y=start_y_row + 1, 
                    w=image_display_size, h=image_display_size
                )
            
            # 2. Move cursor to the next starting point for data cells
            pdf.set_xy(start_x_row + col_widths[0], start_y_row) 

            # 3. DATA CELLS (Draw data cells next to the image cell)
            pdf.cell(col_widths[1], ROW_HEIGHT, row.get('top_prediction', 'N/A'), 1, 0, 'C', fill=True)
            pdf.cell(col_widths[2], ROW_HEIGHT, f"{row.get('top_confidence', 0.0):.2f}", 1, 0, 'C', fill=True)
            pdf.cell(col_widths[3], ROW_HEIGHT, f"{row.get('inference_time_ms', 0.0):.2f}", 1, 0, 'C', fill=True)
            pdf.cell(col_widths[4], ROW_HEIGHT, row.get('img_dimensions', 'N/A'), 1, 0, 'C', fill=True)
            pdf.cell(col_widths[5], ROW_HEIGHT, f"{row.get('img_size_kb', 0.0):.2f}", 1, 0, 'C', fill=True)
            
            # status 
            pdf.cell(col_widths[6], ROW_HEIGHT, status_string, 1, 0, 'C', fill=True) 
            
            pdf.ln() 
            
            pdf.set_fill_color(255, 255, 255)

        pdf.cell(sum(col_widths), 0, '', 'T', 1, 'L')
            
        pdf.output(pdf_path)
        
        # Clean up temporary chart files
        for path in chart_paths.values():
            if os.path.exists(path):
                os.remove(path)
            
        return True

    except Exception as e:
        print(f"FATAL ERROR during PDF generation: {e}")
        return False

def generate_summary_report(results, model_name):
    successful_predictions = [r for r in results if 'error' not in r]
    total_processed_attempts = len(results)
    total_successful_runs = len(successful_predictions)
    
    avg_conf = 0.0
    avg_time = 0.0
    avg_gap = 0.0
    
    if total_successful_runs > 0:
        # Averages 
        avg_conf = np.mean([r['top_confidence'] for r in successful_predictions]) 
        avg_time = np.mean([r['inference_time_ms'] for r in successful_predictions])
        avg_gap = np.mean([r['conf_gap'] for r in successful_predictions])
        
        # Calculate Threshold
        total_adherent_count = 0
        
        for r in successful_predictions:
            confidence_ratio = r['top_confidence'] / 100.0
            
            status = get_prediction_status(confidence_ratio, model_name) 
            
            if status == 'CONFIRMED':
                total_adherent_count += 1
                
        total_non_adherent = total_successful_runs - total_adherent_count
        
    else:
        total_adherent_count = 0
        total_non_adherent = 0
        
        if total_processed_attempts == 0:
            return {"total_processed": 0, "message": "No input data provided."}

    return {
        "total_processed": total_successful_runs,
        "model_used": model_name,
        "avg_confidence": round(avg_conf, 2),
        "avg_inference_time_ms": round(avg_time, 2),
        "avg_conf_gap": round(avg_gap, 2),
        
        "total_successful": total_adherent_count, 
        "total_failures": total_non_adherent,   
    }

# Load Custom CNN
if os.path.exists(CUSTOM_CNN_MODEL_PATH):
    MODELS['custom_cnn'] = load_model(CUSTOM_CNN_MODEL_PATH)
    PREPROCESS_FUNCTIONS['custom_cnn'] = custom_cnn_preprocess
    print("Custom CNN loaded.")
else:
    print(f"WARNING: Custom CNN model not found at {CUSTOM_CNN_MODEL_PATH}")

# Load MobileNetV2
if os.path.exists(MOBILENET_V2_MODEL_PATH):
    def mobilenet_wrapper_preprocess(image_path):
        img_array = preprocess_image_to_array(image_path, TARGET_SIZE) # Get raw numpy array
        return mobilenet_v2_preprocess(img_array)

    MODELS['mobilenet_v2'] = load_model(MOBILENET_V2_MODEL_PATH)
    PREPROCESS_FUNCTIONS['mobilenet_v2'] = mobilenet_wrapper_preprocess
    print("MobileNetV2 loaded.")
else:
    print(f"WARNING: MobileNetV2 model not found at {MOBILENET_V2_MODEL_PATH}")

if not MODELS:
    print("ERROR: No models loaded. Exiting.")
    exit(1)

def get_all_images():
    images = os.listdir(app.config['IMAGE_FOLDER'])
    images.sort()
    return images

@app.route('/')
def index():
    images = get_all_images()
    return render_template('index.html', images=images)

@app.route('/images')
def all_images():
    images = get_all_images()
    return jsonify(images)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'status': 'error', 'error': 'No selected file'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        
        file_path = os.path.join(app.config['IMAGE_FOLDER'], filename)
        
        os.makedirs(app.config['IMAGE_FOLDER'], exist_ok=True)
        
        try:
            file.save(file_path)
            
            return jsonify({'status': 'success', 'filename': filename}), 200
        
        except Exception as e:
            return jsonify({'status': 'error', 'error': f'Failed to save file to disk: {e}'}), 500
            
    return jsonify({'status': 'error', 'error': 'Unknown upload failure'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try: 
        image_name = request.json['image']
        model_id = request.json.get('model_id', 'mobilenet_v2') 

        if model_id not in MODELS:
            return jsonify({'status': 'error', 'error': f'Invalid model ID: {model_id}'}), 400
        
        # Select the model and preprocessor based on the ID
        model = MODELS[model_id]
        preprocessor = PREPROCESS_FUNCTIONS[model_id]

        current_threshold = CONFIDENCE_THRESHOLDS[model_id]

        image_path = os.path.join(app.config['IMAGE_FOLDER'], image_name)

        if not os.path.exists(image_path):
            return jsonify({'status': 'error', 'error': f'Image file not found at: {image_path}'}), 404
        
        processed_img = preprocessor(image_path)

        if processed_img.ndim == 3:
            processed_img = np.expand_dims(processed_img, axis=0)
        
        # run prediction: returns prob_class1, ..., prob_class7
        predictions = model.predict(processed_img)[0] 
        

        # structure all 7 classes
        results = []
        for i, class_name in enumerate(CLASS_NAMES):
            results.append({
                'class': class_name,
                'confidence': round(float(predictions[i]) * 100, 2) 
            })

        results.sort(key=lambda x: x['confidence'], reverse=True) # Sort Results by confidence in descending order.
        
        top_3_predictions = results[:3] # Top 3 Predictions

        # THRESHOLD CHECK
        raw_confidence_top = results[0]['confidence'] / 100.0 # Convert 0-100% back to 0.0-1.0
        
        if raw_confidence_top >= current_threshold:
            status_message = "CONFIRMED"
        elif raw_confidence_top >= 0.50: 
            status_message = "WARNING - AMBIGUOUS"
        else:
            status_message = "LOW CONFIDENCE - UNRELIABLE"
        
        # Return JSON Response
        return jsonify({
            'status': 'success',
            'model_used': model_id,
            'top_prediction': results[0]['class'],
            'top_confidence': results[0]['confidence'],
            'prediction_status': status_message,
            'top_3_predictions': top_3_predictions,
            'all_classes': results # return all 7 classes
        })

    except Exception as e:
        return jsonify({'status': 'error', 'error': f'Prediction failed: {e}'}), 500

@app.route('/batch_predict_and_report', methods=['GET'])
def batch_predict_and_report():
    try:
        selected_model_name = request.args.get('model', 'mobilenet_v2') 
        model = MODELS.get(selected_model_name)
        preprocessor = PREPROCESS_FUNCTIONS.get(selected_model_name)
        
        if not model or not preprocessor:
            return jsonify({"error": f"Invalid or unavailable model or preprocessor: {selected_model_name}"}), 400
            
        image_filenames = [f for f in os.listdir(app.config['IMAGE_FOLDER']) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_filenames:
            return jsonify({"message": f"No images found in {app.config['IMAGE_FOLDER']}"}), 200

        results_data = []
        
        for filename in image_filenames:
            file_path = os.path.join(app.config['IMAGE_FOLDER'], filename)
            
            try:
                if not os.path.exists(file_path):
                    results_data.append({'filename': filename, 'error': f'Image file not found on server.'})
                    continue

                # image quality
                img_size_bytes = os.path.getsize(file_path)
                img_size_kb = round(img_size_bytes / 1024, 2)
                
                img_pil = Image.open(file_path)
                width, height = img_pil.size
                img_dimensions = f"{width}x{height}"
                
                # predicting and processing
                processed_img = preprocessor(file_path)
                if processed_img.ndim == 3:
                    processed_img = np.expand_dims(processed_img, axis=0)

                start_time = time.time()
                predictions = model.predict(processed_img, verbose=0)[0] 
                end_time = time.time()
                inference_time = (end_time - start_time) * 1000 # in ms
                
                image_results = []
                for i, class_name in enumerate(CLASS_NAMES):
                    image_results.append({'class': class_name, 'confidence': round(float(predictions[i]) * 100, 2)})

                image_results.sort(key=lambda x: x['confidence'], reverse=True) 
                
                top_confidence_value = image_results[0]['confidence']
                # is_successful = top_confidence_value >= CONFIDENCE_THRESHOLDS
                
                # CONFIDENCE GAP CALCULATION
                conf_gap = round(top_confidence_value - image_results[1]['confidence'], 2) if len(image_results) > 1 else 100.0
                
                top_3_predictions = image_results[:3]

                results_data.append({
                    'filename': filename,
                    'file_path': file_path,
                    'top_prediction': image_results[0]['class'],
                    'top_confidence': top_confidence_value,
                    'conf_gap': conf_gap,
                    'inference_time_ms': round(inference_time, 2),
                    # 'is_successful': 'YES' if is_successful else 'NO',
                    'img_dimensions': img_dimensions,
                    'img_size_kb': img_size_kb,
                    'top_3_predictions': top_3_predictions, 
                    'all_classes': image_results
                })
                
            except Exception as e:
                results_data.append({'filename': filename, 'error': f"Processing failed: {str(e)}"})

        # 4. Generate Summary and PDF
        report_summary = generate_summary_report(results_data, selected_model_name)
        pdf_filename = f"batch_report_{selected_model_name}_{int(time.time())}.pdf"
        pdf_path = os.path.join(app.root_path, pdf_filename)
        
        if create_pdf_report(report_summary, results_data, selected_model_name, pdf_path):
            response = send_file(pdf_path, as_attachment=True, download_name=pdf_filename)
    
            @response.call_on_close
            def cleanup():
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)
                    
            return response
        else:
            return jsonify({
                'error': 'Failed to generate PDF. Check server logs.',
                'model_used': selected_model_name,
                'summary': report_summary,
                'details': results_data
            }), 500

    except Exception as e:
        return jsonify({'status': 'error', 'error': f'Batch prediction failed: {e}'}), 500

if __name__ == '__main__':
    if not os.path.exists(app.config['IMAGE_FOLDER']):
        os.makedirs(app.config['IMAGE_FOLDER'])

    app.run(debug=True)