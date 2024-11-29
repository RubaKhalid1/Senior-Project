from flask import Flask, render_template, jsonify, send_from_directory, request
import os
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from PIL.ExifTags import TAGS, GPSTAGS
import torch
import sys
from werkzeug.utils import secure_filename
from datetime import datetime
import csv
import cv2

app = Flask(__name__)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best_weights_lr0.001_batch16_epochs100_patience10.pt')
IMAGE_DIR = os.path.join(BASE_DIR, 'images')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
CSV_PATH = os.path.join(RESULTS_DIR, 'detection_history.csv')

# Create necessary directories
for directory in [IMAGE_DIR, RESULTS_DIR, UPLOAD_DIR]:
    os.makedirs(directory, exist_ok=True)

# Upload configuration
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def initialize_model():
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")

        # Load model with error handling
        try:
            model = YOLO(MODEL_PATH)
            print("Model loaded successfully")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
    except Exception as e:
        print(f"Error initializing model: {e}")
        sys.exit(1)

# Initialize model
yolo_model = initialize_model()
class_names_list = ["garbage", "graffiti", "sand on road"]

def format_date(date_str):
    """Format date string into a consistent format"""
    try:
        if isinstance(date_str, str):
            # Handle EXIF date format (YYYY:MM:DD HH:MM:SS)
            if ':' in date_str:
                date_str = date_str.replace(':', '-', 2)
                date_obj = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
            else:
                # Try parsing the date string
                try:
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    date_obj = datetime.now()
        else:
            date_obj = datetime.now()

        return date_obj.strftime('%Y-%m-%d %H:%M:%S')
    except Exception as e:
        print(f"Error formatting date: {e}")
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
def get_exif_data(image_path):
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        if exif_data is None:
            return None, None

        gps_info = None
        date_info = None

        for tag, value in exif_data.items():
            decoded_tag = TAGS.get(tag, tag)
            if decoded_tag == 'GPSInfo':
                gps_info = {GPSTAGS.get(t, t): value[t] for t in value}
            if decoded_tag == 'DateTimeOriginal':
                date_info = format_date(value)

        return gps_info, date_info
    except Exception as e:
        print(f"Error reading EXIF data: {e}")
        return None, None

def convert_to_degrees(value):
    try:
        d = float(value[0])
        m = float(value[1])
        s = float(value[2])
        return d + (m / 60.0) + (s / 3600.0)
    except (TypeError, ValueError , IndexError) as e:
        print(f"Error converting coordinates: {e}")
        return None

def get_coordinates(gps_info):
    if gps_info is None:
        return None, None

    try:
        lat = convert_to_degrees(gps_info.get('GPSLatitude'))
        lon = convert_to_degrees(gps_info.get('GPSLongitude'))

        if lat is None or lon is None:
            return None, None

        if gps_info.get('GPSLatitudeRef', 'N') == 'S':
            lat = -lat
        if gps_info.get('GPSLongitudeRef', 'E') == 'W':
            lon = -lon

        return lat, lon
    except Exception as e:
        print(f"Error processing coordinates: {e}")
        return None, None

def save_detection_result(image_path, results, image_name):
    """Process detection and save results"""
    try:
        # Get EXIF data
        gps_info, date_info = get_exif_data(image_path)
        lat, lon = get_coordinates(gps_info)
        
        # Use default coordinates if none found
        if lat is None:
            lat, lon = 21.492500, 39.177570
        
        # Use current time if no date found
        if not date_info:
            date_info = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Get detections
        detections = [class_names_list[int(box.cls)] for box in results[0].boxes]
        detection_classes = ', '.join(detections) if detections else 'No detections'
        
        # Save processed image
        output_name = f'processed_{image_name}'
        output_path = os.path.join(RESULTS_DIR, output_name)
        results[0].save(output_path)
        
        # Prepare data for CSV
        data = {
            'Image': image_name,
            'Processed_Image': output_name,
            'Latitude': lat,
            'Longitude': lon,
            'Date_Taken': date_info,
            'Classes': detection_classes,
            'Processing_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save to CSV
        csv_exists = os.path.exists(CSV_PATH)
        with open(CSV_PATH, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            if not csv_exists:
                writer.writeheader()
            writer.writerow(data)
        
        return data
        
    except Exception as e:
        print(f"Error saving detection result: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file part'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file'})
        
        if file and allowed_file(file.filename):
            try:
                # Save the uploaded file
                filename = secure_filename(file.filename)
                filepath = os.path.join(IMAGE_DIR, filename)  # Save directly to images directory
                file.save(filepath)
                
                print(f"Processing image: {filepath}")  # Debug print
                
                # Run YOLO detection
                try:
                    results = yolo_model(filepath)
                    # Get class names from results
                    detections = []
                    for r in results:
                        for box in r.boxes:
                            class_idx = int(box.cls)
                            if class_idx < len(class_names_list):
                                detections.append(class_names_list[class_idx])
                    
                    # Get image metadata
                    gps_info, date_info = get_exif_data(filepath)
                    lat, lon = get_coordinates(gps_info)
                    
                    # Use default values if metadata is missing
                    if lat is None:
                        lat, lon = 24.7136, 46.6753
                    if not date_info:
                        date_info = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Save the processed image
                    results_image_path = os.path.join(RESULTS_DIR, f'processed_{filename}')
                    results[0].save(results_image_path)
                    
                    # Prepare data for CSV
                    detection_data = {
                        'Image': filename,
                        'Processed_Image': f'processed_{filename}',
                        'Latitude': lat,
                        'Longitude': lon,
                        'Date_Taken': date_info,
                        'Classes': ', '.join(detections) if detections else 'No detections',
                        'Processing_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    # Append to CSV
                    csv_exists = os.path.exists(CSV_PATH)
                    with open(CSV_PATH, 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=detection_data.keys())
                        if not csv_exists:
                            writer.writeheader()
                        writer.writerow(detection_data)
                    
                    return jsonify({
                        'success': True,
                        'message': 'File uploaded and processed successfully',
                        'data': detection_data
                    })
                    
                except Exception as e:
                    print(f"Error in YOLO detection: {str(e)}")  # Debug print
                    return jsonify({
                        'success': False,
                        'error': f'Error processing image: {str(e)}'
                    })
                    
            except Exception as e:
                print(f"Error saving file: {str(e)}")  # Debug print
                return jsonify({
                    'success': False,
                    'error': f'Error saving file: {str(e)}'
                })
                
    except Exception as e:
        print(f"Upload error: {str(e)}")  # Debug print
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/data')
def get_data():
    try:
        # Read from CSV if it exists
        if os.path.exists(CSV_PATH):
            df = pd.read_csv(CSV_PATH)
        else:
            return jsonify({
                'success': False,
                'error': 'No data available'
            })

        data = df.to_dict('records')
        
        # Calculate statistics
        total_images = len(data)
        detection_counts = {}
        for record in data:
            classes = str(record['Classes']).split(', ')
            for cls in classes:
                if cls != 'No detections':
                    detection_counts[cls] = detection_counts.get(cls, 0) + 1

        return jsonify({
            'success': True,
            'data': data,
            'stats': {
                'total_images': total_images,
                'detection_counts': detection_counts
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print(f"Model path: {MODEL_PATH}")
    print(f"Image directory: {IMAGE_DIR}")
    print("Available classes:", class_names_list)
    
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        sys.exit(1)
    
    app.run(debug=True)