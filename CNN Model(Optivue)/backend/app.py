from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
import numpy as np
from PIL import Image
import os
import json
import torch
import cv2
from typing import Dict, List, Any, Tuple, Optional
import logging
from pathlib import Path
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

class ModelManager:
    """Advanced YOLO model manager for multi-model detection"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.model_cache = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_configs = self._load_model_configs()
        
        # Ensure models directory exists
        self.models_dir.mkdir(exist_ok=True)
        
        logger.info(f"Model Manager initialized - Device: {self.device}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    def _load_model_configs(self) -> Dict[str, Any]:
        """Load model configurations from JSON file"""
        config_path = self.models_dir / 'model_config.json'
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    logger.info(f"Loaded model configs for: {list(config.keys())}")
                    return config
            except Exception as e:
                logger.warning(f"Failed to load model config: {e}")
        
        # Default configuration for all models
        return {
            "model1": {
                "name": "Human Face Detection",
                "confidence_threshold": 0.5,
                "iou_threshold": 0.45,
                "input_size": 640,
                "class_filter": ["person", "face", "human"],
                "detection_type": "face"
            },
            "model2": {
                "name": "Fire Detection", 
                "confidence_threshold": 0.4,
                "iou_threshold": 0.45,
                "input_size": 640,
                "class_filter": [],  # Show all detections
                "detection_type": "fire"
            },
            "model3": {
                "name": "Fall Detection",
                "confidence_threshold": 0.3,
                "iou_threshold": 0.45, 
                "input_size": 640,
                "class_filter": [],  # Show all detections
                "detection_type": "Predictions"
            }
        }
    
    def _get_model_path(self, model_name: str) -> Optional[Path]:
        """Get the full path for a model file"""
        model_path = self.models_dir / f"{model_name}.pt"
        if model_path.exists():
            return model_path
        return None
    
    def _extract_class_names(self, model) -> List[str]:
        """Extract class names from YOLO model"""
        try:
            if hasattr(model, 'names'):
                names = model.names
                if isinstance(names, dict):
                    return list(names.values())
                elif isinstance(names, list):
                    return names
            elif hasattr(model, 'module') and hasattr(model.module, 'names'):
                names = model.module.names
                if isinstance(names, dict):
                    return list(names.values())
                elif isinstance(names, list):
                    return names
            
            # Fallback to COCO classes
            return self._get_default_class_names()
            
        except Exception as e:
            logger.warning(f"Could not extract class names: {e}")
            return self._get_default_class_names()
    
    def _get_default_class_names(self) -> List[str]:
        """Get default class names based on common detection scenarios"""
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush', 'fire', 'smoke', 'flame', 'fall', 'accident', 'emergency'
        ]
    
    def load_model(self, model_name: str) -> Dict[str, Any]:
        """Load and cache a YOLO model"""
        if model_name in self.model_cache:
            logger.info(f"Using cached model: {model_name}")
            return self.model_cache[model_name]
        
        model_path = self._get_model_path(model_name)
        if not model_path:
            raise FileNotFoundError(f"Model '{model_name}.pt' not found in {self.models_dir}")
        
        try:
            logger.info(f"Loading YOLOv5 model: {model_name} from {model_path}")
            
            # Load model using torch.hub
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(model_path), device=self.device, force_reload=False)
            
            # Extract class names
            class_names = self._extract_class_names(model)
            
            # Get model configuration
            config = self.model_configs.get(model_name, {
                "name": f"Model {model_name}",
                "confidence_threshold": 0.4,
                "iou_threshold": 0.45,
                "input_size": 640,
                "class_filter": [],
                "detection_type": "object"
            })
            
            model_info = {
                'model': model,
                'class_names': class_names,
                'config': config,
                'name': config.get('name', model_name),
                'confidence_threshold': config.get('confidence_threshold', 0.4),
                'iou_threshold': config.get('iou_threshold', 0.45),
                'input_size': config.get('input_size', 640),
                'class_filter': config.get('class_filter', []),
                'detection_type': config.get('detection_type', 'object')
            }
            
            self.model_cache[model_name] = model_info
            logger.info(f"Successfully loaded {config.get('name', model_name)} with {len(class_names)} classes")
            return model_info
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to load model {model_name}: {str(e)}")
    
    def get_available_models(self) -> List[str]:
        """Get list of available YOLO models"""
        models = []
        if self.models_dir.exists():
            for file_path in self.models_dir.glob("*.pt"):
                models.append(file_path.stem)
        return sorted(models)
    
    def predict(self, model_info: Dict[str, Any], image_array: np.ndarray) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """Run YOLO inference and return detections with annotated image"""
        model = model_info['model']
        class_names = model_info['class_names']
        confidence_threshold = model_info['confidence_threshold']
        iou_threshold = model_info['iou_threshold']
        class_filter = model_info['class_filter']
        
        try:
            # Configure model thresholds
            model.conf = confidence_threshold
            model.iou = iou_threshold
            
            logger.debug(f"Running inference with conf={confidence_threshold}, iou={iou_threshold}")
            
            # Run inference
            results = model(image_array)
            
            # Parse results
            detections = []
            pred_df = results.pandas().xyxy[0]
            
            logger.debug(f"Raw detections found: {len(pred_df)}")
            
            for _, detection in pred_df.iterrows():
                class_id = int(detection['class'])
                confidence = float(detection['confidence'])
                
                # Get class name
                class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                
                # Apply class filtering for Model 1 (face detection)
                if class_filter:
                    class_name_lower = class_name.lower()
                    if not any(filter_class.lower() in class_name_lower for filter_class in class_filter):
                        continue
                
                detection_dict = {
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': {
                        'x1': float(detection['xmin']),
                        'y1': float(detection['ymin']),
                        'x2': float(detection['xmax']),
                        'y2': float(detection['ymax']),
                        'width': float(detection['xmax'] - detection['xmin']),
                        'height': float(detection['ymax'] - detection['ymin'])
                    }
                }
                
                detections.append(detection_dict)
            
            # Sort by confidence (highest first)
            detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            logger.info(f"Filtered detections: {len(detections)}")
            
            # Create annotated image
            annotated_image = self._draw_detections(image_array.copy(), detections, model_info['detection_type'])
            
            return detections, annotated_image
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def _draw_detections(self, image: np.ndarray, detections: List[Dict[str, Any]], detection_type: str) -> np.ndarray:
        """Draw red bounding boxes on image for all detections"""
        annotated_image = image.copy()
        
        # Color scheme - red for all detection types
        box_color = (0, 0, 255)  # BGR format for OpenCV
        text_bg_color = (0, 0, 255)
        text_color = (255, 255, 255)
        
        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Convert coordinates to integers
            x1, y1 = int(bbox['x1']), int(bbox['y1'])
            x2, y2 = int(bbox['x2']), int(bbox['y2'])
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), box_color, 2)
            
            # Create label based on detection type
            if detection_type == 'face':
                label = f"Human Face {confidence:.0%}"
            elif detection_type == 'fire':
                label = f"Fire {confidence:.0%}"
            elif detection_type == 'Predictions':
                label = f"Predictions {confidence:.0%}"
            else:
                label = f"{class_name} {confidence:.0%}"
            
            # Get text size for background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Draw background for text
            cv2.rectangle(annotated_image, 
                         (x1, y1 - text_height - 10), 
                         (x1 + text_width + 10, y1), 
                         text_bg_color, -1)
            
            # Draw text
            cv2.putText(annotated_image, label, 
                       (x1 + 5, y1 - 5), 
                       font, font_scale, text_color, thickness)
        
        return annotated_image

def decode_base64_image(image_data: str) -> np.ndarray:
    """Decode base64 image to numpy array"""
    try:
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(image)
        
        return image_array
        
    except Exception as e:
        logger.error(f"Image decoding failed: {e}")
        raise ValueError(f"Invalid image data: {str(e)}")

def encode_image_to_base64(image_array: np.ndarray) -> str:
    """Encode numpy array image to base64 string"""
    try:
        # Convert BGR to RGB if needed
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image_array
        
        # Convert to PIL Image
        image = Image.fromarray(image_rgb.astype('uint8'))
        
        # Save to bytes buffer
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=95)
        
        # Encode to base64
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return f"data:image/jpeg;base64,{image_base64}"
        
    except Exception as e:
        logger.error(f"Image encoding failed: {e}")
        raise ValueError(f"Failed to encode image: {str(e)}")

# Initialize model manager
model_manager = ModelManager()

@app.route('/predict', methods=['POST'])
def predict():
    """Main object detection endpoint"""
    try:
        logger.info("=== PREDICTION REQUEST RECEIVED ===")
        
        # Validate request
        if not request.json:
            logger.error("No JSON data provided")
            return jsonify({'error': 'No JSON data provided'}), 400
        
        image_data = request.json.get('image')
        model_name = request.json.get('model')
        
        logger.info(f"Model requested: {model_name}")
        logger.info(f"Image data length: {len(image_data) if image_data else 0}")
        
        if not image_data or not model_name:
            return jsonify({'error': 'Both "image" and "model" fields are required'}), 400
        
        # Step 1: Decode image
        try:
            image_array = decode_base64_image(image_data)
            logger.info(f"✓ Image decoded successfully: {image_array.shape}")
        except Exception as e:
            logger.error(f"✗ Image decoding failed: {e}")
            return jsonify({'error': f'Image decoding failed: {str(e)}'}), 400
        
        # Step 2: Load model
        try:
            model_info = model_manager.load_model(model_name)
            logger.info(f"✓ Model loaded: {model_info['name']}")
        except Exception as e:
            logger.error(f"✗ Model loading failed: {e}")
            return jsonify({'error': f'Model loading failed: {str(e)}'}), 404
        
        # Step 3: Run inference
        try:
            detections, annotated_image = model_manager.predict(model_info, image_array)
            logger.info(f"✓ Inference completed: {len(detections)} detections found")
        except Exception as e:
            logger.error(f"✗ Inference failed: {e}")
            return jsonify({'error': f'Inference failed: {str(e)}'}), 500
        
        # Step 4: Encode result image
        try:
            annotated_image_b64 = encode_image_to_base64(annotated_image)
            logger.info("✓ Image encoding completed")
        except Exception as e:
            logger.error(f"✗ Image encoding failed: {e}")
            return jsonify({'error': f'Image encoding failed: {str(e)}'}), 500
        
        # Prepare response
        response = {
            'success': True,
            'model_used': model_name,
            'model_name': model_info['name'],
            'detection_type': model_info['detection_type'],
            'image_shape': {
                'width': int(image_array.shape[1]),
                'height': int(image_array.shape[0])
            },
            'detections_count': len(detections),
            'detections': detections,
            'annotated_image': annotated_image_b64,
            'model_info': {
                'confidence_threshold': model_info['confidence_threshold'],
                'iou_threshold': model_info['iou_threshold'],
                'input_size': model_info['input_size'],
                'total_classes': len(model_info['class_names']),
                'detection_type': model_info['detection_type']
            },
            'timestamp': str(pd.Timestamp.now()) if 'pd' in globals() else 'N/A'
        }
        
        logger.info(f"=== PREDICTION COMPLETED SUCCESSFULLY ===")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"=== PREDICTION FAILED ===")
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Unexpected server error: {str(e)}'}), 500

@app.route('/models', methods=['GET'])
def list_models():
    """List available models with detailed information"""
    try:
        available_models = model_manager.get_available_models()
        model_details = []
        
        for model_name in available_models:
            try:
                # Get config without loading the actual model
                config = model_manager.model_configs.get(model_name, {})
                details = {
                    'name': model_name,
                    'display_name': config.get('name', f'Model {model_name}'),
                    'detection_type': config.get('detection_type', 'object'),
                    'confidence_threshold': config.get('confidence_threshold', 0.4),
                    'iou_threshold': config.get('iou_threshold', 0.45),
                    'input_size': config.get('input_size', 640),
                    'status': 'available'
                }
                model_details.append(details)
            except Exception as e:
                logger.warning(f"Could not get details for model {model_name}: {e}")
                model_details.append({
                    'name': model_name,
                    'status': 'error',
                    'error': str(e)
                })
        
        return jsonify({
            'available_models': available_models,
            'model_details': model_details,
            'total_models': len(available_models)
        })
        
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        return jsonify({'error': 'Failed to retrieve model list'}), 500

@app.route('/model/<model_name>/info', methods=['GET'])
def get_model_info(model_name: str):
    """Get detailed information about a specific model"""
    try:
        model_info = model_manager.load_model(model_name)
        return jsonify({
            'model': model_name,
            'name': model_info['name'],
            'detection_type': model_info['detection_type'],
            'classes': model_info['class_names'],
            'total_classes': len(model_info['class_names']),
            'configuration': {
                'confidence_threshold': model_info['confidence_threshold'],
                'iou_threshold': model_info['iou_threshold'],
                'input_size': model_info['input_size']
            }
        })
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        logger.error(f"Failed to get info for model {model_name}: {e}")
        return jsonify({'error': 'Failed to retrieve model information'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Comprehensive health check endpoint"""
    try:
        available_models = model_manager.get_available_models()
        return jsonify({
            'status': 'healthy',
            'service': 'BURNPEPPER AI - Multi-Model Detection API',
            'version': '1.0.0',
            'device': str(model_manager.device),
            'cuda_available': torch.cuda.is_available(),
            'models_available': len(available_models),
            'models': available_models,
            'endpoints': ['/predict', '/models', '/model/<name>/info', '/health', '/test']
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/test', methods=['GET'])
def test_endpoint():
    """Simple test endpoint"""
    return jsonify({
        'status': 'API is working perfectly',
        'message': 'BURNPEPPER AI Backend is responding',
        'port': 5050
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logger.info("="*60)
    logger.info("🔥 BURNPEPPER AI - Multi-Model Detection API 🔥")
    logger.info("="*60)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Log startup information
    available_models = model_manager.get_available_models()
    logger.info(f"Device: {model_manager.device}")
    logger.info(f"Available models: {available_models}")
    
    if not available_models:
        logger.warning("⚠️  No .pt models found in models/ directory!")
        logger.info("📝 Please add your trained models:")
        logger.info("   - model1.pt (Human Face Detection)")
        logger.info("   - model2.pt (Fire Detection)")  
        logger.info("   - model3.pt (Fall Detection)")
    else:
        logger.info("✅ Models ready for detection!")
    
    logger.info("="*60)
    logger.info("🚀 Starting server on http://localhost:5050")
    logger.info("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5050)