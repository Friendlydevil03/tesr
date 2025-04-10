import torch
from torchvision.models import detection
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import numpy as np
import time

class VehicleDetector:
    def __init__(self, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Force CPU usage to avoid CUDA issues
        self.device = torch.device('cpu')
        print(f"Using device: {self.device}")
        
        # Cache for previously detected frames to improve performance
        self.detection_cache = {}
        self.cache_max_size = 20
        self.last_inference_time = 0
        self.inference_interval = 0.5  # Minimum time between full inferences (seconds)

        try:
            # Load pre-trained model with proper error handling
            print("Loading ML model...")
            self.model = detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
            
            # Optimize model for inference
            self.model.eval()
            
            # Move model to device
            self.model = self.model.to(self.device)
            
            # Use TorchScript to optimize model if possible
            try:
                self.model = torch.jit.script(self.model)
                print("Model optimized with TorchScript")
            except Exception as e:
                print(f"Could not optimize with TorchScript: {e}")
                
            print("Model loaded successfully")

            # COCO class names (we're interested in vehicles)
            self.classes = [
                'background', 'person', 'bicycle', 'car', 'motorcycle',
                'airplane', 'bus', 'train', 'truck', 'boat'
            ]
            self.vehicle_classes = [2, 3, 5, 6, 7, 8]  # Indices of vehicle classes
        except Exception as e:
            print(f"Error loading ML model: {str(e)}")
            raise

    def detect_vehicles(self, frame):
        """Detect vehicles in a frame with caching and optimization"""
        # Handle invalid input
        if frame is None or frame.size == 0:
            print("Error: Empty frame received")
            return []

        try:
            # Generate a frame hash for cache lookup
            # Simple hash based on downsampled frame to save computation
            small_frame = cv2.resize(frame, (32, 32))
            frame_hash = hash(small_frame.tobytes())
            
            # Check if we have this frame in cache
            if frame_hash in self.detection_cache:
                return self.detection_cache[frame_hash]
                
            # Check if we should do a full inference based on time
            current_time = time.time()
            if current_time - self.last_inference_time < self.inference_interval:
                # Return the most recent detection if available
                if self.detection_cache:
                    return list(self.detection_cache.values())[-1]
            
            # Update last inference time
            self.last_inference_time = current_time

            # Convert frame to tensor efficiently
            # Pre-process the frame for better performance
            img = torch.from_numpy(frame.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
            img = img.to(self.device)

            with torch.no_grad():
                predictions = self.model(img)

            # Extract detections
            boxes = predictions[0]['boxes'].cpu().numpy().astype(int)
            scores = predictions[0]['scores'].cpu().numpy()
            labels = predictions[0]['labels'].cpu().numpy()

            # Filter by confidence and vehicle classes
            vehicle_detections = []
            for box, score, label in zip(boxes, scores, labels):
                if score > self.confidence_threshold and label in self.vehicle_classes:
                    # Return in expected format: (box, score, label)
                    vehicle_detections.append(([box[0], box[1], box[2], box[3]], float(score), int(label)))

            # Store in cache
            self.detection_cache[frame_hash] = vehicle_detections
            
            # Limit cache size
            if len(self.detection_cache) > self.cache_max_size:
                # Remove oldest entry
                oldest_key = list(self.detection_cache.keys())[0]
                del self.detection_cache[oldest_key]

            return vehicle_detections

        except Exception as e:
            print(f"Error in vehicle detection: {str(e)}")
            # Return empty list on error to prevent crashing
            return []
            
    def set_confidence_threshold(self, threshold):
        """Update the confidence threshold"""
        self.confidence_threshold = threshold
        # Clear cache when threshold changes
        self.detection_cache.clear()
