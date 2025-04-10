import cv2
import numpy as np
import threading
import time
import os
import pickle
from datetime import datetime


class ParkingManager:
    """Core parking management functionality separate from UI"""

    DEFAULT_CONFIDENCE = 0.6
    DEFAULT_THRESHOLD = 500
    MIN_CONTOUR_SIZE = 40
    DEFAULT_OFFSET = 10
    DEFAULT_LINE_HEIGHT = 400

    def __init__(self, config_dir="config", log_dir="logs"):
        # Initialize directories
        self.config_dir = config_dir
        self.log_dir = log_dir
        self._ensure_directories_exist()

        # Initialize tracking variables
        self.posList = []
        self.total_spaces = 0
        self.free_spaces = 0
        self.occupied_spaces = 0
        self.vehicle_counter = 0
        self.matches = []

        # Detection parameters
        self.parking_threshold = self.DEFAULT_THRESHOLD
        self.detection_mode = "parking"
        self.line_height = self.DEFAULT_LINE_HEIGHT
        self.min_contour_width = self.MIN_CONTOUR_SIZE
        self.min_contour_height = self.MIN_CONTOUR_SIZE
        self.offset = self.DEFAULT_OFFSET

        # Video/image references
        self.video_reference_map = {}
        self.reference_dimensions = {}
        self.current_reference_image = None
        self.current_video = None

        # ML detection
        self.use_ml_detection = False
        self.ml_detector = None
        self.ml_confidence = self.DEFAULT_CONFIDENCE

        # Thread safety
        self._cleanup_lock = threading.Lock()
        self.data_lock = threading.Lock()
        self.log_data = []

        # For the parking allocation system
        self.parking_visualizer = None
        self.parking_data = {}

    def _ensure_directories_exist(self):
        """Ensure necessary directories exist"""
        for directory in [self.config_dir, self.log_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)

    def load_parking_positions(self, reference_image):
        """Load parking positions from file"""
        try:
            pos_file = os.path.join(self.config_dir, f'CarParkPos_{os.path.splitext(reference_image)[0]}')

            if os.path.exists(pos_file):
                with open(pos_file, 'rb') as f:
                    self.posList = pickle.load(f)
                    self.total_spaces = len(self.posList)
                    self.free_spaces = 0
                    self.occupied_spaces = self.total_spaces
                    return True
            else:
                self.posList = []
                self.total_spaces = 0
                self.free_spaces = 0
                self.occupied_spaces = 0
                return False
        except Exception as e:
            print(f"Error loading parking positions: {str(e)}")
            return False

    def save_parking_positions(self, reference_image):
        """Save parking positions to file"""
        try:
            pos_file = os.path.join(self.config_dir, f'CarParkPos_{os.path.splitext(reference_image)[0]}')
            with open(pos_file, 'wb') as f:
                pickle.dump(self.posList, f)
            return True
        except Exception as e:
            print(f"Error saving parking positions: {str(e)}")
            return False

    def check_parking_space(self, img_pro, img):
        """Process frame to check parking spaces"""
        space_counter = 0
        for i, (x, y, w, h) in enumerate(self.posList):
            # Ensure coordinates are within image bounds
            if (y >= 0 and y + h < img_pro.shape[0] and
                    x >= 0 and x + w < img_pro.shape[1]):

                # Get crop of parking space
                img_crop = img_pro[y:y + h, x:x + w]
                count = cv2.countNonZero(img_crop)

                if count < self.parking_threshold:
                    color = (0, 255, 0)  # Green for free
                    space_counter += 1
                else:
                    color = (0, 0, 255)  # Red for occupied

                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

                # Add count text
                text_scale = 0.6
                text_thickness = 2
                (text_width, text_height), _ = cv2.getTextSize(
                    str(count), cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness
                )
                text_x = x + (w - text_width) // 2
                text_y = y + h - 5
                cv2.putText(img, str(count), (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), text_thickness)

        # Update counters
        self.free_spaces = space_counter
        self.occupied_spaces = self.total_spaces - self.free_spaces

        return img

    # In models/parking_manager.py update_allocation_status method
    def update_allocation_status(self, img_pro, img):
        """Update both the original parking status and the allocation system"""
        # First, process with the existing method
        img = self.check_parking_space(img_pro, img)

        # Then update the allocation system
        space_ids = []
        statuses = []

        # Create parking_data dictionary if it doesn't exist
        if not hasattr(self, 'parking_data'):
            self.parking_data = {}

        for i, (x, y, w, h) in enumerate(self.posList):
            space_id = f"S{i + 1}"
            section = "A" if x < img.shape[1] / 2 else "B"
            section += "1" if y < img.shape[0] / 2 else "2"
            full_space_id = f"{space_id}-{section}"

            # Get status from existing system
            img_crop = img_pro[y:y + h, x:x + w]
            count = cv2.countNonZero(img_crop)
            is_occupied = count >= self.parking_threshold

            space_ids.append(space_id)
            statuses.append(is_occupied)

            # Update the parking_data dictionary used by allocation engine
            if full_space_id not in self.parking_data:
                self.parking_data[full_space_id] = {
                    'position': (x, y, w, h),
                    'occupied': is_occupied,
                    'vehicle_id': None,
                    'last_state_change': datetime.now(),
                    'distance_to_entrance': x + y,  # Simple distance estimation
                    'section': section
                }
            else:
                self.parking_data[full_space_id]['occupied'] = is_occupied

        # Update the allocation system if available
        if self.parking_visualizer:
            self.parking_visualizer.update_parking_status(space_ids, statuses)

        return img


    def get_centroid(self, x, y, w, h):
        """Calculate centroid of a rectangle"""
        return x + w // 2, y + h // 2

    def detect_vehicles(self, frame1, frame2):
        """Process frames to detect and count vehicles"""
        # Get difference between frames
        d = cv2.absdiff(frame1, frame2)
        grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(grey, (5, 5), 0)

        _, th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(th, np.ones((3, 3)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

        closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Draw detection line
        line_y = self.line_height
        if line_y >= frame1.shape[0]:
            line_y = frame1.shape[0] - 50
        cv2.line(frame1, (0, line_y), (frame1.shape[1], line_y), (0, 255, 0), 2)

        # Process contours
        for (i, c) in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(c)
            contour_valid = (w >= self.min_contour_width) and (h >= self.min_contour_height)

            if not contour_valid:
                continue

            cv2.rectangle(frame1, (x - 10, y - 10), (x + w + 10, y + h + 10), (255, 0, 0), 2)

            centroid = self.get_centroid(x, y, w, h)
            self.matches.append(centroid)
            cv2.circle(frame1, centroid, 5, (0, 255, 0), -1)

        # Check for vehicles crossing the line
        new_matches = []
        for (x, y) in self.matches:
            if (line_y - self.offset) < y < (line_y + self.offset):
                self.vehicle_counter += 1
            else:
                new_matches.append((x, y))

        self.matches = new_matches

        # Display count
        cv2.putText(frame1, f"Vehicle Count: {self.vehicle_counter}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)

        return frame1

    def scale_positions(self, orig_width, orig_height, new_width, new_height):
        """Scale parking positions based on current video dimensions"""
        # Calculate scale factors
        width_scale = new_width / orig_width
        height_scale = new_height / orig_height

        # Scale all positions
        scaled_positions = []
        for x, y, w, h in self.posList:
            new_x = int(x * width_scale)
            new_y = int(y * height_scale)
            new_w = int(w * width_scale)
            new_h = int(h * height_scale)
            scaled_positions.append((new_x, new_y, new_w, new_h))

        self.posList = scaled_positions

    def cleanup(self):
        """Clean up resources"""
        with self._cleanup_lock:
            if hasattr(self, 'ml_detector') and self.ml_detector:
                del self.ml_detector

            # Clean up any other resources here
            import gc
            gc.collect()