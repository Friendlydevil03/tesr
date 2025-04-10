import cv2
from PIL import Image, ImageTk
import threading
from tkinter import Frame, Label, Button, StringVar, OptionMenu, BOTH, X, Y, LEFT, RIGHT, Canvas
from tkinter import ttk, messagebox, NSEW, W, E, N, S
from datetime import datetime
from models.vehicle_detector import VehicleDetector
from utils.media_paths import get_video_path
from utils.image_processor import (
    preprocess_frame_for_parking_detection,
    process_parking_spaces,
    detect_vehicles_traditional,
    process_ml_detections
)


class DetectionTab:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app

        # Initialize variables specific to this tab
        self.prev_frame = None
        self.frame_count = 0
        self.frame_skip = 3  # Process every 3rd frame
        self.last_detections = []

        # Set default values for vehicle detection matching your code
        self.app.min_contour_width = 40
        self.app.min_contour_height = 40
        self.app.offset = 10
        self.app.line_height = 550
        self.app.matches = []
        self.app.vehicle_counter = 0

        # Setup UI components
        self.setup_ui()

    def setup_ui(self):
        """Set up the detection tab UI with responsive design"""
        # Configure grid layout
        self.parent.grid_columnconfigure(0, weight=3)  # Video takes 3/4
        self.parent.grid_columnconfigure(1, weight=1)  # Controls take 1/4
        self.parent.grid_rowconfigure(0, weight=1)

        # Left side - Video feed (resizable)
        self.video_frame = Frame(self.parent, bg='black')
        self.video_frame.grid(row=0, column=0, sticky=NSEW, padx=5, pady=5)
        self.video_frame.grid_rowconfigure(0, weight=1)
        self.video_frame.grid_columnconfigure(0, weight=1)

        # Canvas for the video display
        self.video_canvas = Frame(self.video_frame, bg='black')
        self.video_canvas.grid(row=0, column=0, sticky=NSEW)

        # Right side - Controls (fixed width but scrollable)
        self.control_outer_frame = Frame(self.parent)
        self.control_outer_frame.grid(row=0, column=1, sticky=NSEW, padx=5, pady=5)
        self.control_outer_frame.grid_rowconfigure(0, weight=1)
        self.control_outer_frame.grid_columnconfigure(0, weight=1)

        # Add a canvas and scrollbar for the controls
        self.control_canvas = Canvas(self.control_outer_frame)
        self.control_scrollbar = ttk.Scrollbar(self.control_outer_frame, orient="vertical",
                                               command=self.control_canvas.yview)
        self.control_scrollbar.grid(row=0, column=1, sticky=N + S)
        self.control_canvas.grid(row=0, column=0, sticky=NSEW)
        self.control_canvas.configure(yscrollcommand=self.control_scrollbar.set)

        # Create a frame inside the canvas for actual controls
        self.control_frame = Frame(self.control_canvas)
        self.control_canvas.create_window((0, 0), window=self.control_frame, anchor="nw", tags="self.control_frame")

        # Configure control frame to fill canvas width
        self.control_frame.bind("<Configure>", self._configure_control_frame)

        # Mode selection
        self._create_section_label("Detection Mode")
        self.mode_var = StringVar(value="parking")
        self.mode_menu = ttk.Combobox(self.control_frame, textvariable=self.mode_var,
                                      values=["parking", "vehicle"])
        self.mode_menu.pack(fill=X, pady=5, padx=10)
        self.mode_menu.bind("<<ComboboxSelected>>", self.switch_detection_mode)

        # ML detection toggle
        self._create_section_label("ML Detection")
        self.ml_frame = Frame(self.control_frame)
        self.ml_frame.pack(fill=X, pady=5, padx=10)
        self.ml_var = StringVar(value="Off")
        ttk.Label(self.ml_frame, text="Enable ML:").pack(side=LEFT, padx=5)
        self.ml_toggle = ttk.Combobox(self.ml_frame, textvariable=self.ml_var, values=["Off", "On"],
                                      width=5, state="readonly")
        self.ml_toggle.pack(side=LEFT, padx=5, fill=X, expand=True)
        self.ml_toggle.bind("<<ComboboxSelected>>", self.toggle_ml_detection)

        # Create the ML confidence label first
        self.ml_confidence_label = Label(self.control_frame, text=f"Value: {self.app.ml_confidence:.1f}")

        # ML confidence slider - capture the returned slider object
        self.ml_confidence_scale, _ = self._create_control_slider(
            "ML Confidence:", 0.1, 0.9, self.app.ml_confidence,
            self.update_ml_confidence, self.ml_confidence_label)

        # Video source selection
        self._create_section_label("Video Source")
        self.video_source_var = StringVar(value=self.app.video_sources[0] if self.app.video_sources else "0")

        # Use the video sources from the app
        self.video_menu = ttk.Combobox(self.control_frame, textvariable=self.video_source_var,
                                       values=self.app.video_sources)
        self.video_menu.pack(fill=X, pady=5, padx=10)
        self.video_menu.bind("<<ComboboxSelected>>", self.switch_video_source)

        # Status information
        self._create_section_label("Status Information")
        self.status_info = Label(self.control_frame,
                                 text="Total Spaces: 0\nFree Spaces: 0\nOccupied: 0\nVehicles Counted: 0",
                                 font=("Arial", 12), justify=LEFT, relief="groove", bd=2)
        self.status_info.pack(pady=5, fill=X, padx=10)

        # Status indicator
        self.status_label = Label(self.control_frame, text="Status: Stopped", fg="red", font=("Arial", 12))
        self.status_label.pack(pady=5, fill=X, padx=10)

        # Control buttons
        self._create_section_label("Controls")
        self.button_frame = Frame(self.control_frame)
        self.button_frame.pack(fill=X, pady=5, padx=10)

        # Use grid for button layout (3 buttons in a row)
        self.button_frame.columnconfigure(0, weight=1)
        self.button_frame.columnconfigure(1, weight=1)
        self.button_frame.columnconfigure(2, weight=1)

        self.start_button = ttk.Button(self.button_frame, text="Start", command=self.start_detection)
        self.start_button.grid(row=0, column=0, sticky=W + E, padx=2)

        self.stop_button = ttk.Button(self.button_frame, text="Stop", command=self.stop_detection,
                                      state="disabled")
        self.stop_button.grid(row=0, column=1, sticky=W + E, padx=2)

        self.reset_button = ttk.Button(self.button_frame, text="Reset", command=self.reset_counters)
        self.reset_button.grid(row=0, column=2, sticky=W + E, padx=2)

        # Advanced settings
        self._create_section_label("Detection Settings")

        # Create the threshold label first
        self.threshold_label = Label(self.control_frame, text=f"Value: {self.app.parking_threshold}")

        # Threshold slider - capture the returned slider object
        self.threshold_scale, _ = self._create_control_slider(
            "Detection Threshold:", 100, 1000, self.app.parking_threshold,
            self.update_threshold, self.threshold_label)

        # Vehicle detection settings - only show when in vehicle mode
        self.vehicle_settings_frame = Frame(self.control_frame)

        # Line height slider
        self.line_height_label = Label(self.vehicle_settings_frame, text=f"Value: {self.app.line_height}")
        self.line_height_scale, _ = self._create_control_slider(
            "Line Height:", 100, self.app.image_height - 100, self.app.line_height,
            self.update_line_height, self.line_height_label)

        # Contour width slider
        self.contour_width_label = Label(self.vehicle_settings_frame, text=f"Value: {self.app.min_contour_width}")
        self.contour_width_scale, _ = self._create_control_slider(
            "Min Contour Width:", 10, 100, self.app.min_contour_width,
            self.update_contour_width, self.contour_width_label)

        # Contour height slider
        self.contour_height_label = Label(self.vehicle_settings_frame, text=f"Value: {self.app.min_contour_height}")
        self.contour_height_scale, _ = self._create_control_slider(
            "Min Contour Height:", 10, 100, self.app.min_contour_height,
            self.update_contour_height, self.contour_height_label)

        # Offset slider
        self.offset_label = Label(self.vehicle_settings_frame, text=f"Value: {self.app.offset}")
        self.offset_scale, _ = self._create_control_slider(
            "Line Offset:", 5, 50, self.app.offset,
            self.update_offset, self.offset_label)

        # Only show vehicle settings if vehicle mode is selected
        if self.app.detection_mode == "vehicle":
            self.vehicle_settings_frame.pack(fill=X, pady=10, padx=10)

        # Debug mode
        self.debug_frame = Frame(self.control_frame)
        self.debug_frame.pack(fill=X, pady=5, padx=10)
        self.debug_var = StringVar(value="Off")
        ttk.Label(self.debug_frame, text="Debug Mode:").pack(side=LEFT, padx=5)
        self.debug_toggle = ttk.Combobox(self.debug_frame, textvariable=self.debug_var,
                                         values=["Off", "On"], width=5, state="readonly")
        self.debug_toggle.pack(side=LEFT, padx=5, fill=X, expand=True)

    def update_line_height(self, value):
        """Update the line height for vehicle counting"""
        self.app.line_height = int(float(value))
        self.line_height_label.config(text=f"Value: {self.app.line_height}")

    def update_contour_width(self, value):
        """Update the minimum contour width for vehicle detection"""
        self.app.min_contour_width = int(float(value))
        self.contour_width_label.config(text=f"Value: {self.app.min_contour_width}")

    def update_contour_height(self, value):
        """Update the minimum contour height for vehicle detection"""
        self.app.min_contour_height = int(float(value))
        self.contour_height_label.config(text=f"Value: {self.app.min_contour_height}")

    def update_offset(self, value):
        """Update the line offset for vehicle counting"""
        self.app.offset = int(float(value))
        self.offset_label.config(text=f"Value: {self.app.offset}")

    def _create_section_label(self, text):
        """Create a section header label"""
        label = ttk.Label(self.control_frame, text=text, font=("Arial", 11, "bold"))
        label.pack(anchor="w", pady=(10, 0), padx=10)
        # Add separator line
        separator = ttk.Separator(self.control_frame, orient="horizontal")
        separator.pack(fill=X, pady=2, padx=10)

    def _create_control_slider(self, label_text, from_val, to_val, default_val, command, value_label=None):
        """Create a labeled slider control"""
        frame = Frame(self.control_frame)
        frame.pack(fill=X, pady=5, padx=10)

        # Label
        ttk.Label(frame, text=label_text).pack(anchor="w")

        # Create a container for slider and value label
        slider_frame = Frame(frame)
        slider_frame.pack(fill=X, pady=2)
        slider_frame.columnconfigure(0, weight=1)
        slider_frame.columnconfigure(1, weight=0)

        # Slider
        slider = ttk.Scale(slider_frame, from_=from_val, to=to_val, orient="horizontal", value=default_val)
        slider.grid(row=0, column=0, sticky=W + E, padx=(0, 5))

        # Value label
        if value_label is None:
            value_label = Label(slider_frame, text=f"Value: {default_val}", width=8)
            value_label.grid(row=0, column=1, sticky=E)
        else:
            # If a label is already provided, just place it in the grid
            value_label.config(width=8)
            value_label.grid(row=0, column=1, sticky=E)

        # Set command
        def update_and_callback(val):
            value_label.config(text=f"Value: {float(val):.1f}")
            command(val)

        slider.config(command=update_and_callback)
        return slider, value_label

    def _configure_control_frame(self, event):
        """Configure the control frame scrolling region when it changes size"""
        # Update the scroll region to include the entire frame
        self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all"))
        # Set the control frame width to match canvas width
        self.control_canvas.itemconfig("self.control_frame", width=self.control_canvas.winfo_width())

    def _create_section_label(self, text):
        """Create a section header label"""
        label = ttk.Label(self.control_frame, text=text, font=("Arial", 11, "bold"))
        label.pack(anchor="w", pady=(10, 0), padx=10)
        # Add separator line
        separator = ttk.Separator(self.control_frame, orient="horizontal")
        separator.pack(fill=X, pady=2, padx=10)

    def _create_control_slider(self, label_text, from_val, to_val, default_val, command, label_var=None):
        """Create a labeled slider control"""
        frame = Frame(self.control_frame)
        frame.pack(fill=X, pady=5, padx=10)

        # Label
        ttk.Label(frame, text=label_text).pack(anchor="w")

        # Create a container for slider and value label
        slider_frame = Frame(frame)
        slider_frame.pack(fill=X, pady=2)
        slider_frame.columnconfigure(0, weight=1)
        slider_frame.columnconfigure(1, weight=0)

        # Slider
        slider = ttk.Scale(slider_frame, from_=from_val, to=to_val, orient="horizontal", value=default_val)
        slider.grid(row=0, column=0, sticky=W + E, padx=(0, 5))

        # Value label
        if not label_var:
            label_var = Label(slider_frame, text=f"Value: {default_val}", width=8)
        else:
            label_var = Label(slider_frame, text=f"Value: {default_val}", width=8)
        label_var.grid(row=0, column=1, sticky=E)

        # Set command
        def update_and_callback(val):
            label_var.config(text=f"Value: {float(val):.1f}")
            command(val)

        slider.config(command=update_and_callback)
        return slider, label_var

    def _configure_control_frame(self, event):
        """Configure the control frame scrolling region when it changes size"""
        # Update the scroll region to include the entire frame
        self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all"))
        # Set the control frame width to match canvas width
        self.control_canvas.itemconfig("self.control_frame", width=self.control_canvas.winfo_width())

    def update_status_info(self, total_spaces, free_spaces, occupied_spaces, vehicle_counter):
        """Update the status information display"""
        status_text = f"Total Spaces: {total_spaces}\n"
        status_text += f"Free Spaces: {free_spaces}\n"
        status_text += f"Occupied: {occupied_spaces}\n"
        status_text += f"Vehicles Counted: {vehicle_counter}"

        # Update the status info label
        self.status_info.config(text=status_text)

    def start_detection(self):
        """Start the detection process"""
        if not self.app.running:
            self.switch_video_source(self.video_source_var.get())

            if self.app.video_capture and self.app.video_capture.isOpened():
                self.app.running = True
                self.status_label.config(text="Status: Running", fg="green")
                self.start_button.config(state="disabled")
                self.stop_button.config(state="normal")
                self.app.log_event(f"Started {self.app.detection_mode} detection with source {self.app.current_video}")
                self.process_frame()
            else:
                messagebox.showerror("Error", "Could not open video source.")

    def stop_detection(self):
        """Stop the detection process"""
        self.app.running = False
        self.status_label.config(text="Status: Stopped", fg="red")
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.app.log_event(f"Stopped {self.app.detection_mode} detection")


    def reset_counters(self):
        """Reset the detection counters"""
        self.app.vehicle_counter = 0
        self.app.matches = []
        self.update_status_info(self.app.total_spaces, self.app.free_spaces, self.app.occupied_spaces,
                                self.app.vehicle_counter)
        self.app.log_event("Reset counters")

    # Replace the switch_detection_mode method in ui/detection_tab.py
    def switch_detection_mode(self, event=None):
        """Switch between parking space detection and vehicle counting modes"""
        # Get requested mode
        mode = self.mode_var.get()
        previous_mode = self.app.detection_mode

        # Check if we're already running
        was_running = self.app.running

        # Always stop detection first to release resources
        if was_running:
            self.stop_detection()

        # Update mode
        self.app.detection_mode = mode
        self.app.log_event(f"Switched to {mode} detection mode")

        # Store current reference image before switching
        current_ref = self.app.current_reference_image

        # Update UI elements based on mode
        if mode == "parking":
            self.threshold_scale.configure(from_=100, to=1000)
            self.threshold_scale.set(self.app.parking_threshold)
            self.threshold_label.config(text=f"Value: {self.app.parking_threshold}")

            # Hide vehicle settings
            if hasattr(self, 'vehicle_settings_frame'):
                self.vehicle_settings_frame.pack_forget()

            # Make sure we're using the proper reference image for the current video
            if self.app.current_video in self.app.video_reference_map:
                self.app.current_reference_image = self.app.video_reference_map[self.app.current_video]
                self.app.load_parking_positions(self.app.current_reference_image)

        else:  # vehicle mode
            self.threshold_scale.configure(from_=10, to=100)
            self.threshold_scale.set(20)  # Default for vehicle detection
            self.threshold_label.config(text=f"Value: 20")

            # Show vehicle settings
            if hasattr(self, 'vehicle_settings_frame'):
                self.vehicle_settings_frame.pack(fill=X, pady=10, padx=10)

            # Keep the current reference image, don't change it
            if previous_mode != mode and current_ref:
                self.app.current_reference_image = current_ref

            # Reset vehicle detection parameters
            self.reset_vehicle_detection()

        # Force video source reset
        self.force_reset_video()

        # Restart if was running
        if was_running:
            # Use after to ensure UI has time to update
            self.parent.after(100, self.start_detection)

    def reset_vehicle_detection(self):
        """Reset vehicle detection parameters and counters"""
        # Reset counters
        self.app.matches = []
        self.app.vehicle_counter = 0

        # Reset frame tracking variables
        if hasattr(self, 'prev_frame'):
            self.prev_frame = None

        self.frame_count = 0
        self.last_detections = []

        # Update the UI
        self.update_status_info(
            self.app.total_spaces,
            self.app.free_spaces,
            self.app.occupied_spaces,
            self.app.vehicle_counter
        )

        self.app.log_event("Reset vehicle detection parameters")

    def toggle_ml_detection(self, event=None):
        new_value = self.ml_var.get()
        self.app.log_event(f"ML detection toggle: {new_value}")

        if new_value == "On":
            # Initialize in separate thread without blocking UI
            def init_ml_detector():
                try:
                    self.app.ml_detector = VehicleDetector(confidence_threshold=self.app.ml_confidence)
                    # Update UI from main thread
                    self.parent.after(0, lambda: self.app.log_event("ML detector initialized successfully"))
                    self.app.use_ml_detection = True
                except Exception as e:
                    # Handle error in main thread
                    self.parent.after(0, lambda: self.handle_ml_error(str(e)))

            # Show loading message
            self.status_label.config(text="Loading ML model...", fg="orange")
            # Start initialization thread
            threading.Thread(target=init_ml_detector, daemon=True).start()
        else:
            # Disable ML detection
            self.app.use_ml_detection = False
            self.app.log_event("ML detection disabled")

    def reset_ml_state(self):
        """Reset the ML detection state"""
        # Reset frame tracking variables
        self.prev_frame = None
        self.frame_count = 0
        self.last_detections = []

        # Clear any existing matches
        self.app.matches = []

        # Update the UI
        self.update_status_info(
            self.app.total_spaces,
            self.app.free_spaces,
            self.app.occupied_spaces,
            self.app.vehicle_counter
        )

        self.app.log_event("Reset ML detection state")

    def initialize_ml_detector(self):
        """Initialize the machine learning detector with progress display"""
        loading_window = None
        try:
            # Show loading message
            from tkinter import Toplevel
            loading_window = Toplevel(self.app.master)
            loading_window.title("Loading Model")
            loading_window.geometry("300x100")
            loading_window.resizable(False, False)
            loading_window.transient(self.app.master)
            loading_window.grab_set()

            Label(loading_window, text="Loading ML model...", font=("Arial", 12)).pack(pady=20)
            loading_window.update()

            self.app.log_event("Initializing ML detector...")
            self.app.ml_detector = VehicleDetector(confidence_threshold=self.app.ml_confidence)

            # Close loading window
            loading_window.destroy()

            self.app.log_event("ML detector initialized successfully")
            return True
        except Exception as e:
            if 'loading_window' in locals() and loading_window.winfo_exists():
                loading_window.destroy()

            messagebox.showerror("ML Error", f"Failed to initialize ML detector: {str(e)}")
            self.app.log_event(f"ML detector initialization failed: {str(e)}")
            return False

    def update_ml_confidence(self, value):
        """Update ML confidence threshold"""
        confidence = float(value)
        self.app.ml_confidence = confidence
        self.ml_confidence_label.config(text=f"Value: {confidence:.1f}")
        if self.app.ml_detector:
            self.app.ml_detector.confidence_threshold = confidence

    def update_threshold(self, value):
        """Update the detection threshold value"""
        threshold = int(float(value))
        if self.app.detection_mode == "parking":
            self.app.parking_threshold = threshold
        self.threshold_label.config(text=f"Value: {threshold}")

    # Modify the switch_video_source method to include a timeout
    # Fix for the switch_video_source method to include except/finally blocks
    def switch_video_source(self, source):
        # Stop current detection if running
        was_running = self.app.running
        if was_running:
            self.stop_detection()

        # Close existing capture if any
        if self.app.video_capture is not None:
            self.app.video_capture.release()

        try:
            # Handle webcam (integer) or video file
            self.app.current_video = source

            # Use get_video_path to get the correct path
            video_path = get_video_path(source)

            # Use a separate thread for opening the video with timeout
            capture_success = [False]
            capture_complete = threading.Event()

            def open_video_with_timeout():
                try:
                    if source == "0" or video_path == 0:
                        self.app.video_capture = cv2.VideoCapture(0)  # Webcam
                    else:
                        self.app.video_capture = cv2.VideoCapture(video_path)  # Video file

                    capture_success[0] = self.app.video_capture.isOpened()
                finally:
                    capture_complete.set()

            # Start thread to open video
            thread = threading.Thread(target=open_video_with_timeout)
            thread.daemon = True
            thread.start()

            # Wait with timeout (3 seconds)
            capture_complete.wait(3)

            if not capture_complete.is_set() or not capture_success[0]:
                raise Exception(f"Could not open video source: {source} (timeout or failed)")
        except Exception as e:
            # Handle any errors that occur during video source switching
            self.app.log_event(f"Error switching video source: {str(e)}")
            messagebox.showerror("Video Error", f"Could not open video source: {str(e)}")
            return False

        return True  # Successfully switched video source

    def reset_detection_parameters(self):
        """Reset detection parameters when switching sources"""
        if self.app.detection_mode == "vehicle":
            # Reset vehicle detection parameters
            self.app.matches = []
            self.app.vehicle_counter = 0

            # Reset frame tracking variables
            if hasattr(self, 'prev_frame'):
                self.prev_frame = None

            self.frame_count = 0
            self.last_detections = []
        else:
            # Make sure we have the right parking positions
            self.app.load_parking_positions()

    # Add this method to the DetectionTab class in ui/detection_tab.py
    def force_reset_video(self):
        """Reset the video capture object to ensure clean state"""
        try:
            # Close existing capture
            if self.app.video_capture is not None:
                self.app.video_capture.release()
                self.app.video_capture = None

            # Reset frame variables
            self.prev_frame = None
            self.frame_count = 0
            self.last_detections = []

            # Re-open the video source
            video_path = get_video_path(self.app.current_video)

            if self.app.current_video == "0" or video_path == 0:
                self.app.video_capture = cv2.VideoCapture(0)
            else:
                self.app.video_capture = cv2.VideoCapture(video_path)

            # Get frame dimensions
            if self.app.video_capture.isOpened():
                ret, first_frame = self.app.video_capture.read()
                if ret and first_frame is not None:
                    self.app.image_height, self.app.image_width = first_frame.shape[:2]
                    # Reset video to beginning
                    self.app.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

            self.app.log_event(f"Reset video source: {self.app.current_video}")
        except Exception as e:
            self.app.log_event(f"Error resetting video: {str(e)}")

    def process_frame(self):
        """Process video frames for the selected detection mode"""
        if not self.app.running:
            return

        try:
            # Read frame from video
            success, img = self.app.video_capture.read()

            # Reset video if at end
            if not success:
                self.app.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success, img = self.app.video_capture.read()
                if not success:
                    self.status_label.config(text="Status: Video Error", fg="red")
                    self.app.log_event("Video error occurred")
                    self.stop_detection()
                    return

            # Resize frame if necessary
            if img.shape[1] != self.app.image_width or img.shape[0] != self.app.image_height:
                img = cv2.resize(img, (self.app.image_width, self.app.image_height))

             # Process based on selected mode
            if self.app.detection_mode == "parking":
                # Process for parking detection
                imgProcessed = preprocess_frame_for_parking_detection(img)

                # Check parking spaces
                debug_mode = hasattr(self, 'debug_var') and self.debug_var.get() == "On"
                processed_img, free_spaces, occupied_spaces, total_spaces = process_parking_spaces(
                    imgProcessed, img.copy(), self.app.posList, self.app.parking_threshold, debug=debug_mode
                )

                # Update app state
                self.app.free_spaces = free_spaces
                self.app.occupied_spaces = occupied_spaces
                self.app.total_spaces = total_spaces

                # Update the processed image
                img = processed_img
                self.update_parking_data_for_allocation(imgProcessed)

            if self.prev_frame is None or self.frame_count == 0:
                self.prev_frame = img.copy()
                self.frame_count = 1

                if hasattr(self.app, 'allocation_tab'):
                    self.parent.after(100, self.app.allocation_tab.update_visualization)
                    self.parent.after(100, self.app.allocation_tab.update_statistics)

                # Schedule next frame
                self.parent.after(10, self.process_frame)
                return  # Skip processing for initialization frame
            self.frame_count += 1
            # Use traditional method by default (which is more stable)
            use_ml = False
            if self.app.use_ml_detection and self.app.ml_detector:
                try:
                    # Add debug logs
                    print(f"Using ML detection for frame {self.frame_count}")

                    # Only run ML detection on certain frames to improve performance
                    if self.frame_count % self.frame_skip == 0:
                        # Get vehicle detections
                        detections = self.app.ml_detector.detect_vehicles(img)

                        # Ensure detections isn't None
                        if detections is None:
                            detections = []
                            print("Warning: ML detector returned None")

                        # Store for use in skipped frames
                        self.last_detections = detections
                        print(f"Detected {len(detections)} vehicles with ML")
                    else:
                        # Use the last known detections for in-between frames
                        detections = self.last_detections if hasattr(self,
                                                                     'last_detections') and self.last_detections is not None else []

                    # Check if we have valid detections to process
                    if not isinstance(detections, list):
                        raise TypeError(f"Expected list of detections but got {type(detections)}")

                    # Process the ML detections
                    processed_img, new_matches, new_vehicle_counter = process_ml_detections(
                        img.copy(),
                        detections,
                        self.app.line_height,
                        self.app.offset,
                        self.app.matches,
                        self.app.vehicle_counter,
                        self.app.ml_detector.classes
                    )

                    # Update app state
                    self.app.matches = new_matches
                    self.app.vehicle_counter = new_vehicle_counter

                    # Update the processed image
                    img = processed_img

                except Exception as e:
                    print(f"ML detection error: {str(e)}")
                    self.app.log_event(f"ML detection error: {str(e)}")
                    # Fallback to traditional method
                    processed_img, new_matches, new_vehicle_counter = detect_vehicles_traditional(
                        img.copy(),
                        self.prev_frame,
                        self.app.line_height,
                        self.app.min_contour_width,
                        self.app.min_contour_height,
                        self.app.offset,
                        self.app.matches,
                        self.app.vehicle_counter
                    )

                    # Update app state
                    self.app.matches = new_matches
                    self.app.vehicle_counter = new_vehicle_counter

                    # Update the processed image
                    img = processed_img

            # Update the previous frame for the next iteration
            self.prev_frame = img.copy()

            # Convert to RGB for display
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Convert to PIL format and then to Tkinter format
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)

            # Display the image
            if hasattr(self, 'image_label'):
                self.image_label.configure(image=img_tk)
                self.image_label.image = img_tk
            else:
                self.image_label = Label(self.video_canvas, image=img_tk)
                self.image_label.pack(fill=BOTH, expand=True)
                self.image_label.image = img_tk

            # Update status information
            self.update_status_info(
                self.app.total_spaces,
                self.app.free_spaces,
                self.app.occupied_spaces,
                self.app.vehicle_counter
            )

            # Add this line to ensure the allocation tab is updated
            if hasattr(self.app, 'allocation_tab'):
                self.app.allocation_tab.update_visualization()
                self.app.allocation_tab.update_statistics()

            # Schedule next frame processing
            if self.app.detection_mode == "parking" or not self.app.use_ml_detection:
                self.parent.after(10, self.process_frame)  # Standard delay
            else:
                self.parent.after(1, self.process_frame)  # Faster for ML detection

        except Exception as e:
            self.app.log_event(f"Error processing frame: {str(e)}")
            messagebox.showerror("Error", f"Error processing video frame: {str(e)}")
            self.stop_detection()


    # Add this new method to DetectionTab class
    def update_parking_data_for_allocation(self, img_pro):
        """Update parking data for allocation system"""
        try:
            # Make sure app has parking_manager
            if not hasattr(self.app, 'parking_manager'):
                self.app.log_event("No parking manager found")
                return

            # Create parking_data if it doesn't exist in parking_manager
            if not hasattr(self.app.parking_manager, 'parking_data'):
                self.app.parking_manager.parking_data = {}

            # Update parking spaces data
            for i, (x, y, w, h) in enumerate(self.app.posList):
                # Ensure coordinates are within image bounds
                if (y >= 0 and y + h < img_pro.shape[0] and x >= 0 and x + w < img_pro.shape[1]):
                    # Get crop of parking space
                    img_crop = img_pro[y:y + h, x:x + w]
                    count = cv2.countNonZero(img_crop)
                    is_occupied = count >= self.app.parking_threshold

                    # Generate section based on position
                    section = "A" if x < img_pro.shape[1] / 2 else "B"
                    section += "1" if y < img_pro.shape[0] / 2 else "2"

                    # Full space ID
                    space_id = f"S{i + 1}-{section}"

                    # Update or create parking space data
                    if space_id not in self.app.parking_manager.parking_data:
                        self.app.parking_manager.parking_data[space_id] = {
                            'position': (x, y, w, h),
                            'occupied': is_occupied,
                            'vehicle_id': None,
                            'last_state_change': datetime.now(),
                            'distance_to_entrance': x + y,  # Simple distance estimation
                            'section': section
                        }
                    else:
                        # Just update occupancy status
                        self.app.parking_manager.parking_data[space_id]['occupied'] = is_occupied

            self.app.log_event(f"Updated parking data for {len(self.app.posList)} spaces")
        except Exception as e:
            self.app.log_event(f"Error updating parking allocation data: {str(e)}")