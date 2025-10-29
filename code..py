import cv2  # OpenCV library for image processing
import numpy as np  # Numpy library for array operations
import os  # OS library for file operations
from picamera2 import Picamera2  # Picamera2 library for interfacing with the Raspberry Pi camera

# Initialize Picamera2 object
picam2 = Picamera2()
# Configure the camera for video mode with a specified resolution
config = picam2.create_video_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()  # Start the camera


def nothing(x):
    # Placeholder function for trackbar callback
    pass


# Create a full-screen window for the live feed
cv2.namedWindow("Live Feed", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Live Feed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Create trackbars for adjusting various settings
cv2.createTrackbar("Contrast", "Live Feed", 12, 30, nothing)
cv2.createTrackbar("Brightness", "Live Feed", 15, 100, nothing)
cv2.createTrackbar("Canny Min", "Live Feed", 30, 255, nothing)
cv2.createTrackbar("Canny Max", "Live Feed", 50, 255, nothing)
cv2.createTrackbar("CLAHE clipLimit", "Live Feed", 40, 100, nothing)
cv2.createTrackbar("CLAHE tileGridSize", "Live Feed", 10, 16, nothing)
cv2.createTrackbar("Blur Kernel", "Live Feed", 7, 10, nothing)
cv2.createTrackbar("Adaptive Thresh", "Live Feed", 25, 30, nothing)
cv2.createTrackbar("Exposure Time", "Live Feed", 8, 100, nothing)
cv2.createTrackbar("Analogue Gain", "Live Feed", 1, 10, nothing)

# Variable to track the state of the capture button
button_pressed = False


# Function to adjust camera settings based on trackbar positions
def adjust_camera_settings():
    exposure_time = cv2.getTrackbarPos("Exposure Time", "Live Feed")
    analogue_gain = cv2.getTrackbarPos("Analogue Gain", "Live Feed")
    # Set camera controls for exposure time and gain
    picam2.set_controls({"ExposureTime": exposure_time * 1000, "AnalogueGain": analogue_gain})


# Function to process the frame captured from the camera
def process_frame(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Retrieve trackbar positions for image processing parameters
    contrast = cv2.getTrackbarPos("Contrast", "Live Feed") / 10
    brightness = cv2.getTrackbarPos("Brightness", "Live Feed")
    canny_min = cv2.getTrackbarPos("Canny Min", "Live Feed")
    canny_max = cv2.getTrackbarPos("Canny Max", "Live Feed")
    clip_limit = cv2.getTrackbarPos("CLAHE clipLimit", "Live Feed") / 10
    tile_grid_size = cv2.getTrackbarPos("CLAHE tileGridSize", "Live Feed")
    blur_kernel = cv2.getTrackbarPos("Blur Kernel", "Live Feed") * 2 + 1
    adaptive_thresh = cv2.getTrackbarPos("Adaptive Thresh", "Live Feed")

    # Ensure adaptive threshold value is odd
    adaptive_thresh = max(3, (adaptive_thresh // 2) * 2 + 1)
    # Adjust contrast and brightness
    adjusted = cv2.convertScaleAbs(gray, alpha=contrast, beta=brightness)
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    enhanced = clahe.apply(adjusted)
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(enhanced, (blur_kernel, blur_kernel), 0)
    # Perform Canny edge detection
    canny_edges = cv2.Canny(blurred, canny_min, canny_max)
    # Apply adaptive thresholding
    adaptive_thresh_image = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                                  adaptive_thresh, 2)
    # Morphological transformation to close gaps
    kernel = np.ones((3, 3), np.uint8)
    morph_image = cv2.morphologyEx(adaptive_thresh_image, cv2.MORPH_CLOSE, kernel)

    # Define region of interest (ROI) dimensions
    h, w = frame.shape[:2]
    rect_x, rect_y = w // 4, h // 4
    rect_w, rect_h = w // 2 + 50, h // 2
    roi = (slice(rect_y, rect_y + rect_h), slice(rect_x, rect_x + rect_w))
    # Create a mask for the ROI
    mask = np.zeros_like(frame)
    mask[roi] = cv2.cvtColor(morph_image[roi], cv2.COLOR_GRAY2BGR)

    # Combine the original frame with the mask for visual feedback
    combined = frame.copy()
    combined[roi] = cv2.addWeighted(frame[roi], 0.5, mask[roi], 0.5, 0)
    # Draw rectangle around the ROI
    cv2.rectangle(combined, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (0, 255, 0), 2)
    # Add text prompt within the ROI
    cv2.putText(combined, "Place your arm here", (rect_x + 10, rect_y + rect_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0), 2)

    return frame, combined, np.count_nonzero(canny_edges)  # Return processed frames and edge count


# Function to save the processed image
def save_image(image, folder='captured_images'):
    if not os.path.exists(folder):  # Create directory if it does not exist
        os.makedirs(folder)
    filename = os.path.join(folder, "captured_image.png")
    cv2.imwrite(filename, image)  # Save image to specified path
    print(f"Image saved as {filename}")


# Mouse callback function to detect button press events
def mouse_callback(event, x, y, flags, param):
    global button_pressed
    if event == cv2.EVENT_LBUTTONDOWN:
        if 10 < x < 110 and 10 < y < 60:
            button_pressed = True


# Set the mouse callback function for the live feed window
cv2.setMouseCallback("Live Feed", mouse_callback)

try:
    while True:
        adjust_camera_settings()  # Adjust camera settings based on trackbars
        frame = picam2.capture_array()  # Capture frame from the camera
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)  # Convert frame to BGR format
        original, combined_view, edge_count = process_frame(frame_bgr)  # Process the captured frame
        combined_window = np.hstack((frame_bgr, combined_view))  # Combine original and processed views

        # Draw capture button on the combined view
        cv2.rectangle(combined_window, (10, 10), (110, 60), (0, 0, 255), -1)
        cv2.putText(combined_window, "Capture", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Live Feed", combined_window)  # Display the combined view in the window
        print(f"Edges detected: {edge_count}")  # Print the number of edges detected

        if button_pressed:  # Check if the capture button was pressed
            save_image(combined_view)  # Save the processed image
            button_pressed = False

        key = cv2.waitKey(1) & 0xFF  # Check for key press
        if key == ord('q'):  # Exit loop if 'q' is pressed
            break

finally:
    picam2.stop()  # Stop the camera
    cv2.destroyAllWindows()  # Close all OpenCV windows
