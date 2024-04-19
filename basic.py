import cv2
import numpy as np
# Function to perform Non-Maximum Suppression

def estimate_distance_from_camera(human_height, focal_length, object_height_in_pixels):
    # Calculate distance using the pinhole camera model
    distance = (human_height * focal_length) / object_height_in_pixels
    return distance

def apply_nms(boxes, confidences, threshold=0.5):
    indices = cv2.dnn.NMSBoxes(boxes, confidences, threshold, 0.4)
    return indices


# Load YOLO model and COCO dataset names
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Camera focal length (in pixels), you need to measure or estimate this value
focal_length = 1000

# Height of the human in the real world (in meters), you can adjust this value as needed
human_height = 1.7

# Variables for motion tracking
previous_center_x = None
previous_center_y = None
direction_x = None
direction_y = None

# Function to perform human detection on each frame, estimate distance, and track motion
def detect_humans_estimate_distance_and_track_motion(frame):
    global previous_center_x, previous_center_y, direction_x, direction_y

    height, width = frame.shape[:2]

    # Preprocess the frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Get detections
    outs = net.forward(output_layers)
    bboxes = []
    confidences = []

    # Process detections and draw bounding boxes
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and class_id == 0:  # 0 corresponds to the "person" class in COCO dataset
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Calculate the top-left corner of the bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                # Store bounding box and confidence for NMS
                bboxes.append([x, y, w, h])
                confidences.append(float(confidence))
    nms_indices = apply_nms(bboxes, confidences)
    for i in nms_indices:
        x, y, w, h = bboxes[i]
        center_x = x + w // 2
        center_y = y + h // 2
        # Draw bounding box around the human
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Estimate distance from the camera
        object_height_in_pixels = h
        distance = estimate_distance_from_camera(human_height, focal_length, object_height_in_pixels)
        distance_text = f"Distance: {distance:.2f} meters"
        cv2.putText(frame, distance_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Track horizontal motion and determine direction (left or right)
        if previous_center_x is not None:
            if center_x < previous_center_x:
                direction_x = "Moving Left"
            elif center_x > previous_center_x:
                direction_x = "Moving Right"

        previous_center_x = center_x

                # Track vertical motion and determine direction (up or down)
        if previous_center_y is not None:
            if center_y < previous_center_y:
                direction_y = "Moving Front"
            elif center_y > previous_center_y:
                direction_y = "Moving Back"

        previous_center_y = center_y
        # Apply Non-Maximum Suppression to remove redundant bounding boxes


    return frame, direction_x, direction_y

# Open video capture
cap = cv2.VideoCapture(0)  # Change to 1 if you have multiple cameras

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame, direction_x, direction_y = detect_humans_estimate_distance_and_track_motion(frame)

    # Print direction messages on the laptop's console
    if direction_x is not None:
        print(f"Horizontal Direction: {direction_x}")
    if direction_y is not None:
        print(f"Vertical Direction: {direction_y}")

    # Display the resulting frame
    cv2.imshow("Human Detection, Distance Estimation, and Motion Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
