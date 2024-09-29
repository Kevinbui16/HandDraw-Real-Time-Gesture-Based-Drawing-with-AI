import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands.
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Open the webcam.
cap = cv2.VideoCapture(0)

# Create a blank image for drawing.
drawing_image = np.zeros((480, 640, 3), dtype=np.uint8)

# Variables to store the previous position of the index finger tip.
prev_x, prev_y = None, None

# Define the color palette.
colors = {
    'red': (0, 0, 255),
    'yellow': (0, 255, 255),
    'blue': (255, 0, 0),
    'green': (0, 255, 0)
}
color_names = list(colors.keys())
current_color = colors['red']

def draw_color_palette(image):
    """Draw color palette on the top right horizontally with spacing."""
    height, width, _ = image.shape
    palette_x = width - 540  # Start position from right side, moved further left
    spacing = 20  # Spacing between color boxes
    box_size = 100  # Size of the color boxes
    for i, color_name in enumerate(color_names):
        color = colors[color_name]
        # Draw larger color boxes horizontally with spacing
        cv2.rectangle(image, (palette_x + i * (box_size + spacing), 10), (palette_x + box_size + i * (box_size + spacing), 10 + box_size), color, -1)
        # cv2.putText(image, color_name, (palette_x + i * (box_size + spacing), 10 + box_size + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

def check_if_hand_open(hand_landmarks, image_shape):
    """Check if all five fingers are open."""
    image_height, image_width = image_shape[:2]
    fingers_tips_ids = [mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP,
                        mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP,
                        mp_hands.HandLandmark.PINKY_TIP]
    # Get y-coordinates of finger tips and corresponding MCP joints (bottom of each finger)
    fingers_tips = [hand_landmarks.landmark[tip].y * image_height for tip in fingers_tips_ids]
    fingers_mcp = [hand_landmarks.landmark[mcp].y * image_height for mcp in
                   [mp_hands.HandLandmark.THUMB_CMC, mp_hands.HandLandmark.INDEX_FINGER_MCP,
                    mp_hands.HandLandmark.MIDDLE_FINGER_MCP, mp_hands.HandLandmark.RING_FINGER_MCP,
                    mp_hands.HandLandmark.PINKY_MCP]]

    # Check if all fingers are open (tip is higher than corresponding MCP joint)
    return all(tip < mcp for tip, mcp in zip(fingers_tips, fingers_mcp))

with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Resize the drawing image to match the webcam image size.
        drawing_image = cv2.resize(drawing_image, (image.shape[1], image.shape[0]))

        # Convert the BGR image to RGB.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image to detect hand landmarks.
        results = hands.process(image_rgb)

        # Draw hand landmarks on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=4, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=4, circle_radius=2)
                )

                # Check if all fingers are open, if so, clear the drawing.
                if check_if_hand_open(hand_landmarks, image.shape):
                    drawing_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
                    continue

                # Get the coordinates of the index finger tip and thumb tip.
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                x1 = int(index_finger_tip.x * image.shape[1])
                y1 = int(index_finger_tip.y * image.shape[0])
                x2 = int(thumb_tip.x * image.shape[1])
                y2 = int(thumb_tip.y * image.shape[0])

                # Check if the index finger tip is over the color palette.
                palette_x = image.shape[1] - 540  # Adjust starting position
                spacing = 20  # Spacing between boxes
                box_size = 100  # Size of the color boxes
                for i, color_name in enumerate(color_names):
                    if palette_x + i * (box_size + spacing) <= x1 <= palette_x + box_size + i * (box_size + spacing) and 10 <= y1 <= 10 + box_size:
                        current_color = colors[color_name]

                # Only draw if the index finger and thumb form an "L" shape as per new condition.
                if abs(x1 - x2) > 50 and abs(y1 - y2) > 50 and y2 > y1 and x2 > x1:
                    # Draw a line from the previous position to the current position.
                    if prev_x is not None and prev_y is not None:
                        cv2.line(drawing_image, (prev_x, prev_y), (x1, y1), current_color, 5)

                    # Update the previous position.
                    prev_x, prev_y = x1, y1
                else:
                    # If the fingers do not form an "L" shape, stop drawing.
                    prev_x, prev_y = None, None
        else:
            # Reset the previous position if no hand is detected.
            prev_x, prev_y = None, None

        # Draw the color palette on the image.
        draw_color_palette(image)

        # Combine the webcam image with the drawing image.
        combined_image = cv2.addWeighted(image, 0.5, drawing_image, 0.5, 0)

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Hands Drawing', cv2.flip(combined_image, 1))

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
