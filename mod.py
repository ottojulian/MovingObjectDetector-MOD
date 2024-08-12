import cv2
from pythonosc import udp_client
import numpy as np
from PIL import ImageFont, ImageDraw, Image

# Initialize the video capture from camera index 0
cap = cv2.VideoCapture(0)  # Change the camera index as needed
#cap = cv2.VideoCapture('C:/Users/Laboratorio01/Videos/2023-11-03 14-39-52.mov')

# Initialize the background subtractor (MOG2 is used here)
bg_subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()

# Initialize an OSC client to send messages
osc_client = udp_client.SimpleUDPClient("127.0.0.1", 12346)  # Replace with the appropriate IP and port

# Number of frames to skip before evaluating
frame_skip = 1  # Adjust as needed

frame_count = 0  # Initialize the frame counter

# Flags to control the visualization mode
visualization_mode = 0  # 0 - Camera view, 1 - Contour mask view, 2 - Background Subtraction Mask view

# Flags to control the visibility of the on-screen menu
show_menu = True

# Color of the menu text
menu_color = (0, 0, 0)

# Font for the menu text
font = cv2.FONT_HERSHEY_DUPLEX

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Salteador de frames
    frame_count += 1

    if frame_count % frame_skip != 0:
        continue  # Skip frames

    # Apply background subtraction
    fg_mask = bg_subtractor.apply(frame)

    # Find contours in the foreground mask
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frame_height, frame_width, _ = frame.shape  # Get the dimensions of the frame

    # Default values for the normalized centroid coordinates
    normalized_cx = 0.0
    normalized_cy = 0.0

    for i, contour in enumerate(contours):
        if 3000 < cv2.contourArea(contour) < 70000:  # Adjust the minimum and maximum contour area as needed
            # Get the bounding rectangle for the contour
            x, y, w, h = cv2.boundingRect(contour)

            # Calculate the centroid (center) of the bounding box
            cx = x + w // 2
            cy = y + h // 2

            # Normalize the centroid coordinates between 0 and 1
            normalized_cx = cx / frame_width
            normalized_cy = cy / frame_height

            # Draw a bounding box around the object
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw a central cross (plus shape)
            cross_size = 5  # Adjust the size of the cross as needed
            cv2.circle(frame, (cx, cy), 1, (0, 0, 255), 2)
            txtOffset = 10
            cv2.putText(frame, f'{normalized_cx:.2f}, {normalized_cy:.2f}', (cx + txtOffset, cy - txtOffset), font, 0.35, (0, 0, 255), 1)
            lineOffset = 4
            cv2.line(frame, (cx, cy), (cx + txtOffset - lineOffset, cy - txtOffset + lineOffset), (0, 0, 255), 1)
            cv2.line(frame, (cx + txtOffset - lineOffset, cy - txtOffset + lineOffset), (cx + txtOffset + 20, cy - txtOffset + lineOffset), (0, 0, 255), 1)
            osc_client.send_message("/Xcoord", [normalized_cx])
            osc_client.send_message("/Ycoord", [normalized_cy])
    # Send normalized coordinates over OSC
    

    if visualization_mode == 1:
        # Contour mask view
        contour_mask = np.zeros_like(fg_mask)
        cv2.drawContours(contour_mask, contours, -1, (255), 1)
        frame = contour_mask
    elif visualization_mode == 2:
        # Background Subtraction Mask view
        frame = fg_mask

    # Display or hide the on-screen menu based on the '0' key
    if show_menu:
        # Add the keyboard controls text in the top left corner
        cv2.putText(frame, "Controls:", (5, 30), font, 0.5, menu_color, 1)
        cv2.putText(frame, "1_Camera View", (15, 50), font, 0.5, menu_color, 1)
        cv2.putText(frame, "2_Contour Mask", (15, 70), font, 0.5, menu_color, 1)
        cv2.putText(frame, "3_BG Subtraction Mask", (15, 90), font, 0.5, menu_color, 1)
        cv2.putText(frame, "0_Show/Hide Menu", (15, 110), font, 0.5, menu_color, 1)

    # Display the frame based on the selected visualization mode
    cv2.imshow('Moving Objects Detection', frame)

    key = cv2.waitKey(30) & 0xFF
    if key == 27:  # Press 'Esc' to exit
        break
    elif key == 49:  # Press '1' for Camera View
        visualization_mode = 0
    elif key == 50:  # Press '2' for Contour Mask View
        visualization_mode = 1
    elif key == 51:  # Press '3' for Background Subtraction Mask View
        visualization_mode = 2
    elif key == 48:  # Press '0' to show or hide the menu
        show_menu = not show_menu

cap.release()
cv2.destroyAllWindows()
