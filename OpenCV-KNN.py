import cv2

# Create a background subtractor
bg_subtractor = cv2.createBackgroundSubtractorKNN()

# Open a video capture
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(r"Media/Original Video.mpg")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction
    fg_mask = bg_subtractor.apply(frame)

    # Get the foreground image
    fg_image = cv2.bitwise_and(frame, frame, mask=fg_mask)

    # Display the original frame and the foreground mask
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Foreground Mask', fg_mask)
    cv2.imshow('Foreground Image', fg_image)

    # Press 'q' to exit the loop
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
