import cv2
import platform

# Path to the Haar Cascade classifier and video files
vehicle_cascade_path = 'car.xml'  # Path to the Haar cascade classifier
input_file = 'inputvideo.webm'  # Input video file
output_file = 'output_video.webm'  # Output video file

# Check system architecture
def check_architecture():
    arch = platform.machine()  # Returns something like 'x86_64', 'aarch64', 'riscv64', etc.
    print(f"Detected architecture: {arch}")
    
    if 'x86' in arch:
        return 'x86'
    elif 'riscv' in arch:
        return 'riscv'
    else:
        return 'unknown'

# Initialize video capture and the classifier
video_stream = cv2.VideoCapture(input_file)
vehicle_classifier = cv2.CascadeClassifier(vehicle_cascade_path)

# Video properties for setting up output
width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = video_stream.get(cv2.CAP_PROP_FPS)

# Define the codec and output video writer
codec = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter(output_file, codec, frame_rate, (width, height))

# Initialize frame processing count
frame_index = 0

# Check system architecture and determine the behavior
architecture = check_architecture()

# Decide whether to open a window or print to terminal based on architecture
if architecture == 'x86':
    try:
        cv2.namedWindow('Vehicle Detection Window', cv2.WINDOW_NORMAL)
        window_opened = True
    except cv2.error:
        window_opened = False  # If cv2.error occurs, we can't open a display window
elif architecture == 'riscv':
    window_opened = False  # RISC-V: no window display, print detections in the terminal
else:
    window_opened = False  # Default behavior if architecture is unknown

# Process the video frames
while True:
    success, frame = video_stream.read()
    if not success:
        break

    # Stop after processing 300 frames
    if frame_index >= 300:
        break

    # Convert frame to grayscale for vehicle detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect vehicles (cars) in the grayscale frame
    detected_vehicles = vehicle_classifier.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=1)

    # Draw rectangles around the detected vehicles
    for (x, y, w, h) in detected_vehicles:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Write the processed frame to the output video
    video_writer.write(frame)

    # If the window can be opened (x86), display the frame
    if window_opened:
        cv2.imshow('Vehicle Detection Window', frame)
    else:
        # If RISC-V or no GUI support, print the detections to the terminal
        print(f"Frame {frame_index}: Detected vehicles - {len(detected_vehicles)}")
        for (x, y, w, h) in detected_vehicles:
            print(f"  Vehicle at [x: {x}, y: {y}, width: {w}, height: {h}]")

    # Increment the frame counter
    frame_index += 1

    # Exit on pressing the 'Esc' key
    if cv2.waitKey(33) == 27:  # 27 is the ASCII value for 'Esc'
        break

# Release the resources after processing is done
video_stream.release()
video_writer.release()
cv2.destroyAllWindows()
