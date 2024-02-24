import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import math
from ultralytics import YOLO


class ObjectDetectionApp:
    def __init__(self, root):
        # Initialize the GUI
        self.root = root
        self.root.title("Object Detection GUI")
        self.root.geometry("800x600")
        self.root.configure(bg="lightblue")  # Set the background color

        # Initialize variables
        self.cap = None
        self.model = YOLO("ppe.pt")
        self.class_names = [
            'Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'E',
            'Safety Vest', 'E', 'E'
        ]
        self.confidence_threshold = 0.5

        # Create GUI elements
        self.create_widgets()

    def create_widgets(self):
        # Create a frame
        frame = tk.Frame(self.root, bg="lightblue")  # Set the frame background color
        frame.place(relx=0.5, rely=0.5, anchor="center")

        # Create buttons
        select_video_button = tk.Button(
            frame, text="Select Video for Detection", command=self.open_and_start_detection,
            width=30, height=2
        )
        live_button = tk.Button(
            frame, text="Start Live Detection", command=self.start_live_detection,
            width=20, height=2
        )

        # Set button colors
        select_video_button.configure(bg="green", fg="white")
        live_button.configure(bg="orange", fg="white")

        # Place buttons in the frame
        select_video_button.grid(row=0, column=0, pady=10)
        live_button.grid(row=1, column=0, pady=10)

    def open_and_start_detection(self):
        # Open a video file for detection
        file_path = filedialog.askopenfilename()
        self.cap = cv2.VideoCapture(file_path)

        if self.cap is None:
            messagebox.showerror("Error", "Please select a valid video file.")
            return

        while True:
            success, img = self.cap.read()
            if not success:
                break

            self.process_frame(img)

        self.release_capture()

    def start_live_detection(self):
        # Start live detection from the camera
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            messagebox.showerror("Error", "Failed to open the camera.")
            return

        while True:
            success, img = self.cap.read()
            if not success:
                break

            self.process_frame(img, resize=True)

        self.release_capture()

    def process_frame(self, img, resize=False):
        if resize:
            img = cv2.resize(img, (800, 600))

        # Perform object detection
        results = self.model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, img.shape[1]), min(y2, img.shape[0])

                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])

                print("Detected class index:", cls)  # Print for debugging

                try:
                    current_class = self.class_names[cls]
                except IndexError:
                    print("Error: Class index out of range in self.class_names.")
                    current_class = "Unknown"

                if conf > self.confidence_threshold:
                    my_color = (0, 0, 255) if "NO-" in current_class else (
                        0, 255, 0) if current_class in ['Hardhat', 'Safety Vest', 'Mask'] else (255, 0, 0)

                    label = f'{current_class} {conf}'
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                    text_x, text_y = x1, max(y1 - 10, 0)
                    box_color = my_color

                    cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
                    cv2.rectangle(img, (text_x, text_y), (text_x + text_size[0], text_y - text_size[1]), box_color, -1)
                    cv2.putText(img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display the image with detection results
        cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)  # Set window to normal size
        cv2.setWindowProperty("Object Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Object Detection", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            self.release_capture()

    def release_capture(self):
        # Release video capture resources
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Run the application
    root_instance = tk.Tk()
    app = ObjectDetectionApp(root_instance)
    root_instance.mainloop()
