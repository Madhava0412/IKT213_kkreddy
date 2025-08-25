import cv2
import os

base_path = r"D:\PROJECTS\UIA\MACHINE_VISION\Assignment_1\IKT213_kkreddy\assignment_1"
output_dir = os.path.join(base_path, "solutions")
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, "camera_outputs.txt")

# Try to open camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ No webcam detected. Writing default values.")
    fps, width, height = "None", "None", "None"
else:
    fps = cap.get(cv2.CAP_PROP_FPS) or "Unknown"
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

# Save results
with open(output_file, "w") as f:
    f.write(f"fps: {fps}\n")
    f.write(f"width: {width}\n")
    f.write(f"height: {height}\n")

print(f"✅ Camera info saved to {output_file}")
