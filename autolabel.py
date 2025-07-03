import cv2
import os
import glob

# Set paths
image_folder = r"D:\mosquito_dataset\all_images"
label_folder = os.path.join(image_folder, "labels")
os.makedirs(label_folder, exist_ok=True)

# Get all image paths
image_paths = glob.glob(os.path.join(image_folder, "*.jpg"))

for img_path in image_paths:
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Preprocessing
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        continue

    # Use the largest contour
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    if w * h < 100:  # Skip tiny areas
        continue

    # Normalize bounding box
    img_h, img_w = img.shape[:2]
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h

    # Save YOLO-format label
    label_path = os.path.join(label_folder, os.path.splitext(os.path.basename(img_path))[0] + ".txt")
    with open(label_path, "w") as f:
        f.write(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
