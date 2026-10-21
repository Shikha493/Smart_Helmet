import cv2
import numpy as np
import os
from ultralytics import YOLO

# ==========================================
# 1. USER CONFIGURATION
# ==========================================
YOLO_MODEL_PATH = r"C:\Users\athar\OneDrive\Desktop\Atharva\smart_helmet\yolo11n_40_epochs\weights\best.pt"     # Your trained YOLOv8-seg model
IMAGE_PATH = r"C:\Users\athar\OneDrive\Desktop\Atharva\smart_helmet\image\2020_0328_94_jpg.rf.837331e8e2e53490dea5a6b594f95e30.jpg"  # The image you want to process

# Paths to your transparent PNGs for association
PNG_PATHS = {
    "Plastic": "path/to/bouquet.png",        
    "Paper": "path/to/paper_replacement.png",
    "Metal": "path/to/metal_replacement.png"
}

# ==========================================
# 2. CLASS MAPPINGS & FALLBACK COLORS (BGR)
# ==========================================
class_mapping = {
    "plastic": "Plastic",
    "plastic pet": "Plastic",
    "paper": "Paper",
    "paper container": "Paper",
    "metal": "Metal"
}

# Fallback colors in OpenCV's native BGR format (Blue, Green, Red)
fallback_colors = {
    "Plastic": (0, 255, 0),     # Green
    "Paper": (0, 255, 255),     # Yellow
    "Metal": (150, 150, 150)    # Gray
}

# ==========================================
# 3. HELPER FUNCTION: OVERLAY PNG
# ==========================================
def overlay_transparent(background, overlay, x, y):
    """Overlays a transparent PNG (BGRA) onto a background image (BGR)."""
    bg_h, bg_w = background.shape[:2]
    h, w = overlay.shape[:2]

    # Ensure overlay fits within background bounds
    if x >= bg_w or y >= bg_h:
        return background

    h = min(h, bg_h - y)
    w = min(w, bg_w - x)
    overlay = overlay[:h, :w]

    # Split overlay into BGR and Alpha channels
    overlay_img = overlay[..., :3]
    alpha = overlay[..., 3] / 255.0 

    y1, y2 = y, y + h
    x1, x2 = x, x + w

    # Perform alpha blending (works perfectly for BGR)
    for c in range(3):
        background[y1:y2, x1:x2, c] = (alpha * overlay_img[:, :, c] +
                                       (1 - alpha) * background[y1:y2, x1:x2, c])
    return background

# ==========================================
# 4. LOAD MODEL & RUN INFERENCE
# ==========================================
print("Loading YOLO model...")
model = YOLO(YOLO_MODEL_PATH)

print("Processing image...")
# Load image natively in BGR (no conversion needed for cv2.imshow)
original_bgr = cv2.imread(IMAGE_PATH)
if original_bgr is None:
    raise ValueError("Image not found. Check IMAGE_PATH.")

img_h, img_w = original_bgr.shape[:2]

# Run YOLO inference
results = model(original_bgr)[0]

# --- Generate the 3 Visualizations ---
detection_img = results.plot(boxes=True, masks=False)
segmentation_img = results.plot(boxes=False, masks=True)
association_img = original_bgr.copy()

# ==========================================
# 5. APPLY ASSOCIATIONS
# ==========================================
if results.boxes is not None and results.masks is not None:
    for box, mask_data in zip(results.boxes, results.masks.xy):
        
        class_id = int(box.cls[0])
        original_label = model.names[class_id].lower()
        simplified_label = class_mapping.get(original_label, "Plastic") 

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        box_w, box_h = x2 - x1, y2 - y1

        png_path = PNG_PATHS.get(simplified_label, "")

        # PNG Overlay
        if os.path.exists(png_path):
            # Load PNG with Alpha Channel (BGRA)
            replacement_png = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
            resized_png = cv2.resize(replacement_png, (box_w, box_h))
            association_img = overlay_transparent(association_img, resized_png, x1, y1)

        # Fallback: Color Mask
        else:
            color = fallback_colors.get(simplified_label, (0, 0, 255)) # Default Red (BGR)
            
            binary_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            pts = np.array(mask_data, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(binary_mask, [pts], 255)

            colored_overlay = np.zeros_like(association_img)
            colored_overlay[binary_mask == 255] = color

            mask_indices = binary_mask == 255
            association_img[mask_indices] = cv2.addWeighted(
                association_img[mask_indices], 0.4, 
                colored_overlay[mask_indices], 0.6, 
                0
            )

# ==========================================
# 6. DISPLAY RESULTS & WAIT FOR 'Q'
# ==========================================
print("Press 'q' on any image window to close everything.")

# Show all 3 windows
cv2.imshow("1. Detection", detection_img)
cv2.imshow("2. Segmentation", segmentation_img)
cv2.imshow("3. Association", association_img)

# Infinite loop to keep windows open until 'q' is pressed
while True:
    # Wait for 1 millisecond and get the pressed key
    key = cv2.waitKey(1) & 0xFF
    
    # If the pressed key is 'q', break the loop
    if key == ord('q'):
        break

# Clean up and close all windows
cv2.destroyAllWindows()
print("Windows closed.")