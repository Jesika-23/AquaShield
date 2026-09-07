"""
backend/smoke_test.py
Quick manual check for Task 2.
Run from repo root: python -m backend.smoke_test path/to/some_test_image.jpg
"""
import sys

from PIL import Image

from backend.services.inference_service import run_detection


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m backend.smoke_test <path-to-image>")
        sys.exit(1)

    image_path = sys.argv[1]
    image = Image.open(image_path)

    result = run_detection(image, file_name=image_path, conf_threshold=0.25)

    print(f"file_name:         {result.file_name}")
    print(f"image_size:        {result.image_size.width}x{result.image_size.height}")
    print(f"inference_time_ms: {result.inference_time_ms:.1f}")
    print(f"detections:        {len(result.detections)}")
    for det in result.detections:
        print(
            f"  - {det.class_name} (id={det.class_id}) "
            f"conf={det.confidence:.2f} "
            f"bbox=({det.bbox.x1:.0f},{det.bbox.y1:.0f},{det.bbox.x2:.0f},{det.bbox.y2:.0f})"
        )
    print(f"annotated_image_base64: {len(result.annotated_image_base64)} chars")


if __name__ == "__main__":
    main()
