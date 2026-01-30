# Clean Next.js build cache and check for stray images in all public/static folders
import os
import shutil

FRONTEND_DIR = os.path.dirname(os.path.abspath(__file__))
NEXT_DIR = os.path.join(FRONTEND_DIR, ".next")
PUBLIC_DIR = os.path.join(FRONTEND_DIR, "public")

# Remove .next build cache
if os.path.exists(NEXT_DIR):
    shutil.rmtree(NEXT_DIR)
    print(f"Deleted build cache: {NEXT_DIR}")
else:
    print("No .next build cache found.")

# Recursively check for images in public/static folders
for root, dirs, files in os.walk(PUBLIC_DIR):
    for file in files:
        if file.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff")):
            print(f"Found image: {os.path.join(root, file)}")

print("Scan complete. If any images were found above, delete them for a clean demo.")
