#!pip install pillow



#open the image
image = Image.open("/content/drive/MyDrive/pro.imagem/word-written.jpg")

# Print the image details

print(f"Image format: {image.format}")
print(f"Image size: {image.size}")
print(f"Image mode: {image.mode}")
print(f"Image width: {image.width}")
print(f"Image height: {image.height}")

# Access the image data
pixels = image.load()

# Get a specific pixel value
pixel_value = pixels[0, 0]
print(f"\nPixel value at (0, 0): {pixel_value}")

# Iterate over all pixels
for x in range(image.width):
  for y in range(image.height):
     pixel_value = pixels[x, y]
     # Do something with the pixel value




from PIL import Image
import matplotlib.pyplot as plt

# Open the image

image = Image.open("/content/drive/MyDrive/pro.imagem/word-written.jpg")

# Convert to grayscale
image_grayscale = image.convert("L")

# Display both images
display(image_grayscale)
