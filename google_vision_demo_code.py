from google.cloud import vision
import os

# ✅ Use raw string for the credentials path
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\Users\Obad\Downloads\chatable_google_vision_apikey.json'

client = vision.ImageAnnotatorClient()

# ✅ Use raw string for the image path
with open(r'C:\Users\Obad\Downloads\20250615_1644_Logo Integration_remix_01jxspcp1qfrtaqa46s26wzdf8.png', 'rb') as img_file:
    content = img_file.read()

image = vision.Image(content=content)

response = client.text_detection(image=image)
texts = response.text_annotations

for text in texts:
    print(f'Detected text: {text.description}')
