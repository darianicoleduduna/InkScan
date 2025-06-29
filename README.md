# InkScan - Features

This project is a web application for recognizing and managing handwritten text from images. The main features are:

- **User authentication and registration**  
  Users can create accounts, log in, and log out.

- **Upload handwritten images**  
  Users can upload images (JPG, JPEG, PNG) containing handwritten text for analysis.

- **Handwriting recognition from images**  
  The application automatically extracts handwritten text from images using advanced OCR models.

- **Automatic text correction**  
  The recognized text is cleaned and automatically corrected to improve readability.

- **Text-to-speech**  
  Users can listen to the recognized text using speech synthesis functionality.

- **Download results**  
  The recognized text can be downloaded in DOCX or PDF format.

- **Scan history**  
  Each user has access to their history of uploaded images and recognized texts, with options to delete individual entries or clear the entire history.

- **Drawing interface**  
  Users can write by hand directly in the application, and the drawn text is recognized and processed similarly to uploaded images.

- **Save drawing with watermark**  
  Drawings can be downloaded with a personalized watermark.

These features provide a complete experience for digitizing and managing handwritten text.

---

## Technologies Used

- **Flask** – Web framework for Python.
- **OpenCV** – Image processing and manipulation.
- **EasyOCR** – For word detection and bounding box extraction.
- **TrOCR (Transformers OCR)** – Handwriting recognition using Microsoft's TrOCR model (`microsoft/trocr-large-handwritten`).
- **Pytesseract** – For orientation and script detection.
- **TextBlob** & **language_tool_python** – For automatic text correction.
- **wordninja** – For splitting concatenated words.
- **ReportLab** & **python-docx** – For exporting recognized text as PDF and DOCX.
- **Speech Synthesis (Web Speech API)** – For text-to-speech in the browser.
- **Pillow (PIL)** – Image manipulation and watermarking.

These technologies ensure accurate OCR, robust text correction, and a user-friendly experience.
