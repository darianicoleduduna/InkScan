#importam bibliotecile necesare 
from datetime import datetime
from flask import Flask, jsonify, render_template, request, redirect, url_for, session, flash, send_file
import json
import os
import re
import cv2
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import easyocr
from textblob import TextBlob
import wordninja
import base64
from io import BytesIO
import pytesseract
import imutils
from docx import Document
import language_tool_python
from nltk.corpus import words as nltk_words
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
from textwrap import wrap
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

#facem setup pentru aplicatia flask, si definim fisierul unde vom salva date despre utilizatori
app = Flask(__name__)
app.secret_key = "secretkey"
USER_DB = "users.json"


#initiaizam procesorul si modelul TrOCR 
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')

tool = language_tool_python.LanguageTool('en-US')
VALID_WORDS = set(nltk_words.words())


#metoda pentru deschiderea fisierului json
def load_users():
    if not os.path.exists(USER_DB):
        return {}
    with open(USER_DB, "r") as f:
        return json.load(f)

#metoda de salvare in fisierul json 
def save_users(users):
    with open(USER_DB, "w") as f:
        json.dump(users, f)



def clean_ocr_text(text): 
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\.+', '.', text)
    words = text.split()
    cleaned = []

    for i, word in enumerate(words):
        #eliminam punctele din interiorul cuvintelor afectate de zgomot
        if '.' in word and not word.endswith('.'):
            word = word.replace('.', '')

        #pastram punctul doar daca e sf de prop
        if word.endswith('.'):
            next_word = words[i+1] if i+1 < len(words) else ''
            if next_word and next_word[0].isupper():
                cleaned.append(word) 
            elif i == len(words) - 1:
                cleaned.append(word)
            else:
                cleaned.append(word.rstrip('.'))
        elif word.startswith('.') and len(word) > 1:
            cleaned.append(word[1:])
        else:
            cleaned.append(word)

    return ' '.join(cleaned).strip()


#cautam daca au fost interpretate 2 cuvinte ca fiind unul singur 
def split_glued_words(text):
    print(text)
    words = text.split()
    new_words = []
    for word in words:
        cleaned = ''.join(filter(str.isalpha, word)).lower()
        if cleaned in VALID_WORDS:
            new_words.append(word)
        else:
            split = wordninja.split(word)
            if all(w.lower() in VALID_WORDS for w in split) and len(split) > 1:
                new_words.extend(split)
            else:
                new_words.append(word)
    return " ".join(new_words)

#folosim metode de corectare automata
def autocorrect_text(text):
    blob=TextBlob(text)
    inter=blob.correct()
    result=tool.correct(str(text))
    return result

#grupam cuvintele extrase in functie de pozitii pe linii
def group_lines(results, y_thresh=25):
    results.sort(key=lambda x: (x[0][0][1] + x[0][2][1]) / 2)
    lines = []
    for r in results:
        y_center = (r[0][0][1] + r[0][2][1]) / 2
        placed = False
        for line in lines:
            line_y_center = sum([(item[0][0][1] + item[0][2][1]) / 2 for item in line]) / len(line)
            if abs(y_center - line_y_center) < y_thresh:
                line.append(r)
                placed = True
                break
        if not placed:
            lines.append([r])
    for line in lines:
        line.sort(key=lambda x: x[0][0][0])
    return lines


#preprocesam imaginea pentru a o netezi si a elimina din diferentele mari de culori 
def preprocess_image_grayscale(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    cv2.imwrite(image_path, rgb)

    return image_path

#functie pentru a arata utilizatorului ce cuvinte au fost detectate, incadrandu-le 
def show_detected_words(image_path, results, output_path="static/annotated.png"):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for (bbox, text, conf) in results:
        pts = np.array(bbox, dtype=np.int32)
        cv2.polylines(image, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
        
    annotated_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, annotated_bgr)
    return output_path

#verificam daca s-u suprapus 2 boxuri si a fost detectat acelasi cuvant de 2 ori consecutiv
def remove_duplicate_words(line_words):
    result = []
    for word in line_words:
        if not result or word != result[-1]:
            result.append(word)
    return result

#corectam orientarea imaginii, prin rotatii de 90/180/270 de grade
def image_orientation_corrector(image_path, output_path="static/rotated_image.jpg", angle_threshold=3):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    try:
        results = pytesseract.image_to_osd(rgb, config='--psm 0 -c min_characters_to_try=6', output_type=pytesseract.Output.DICT)
        angle = int(results.get("rotate", 0))
    except pytesseract.TesseractError as e:
        return image_path
    
    #ignoram unghiuri mici iar cazul cu 180 de grade il tratam ulterior pt a ne asigura ca nu a determinat eronat uunghiul
    if abs(angle) <= angle_threshold or angle == 180:
        corrected = image
    else:
        corrected = imutils.rotate_bound(image, angle=-angle)
    
    # Comparăm scorurile OCR între imaginea corectată și varianta rotită 180
    reader = easyocr.Reader(['en'], gpu=False)
    results_corrected = reader.readtext(corrected)
    results_rotated_180 = reader.readtext(cv2.rotate(corrected, cv2.ROTATE_180))

    #verif scorurile de confidence easyocr si facem media pentru a le compara mai tarziu
    conf_corrected = np.mean([r[2] for r in results_corrected]) if results_corrected else 0
    conf_rotated = np.mean([r[2] for r in results_rotated_180]) if results_rotated_180 else 0

    #in functie de acele scoruri decidem ce varianta pastram 
    final_image = corrected if conf_corrected >= conf_rotated else cv2.rotate(corrected, cv2.ROTATE_180)
    cv2.imwrite(output_path, final_image)

    return output_path

#vrem sa aliniam textul perfect pe orizontala asa ca vrem sa corectam inclinarea
def deskew_image(image_path, save_path="static/deskewed.png"):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edged, 1, np.pi / 180, 200)
    if lines is None:
        return image_path 

    angles = []
    for rho, theta in lines[:, 0]:
        angle = (theta * 180 / np.pi) - 90
        if -45 < angle < 45:
            angles.append(angle)

    if not angles:
        return image_path

    median_angle = np.median(angles)

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    cv2.imwrite(save_path, rotated)
    return save_path


#metoda principala care extrage textul prin analiza imaginior
def extract_words_with_trocr(image_path, processor, model):
    reader = easyocr.Reader(['en'], gpu=False)
    #apelam metodele pe care le-am definit anterior pentru a preprocesa imaginea si a ne asigura ca este corect alinita si orientata
    image_path = preprocess_image_grayscale(image_path)
    image_path = image_orientation_corrector(image_path)
    image_path=deskew_image(image_path)
    #folosim EasyOcr pt a detecta cuvintele 
    results = reader.readtext(image_path)
    if not results:
        return "No words detected."
    annotated_path = show_detected_words(image_path, results)
    image = cv2.imread(image_path)
    #impartim aceste cuvinte pe linii
    grouped = group_lines(results)
    recognized_lines = []

    for line in grouped:
        line_words = []
        last_x_max = None

        # Calculează distanțele între cuvintele detectate în linie
        x_gaps = []
        for i in range(1, len(line)):
            prev_bbox = line[i - 1][0]
            curr_bbox = line[i][0]
            prev_x_max = max(prev_bbox[2][0], prev_bbox[3][0])
            curr_x_min = min(curr_bbox[0][0], curr_bbox[1][0])
            x_gaps.append(curr_x_min - prev_x_max)

        # Threshold dinamic: medie - deviație standard
        avg_gap = np.mean(x_gaps) if x_gaps else 12
        std_gap = np.std(x_gaps) if x_gaps else 3
        dynamic_threshold = avg_gap - std_gap  # ajustabil

        #dorim sa parcurgem cuvintele in ordine 
        for (bbox, _, _) in line:
            (tl, tr, br, bl) = bbox
            x_min = int(min(tl[0], bl[0]))
            y_min = int(min(tl[1], tr[1]))
            x_max = int(max(tr[0], br[0]))
            y_max = int(max(bl[1], br[1]))

            if x_min >= x_max or y_min >= y_max:
                continue

            cropped = image[y_min:y_max, x_min:x_max]

            if cropped.size == 0:
                continue

            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (gray.shape[1]*2, gray.shape[0]*2), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
            pil_image = Image.fromarray(rgb)

            try:
                pixel_values = processor(pil_image, return_tensors="pt").pixel_values
                output_ids = model.generate(pixel_values)
                text = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

                if last_x_max is not None:
                    gap = x_min - last_x_max
                    prev_word = line_words[-1] if line_words else ''
                    if gap < dynamic_threshold and text != prev_word and not (prev_word.isdigit() and text.isalpha()) and not (prev_word.isalpha() and text.isdigit()):
                        line_words[-1] += text
                    else:
                        line_words.append(text)
                else:
                    line_words.append(text)

                last_x_max = x_max
            except Exception:
                line_words.append("")
        #in timp ce recompunem textul vrem sa eliminam cuvintele duplicate consecutive
        line_words = remove_duplicate_words(line_words)
        recognized_lines.append(line_words)
        

    lines_text = [" ".join(line) for line in recognized_lines]
    full_text = "\n".join(lines_text)
    #curatam textul
    full_text = clean_ocr_text(full_text)
    #despartim cuvintele lipite eronat
    full_text = split_glued_words(full_text)
    #trecem textul prin metode de corectare automata
    corrected_text = autocorrect_text(full_text)
    
    
    #returnam mesajul final
    return corrected_text

#metoda de salvare in istoric
def add_to_history(username, image_path, recognized_text):
    users = load_users()
    if username not in users:
        return

    entry = {
        "image": image_path,
        "text": recognized_text,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    if "history" not in users[username]:
        users[username]["history"] = []

    users[username]["history"].append(entry)
    save_users(users)




#mapare cai din aplicatie
# autentificare
@app.route("/", methods=["GET", "POST"])
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        users = load_users()
        username = request.form["username"]
        password = request.form["password"]
        if username in users and users[username]["password"] == password:
            session["user"] = username
            return redirect(url_for("welcome"))
        flash("Invalid username or password.")
    return render_template("login.html")

#inregistrare
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        users = load_users()
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]
        confirm_password = request.form["confirm_password"]
        if username in users:
            flash("Username already exists.")
        elif password != confirm_password:
            flash("Passwords do not match.")
        else:
            users[username] = {"password": password, "email": email}
            save_users(users)
            flash("Account created! You can now log in.")
            return redirect(url_for("login"))
    return render_template("register.html")

#pagina intermediara dupa autentificare
@app.route("/welcome")
def welcome():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("welcome.html", user=session["user"])


#iesire din cont
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

#pagina principala
@app.route("/main", methods=["GET", "POST"])
def main():
    extracted_text = None
    image_url = None
    if request.method == "POST":
        if "image" in request.files:
            file = request.files["image"]
            if file.filename:
                filepath = os.path.join("static/uploads", file.filename)
                file.save(filepath)
                extracted_text = extract_words_with_trocr(filepath, processor, model)
                add_to_history(session["user"], filepath, extracted_text)
                image_url = filepath
    return render_template("main.html", text=extracted_text, image_url=image_url)

#interfata pentru scris de mana 
@app.route("/draw", methods=["GET", "POST"])
def draw():
    return render_template("draw.html")

#recunoasterea textului scris in interfata
@app.route("/recognize-drawing", methods=["POST"])
def recognize_drawing():
    data = request.get_json()
    image_data = data["image"].split(",")[1]  
    image_bytes = base64.b64decode(image_data)

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    path = "static/temp_drawing.png"
    image.save(path)

    recognized_text = extract_words_with_trocr(path, processor, model)
    return {"text": recognized_text}

#pagina cu istoricul
@app.route("/history")
def history():
    if "user" not in session:
        return redirect(url_for("login"))

    users = load_users()
    history_data = users.get(session["user"], {}).get("history", [])
    return render_template("history.html", history=history_data)


@app.route("/delete_entry", methods=["POST"])
def delete_entry():
    if "user" not in session:
        return redirect(url_for("login"))

    index = int(request.form["index"])
    users = load_users()
    history = users[session["user"]].get("history", [])

    if 0 <= index < len(history):
        del history[index]
        save_users(users)

    return redirect(url_for("history"))


@app.route("/clear_history", methods=["POST"])
def clear_history():  
    if "user" not in session:
        return redirect(url_for("login"))

    users = load_users()
    users[session["user"]]["history"] = []
    save_users(users)

    return redirect(url_for("history"))


@app.route("/download_docx", methods=["POST"])
def download_docx():
    text = request.form.get("text", "")
    doc = Document()

    wrapped_lines = wrap(text, width=90)
    for line in wrapped_lines:
        doc.add_paragraph(line)

    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="output.docx")


@app.route("/download_pdf", methods=["POST"])
def download_pdf():
    text = request.form.get("text", "")
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    y = height - 50
    wrapped_lines = wrap(text, width=90)
    for line in wrapped_lines:
        p.drawString(50, y, line)
        y -= 18
        if y < 50:
            p.showPage()
            y = height - 50

    p.save()
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="output.pdf")


@app.route('/save-drawing-watermark', methods=['POST'])
def save_drawing_with_watermark():
    data = request.get_json()
    image_data = data['image'].split(',')[1]
    image_bytes = base64.b64decode(image_data)
    base_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")

    watermark_img = Image.open("static/images/InkScanlogo.png").convert("RGBA")

    scale_factor = 0.08
    target_width = int(base_image.width * scale_factor)
    aspect_ratio = target_width / watermark_img.width
    target_height = int(watermark_img.height * aspect_ratio)
    watermark_img = watermark_img.resize((target_width, target_height), Image.LANCZOS)

    
    position_img = (10, base_image.height - watermark_img.height )
    position_text = (position_img[0] + watermark_img.width + 5, position_img[1] + watermark_img.height // 2 - 20 )

    txt_layer = Image.new("RGBA", base_image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(txt_layer)
    try:
        font = ImageFont.truetype("static/fontwatermark.ttf", 40)
    except:
        font = ImageFont.load_default()

    draw.text(position_text, "InkScan", font=font, fill=(227, 115, 131, 160))

    combined = Image.new("RGBA", base_image.size)
    combined.paste(base_image, (0, 0))
    combined.paste(watermark_img, position_img, watermark_img)
    combined = Image.alpha_composite(combined, txt_layer)

    img_io = io.BytesIO()
    combined.convert("RGB").save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png', as_attachment=True, download_name='inkscan_watermarked.png')

if __name__ == "__main__":
    app.run(debug=True)
