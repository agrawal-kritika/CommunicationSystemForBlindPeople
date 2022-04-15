
import requests
import matplotlib.image as mpimg

from flask import Flask, flash, request, redirect, url_for, render_template
# import urllib.request
import os
from werkzeug.utils import secure_filename
import cv2
import tensorflow as tf
import numpy as np
import pyttsx3


app = Flask(__name__)
# engine = pyttsx3.init()
model = tf.keras.models.load_model("model_braille_v17.h5")
space_model = tf.keras.models.load_model("model_braille_space_ignore.h5")
UPLOAD_FOLDER = 'static/uploads/'
asciicodes = [' ',
          '0','1','2','3','4','5','6','7','8','9',
          'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q',
          'r','s','t','u','v','w','x','y','z','upper','cap']

# Braille symbols

brailles = ['⠀', '⠼⠚', '⠼⠁',
            '⠼⠃', '⠼⠉', '⠼⠙', '⠼⠑', '⠼⠋', '⠼⠛', '⠼⠓', '⠼⠊',  '⠁', '⠃',
            '⠉', '⠙', '⠑', '⠋', '⠛', '⠓', '⠊', '⠚', '⠅', '⠇', '⠍', '⠝', '⠕', '⠏', '⠟', '⠗', '⠎', '⠞', '⠥',
            '⠧', '⠺', '⠭', '⠽', '⠵', '⠠', '⠠']

number = {'a':'1','b':'2','c':'3','d':'4','e':'5','f':'6','g':'7','h':'8','i':'9','j':'0'}
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png','jpg'])
def text_to_braille(org_text):
    if(len(org_text)>0):
            l = org_text.split("\n")
            s =""
            for text in l:
                if(text[-1] == "\r"):
                    print(text)
                    text = text[:-1]
                if(text[-4:] == "<br>"):
                    print(text)
                    text = text[:-4]
                for i in text:
                    if(i.isupper()):
                        s += '⠠'
                        s = s + str(brailles[asciicodes.index(i.lower())])
                    else:
                        s = s + str(brailles[asciicodes.index(i)])
                s += "<br>"
    return(s)


def braille_to_english(org_text):
    upper_flag = 0
    numeric_flag = 0
    l = org_text.split("\n")
    s =""
    for text in l:
        if(text[-1] == "\r"):
            print(text)
            text = text[:-1]
        if(text[-4:] == "<br>"):
            print(text)
            text = text[:-4]
        for i in text:
            if(i == "⠠"):
                upper_flag = 1
            elif(i == "⠀"):
                s += " "
            elif(i == "⠼"):
                numeric_flag = 1
            elif(upper_flag == 1):
                s += str(asciicodes[brailles.index(i)]).upper()
                upper_flag = 0
            elif(numeric_flag == 1):
                s += (number[str(asciicodes[brailles.index(i)])])
                numeric_flag = 0
            else:
                s += str(asciicodes[brailles.index(i)])
        s += "<br>"
    return(s)





def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index/', methods=['POST','GET'])
def index():
    return render_template('index.html')

@app.route('/ttob/', methods=['POST','GET'])
def ttob():
    return render_template('ttob.html')

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # print('upload_image filename: ' + filename)
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        fontScale = 0.5
        color = (255, 0, 0)
        thickness = 2
        img      = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        org_img = img.copy()
        text_img = img.copy()
        gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur     = cv2.GaussianBlur(gray,(3,3),0)
        thres    = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,5,4)
        blur2    = cv2.medianBlur(thres,3)
        ret2,th2 = cv2.threshold(blur2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        blur3    = cv2.GaussianBlur(th2,(3,3),0)
        ret3,th3 = cv2.threshold(blur3,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        inter_img = th3.copy()
        empty_rows = []
        good_rows = []
        empty_cols = []
        check_count = []
        dim = img.shape
        for i in range(th3.shape[0]):
            if np.mean(255-th3[i,:]) ==0:
                empty_rows.append(i)
        check_len = 0
        d = np.ediff1d(empty_rows, to_begin=1)
        l = []
        flag = 0
        good_rows =[]
        for i in range(len(d)):
            if(d[i] != 1 and d[i]!=0 and len(good_rows)%2 == 0):
                good_rows.append(empty_rows[i-1])
                flag = 1
            if(d[i] != 1 and d[i]!=0 and flag == 0):
                good_rows.append(empty_rows[i])
            if(d[i] != 1 and d[i]!=0 and flag < 4):
                flag += 1
            if(d[i] != 1 and d[i]!=0 and flag == 3):
                flag = 0

        for i in good_rows:
            cv2.line(img, (0,i), (dim[1],i), (0,255,0), 1)

        #   ----------------------------------------------------------------------------------------------------------------------
        upper_flag = 0
        numeric_flag = 0
        imgs = []
        ans = []
        for wi in range(1,len(good_rows)):
            check_img = th3[good_rows[wi-1]:good_rows[wi],:]
            imgs.append(check_img)
            empty_cols = []
            good_cols = []
            for i in range(check_img.shape[1]):
                if np.mean(255-check_img[:,i]) ==0 :
                    empty_cols.append(i)
            d1 = np.ediff1d(empty_cols, to_begin=1)
            for i in range(len(d1)):
                if(d1[i] != 1 and d1[i]!=0 and len(good_cols)%2 == 0):
                    good_cols.append(empty_cols[i-1])
                    flag = 1
                if(d1[i] != 1 and d1[i]!=0 and flag == 0):
                    good_cols.append(empty_cols[i])
                if(d1[i] != 1 and d1[i]!=0 and flag < 4):
                    flag += 1
                if(d1[i] != 1 and d1[i]!=0 and flag == 2):
                    flag = 0
            for i in good_cols:
                cv2.line(img, (i,good_rows[wi-1]), (i,good_rows[wi]), (0,255,0), 1)
            num = 0
            for hi in range(1,len(good_cols)):
                img1 = org_img[good_rows[wi-1]:good_rows[wi],good_cols[hi-1]:good_cols[hi]]
                resized = cv2.resize(img1, (28,28), interpolation = cv2.INTER_AREA)
                resized  = resized/255
                pred = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'ignore', 'j', 'k', 'l', 'm', 'n', 'numeric', 'o', 'p', 'q', 'r', 's', 'space', 't', 'u', 'upper', 'v', 'w', 'x', 'y', 'z']
                ans1 = pred[np.argmax(model.predict(np.expand_dims(resized,axis=0)))]
                print(ans1)
                if(ans1 != "ignore"):
                    if(ans1 == "upper"):
                        upper_flag = 1
                    elif(ans1 == "space"):
                        ans.append(" ")
                    elif(ans1 == "numeric"):
                        numeric_flag = 1
                    elif(upper_flag == 1):
                        ans.append(ans1.upper())
                        upper_flag = 0
                    elif(numeric_flag == 1):
                        ans.append(number[ans1])
                        numeric_flag = 0
                    else:
                        ans.append(ans1)
                else:
                    if(np.argmax(space_model.predict(np.expand_dims(resized,axis=0))) == 1):
                        ans.append(" ")
            ans.append("\n")
        text = ""
        for i in ans:
            text += i

#         engine.say(text)
#         engine.runAndWait()
#         engine.endLoop()
#         engine.stop()
        flash(text)
        return render_template('index.html')
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect("hi")


@app.route('/display/<filename>')
def display_image(filename):
    print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/t_to_b', methods=['POST'])
def t_to_b():
    features = [x for x in request.form.values()]
    org_text = features[0]
    if(len(org_text)>0):
        s = text_to_braille(org_text)
#         engine.say(org_text)
#         engine.runAndWait()
        flash(s)
    else:
        flash("no input")
    return render_template('ttob.html')

@app.route('/b_to_t', methods=['POST'])
def b_to_t():
    features = [x for x in request.form.values()]
    org_text = features[0]
    if(len(org_text)>0):
        s = braille_to_english(org_text)
#         engine.say(org_text)
#         engine.runAndWait()
        flash(s)
    else:
        flash("no input")
    return render_template('ttob.html')

if __name__ == "__main__":
    app.run()
