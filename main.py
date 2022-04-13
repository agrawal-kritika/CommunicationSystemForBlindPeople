# Braille
import numpy as np
import sys
import cv2
import tensorflow as tf
def image_file_text(img_path):
    model = tf.keras.models.load_model("model121.h5")

    # font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # org
    org = (50, 50)

    # fontScale
    fontScale = 0.5

    # Blue color in BGR
    color = (255, 0, 0)

    # Line thickness of 2 px
    thickness = 2




    img      = cv2.imread(img_path)
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
    #remove front and back

    for i in range(len(d)):
        if d[i] == 1:
            d[i] = 0
        else:
            start_row = empty_rows[i-1]
            good_rows.append(empty_rows[i-1])
            break
    for i in reversed(range(len(d))):
        if d[i] == 1:
            d[i] = 0
        else:
            if(i < len(d) and i >0):
                good_rows.append(empty_rows[i])
                break

    for i in range(len(d)):
        if d[i] == 1:
            check_len += 1
        else:
            if(check_len != 1):
                check_count.append(check_len)
            check_len = 1

    bias = int(np.mean(check_count[0:4])*0.9)
    check_len = 0
    for i in range(len(d)):
        if d[i] == 1:
            check_len += 1
        else:
            if(check_len>bias):
                good_rows.append(empty_rows[i-int(check_len/2)])
            check_len = 1
    if(check_len>bias):
        good_rows.append(empty_rows[len(d)-int(check_len/2)])
    for i in good_rows:
        f = cv2.line(img, (0,i), (dim[1],i), (0,255,0), 1)

    good_rows.sort()
    f1 = cv2.line(img, (0,1500), (50,1500), (255,0,0), 1)
    #   ----------------------------------------------------------------------------------------------------------------------

    imgs = []
    ans = []
    upper_flag = 0
    numeric_flag = 0
    numeric = {'a':1, 'b':2, 'c':3, 'd':4, 'e':5, 'f':6, 'g':7, 'h':8, 'i':9, 'j':10, 'k':11, 'l':12, 'n':13, 'n':14, 'o':15, 'p':16, 'q':17,
                    'r':18, 's':19, 't':20, 'u':21, 'v':22, 'w':23, 'x':24, 'y':25, 'z':26,}
    

    for wi in range(1,len(good_rows)):
        check_img = th3[good_rows[wi-1]:good_rows[wi],:]
        imgs.append(check_img)

        empty_cols = []

        for i in range(check_img.shape[1]):
            if np.mean(255-check_img[:,i]) ==0 :
                empty_cols.append(i)
        d1 = np.ediff1d(empty_cols, to_begin=1)

        good_cols=[]
        #remove front and back
        for i in range(len(d1)):
            if d1[i] == 1:
                d1[i] = 0
            else:
                start_col = d1[i-1]
                if(i>1):
                    good_cols.append(empty_cols[i-1])
                else:
                    good_cols.append(0)
                break

        for i in reversed(range(len(d1))):
            if d1[i] == 1:
                d1[i] = 0
            else:
                if(i < len(d1) and i >0):
                    good_cols.append(empty_cols[i])
                    break

        bias = int(np.mean(check_count[0:4])*0.9)



        check_len = 0

        for i in range(len(d1)):
            if d1[i] == 1:
                check_len += 1
            else:
                if(check_len>bias):
                    good_cols.append(empty_cols[i-int(check_len)])
                check_len = 1

        if(check_len > bias):
            good_cols.append(empty_rows[len(d)-int(check_len/2)])


        check_c = []
        good_cols = list(set(good_cols))
        num = 1
        good_cols.sort()
        for hi in range(1,len(good_cols)):
            img1 = img[good_rows[wi-1]:good_rows[wi],good_cols[hi-1]:good_cols[hi]]
            resized = cv2.resize(img1, (28,28), interpolation = cv2.INTER_AREA)
            filename = "/Users/saisoorya/PycharmProjects/braille/check/img_4_" +  str(wi) + "_"+ str(num) +'.jpg'
            num += 1
            cv2.imwrite(filename, resized)
            pred = [" ", '-', '@', "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "n", "n", "o", "p", "q",
                    "r", "s", "t", "u", "v", "w", "x", "y", "z", ]

            ans1 = pred[np.argmax(model.predict(np.expand_dims(resized, axis=0)))]

            if (ans1 == '-'):
                upper_flag = 1
            elif (ans1 == '@'):
                numeric_flag = 1
            elif (upper_flag == 1):
                ans = ans.upper()
                upper_flag = 0
            elif (numeric_flag == 1):
                ans1 = number[ans1]
                numeric_flag = 0

            ans.append(ans1)
            image12 = cv2.putText(text_img, ans1,(good_cols[hi],good_rows[wi-1]) , font,
                                        fontScale, color, thickness, cv2.LINE_AA)


        for i in good_cols:
            f1 = cv2.line(img, (i,good_rows[wi-1]), (i,good_rows[wi]), (0,255,0), 1)
    while 1:
        cv2.imshow('ORGINAL IMAGE',org_img)
        cv2.imshow('INTERMEDIATE IMAGE',th3)
        cv2.imshow('IMAGE WITH GRID',img)
        cv2.imshow('OUTPUT IMAGE',text_img)
        cv2.waitKey(0)

if __name__ == "__main__":

    image_file_text(sys.argv[1])