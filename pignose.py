import cv2 as cv


#img = cv.imread('smile.jpg')
pig = cv.imread('pig.png')
#img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
pig_gray = cv.cvtColor(pig,cv.COLOR_BGR2GRAY)

nose_cascade = cv.CascadeClassifier('haarcascade_nose.xml')

cap = cv.VideoCapture(0)

while True:
    _, img = cap.read()
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    nose = nose_cascade.detectMultiScale(img_gray,1.3,5)

    for (x,y,w,h) in nose:
        x1=int(x-5)
        y1 = int(y-10)
        w1 = int(w+14)
        h1 = int(h+10)
        roi = img[y1:y1+h1,x1:x1+w1]
        width,height,channels = roi.shape
        resize_pig = cv.resize(pig,(int(height),int(width)),interpolation=cv.INTER_AREA)
        resize_pig_gray = cv.cvtColor(resize_pig,cv.COLOR_BGR2GRAY)
        ret, mask = cv.threshold(resize_pig_gray,210,255,cv.THRESH_BINARY_INV)
        mask_inverse = cv.bitwise_not(mask)
        img_bg = cv.bitwise_and(roi,roi,mask=mask_inverse)
        pig_fg = cv.bitwise_and(resize_pig,resize_pig,mask=mask)
        dst = cv.add(img_bg,pig_fg)
        img[y1:y1 + h1, x1:x1 + w1] = dst

    cv.imshow('frame',img)

    if cv.waitKey(1) == ord('q'):
        break






