import cv2

detect=cv2.CascadeClassifier(".\haarcascade-frontalface-default.xml")
imp_img=cv2.VideoCapture("images\\elon.jpg")

#res is True/False whether image got read succesfully or not
# img will have image dimensions
res,img=imp_img.read()

#Grayscale conversion for haarcascade_frontalface_default

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces=detect.detectMultiScale(gray,1.3,5)

#Now faces will return x,y width and height
# So drawing a square

for x,y,w,h in faces:
    #now drawing rectangle
    #5 parameters image,pt1,pt2,color,thickness
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)


cv2.imshow("Elon Image",img)
cv2.waitKey(0)
imp_img.release()
cv2.destroyAllWindows()
