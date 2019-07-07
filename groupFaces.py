import cv2,glob

# glob.glob returns the list of files with their full path unlike os.listdir() and is more
# powerful because it uses wildcards



gimg=glob.glob(".\GroupImages\*.jpg")
detect=cv2.CascadeClassifier(".\haarcascade-frontalface-default.xml")

for timg in gimg:
    img=cv2.imread(timg)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=detect.detectMultiScale(gray,1.25,5)

    for x,y,w,h in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
    cv2.imshow("Detect Multiple Images",img)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
