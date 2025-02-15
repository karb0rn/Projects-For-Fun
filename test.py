import cv2 as cv

img = cv.imread(r"C:\Users\Karthick\Downloads\messi.jpg")

if img is None:
    print("Error: Image not loaded. Check file path and format.")
else:
    cv.imshow('window', img)  
    cv.imwrite('messi.png',img)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    cv.imwrite('messi_gray.png',gray)
    cv.imshow('messi_gray.png', gray)
    cv.waitKey(0)
    cv.destroyAllWindows() 
