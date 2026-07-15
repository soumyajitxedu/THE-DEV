import cv2
# type: ignore #colour changing 

'''
img = cv2.imread("PRIMARY nny Aspirant (Blade of Kibou) GFX Design - MLBB.jpe") # type: ignore
if img is None:
    print("error")
    
   
else:
    print("processing")
    
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2RGBA)
    cv2.imshow("image ", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

'''
#mini project


data = (input("PASTE THE LOCATION OF THE IMAGE : "))

img = cv2.imread(data)

if img is None:
    print("error")
else:
    print("wait image loading")
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imshow("image", gray)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()












#1:6:30