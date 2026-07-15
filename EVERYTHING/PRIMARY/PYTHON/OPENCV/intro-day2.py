import cv2


image = cv2.imread("PRIMARY\PYTHON\OPENCV\DATASETS (IMGAGE)\Editorial Magazine Aesthetic _ High Fashion Poster Inspo.jpe") # type: ignore

if image is not None:
    cv2.imshow("image is showing ", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("unable to find the image")

#saving 
success = cv2.imwrite("PRIMARY\PYTHON\OPENCV\DATASETS (IMGAGE)\Editorial Magazine Aesthetic _ High Fashion Poster Inspo.jpg", image) # type: ignore
if success:
    print("image saved sucessfully ")
else:
    print("error saving the image")

