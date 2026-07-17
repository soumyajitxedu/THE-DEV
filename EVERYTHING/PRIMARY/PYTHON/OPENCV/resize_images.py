#image transformation and manipulation


#resize image 


import cv2

image = cv2.imread('EVERYTHING\PRIMARY\PYTHON\OPENCV\DATASETS (IMGAGE)\Editorial Magazine Aesthetic _ High Fashion Poster Inspo.jpg')  # type: ignore


if image is None:
    print("error: image not found")
else:

    print("Original Image Shape:", image.shape)
    print(f"Dimensions of the image: {image.shape[0]} x {image.shape[1]}")
    resized_image =cv2.resize(image,(100,100))
    print("Resized Image Shape:", resized_image.shape)
    print(f"Dimensions of the resized image: {resized_image.shape[0]} x {resized_image.shape[1]}")
    cv2.imshow('Resized Image', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#advace resizing methods


