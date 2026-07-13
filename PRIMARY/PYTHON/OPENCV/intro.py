import cv2

img = cv2.imread(r"C:\Users\maina\OneDrive\Pictures\Screenshots\Screenshot 2026-06-16 205729.png", cv2.IMREAD_COLOR)

if img is None:
    print("error loading image")
    exit()
cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.imshow("image",img)
cv2.waitKey(0)

## next ?

print(f"image size : {img.shape}")
print(f"image height  : {img.shape[0]}")
print(f"image width: {img.shape[1]}")

print("size of the image is :",img.size,"pixels")
print("now filtering the image")

avg = cv2.blur(img,(5,5))
cv2.imshow("average filter",avg)
cv2.namedWindow("average filter", cv2.WINDOW_NORMAL)