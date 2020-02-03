import cv2
import os

image_folder = 'output'
video_name = 'video.avi'
image_list = os.listdir(image_folder)
image_list.sort()
print(image_list)

images = [img for img in image_list if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 15, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()