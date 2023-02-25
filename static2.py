import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
#imatplotlib inline

def looking_imag(img):
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_RGB)
    plt.show()

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(static_image_mode = True, model_complexity = 2, 
smooth_landmarks = False, enable_segmentation = True, min_detection_confidence = 0.5, min_tracking_confidence = 0.5)

img = cv2.imread('kicking2.jpg')
img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
result = pose.process(img_RGB)

mp_drawing.plot_landmarks(result.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
looking_imag(img)