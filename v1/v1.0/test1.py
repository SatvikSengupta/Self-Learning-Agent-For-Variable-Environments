import pybullet as p
import pybullet_data
import time
import numpy as np
import cv2

# Initialize the physics engine with a graphical interface
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# Load the ground and the robotic arm
planeId = p.loadURDF("plane.urdf")
startPos = [0, 0, 0]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])
armId = p.loadURDF("arm.urdf", startPos, startOrientation, useFixedBase=True)
# Spawn a small cube half a meter in front of the robot
targetId = p.loadURDF("cube_small.urdf", [0.5, 0, 0.05])

# Dynamically find the index of the camera link defined in the URDF
camera_link_index = -1
for i in range(p.getNumJoints(armId)):
    info = p.getJointInfo(armId, i)
    if info[12].decode('utf-8') == 'camera_link':
        camera_link_index = i
        break

# Configure the camera rendering parameters
width, height = 128, 128
fov, near, far = 60, 0.02, 5.0
projection_matrix = p.computeProjectionMatrixFOV(fov, width / height, near, far)

print("Simulation running... Select the PyBullet window and use the mouse to move the camera.")
print("Press Ctrl+C in the terminal to exit.")

try:
    while True:
        # Step the physics engine forward
        p.stepSimulation()

        if camera_link_index != -1:
            # Retrieve the precise 3D position and orientation of the claw's camera mount
            link_state = p.getLinkState(armId, camera_link_index)
            cam_pos = link_state[0]
            cam_orn = link_state[1]

            # Calculate the forward-facing and upward-facing vectors for the camera lens
            rot_matrix = np.array(p.getMatrixFromQuaternion(cam_orn)).reshape(3, 3)
            forward_vec = rot_matrix.dot([1, 0, 0])
            up_vec = rot_matrix.dot([0, 0, 1])
            target_pos = cam_pos + forward_vec

            # Render the environment from the claw's perspective
            view_matrix = p.computeViewMatrix(cam_pos, target_pos, up_vec)
            images = p.getCameraImage(width, height, view_matrix, projection_matrix, 
                                      renderer=p.ER_BULLET_HARDWARE_OPENGL)
            
            # Extract the raw RGB array, cast to 8-bit integers, and convert it for OpenCV
            rgb_img = np.reshape(images[2], (height, width, 4))[:, :, :3].astype(np.uint8)
            bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            
            # Upscale the OpenCV display purely so it's easier for you to see on screen
            display_img = cv2.resize(bgr_img, (512, 512), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Agent Point of View", display_img)
            cv2.waitKey(1)

        time.sleep(1./240.) 
except KeyboardInterrupt:
    p.disconnect()
    cv2.destroyAllWindows()