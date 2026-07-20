import pybullet as p
import pybullet_data
import time
import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces
import socket

class VisionArmEnv(gym.Env):
    def __init__(self):
        super(VisionArmEnv, self).__init__()
        
        # Action Space: 4 joints (shoulder, elbow, wrist, neck)
        self.action_space = spaces.Box(low=-1.57, high=1.57, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8)
        
        self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        self.width, self.height = 128, 128
        self.fov, self.near, self.far = 60, 0.02, 5.0
        self.proj_matrix = p.computeProjectionMatrixFOV(self.fov, self.width / self.height, self.near, self.far)
        
        self.armId = None
        self.targetId = None
        self.camera_link_index = -1

        # IPC Socket Listener (Kept warm for Phase 2)
        self.udp_ip = "127.0.0.1"
        self.udp_port = 5005
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.udp_ip, self.udp_port))
        self.sock.setblocking(False) 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        
        self.armId = p.loadURDF("arm.urdf", [0, 0, 0], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True)
        # Pushed the cube slightly further away so it really has to reach for it
        self.targetId = p.loadURDF("cube_small.urdf", [0.7, 0, 0.05])
        p.changeVisualShape(self.targetId, -1, rgbaColor=[0, 1, 1, 1])
        
        self.camera_link_index = -1
        for i in range(p.getNumJoints(self.armId)):
            if p.getJointInfo(self.armId, i)[12].decode('utf-8') == 'camera_link':
                self.camera_link_index = i
                break
                
        return self._get_observation(), {}

    def step(self, action):
        joint_indices = [0, 1, 2] 
        
        # Now driving 4 motors: Shoulder, Elbow, Wrist, Neck
        joint_indices = [0, 1, 2, 3] 
        
        p.setJointMotorControlArray(self.armId, jointIndices=joint_indices, 
                                    controlMode=p.POSITION_CONTROL, targetPositions=action)
        
        # Step the physics engine 10 times holding the SAME action (Frame Skipping)
        for _ in range(10):
            p.stepSimulation()
            time.sleep(1./2400.)
        
        # Now calculate the reward based on where the arm ended up
        reward = self._get_auto_reward()

        #RHLF ONLY!!
        # reward = self._get_human_reward() 
         
        obs = self._get_observation()
        
        # Only update the UI after the physical movement completes
        display_img = cv2.resize(cv2.cvtColor(obs, cv2.COLOR_RGB2BGR), (512, 512), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Agent POV", display_img)
        cv2.waitKey(1)
        
        done = False 
        truncated = False
        
        return obs, reward, done, truncated, {}

    def _get_auto_reward(self):
        # 1. Get the exact 3D coordinates of the camera anchor
        cam_pos = np.array(p.getLinkState(self.armId, self.camera_link_index)[0])
        
        # 2. Get the exact 3D coordinates of the cyan cube
        target_pos = np.array(p.getBasePositionAndOrientation(self.targetId)[0])
        
        # 3. Calculate Euclidean distance
        distance = np.linalg.norm(cam_pos - target_pos)
        
        # 4. Dense penalty: the further away, the more negative the score
        reward = -distance 
        
        # 5. Success bonus: If it gets within 15cm of the center of the cube, massive reward
        if distance < 0.15:
            reward += 10.0
            
        return reward

    def _get_human_reward(self):
        reward = 0.0 
        try:
            while True:
                data, addr = self.sock.recvfrom(1024)
                reward = float(data.decode('utf-8'))
        except BlockingIOError:
            pass 
        return reward

    def _get_observation(self):
        if self.camera_link_index == -1:
            return np.zeros((128, 128, 3), dtype=np.uint8)
            
        link_state = p.getLinkState(self.armId, self.camera_link_index)
        cam_pos, cam_orn = link_state[0], link_state[1]
        
        rot_matrix = np.array(p.getMatrixFromQuaternion(cam_orn)).reshape(3, 3)
        target_pos = cam_pos + rot_matrix.dot([1, 0, 0])
        up_vec = rot_matrix.dot([0, 0, 1])
        
        view_matrix = p.computeViewMatrix(cam_pos, target_pos, up_vec)
        images = p.getCameraImage(self.width, self.height, view_matrix, self.proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        
        return np.reshape(images[2], (self.height, self.width, 4))[:, :, :3].astype(np.uint8)