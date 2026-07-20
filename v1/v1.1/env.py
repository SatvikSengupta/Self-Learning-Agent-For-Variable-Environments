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
        
        self.action_space = spaces.Box(low=-1.57, high=1.57, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8)
        
        self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        self.width, self.height = 128, 128
        self.fov, self.near, self.far = 60, 0.02, 5.0
        self.proj_matrix = p.computeProjectionMatrixFOV(self.fov, self.width / self.height, self.near, self.far)
        
        self.armId = None
        self.targetId = None
        self.camera_link_index = -1

        # --- NEW: IPC Socket Listener ---
        self.udp_ip = "127.0.0.1"
        self.udp_port = 5005
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.udp_ip, self.udp_port))
        # Non-blocking is critical so the simulation doesn't freeze waiting for a click
        self.sock.setblocking(False) 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        
        self.armId = p.loadURDF("arm.urdf", [0, 0, 0], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True)
        self.targetId = p.loadURDF("cube_small.urdf", [0.5, 0, 0.05])
        p.changeVisualShape(self.targetId, -1, rgbaColor=[0, 1, 1, 1])
        
        self.camera_link_index = -1
        for i in range(p.getNumJoints(self.armId)):
            if p.getJointInfo(self.armId, i)[12].decode('utf-8') == 'camera_link':
                self.camera_link_index = i
                break
                
        return self._get_observation(), {}

    def step(self, action):
        joint_indices = [0, 1, 2] 
        p.setJointMotorControlArray(self.armId, jointIndices=joint_indices, 
                                    controlMode=p.POSITION_CONTROL, targetPositions=action)
        
        p.stepSimulation()
        
        reward = self._get_human_reward()
        obs = self._get_observation()
        
        display_img = cv2.resize(cv2.cvtColor(obs, cv2.COLOR_RGB2BGR), (512, 512), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Agent POV", display_img)
        cv2.waitKey(1)
        
        time.sleep(1./240.) 
        
        done = False 
        truncated = False
        
        return obs, reward, done, truncated, {}

    def _get_human_reward(self):
        # Default reward if no buttons were pressed
        reward = 0.0 
        try:
            # Drain the buffer and grab the most recent packet sent by the UI
            while True:
                data, addr = self.sock.recvfrom(1024)
                reward = float(data.decode('utf-8'))
        except BlockingIOError:
            # No data waiting in the socket, which is perfectly normal
            pass 
        
        if reward != 0.0:
            print(f"Received Feedback: {reward}")
            
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