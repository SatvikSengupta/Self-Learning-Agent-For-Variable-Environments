import pybullet as p
import pybullet_data
import time
import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces

class VisionArmEnv(gym.Env):
    def __init__(self):
        super(VisionArmEnv, self).__init__()
        
        # Action Space: 3 joints (shoulder, elbow, wrist), values between -1.57 and +1.57 radians
        self.action_space = spaces.Box(low=-1.57, high=1.57, shape=(3,), dtype=np.float32)
        
        # Observation Space: 128x128 RGB image from the camera
        self.observation_space = spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8)
        
        # Initialize PyBullet
        self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Camera configuration
        self.width, self.height = 128, 128
        self.fov, self.near, self.far = 60, 0.02, 5.0
        self.proj_matrix = p.computeProjectionMatrixFOV(self.fov, self.width / self.height, self.near, self.far)
        
        self.armId = None
        self.targetId = None
        self.camera_link_index = -1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        
        # Load Arm and Target
        self.armId = p.loadURDF("arm.urdf", [0, 0, 0], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True)
        self.targetId = p.loadURDF("cube_small.urdf", [0.5, 0, 0.05])
        p.changeVisualShape(self.targetId, -1, rgbaColor=[0, 1, 1, 1])
        
        # Find camera link
        for i in range(p.getNumJoints(self.armId)):
            if p.getJointInfo(self.armId, i)[12].decode('utf-8') == 'camera_link':
                self.camera_link_index = i
                break
                
        return self._get_observation(), {}

    def step(self, action):
        # 1. Apply the neural network's chosen action to the virtual motors
        joint_indices = [0, 1, 2] # Shoulder, Elbow, Wrist
        p.setJointMotorControlArray(self.armId, jointIndices=joint_indices, 
                                    controlMode=p.POSITION_CONTROL, targetPositions=action)
        
        # 2. Step the physics forward
        p.stepSimulation()
        
        # 3. Capture RLHF Reward from your keyboard
        reward = self._get_human_reward()
        
        # 4. Get the new visual state
        obs = self._get_observation()
        
        # (Optional UI update so you can watch it)
        cv2.imshow("Agent POV", cv2.resize(cv2.cvtColor(obs, cv2.COLOR_RGB2BGR), (512, 512), interpolation=cv2.INTER_NEAREST))
        cv2.waitKey(1)
        time.sleep(1./240.) # Keep it real-time for human feedback
        
        # done condition (e.g., did it hit the floor or run out of time? We'll leave it False for infinite continuous learning for now)
        done = False 
        truncated = False
        
        return obs, reward, done, truncated, {}

    def _get_human_reward(self):
        keys = p.getKeyboardEvents()
        if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_WAS_TRIGGERED:
            print("Reward: +1")
            return 1.0
        elif p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_WAS_TRIGGERED:
            print("Reward: -1")
            return -1.0
        return 0.0 # Default reward is 0 (which handles your RIGHT_ARROW 'Okay' state naturally)

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
        
        # Return strict 128x128x3 uint8 array for PyTorch
        return np.reshape(images[2], (self.height, self.width, 4))[:, :, :3].astype(np.uint8)

# To test if the environment compiles:
if __name__ == '__main__':
    env = VisionArmEnv()
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    
    try:
        while True:
            # Random action sampling just to test the motors and rendering
            random_action = env.action_space.sample() 
            obs, reward, done, truncated, info = env.step(random_action)
    except KeyboardInterrupt:
        p.disconnect()
        cv2.destroyAllWindows()