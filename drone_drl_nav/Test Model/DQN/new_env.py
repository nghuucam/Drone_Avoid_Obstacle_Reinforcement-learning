from gymnasium import spaces
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType
from gym_pybullet_drones.My_Coding import config

import pybullet_data
import numpy as np
import pybullet as p
import random
import math

MAX_STEPS = 10000
class DroneEnv(BaseAviary):
    def __init__(self,
                 gui,
                 initial_xyzs = None, 
                 initial_rpys = np.array([[0.0, 0.0, -np.pi/2]]), 
                 drone_model = DroneModel.CF2X, 
                 num_drones = 1, 
                 neighbourhood_radius = np.inf, 
                 physics = Physics.PYB, 
                 pyb_freq = 240, 
                 ctrl_freq = 240,  
                 record=False, 
                 obstacles=True, 
                 user_debug_gui=True, 
                 vision_attributes=True, 
                 output_folder='results'):
        self.obstacles_id = []
        self.obstacles_position = [] 
        self.collision = False
        self.win = False
        self.over_step = False
        self.over_map = False
        self.step_count = 0
        self.reward = 0
        self.start, self.goal = self.add_start_goal()
        #self.IMG_RES = np.array([64, 64])
        super().__init__(drone_model = drone_model, 
                         num_drones = num_drones, 
                         neighbourhood_radius = neighbourhood_radius, 
                         initial_xyzs = self.start, 
                         initial_rpys = initial_rpys, 
                         physics = physics, 
                         pyb_freq = pyb_freq, 
                         ctrl_freq = ctrl_freq, 
                         gui = gui, 
                         record = record, 
                         obstacles = obstacles, 
                         user_debug_gui = user_debug_gui, 
                         vision_attributes = vision_attributes, 
                         output_folder = output_folder)
    
    def add_start_goal(self):
        start_index = random.choice(config.start_space)
        goal_index = random.choice(config.goal_space)
        start_point = np.array([[start_index[0], start_index[1], 0]])
        goal_point = np.array([[goal_index[0], goal_index[1], 0]])
        return start_point, goal_point
    
    def _addObstacles(self):
        ##################### TẠO VẬT CẢN #########################################################################        
        cylinder_collision_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=1.0, height=3.0)
        cylinder_vis_id = p.createVisualShape(p.GEOM_CYLINDER, radius=1.0, length=3.0, rgbaColor=[0, 0.7, 0.3, 1])
       
        self.obstacles_id = []
        for pos in self.obstacles_position:
            body_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=cylinder_collision_id,
                baseVisualShapeIndex=cylinder_vis_id,
                basePosition=[pos[0], pos[1], 1.5]
            )
            self.obstacles_id.append(body_id)
        ##############################################################################################################
        
        ######################## TẠO RA VẬT CẢN Ở ĐÍCH ###############################################################
        goal_x, goal_y, _ = self.goal[0]

        # Tạo cột mỏng cao làm đích
        pole_collision_id = p.createCollisionShape(
            p.GEOM_CYLINDER, 
            radius=0.1,   # ← Bán kính nhỏ
            height=3.0    # ← Cao
        )
        pole_vis_id = p.createVisualShape(
            p.GEOM_CYLINDER, 
            radius=0.1, 
            length=3.0, 
            rgbaColor=[1, 0, 0, 1]  # ← Màu đỏ dễ thấy
        )
        self.goal_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=pole_collision_id,
            baseVisualShapeIndex=pole_vis_id,
            basePosition=[goal_x, goal_y, 1.5],  # ← Giữa không trung
            physicsClientId=self.CLIENT
        )

        p.addUserDebugText(
            text="GOAL", 
            textPosition=[0, 0, 1.8],        
            textColorRGB=[0, 1, 0],          
            textSize=2.0,                    
            parentObjectUniqueId=self.goal_id, 
            lifeTime=0,                      
            physicsClientId=self.CLIENT
        )
 
    
    # def _computeReward(self):
    #     self.reward = 0
    #     self.collision = self.check_collision()
    #     self.win = self.check_win()
    #     self.over_map = self.check_overmap()
    #     state = self._getDroneStateVector(0)
        
    #     current_dist_to_goal = np.linalg.norm(state[0:3] - self.goal[0])
    #     progress = self.previous_dist - current_dist_to_goal

    #     reward1 = 5.0 * progress
    #     reward1 -= 1.0

    #     goal_vec = self.goal[0] - state[0:3]
    #     goal_dir = goal_vec / (np.linalg.norm(goal_vec) + 1e-8)
    #     vel = state[10:13]
    #     vel_norm = vel / (np.linalg.norm(vel) + 1e-8)
    #     alignment = np.dot(vel_norm, goal_dir)
    #     reward1 += 2.0 * alignment

    #     if self.collision: reward1 -= 500
    #     if self.over_map:  reward1 -= 500
    #     if self.win:       reward1 += 1000

    #     self.previous_dist = current_dist_to_goal
    #     return reward1
    
    def _computeReward(self):
        self.reward = 0
        self.collision = self.check_collision()
        self.win = self.check_win()
        self.over_map = self.check_overmap()
        state = self._getDroneStateVector(0)
        
        current_dist_to_goal = np.linalg.norm(state[0:3] - self.goal[0])
        progress = self.previous_dist - current_dist_to_goal

        reward1 = (2.0 * progress) - 0.001

        if self.collision: reward1 -= 200
        if self.over_map:  reward1 -= 200
        if self.win:       reward1 += 500

        self.previous_dist = current_dist_to_goal
        return reward1

    
    def _computeInfo(self):
        status1 = "Drone đã va chạm với cột. " if self.collision else "Drone chưa có va chạm. "
        status2 = "Drone đã tới đích thành công . " if self.win else "Drone chưa tới đích. "
        return status1 + status2
    
    def _computeTruncated(self):
        status1 = "WIN" if self.win else "NONE"
        status2 = "OVER_STEP" if self.over_step else "NONE"
        status3 = "OVER_MAP" if self.over_map else "NONE"
        check = False if ((status1 == "NONE") and (status2 == "NONE") and (status3 == "NONE")) else True
        return (check, status1, status2, status3)
    
    def _computeTerminated(self):
        if self.collision:
            return True
        return False
    
    def _computeObs(self):
        return self._getDroneStateVector(0)

    def reset(self, options, seed = None):
        JITTERING = 1.0
        self.previous_dist = np.linalg.norm(self.goal - self.start)
        #######     TH1 : Bắt đầu môi trường    ######################
        if options == "begin":
            collision_list = random.sample(config.obstacle_position, 10)
            for pos in collision_list:
                jitter_x = random.uniform(-JITTERING, JITTERING)
                jitter_y = random.uniform(-JITTERING, JITTERING)
                
                new_x = pos[0] + jitter_x
                new_y = pos[1] + jitter_y
                self.obstacles_position.append([new_x, new_y, 0])
            return super().reset()
        #######     TH2  : Reset nhưng vẫn cho thử lại ###########
        elif (options == "try_again"):
            return super().reset()
        ######      TH3  : Reset tất cả                ###########
        else:
            self.obstacles_id = []
            self.obstacles_position = [] 
            self.collision = False
            self.win = False
            self.over_step = False
            self.over_map = False
            self.step_count = 0
            self.reward = 0
            self.start, self.goal = self.add_start_goal()
            self.INIT_XYZS = self.start
            collision_list = random.sample(config.obstacle_position, 10)
            for pos in collision_list:
                jitter_x = random.uniform(-JITTERING, JITTERING)
                jitter_y = random.uniform(-JITTERING, JITTERING)
                
                new_x = pos[0] + jitter_x
                new_y = pos[1] + jitter_y
                
                self.obstacles_position.append([new_x, new_y, 0])

        
            self.previous_dist = np.linalg.norm(self.goal - self.start)
            return super().reset()
    
    def _preprocessAction(self, action):
        return np.array(action).flatten() 

    def _applyAction(self, action):
        super()._applyAction(action)

    def _observationSpace(self):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)  
     
    def _actionSpace(self):
        return spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action=action)
        img = self._getDroneImages(nth_drone=0, segmentation=0)
        self.step_count += 1
        return obs, img, reward, terminated, truncated, info
    
    def check_collision(self):
        contact_points = p.getContactPoints(bodyA=self.DRONE_IDS[0], physicsClientId=self.CLIENT)
        for contact in contact_points:
            id_vat_the_bi_dam = contact[2]
            if id_vat_the_bi_dam != self.DRONE_IDS[0] and id_vat_the_bi_dam != self.PLANE_ID:
                return True
        return False
    
    def check_win(self):
        state = self._getDroneStateVector(0)
        cur_x, cur_y, _ = state[0:3]
        goal_x, goal_y, _ = self.goal[0][0:3]

        if abs(goal_x - cur_x) <= 1 and abs(goal_y - cur_y) <= 1:
            return True
        return False
    
    def check_overmap(self):
        state = self._getDroneStateVector(0)
        if abs(state[0]) > 6.5 or abs(state[1]) > 12:
            return True
        return False

    def _getDroneImages(self, nth_drone, segmentation: bool=True):
        if self.IMG_RES is None:
            print("[ERROR] in BaseAviary._getDroneImages()")
            exit()

        # Vị trí drone
        drone_pos = np.array(self.pos[nth_drone, :])

        # Camera đặt ngay tại drone, nhìn thẳng về hướng -Y
        eye_position = drone_pos + np.array([0, 0, 0.05])  # Nhích lên 5cm

        # Điểm nhìn đích: thẳng về -Y
        target = drone_pos + np.array([0, -1000, 0])  # Nhìn về -Y

        # Up vector: hướng lên Z
        DRONE_CAM_VIEW = p.computeViewMatrix(
            cameraEyePosition=eye_position,
            cameraTargetPosition=target,
            cameraUpVector=[0, 0, 1],
            physicsClientId=self.CLIENT
        )

        DRONE_CAM_PRO = p.computeProjectionMatrixFOV(
            #fov=60.0,
            fov=90,
            aspect=1.0,
            nearVal=self.L,
            farVal=1000.0
        )

        SEG_FLAG = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX if segmentation else p.ER_NO_SEGMENTATION_MASK

        [w, h, rgb, dep, seg] = p.getCameraImage(
            width=self.IMG_RES[0],
            height=self.IMG_RES[1],
            shadow=1,
            viewMatrix=DRONE_CAM_VIEW,
            projectionMatrix=DRONE_CAM_PRO,
            flags=SEG_FLAG,
            physicsClientId=self.CLIENT
        )

        rgb = np.reshape(rgb, (h, w, 4))
        dep = np.reshape(dep, (h, w))
        seg = np.reshape(seg, (h, w))

        return rgb, dep, seg
    
    def get_raycast_sensors(self, nth_drone=0):
        drone_pos = np.array(self.pos[nth_drone, :])
        MAX_RANGE = 5.0 
        
        # --- 1. TIA CHÉO (45 ĐỘ) ---
        offset_x_45 = MAX_RANGE * math.sin(math.pi / 4)
        offset_y_45 = MAX_RANGE * math.cos(math.pi / 4)
        
        target_left_diag  = drone_pos + np.array([-offset_x_45, -offset_y_45, 0])
        target_right_diag = drone_pos + np.array([offset_x_45, -offset_y_45, 0])
        
        # --- 2. TIA NGANG (90 ĐỘ - TRÁI/PHẢI THUẦN TÚY) ---
        # Sang trái là -X, sang phải là +X. Trục Y giữ nguyên = 0
        target_left_90  = drone_pos + np.array([-MAX_RANGE, 0, 0])
        target_right_90 = drone_pos + np.array([MAX_RANGE, 0, 0])
        
        # --- 3. BẮN 4 TIA CÙNG LÚC ---
        ray_left_diag  = p.rayTest(drone_pos, target_left_diag)[0]
        ray_right_diag = p.rayTest(drone_pos, target_right_diag)[0]
        ray_left_90    = p.rayTest(drone_pos, target_left_90)[0]
        ray_right_90   = p.rayTest(drone_pos, target_right_90)[0]
        
        # --- 4. LẤY KẾT QUẢ CHUẨN HÓA (0.0 đến 1.0) ---
        norm_left_diag  = ray_left_diag[2]
        norm_right_diag = ray_right_diag[2]
        norm_left_90    = ray_left_90[2]
        norm_right_90   = ray_right_90[2]
        
        # Trả về mảng 4 giá trị
        return np.array([norm_left_diag, norm_right_diag, norm_left_90, norm_right_90])