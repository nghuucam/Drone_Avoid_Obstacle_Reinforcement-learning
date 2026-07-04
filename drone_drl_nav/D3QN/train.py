import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import time
import torch
import pybullet as p
import csv
import matplotlib.pyplot as plt

from gym_pybullet_drones.utils.enums import DroneModel
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.My_Coding.D3QN.new_env import DroneEnv 
from gym_pybullet_drones.My_Coding.D3QN.agent import D3QN, DroneNet

def get_action_hints(obs, env, direct):
    state = obs[0:3]
    current_dist = np.linalg.norm(state - env.goal[0])
    
    hints = []
    for target in direct:
        future_dist = np.linalg.norm(target - env.goal[0])
        progress = current_dist - future_dist
        hints.append(progress)
    
    hints = np.array(hints, dtype=np.float32)
    max_val = np.max(np.abs(hints)) + 1e-8
    hints = (hints / max_val) * 0.3
    
    return hints

def get_vec_state(obs, env, action_hints):
    pos   = obs[0:3]    # 3
    rpy   = obs[7:10]   # 3
    vel   = obs[10:13]  # 3
    ang_v = obs[13:16]  # 3
    hints = action_hints  # 5
    
    # Gọi hàm 4 cảm biến mới
    four_sensors = env.get_raycast_sensors() # <--- Đổi tên hàm ở đây

    # Nối tất cả lại (17 + 4 = 21 chiều)
    vec = np.concatenate([pos, rpy, vel, ang_v, hints, four_sensors]) 
    return np.expand_dims(vec, axis=0)

def get_img_state(rgb_img):
    img = rgb_img[:, :, :3].astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)

def speech_bubble(cur_pos, env):
    bubble_text = f"X: {cur_pos[0]:.2f} Y: {cur_pos[1]:.2f} Z: {cur_pos[2]:.2f}"
    env.bubble_id = p.addUserDebugText(
        text=bubble_text,
        textPosition=[0, 0, 0.2],         
        textColorRGB=[69/255, 29/255, 23/255],           
        textSize=1.0,                     
        lifeTime=0,                     
        parentObjectUniqueId=env.DRONE_IDS[0],
        replaceItemUniqueId=getattr(env, 'bubble_id', -1),
        physicsClientId=env.CLIENT
    )

def go_up(obs, control, env):
    cur_obs = env._getDroneStateVector(0)
    current_x, current_y = cur_obs[0:2]

    start_time = time.time()
    TIMEOUT = 10.0

    while True:
        cur_obs = env._getDroneStateVector(0)
        cur_pos = cur_obs[0:3]
        cur_quat = cur_obs[3:7]  
        cur_vel = cur_obs[10:13]  
        cur_ang_v = cur_obs[13:16]

        # Chỉ bay thẳng đứng lên trời từ vị trí hiện tại
        target_pos = np.array([current_x, current_y, 1.0])
        
        action_go_up, _, _ = control.computeControl(
            control_timestep=env.CTRL_TIMESTEP,
            cur_pos=cur_pos,
            cur_quat=cur_quat,
            cur_vel=cur_vel,
            cur_ang_vel=cur_ang_v,
            target_pos=target_pos,
            target_rpy=np.array([0.0, 0.0, 0.0]),
        )
        speech_bubble(cur_pos=cur_pos, env=env)

        obs_ret, _, _, _, _, _ = env.step(np.array([action_go_up]))
        if isinstance(obs_ret, tuple):
            obs_ret = obs_ret[0]

        if abs(cur_pos[2] - 1.0) < 0.01:
            print("🚀 Đã bay lên độ cao an toàn!")
            break
            
        if time.time() - start_time > TIMEOUT:
            raise RuntimeError("Drone bị lật úp hoặc kẹt dưới sàn không thể cất cánh!")

    return obs_ret

def move(target_pos, obs, control, env):
    accumulated_reward = 0
    
    for step in range(200):
        cur_pos = obs[0:3]
        cur_quat = obs[3:7]
        cur_vel = obs[10:13]
        cur_ang_v = obs[13:16]
        
        _, _, z_point = cur_pos[0:3]
        
        if z_point - 0.9 < 0:
            obs = go_up(obs=obs, control=control, env=env)
            cur_pos = obs[0:3]
            cur_quat = obs[3:7]
            cur_vel = obs[10:13]
            cur_ang_v = obs[13:16]
            state = env._getDroneStateVector(0)
            env.previous_dist = np.linalg.norm(state[0:3] - env.goal[0]) 
            
        action_go, _, _ = control.computeControl(
            control_timestep=env.CTRL_TIMESTEP,
            cur_pos=cur_pos, 
            cur_quat=cur_quat,
            cur_vel=cur_vel, 
            cur_ang_vel=cur_ang_v,
            target_pos=target_pos,
            target_rpy=np.array([0.0, 0.0, 0.0]) 
            #target_rpy=np.array([0.0, 0.0, -np.pi/2])
        )
        speech_bubble(cur_pos=cur_pos, env=env)
        
        next_obs, img, reward, terminated, truncated, info = env.step(np.array([action_go]))
        if isinstance(next_obs, tuple): next_obs = next_obs[0]
        
        accumulated_reward += reward
        obs = next_obs
        
        dist_to_target = np.linalg.norm(cur_pos[0:2] - target_pos[0:2])
        speed = np.linalg.norm(cur_vel)
        wobble_speed = np.linalg.norm(cur_ang_v)
        
        if step > 20 and dist_to_target < 0.05 and speed < 0.1 and wobble_speed < 0.2:
            break 
            
        if terminated or truncated[0]:
            break            
    
    return obs, img, accumulated_reward, terminated, truncated, info 

def main():
    env = DroneEnv(gui=True)
    model = DroneNet(n_actions=5, state_vector_dim=21)
    d3qn_agent = D3QN(model, n_actions=5)
    control = DSLPIDControl(drone_model=DroneModel.CF2X)
    start_step = 1

    eposide = 1
    max_eposide0 = 200
    max_eposide = 500
    
    max_epsilon =  1.0
    min_epsilon = 0.1
    
    MAX_ACTIONS = 70
    action_count = 0

    accum_reward = 0

    log_file = "drone_flight_log_test_D3QN.csv"
    log_entropy_cnn = "neutron_array_D3QN.csv"
    log_loss = "log_loss_D3QN.csv"
    eposide_losses = []

    reduce_epison = (max_epsilon - min_epsilon) / (max_eposide0 - 0)
    
    ############    Đầu tiên cho máy bay bay lên    ######################
    obs, info = env.reset(options="begin")
    print(f"##################      EPOSIDE {eposide}       ######################### ")
    obs = go_up(obs=obs, control=control, env=env)
    env.previous_dist = np.linalg.norm(obs[0:3] - env.goal[0])

    start_time = time.perf_counter()
    ############    Lấy bức ảnh đầu tiên khi drone bay lên ###############
    rgb, _, _ = env._getDroneImages(nth_drone=0, segmentation=False)

    ############    Chuẩn bị thư mục để ghi log    #######################
    if not os.path.exists(log_file):
        with open(log_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Eposide',
                             'Step',
                             'Start',
                             'End', 
                             'X', 
                             'Y', 
                             'Z', 
                             'Collision',
                             'Win',
                             'Over_Step',
                             'Over_Map',
                             'Epsilon',
                             'Action',
                             'AccumReward'])
    log_f = open(log_file, mode='a', newline='', encoding='utf-8')
    log_writer = csv.writer(log_f)
    
    
    ############    Chuẩn bị thư mục để xem mạng    #########################
    if not os.path.exists(log_entropy_cnn):
        with open(log_entropy_cnn, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Eposide',
                             'Step',
                             'Start',
                             'End',
                             'List_Actions', 
                             'Epsilon', 
                             'Action_Decide'])
    log_f1 = open(log_entropy_cnn, mode='a', newline='', encoding='utf-8')
    log_neutrol = csv.writer(log_f1)
    
    ##########      Ghi lại loss            #############################
    if not os.path.exists(log_loss):
        with open(log_loss, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Eposide',
                             'Loss'])
    
    log_f2 = open(log_loss, mode='a', newline='', encoding='utf-8')
    log_loss_writer = csv.writer(log_f2)


    ###########     Quá trình huấn luyện    #############################
    while(True):
        try:
            print(f"Step: {start_step}")
            x_point, y_point, z_point = obs[0:3]

        ###########     Bay lên khi rớt xuống   #############################
            if z_point - 0.9 < 0:
                obs = go_up(obs=obs, control=control, env=env)
                x_point, y_point, z_point = obs[0:3]
                env.previous_dist = np.linalg.norm(obs[0:3] - env.goal[0])
            rgb, _, _ = env._getDroneImages(nth_drone=0, segmentation=False)
            
        ###########     Đưa ra quyết định   ####################################
            RANGE = 0.5
            direct = [np.array([x_point, y_point - RANGE, 1]),
                    np.array([x_point - RANGE, y_point, 1]),
                    np.array([x_point + RANGE, y_point, 1]),
                    np.array([x_point + RANGE, y_point - RANGE, 1]),
                    np.array([x_point - RANGE, y_point - RANGE, 1])]
            action_hints = get_action_hints(obs, env, direct)
            s_img = get_img_state(rgb)
            s_vec = get_vec_state(obs, env, action_hints)



            list_action, action_idx = d3qn_agent.get_action(s_img, s_vec, max_epsilon)

            action_names = ["Đi thẳng", "Đi phải", "Đi trái", "Đi chéo trái", "Đi chéo phải"]
            print(f"Q-Values: {list_action}  -> Chọn: {action_names[action_idx]}")
            
            target_pos = direct[action_idx]
            next_obs, three_img, reward, terminated, truncated, info = move(target_pos=target_pos, obs=obs, control=control, env=env)
            accum_reward += reward
            action_count += 1
            if isinstance(next_obs, tuple): next_obs = next_obs[0]


            #time.sleep(5.0)
            ############    Đưa ra đúng hay sai     ##############################
            done = truncated[0] or terminated or (action_count >= MAX_ACTIONS)
            rgb, _, _ = three_img[0:3]
            
            if three_img is not None and not isinstance(three_img, float):
                rgb, _, _ = three_img[0:3]
            else:
                rgb, _, _ = env._getDroneImages(nth_drone=0, segmentation=False)

            x1, y1, z1 = next_obs[0:3]
            direct1 = [np.array([x1, y1 - RANGE, 1]),
                np.array([x1 - RANGE, y1, 1]),
                np.array([x1 + RANGE, y1, 1]),
                np.array([x1 + RANGE, y1 - RANGE, 1]),
                np.array([x1 - RANGE, y1 - RANGE, 1])]
            action_hints1 = get_action_hints(next_obs, env, direct1)
            ns_img = get_img_state(rgb)
            ns_vec = get_vec_state(next_obs, env, action_hints1)



            if action_count >= MAX_ACTIONS: 
                reward -= 200
                print("Hết lượt di chuyển!!!\n")
            
            d3qn_agent.store_transition(s_img, s_vec, action_idx, reward, ns_img, ns_vec, done)
            loss = d3qn_agent.learn()
            if loss > 0:
                eposide_losses.append(loss)
        ############    Ghi log dữ liệu         ##############################
            WIN = "1" if truncated[1] != "NONE" else "0"
            OVER_STEP = "1" if action_count >= MAX_ACTIONS else "0"
            OVER_MAP = "1" if truncated[3] != "NONE" else "0"   
            COLLISION = "1" if terminated else "0"

            log_writer.writerow([eposide, start_step, env.start, env.goal, x1, y1, z1, COLLISION, WIN, OVER_STEP, OVER_MAP, max_epsilon, action_names[action_idx], accum_reward])
            log_neutrol.writerow([eposide, start_step, env.start, env.goal, list_action, max_epsilon, action_names[action_idx]])

            start_step += 1
        ############    Khi bị truncated hoặc terminated    ##################
            if done:
                print(f"Màn chơi này thực hiện tổng cộng {action_count} hành động")
                if (truncated[0] or terminated or (action_count >= MAX_ACTIONS)):
                    obs, _ = env.reset(options="reset")
                    start_step = 1
                    eposide += 1
                    
                    avg_loss = np.mean(eposide_losses) if eposide_losses else 0
                    log_loss_writer.writerow([eposide,avg_loss])
                    eposide_losses = []
                    accum_reward = 0

                    if max_epsilon > min_epsilon : max_epsilon -= reduce_epison
                    if WIN == "1": print("Màn chơi thành công tới đích !!!\n")
                    if OVER_STEP == "1": print("Đi quá số bước !!!\n")
                    if OVER_MAP == "1": print("Bay ra ngoài map !!!\n")
                    if COLLISION == "1": print("Bị va chạm với cột !!!\n")
                    print(f"##################      EPOSIDE {eposide}       ######################### ")
                action_count = 0
            else:
                obs = next_obs
                s_img = ns_img
                s_vec = ns_vec

            if eposide > 0 and eposide % 100 == 0:
                save_path = f"Model/drone_model_d3qn_eposide{eposide}.pth"
                d3qn_agent.save(save_path)
            
            if eposide == max_eposide: 
                end_time = time.perf_counter()
                print(f"###     THỜI GIAN ĐỂ TRAIN BẰNG THUẬT TOÁN D3QN LÀ {end_time-start_time} GIÂY       ###\n")
                break
        
        except Exception as e:
            print(f"\n💥 MÔI TRƯỜNG BỊ KẸT: {e}")
            print("🔄 Đang dọn dẹp xác drone và khởi động lại map mới...")
            obs, _ = env.reset(options="reset")
            
            eposide_losses = []
            eposide += 1
            action_count = 1
            accum_reward = 0
            env.previous_dist = np.linalg.norm(obs[0:3] - env.goal[0])

            print(f"################## BẮT ĐẦU LẠI EPISODE {eposide} #########################\n")
            continue
    
    log_f.close()
    log_f1.close()
    log_f2.close()

if __name__ == "__main__":
    main()