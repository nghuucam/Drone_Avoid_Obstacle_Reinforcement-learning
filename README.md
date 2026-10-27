<p align="center">
  <a href="https://www.uit.edu.vn/" title="Trường Đại học Công nghệ Thông tin" style="border: none;">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="Trường Đại học Công nghệ Thông tin | University of Information Technology">
  </a>
</p>

# 🚁 Project: Training a Drone for Obstacle Avoidance using Reinforcement Learning (RL)

## 📑 TABLE OF CONTENTS

* [Course Introduction](#gioithieumonhoc)
* [Instructor](#giangvien)
* [Team Members](#thanhvien)
* [Project Overview](#doan)
* [Technologies & Algorithms](#congnghe)
* [Simulation Environment](#moitruong)
* [Results & Evaluation](#ketqua)
* [Installation & Deployment](#caidat)
* [Training Results Comparison (DQN vs. D3QN)](#sosanh)

## 🎓 COURSE INTRODUCTION <a name="gioithieumonhoc"></a>

* **Course Name:** Specialized Project
* **Course Code:** NT114
* **Class:** NT114.Q21

## 👨‍🏫 INSTRUCTOR <a name="giangvien"></a>

* MSc. **Nguyễn Khánh Thuật**

## 👨‍💻 TEAM MEMBERS <a name="thanhvien"></a>

| No. | Student ID | Full Name | Github | Email | 
| :---: | :---: | :--- | :--- | :--- | 
| 1 | 23520164 | Nguyễn Hữu Cảm | [nghuucam](https://github.com/nghuucam) | 23520164@gm.uit.edu.vn | 

## 🚀 PROJECT OVERVIEW <a name="doan"></a>

**Project Title:** Training a Drone for Obstacle Avoidance based on Reinforcement Learning.

**Problem Description:**
This project aims to build an autonomous system for a Drone in a 3D simulation environment. In this environment, obstacles (cylindrical blocks) are generated at random positions in each episode. The Drone's task is to use its "vision" and "sensors" to find the shortest flight path to the destination safely without colliding.

The project implements, optimizes, and compares the performance of two advanced Deep Reinforcement Learning algorithms:

1. **DQN** (Deep Q-Network)
2. **D3QN** (Dueling Double Deep Q-Network)

## 🛠 TECHNOLOGIES & ALGORITHMS <a name="congnghe"></a>

* **Programming Language:** Python
* **Deep Learning Framework:** PyTorch
* **Data & Image Processing Libraries:** NumPy, OpenCV, Pandas, Matplotlib
* **Optimization Techniques (Applied):**
  * **Sensor Fusion:** Combining spatial images and 4 virtual Lidar rays (Raycast) to eliminate blind spots.
  * **Soft Target Update:** Continuously updating the Target network weights at a small rate (0.5% per step) to prevent Gradient shock and thoroughly resolve the "Q-Value Overestimation" issue of D3QN.
  * **Epsilon Decay:** Gradually decreasing the random exploration rate over episodes.

## 🌍 SIMULATION ENVIRONMENT <a name="moitruong"></a>

The system uses the **`gym-pybullet-drones`** framework (realistic physics simulation of Drones).

* **State Space (21 dimensions):** Includes Coordinates (X, Y, Z), Rotation Angles (Roll, Pitch, Yaw), Velocity, Angular Velocity, Action Hints, and Data from 4 Lidar sensor rays.
* **Action Space (5 Discrete):** Move Forward, Move Left, Move Right, Move Diagonally Left, Move Diagonally Right.
* **Reward System:**
  * Reaching the destination: `+1000` points.
  * Colliding with an obstacle / Flying out of bounds / Running out of time: `-100` to `-200` points.
  * Movement reward: Based on the distance moving closer to the destination.

## 📊 RESULTS & EVALUATION <a name="ketqua"></a>

Based on the training results (500 episodes) and testing (100 test episodes per algorithm):

* **DQN Algorithm:** Demonstrates fast convergence speed, learns how to survive well in the early episodes, and achieves a highly positive success rate of reaching the target in a dynamic environment.
* **D3QN Algorithm:** Excels in stability during the later stages. Thanks to the Dueling architecture that separates State Value and Action Advantage, combined with Soft Update, D3QN makes smoother and safer obstacle-avoidance decisions.

## 💻 INSTALLATION & DEPLOYMENT <a name="caidat"></a>

### System Requirements

* Python 3.8+
* Virtual environment (Conda / Venv) recommended.

### Installation Steps

**1. Initialize Environment:** 
Download the Gym-pybullet-drones simulation environment

```bash
git clone https://github.com/nghuucam/Drone_Avoid_Obstacle_Reinforcement-learning.git
cd Drone_Avoid_Obstacle_Reinforcement-learning/

conda create -n drones python=3.11
conda activate drones

# You must install Microsoft C++ Build Tools with Desktop development with C++
pip3 install -e . # if needed, `sudo apt install build-essential` to install `gcc` and build `pybullet`
pip install -r requirements.txt

# check installed packages with `conda list`, deactivate with `conda deactivate`, remove with `conda env remove -n drones`
```


**2. Training the Models**

For the DQN algorithm:
```bash
cd DQN
python train.py

```

For the D3QN algorithm:

```bash
cd D3QN
python train.py

```

*After the training is complete, three `.xlsx` log files and the trained model weights will be generated and saved in the `Model` directory.*

**3. Results Evaluation**

**3.1. Training Analysis**

Copy the three `.xlsx` files generated during training and paste them into the respective `Analyze/DQN` or `Analyze/D3QN` folders. Then, run the following commands to generate the analysis charts:

For DQN:

```bash
cd Analyze/DQN
python ana1.py
python ana2.py
python ana3.py
python ana4.py
python ana5.py

```

For D3QN:

```bash
cd Analyze/D3QN
python ana1.py
python ana2.py
python ana3.py
python ana4.py
python ana5.py

```

**Chart Descriptions:**

* **Chart 1 (`ana1.py`):** Overall test results analysis.
* **Chart 2 (`ana2.py`):** Analysis of Q-Value, Epsilon decay, and Action distribution.
* **Chart 3 (`ana3.py`):** Evaluation of Accumulated Reward and Win Rate.
* **Chart 4 (`ana4.py`):** Tracking the model's Loss.
* **Chart 5 (`ana5.py`):** Assessment of survival capability based on Reward and Step count.

**3.2. Testing in a Randomized Environment**

You can evaluate the trained models by running test scenarios in an environment with randomly generated obstacles.

*(**Note:** Before running the test, please update line 160 in the `test.py` script to point to the exact path of your trained model file).*

Testing the DQN model:

```bash
cd Test/DQN
python test.py

```

Testing the D3QN model:

```bash
cd Test/D3QN
python test.py

```

---

## 📈 TRAINING RESULTS COMPARISON: DQN vs. D3QN <a name="sosanh"></a>

Both **DQN** and **D3QN** were trained over **500 episodes** under identical 3D obstacle-avoidance simulation conditions (`gym-pybullet-drones`). Below is a detailed side-by-side comparison across four core evaluation metrics.

### 1. Summary Comparison Table

| Evaluation Metric | DQN | D3QN | Comparison & Key Takeaway |
| :--- | :---: | :---: | :--- |
| **Zero-Reward Crossing Point** *(100-ep MA)* | ~Episode 205 | **~Episode 180** | **D3QN** learns positive-reward trajectories faster. |
| **Peak Accumulated Reward** *(100-ep MA)* | ~395 pts *(Ep 408)* | **~475 pts** *(Ep 435)* | **D3QN** achieves a **~20% higher** peak moving-average score. |
| **Final Accumulated Reward** *(Ep 500, 100-ep MA)* | ~335 pts | **~415 pts** | **D3QN** maintains higher and more consistent rewards at convergence. |
| **Peak Win Rate** *(100-ep MA)* | ~81% *(Ep 408)* | **~92%** *(Ep 435)* | **D3QN** outperforms DQN by **+11%** in peak success rate. |
| **Final Win Rate** *(Ep 400–500 range)* | 72% – 81% | **83% – 92%** | **D3QN** demonstrates superior reliability in the late stage. |
| **Average Flight Steps** *(Converged)* | ~30 – 33 steps | **~30 – 32 steps** | Both converge to optimal path lengths; **D3QN** has fewer crash drops. |
| **Early Training Loss Peak** *(20-ep MA)* | ~15.8 *(spike: 21.8)* | **~13.7** *(spike: 18.6)* | **D3QN** suppresses early gradient shock and Q-value overestimation. |

---

### 2. Accumulated Reward & Win Rate

| DQN | D3QN |
| :---: | :---: |
| ![DQN Accumulated Reward vs Win Rate](img/DQN%20_%20Accumulated%20Reward%20vs%20Win%20Rate.png) | ![D3QN Accumulated Reward vs Win Rate](img/D3QN%20Accumulated%20Reward%20vs%20Win%20Rate.png) |

* **DQN:** Starts recording wins around Episode 40 and steadily improves after Episode 100. Its 100-episode moving average reward crosses `0` at Episode ~205 and peaks at **~395 points** (with an **~81% Win Rate**) around Episode 408, before fluctuating between **72%–77%** near Episode 500.
* **D3QN:** Begins winning earlier (around Episode 30) and crosses the `0` reward threshold faster (at Episode ~180). In the late stage (Episodes 370–500), D3QN reaches a peak moving-average reward of **~475 points** and a **Win Rate of ~92%**, sustaining **83%–92%** until the end of training with noticeably fewer failed episodes (grey drop lines).

---

### 3. Survival Capability & Path Optimization (Reward vs. Average Steps)

| DQN | D3QN |
| :---: | :---: |
| ![DQN Accumulated Reward vs Average Step](img/DQN%20_%20Accumulated%20Reward%20vs%20Average%20Step.png) | ![D3QN Accumulated Reward vs Average Step](img/D3QN%20_%20Accumulated%20Reward%20vs%20Average%20Step.png) |

* **DQN:** During the initial exploration phase, the average step count drops to **~12 steps** due to frequent early collisions. As the agent learns obstacle avoidance, flight duration gradually rises and stabilizes around **30–33 steps** per episode after Episode 300.
* **D3QN:** Reaches the optimal **~30 steps** plateau much earlier (by Episode 220) and maintains a remarkably flat step curve (**30–32 steps**) through Episode 500. The raw step trace (light red) in D3QN exhibits significantly fewer sudden downward spikes in the final 150 episodes, confirming safer flight trajectories.

---

### 4. Exploration (Epsilon), Q-Value Estimation & Action Distribution

| DQN | D3QN |
| :---: | :---: |
| ![DQN Epsilon vs Q-Values vs Action Distribution](img/DQN%20_%20Epsilon%20Delay%20vs%20Q-Values%20vs%20Action%20Distribution.png) | ![D3QN Epsilon vs Q-Values vs Action Distribution](img/D3QN%20_%20Epsilon%20Delay%20vs%20Q-Values%20vs%20Action%20Distribution.png) |

* **Epsilon Decay & Q-Value Trend:** Both algorithms decay $\epsilon$ linearly from `1.0` down to `0.1` at Episode 200.
  * **DQN** exhibits a rapid Q-value surge up to **~17.2** around Episode 265 accompanied by high episode-to-episode variance (raw spikes reaching `~21`), then plateaus around `15.5–16.0`.
  * **D3QN** controls overestimation during the transition from exploration to exploitation (staying stable at `14.5–16.0` between Episodes 200–400 with tighter variance) and smoothly scales up to **~19.2** by Episode 500 as its policy locks onto high-return paths.
* **Action Distribution:**
  * Both agents prioritize forward-advancing maneuvers (**Move Diagonally Right**, **Move Diagonally Left**, and **Move Forward**) over pure sideways movements (**Move Left**, **Move Right**), enabling the drone to dodge cylindrical obstacles while continuously closing the distance to the goal.
  * **D3QN** shows an even stronger preference for diagonal evasive maneuvers (**>4,200 selections** each for *Move Diagonally Left* and *Move Diagonally Right* vs. **~3,850** in DQN), resulting in smoother obstacle-bypassing curves instead of abrupt lateral turns.

---

### 5. Training Loss Stability

| DQN | D3QN |
| :---: | :---: |
| ![DQN Loss](img/DQN%20_%20Loss.png) | ![D3QN Loss](img/D3QN%20_%20Loss.png) |

* **DQN:** Experiences a sharp initial Loss spike during Episodes 25–50, where the 20-episode moving average hits **~15.8** (and raw episode loss spikes up to **~21.8**) due to target instability and Q-value overestimation in standard DQN, before settling around `10.0–12.5`.
* **D3QN:** By decoupling action selection from evaluation (**Double DQN**) and separating State Value $V(s)$ from Action Advantage $A(s, a)$ (**Dueling Architecture**), D3QN dampens the initial gradient shock—keeping the early moving-average Loss peak down to **~13.7**—and maintains a steady, controlled loss curve throughout the 500 episodes.

---

### 🏆 Overall Conclusion

While **DQN** converges quickly and achieves a solid success rate (**~72%–81%**), **D3QN** proves superior across all evaluation criteria in dynamic 3D obstacle avoidance:
1. **Higher Success Rate & Reward:** Reaches up to **92% Win Rate** and **~475 average reward** (vs. 81% and ~395 for DQN).
2. **Smarter Navigation Strategy:** Strongly favors diagonal evasive trajectories, maintaining a steady **~30–32 steps** to reach the target with minimal collision drops.
3. **Greater Training Stability:** Mitigates Q-value overestimation and reduces early-stage Loss spikes.

<!-- ///**2. Chạy thử các thuật toán**
DQN
```bash
cd DQN
python train.py
```

D3QN
```bash
cd D3QN
python train.py
```

Sau thực hiện việc huấn luyện, chúng ta sẽ nhận được 3 file .xlsx và các file Model được lưu trong thư mục Model

**3. Đánh giá các kết quả**
**3.1. Đánh giá quá trình huấn luyện**
Thực hiện copy 3 file .xlsx sau quá trình huấn luyện, paste vào folder Analyze với các thư mục DQN/D3QN sau đó chạy các câu lệnh sau để phân tích
Đổi với DQN

```bash
cd Analyze/DQN
python ana1.py
python ana2.py
python ana3.py
python ana4.py
python ana5.py
```

Đối với D3QN
```bash
cd Analyze/D3QN
python ana1.py
python ana2.py
python ana3.py
python ana4.py
python ana5.py
```

Ý nghĩa của các biểu đồ:
- Biểu đồ 1: Phân tích kết quả kiểm thử tổng quan
- Biểu đồ 2: Phân tích Q-Value, Epsilon và Hành động
- Biểu đồ 3: Đánh giá Điểm thưởng và Tỷ lệ chiến thắng
- Biểu đồ 4: Theo dõi mức độ lỗi của mô hình - Loss
- Biểu đồ 5: Đánh giá Khả năng sinh tồn qua Điểm số và Số bước

**3.2. Kiểm thử trong môi trường ngẫu nhiên**
Với các file Model đã có sau khi huấn luyện thì có thể thực hiện sử dụng các bài kiểm thử trong môi trường có vật cản được sinh ngẫu nhiên để đánh giá được mô hình. Đoạn code cần thay đổi các trường ở trong dòng thứ 160 để có đường dẫn mô hình.

Kiểm thử mô hình với thuật toán DQN
```bash
cd Test/DQN
python test.py
```

Kiểm thử mô hình với thuật toán D3QN
```bash
cd Test/D3QN
python test.py
```
///-->
