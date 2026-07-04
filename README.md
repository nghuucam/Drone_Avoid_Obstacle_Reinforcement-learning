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

Based on the training results (1000 episodes) and testing (100 test episodes per algorithm):

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

conda create -n drones python=3.10
conda activate drones

pip3 install -e . # if needed, `sudo apt install build-essential` to install `gcc` and build `pybullet`

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
