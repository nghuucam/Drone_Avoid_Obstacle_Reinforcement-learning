import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Tên file log (Đổi thành tên file thực tế của bạn)
file_name = 'drone_flight_log_test_DQN.csv' # hoặc 'drone_flight_log_fixed.csv'

# Đọc dữ liệu
df = pd.read_csv(file_name)

# 2. Lọc ra bước (step) cuối cùng của mỗi Eposide
# Lệnh .last() sẽ lấy ra dòng cuối cùng của mỗi nhóm
episodes_summary = df.groupby('Eposide').last().reset_index()

# 3. Tổng hợp điểm AccumReward từng tập
reward_per_episode = episodes_summary[['Eposide', 'AccumReward', 'Win', 'Collision']]

# 4. Tính toán các chỉ số thống kê
total_episodes = len(episodes_summary)
overall_win_rate = episodes_summary['Win'].mean() * 100
overall_collision_rate = episodes_summary['Collision'].mean() * 100
avg_reward = episodes_summary['AccumReward'].mean()

print(f"========== TỔNG KẾT HUẤN LUYỆN ==========")
print(f"Tổng số tập (Episodes) : {total_episodes}")
print(f"Tỷ lệ chiến thắng (Win Rate)  : {overall_win_rate:.2f}%")
print(f"Tỷ lệ đâm đụng (Collision Rate): {overall_collision_rate:.2f}%")
print(f"Điểm thưởng trung bình   : {avg_reward:.2f}")
print("=========================================")

print("\n--- 5 Tập gần nhất ---")
print(reward_per_episode.tail(5).to_string(index=False))

# 5. (Tùy chọn) Tính trung bình trượt (Moving Average) để vẽ biểu đồ cho mượt
# Tính Win Rate và Reward trên mỗi cửa sổ 50 tập
window_size = 100
episodes_summary['Moving_Reward'] = episodes_summary['AccumReward'].rolling(window=window_size, min_periods=1).mean()
episodes_summary['Moving_Win_Rate'] = episodes_summary['Win'].rolling(window=window_size, min_periods=1).mean() * 100

# ================= VẼ BIỂU ĐỒ =================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Biểu đồ 1: Điểm thưởng (AccumReward)
ax1.plot(episodes_summary['Eposide'], episodes_summary['AccumReward'], alpha=0.3, color='gray', label='Điểm từng tập')
ax1.plot(episodes_summary['Eposide'], episodes_summary['Moving_Reward'], color='blue', linewidth=2, label=f'Trung bình trượt ({window_size} tập)')
ax1.set_title('Biểu đồ Điểm thưởng (Accumulated Reward) qua các hiệp của DQN')
ax1.set_ylabel('Điểm')
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()

# Biểu đồ 2: Tỷ lệ chiến thắng (Win Rate)
ax2.plot(episodes_summary['Eposide'], episodes_summary['Moving_Win_Rate'], color='green', linewidth=2, label=f'Win Rate trượt ({window_size} tập)')
ax2.set_title('Biểu đồ Tỷ lệ chiến thắng (Win Rate) của DQN')
ax2.set_xlabel('Eposide')
ax2.set_ylabel('Tỷ lệ thắng (%)')
ax2.set_ylim(0, 105) # Giới hạn trục Y từ 0 - 100%
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend()


plt.tight_layout()
plt.show()