import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# 1. Đọc dữ liệu
df = pd.read_csv('neutron_array_DQN.csv', comment='#')

# 2. Xử lý dữ liệu cột List_Actions (Đã fix lỗi float32)
def extract_max_q(action_string):
    if pd.isna(action_string):
        return np.nan
    clean_string = str(action_string).replace('float32', '').replace('float64', '')
    numbers = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", clean_string)
    if not numbers:
        return np.nan
    return max([float(num) for num in numbers])

df['Max_Q_Value'] = df['List_Actions'].apply(extract_max_q)

# 3. Tổng hợp dữ liệu theo Eposide
episode_stats = df.groupby('Eposide').agg(
    Epsilon=('Epsilon', 'last'),
    Avg_Max_Q=('Max_Q_Value', 'mean')
).reset_index()

# Tính trung bình trượt 20 tập cho Q-Value
episode_stats['Moving_Q'] = episode_stats['Avg_Max_Q'].rolling(window=20, min_periods=1).mean()

# 4. Thống kê hành động
action_counts = df['Action_Decide'].value_counts()

# ================= VẼ BIỂU ĐỒ =================
# Chuyển thành 2 subplots thay vì 3
fig, axes = plt.subplots(2, 1, figsize=(12, 12))

# --- BIỂU ĐỒ 1: Epsilon Decay & Q-Value (Sử dụng 2 trục Y) ---
ax1 = axes[0]

# Vẽ Q-Value trên trục Y bên trái
line1 = ax1.plot(episode_stats['Eposide'], episode_stats['Avg_Max_Q'], alpha=0.3, color='orange', label='Q-Value từng tập')
line2 = ax1.plot(episode_stats['Eposide'], episode_stats['Moving_Q'], color='red', linewidth=2, label='Q-Value trung bình (20 tập)')
ax1.set_xlabel('Tập (Episode)')
ax1.set_ylabel('Giá trị Q-Value', color='red', fontweight='bold')
ax1.tick_params(axis='y', labelcolor='red')
ax1.grid(True, linestyle='--', alpha=0.6)

# Tạo trục Y thứ hai (bên phải) dùng chung trục X cho Epsilon
ax2 = ax1.twinx()
line3 = ax2.plot(episode_stats['Eposide'], episode_stats['Epsilon'], color='purple', linewidth=2.5, linestyle='--', label='Epsilon')
ax2.set_ylabel('Tỷ lệ khám phá (Epsilon)', color='purple', fontweight='bold')
ax2.tick_params(axis='y', labelcolor='purple')
ax2.set_ylim(0, 1.05)
ax2.set_yticks(np.arange(0, 1.1, 0.1))

# Gộp Legend (chú thích) của cả 2 trục vào chung 1 bảng
# ==========================================
# THAY THẾ TOÀN BỘ PHẦN CUỐI TỪ ĐÂY
# ==========================================

# Gộp Legend của cả 2 trục vào chung 1 bảng
lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]

# Cách an toàn 100%: Nhét Legend vào bên trong đồ thị ở vị trí trống (giữa bên trái)
ax1.legend(lines, labels, loc='center right') 
ax1.set_title('1. Tương quan giữa Tỷ lệ Khám phá (Epsilon) và Nhận thức của AI (Q-Value) của DQN', pad=15)

# Xóa nhãn trục X của biểu đồ trên cho đỡ chật (vì biểu đồ dưới đã có chữ rồi)
ax1.set_xlabel('') 

# --- BIỂU ĐỒ 2: Phân phối Hành động ---
ax_bar = axes[1]
ax_bar.bar(action_counts.index, action_counts.values, color='teal')
ax_bar.set_title('2. Phân phối lựa chọn Hành động (Action Distribution) của DQN')
ax_bar.set_ylabel('Số lần chọn')
ax_bar.set_xlabel('Tập (Episode)') # Đưa nhãn Tập xuống dưới cùng
ax_bar.tick_params(axis='x', rotation=45) 
ax_bar.grid(axis='y', linestyle='--', alpha=0.6)

# Ép khoảng cách dọc (h_pad) giữa 2 biểu đồ một cách mạnh bạo (tăng số 4.0 lên 5.0 nếu màn hình bạn phân giải cao)
plt.tight_layout(h_pad=4.0) 

plt.show()