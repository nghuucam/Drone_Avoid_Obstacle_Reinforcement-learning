import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# 1. Đọc dữ liệu
df = pd.read_csv('log_loss_DQN.csv')

# Đổi tên cột 'Eposide' thành 'Episode' cho chuẩn chính tả nếu cần
df.rename(columns={'Eposide': 'Episode'}, inplace=True)

# 2. Tính trung bình trượt (Moving Average) với cửa sổ = 20 tập
window_size = 20
df['Moving_Loss'] = df['Loss'].rolling(window=window_size, min_periods=1).mean()

# 3. Vẽ biểu đồ
plt.figure(figsize=(10, 6))

# Vẽ đường Loss gốc (màu nhạt để làm nền)
plt.plot(df['Episode'], df['Loss'], alpha=0.3, color='orange', label='Loss từng tập (Gốc)')

# Vẽ đường Trung bình trượt (màu đậm để nhìn rõ xu hướng)
plt.plot(df['Episode'], df['Moving_Loss'], color='red', linewidth=2, label=f'Loss Trung bình trượt ({window_size} tập)')

# Trang trí biểu đồ
plt.title('Biểu đồ Loss qua các tập huấn luyện (DQN)')
plt.xlabel('Tập (Episode)')
plt.ylabel('Giá trị Loss')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Hiển thị
plt.ylim(-1, 25) 
plt.yticks(np.arange(0, 26, 5))
plt.tight_layout()
plt.show()