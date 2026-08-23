import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv('log_loss_DQN.csv')

df.rename(columns={'Eposide': 'Episode'}, inplace=True)

window_size = 20
df['Moving_Loss'] = df['Loss'].rolling(window=window_size, min_periods=1).mean()

plt.figure(figsize=(10, 6))

plt.plot(df['Episode'], df['Loss'], alpha=0.3, color='orange', label='Loss từng tập (Gốc)')

plt.plot(df['Episode'], df['Moving_Loss'], color='red', linewidth=2, label=f'Loss Trung bình trượt ({window_size} tập)')

plt.title('Biểu đồ Loss qua các tập huấn luyện (DQN)')
plt.xlabel('Tập (Episode)')
plt.ylabel('Giá trị Loss')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.ylim(-1, 25) 
plt.yticks(np.arange(0, 26, 5))
plt.tight_layout()
plt.show()