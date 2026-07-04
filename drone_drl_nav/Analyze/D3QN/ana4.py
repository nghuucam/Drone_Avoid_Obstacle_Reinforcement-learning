import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_moving_average(csv_file, window_size=50):
    try:
        # 1. Đọc dữ liệu từ file log
        print(f"Đang đọc dữ liệu từ {csv_file}...")
        df = pd.read_csv(csv_file)
        
        # 2. Lấy kết quả tổng kết của từng màn (Episode)
        # Vì bạn ghi log ở mỗi bước, dòng cuối cùng của mỗi Eposide sẽ chứa 
        # số bước tối đa và tổng điểm (AccumReward) cuối cùng.
        episodes_data = df.drop_duplicates(subset=['Eposide'], keep='last').copy()
        
        # 3. Tính toán Cửa sổ trượt (Moving Average)
        # Tính trung bình điểm số và số bước trong 'window_size' màn gần nhất
        episodes_data['MA_Reward'] = episodes_data['AccumReward'].rolling(window=window_size, min_periods=1).mean()
        episodes_data['MA_Steps'] = episodes_data['Step'].rolling(window=window_size, min_periods=1).mean()
        
        # 4. Vẽ biểu đồ
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle(f'Tiến trình Huấn luyện D3QN (Trung bình trượt {window_size} màn)', fontsize=16, fontweight='bold')
        
        # --- Biểu đồ 1: Điểm thưởng (Reward) ---
        # Điểm gốc mờ mờ ở nền
        ax1.plot(episodes_data['Eposide'], episodes_data['AccumReward'], color='lightgray', alpha=0.6, label='Điểm từng màn (Gốc)')
        # Đường trung bình đậm nét
        ax1.plot(episodes_data['Eposide'], episodes_data['MA_Reward'], color='blue', linewidth=2, label=f'Điểm trung bình ({window_size} màn)')
        ax1.set_ylabel('Điểm thưởng (Reward)')
        ax1.set_title('Sự tăng trưởng Điểm số')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()
        
        # --- Biểu đồ 2: Số bước bay (Steps) ---
        # Số bước gốc
        ax2.plot(episodes_data['Eposide'], episodes_data['Step'], color='mistyrose', alpha=0.6, label='Số bước từng màn')
        # Đường trung bình đậm nét
        ax2.plot(episodes_data['Eposide'], episodes_data['MA_Steps'], color='red', linewidth=2, label=f'Số bước trung bình ({window_size} màn)')
        ax2.set_xlabel('Màn chơi (Episode)')
        ax2.set_ylabel('Số bước bay (Steps)')
        ax2.set_title('Khả năng sống sót / Tối ưu đường đi')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print(f"❌ Không tìm thấy file {csv_file}. Hãy đảm bảo tên file chính xác và nằm cùng thư mục.")
    except Exception as e:
        print(f"❌ Có lỗi xảy ra: {e}")

if __name__ == "__main__":
    # Thay tên file CSV của bạn vào đây nếu cần
    LOG_FILE = "drone_flight_log_test_D3QN.csv" 
    
    # Cài đặt cửa sổ trượt (window_size). 
    # Ví dụ: 50 nghĩa là mỗi điểm trên đường nét đậm là trung bình cộng của 50 màn trước đó.
    plot_moving_average(LOG_FILE, window_size=100)