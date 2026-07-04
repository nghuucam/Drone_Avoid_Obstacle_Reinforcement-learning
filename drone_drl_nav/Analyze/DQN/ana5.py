import pandas as pd

def analyze_flight_log(csv_file):
    try:
        # Đọc dữ liệu từ file CSV
        df = pd.read_csv(csv_file)
        
        # Lấy dòng cuối cùng của mỗi màn chơi (Lưu ý: cột trong file của bạn tên là 'Eposide')
        # keep='last' giúp ta lấy trạng thái chốt hạ của màn chơi đó
        episodes = df.drop_duplicates(subset=['Eposide'], keep='last')
        
        # 1. Thống kê tổng quan (Ép kiểu về chuỗi để so sánh an toàn)
        total_episodes = len(episodes)
        wins = (episodes['Win'].astype(str) == '1').sum()
        losses = total_episodes - wins
        
        # 2. Bóc tách nguyên nhân thua
        collisions = (episodes['Collision'].astype(str) == '1').sum()
        over_maps = (episodes['Over_Map'].astype(str) == '1').sum()
        over_steps = (episodes['Over_Step'].astype(str) == '1').sum()
        
        # 3. In báo cáo ra màn hình
        print("=" * 55)
        print(f"📊 BÁO CÁO CHI TIẾT TỪ FILE: {csv_file}")
        print("=" * 55)
        print(f"Tổng số màn chơi đã test : {total_episodes}")
        print(f"🏆 Số lần THẮNG (Tới đích): {wins} ({(wins/total_episodes)*100:.1f}%)")
        print(f"💥 Tổng số lần THUA      : {losses} ({(losses/total_episodes)*100:.1f}%)")
        print("-" * 55)
        print("Chi tiết nguyên nhân thất bại:")
        print(f"  - Thua do va chạm vật cản   : {collisions} lần")
        print(f"  - Thua do bay ra ngoài map  : {over_maps} lần")
        print(f"  - Thua do hết lượt di chuyển: {over_steps} lần")
        print("=" * 55)
        
    except FileNotFoundError:
        print(f"❌ Lỗi: Không tìm thấy file '{csv_file}'. Vui lòng để chung thư mục hoặc điền lại đường dẫn.")
    except Exception as e:
        print(f"❌ Có lỗi xảy ra trong quá trình đọc file: {e}")

if __name__ == "__main__":
    # Điền tên file CSV bạn muốn phân tích vào đây
    # Ví dụ thống kê cho cả 2 thuật toán:
    
    print("\n--- ĐÁNH GIÁ MÔ HÌNH DQN ---")
    analyze_flight_log('drone_flight_log_test_DQN.csv')