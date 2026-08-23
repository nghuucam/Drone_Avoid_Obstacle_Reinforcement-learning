import pandas as pd

def analyze_flight_log(csv_file):
    try:
        df = pd.read_csv(csv_file)
        
        episodes = df.drop_duplicates(subset=['Eposide'], keep='last')
        
        total_episodes = len(episodes)
        wins = (episodes['Win'].astype(str) == '1').sum()
        losses = total_episodes - wins
        
        collisions = (episodes['Collision'].astype(str) == '1').sum()
        over_maps = (episodes['Over_Map'].astype(str) == '1').sum()
        over_steps = (episodes['Over_Step'].astype(str) == '1').sum()
        
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
    print("\n--- ĐÁNH GIÁ MÔ HÌNH DQN ---")
    analyze_flight_log('drone_flight_log_test_D3QN.csv')