🚀 Lệnh thực thi nhanh:
Chạy đánh giá VLA Octo với SimplerEnv:
bash
python src/main.py --task google_robot_pick_coke_can --mode policy --chunk 4
Thu thập trajectory mẫu từ Oracle Expert:
bash
python src/main.py --task google_robot_pick_coke_can --mode expert --max-steps 35
Khởi chạy Web Visualization & Digital Twin:
bash
python src/main.py --web --port 8080
# Mở trình duyệt tại: http://localhost:8080
