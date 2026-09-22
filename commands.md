Dưới đây là tổng hợp các câu lệnh có thể chạy (runnable commands) của dự án hiện tại, được phân loại theo mục đích sử dụng:

1. Đánh giá Mô hình VLA (SimplerEnv Policy Evaluation)
Chạy vòng lặp điều khiển closed-loop với mô hình VLA (Octo, OpenVLA, Mock) kết hợp Action Chunking và cơ chế Memory-as-a-Prompt (Chronicler):

bash
# Đánh giá mặc định (Octo-Small, task Pick Coke Can, Action Chunk = 4)
python src/main.py --task google_robot_pick_coke_can --mode policy --chunk 4
# Đánh giá tác vụ mở ngăn kéo (Google Robot Open Drawer)
python src/main.py --task google_robot_open_drawer --mode policy --max-steps 60
# Đánh giá trên robot WidowX (WidowX Put Eggplant in Basket)
python src/main.py --task widowx_put_eggplant_in_basket --mode policy --chunk 4
# Đánh giá với seed ngẫu nhiên khác và tắt ghi trajectory
python src/main.py --task google_robot_pick_coke_can --seed 123 --no-record
2. Thu thập Dữ liệu Trajectory & Điều khiển (Teleop / Expert)
Tạo và ghi lại tập dữ liệu trajectory ($\text{obs} \to \text{action} \to \text{reward/success}$) xuất ra các file .json, .npz và .mp4:

bash
# Thu demo tự động từ Scripted Oracle Expert (Tự động thành công 100%)
python src/main.py --task google_robot_pick_coke_can --mode expert --max-steps 35
# Thu demo tự động tác vụ mở ngăn kéo
python src/main.py --task google_robot_open_drawer --mode expert
# Điều khiển thủ công bằng bàn phím (W/S/A/D/R/F/Q/E/Space)
python src/main.py --task google_robot_pick_coke_can --mode teleop
Hướng dẫn phím bấm Teleop:

W / S: Di chuyển +X (Tới) / -X (Lùi)
A / D: Di chuyển +Y (Trái) / -Y (Phải)
R / F: Nâng +Z (Lên) / -Z (Xuống)
Q / E: Xoay Yaw (-Yaw / +Yaw)
Space: Đóng / Mở tay kẹp (Gripper)
R: Reset lại episode
Q / Esc: Thoát
3. Tái lập Thực nghiệm (Deterministic Trajectory Replay)
Tái hiện lại chính xác chuỗi hành động đã lưu từ file trajectory để kiểm chứng sai số Cartesian drift và kết quả success:

bash
# Replay và hiển thị cửa sổ hình ảnh trực quan
python scripts/replay_trajectory.py --file data/trajectories/<tệp_trajectory>.json --visualize
# Chạy replay tự động sinh demo để kiểm thử độ lệch (Zero-Drift Check)
python scripts/replay_trajectory.py --tolerance 0.005
4. Giao diện Tương tác Chính (Franka Panda MuJoCo VLA Workspace)
Khởi chạy toàn bộ nền tảng mô phỏng MuJoCo, stream camera kép thời gian thực (Top Cam + Wrist Cam), nhận lệnh Chat Console và điều khiển VLA trực tiếp:

bash
# Khởi chạy mặc định (Mở Web Workspace tại cổng 8080)
python run.py
# Khởi chạy kèm tham số tác vụ và mô hình VLA
python run.py --task "put can in basket" --model octo --port 8080
# Hoặc chạy trực tiếp qua web_server.py
python src/web_server.py --port 8080
👉 Truy cập giao diện trình duyệt tại: http://localhost:8080

5. Chạy Kiểm thử Tự động (Unit & Integration Tests)
Chạy bộ test tự động xác thực SimplerEnv Adapter, Action Chunking, Trajectory Recorder, và Replay Engine:

bash
# Chạy bộ test SimplerEnv Platform & Replay Engine
python -m unittest tests/test_simpler_env.py -v
# Chạy bộ test VLA Middleware (Chronicler, Tracker, SyncLayer)
python -m unittest tests/test_vla.py -v
# Chạy thử nghiệm độc lập từng module core
python src/simpler_env_adapter.py
python src/trajectory_recorder.py
python src/teleop_controller.py
6. Cài đặt Môi trường & Docker
bash
# Cài đặt các thư viện Python cơ bản
pip install -r requirements.txt
# Cài đặt Native SimplerEnv (Khuyên dùng trên Linux / WSL2 GPU)
pip install mani-skill2==0.5.3 sapien==2.2.2
pip install git+https://github.com/simpler-env/SimplerEnv.git
python -m mani_skill2.utils.download_asset all
# Build & Khởi chạy Docker Container
docker-compose up --build -d
docker exec -it robot_memory_container bash
