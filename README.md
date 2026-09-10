# Robotic Grasp Planning & SimplerEnv Evaluation Platform

Cơ chế trí nhớ đa phương thức cho mô hình VLA (Octo, RT-1, OpenVLA) trong điều khiển Robot qua bộ nhớ trạng thái, tích hợp nền tảng mô phỏng **SimplerEnv** (ManiSkill2 / SAPIEN) để đánh giá zero-shot sim-to-real cho robot đời thực.

---

## 🚀 Các Tính Năng Nổi Bật

1. **Nền tảng Đánh giá Chuẩn SimplerEnv**:
   - Hỗ trợ các tác vụ chuẩn hoá: `google_robot_pick_coke_can`, `google_robot_open_drawer`, `widowx_put_eggplant_in_basket`.
   - Giữ nguyên vẹn quan sát RGB/RGB-D từ camera thật (300x300 / 256x256), ma trận camera intrinsics ($K$) và extrinsics ($T$).
   - Đồng nhất Action Space (7-DoF delta Cartesian) và Kinematics với các mô hình VLA huấn luyện trên Fractal / Bridge V2.
   - Ground-truth **Success Condition Evaluator** xác thực điều kiện gắp/nâng/mở ngăn kéo khách quan.

2. **Cơ chế Trí nhớ Đa Phương Thức (Middleware)**:
   - **The Chronicler** (Memory-as-a-Prompt): Duy trì nhận thức thời gian (temporal awareness) và logic trạng thái qua dynamic prompt.
   - **The Spatial Tracker & Painter**: Theo vết persistent tracking points dưới điều kiện che khuất (occlusion).
   - **Action Chunking**: Hỗ trợ horizon $H=4, 8$ giúp robot thao tác mượt mà ở tần số cao.

3. **Thu thập Trajectory & Deterministic Replay**:
   - Ghi nhận đầy đủ: Observation $\to$ Action $\to$ Reward/Success $\to$ Metadata (Seed, Timestamp, Calibration, Config Hash).
   - Xuất dữ liệu đa định dạng: JSONL, NumPy Compressed (`.npz`), và Video rollout (`.mp4`).
   - Tái lập thực nghiệm (Deterministic Replay) với sai số Cartesian drift $\le 10^{-4}\text{ m}$.

4. **Nguồn Dữ liệu Đa dạng & Teleoperation**:
   - Điều khiển bàn phím (Keyboard Teleop) trực tiếp trên giao diện.
   - **Scripted Oracle Expert Policy**: Tự động sinh trajectory mẫu thành công phục vụ training và benchmark.
   - Thông báo nổi bật **🏆 TASK SUCCESSFUL** và **❌ TASK FAILED** ngay trên OpenCV Dashboard và Web UI.

---

## 🛠️ Hướng Dẫn Cài Đặt (Native SimplerEnv & Dependencies)

### Cài đặt thư viện Python cơ bản:
```bash
pip install -r requirements.txt
```

### Cài đặt Native SimplerEnv (Khuyến nghị trên Linux / WSL2 có GPU NVIDIA):
```bash
# 1. Cài đặt ManiSkill2 & SAPIEN
pip install mani-skill2==0.5.3 sapien==2.2.2

# 2. Cài đặt SimplerEnv từ source
pip install git+https://github.com/simpler-env/SimplerEnv.git

# 3. Tải assets 3D cho Google Robot & WidowX
python -m mani_skill2.utils.download_asset all
```

> **Lưu ý**: Khi chạy trên Windows hoặc môi trường chưa cài đặt SAPIEN/Vulkan, hệ thống sẽ tự động kích hoạt **High-Fidelity Standalone Emulation Mode** trong `src/simpler_env_adapter.py` để đảm bảo 100% API, Observation, Action Space, và Success Condition tương thích mà không bị phụ thuộc driver.

---

## 💻 Hướng Dẫn Sử Dụng

### 1. Đánh giá Chính sách VLA (Octo / OpenVLA / Mock):
```bash
# Chạy đánh giá task gắp lon nước ngọt với Octo + Action Chunking
python src/main.py --task google_robot_pick_coke_can --mode policy --chunk 4

# Chạy đánh giá task mở ngăn kéo
python src/main.py --task google_robot_open_drawer --mode policy
```

### 2. Thu thập Trajectory từ Oracle Expert:
```bash
# Sinh trajectory mẫu thành công và lưu vào data/trajectories/
python src/main.py --task google_robot_pick_coke_can --mode expert --max-steps 35
```

### 3. Điều khiển Thủ công (Teleoperation):
```bash
# Điều khiển bằng bàn phím (W/S/A/D: di chuyển XY, R/F: nâng hạ Z, Space: kẹp/nhả)
python src/main.py --task google_robot_pick_coke_can --mode teleop
```

### 4. Tái lập Thực nghiệm (Deterministic Replay):
```bash
# Chạy replay và kiểm chứng sai số trạng thái
python scripts/replay_trajectory.py --file data/trajectories/<tệp_trajectory>.json --visualize
```

### 5. Giao diện Web Visualization & Digital Twin:
```bash
# Khởi chạy Web Server tại cổng 8080
python src/main.py --web --port 8080
# Mở trình duyệt tại: http://localhost:8080
```

### 6. Chạy Kiểm thử Tự động (Unit & Integration Tests):
```bash
python -m unittest tests/test_simpler_env.py -v
python -m unittest tests/test_vla.py -v
```
