# Architecture Document — Cơ chế trí nhớ đa phương thức cho mô hình VLA trong điều khiển Robot qua bộ nhớ trạng thái.

## System Overview

Hệ thống hoạt động như một lớp Middleware (Phần mềm trung gian) nằm giữa đầu vào (Camera/Cảm biến/Prompt) và Mô hình VLA (OpenVLA, $\pi_0$, etc.). Nhằm giải quyết các thách thức lớn trong điều khiển robot với các tác vụ dài:
1. **Thiếu nhận thức thời gian (Temporal Awareness)** trong các tác vụ dài (Long-horizon tasks).
2. **Hiện tượng che khuất vật thể (Occlusion)** khi tay kẹp robot thao tác gần hoặc tiếp xúc trực tiếp với mục tiêu.

Hệ thống kết hợp hai cơ chế độc lập chạy ở các tần số khác nhau: **The Chronicler** (Memory-as-a-Prompt) và **The Spatial Tracker** (Persistent Visual Tracking Overlay), giúp robot duy trì bộ nhớ trạng thái đa phương thức hiệu quả mà không cần thay đổi cấu trúc hay kiến trúc gốc của mô hình VLA.

---

## Architecture Diagram

```mermaid
graph TD
    classDef memory fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef vision fill:#efebe9,stroke:#3e2723,stroke-width:2px;
    classDef sync fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px;
    classDef model fill:#fff3e0,stroke:#e65100,stroke-width:2px;

    Command[Human Command/Instruction] --> |"Put apple in drawer"| Sync[Synchronization Layer / Buffer]
    Camera[Camera RGB Stream] --> Chronicler[The Chronicler<br>Memory-as-a-Prompt<br>1-2 Hz]
    Camera --> Tracker[The Spatial Tracker<br>Persistent Visual Tracking<br>10-20 Hz]
    Proprioception[Robot Proprioception<br>Gripper State] --> Chronicler
    
    Chronicler --> |"Dynamic Prompt State<br>[Task] ... | [Status] ..."| Sync
    Tracker --> |Paint: Green dots for visible<br>Red dots for occluded| Painter[The Painter Filter]
    Painter --> |Painted Image RGB| Sync
    
    Sync --> |Painted Image + Dynamic Prompt| VLA[VLA Model<br>OpenVLA / pi_0 / RT-X<br>10 Hz]
    VLA --> |Action Trajectory| Robot[Robot Controller]

    style Chronicler class:memory
    style Tracker class:vision
    style Sync class:sync
    style VLA class:model
```

---

## Component Descriptions

### 1. The Chronicler (Cơ chế 1 - Memory-as-a-Prompt)
(`src/chronicler.py`)
- **Nhiệm vụ**: Duy trì "nhận thức thời gian" (Temporal Awareness) và trạng thái logic của robot mà không bắt VLA phải tự nhớ toàn bộ lịch sử.
- **Tần số hoạt động (Tick Rate)**: 5 - 10 Hz (Rất nhẹ, chạy ngầm, không gây thắt cổ chai cho hệ thống control).
- **Đầu vào (Inputs)**:
  - Frames từ Camera (định kỳ).
  - Lệnh từ Human Command.
  - Dữ liệu Proprioception (Đặc biệt là độ mở của tay kẹp - Gripper Width/State).
- **Mô hình cốt lõi**: Một VLM nhỏ, nhạy bén (VD: LLaVA-1.5, Qwen-VL-Chat) kết hợp với Object Detection.
- **Đầu ra (Output)**: Một chuỗi `State_String` tuân thủ template cố định:
  - `[Task]: <Human Command> | [Status]: <Past Action>. <Object States>. <Gripper State>. <Environment States>.`
  - *Ví dụ*: `[Task]: Make fried egg | [Status]: Egg was grasped. Egg is in hand. Gripper is closed. Pan is heated.`
  thử thêm nhiều template 
  có thể dùng nguyên một projecter/adapter/anything.. được finetune chuyên biệt để sinh ra **memory token** tương tự như image token của VLM hay các end of sentence token,...

### 2. The Spatial Tracker (Cơ chế 2 - Persistent Visual Tracking)
(`src/spatial_tracker.py`)
- **Nhiệm vụ**: Cung cấp "nhận thức không gian" (Spatial Awareness) liên tục, giải quyết triệt để vấn đề Occlusion (Che khuất) khi tay kẹp đến gần vật thể.
- **Tần số hoạt động (Tick Rate)**: 10 - 20 Hz (Đồng bộ với Control Frequency của VLA).
- **Mô hình cốt lõi**: TAPIR (DeepMind) hoặc CoTracker (Meta). Các mô hình này có khả năng dự đoán vị trí điểm ảnh khi bị che khuất (Occluded Point Prediction).
- **Cơ chế hoạt động**:
  1. *Initialize (Khởi tạo)*: Khi Human đưa ra lệnh ("Grasp mug"), hệ thống dùng Grounding DINO (hoặc YOLO/SAM) để tìm cái cốc lần đầu tiên. Rải một lưới (grid) khoảng 10 điểm (tracking points) lên chiếc cốc.
  2. *Track (Theo vết)*: TAPIR liên tục track 10 điểm này xuyên suốt video stream. Khi tay kẹp che khuất cốc, TAPIR xuất ra tọa độ dự đoán (Predicted Coordinates) + Cờ che khuất (Occluded Flag).
  3. *The Painter (Bộ lọc vẽ đè - Yếu tố Plug-and-Play quyết định)*: Thay vì thêm channel (vì việc đổi input từ RGB 3 channels sang 4 channels sẽ phá vỡ kiến trúc CNN/ViT của VLA, vi phạm constraint), module Painter sẽ trực tiếp vẽ các điểm lên ảnh RGB:
     - Điểm nhìn thấy (Visible): Vẽ chấm màu xanh lá.
     - Điểm bị che khuất (Occluded): Vẽ chấm màu đỏ (hoặc vẽ bounding box nét đứt).

### 3. Synchronization Layer (Quản lý Đồng bộ)
(`src/sync_layer.py`)
- **Nhiệm vụ**: Điều phối và đồng bộ các tiến trình bất đồng bộ chạy ở các tần số khác nhau thông qua bộ đệm (Asynchronous Buffer).
- **Chi tiết luồng xử lý**:
  - *Thread 1 (Tracker Pipeline)*: Chạy ở ~ 10Hz. Liên tục nhận ảnh RGB $\rightarrow$ Track $\rightarrow$ Vẽ đè (Paint) $\rightarrow$ Đẩy `Painted_Image` vào Buffer.
  - *Thread 2 (Chronicler Pipeline)*: Chạy ở 5 - 10Hz. Liên tục phân tích trạng thái $\rightarrow$ Đẩy `Dynamic_Prompt` vào Buffer.
  - *Thread 3 (VLA Engine)*: Lặp ở 10Hz. Ở mỗi step, VLA lấy `Painted_Image` mới nhất và `Dynamic_Prompt` mới nhất từ Buffer $\rightarrow$ Suy luận $\rightarrow$ Xuất Action Chunk $\rightarrow$ Gửi xuống Robot.

### 4. Evaluattion (Giai đoạn Tích hợp & Đánh giá)
- **Quy trình Alignment (Lightweight Finetuning)**:
  Mặc dù kiến trúc VLA không bị thay đổi, nhưng để VLA hoạt động tốt với các chấm xanh/đỏ (Cơ chế 2) và tuân lệnh cấu trúc Text mới (Cơ chế 1), chúng ta cần một bước Alignment (Căn chỉnh). Bước này chỉ mất vài giờ trên 1 GPU RTX 4090/A6000:
  - *Data Preparation*: Lấy một tập dataset hành động robot có sẵn (VD: Open X-Embodiment dataset).
  - *Data Processing (Tự động hoàn toàn)*:
    1. Chạy Tracker qua các video để "vẽ" chấm xanh/đỏ lên mục tiêu.
    2. Dùng Heuristics (độ mở tay kẹp) để tự động sinh ra các đoạn Text `[Status]: Object is IN_HAND`.
  - *LoRA Fine-tuning*: Finetune mô hình VLA (VD: OpenVLA) bằng phương pháp LoRA (Low-Rank Adaptation). Không thay đổi trọng số gốc của VLA, chỉ dạy cho cơ chế Attention của nó biết rằng: "Hãy chú ý vào các chấm màu đỏ/xanh trên ảnh, và tin tưởng tuyệt đối vào thông tin IN_HAND trong prompt".
- **Giá trị Thực tiễn (Value Proposition)**:
  - *Tính Decoupling (Tách bạch) tuyệt đối*: Cho phép tháo lắp và cắm các mô hình VLA mới hơn (ví dụ: RT-3 hoặc $\pi_1$) mà không cần viết lại mã nguồn quản lý bộ nhớ.
  - *Độ tin cậy cao cho Handheld Occlusion*: Khi tay robot che khuất $>80\%$ vật thể, robot vẫn hoàn thành quỹ đạo nhờ thông tin nội suy tọa độ và chỉ dẫn trạng thái logic.
  - *Tối ưu tài nguyên*: Chuyển giao Long-horizon logic sang VLM nhỏ hơn chạy ở tần số thấp (1Hz), giúp giải phóng tài nguyên tính toán cho mô hình VLA chuyên trách phản ứng tốc độ cao (10Hz).

---

## Tech Stack

| Layer | Technology |
|---|---|
| **VLA Controllers & Models** | OpenVLA/$\pi_0$/RT-X |
| **Object Grounding & Segmentation** | Grounding DINO, YOLO, Segment Anything (SAM) |
| **Point Tracking** | TAPIR (DeepMind), CoTracker (Meta) |
| **State VLM** | LLaVA-1.5-7B, Qwen-VL-Chat |
| **Finetuning & Optimization** | PyTorch, PEFT / LoRA (Low-Rank Adaptation) |
| **System Orchestration** | ROS2, Docker (WSL2), Python Multiprocessing, Asynchronous Buffering |
| **Computing Resource** | GPU RTX 4090 / NVIDIA A6000 |

---

## Folder Structure

```
project_root/
├── configs/                     # Cấu hình cho Tracker, Chronicler và VLA
├── scripts/                     # Kịch bản huấn luyện LoRA và tiền xử lý dữ liệu
├── src/
│   ├── chronicler.py            # Memory-as-a-Prompt
│   ├── spatial_tracker.py       # Persistent Visual Tracking
│   ├── painter.py               # Bộ lọc vẽ đè chấm trạng thái trực tiếp lên ảnh RGB
│   ├── sync_layer.py            # Synchronization Layer (Asynchronous Buffer)
│   └── main.py                  # Entrypoint điều khiển robot chính
├── tests/                       # Các ca kiểm thử (Unit / Integration Tests)
├── Dockerfile                   # Dockerfile cấu hình môi trường ROS 2 + CUDA + PyTorch
├── docker-compose.yml           # Docker Compose cấu hình GPU và mạng ROS 2
└── README.md                    # Hướng dẫn triển khai chi tiết
```
