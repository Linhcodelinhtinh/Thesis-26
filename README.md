# Robotic Grasp Planning Architecture

Cơ chế trí nhớ đa phương thức cho mô hình VLA trong điều khiển Robot qua bộ nhớ trạng thái.

## Setup & Deployment (Docker + WSL2)

Dự án sử dụng Docker để đồng bộ hóa môi trường chạy ROS 2 Humble, CUDA 12.1, và PyTorch.

### Prerequisites
1. **Windows Subsystem for Linux (WSL2)** cài đặt Ubuntu.
2. **Docker Desktop** đã bật tính năng tích hợp WSL2.
3. **NVIDIA Container Toolkit** đã được cài đặt trên host Windows & WSL2 để container truy cập GPU.

### Build and Run Docker Container

1. **Khởi chạy container và build image:**
   ```bash
   docker-compose up --build -d
   ```

2. **Truy cập vào container:**
   ```bash
   docker exec -it robot_memory_container bash
   ```

3. **Chạy ứng dụng chính bên trong container:**
   ```bash
   python3 src/main.py
   ```

## Project Structure
Xem chi tiết tại [architecture.md](architecture.md).
