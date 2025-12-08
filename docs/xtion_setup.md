# ASUS Xtion ROS2 Humble セットアップガイド

ASUS Xtion Pro LiveをUSB接続し、ROS2 Humbleで使用するためのセットアップ手順です。

## 前提条件

- Ubuntu 22.04
- ROS2 Humble
- ASUS Xtion Pro Live（USB接続）

## セットアップ手順

### 1. OpenNI2ライブラリのインストール

```bash
sudo apt update
sudo apt install libopenni2-dev libopenni2-0
```

### 2. openni2_cameraパッケージのインストール

#### 方法A: aptでインストール（推奨）

```bash
sudo apt install ros-humble-openni2-camera
```

> **Note**: パッケージが見つからない場合は方法Bを使用してください。

#### 方法B: ソースからビルド

```bash
# ROS2ワークスペースに移動
cd ~/ros2_ws/src

# openni2_cameraをクローン（ironブランチがHumbleをサポート）
git clone -b iron https://github.com/ros-drivers/openni2_camera.git

# ビルド
cd ~/ros2_ws
colcon build --packages-select openni2_camera

# 環境をソース
source install/setup.bash
```

### 3. udevルールの設定

Xtionデバイスへのアクセス権限を設定します。

```bash
# ユーザーをvideoグループに追加
sudo usermod -aG video $USER

# udevルールファイルを作成
sudo tee /etc/udev/rules.d/55-primesense.rules << 'EOF'
# PrimeSense / ASUS Xtion
SUBSYSTEM=="usb", ATTR{idVendor}=="1d27", MODE="0666"
SUBSYSTEM=="usb", ATTR{idVendor}=="0409", ATTR{idProduct}=="005e", MODE="0666"
EOF

# udevルールを再読み込み
sudo udevadm control --reload-rules
sudo udevadm trigger

# ログアウト・ログインして権限を反映
```

### 4. Xtionの接続確認

```bash
# USBデバイスの確認
lsusb | grep -i -E "asus|primesense|1d27"

# OpenNI2でデバイス確認
NiViewer  # OpenNI2のビューワー（オプション）
```

正常に接続されている場合、以下のような出力が表示されます：
```
Bus 00X Device 00X: ID 1d27:0601 ASUS Xtion Pro
```

### 5. カメラの起動

```bash
# ROS2環境をソース
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash  # ソースビルドの場合

# カメラノードを起動
ros2 launch openni2_camera camera_only.launch.py
```

PointCloud2も取得する場合：
```bash
ros2 launch openni2_camera camera_with_cloud.launch.py
```

### 6. 発行されるトピック

カメラ起動後、以下のトピックが発行されます：

| トピック | 型 | 説明 |
|---------|-----|------|
| `/camera/rgb/image_raw` | sensor_msgs/Image | RGBカラー画像 |
| `/camera/depth/image_raw` | sensor_msgs/Image | 深度画像 |
| `/camera/ir/image_raw` | sensor_msgs/Image | 赤外線画像 |
| `/camera/rgb/camera_info` | sensor_msgs/CameraInfo | RGBカメラ情報 |
| `/camera/depth/camera_info` | sensor_msgs/CameraInfo | 深度カメラ情報 |

トピック一覧の確認：
```bash
ros2 topic list | grep camera
```

## Data Collectionアプリとの連携

1. Xtionカメラを起動
   ```bash
   ros2 launch openni2_camera camera_only.launch.py
   ```

2. アプリを起動
   ```bash
   streamlit run app/main.py
   ```

3. **Data Collection** → **ROS2 Camera** タブを選択

4. **Image Topic** で `/camera/rgb/image_raw` を選択

5. バースト撮影またはフォルダインポートでデータを収集

## トラブルシューティング

### デバイスが認識されない

```bash
# USB接続を確認
lsusb

# カーネルメッセージを確認
dmesg | tail -20

# デバイスファイルを確認
ls -la /dev/bus/usb/
```

### 権限エラー (Permission denied)

```bash
# 現在のユーザーがvideoグループに属しているか確認
id | grep video

# グループに追加（要ログアウト/ログイン）
sudo usermod -aG video $USER
```

### "No devices found" エラー

1. udevルールが正しく設定されているか確認
2. Xtionを一度抜いて再接続
3. 別のUSBポート（USB 2.0推奨）を試す

### libudev.so.0 が見つからない

```bash
# シンボリックリンクを作成
sudo ln -s /lib/x86_64-linux-gnu/libudev.so.1 /lib/x86_64-linux-gnu/libudev.so.0
```

### ノードが起動しない

```bash
# 依存関係を確認
rosdep install --from-paths ~/ros2_ws/src --ignore-src -r -y

# 再ビルド
cd ~/ros2_ws
colcon build --packages-select openni2_camera --cmake-clean-first
```

### Streamlit起動後にトピックが見えない

**症状**: `streamlit run app/main.py` を先に起動してから Xtion を起動すると、トピックが発行されない・見えない。

**原因**: FastDDS の Shared Memory (SHM) トランスポートが起動順序によってディスカバリに失敗する。

**解決策**: `run_app.sh` を使用してStreamlitを起動することで、FastDDSプロファイルが自動的に適用されます。

```bash
# 推奨: run_app.sh経由で起動（DDS設定済み）
./run_app.sh

# 別ターミナルでXtion起動
ros2 launch openni2_camera camera_only.launch.py
```

**手動設定が必要な場合**:

```bash
# FastDDSのShared Memoryを無効化
export FASTRTPS_DEFAULT_PROFILES_FILE=/path/to/config/fastdds_profile.xml
export ROS_DOMAIN_ID=0

streamlit run app/main.py
```

**確認方法**:

```bash
# 環境変数が設定されているか確認
echo $FASTRTPS_DEFAULT_PROFILES_FILE
echo $ROS_DOMAIN_ID

# トピック一覧
ros2 topic list
```

## 参考リンク

- [openni2_camera (GitHub)](https://github.com/ros-drivers/openni2_camera)
- [libopenni2-dev (Ubuntu Packages)](https://launchpad.net/ubuntu/jammy/+package/libopenni2-dev)
- [OpenNI2 Installation Guide](https://robots.uc3m.es/installation-guides/install-openni-nite.html)
