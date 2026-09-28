#!/usr/bin/env python3
"""AstraArmController 的零点初始化 (setupTorque(128)) —— 单臂、不自动上力矩的版本。

diff with  examples/02_record_zero.py:
  - only for 1 arm (default: /dev/ttyUSB0)
  - automatically set_torque(1)
  - set_torque(0) when finish

before exec:
  - initial all joint and wrist
  - ready for emergency bottom

utilize:
  uv run init_arm.py [端口]
"""

import os
import sys
import time

REPO = os.path.expanduser("~/aharobot/AhaRobot")
sys.path.insert(0, os.path.join(REPO, "upstream", "astra_controller"))

from astra_controller.arm_controller import ArmController  # noqa: E402

port = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyUSB0"

print("=" * 60)
print("即将执行零点初始化（会写入舵机 EEPROM 与 ESP32 LittleFS）")
print("  - 夹爪完全闭合?")
print("  - joint0/1 摆成 URDF 零点姿态（上臂偏 10.6°，前臂与安装座 x 轴垂直），底座已固定?")
print("  - 手在电源开关旁?")
if input("全部确认后输入 yes 继续: ").strip() != "yes":
    sys.exit("已取消")

ctrl = ArmController(port, do_init=True)  # 发送 set_torque(128) 后立即返回
print(f"\n已发送初始化命令，等待 15 秒，固件输出如下（出现 'Gap is too wide' 或 "
      "'Maybe cause wrong init_pos0' 时立即断电）:\n")
time.sleep(15)

print("\n发送 set_torque(0)，所有舵机脱力")
ctrl.set_torque(0)
time.sleep(1.0)
print("完成。")
