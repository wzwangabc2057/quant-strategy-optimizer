#!/usr/bin/env python3
"""
API 服务管理脚本
支持 start / stop / restart / status
"""
import os
import sys
import subprocess
import signal
import time

PID_FILE = "/tmp/quant_api.pid"
LOG_FILE = "/tmp/quant_api.log"
DEFAULT_PORT = 8000

def get_pid():
    """获取进程 PID"""
    if os.path.exists(PID_FILE):
        with open(PID_FILE, 'r') as f:
            try:
                return int(f.read().strip())
            except:
                return None
    return None

def is_running():
    """检查服务是否运行"""
    pid = get_pid()
    if pid:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    return False

def start(port=DEFAULT_PORT):
    """启动服务"""
    if is_running():
        print(f"服务已在运行中 (PID: {get_pid()})")
        return

    print(f"启动 API 服务 (端口: {port})...")

    # 切换到项目目录
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)

    # 激活虚拟环境
    venv_python = os.path.join(project_dir, '.venv', 'bin', 'python')
    if not os.path.exists(venv_python):
        venv_python = sys.executable

    # 启动命令
    cmd = [
        venv_python, '-m', 'uvicorn',
        'api.server:app',
        '--host', '0.0.0.0',
        '--port', str(port),
        '--log-level', 'info'
    ]

    # 后台启动
    with open(LOG_FILE, 'w') as log:
        process = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=log,
            start_new_session=True
        )

    # 保存 PID
    with open(PID_FILE, 'w') as f:
        f.write(str(process.pid))

    # 等待启动
    time.sleep(2)

    if is_running():
        print(f"✅ 服务启动成功!")
        print(f"   PID: {process.pid}")
        print(f"   API: http://0.0.0.0:{port}")
        print(f"   文档: http://0.0.0.0:{port}/docs")
        print(f"   日志: {LOG_FILE}")
    else:
        print("❌ 服务启动失败，请检查日志")
        print(f"   日志: {LOG_FILE}")

def stop():
    """停止服务"""
    pid = get_pid()
    if not pid:
        print("服务未运行")
        return

    print(f"停止服务 (PID: {pid})...")

    try:
        os.kill(pid, signal.SIGTERM)
        time.sleep(2)

        # 检查是否停止
        try:
            os.kill(pid, 0)
            # 还在运行，强制杀死
            os.kill(pid, signal.SIGKILL)
            print("⚠️ 强制终止服务")
        except OSError:
            print("✅ 服务已停止")

        # 删除 PID 文件
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)

    except OSError:
        print("服务未运行")
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)

def restart(port=DEFAULT_PORT):
    """重启服务"""
    print("重启服务...")
    stop()
    time.sleep(1)
    start(port)

def status():
    """查看状态"""
    pid = get_pid()
    if pid and is_running():
        print(f"✅ 服务运行中 (PID: {pid})")
        print(f"   API: http://0.0.0.0:{DEFAULT_PORT}")
        print(f"   日志: {LOG_FILE}")

        # 显示最近日志
        if os.path.exists(LOG_FILE):
            print("\n最近日志:")
            os.system(f"tail -5 {LOG_FILE}")
    else:
        print("❌ 服务未运行")

def logs():
    """查看日志"""
    if os.path.exists(LOG_FILE):
        os.system(f"tail -50 {LOG_FILE}")
    else:
        print("日志文件不存在")

def main():
    if len(sys.argv) < 2:
        print("用法: python api_service.py {start|stop|restart|status|logs} [port]")
        print("")
        print("命令:")
        print("  start [port]  - 启动服务 (默认端口 8000)")
        print("  stop          - 停止服务")
        print("  restart [port]- 重启服务")
        print("  status        - 查看状态")
        print("  logs          - 查看日志")
        sys.exit(1)

    command = sys.argv[1]
    port = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_PORT

    if command == 'start':
        start(port)
    elif command == 'stop':
        stop()
    elif command == 'restart':
        restart(port)
    elif command == 'status':
        status()
    elif command == 'logs':
        logs()
    else:
        print(f"未知命令: {command}")
        sys.exit(1)

if __name__ == '__main__':
    main()
