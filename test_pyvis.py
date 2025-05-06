import os
from pyvis import network as net

# 获取 pyvis 的安装路径
pyvis_install_path = os.path.dirname(net.__file__)
print(f"pyvis 安装路径: {pyvis_install_path}")

# 拼接模板路径
template_path = os.path.join(pyvis_install_path, "templates")
print(f"模板路径: {template_path}")

# 检查模板文件是否存在
template_file = os.path.join(template_path, "network.html")
print(f"模板文件是否存在: {os.path.exists(template_file)}")