"""
最短路径路由计算(Dijkstra 算法）
计算各个节点的时延
"""

import math
import heapq
import argparse

# 添加命令行参数解析
parser = argparse.ArgumentParser(description="最短路径路由计算 (Dijkstra 算法)")
parser.add_argument("--c", type=float, default=3e8, help="光速 (m/s)")
parser.add_argument("--f", type=float, default=5e9, help="频率 (Hz)")
parser.add_argument("--B", type=float, default=0.1 * 5e9, help="带宽 (Hz)")
parser.add_argument("--SNR_dB", type=float, default=20, help="信噪比 (dB)")
parser.add_argument("--packet_size", type=float, default=1, help="数据包大小 (MB)")
parser.add_argument("--P_tx", type=float, default=20, help="发射功率 (dBm)")
parser.add_argument("--P_rx", type=float, default=-90, help="接收灵敏度 (dBm)")
# 添加默认坐标列表
parser.add_argument("--nodes", type=str, default="[(0,0),(1000,0),(2000,0),(1000,1000),(3000,500),(2500,1500)]", help="节点的坐标列表，格式为 [(x1,y1),(x2,y2),...]，单位：米")

args = parser.parse_args()

# 使用命令行参数
c = args.c
f = args.f
B = args.B
SNR_dB = args.SNR_dB
SNR = 10 ** (SNR_dB / 10)
packet_size = args.packet_size
L = packet_size * 8 * 10**6  # 数据包大小 1 MB in bits
P_tx = args.P_tx
P_rx = args.P_rx

# 使用命令行参数解析节点
nodes = eval(args.nodes)  # 将字符串解析为列表
n = len(nodes)

# 计算最大通信距离（自由空间路径损耗模型）
def compute_max_range(P_tx, P_rx, f):
    FSPL_max = P_tx - P_rx  # 最大可容忍的路径损耗（单位 dB）
    f_mhz = f / 1e6  # 转换为 MHz
    # 解路径损耗公式反推距离（单位 km）
    d_km = 10 ** ((FSPL_max - 32.44 - 20 * math.log10(f_mhz)) / 20)
    return d_km * 1000  # 转换为米

# 计算两个节点间距离
def distance(i, j):
    xi, yi = nodes[i]
    xj, yj = nodes[j]
    return math.hypot(xi - xj, yi - yj)

# 传播时延
def propagation_delay(d):
    return d / c

# 传输时延（香农速率）
def transmission_delay(L, B, SNR):
    capacity = B * math.log2(1 + SNR)
    return L / capacity

# Dijkstra 最短路径算法
def dijkstra(start):
    dist = [float('inf')] * n
    dist[start] = 0
    prev = [-1] * n  # 用于存储路径
    heap = [(0, start)]

    while heap:
        t, u = heapq.heappop(heap)
        if t > dist[u]:
            continue
        for v, delay in graph[u]:
            if dist[u] + delay < dist[v]:
                dist[v] = dist[u] + delay
                prev[v] = u  # 记录前驱节点
                heapq.heappush(heap, (dist[v], v))
    return dist, prev


def reconstruct_path(prev, target):
    path = []
    while target != -1:
        path.append(target)
        target = prev[target]
    return path[::-1]  # 反转路径


if __name__ == "__main__":
    max_range = compute_max_range(P_tx, P_rx, f)
    print(f"最大通信距离: {max_range/1000:.2f} km")
    # 构建邻接表（仅在通信范围内）
    graph = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                d = distance(i, j)
                if d <= max_range:
                    t_prop = propagation_delay(d)
                    t_trans = transmission_delay(L, B, SNR)
                    total_delay = t_prop + t_trans
                    graph[i].append((j, total_delay))
    # 执行最短路径计算
    delays, prev = dijkstra(0)

    # 输出结果
    for i, t in enumerate(delays):
        if t < float('inf'):
            path = reconstruct_path(prev, i)
            path_str = " -> ".join(map(str, path))
            print(f"节点0 到 节点{i} 的最短时延: {t * 1000:.3f} 毫秒, 路径: {path_str}")
        else:
            print(f"节点0 到 节点{i} 不可达")

    # 输出结果
    for i, t in enumerate(delays):
        print(f"节点0 到 节点{i} 的最短时延: {t * 1000:.3f} 毫秒")

