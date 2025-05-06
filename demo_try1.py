"""
只切换频率来计算时延
"""


import math
import heapq

# 常量定义
c = 3 * 10**8  # 光速 (m/s)
packet_size = 1  # 数据包大小 (MB)  
L = packet_size * 8 * 10**6  # bits
P_tx = 20  # dBm
P_rx = -90  # dBm
k = 1.38e-23  # Boltzmann constant
T = 290  # noise temperature in Kelvin

SNR_dB = 20
static_SNR = 10 ** (SNR_dB / 10)
max_delay_threshold = 0.05  # 例如：200 毫秒

# 节点坐标
nodes = [
    (0, 0),
    (1000, 0),
    (2000, 0),
    (1000, 1000),
    (3000, 500),
    (2500, 1500)
]

n = len(nodes)

# 距离计算
def distance(i, j):
    xi, yi = nodes[i]
    xj, yj = nodes[j]
    return math.hypot(xi - xj, yi - yj)

# 最大通信距离（dBm）
def compute_max_range(P_tx, P_rx, f):
    FSPL_max = P_tx - P_rx
    f_mhz = f / 1e6
    d_km = 10 ** ((FSPL_max - 32.44 - 20 * math.log10(f_mhz)) / 20)
    return d_km * 1000

def propagation_delay(d):
    return d / c

def transmission_delay(L, B, f, d):
    # 动态计算snr
    f_mhz = f / 1e6
    d_km = d / 1000
    FSPL_dB = 32.44 + 20 * math.log10(f_mhz) + 20 * math.log10(d_km)
    Pr_dBm = P_tx - FSPL_dB
    noise_power = k * T * B  # in watts
    Pr_watts = 10 ** ((Pr_dBm - 30) / 10)
    SNR = Pr_watts / noise_power
    if SNR <= 0:
        print("信号过弱，无法通信")
        return float('inf')
    capacity = B * math.log2(1 + SNR)
    return L / capacity

def transmission_static_delay(L, B, SNR):
    capacity = B * math.log2(1 + SNR)
    return L / capacity

def dijkstra(start, graph):
    dist = [float('inf')] * n
    dist[start] = 0
    heap = [(0, start)]
    while heap:
        t, u = heapq.heappop(heap)
        if t > dist[u]:
            continue
        for v, delay in graph[u]:
            if dist[u] + delay < dist[v]:
                dist[v] = dist[u] + delay
                heapq.heappush(heap, (dist[v], v))
    return dist

# 枚举频率，寻找最小最大时延
best_f = None
min_max_delay = float('inf')
best_delays = []

for f in range(100_000_000, 6_100_000_000, 200_000_000):  # 2 GHz 到 6 GHz，步长 100 MHz
    B = 0.1 * f
    max_range = compute_max_range(P_tx, P_rx, f)
    
    # 构建图
    graph = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                d = distance(i, j)
                if d <= max_range:
                    delay = propagation_delay(d) + transmission_delay(L, B, f, d)
                    # delay = propagation_delay(d) + transmission_static_delay(L, B, static_SNR)
                    graph[i].append((j, delay))
    
    # 若图不连通则跳过
    delays = dijkstra(0, graph)
    if float('inf') in delays:
        continue
    
    max_delay = max(delays)
    if max_delay < min_max_delay:
        min_max_delay = max_delay
        best_f = f
        best_delays = delays[:]

# 输出最优频率及对应延迟
if best_f:
    print(f"最优频率: {best_f / 1e9:.2f} GHz")
    print(f"最小最大时延: {min_max_delay:.6f} 秒")
    for i, t in enumerate(best_delays):
        print(f"节点0 到 节点{i} 的时延: {t * 1000:.3f} 毫秒")
else:
    print("未找到可用频率使网络连通")
