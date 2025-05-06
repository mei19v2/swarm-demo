import math, heapq, optuna, random

# ———— 全局常量 ————
c = 3e8                     # 光速 (m/s)
k = 1.38e-23; T = 290       # 热噪声常数
L = 8e6                     # 1 MB = 8e6 bits
P_rx = -90                  # 接收灵敏度 dBm
T_min = 0.005               # 最小时延约束 (s)
alpha, beta = 1.0, 1.0      # 时延/能耗权重
D_min_est, D_max_est = 0, 0.1  # 预估最大最小延时
E_min_est, E_max_est = 0, 0.5  # 预估最大最小能耗

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

def dist(i, j):
    xi, yi = nodes[i]; xj, yj = nodes[j]
    return math.hypot(xi - xj, yi - yj)

def max_range(Pt_dBm, Pr_dBm, f):
    fspl_max = Pt_dBm - Pr_dBm
    f_mhz = f / 1e6
    d_km = 10 ** ((fspl_max - 32.44 - 20 * math.log10(f_mhz)) / 20)
    return d_km * 1000

def propagation_delay(d):
    return d / c

def tx_delay_and_energy(Pt_dBm, B, f, d):
    # 自由空间路径损耗 (FSPL)
    f_mhz, d_km = f / 1e6, d / 1000
    FSPL = 32.44 + 20 * math.log10(f_mhz) + 20 * math.log10(d_km)
    Pr_dBm = Pt_dBm - FSPL
    Pr_w = 10 ** ((Pr_dBm - 30) / 10)
    # 噪声功率
    noise = k * T * B
    snr = Pr_w / noise
    if snr <= 0:
        return float('inf'), 0.0
    # 信道容量 & 传输时延
    cap = B * math.log2(1 + snr)
    t_tx = L / cap
    # 发射能量
    Pt_w = 10 ** ((Pt_dBm - 30) / 10)
    e_tx = Pt_w * t_tx
    return t_tx, e_tx

def dijkstra_with_pred(start, graph):
    dist_arr = [math.inf] * n
    pred = [-1] * n
    dist_arr[start] = 0
    pq = [(0, start)]
    while pq:
        d_u, u = heapq.heappop(pq)
        if d_u > dist_arr[u]:
            continue
        for v, (delay, energy) in graph[u]:
            nd = dist_arr[u] + delay
            if nd < dist_arr[v]:
                dist_arr[v] = nd
                pred[v] = u
                heapq.heappush(pq, (nd, v))
    return dist_arr, pred

def warm():
    # ———— 1. 预采样：随机采若干组 (f, Pt_list)，统计 D_avg/E_total 的 min/max ————
    warm_up = 100
    D_vals = []
    E_vals = []
    for _ in range(warm_up):
        f = random.uniform(2e9,6e9)
        B = 0.1*f
        Pt_list = [random.uniform(1,100) for _ in range(n)]
        R = max_range(max(Pt_list), P_rx, f)

        # 构建图
        graph = [[] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i==j: continue
                d = dist(i,j)
                if d > R: continue
                t_tx,e_tx = tx_delay_and_energy(Pt_list[i], B, f, d)
                if t_tx == math.inf: continue
                graph[i].append((j,(propagation_delay(d)+t_tx,e_tx)))

        dist_arr, pred = dijkstra_with_pred(0, graph)
        if any(math.isinf(x) for x in dist_arr): 
            continue
        if max(dist_arr)>T_min:
            continue

        D_avg = sum(dist_arr) / (n - 1)
        total_e = 0
        for node in range(n):
            cur=node
            while cur!=0:
                p=pred[cur]
                for v,(_,e_tx) in graph[p]:
                    if v==cur:
                        total_e+=e_tx; break
                cur = p
        E_total = total_e

        D_vals.append(D_avg)
        E_vals.append(E_total)

    # 如果没有任何可行样本，用一个默认范围
    if not D_vals: D_vals = [T_min, T_min*2]
    if not E_vals: E_vals = [0,1]

    D_min_est, D_max_est = min(D_vals), max(D_vals)
    E_min_est, E_max_est = min(E_vals), max(E_vals)

    print(f"预估延时范围: [{D_min_est:.4f}s, {D_max_est:.4f}s]")
    print(f"预估能耗范围: [{E_min_est:.4e}J, {E_max_est:.4e}J]")
    return D_max_est, D_min_est, E_max_est, E_min_est

def objective(trial):
    # 1. 采样频率和各节点发射功率
    f = trial.suggest_float("f", 1e9, 6e9)
    B = 0.1 * f
    Pt_list = [trial.suggest_float(f"Pt_{i}", 1.0, 100.0) for i in range(n)]

    # 2. 构图
    R = max_range(max(Pt_list), P_rx, f)
    graph = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j: continue
            d = dist(i, j)
            if d > R: continue
            t_tx, e_tx = tx_delay_and_energy(Pt_list[i], B, f, d)
            if t_tx == math.inf: continue
            graph[i].append((j, (propagation_delay(d) + t_tx, e_tx)))

    # 3. Dijkstra + 约束
    dist_arr, pred = dijkstra_with_pred(0, graph)
    if any(math.isinf(x) for x in dist_arr):
        return float("inf")
    if max(dist_arr) > T_min:
        return float("inf")

    # 4. 计算平均时延
    D_avg = sum(dist_arr) / (n - 1)

    # 5. 计算能耗
    total_energy = 0.0
    for node in range(n):
        cur = node
        while cur != 0:
            p = pred[cur]
            for v, (_, e_tx) in graph[p]:
                if v == cur:
                    total_energy += e_tx
                    break
            cur = p
    E_total = total_energy

    # 6. 重构每个节点的路径
    paths = []
    for node in range(n):
        path = []
        cur = node
        while cur != -1:
            path.append(cur)
            cur = pred[cur]
        paths.append(path[::-1])

    # 7. 归一化
    D_norm = (D_avg - D_min_est) / (D_max_est - D_min_est)
    E_norm = (E_total - E_min_est) / (E_max_est - E_min_est)
    D_norm = min(max(D_norm,0),1)
    E_norm = min(max(E_norm,0),1)

    # 8. 保存所有感兴趣的属性
    trial.set_user_attr("delays", dist_arr)
    trial.set_user_attr("Pt_list", Pt_list)
    trial.set_user_attr("avg_delay", D_avg)
    trial.set_user_attr("avg_energy", E_total)
    trial.set_user_attr("paths", paths)
    trial.set_user_attr("norm_delay", D_norm)
    trial.set_user_attr("norm_energy", E_norm)

    # 9. 加权目标返回
    return alpha * D_norm + beta * E_norm


# ———— 启动优化 ————
D_max_est, D_min_est, E_max_est, E_min_est = warm()
optuna.logging.set_verbosity(optuna.logging.ERROR)
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=200)

# ———— 输出最优解 ————
best = study.best_params
print(f"最优频率: {best['f'] / 1e9:.3f} GHz")
for i in range(n):
    print(f"节点 {i} 发射功率: {best[f'Pt_{i}']:.2f} dBm")
print(f"最小化目标值: {study.best_value:.6e} (α · D + β · E)")

# 计算最优解对应的路径和时延
def print_best_paths():
    best = study.best_trial
    print("=== 最优解 ===")
    print(f"频率: {best.params['f'] / 1e9:.3f} GHz")
    for i in range(n):
        print(f"节点 {i} 发射功率: {best.params[f'Pt_{i}']:.2f} dBm")

    # 从 user_attrs 拿出结果
    delays = best.user_attrs["delays"]
    paths = best.user_attrs["paths"]
    avg_delay = best.user_attrs["avg_delay"]
    avg_energy = best.user_attrs["avg_energy"]

    print(f"\n平均时延: {avg_delay:.6f} s, 平均能耗: {avg_energy:.6f} J")
    print("\n各节点时延与路径：")
    for i, (d, path) in enumerate(zip(delays, paths)):
        print(f"  节点 {i}: 时延 {d * 1000:.3f} ms, 路径: {' -> '.join(map(str, path))}")

print_best_paths()
