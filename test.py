import math, heapq, optuna

class WirelessNetworkOptimization:
    def __init__(self):
        # 全局常量
        self.c = 3e8  # 光速 (m/s)
        self.k = 1.38e-23
        self.T = 290  # 热噪声常数
        self.L = 8e6  # 1 MB = 8e6 bits
        self.P_rx = -90  # 接收灵敏度 dBm
        self.T_min = 0.05  # 最小时延约束 (s)
        self.alpha = 0   # 时延权重
        self.beta = 1  # 能耗权重

        # 节点坐标
        self.nodes = [
            (0, 0),
            (1000, 0),
            (2000, 0),
            (1000, 1000),
            (3000, 500),
            (2500, 1500)
        ]
        self.n = len(self.nodes)

    def dist(self, i, j):
        xi, yi = self.nodes[i]
        xj, yj = self.nodes[j]
        return math.hypot(xi - xj, yi - yj)

    def max_range(self, Pt_dBm, Pr_dBm, f):
        fspl_max = Pt_dBm - Pr_dBm
        f_mhz = f / 1e6
        d_km = 10 ** ((fspl_max - 32.44 - 20 * math.log10(f_mhz)) / 20)
        return d_km * 1000

    def propagation_delay(self, d):
        return d / self.c

    def tx_delay_and_energy(self, Pt_dBm, B, f, d):
        # 自由空间路径损耗 (FSPL)
        f_mhz, d_km = f / 1e6, d / 1000
        FSPL = 32.44 + 20 * math.log10(f_mhz) + 20 * math.log10(d_km)
        Pr_dBm = Pt_dBm - FSPL
        Pr_w = 10 ** ((Pr_dBm - 30) / 10)
        # 噪声功率
        noise = self.k * self.T * B
        snr = Pr_w / noise
        if snr <= 0:
            return float('inf'), 0.0
        # 信道容量 & 传输时延
        cap = B * math.log2(1 + snr)
        t_tx = self.L / cap
        # 发射能量
        Pt_w = 10 ** ((Pt_dBm - 30) / 10)
        e_tx = Pt_w * t_tx
        return t_tx, e_tx

    def dijkstra_with_pred(self, start, graph):
        dist_arr = [math.inf] * self.n
        pred = [-1] * self.n
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

    def objective(self, trial):
        # 1. 采样
        f = trial.suggest_float("f", 5e8, 6e9)
        B = 0.1 * f
        Pt_list = [trial.suggest_float(f"Pt_{i}", 1.0, 100.0) for i in range(self.n)]

        # 2. 构建图
        R = self.max_range(max(Pt_list), self.P_rx, f)
        graph = [[] for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                if i == j: continue
                d = self.dist(i, j)
                if d > R: continue
                t_tx, e_tx = self.tx_delay_and_energy(Pt_list[i], B, f, d)
                if t_tx == math.inf: continue
                graph[i].append((j, (self.propagation_delay(d) + t_tx, e_tx)))

        # 3. Dijkstra
        dist_arr, pred = self.dijkstra_with_pred(0, graph)
        if any(math.isinf(d) for d in dist_arr):
            return float("inf")
        if max(dist_arr) > self.T_min:
            return float("inf")

        # 4. 计算平均时延 & 能耗
        D_avg = sum(dist_arr) / (self.n - 1)
        total_energy = 0.0
        for node in range(self.n):
            cur = node
            while cur != 0:
                p = pred[cur]
                for v, (_, e_tx) in graph[p]:
                    if v == cur:
                        total_energy += e_tx
                        break
                cur = p
        E_total = total_energy

        # 5. 重构路径
        paths = []
        for node in range(self.n):
            path = []
            cur = node
            while cur != -1:
                path.append(cur)
                cur = pred[cur]
            path.reverse()
            paths.append(path)

        # 6. 保存 user_attrs
        trial.set_user_attr("delays", dist_arr)
        trial.set_user_attr("Pt_list", Pt_list)
        trial.set_user_attr("avg_delay", D_avg)
        trial.set_user_attr("avg_total", E_total)
        trial.set_user_attr("paths", paths)

        # 7. 返回目标
        return self.alpha * D_avg + self.beta * E_total

    def optimize(self, n_trials=200):
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=n_trials)
        return study

    def print_best_paths(self, study):
        best = study.best_trial
        print("=== 最优解 ===")
        print(f"频率: {best.params['f'] / 1e9:.3f} GHz")
        for i in range(self.n):
            print(f"节点 {i} 发射功率: {best.params[f'Pt_{i}']:.2f} dBm")

        # 从 user_attrs 拿出结果
        delays = best.user_attrs["delays"]
        paths = best.user_attrs["paths"]
        avg_delay = best.user_attrs["avg_delay"]
        avg_total = best.user_attrs["avg_total"]

        print(f"\n平均时延: {avg_delay:.6f} s, 总能耗: {avg_total:.6f} J")
        print("\n各节点时延与路径：")
        for i, (d, path) in enumerate(zip(delays, paths)):
            print(f"  节点 {i}: 时延 {d * 1000:.3f} ms, 路径: {' -> '.join(map(str, path))}")
    

# 使用类
if __name__ == "__main__":
    optimizer = WirelessNetworkOptimization()
    study = optimizer.optimize(n_trials=500)
    optimizer.print_best_paths(study)
