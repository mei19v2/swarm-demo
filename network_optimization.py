import math
import heapq
import argparse
import optuna
import random
from enum import Enum

class Mode(Enum):
    MODE1 = 1
    MODE2 = 2
    MODE3 = 3
    MODE4 = 4

class NetworkOptimization:
    def __init__(self, args):
        # 全局常量
        self.c = args.c
        self.k = 1.38e-23
        self.T = 290   # 热噪声系数
        self.L = args.packet_size * 8 * 10**6  # 数据包大小 (bits) 输入的单位是MB
        self.P_rx = args.P_rx    # 接收灵敏度 (dBm)
        self.T_min = args.T_min  # 最小实验约束
        self.alpha = args.alpha  # 时延权重
        self.beta = args.beta  # 能耗权重
        self.mode = args.mode  # 工作模式
        self.max_freq = args.max_freq  # 最大频率 (Hz)
        self.min_freq = args.min_freq  # 最小频率 (Hz)
        self.max_power = args.max_power  # 最大发射功率 (dBm)
        self.min_power = args.min_power  # 最小发射功率 (dBm)
        self.n_trials = args.n_trials  # 优化迭代次数
       
        # 节点坐标
        self.nodes = eval(args.nodes) if args.nodes else [(0, 0), (1000, 0), (2000, 0), (1000, 1000), (3000, 500), (2500, 1500)]
        self.n = len(self.nodes)

        # 静态模式下发送功率
        self.static_P_tx = args.static_P_tx  # dBm
        self.static_freq = args.static_freq  # Hz

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
    
    def print_link_details(self, f, pred, delays):
        res = []
        for i, t in enumerate(delays):
            if t < float('inf'):
                prev = i
                path = []
                while prev != -1:
                    path.append(prev)
                    prev = pred[prev]
                path.reverse()  # 原地反转路径
                path_str = " -> ".join(map(str, path))
                print(f"节点0 到 节点{i} 的最短时延: {t * 1000:.3f} 毫秒, 路径: {path_str}")
                res.append(path)
            else:
                print(f"节点0 到 节点{i} 不可达")
        return res
    
    def build_graph(self, Pt_list, f, R):
        B = 0.1 * f
        graph = [[] for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                if i == j: continue
                d = self.dist(i, j)
                if d > R: continue
                t_tx, e_tx = self.tx_delay_and_energy(Pt_list[i], B, f, d)
                if t_tx == math.inf: continue
                graph[i].append((j, (self.propagation_delay(d) + t_tx, e_tx)))
        return graph

    def optimize(self, n_trials):
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=n_trials)
        return study
    
    def objective(self, trial):
        # 1. 采样
        f = trial.suggest_float("f", self.min_freq, self.max_freq)
        B = 0.1 * f
        Pt_list = [trial.suggest_float(f"Pt_{i}", self.min_power, self.max_power) for i in range(self.n)]

        # 2. 构建图
        R = self.max_range(max(Pt_list), self.P_rx, f)
        graph = self.build_graph(Pt_list, f, R)
        
        # 3. Dijkstra
        dist_arr, pred = self.dijkstra_with_pred(0, graph)
        if any(math.isinf(d) for d in dist_arr):
            return float("inf")   # 不连通时跳过
        if self.mode == 3 and max(dist_arr) > self.T_min:  #  # 仅在模式3下检查时延约束
            return float("inf")

        # 4. 计算平均时延 & 能耗
        D_avg = sum(dist_arr) / (self.n - 1)
        E_total = 0.0
        for node in range(self.n):
            cur = node
            while cur != 0:
                p = pred[cur]
                for v, (_, e_tx) in graph[p]:
                    if v == cur:
                        E_total += e_tx
                        break
                cur = p

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
        trial.set_user_attr("total_energy", E_total)
        trial.set_user_attr("paths", paths)

        # 7. 返回目标
        if self.mode == 3:
            return self.alpha * D_avg + self.beta * E_total
        elif self.mode == 4:
            return E_total
        return self.alpha * D_avg + self.beta * E_total
    

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
        avg_total = best.user_attrs["total_energy"]

        print(f"\n平均时延: {avg_delay:.6f} s, 总能耗: {avg_total:.6f} J")
        print("\n各节点时延与路径：")
        for i, (d, path) in enumerate(zip(delays, paths)):
            print(f"  节点 {i}: 时延 {d * 1000:.3f} ms, 路径: {' -> '.join(map(str, path))}")
    
    
    def mode1(self):
        # 对应 demo_try1.py 的功能
        print("运行模式 1：普通固定最短路径")
        # ...existing code from demo_try1.py...
        B = 0.1 * self.static_freq
        Pt_list = [self.static_P_tx] * self.n
        R = self.max_range(self.static_P_tx, self.P_rx, self.static_freq)
        # 2. 构建图
        graph = self.build_graph(Pt_list, self.static_freq, R)
        dist_arr, pred = self.dijkstra_with_pred(0, graph)
        # 输出结果
        path = self.print_link_details(self.static_freq, pred, dist_arr)
        total_energy = sum(dist_arr) * 10 ** ((self.static_P_tx - 30) / 10)  # 总能耗
        # print(total_energy)
        return dist_arr, path, total_energy, self.static_freq
    
    def mode2(self):
        # 对应 demo_try2.py 的功能
        print("运行模式 2：不修改功率只切换频率")
        # ...existing code from demo_try2.py...
        min_delay = float('inf')
        best_f = None
        Pt_list = [self.static_P_tx] * self.n
        for f in range(int(self.min_freq), int(self.max_freq) + 1, 10_000_000):
            max_range = self.max_range(self.static_P_tx, self.P_rx, f)
            # 构建图
            graph = self.build_graph(Pt_list, f, max_range)
            dist_arr, pred = self.dijkstra_with_pred(0, graph)
            if any(math.isinf(d) for d in dist_arr):
                continue
            max_delay = max(dist_arr)
            if max_delay < self.T_min:
                avg_delay = sum(dist_arr) / (self.n - 1)
                if(avg_delay < min_delay):
                    min_delay = avg_delay
                    best_f = f
                    best_delays = dist_arr[:]
                    best_pred = pred[:]
        if best_f:
            print(f"最优频率: {best_f / 1e9:.2f} GHz")
            path = self.print_link_details(best_f, best_pred, best_delays)
            total_energy = sum(best_delays) * 10 ** ((self.static_P_tx - 30) / 10)
        else:
            print("未找到可用频率使网络连通")
            return None, None, None, None
        print(path)
        return best_delays, path, total_energy, best_f
            
    def mode3(self):
        # 对应 test.py 的功能
        print("运行模式 3：同时优化时延与能耗")
        study = self.optimize(self.n_trials)
        self.print_best_paths(study)
        # ...existing code from test.py...

    def mode4(self):
        # 对应 demo_try4.py 的功能
        print("运行模式 4：节能")
        study = self.optimize(self.n_trials)
        self.print_best_paths(study)
        # ...existing code from demo_try4.py...

    def run(self):
        if self.mode == 1:
            self.mode1()
        elif self.mode == 2:
            self.mode2()
        elif self.mode == 3:
            self.mode3()
        elif self.mode == 4:
            self.mode4()
        else:
            print("无效的模式选择，请选择 1-4 之间的模式")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="网络优化工具")
    parser.add_argument("--c", type=float, default=3e8, help="光速 (m/s)")
    parser.add_argument("--packet_size", type=float, default=1, help="数据包大小 (MB)")
    parser.add_argument("--P_rx", type=float, default=-90, help="接收灵敏度 (dBm)")
    parser.add_argument("--T_min", type=float, default=0.005, help="最小时延约束 (s)")
    parser.add_argument("--alpha", type=float, default=0.5, help="时延权重")
    parser.add_argument("--beta", type=float, default=0.5, help="能耗权重")
    parser.add_argument("--mode", type=int, default=3, help="工作模式 (1-4)")
    parser.add_argument("--nodes", type=str, default="[(0,0),(1000,0),(2000,0),(1000,1000),(3000,500),(2500,1500)]", help="节点的坐标列表，格式为 [(x1,y1),(x2,y2),...]，单位：米")
    parser.add_argument("--max_freq", type=float, default=3e10, help="最大工作频率(hz)")
    parser.add_argument("--min_freq", type=float, default=5e8, help="最小工作频率(hz)")
    parser.add_argument("--max_power", type=int, default=60, help="最大发射功率(dBm)")
    parser.add_argument("--min_power", type=int, default=5, help="最小发射功率(dBm)")
    parser.add_argument("--static_P_tx", type=int, default=20, help="静态模式下发送功率(dBm)")
    parser.add_argument("--n_trials", type=int, default=200, help="优化迭代次数")
    parser.add_argument("--static_freq", type=int, default=5e9, help="静态模式下频率(hz)")
    args = parser.parse_args()

    optimizer = NetworkOptimization(args)
    optimizer.run()
