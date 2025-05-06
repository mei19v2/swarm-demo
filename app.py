# app.py
import streamlit as st
import argparse
import tempfile
from network_optimization import NetworkOptimization
from pyvis.network import Network
import streamlit.components.v1 as components
import os
import uuid

st.set_page_config(layout="wide")
st.title("📡 无线自组织网 优化与可视化")

# —— Sidebar: 参数输入 —— 
st.sidebar.header("网络参数")
mode = st.sidebar.selectbox("运行模式", [1, 2, 3, 4], index=2)
nodes_str = st.sidebar.text_area(
    "节点坐标 (米)",
    "[(0,0),(1000,0),(2000,0),(1000,1000),(3000,500),(2500,1500)]"
)
packet_size = st.sidebar.number_input("数据包大小 (MB)", 0.1, 10.0, 1.0, 0.1)
P_rx = st.sidebar.number_input("接收灵敏度 (dBm)", -120.0, 0.0, -90.0, 1.0)
T_min = st.sidebar.number_input("最小时延约束 (s)", 0.001, 1.0, 0.005, 0.001)
alpha = st.sidebar.slider("时延权重 α", 0.0, 1.0, 0.5, 0.1)
beta  = st.sidebar.slider("能耗权重 β",  0.0, 1.0, 0.5, 0.1)

st.sidebar.header("搜索范围")
min_freq   = st.sidebar.number_input("最小频率 (Hz)", 1e8, 1e11, 5e8, 1e7)
max_freq   = st.sidebar.number_input("最大频率 (Hz)", min_freq, 1e11, 5e9, 1e7)
min_power  = st.sidebar.number_input("最小发射功率 (dBm)", 0.0, 60.0, 5.0, 1.0)
max_power  = st.sidebar.number_input("最大发射功率 (dBm)", min_power, 100.0, 30.0, 1.0)

st.sidebar.header("静态模式参数")
static_P_tx = st.sidebar.number_input("静态模式功率 (dBm)", 0.0, 100.0, 20.0, 1.0)
static_freq = st.sidebar.number_input("静态模式频率 (Hz)", 1e8, 1e11, 5e9, 1e7)

n_trials = st.sidebar.number_input("优化迭代次数", 10, 1000, 200, 10)

# —— 按钮触发优化 —— 
if st.sidebar.button("开始优化"):

    # 构造 args
    args = argparse.Namespace(
        c=3e8,
        packet_size=packet_size,
        P_rx=P_rx,
        T_min=T_min,
        alpha=alpha,
        beta=beta,
        mode=mode,
        nodes=nodes_str,
        max_freq=max_freq,
        min_freq=min_freq,
        max_power=max_power,
        min_power=min_power,
        static_P_tx=static_P_tx,
        static_freq=static_freq,
        n_trials=n_trials
    )

    optimizer = NetworkOptimization(args)

    # 根据模式运行
    if mode in (3,4):
        study = optimizer.optimize(n_trials)
        best = study.best_trial
        print("best trial user_attrs:", best.user_attrs)
        required_keys = ["delays", "paths", "avg_delay", "total_energy"]
        missing_keys = [k for k in required_keys if k not in best.user_attrs]
        if missing_keys:
            st.error(f"无法满足约束，无解")
            st.stop()
        n = len(eval(args.nodes))
        f_opt = best.params["f"]
        Pt_list = best.user_attrs["Pt_list"]
        delays  = best.user_attrs["delays"]
        paths   = best.user_attrs["paths"]

        st.write("### 优化后参数")
        st.write(f"- 频率: **{f_opt/1e9:.3f} GHz**")
        # for i in range(n):
        #     st.write(f"节点 {i} 发射功率: {best.params[f'Pt_{i}']:.2f} dBm")
        st.write(f"- 发射功率数组: {['{:.1f} dBm'.format(p) for p in Pt_list]}")
        st.write(f"- 平均时延: **{best.user_attrs['avg_delay']*1000:.2f} ms**")
        st.write(f"- 总能耗: **{best.user_attrs['total_energy']:.3f} J**")

    else:
        # 模式 1 或 2：直接调用并打印
        if mode == 1:
            optimizer.mode1()
            # 模式1里会打印到控制台
        else:
            optimizer.mode2()
        # 获得 static result
        # 由于 mode1/2 直接打印，你可以在控制台查看

        # 为简单起见，接下来使用 last-recorded user_attrs:
        best = optimizer

        f_opt    = static_freq if mode==1 else optimizer.mode2_best_f
        Pt_list  = [static_P_tx]*optimizer.n
        delays   = best.user_attrs.get("delays", [])
        paths    = best.user_attrs.get("paths", [])

    # —— 构建图用于可视化 —— 
    # 重建图（只展示链路，不再考量时延约束）
    R = optimizer.max_range(max(Pt_list), optimizer.P_rx, f_opt)
    graph = optimizer.build_graph(Pt_list, f_opt, R)

    # —— pyvis 可视化 —— 
    net = Network(height="600px", width="100%", notebook=False)
    net.barnes_hut()
    # 添加节点
    for idx, (x,y) in enumerate(optimizer.nodes):
        label = f"{idx}"
        title = f"Node {idx}<br>Power: {Pt_list[idx]:.1f} dBm"
        net.add_node(idx, label=label, title=title, x=x, y=-y, physics=False)
    # 添加边
    for i, neigh in enumerate(graph):
        for j,(delay,e) in neigh:
            net.add_edge(i, j, 
                         title=f"{delay*1000:.2f} ms, {e:.3f} J",
                         value=1, 
                         label=f"{delay*1000:.1f}ms")
    # 保存并展示
    # path = tempfile.mktemp(".html")
    temp_filename = f"temp_{os.getpid()}.html"
    path = os.path.join(tempfile.gettempdir(), f"temp_{uuid.uuid4().hex}.html")  # 生成合法的临时文件路径
    if path.startswith("./"):
        path = path[2:]
    print(path)
    net.show(path)
    components.html(open(path, 'r').read(), height=650)

    # —— 输出路径列表 —— 
    st.write("### 各节点最短路径 & 时延")
    for i, (d,path) in enumerate(zip(delays, paths)):
        path_str = " → ".join(map(str,path))
        st.write(f"- 节点 {i}: 时延 **{d*1000:.2f} ms**, 路径 `{path_str}`")
    # 生成合法的临时 HTML 路径
    