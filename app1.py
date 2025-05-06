import streamlit as st
import argparse
import matplotlib.pyplot as plt
import networkx as nx
import math
from network_optimization import NetworkOptimization

st.set_page_config(layout="wide")
st.title("📡 无线自组织网 优化与可视化")

# Sidebar parameters
mode = st.sidebar.selectbox("运行模式", ["普通最短路径", "考虑时延固定功率只切换频率", "可切换频率和功率综合考虑能耗时延", "可切换频率和功率只考虑能耗"], index=2)
nodes_str = st.sidebar.text_area("节点坐标 (米)", "[(0,0),(1000,0),(2000,0),(1000,1000),(3000,500),(2500,1500)]")
packet_size = st.sidebar.number_input("数据包大小 (MB)", 0.1, 10.0, 1.0, 0.1)
P_rx = st.sidebar.number_input("接收灵敏度 (dBm)", -120.0, 0.0, -90.0, 1.0)
T_min = st.sidebar.number_input("最小时延约束 (ms)", 0.01, 1000.0, 5.0, 1.0) / 1000
alpha = st.sidebar.slider("时延权重 α", 0.0, 1.0, 0.5, 0.1)
beta  = st.sidebar.slider("能耗权重 β",  0.0, 1.0, 0.5, 0.1)

# Search ranges
min_freq   = st.sidebar.number_input("最小频率 (GHz)", 0.1, 100.0, 0.5, 0.1) * 1e9
max_freq   = st.sidebar.number_input("最大频率 (GHz)", min_freq / 1e9, 100.0, 5.0, 0.1) * 1e9
min_power  = st.sidebar.number_input("最小发射功率 (dBm)", 0.0, 60.0, 5.0, 1.0)
max_power  = st.sidebar.number_input("最大发射功率 (dBm)", min_power, 100.0, 40.0, 1.0)

static_P_tx = st.sidebar.number_input("静态模式功率 (dBm)", 0.0, 100.0, 20.0, 1.0)
static_freq = st.sidebar.number_input("静态模式频率 (GHz)", 0.1, 100.0, 5.0, 0.1) * 1e9

n_trials = st.sidebar.number_input("优化迭代次数", 10, 1000, 200, 10)

run = st.sidebar.button("开始优化")

if run:
    # Map mode from string to integer
    mode_mapping = {
        "普通最短路径": 1,
        "考虑时延固定功率只切换频率": 2,
        "可切换频率和功率综合考虑能耗时延": 3,
        "可切换频率和功率只考虑能耗": 4
    }
    mode_int = mode_mapping.get(mode, 1)  # Default to 1 if mode is not found

    args = argparse.Namespace(
        c=3e8,
        packet_size=packet_size,
        P_rx=P_rx,
        T_min=T_min,
        alpha=alpha,
        beta=beta,
        mode=mode_int,
        nodes=nodes_str,
        max_freq=max_freq,
        min_freq=min_freq,
        max_power=max_power,
        min_power=min_power,
        static_P_tx=static_P_tx,
        static_freq=static_freq,
        n_trials=n_trials
    )
    opt = NetworkOptimization(args)
    st.warning(mode)
    if mode_int in (3, 4):
        study = opt.optimize(n_trials)
        best = study.best_trial
        user = best.user_attrs

        required_keys = ["delays", "paths", "avg_delay", "total_energy"]
        missing_keys = [k for k in required_keys if k not in user]
        if missing_keys:
            st.error(f"无法满足约束，无解")
            st.stop()
        f_opt = best.params['f']
        Pt_list = user['Pt_list']
        delays  = user['delays']
        paths   = user['paths']
        total_energy = user['total_energy']
        avg_delay = user['avg_delay']
    
    elif mode_int == 1:
        Pt_list = [static_P_tx] * len(opt.nodes)
        delays, paths, total_energy, f_opt = opt.mode1()
        avg_delay = sum(delays) / (len(delays) - 1)
        Pt_list = [static_P_tx] * len(opt.nodes)

    else:
        Pt_list = [static_P_tx]*len(opt.nodes)
        delays, paths, total_energy, f_opt = opt.mode2()
        if delays is None:
            st.error(f"无法满足约束，无解")
            st.stop()
        avg_delay = sum(delays) / (len(delays) - 1)
        Pt_list = [static_P_tx] * len(opt.nodes)
        
    print(paths)
    
    # show paths
    # st.write(f"集群切换到频率{f_opt/1e9:.3f} GHz")
    # st.write("各节点最短路径 & 时延")
    # for i,path in enumerate(paths):
    #     st.write(f"节点 {i}: {'->'.join(map(str,path))}, 时延 {delays[i]*1000:.2f} ms, 发送功率 {Pt_list[i]:.2f} dBm")
    # st.write(f"本次消息分发能耗：{total_energy:.3f} J, 平均时延 {avg_delay*1000:.2f} ms")
    # Build graph and networkx
    
    G = nx.DiGraph()
    coords = eval(nodes_str)
    for i,(x,y) in enumerate(coords):
        G.add_node(i, pos=(x,y))
    R = opt.max_range(max(Pt_list), opt.P_rx, f_opt)
    B = 0.1 * f_opt
    for i in range(len(coords)):
        for j in range(len(coords)):
            if i==j: continue
            d = opt.dist(i,j)
            if d > R: continue
            t_tx, e = opt.tx_delay_and_energy(Pt_list[i],B, f_opt , d)
            if t_tx==math.inf: continue
            G.add_edge(i,j, delay=opt.propagation_delay(d) + t_tx, energy=e)

    # Draw with matplotlib
    pos = nx.get_node_attributes(G, 'pos')
    fig, ax = plt.subplots(figsize=(6, 4))  # Adjusted figure size to make the graph smaller
    nx.draw(G, pos, with_labels=True, node_size=250, node_color='skyblue', font_size=6 , edgelist=[], ax=ax)
    # edge labels
    # edge_labels = {(u,v): f"{data['delay']*1000:.1f}ms/{data['energy']:.2e}J" for u,v,data in G.edges(data=True)}
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=4, ax=ax)

    # Calculate and store individual segment delays
    segment_delays = []
    for path, total_delay in zip(paths, delays):
        segment_delay = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if G.has_edge(u, v):
                edge_data = G.get_edge_data(u, v)
                segment_delay.append(edge_data['delay'])
        segment_delays.append(segment_delay)

    # Draw paths based on the paths array with individual segment delays
    for path, segment_delay in zip(paths, segment_delays):
        if len(path) > 1:
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                if G.has_edge(u, v):
                    edge_label = f"{segment_delay[i] * 1000:.1f}ms"
                    nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color='red', width=1, ax=ax)
                    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): edge_label}, font_size=4, ax=ax)
    col1, col2 = st.columns([0.4, 0.6])  # 1:1的比例，可以根据需要调整
    with col1:
        st.write(f"**集群切换到频率 {f_opt/1e9:.3f} GHz**")
        st.write("**各节点最短路径 & 时延**")
        for i, path in enumerate(paths):
            st.write(f"节点 {i}: {'→'.join(map(str,path))}, 时延 {delays[i]*1000:.2f} ms, 发送功率 {Pt_list[i]:.2f} dBm")
        st.write(f"**本次消息分发能耗**: {total_energy:.3f} J")
        st.write(f"**平均时延**: {avg_delay*1000:.2f} ms")

    with col2:
        # 绘制图形
        st.pyplot(fig, use_container_width=True)
    # st.pyplot(fig)


