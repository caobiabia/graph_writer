import networkx as nx
import json

def is_dag(G):
    """
    判断 NetworkX 有向图 G 是否为有向无环图 (DAG)。
    """
    return nx.is_directed_acyclic_graph(G)


def make_dag(G: nx.DiGraph) -> nx.DiGraph:
    """
    通过迭代移除强连通分量 (SCC) 内权重最低的边，将有环图转换为有向无环图 (DAG)。
    这是解决反馈弧集问题的启发式方法。

    Args:
        G (nx.DiGraph): 输入的有向图，边需要有 'weight' 属性。

    Returns:
        nx.DiGraph: 转换后的有向无环图 (DAG)。
    """
    dag_G = G.copy()
    removed_edges_count = 0
    removed_edges_log = []

    print("\n--- 开始将图转换为 DAG ---")

    # 循环直到图成为 DAG
    while not nx.is_directed_acyclic_graph(dag_G):
        # 找到所有强连通分量 (SCCs)
        sccs = list(nx.strongly_connected_components(dag_G))
        
        # 筛选出包含循环的 SCCs (节点数 > 1 的分量)
        cyclic_sccs = [scc for scc in sccs if len(scc) > 1]
        
        if not cyclic_sccs:
            # 理论上不会发生，但作为安全措施
            break 
            
        # 1. 找到所有 SCC 内部的边
        edges_in_cycles = []
        for scc in cyclic_sccs:
            # 遍历 SCC 内部的节点对，检查它们之间的边
            for u, v, data in dag_G.edges(data=True):
                if u in scc and v in scc:
                    edges_in_cycles.append((u, v, data.get('weight', 0)))

        if not edges_in_cycles:
            # 无法找到更多循环边 (可能图结构异常或网络库问题)
            print("警告：存在大于1个节点的SCC但未找到内部边，停止循环。")
            break

        # 2. 找到所有循环边中权重最低的那条边
        # 假设权重越低，依赖关系越弱，移除越合理
        edges_in_cycles.sort(key=lambda x: x[2])
        u_remove, v_remove, weight_remove = edges_in_cycles[0]
        
        # 3. 移除该边
        dag_G.remove_edge(u_remove, v_remove)
        
        # 记录移除操作
        node_u_title = dag_G.nodes[u_remove]['title']
        node_v_title = dag_G.nodes[v_remove]['title']
        
        removed_edges_log.append({
            "source": u_remove, 
            "target": v_remove, 
            "weight": weight_remove,
            "reason": f"打破 {len(cyclic_sccs)} 个循环中的一个, 移除最弱依赖: {node_u_title} -> {node_v_title}"
        })
        removed_edges_count += 1
        print(f"[{removed_edges_count}] 移除边: {node_u_title} -> {node_v_title} (权重: {weight_remove:.4f})")

    print(f"--- DAG 转换完成。共移除 {removed_edges_count} 条边 ---")
    
    # 打印最终的移除日志
    if removed_edges_log:
        print("\n最终移除日志:")
        for log in removed_edges_log:
             print(f"- 移除了 {log['source']} ({log['reason'].split(': ')[1]}) -> {log['target']}, 权重: {log['weight']:.4f}")
    
    return dag_G


def save_graph_json(G, json_path="graph.json"):
    """仅保存图的 JSON 结构"""
    graph_json = {
        "nodes": [{"id": n, "title": d["title"]} for n, d in G.nodes(data=True)],
        "edges": [{"source": u, "target": v, "weight": d["weight"]} for u, v, d in G.edges(data=True)]
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(graph_json, f, ensure_ascii=False, indent=4)
    print(f"图的JSON已保存到 {json_path}")


def dag_to_levels(G: nx.DiGraph):
    """将 DAG 转换为层级列表"""
    levels = []
    # 拷贝一份避免修改原始图
    dag = G.copy()
    while True:
        # 当前可执行层 = 所有 in_degree == 0 的节点
        zero_in = [n for n in dag.nodes if dag.in_degree(n) == 0]
        if not zero_in:
            break
        levels.append(zero_in)
        dag.remove_nodes_from(zero_in)
    return levels
