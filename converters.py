import numpy as np
import tensornetwork as tn


def rtni_to_tn(rtni_tensors, rtni_tn_correspondance, d_symbol, d_value, n_tensor_factors):
    final_tensor = None
    for (contracted_edges, coeff) in rtni_tensors:
        tn_copies = tn.copy(rtni_tn_correspondance.values())[0]
        num_coeff = float(coeff.subs({ d_symbol: d_value }))
        #print(f"num_coeff: {num_coeff}")
        root_tn = None
        input_nodes = [
            tn.Node(np.eye(d_value))
            for _ in range(n_tensor_factors)
        ]
        output_nodes = [
            tn.Node(np.eye(d_value))
            for _ in range(n_tensor_factors)
        ]
        encountered_dummy_tensor = False
        for contracted_edge in contracted_edges:
            contracted_edge_start, contracted_edge_end = contracted_edge
            if contracted_edge_start[0] == "@U*" and contracted_edge_start[2] == "in" \
                and contracted_edge_end[0] == "@U" and contracted_edge_end[2] == "out":
                #print("@U and @U* where expected")
                output_nodes[contracted_edge_end[1] - 1][1] ^ input_nodes[contracted_edge_start[1] - 1][0]
                encountered_dummy_tensor = True
            elif contracted_edge_start[0] != "@U*" and contracted_edge_end[0] != "@U":
                #print("normal vertex")
                def rtni_to_tn_edge(tn_tensor, rtni_edge):
                    tn_edge = rtni_edge[1] - 1
                    if rtni_edge[0] == "in":
                        tn_edge += int(len(tn_tensor.edges) / 2)
                    return tn_edge
                tn_start = tn_copies[rtni_tn_correspondance[contracted_edge_start[0]]]
                tn_end = tn_copies[rtni_tn_correspondance[contracted_edge_end[0]]]
                edge_start = rtni_to_tn_edge(tn_start, contracted_edge_start[-2:])
                edge_end = rtni_to_tn_edge(tn_end, contracted_edge_end[-2:])
                tn_start[edge_start] ^ tn_end[edge_end]
                if root_tn == None:
                    root_tn = tn_start
            else:
                raise Exception("unexpected contracted edge {}".format(contracted_edge))
        if root_tn == None:
            raise Exception("encountered no RTNI tensors apart from @U, @U*, should not happen")
        output_edge_order = []
        if encountered_dummy_tensor:
            output_edge_order.extend([
                output_node[0]
                for output_node in output_nodes if output_node[0].is_dangling()
            ])
            output_edge_order.extend([
                input_node[1]
                for input_node in input_nodes if input_node[1].is_dangling()
            ])
        result = tn.contractors.greedy(
            list(tn_copies.values()) + ((output_nodes + input_nodes) if encountered_dummy_tensor else []),
            output_edge_order=output_edge_order
        )
        if final_tensor is None:
            final_tensor = num_coeff * result.tensor
        else:
            final_tensor += num_coeff * result.tensor
    return final_tensor