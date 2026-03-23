use log::{debug, info};
use petgraph::{
    algo::{greedy_feedback_arc_set, is_cyclic_directed},
    stable_graph::{EdgeIndex, StableDiGraph},
    visit::EdgeRef,
};

use super::{Edge, Vertex};

/// Removes all the edges that contribute to cycles in the graph
/// Does so by finding a greedy feedback arc set and then reversing the
/// direction of the edges from that set.
/// Is not guaranteed to find the minimum fas.
pub(crate) fn remove_cycles(graph: &mut StableDiGraph<Vertex, Edge>) -> Vec<EdgeIndex> {
    if !is_cyclic_directed(&*graph) {
        info!(target: "Cycle Removal", "Graph contains no cycle");
        return Vec::new();
    }

    info!(target: "Cycle Removal", "Graph contains cycle, reversing edges");

    // get the feedback arc set
    let fas: Vec<EdgeIndex> = greedy_feedback_arc_set(&*graph).map(|e| e.id()).collect();
    let mut reversed_edges = Vec::new();

    // reverse the direction of the edges
    for edge in fas {
        if let Some((tail, head)) = graph.edge_endpoints(edge) {
            // get the weight
            let weight = graph[edge];
            // add new edge in reversed direction
            let reversed_edge = graph.add_edge(head, tail, weight);
            reversed_edges.push(reversed_edge);
            // remove the old edge
            graph.remove_edge(edge);
        }
    }

    assert!(!is_cyclic_directed(&*graph));

    debug!(target: "Cycle Removal", "Reversed {} edges", reversed_edges.len());

    reversed_edges
}
