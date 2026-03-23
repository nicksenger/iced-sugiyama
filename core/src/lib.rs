use std::collections::HashMap;

use algorithm::{Edge, Vertex};

use configure::Config;
use log::info;
use petgraph::{graph::NodeIndex, stable_graph::StableDiGraph};

pub mod advanced;
mod algorithm;
pub mod configure;

type Layout = (Vec<(usize, (f64, f64))>, f64, f64);
type Layouts<T> = Vec<(Vec<(T, (f64, f64))>, f64, f64)>;

/// Creates a graph layout from edges, which are given as a `&[(u32, u32)]`.
///
/// The layouts are returned as a list of disjoint subgraphs containing the
/// subgraph layout, the width, and the height. The layout of a subgraph is a
/// list of the vertex number (as specified in the edges) and its x and y
/// position respectively.
pub fn from_edges(edges: &[(u32, u32)], config: &Config) -> Layouts<usize> {
    info!(target: "initializing", "Creating new layout from edges, containing {} edges", edges.len());
    let graph = StableDiGraph::from_edges(edges);
    algorithm::start(graph, config)
}

/// Creates a graph layout from a preexisting [StableDiGraph<V, E>].
///
/// The layouts are returned as a list of disjoint subgraphs containing the
/// subgraph layout, the width, and the height. The layout of a subgraph is a
/// list of the [NodeIndex] and its x and y position respectively.
pub fn from_graph<V, E>(
    graph: &StableDiGraph<V, E>,
    vertex_size: &impl Fn(NodeIndex, &V) -> (f64, f64),
    config: &Config,
) -> Layouts<NodeIndex> {
    info!(target: "initializing", 
        "Creating new layout from existing graph, containing {} vertices and {} edges.", 
        graph.node_count(), 
        graph.edge_count());

    let graph = graph.map(
        |id, v| Vertex::new(id.index(), vertex_size(id, v)),
        |_, _| Edge::default(),
    );

    algorithm::start(graph, config)
        .into_iter()
        .map(|(l, w, h)| {
            (
                l.into_iter()
                    .map(|(id, coords)| (NodeIndex::from(id as u32), coords))
                    .collect(),
                w,
                h,
            )
        })
        .collect()
}

/// Creates a graph layout from `&[(u32, (f64, f64))]` (vertices as vertex id
/// and vertex size) and `&[(u32, u32)]` (edges).
///
/// The layouts are returned as a list of disjoint subgraphs containing the
/// subgraph layout, the width, and the height. The layout of a subgraph is a
/// list of the vertex number and its x and y position respectively.
///
/// # Panics
///
/// Panics if `edges` contain vertices which are not contained in `vertices`
pub fn from_vertices_and_edges<'a>(
    vertices: &'a [(u32, (f64, f64))],
    edges: &'a [(u32, u32)],
    config: &Config,
) -> Layouts<usize> {
    info!(target: "initializing", 
        "Creating new layout from existing graph, containing {} vertices and {} edges.", 
        vertices.len(), 
        edges.len());

    let mut graph = StableDiGraph::new();
    let mut id_map = HashMap::new();
    for &(v, size) in vertices {
        let id = graph.add_node(Vertex::new(v as usize, size));
        id_map.insert(v, id);
    }

    for (tail, head) in edges {
        graph.add_edge(
            *id_map.get(tail).unwrap(),
            *id_map.get(head).unwrap(),
            Edge::default(),
        );
    }

    algorithm::start(graph, config)
}
