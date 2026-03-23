use std::collections::{BTreeMap, HashMap};

use advanced::{from_graph_with_features, ClusterSpec, DetailedLayout, RoutedEdge, RoutedNode};
use algorithm::{Edge, Vertex};
use log::info;
use petgraph::{graph::NodeIndex, stable_graph::StableDiGraph};

mod advanced;
mod algorithm;
mod configure;
mod util;

pub use advanced::RenderConfig;
pub use configure::{Config, CrossingMinimization, RankingType};

type Layout = (Vec<(usize, (f64, f64))>, f64, f64);
type Layouts<T> = Vec<(Vec<(T, (f64, f64))>, f64, f64)>;

const DEFAULT_NODE_SIZE: (f64, f64) = (56.0, 32.0);
const COMPONENT_GAP: f64 = 80.0;

#[derive(Debug, Clone, PartialEq)]
pub struct Cluster {
    nodes: Vec<u32>,
    padding: Option<f64>,
    parent: Option<usize>,
}

impl Cluster {
    pub fn new(nodes: Vec<u32>) -> Self {
        Self {
            nodes,
            padding: None,
            parent: None,
        }
    }

    pub fn padding(self, padding: f64) -> Self {
        Self {
            padding: Some(padding),
            ..self
        }
    }

    pub fn parent(self, parent: usize) -> Self {
        Self {
            parent: Some(parent),
            ..self
        }
    }

    pub fn nodes(&self) -> &[u32] {
        &self.nodes
    }

    pub fn padding_value(&self) -> Option<f64> {
        self.padding
    }

    pub fn parent_index(&self) -> Option<usize> {
        self.parent
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EdgeLayout {
    index: usize,
    points: Vec<(f64, f64)>,
    curve_points: Vec<(f64, f64)>,
    label: Option<String>,
    label_position: Option<(f64, f64)>,
}

impl EdgeLayout {
    pub fn index(&self) -> usize {
        self.index
    }

    pub fn points(&self) -> &[(f64, f64)] {
        &self.points
    }

    pub fn curve_points(&self) -> &[(f64, f64)] {
        &self.curve_points
    }

    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    pub fn label_position(&self) -> Option<(f64, f64)> {
        self.label_position
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClusterLayout {
    index: usize,
    parent: Option<usize>,
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
}

impl ClusterLayout {
    pub fn index(&self) -> usize {
        self.index
    }

    pub fn parent(&self) -> Option<usize> {
        self.parent
    }

    pub fn min_x(&self) -> f64 {
        self.min_x
    }

    pub fn min_y(&self) -> f64 {
        self.min_y
    }

    pub fn max_x(&self) -> f64 {
        self.max_x
    }

    pub fn max_y(&self) -> f64 {
        self.max_y
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GraphLayout {
    max_x: f64,
    max_y: f64,
    coords: BTreeMap<usize, (f64, f64)>,
    edges: Vec<EdgeLayout>,
    clusters: Vec<ClusterLayout>,
}

impl GraphLayout {
    fn empty() -> Self {
        Self {
            max_x: 0.0,
            max_y: 1.0,
            coords: BTreeMap::new(),
            edges: Vec::new(),
            clusters: Vec::new(),
        }
    }

    pub fn max_x(&self) -> f64 {
        self.max_x
    }

    pub fn max_y(&self) -> f64 {
        self.max_y
    }

    pub fn position(&self, index: usize) -> Option<(f64, f64)> {
        self.coords.get(&index).copied()
    }

    pub fn edges(&self) -> &[EdgeLayout] {
        &self.edges
    }

    pub fn clusters(&self) -> &[ClusterLayout] {
        &self.clusters
    }
}

#[cfg(test)]
/// Creates a graph layout from edges, which are given as a `&[(u32, u32)]`.
///
/// The layouts are returned as a list of disjoint subgraphs containing the
/// subgraph layout, the width, and the height. The layout of a subgraph is a
/// list of the vertex number (as specified in the edges) and its x and y
/// position respectively.
pub(crate) fn from_edges(edges: &[(u32, u32)], config: &Config) -> Layouts<usize> {
    info!(target: "initializing", "Creating new layout from edges, containing {} edges", edges.len());
    let graph = StableDiGraph::from_edges(edges);
    algorithm::start(graph, config)
}

/// Creates a graph layout from a preexisting [StableDiGraph<V, E>].
///
/// The layouts are returned as a list of disjoint subgraphs containing the
/// subgraph layout, the width, and the height. The layout of a subgraph is a
/// list of the [NodeIndex] and its x and y position respectively.
pub(crate) fn from_graph<V, E>(
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

#[cfg(test)]
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
pub(crate) fn from_vertices_and_edges<'a>(
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

pub fn layout_graph(
    nodes: &[u32],
    edges: &[(u32, u32)],
    config: &Config,
    node_size: impl Fn(u32) -> (f64, f64),
    edge_label: impl Fn(usize, (u32, u32)) -> Option<String>,
    clusters: &[Cluster],
    render_config: &RenderConfig,
) -> GraphLayout {
    if nodes.is_empty() {
        return GraphLayout::empty();
    }

    let mut graph_node_indices = HashMap::<u32, NodeIndex>::new();
    let mut graph = StableDiGraph::<u32, usize>::new();
    for node in nodes {
        let index = graph.add_node(*node);
        graph_node_indices.insert(*node, index);
    }

    for (edge_index, (from, to)) in edges.iter().copied().enumerate() {
        let (Some(from_idx), Some(to_idx)) = (
            graph_node_indices.get(&from).copied(),
            graph_node_indices.get(&to).copied(),
        ) else {
            continue;
        };
        graph.add_edge(from_idx, to_idx, edge_index);
    }

    let cluster_specs = clusters
        .iter()
        .enumerate()
        .map(|(cluster_index, cluster)| ClusterSpec {
            id: cluster_index,
            nodes: cluster
                .nodes()
                .iter()
                .filter_map(|node| graph_node_indices.get(node).copied())
                .collect(),
            padding: cluster.padding_value(),
            parent: cluster.parent_index(),
        })
        .collect::<Vec<_>>();

    let detailed = from_graph_with_features(
        &graph,
        &|_, node| {
            let size = node_size(*node);
            if size.0 > 0.0 && size.1 > 0.0 {
                size
            } else {
                DEFAULT_NODE_SIZE
            }
        },
        &|_, edge_idx| {
            let edge_index = *edge_idx;
            edges
                .get(edge_index)
                .and_then(|edge| edge_label(edge_index, *edge))
        },
        &cluster_specs,
        config,
        render_config,
    );

    let node_to_position = nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (*node, index))
        .collect::<HashMap<_, _>>();

    merge_components(&graph, detailed, &node_to_position)
}

fn merge_components(
    graph: &StableDiGraph<u32, usize>,
    components: Vec<DetailedLayout<NodeIndex, String, usize>>,
    node_to_position: &HashMap<u32, usize>,
) -> GraphLayout {
    let mut coords = BTreeMap::new();
    let mut edges = Vec::<EdgeLayout>::new();
    let mut clusters = Vec::<ClusterLayout>::new();

    let mut x_offset = 0.0;
    let mut max_y = 1.0f64;

    let total_components = components.len();
    for (component_idx, component) in components.into_iter().enumerate() {
        for node in component.nodes {
            merge_node(graph, node, x_offset, node_to_position, &mut coords);
        }

        for edge in component.edges {
            merge_edge(graph, edge, x_offset, &mut edges);
        }

        for cluster in component.clusters {
            clusters.push(ClusterLayout {
                index: cluster.id,
                parent: cluster.parent,
                min_x: cluster.bounds.min_x + x_offset,
                min_y: cluster.bounds.min_y,
                max_x: cluster.bounds.max_x + x_offset,
                max_y: cluster.bounds.max_y,
            });
        }

        max_y = max_y.max(component.height);
        x_offset += component.width;
        if component_idx + 1 < total_components {
            x_offset += COMPONENT_GAP;
        }
    }

    GraphLayout {
        max_x: x_offset.max(0.0),
        max_y,
        coords,
        edges,
        clusters,
    }
}

fn merge_node(
    graph: &StableDiGraph<u32, usize>,
    node: RoutedNode<NodeIndex>,
    x_offset: f64,
    node_to_position: &HashMap<u32, usize>,
    coords: &mut BTreeMap<usize, (f64, f64)>,
) {
    let graph_node = graph[node.id];
    if let Some(position) = node_to_position.get(&graph_node).copied() {
        coords.insert(position, (node.center.0 + x_offset, node.center.1));
    }
}

fn merge_edge(
    graph: &StableDiGraph<u32, usize>,
    edge: RoutedEdge<NodeIndex, String>,
    x_offset: f64,
    edges: &mut Vec<EdgeLayout>,
) {
    let index = *graph.edge_weight(edge.id).unwrap_or(&edge.id.index());
    edges.push(EdgeLayout {
        index,
        points: edge
            .points
            .into_iter()
            .map(|(x, y)| (x + x_offset, y))
            .collect(),
        curve_points: edge
            .curve_points
            .into_iter()
            .map(|(x, y)| (x + x_offset, y))
            .collect(),
        label: edge.label.as_ref().map(|label| label.value.clone()),
        label_position: edge
            .label
            .as_ref()
            .map(|label| (label.position.0 + x_offset, label.position.1)),
    });
}

#[test]
fn run_algo_empty_graph() {
    let edges = [];
    let g = from_edges(&edges, &Config::default());
    assert!(g.is_empty());
}

#[cfg(test)]
mod benchmark {
    use crate::configure::Config;

    use super::from_edges;

    #[test]
    fn r_100() {
        let edges = graph_generator::RandomLayout::new(100)
            .build_edges()
            .into_iter()
            .map(|(r, l)| (r as u32, l as u32))
            .collect::<Vec<(u32, u32)>>();
        let start = std::time::Instant::now();
        let _ = from_edges(&edges, &Config::default());
        println!("Random 100 edges: {}ms", start.elapsed().as_millis());
    }

    #[test]
    fn r_1000() {
        let edges = graph_generator::RandomLayout::new(1000)
            .build_edges()
            .into_iter()
            .map(|(r, l)| (r as u32, l as u32))
            .collect::<Vec<(u32, u32)>>();
        let start = std::time::Instant::now();
        let _ = from_edges(&edges, &Config::default());
        println!("Random 1000 edges: {}ms", start.elapsed().as_millis());
    }

    #[test]
    fn r_2000() {
        let edges = graph_generator::RandomLayout::new(2000).build_edges();
        let start = std::time::Instant::now();
        let _ = from_edges(&edges, &Config::default());
        println!("Random 2000 edges: {}ms", start.elapsed().as_millis());
    }

    #[test]
    fn r_4000() {
        let edges = graph_generator::RandomLayout::new(4000).build_edges();
        let start = std::time::Instant::now();
        let _ = from_edges(&edges, &Config::default());
        println!("Random 4000 edges: {}ms", start.elapsed().as_millis());
    }

    #[test]
    fn l_1000_2() {
        let n = 1000;
        let e = 2;
        let edges = graph_generator::GraphLayout::new_from_num_nodes(n, e).build_edges();
        let start = std::time::Instant::now();
        let _ = from_edges(&edges, &Config::default());
        println!(
            "{n} nodes, {e} edges per node: {}ms",
            start.elapsed().as_millis()
        );
    }

    #[test]
    fn l_2000_2() {
        let n = 2000;
        let e = 2;
        let edges = graph_generator::GraphLayout::new_from_num_nodes(n, e).build_edges();
        let start = std::time::Instant::now();
        let _ = from_edges(&edges, &Config::default());
        println!(
            "{n} nodes, {e} edges per node: {}ms",
            start.elapsed().as_millis()
        );
    }

    #[test]
    fn l_4000_2() {
        let n = 4000;
        let e = 2;
        let edges = graph_generator::GraphLayout::new_from_num_nodes(n, e).build_edges();
        let start = std::time::Instant::now();
        let _ = from_edges(&edges, &Config::default());
        println!(
            "{n} nodes, {e} edges per node: {}ms",
            start.elapsed().as_millis()
        );
    }

    #[test]
    fn l_8000_2() {
        let n = 8000;
        let e = 2;
        let edges = graph_generator::GraphLayout::new_from_num_nodes(n, e).build_edges();
        let start = std::time::Instant::now();
        let _ = from_edges(&edges, &Config::default());
        println!(
            "{n} nodes, {e} edges per node: {}ms",
            start.elapsed().as_millis()
        );
    }
}

#[cfg(test)]
mod check_visuals {

    use crate::{
        configure::{Config, RankingType},
        from_vertices_and_edges,
    };

    use super::from_edges;

    #[test]
    fn test_no_dummies() {
        let vertices = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        ];
        let edges = [
            (1, 2),
            (1, 3),
            (2, 5),
            (2, 16),
            (4, 5),
            (4, 6),
            (4, 7),
            (6, 17),
            (6, 3),
            (6, 18),
            (8, 3),
            (8, 9),
            (8, 10),
            (9, 16),
            (9, 7),
            (9, 19),
            (11, 7),
            (11, 12),
            (11, 13),
            (12, 18),
            (12, 10),
            (12, 20),
            (14, 10),
            (14, 15),
            (15, 19),
            (15, 13),
        ];
        let _ = from_vertices_and_edges(
            &vertices
                .into_iter()
                .map(|v| (v, (0.0, 0.0)))
                .collect::<Vec<_>>(),
            &edges,
            &Config {
                dummy_vertices: true,
                ..Default::default()
            },
        );
    }
    #[test]
    fn verify_looks_good() {
        // NOTE: This test might fail eventually, since the order of lements in a row canot be guaranteed;
        let edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (2, 4),
            (3, 5),
            (3, 6),
            (3, 7),
            (3, 8),
            (4, 5),
            (4, 6),
            (4, 7),
            (4, 8),
            (5, 9),
            (6, 9),
            (7, 9),
            (8, 9),
        ];
        let (layout, width, height) = &mut from_edges(&edges, &Config::default())[0];
        layout.sort_by(|a, b| a.0.cmp(&b.0));

        assert_eq!(*width, 4.0);
        assert_eq!(*height, 6.0);
        println!("{:?}", layout);
    }

    #[test]
    fn root_vertices_on_top_disabled() {
        let edges = [(1, 0), (2, 1), (3, 0), (4, 0)];
        let layout = from_edges(&edges, &Config::default());
        for (id, (_, y)) in layout[0].0.clone() {
            if id == 2 {
                assert_eq!(y, 0.0);
            } else if id == 3 || id == 4 || id == 1 {
                assert_eq!(y, 10.0);
            } else {
                assert_eq!(y, 20.0)
            }
        }
    }

    #[test]
    fn check_coords_2() {
        let edges = [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 4),
            (4, 5),
            (5, 6),
            (2, 6),
            (3, 6),
            (3, 7),
            (3, 8),
            (3, 9),
        ];
        let layout = from_edges(&edges, &Config::default());
        println!("{:?}", layout);
    }

    #[test]
    fn hlrs_ping() {
        let _nodes = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        ];
        let edges = [
            (1, 2),
            (1, 4),
            (1, 5),
            (1, 3),
            (2, 4),
            (2, 5),
            (3, 9),
            (3, 10),
            (3, 8),
            (4, 6),
            (4, 9),
            (4, 8),
            (5, 6),
            (5, 10),
            (5, 8),
            (6, 7),
            (7, 9),
            (7, 10),
            (8, 14),
            (8, 15),
            (8, 13),
            (9, 11),
            (9, 14),
            (9, 13),
            (10, 11),
            (10, 15),
            (10, 13),
            (11, 12),
            (12, 14),
            (12, 15),
            (13, 18),
            (13, 19),
            (13, 20),
            (14, 16),
            (14, 18),
            (14, 20),
            (15, 16),
            (15, 19),
            (15, 20),
            (16, 17),
            (17, 18),
            (17, 19),
            (18, 21),
            (19, 21),
        ]
        .into_iter()
        .map(|(t, h)| (t - 1, h - 1))
        .collect::<Vec<_>>();

        let layout = from_edges(
            &edges,
            &Config {
                ranking_type: RankingType::Up,
                ..Default::default()
            },
        );
        println!("{layout:?}");
    }

    #[test]
    fn run_algo_empty_graph() {
        use super::from_edges;
        let edges = [];
        let g = from_edges(&edges, &Config::default());
        assert!(g.is_empty());
    }

    #[test]
    fn run_algo_with_duplicate_edges() {
        let edges = [
            (1, 2),
            (2, 5),
            (2, 6),
            (2, 3),
            (3, 4),
            (4, 3),
            (4, 8),
            (8, 4),
            (8, 7),
            (3, 7),
            (6, 7),
            (7, 6),
            (5, 6),
            (5, 1),
        ];

        let layout = from_edges(&edges, &Config::default());
        println!("{layout:?}");
    }
}
