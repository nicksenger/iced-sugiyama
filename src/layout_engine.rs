use std::collections::{BTreeMap, HashMap, HashSet};
use std::hash::{Hash, Hasher};

use petgraph::stable_graph::{NodeIndex, StableDiGraph};
use rust_sugiyama::advanced::{
    ClusterSpec, DetailedLayout, RenderConfig, RoutedEdge, RoutedNode, from_graph_with_features,
};

const DEFAULT_NODE_SIZE: (f64, f64) = (56.0, 32.0);
const COMPONENT_GAP: f64 = 80.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeEndpointKind {
    Source,
    Destination,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EdgeEndpoint {
    pub direction_x: f32,
    pub direction_y: f32,
}

impl EdgeEndpoint {
    pub fn angle_radians(self) -> f32 {
        self.direction_y.atan2(self.direction_x)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Cluster {
    pub nodes: Vec<u32>,
    pub padding: Option<f64>,
    pub parent: Option<usize>,
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
}

#[derive(Clone)]
pub(crate) struct EdgeLayout {
    pub index: usize,
    pub points: Vec<(f64, f64)>,
    pub curve_points: Vec<(f64, f64)>,
    pub label: Option<String>,
    pub label_position: Option<(f64, f64)>,
}

#[derive(Clone)]
pub(crate) struct ClusterLayout {
    pub index: usize,
    pub parent: Option<usize>,
    pub min_x: f64,
    pub min_y: f64,
    pub max_x: f64,
    pub max_y: f64,
}

#[derive(Clone)]
pub(crate) struct GraphLayout {
    pub max_x: f64,
    pub max_y: f64,
    pub coords: BTreeMap<usize, (f64, f64)>,
    pub edges: Vec<EdgeLayout>,
    pub clusters: Vec<ClusterLayout>,
}

impl GraphLayout {
    pub fn empty() -> Self {
        Self {
            max_x: 0.0,
            max_y: 1.0,
            coords: BTreeMap::new(),
            edges: Vec::new(),
            clusters: Vec::new(),
        }
    }

    #[allow(dead_code)]
    pub fn avg(&self, other: &Self, weight_other: f64) -> (Self, HashSet<usize>, HashSet<usize>) {
        let own_keys = self.coords.keys().collect::<HashSet<_>>();
        let other_keys = other.coords.keys().collect::<HashSet<_>>();

        let lost = own_keys
            .difference(&other_keys)
            .copied()
            .copied()
            .collect::<HashSet<usize>>();
        let gained = other_keys
            .difference(&own_keys)
            .copied()
            .copied()
            .collect::<HashSet<usize>>();

        let coords = own_keys
            .intersection(&other_keys)
            .filter_map(|k| {
                let a = self.coords.get(k)?;
                let b = other.coords.get(k)?;

                Some((
                    **k,
                    (
                        (b.0 - a.0) * weight_other + a.0,
                        (b.1 - a.1) * weight_other + a.1,
                    ),
                ))
            })
            .chain(
                lost.iter()
                    .filter_map(|k| self.coords.get(k).copied().map(|pos| (*k, pos))),
            )
            .chain(
                gained
                    .iter()
                    .filter_map(|k| other.coords.get(k).copied().map(|pos| (*k, pos))),
            )
            .collect();

        (
            Self {
                max_x: self.max_x + (other.max_x - self.max_x) * weight_other,
                max_y: self.max_y + (other.max_y - self.max_y) * weight_other,
                coords,
                edges: other.edges.clone(),
                clusters: other.clusters.clone(),
            },
            lost,
            gained,
        )
    }
}

pub(crate) fn compute_layout(
    nodes: &[u32],
    edges: &[(u32, u32)],
    config: &rust_sugiyama::configure::Config,
    node_size: impl Fn(u32) -> (f64, f64),
    edge_label: impl Fn(usize, (u32, u32)) -> Option<String>,
    clusters: &[Cluster],
    render_config: &RenderConfig,
) -> GraphLayout {
    if nodes.is_empty() {
        return GraphLayout::empty();
    }

    let mut graph_node_indices = HashMap::<u32, NodeIndex>::new();
    let mut g = StableDiGraph::<u32, usize>::new();
    for node in nodes {
        let index = g.add_node(*node);
        graph_node_indices.insert(*node, index);
    }

    for (edge_index, (from, to)) in edges.iter().copied().enumerate() {
        let (Some(from_idx), Some(to_idx)) = (
            graph_node_indices.get(&from).copied(),
            graph_node_indices.get(&to).copied(),
        ) else {
            continue;
        };
        g.add_edge(from_idx, to_idx, edge_index);
    }

    let cluster_specs = clusters
        .iter()
        .enumerate()
        .map(|(cluster_index, cluster)| ClusterSpec {
            id: cluster_index,
            nodes: cluster
                .nodes
                .iter()
                .filter_map(|node| graph_node_indices.get(node).copied())
                .collect(),
            padding: cluster.padding,
            parent: cluster.parent,
        })
        .collect::<Vec<_>>();

    let detailed = from_graph_with_features(
        &g,
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

    merge_components(
        &g,
        detailed,
        &nodes
            .iter()
            .enumerate()
            .map(|(index, node)| (*node, index))
            .collect(),
    )
}

pub(crate) fn layout_signature(nodes: &[u32], edges: &[(u32, u32)], clusters: &[Cluster]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    nodes.hash(&mut hasher);
    edges.hash(&mut hasher);
    clusters.len().hash(&mut hasher);
    for cluster in clusters {
        cluster.nodes.hash(&mut hasher);
        cluster.padding.map(f64::to_bits).hash(&mut hasher);
        cluster.parent.hash(&mut hasher);
    }
    hasher.finish()
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
    edges.push(EdgeLayout {
        index: *graph.edge_weight(edge.id).unwrap_or(&edge.id.index()),
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
