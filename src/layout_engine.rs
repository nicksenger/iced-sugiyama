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

#[cfg(test)]
mod tests {
    use super::{Cluster, compute_layout};

    fn initial_graph() -> (Vec<u32>, Vec<(u32, u32)>) {
        let mut rng = fastrand::Rng::with_seed(0x5EED_5EED);
        let mut nodes = vec![0_u32];
        let mut edges = Vec::new();

        for to in 1_u32..=6 {
            let max_edges = usize::min(3, to as usize);
            let edge_count = rng.usize(1..=max_edges);
            let mut connected_from = std::collections::HashSet::new();

            while connected_from.len() < edge_count {
                let from = rng.u32(0..to);
                if connected_from.insert(from) {
                    edges.push((from, to));
                }
            }

            nodes.push(to);
        }

        (nodes, edges)
    }

    fn build_clusters(nodes: &[u32]) -> Vec<Cluster> {
        let even_cluster_nodes = nodes
            .iter()
            .copied()
            .filter(|node| *node != 0 && node % 2 == 0)
            .collect::<Vec<_>>();
        let odd_cluster_nodes = nodes
            .iter()
            .copied()
            .filter(|node| *node % 2 == 1)
            .collect::<Vec<_>>();
        let all_cluster_nodes = nodes
            .iter()
            .copied()
            .filter(|node| *node != 0)
            .collect::<Vec<_>>();

        let mut clusters = Vec::new();
        let mut parent_cluster_index = None;
        if all_cluster_nodes.len() > 2 {
            parent_cluster_index = Some(clusters.len());
            clusters.push(Cluster::new(all_cluster_nodes).padding(10.0));
        }
        if odd_cluster_nodes.len() > 1 {
            let cluster = Cluster::new(odd_cluster_nodes).padding(10.0);
            clusters.push(match parent_cluster_index {
                Some(parent) => cluster.parent(parent),
                None => cluster,
            });
        }
        if even_cluster_nodes.len() > 1 {
            let cluster = Cluster::new(even_cluster_nodes).padding(10.0);
            clusters.push(match parent_cluster_index {
                Some(parent) => cluster.parent(parent),
                None => cluster,
            });
        }

        clusters
    }

    fn graph_to_dot(nodes: &[u32], edges: &[(u32, u32)], clusters: &[Cluster]) -> String {
        fn node_label(node: u32) -> String {
            if node == 0 {
                "Moar".to_string()
            } else {
                node.to_string()
            }
        }

        fn edge_label((from, to): (u32, u32)) -> String {
            format!("{from} -> {to}")
        }

        fn node_size(node: u32) -> (f64, f64) {
            let side = if node == 0 {
                100.0
            } else {
                10.0 * f64::from(node)
            }
            .max(72.0);
            (side, side)
        }

        fn dot_escape(value: &str) -> String {
            value
                .replace('\\', "\\\\")
                .replace('"', "\\\"")
                .replace('\n', "\\n")
        }

        fn edge_color(index: usize) -> &'static str {
            match index % 9 {
                0 => "#7393B3FF",
                1 => "#1434A4FF",
                2 => "#3F00FFFF",
                3 => "#1F51FFFF",
                4 => "#4682B4FF",
                5 => "#088F8FFF",
                6 => "#00A36CFF",
                7 => "#008080FF",
                _ => "#40B5ADFF",
            }
        }

        fn cluster_color(index: usize) -> &'static str {
            match index % 2 {
                0 => "#FF7043E6",
                _ => "#2E7D32E6",
            }
        }

        fn push_node_dot(dot: &mut String, indent: usize, node: u32) {
            let indent_str = "    ".repeat(indent);
            let (width, height) = node_size(node);
            dot.push_str(&format!(
                "{indent_str}{node} [label=\"{}\", width={:.6}, height={:.6}];\n",
                dot_escape(&node_label(node)),
                width / 96.0,
                height / 96.0
            ));
        }

        fn cluster_label(index: usize, cluster: &Cluster) -> String {
            match cluster.parent {
                Some(parent) => format!("cluster {index} (child of {parent})"),
                None => format!("cluster {index}"),
            }
        }

        fn push_cluster_dot(dot: &mut String, clusters: &[Cluster], index: usize, indent: usize) {
            let indent_str = "    ".repeat(indent);
            let cluster = &clusters[index];
            let color = cluster_color(index);
            dot.push_str(&format!("{indent_str}subgraph cluster_{index} {{\n"));
            dot.push_str(&format!(
                "{indent_str}    label=\"{}\";\n",
                dot_escape(&cluster_label(index, cluster))
            ));
            dot.push_str(&format!("{indent_str}    fontname=\"Times-Roman\";\n"));
            dot.push_str(&format!("{indent_str}    color=\"{color}\";\n"));
            dot.push_str(&format!("{indent_str}    fontcolor=\"{color}\";\n"));
            dot.push_str(&format!("{indent_str}    pencolor=\"{color}\";\n"));
            dot.push_str(&format!("{indent_str}    style=\"rounded\";\n"));
            if let Some(padding) = cluster.padding {
                dot.push_str(&format!("{indent_str}    margin={padding:.6};\n"));
            }

            let mut descendant_nodes = std::collections::HashSet::new();
            for child_index in 0..clusters.len() {
                if clusters[child_index].parent == Some(index) {
                    descendant_nodes.extend(clusters[child_index].nodes.iter().copied());
                }
            }

            for node in cluster
                .nodes
                .iter()
                .copied()
                .filter(|node| !descendant_nodes.contains(node))
            {
                push_node_dot(dot, indent + 1, node);
            }

            for child_index in 0..clusters.len() {
                if clusters[child_index].parent == Some(index) {
                    push_cluster_dot(dot, clusters, child_index, indent + 1);
                }
            }

            dot.push_str(&format!("{indent_str}}}\n"));
        }

        let mut dot = String::from("digraph G {\n");
        dot.push_str("    graph [\n");
        dot.push_str("        rankdir=TB,\n");
        dot.push_str("        compound=true,\n");
        dot.push_str("        dpi=96,\n");
        dot.push_str("        pad=0.520833,\n");
        dot.push_str("        nodesep=0.270833,\n");
        dot.push_str("        ranksep=0.270833\n");
        dot.push_str("    ];\n");
        dot.push_str("    node [shape=box, style=\"rounded,filled\", fillcolor=\"#f5f5f5\", color=\"#444444\", fixedsize=true, fontname=\"Times-Roman\"];\n");
        dot.push_str(
            "    edge [dir=both, arrowtail=odot, arrowhead=normal, fontname=\"Times-Roman\"];\n",
        );

        let clustered_nodes = clusters
            .iter()
            .flat_map(|cluster| cluster.nodes.iter().copied())
            .collect::<std::collections::HashSet<_>>();

        for node in nodes
            .iter()
            .copied()
            .filter(|node| !clustered_nodes.contains(node))
        {
            push_node_dot(&mut dot, 1, node);
        }

        for cluster_index in 0..clusters.len() {
            if clusters[cluster_index].parent.is_none() {
                push_cluster_dot(&mut dot, clusters, cluster_index, 1);
            }
        }

        for (index, edge) in edges.iter().copied().enumerate() {
            let color = edge_color(index);
            dot.push_str(&format!(
                "    {} -> {} [label=\"{}\", color=\"{}\", fontcolor=\"{}\", minlen=1];\n",
                edge.0,
                edge.1,
                dot_escape(&edge_label(edge)),
                color,
                color
            ));
        }

        dot.push_str("}\n");
        dot
    }

    #[test]
    fn dump_moar_reference_layout() {
        let (nodes, edges) = initial_graph();
        let clusters = build_clusters(&nodes);
        let layout = compute_layout(
            &nodes,
            &edges,
            &rust_sugiyama::configure::Config {
                vertex_spacing: 26.0,
                ..Default::default()
            },
            |node| {
                let side = if node == 0 {
                    100.0
                } else {
                    10.0 * f64::from(node)
                }
                .max(72.0);
                (side, side)
            },
            |_, (from, to)| Some(format!("{from} -> {to}")),
            &clusters,
            &rust_sugiyama::advanced::RenderConfig {
                routing_padding: 4.0,
                bend_penalty: 6.0,
                cluster_padding: 10.0,
                cluster_constraint_iterations: 4,
                cluster_boundary_gap: 8.0,
            },
        );

        eprintln!("nodes={nodes:?}");
        eprintln!("edges={edges:?}");
        eprintln!("dot:\n{}", graph_to_dot(&nodes, &edges, &clusters));
        eprintln!("layout.max=({}, {})", layout.max_x, layout.max_y);
        for (index, (x, y)) in &layout.coords {
            eprintln!("node[{index}] center=({x:.3}, {y:.3})");
        }
        for cluster in &layout.clusters {
            eprintln!(
                "cluster[{}] parent={:?} bounds=({:.3}, {:.3})..({:.3}, {:.3})",
                cluster.index,
                cluster.parent,
                cluster.min_x,
                cluster.min_y,
                cluster.max_x,
                cluster.max_y
            );
        }
        for edge in &layout.edges {
            eprintln!(
                "edge[{}] label={:?} label_pos={:?} points={:?} curve_points={:?}",
                edge.index, edge.label, edge.label_position, edge.points, edge.curve_points
            );
        }
    }
}
