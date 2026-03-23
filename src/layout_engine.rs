use std::fmt::Write as _;
use std::hash::{Hash, Hasher};

pub(crate) use rust_sugiyama::{ClusterLayout, EdgeLayout, GraphLayout};

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

fn core_clusters(clusters: &[Cluster]) -> Vec<rust_sugiyama::Cluster> {
    clusters
        .iter()
        .map(|cluster| {
            let mut converted = rust_sugiyama::Cluster::new(cluster.nodes.clone());
            if let Some(padding) = cluster.padding {
                converted = converted.padding(padding);
            }
            if let Some(parent) = cluster.parent {
                converted = converted.parent(parent);
            }
            converted
        })
        .collect()
}

pub(crate) fn compute_layout(
    nodes: &[u32],
    edges: &[(u32, u32)],
    config: &rust_sugiyama::Config,
    node_size: impl Fn(u32) -> (f64, f64),
    edge_label: impl Fn(usize, (u32, u32)) -> Option<String>,
    clusters: &[Cluster],
    render_config: &rust_sugiyama::RenderConfig,
) -> GraphLayout {
    let clusters = core_clusters(clusters);
    rust_sugiyama::layout_graph(
        nodes,
        edges,
        config,
        node_size,
        edge_label,
        &clusters,
        render_config,
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

pub fn graphviz_plain_layout<NodeSize, NodeLabel, EdgeLabel, ClusterLabel>(
    nodes: &[u32],
    edges: &[(u32, u32)],
    config: &rust_sugiyama::Config,
    node_size: NodeSize,
    node_label: NodeLabel,
    edge_label: EdgeLabel,
    clusters: &[Cluster],
    render_config: &rust_sugiyama::RenderConfig,
    cluster_label: ClusterLabel,
) -> String
where
    NodeSize: Fn(u32) -> (f64, f64),
    NodeLabel: Fn(u32) -> String,
    EdgeLabel: Fn(usize, (u32, u32)) -> Option<String>,
    ClusterLabel: Fn(usize, &Cluster) -> String,
{
    const PLAIN_UNITS_PER_INCH: f64 = 96.0;

    fn to_plain_x(value: f64) -> f64 {
        value / PLAIN_UNITS_PER_INCH
    }

    fn to_plain_y(layout: &GraphLayout, value: f64) -> f64 {
        (layout.max_y() - value) / PLAIN_UNITS_PER_INCH
    }

    fn format_plain_number(value: f64) -> String {
        let mut value = if value.abs() < 0.000_005 { 0.0 } else { value };
        if value == -0.0 {
            value = 0.0;
        }

        let mut text = format!("{value:.5}");
        while text.contains('.') && text.ends_with('0') {
            text.pop();
        }
        if text.ends_with('.') {
            text.pop();
        }
        if text.is_empty() {
            "0".to_string()
        } else {
            text
        }
    }

    fn format_plain_label(value: &str) -> String {
        let simple = !value.is_empty()
            && value
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.' | '-'));
        if simple {
            value.to_string()
        } else {
            format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""))
        }
    }

    let layout = compute_layout(
        nodes,
        edges,
        config,
        &node_size,
        &edge_label,
        clusters,
        render_config,
    );

    let mut plain = String::new();
    let _ = writeln!(
        plain,
        "graph 1 {} {}",
        format_plain_number(to_plain_x(layout.max_x())),
        format_plain_number(to_plain_x(layout.max_y()))
    );

    let mut sorted_clusters = layout.clusters().to_vec();
    sorted_clusters.sort_by_key(|cluster| cluster.index());
    for cluster_layout in sorted_clusters {
        let Some(cluster) = clusters.get(cluster_layout.index()) else {
            continue;
        };
        let parent = cluster_layout
            .parent()
            .map(|parent| parent.to_string())
            .unwrap_or_else(|| "_".to_string());
        let min_y = to_plain_y(&layout, cluster_layout.max_y());
        let max_y = to_plain_y(&layout, cluster_layout.min_y());
        let _ = writeln!(
            plain,
            "cluster {} {} {} {} {} {} {}",
            cluster_layout.index(),
            parent,
            format_plain_number(to_plain_x(cluster_layout.min_x())),
            format_plain_number(min_y),
            format_plain_number(to_plain_x(cluster_layout.max_x())),
            format_plain_number(max_y),
            format_plain_label(&cluster_label(cluster_layout.index(), cluster)),
        );
    }

    for (position, node) in nodes.iter().copied().enumerate() {
        let Some((x, y)) = layout.position(position) else {
            continue;
        };
        let (width, height) = node_size(node);
        let _ = writeln!(
            plain,
            "node {node} {} {} {} {} {}",
            format_plain_number(to_plain_x(x)),
            format_plain_number(to_plain_y(&layout, y)),
            format_plain_number(to_plain_x(width)),
            format_plain_number(to_plain_x(height)),
            format_plain_label(&node_label(node)),
        );
    }

    let mut sorted_edges = layout.edges().to_vec();
    sorted_edges.sort_by_key(|edge| edge.index());
    for edge in sorted_edges {
        let (tail, head) = edges.get(edge.index()).copied().unwrap_or((0, 0));
        let points = if edge.curve_points().is_empty() {
            edge.points()
        } else {
            edge.curve_points()
        };

        let _ = write!(plain, "edge {tail} {head} {}", points.len());
        for &(x, y) in points {
            let _ = write!(
                plain,
                " {} {}",
                format_plain_number(to_plain_x(x)),
                format_plain_number(to_plain_y(&layout, y))
            );
        }

        if let (Some(label), Some((label_x, label_y))) = (edge.label(), edge.label_position()) {
            let _ = write!(
                plain,
                " {} {} {}",
                format_plain_label(label),
                format_plain_number(to_plain_x(label_x)),
                format_plain_number(to_plain_y(&layout, label_y))
            );
        }

        let _ = writeln!(plain);
    }

    plain.push_str("stop\n");
    plain
}
