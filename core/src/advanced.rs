use petgraph::stable_graph::{EdgeIndex, NodeIndex, StableDiGraph};

use crate::configure::Config;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    pub min_x: f64,
    pub min_y: f64,
    pub max_x: f64,
    pub max_y: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RoutedNode<T> {
    pub id: T,
    pub center: (f64, f64),
    pub size: (f64, f64),
    pub bounds: Rect,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EdgeLabel<L> {
    pub value: L,
    pub position: (f64, f64),
}

#[derive(Debug, Clone, PartialEq)]
pub struct RoutedEdge<T, L> {
    pub id: EdgeIndex,
    pub tail: T,
    pub head: T,
    pub points: Vec<(f64, f64)>,
    pub curve_points: Vec<(f64, f64)>,
    pub label: Option<EdgeLabel<L>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClusterSpec<C> {
    pub id: C,
    pub nodes: Vec<NodeIndex>,
    pub padding: Option<f64>,
    pub parent: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClusterLayout<C> {
    pub id: C,
    pub bounds: Rect,
    pub parent: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderConfig {
    /// Padding around node bounds used as routing obstacles.
    pub routing_padding: f64,
    /// Penalty applied when route direction changes.
    pub bend_penalty: f64,
    /// Default cluster padding when [ClusterSpec::padding] is [None].
    pub cluster_padding: f64,
    /// Number of iterative passes used to enforce cluster ordering constraints.
    pub cluster_constraint_iterations: usize,
    /// Additional spacing inserted when the layout crosses cluster boundaries.
    pub cluster_boundary_gap: f64,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            routing_padding: 2.0,
            bend_penalty: 8.0,
            cluster_padding: 8.0,
            cluster_constraint_iterations: 4,
            cluster_boundary_gap: 6.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DetailedLayout<T, L, C> {
    pub nodes: Vec<RoutedNode<T>>,
    pub edges: Vec<RoutedEdge<T, L>>,
    pub clusters: Vec<ClusterLayout<C>>,
    pub width: f64,
    pub height: f64,
}

/// Creates layouts with routed edges, optional edge label placements and
/// optional cluster rectangles.
///
/// This keeps the current layered node placement and adds post-processing
/// features that are commonly needed by renderers.
pub fn from_graph_with_features<V, E, L, C>(
    graph: &StableDiGraph<V, E>,
    vertex_size: &impl Fn(NodeIndex, &V) -> (f64, f64),
    edge_label: &impl Fn(EdgeIndex, &E) -> Option<L>,
    clusters: &[ClusterSpec<C>],
    config: &Config,
    render_config: &RenderConfig,
) -> Vec<DetailedLayout<NodeIndex, L, C>>
where
    L: Clone,
    C: Clone,
{
    vec![]
}
