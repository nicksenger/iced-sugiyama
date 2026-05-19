// Legacy backend internals are still kept during the backend transition.
#![allow(dead_code)]

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::env;

use log::{error, info};
use petgraph::stable_graph::{NodeIndex, StableDiGraph};

mod ported;

mod configure {
    pub use crate::{Config, CrossingMinimization, RankingType};
}

mod util {
    pub use crate::ported::util::*;
}

type Layout = (Vec<(usize, (f64, f64))>, f64, f64);
type Layouts<T> = Vec<(Vec<(T, (f64, f64))>, f64, f64)>;

const DEFAULT_NODE_SIZE: (f64, f64) = (56.0, 32.0);
const COMPONENT_GAP: f64 = 80.0;
const MINIMUM_LENGTH_DEFAULT: u32 = 1;
const VERTEX_SPACING_DEFAULT: f64 = 10.0;
const DUMMY_VERTICES_DEFAULT: bool = true;
const RANKING_TYPE_DEFAULT: RankingType = RankingType::Hybrid;
const HYBRID_WEIGHT_DEFAULT: f64 = 0.5;
const C_MINIMIZATION_DEFAULT: CrossingMinimization = CrossingMinimization::Median;
const TRANSPOSE_DEFAULT: bool = true;
const DUMMY_SIZE_DEFAULT: f64 = 1.0;

const ENV_MINIMUM_LENGTH: &str = "RUST_GRAPH_MIN_LEN";
const ENV_VERTEX_SPACING: &str = "RUST_GRAPH_V_SPACING";
const ENV_DUMMY_VERTICES: &str = "RUST_GRAPH_DUMMIES";
const ENV_RANKING_TYPE: &str = "RUST_GRAPH_R_TYPE";
const ENV_RANKING_HYBRID_WEIGHT: &str = "RUST_GRAPH_R_HYBRID_WEIGHT";
const ENV_CROSSING_MINIMIZATION: &str = "RUST_GRAPH_CROSS_MIN";
const ENV_TRANSPOSE: &str = "RUST_GRAPH_TRANSPOSE";
const ENV_DUMMY_SIZE: &str = "RUST_GRAPH_DUMMY_SIZE";

macro_rules! read_env {
    ($field:expr, $cb:tt, $env:ident) => {
        #[allow(unused_parens)]
        match env::var($env).map($cb) {
            Ok(Ok(v)) => $field = v,
            Ok(Err(e)) => {
                error!(target: "initialization", "{e}");
            }
            _ => (),
        }
    };
}

/// Used to configure parameters of the graph layout.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// Length between layers.
    pub minimum_length: u32,
    /// The minimum spacing between vertices on the same layer and between
    /// layers.
    pub vertex_spacing: f64,
    /// Whether to include dummy vertices when calculating the layout.
    pub dummy_vertices: bool,
    /// How much space a dummy should take up, as a multiplier of the
    /// [`Self::vertex_spacing`].
    pub dummy_size: f64,
    /// Defines how vertices are placed vertically.
    pub ranking_type: RankingType,
    /// Blend factor for [`RankingType::Hybrid`] where `0.0` is equivalent to
    /// [`RankingType::MinimizeEdgeLength`] and `1.0` is equivalent to
    /// [`RankingType::Original`].
    pub hybrid_weight: f64,
    /// Which heuristic to use when minimizing edge crossings.
    pub c_minimization: CrossingMinimization,
    /// Whether to attempt to further reduce crossings by swapping vertices in a
    /// layer. This may increase runtime significantly.
    pub transpose: bool,
}

impl Config {
    /// Read in configuration values from environment variables.
    pub fn new_from_env() -> Self {
        let mut config = Self::default();

        let parse_bool = |x: String| match x.as_str() {
            "y" => Ok(true),
            "n" => Ok(false),
            v => Err(format!("Invalid argument for dummy vertex env: {v}")),
        };
        let parse_hybrid_weight = |x: String| {
            let value = x.parse::<f64>().map_err(|e| {
                format!("Invalid argument for hybrid ranking weight env '{x}': {e}")
            })?;
            if !(0.0..=1.0).contains(&value) {
                return Err(format!(
                    "Invalid argument for hybrid ranking weight env '{x}': expected value in [0.0, 1.0]"
                ));
            }
            Ok(value)
        };

        read_env!(
            config.minimum_length,
            (|x| x.parse::<u32>()),
            ENV_MINIMUM_LENGTH
        );
        read_env!(
            config.c_minimization,
            (TryFrom::try_from),
            ENV_CROSSING_MINIMIZATION
        );
        read_env!(config.ranking_type, (TryFrom::try_from), ENV_RANKING_TYPE);
        read_env!(
            config.vertex_spacing,
            (|x| x.parse::<f64>()),
            ENV_VERTEX_SPACING
        );
        read_env!(
            config.hybrid_weight,
            parse_hybrid_weight,
            ENV_RANKING_HYBRID_WEIGHT
        );
        read_env!(config.dummy_vertices, parse_bool, ENV_DUMMY_VERTICES);
        read_env!(config.dummy_size, (|x| x.parse::<f64>()), ENV_DUMMY_SIZE);
        read_env!(config.transpose, parse_bool, ENV_TRANSPOSE);

        config
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            minimum_length: MINIMUM_LENGTH_DEFAULT,
            vertex_spacing: VERTEX_SPACING_DEFAULT,
            dummy_vertices: DUMMY_VERTICES_DEFAULT,
            dummy_size: DUMMY_SIZE_DEFAULT,
            ranking_type: RANKING_TYPE_DEFAULT,
            hybrid_weight: HYBRID_WEIGHT_DEFAULT,
            c_minimization: C_MINIMIZATION_DEFAULT,
            transpose: TRANSPOSE_DEFAULT,
        }
    }
}

/// Defines the Ranking type, i.e. how vertices are placed on each layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RankingType {
    Original,
    MinimizeEdgeLength,
    Hybrid,
    Up,
    Down,
}

impl TryFrom<String> for RankingType {
    type Error = String;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        match value.as_str() {
            "original" => Ok(Self::Original),
            "minimize" => Ok(Self::MinimizeEdgeLength),
            "hybrid" => Ok(Self::Hybrid),
            "up" => Ok(Self::Up),
            "down" => Ok(Self::Down),
            s => Err(format!("invalid value for ranking type: {s}")),
        }
    }
}

impl From<RankingType> for &'static str {
    fn from(value: RankingType) -> Self {
        match value {
            RankingType::Up => "up",
            RankingType::Down => "down",
            RankingType::Original => "original",
            RankingType::MinimizeEdgeLength => "minimize",
            RankingType::Hybrid => "hybrid",
        }
    }
}

/// Defines the heuristic used for crossing minimization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CrossingMinimization {
    Barycenter,
    Median,
}

impl TryFrom<String> for CrossingMinimization {
    type Error = String;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        match value.as_str() {
            "barycenter" => Ok(Self::Barycenter),
            "median" => Ok(Self::Median),
            s => Err(format!("invalid value for crossing minimization: {s}")),
        }
    }
}

impl From<CrossingMinimization> for &'static str {
    fn from(value: CrossingMinimization) -> Self {
        match value {
            CrossingMinimization::Median => "median",
            CrossingMinimization::Barycenter => "barycenter",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderConfig {
    pub routing_padding: f64,
    pub bend_penalty: f64,
    pub cluster_padding: f64,
    pub cluster_constraint_iterations: usize,
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

fn resolve_ported_config(config: &Config) -> Config {
    let mut resolved = *config;
    if matches!(resolved.ranking_type, RankingType::Hybrid) {
        resolved.ranking_type = if resolved.hybrid_weight >= 0.5 {
            RankingType::Original
        } else {
            RankingType::MinimizeEdgeLength
        };
    }
    resolved
}

fn from_graph<V, E>(
    graph: &StableDiGraph<V, E>,
    vertex_size: &impl Fn(NodeIndex, &V) -> (f64, f64),
    config: &Config,
) -> Layouts<NodeIndex> {
    let graph = graph.map(
        |id, v| {
            let (w, h) = sanitize_node_size(vertex_size(id, v));
            ported::algorithm::Vertex::new(id.index(), (w, h))
        },
        |_, _| ported::algorithm::Edge::default(),
    );

    ported::algorithm::start(graph, config)
        .into_iter()
        .map(|(layout, width, height)| {
            (
                layout
                    .into_iter()
                    .map(|(id, coords)| (NodeIndex::from(id as u32), coords))
                    .collect(),
                width,
                height,
            )
        })
        .collect()
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

    info!(target: "layout", "Starting phase 0 [build_graph]");
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
        .map(|(cluster_index, cluster)| ported::advanced::ClusterSpec {
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

    let resolved_config = resolve_ported_config(config);
    let render_config = ported::advanced::RenderConfig {
        routing_padding: render_config.routing_padding,
        bend_penalty: render_config.bend_penalty,
        cluster_padding: render_config.cluster_padding,
        cluster_constraint_iterations: render_config.cluster_constraint_iterations,
        cluster_boundary_gap: render_config.cluster_boundary_gap,
    };
    let detailed = ported::advanced::from_graph_with_features(
        &graph,
        &|_, node| sanitize_node_size(node_size(*node)),
        &|_, edge_idx| {
            let edge_index = *edge_idx;
            edges
                .get(edge_index)
                .and_then(|edge| edge_label(edge_index, *edge))
        },
        &cluster_specs,
        &resolved_config,
        &render_config,
    );

    let node_to_position: HashMap<u32, usize> = nodes
        .iter()
        .enumerate()
        .map(|(position, node)| (*node, position))
        .collect();

    let mut coords = BTreeMap::new();
    let mut merged_edges = Vec::new();
    let mut merged_clusters = Vec::new();
    let mut x_offset = 0.0;
    let mut max_y = 1.0f64;
    let total_components = detailed.len();
    for (component_idx, component) in detailed.into_iter().enumerate() {
        for node in component.nodes {
            let graph_node = graph[node.id];
            if let Some(position) = node_to_position.get(&graph_node).copied() {
                coords.insert(position, (node.center.0 + x_offset, node.center.1));
            }
        }

        for edge in component.edges {
            let index = *graph.edge_weight(edge.id).unwrap_or(&edge.id.index());
            let points: Vec<(f64, f64)> = edge
                .points
                .into_iter()
                .map(|(x, y)| (x + x_offset, y))
                .collect();
            let curve_points = polyline_to_curve(&points);
            merged_edges.push(EdgeLayout {
                index,
                points,
                curve_points,
                label: edge.label.as_ref().map(|label| label.value.clone()),
                label_position: edge
                    .label
                    .as_ref()
                    .map(|label| (label.position.0 + x_offset, label.position.1)),
            });
        }

        for cluster in component.clusters {
            merged_clusters.push(ClusterLayout {
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
        edges: merged_edges,
        clusters: merged_clusters,
    }
}

#[derive(Debug, Clone)]
struct SizedNode {
    width: f64,
    height: f64,
}

#[derive(Debug, Clone)]
struct InputEdge {
    index: usize,
    tail: usize,
    head: usize,
    min_len: i32,
    label: Option<String>,
}

#[derive(Debug, Clone)]
struct DirectedEdge {
    index: usize,
    tail: usize,
    head: usize,
    min_len: i32,
    base_min_len: i32,
    weight: i32,
    label: Option<String>,
    original_tail: usize,
    original_head: usize,
    flat: bool,
    cluster_crossings: usize,
    temp_kind: TempEdgeKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TempEdgeKind {
    Original,
    Reversed,
    ClusterBridge,
    Flat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LayerNodeKind {
    Real,
    Dummy { edge_index: usize, step: usize },
    FlatVirtual { edge_index: usize },
}

#[derive(Debug, Clone)]
struct LayerNode {
    rank: usize,
    order: usize,
    width: f64,
    height: f64,
    x: f64,
    y: f64,
    real_node: Option<usize>,
    cluster_path: Vec<usize>,
    kind: LayerNodeKind,
    median_value: f64,
}

#[derive(Debug, Clone, Copy)]
struct SegmentEdge {
    from: usize,
    to: usize,
    edge_index: usize,
    weight: i32,
    cluster_crossings: usize,
    flat: bool,
    temp_kind: TempEdgeKind,
}

#[derive(Debug, Clone)]
struct RoutedEdge {
    index: usize,
    points: Vec<(f64, f64)>,
    label: Option<String>,
    label_position: Option<(f64, f64)>,
}

#[derive(Debug, Clone)]
struct ComponentLayout {
    node_positions: HashMap<usize, (f64, f64)>,
    edges: Vec<RoutedEdge>,
    width: f64,
}

#[derive(Debug, Clone, Copy)]
struct Bounds {
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
}

#[derive(Debug, Clone, Copy)]
struct FlatConstraint {
    left: usize,
    right: usize,
    min_gap: f64,
}

fn sanitize_node_size((width, height): (f64, f64)) -> (f64, f64) {
    let width = if width.is_finite() && width > 0.0 {
        width
    } else {
        DEFAULT_NODE_SIZE.0
    };
    let height = if height.is_finite() && height > 0.0 {
        height
    } else {
        DEFAULT_NODE_SIZE.1
    };
    (width, height)
}

fn connected_components(
    node_count: usize,
    edges: &[InputEdge],
    node_cluster_paths: &[Vec<usize>],
) -> Vec<Vec<usize>> {
    let mut adjacency = vec![Vec::<usize>::new(); node_count];
    for edge in edges {
        adjacency[edge.tail].push(edge.head);
        adjacency[edge.head].push(edge.tail);
    }

    // Cluster-aware decomposition: nodes participating in the same cluster
    // skeleton should be processed in the same component, mirroring how dotgen
    // integrates cluster structure before decomposition.
    let mut cluster_anchor = HashMap::<usize, usize>::new();
    for (node, path) in node_cluster_paths.iter().enumerate() {
        for &cluster in path {
            if let Some(&anchor) = cluster_anchor.get(&cluster) {
                adjacency[node].push(anchor);
                adjacency[anchor].push(node);
            } else {
                cluster_anchor.insert(cluster, node);
            }
        }
    }

    let mut visited = vec![false; node_count];
    let mut components = Vec::new();

    for root in 0..node_count {
        if visited[root] {
            continue;
        }
        let mut queue = VecDeque::new();
        queue.push_back(root);
        visited[root] = true;
        let mut component = Vec::new();

        while let Some(node) = queue.pop_front() {
            component.push(node);
            for &neighbor in &adjacency[node] {
                if visited[neighbor] {
                    continue;
                }
                visited[neighbor] = true;
                queue.push_back(neighbor);
            }
        }

        component.sort_unstable();
        components.push(component);
    }

    components.sort_by_key(|component| component.first().copied().unwrap_or(usize::MAX));
    components
}

fn layout_component(
    nodes: &[SizedNode],
    component_nodes: &[usize],
    component_edges: &[InputEdge],
    config: &Config,
    render_config: &RenderConfig,
    node_cluster_paths: &[Vec<usize>],
) -> ComponentLayout {
    let mut local_index = HashMap::new();
    let mut reverse_local = Vec::with_capacity(component_nodes.len());
    for (local, &global_node) in component_nodes.iter().enumerate() {
        local_index.insert(global_node, local);
        reverse_local.push(global_node);
    }
    let local_count = reverse_local.len();

    let mut acyclic_edges = Vec::with_capacity(component_edges.len());
    for edge in component_edges {
        let Some(&tail) = local_index.get(&edge.tail) else {
            continue;
        };
        let Some(&head) = local_index.get(&edge.head) else {
            continue;
        };
        let flat = edge.tail == edge.head;
        let base_min_len = if flat { 0 } else { edge.min_len.max(1) };
        let (cluster_crossings, cluster_weight_penalty, cluster_len_penalty, temp_kind) =
            classify_cluster_edge(
                &node_cluster_paths[edge.tail],
                &node_cluster_paths[edge.head],
                flat,
            );
        acyclic_edges.push(DirectedEdge {
            index: edge.index,
            tail,
            head,
            min_len: base_min_len + cluster_len_penalty,
            base_min_len,
            weight: (1 + cluster_weight_penalty).max(1),
            label: edge.label.clone(),
            original_tail: tail,
            original_head: head,
            flat,
            cluster_crossings,
            temp_kind,
        });
    }
    break_cycles_acyclic(local_count, &mut acyclic_edges);

    let oriented_edges: Vec<DirectedEdge> = acyclic_edges
        .iter()
        .map(|edge| {
            if matches!(config.ranking_type, RankingType::Up) {
                DirectedEdge {
                    index: edge.index,
                    tail: edge.head,
                    head: edge.tail,
                    min_len: edge.min_len,
                    base_min_len: edge.base_min_len,
                    weight: edge.weight,
                    label: edge.label.clone(),
                    original_tail: edge.original_tail,
                    original_head: edge.original_head,
                    flat: edge.flat,
                    cluster_crossings: edge.cluster_crossings,
                    temp_kind: TempEdgeKind::Reversed,
                }
            } else {
                edge.clone()
            }
        })
        .collect();

    info!(target: "layout", "Starting phase 1 [dot_rank]");
    let mut ranks = assign_ranks(
        local_count,
        &oriented_edges,
        config.ranking_type,
        config.hybrid_weight,
    );
    normalize_ranks(&mut ranks);
    // dot/flat.c reserves an extra rank above rank 0 when non-adjacent flat
    // labels need virtual label nodes.
    let needs_flat_label_above_zero = oriented_edges
        .iter()
        .any(|edge| edge.flat && edge.label.is_some() && ranks[edge.tail] == 0);
    if needs_flat_label_above_zero {
        for rank in &mut ranks {
            *rank += 1;
        }
    }

    info!(target: "layout", "Starting phase 2 [dot_mincross]");
    let mut layer_nodes = Vec::<LayerNode>::new();
    let mut layers = vec![Vec::<usize>::new(); (max_rank(&ranks) + 1).max(1)];
    let mut real_layer_node = vec![0usize; local_count];
    for local in 0..local_count {
        let global = reverse_local[local];
        let rank = ranks[local].max(0) as usize;
        while rank >= layers.len() {
            layers.push(Vec::new());
        }
        let id = layer_nodes.len();
        let cluster_path = node_cluster_paths[global].clone();
        layer_nodes.push(LayerNode {
            rank,
            order: 0,
            width: nodes[global].width,
            height: nodes[global].height,
            x: 0.0,
            y: 0.0,
            real_node: Some(local),
            cluster_path,
            kind: LayerNodeKind::Real,
            median_value: 0.0,
        });
        layers[rank].push(id);
        real_layer_node[local] = id;
    }

    let mut segment_edges = Vec::<SegmentEdge>::new();
    let mut chains: HashMap<usize, Vec<usize>> = HashMap::new();
    for edge in &oriented_edges {
        let mut tail_rank = ranks[edge.tail];
        let mut head_rank = ranks[edge.head];
        if !edge.flat && head_rank <= tail_rank {
            head_rank = tail_rank + 1;
            ranks[edge.head] = head_rank;
        }
        if head_rank < 0 {
            continue;
        }
        tail_rank = tail_rank.max(0);
        let tail_layer = real_layer_node[edge.tail];
        let head_layer = real_layer_node[edge.head];
        let diff = (head_rank - tail_rank) as usize;
        if diff == 0 {
            if edge.flat {
                let mut chain = vec![tail_layer];
                if edge.label.is_some() {
                    let edge_rank = tail_rank as usize;
                    let rank = edge_rank.saturating_sub(1);
                    while rank >= layers.len() {
                        layers.push(Vec::new());
                    }
                    let label_width = edge
                        .label
                        .as_ref()
                        .map(|label| estimate_label_width(label).max(config.vertex_spacing))
                        .unwrap_or(config.vertex_spacing);
                    let label_height = (config.vertex_spacing * config.dummy_size).max(1.0);
                    let tail_path = &node_cluster_paths[reverse_local[edge.tail]];
                    let head_path = &node_cluster_paths[reverse_local[edge.head]];
                    let shared = shared_prefix_len(tail_path, head_path);
                    let cluster_path = tail_path[..shared].to_vec();
                    let label_node = layer_nodes.len();
                    layer_nodes.push(LayerNode {
                        rank,
                        order: 0,
                        width: label_width,
                        height: label_height,
                        x: 0.0,
                        y: 0.0,
                        real_node: None,
                        cluster_path,
                        kind: LayerNodeKind::FlatVirtual {
                            edge_index: edge.index,
                        },
                        median_value: 0.0,
                    });
                    layers[rank].push(label_node);
                    let label_segments_flat = rank == edge_rank;
                    if rank <= edge_rank {
                        segment_edges.push(SegmentEdge {
                            from: label_node,
                            to: tail_layer,
                            edge_index: edge.index,
                            weight: (edge.weight + 1).max(1),
                            cluster_crossings: edge.cluster_crossings,
                            flat: label_segments_flat,
                            temp_kind: edge.temp_kind,
                        });
                        segment_edges.push(SegmentEdge {
                            from: label_node,
                            to: head_layer,
                            edge_index: edge.index,
                            weight: (edge.weight + 1).max(1),
                            cluster_crossings: edge.cluster_crossings,
                            flat: label_segments_flat,
                            temp_kind: edge.temp_kind,
                        });
                    } else {
                        segment_edges.push(SegmentEdge {
                            from: tail_layer,
                            to: label_node,
                            edge_index: edge.index,
                            weight: (edge.weight + 1).max(1),
                            cluster_crossings: edge.cluster_crossings,
                            flat: label_segments_flat,
                            temp_kind: edge.temp_kind,
                        });
                        segment_edges.push(SegmentEdge {
                            from: head_layer,
                            to: label_node,
                            edge_index: edge.index,
                            weight: (edge.weight + 1).max(1),
                            cluster_crossings: edge.cluster_crossings,
                            flat: label_segments_flat,
                            temp_kind: edge.temp_kind,
                        });
                    }
                    chain.push(label_node);
                } else {
                    segment_edges.push(SegmentEdge {
                        from: tail_layer,
                        to: head_layer,
                        edge_index: edge.index,
                        weight: edge.weight.max(1),
                        cluster_crossings: edge.cluster_crossings,
                        flat: true,
                        temp_kind: edge.temp_kind,
                    });
                }
                chain.push(head_layer);
                chains.insert(edge.index, chain);
            }
            continue;
        }

        let mut chain = Vec::with_capacity(diff + 1);
        chain.push(tail_layer);
        let mut current = tail_layer;
        for step in 1..diff {
            let rank = tail_rank as usize + step;
            while rank >= layers.len() {
                layers.push(Vec::new());
            }
            let id = layer_nodes.len();
            let size = (config.vertex_spacing * config.dummy_size).max(1.0);
            let path_tail = &node_cluster_paths[reverse_local[edge.tail]];
            let path_head = &node_cluster_paths[reverse_local[edge.head]];
            let cluster_path = dummy_cluster_path_between(path_tail, path_head, step, diff);
            layer_nodes.push(LayerNode {
                rank,
                order: 0,
                width: size,
                height: size,
                x: 0.0,
                y: 0.0,
                real_node: None,
                cluster_path,
                kind: LayerNodeKind::Dummy {
                    edge_index: edge.index,
                    step,
                },
                median_value: 0.0,
            });
            layers[rank].push(id);
            segment_edges.push(SegmentEdge {
                from: current,
                to: id,
                edge_index: edge.index,
                weight: edge.weight.max(1),
                cluster_crossings: edge.cluster_crossings,
                flat: false,
                temp_kind: edge.temp_kind,
            });
            chain.push(id);
            current = id;
        }
        segment_edges.push(SegmentEdge {
            from: current,
            to: head_layer,
            edge_index: edge.index,
            weight: edge.weight.max(1),
            cluster_crossings: edge.cluster_crossings,
            flat: false,
            temp_kind: edge.temp_kind,
        });
        chain.push(head_layer);
        chains.insert(edge.index, chain);
    }

    initialize_layer_orders(&mut layers, &mut layer_nodes);
    let flat_order_constraints =
        collect_flat_order_constraints(&oriented_edges, &real_layer_node, &layer_nodes);
    run_mincross(
        &mut layers,
        &mut layer_nodes,
        &segment_edges,
        config.c_minimization,
        config.transpose,
        &flat_order_constraints,
    );

    info!(target: "layout", "Starting phase 3 [dot_position]");
    let width = assign_coordinates(
        &layers,
        &mut layer_nodes,
        &segment_edges,
        config,
        render_config,
        &oriented_edges,
        &real_layer_node,
    );

    let mut node_positions = HashMap::new();
    for node in &layer_nodes {
        let Some(local) = node.real_node else {
            continue;
        };
        let global = reverse_local[local];
        node_positions.insert(global, (node.x, node.y));
    }

    info!(target: "layout", "Starting phase 4 [dot_splines]");
    let mut grouped_multi_edges: HashMap<(usize, usize), Vec<&DirectedEdge>> = HashMap::new();
    for edge in &oriented_edges {
        grouped_multi_edges
            .entry((edge.original_tail, edge.original_head))
            .or_default()
            .push(edge);
    }

    let mut edge_offsets = HashMap::<usize, f64>::new();
    let offset_step = (render_config.routing_padding * 1.5).max(1.0);
    for grouped in grouped_multi_edges.values_mut() {
        grouped.sort_by_key(|edge| edge.index);
        let center = (grouped.len() as f64 - 1.0) / 2.0;
        for (i, edge) in grouped.iter().enumerate() {
            let side = i as f64 - center;
            let boundary_bonus = if side.abs() > 0.001 {
                edge.cluster_crossings as f64
                    * (render_config.routing_padding * 0.25)
                    * side.signum()
            } else {
                0.0
            };
            let offset = side * offset_step + boundary_bonus;
            edge_offsets.insert(edge.index, offset);
        }
    }

    let mut routed_edges = Vec::new();
    for edge in &oriented_edges {
        let Some(chain) = chains.get(&edge.index) else {
            continue;
        };
        let mut points: Vec<(f64, f64)> = if edge.flat {
            flat_edge_points(chain, &layer_nodes, render_config.routing_padding)
        } else {
            chain
                .iter()
                .map(|&id| (layer_nodes[id].x, layer_nodes[id].y))
                .collect()
        };
        let reverse_output = edge.tail != edge.original_tail || edge.head != edge.original_head;
        if reverse_output {
            points.reverse();
        }

        if let Some(offset) = edge_offsets.get(&edge.index).copied() {
            offset_polyline(&mut points, offset);
        }

        let label_position = if edge.label.is_some() {
            if edge.flat {
                flat_virtual_label_position(chain, &layer_nodes, render_config.routing_padding)
                    .or_else(|| flat_edge_label_position(&points, render_config.routing_padding))
            } else {
                point_at_fraction(&points, 0.5)
            }
        } else {
            None
        };

        routed_edges.push(RoutedEdge {
            index: edge.index,
            points,
            label: edge.label.clone(),
            label_position,
        });
    }

    ComponentLayout {
        node_positions,
        edges: routed_edges,
        width,
    }
}

fn break_cycles_acyclic(node_count: usize, edges: &mut [DirectedEdge]) {
    if node_count <= 1 || edges.is_empty() {
        return;
    }

    let mut mark = vec![false; node_count];
    let mut onstack = vec![false; node_count];
    let mut changed = false;
    for start in 0..node_count {
        if mark[start] {
            continue;
        }
        dfs_break_cycles(start, edges, &mut mark, &mut onstack, &mut changed);
    }
}

fn dfs_break_cycles(
    node: usize,
    edges: &mut [DirectedEdge],
    mark: &mut [bool],
    onstack: &mut [bool],
    changed: &mut bool,
) {
    if mark[node] {
        return;
    }
    mark[node] = true;
    onstack[node] = true;

    // Mirror Graphviz acyclic DFS behavior: traverse current out edges and
    // reverse back-edges in place while iterating.
    let mut edge_idx = 0usize;
    while edge_idx < edges.len() {
        if edges[edge_idx].flat || edges[edge_idx].tail != node {
            edge_idx += 1;
            continue;
        }
        let head = edges[edge_idx].head;
        if onstack[head] {
            let edge = &mut edges[edge_idx];
            std::mem::swap(&mut edge.tail, &mut edge.head);
            edge.temp_kind = TempEdgeKind::Reversed;
            *changed = true;
            continue; // revisit this slot after reversal
        } else if !mark[head] {
            dfs_break_cycles(head, edges, mark, onstack, changed);
        }
        edge_idx += 1;
    }

    onstack[node] = false;
}

fn assign_ranks(
    node_count: usize,
    edges: &[DirectedEdge],
    ranking_type: RankingType,
    hybrid_weight: f64,
) -> Vec<i32> {
    if node_count == 0 {
        return Vec::new();
    }

    if matches!(ranking_type, RankingType::Hybrid) {
        let minimize = assign_ranks(
            node_count,
            edges,
            RankingType::MinimizeEdgeLength,
            hybrid_weight,
        );
        let original = assign_ranks(node_count, edges, RankingType::Original, hybrid_weight);
        return blend_hybrid_ranks(node_count, edges, &minimize, &original, hybrid_weight);
    }

    let topo = topo_order(node_count, edges);
    let mut rank = vec![0i32; node_count];
    let mut outgoing = vec![Vec::<usize>::new(); node_count];
    let mut incoming = vec![Vec::<usize>::new(); node_count];
    for (edge_idx, edge) in edges.iter().enumerate() {
        outgoing[edge.tail].push(edge_idx);
        incoming[edge.head].push(edge_idx);
    }

    if matches!(ranking_type, RankingType::Original) {
        for (idx, &node) in topo.iter().enumerate() {
            rank[node] = idx as i32;
        }
        relax_forward(&mut rank, &topo, &outgoing, edges);
        return rank;
    }

    let sources: HashSet<usize> = incoming
        .iter()
        .enumerate()
        .filter_map(|(node, inputs)| if inputs.is_empty() { Some(node) } else { None })
        .collect();
    let sinks: HashSet<usize> = outgoing
        .iter()
        .enumerate()
        .filter_map(|(node, outputs)| if outputs.is_empty() { Some(node) } else { None })
        .collect();

    for &source in &sources {
        rank[source] = 0;
    }
    relax_forward(&mut rank, &topo, &outgoing, edges);

    if matches!(
        ranking_type,
        RankingType::MinimizeEdgeLength | RankingType::Down | RankingType::Up
    ) {
        balance_sources(&mut rank, &sources, &topo, &outgoing, edges);
    }

    if matches!(ranking_type, RankingType::MinimizeEdgeLength) {
        network_simplex_refine(&mut rank, edges, &topo, &sources, &incoming, &outgoing);
    } else {
        // Dot2-style fallback: keep levels feasible and compact by alternating
        // forward/backward longest-path scans.
        let mut reverse_topo = topo.clone();
        reverse_topo.reverse();
        for _ in 0..(node_count * 2).max(8) {
            let mut changed = false;
            for &node in &reverse_topo {
                if sinks.contains(&node) {
                    continue;
                }
                let upper = outgoing[node]
                    .iter()
                    .map(|&eid| rank[edges[eid].head] - edges[eid].min_len)
                    .min()
                    .unwrap_or(rank[node]);
                if upper < rank[node] {
                    rank[node] = upper;
                    changed = true;
                }
            }
            for &source in &sources {
                rank[source] = 0;
            }
            relax_forward(&mut rank, &topo, &outgoing, edges);
            if !changed {
                break;
            }
        }
    }

    balance_rank_band(
        &mut rank, &topo, &sources, &sinks, &incoming, &outgoing, edges,
    );

    if !sinks.is_empty() && matches!(ranking_type, RankingType::Down | RankingType::Up) {
        let sink_rank = sinks.iter().map(|&node| rank[node]).max().unwrap_or(0);
        for sink in sinks {
            rank[sink] = sink_rank;
        }
    }

    relax_forward(&mut rank, &topo, &outgoing, edges);
    rank
}

fn blend_hybrid_ranks(
    node_count: usize,
    edges: &[DirectedEdge],
    minimize: &[i32],
    original: &[i32],
    hybrid_weight: f64,
) -> Vec<i32> {
    let weight = hybrid_weight.clamp(0.0, 1.0);
    if weight <= f64::EPSILON {
        return minimize.to_vec();
    }
    if (1.0 - weight) <= f64::EPSILON {
        return original.to_vec();
    }

    let mut rank = vec![0i32; node_count];
    for node in 0..node_count {
        let compact = minimize.get(node).copied().unwrap_or_default() as f64;
        let tall = original.get(node).copied().unwrap_or_default() as f64;
        rank[node] = ((1.0 - weight) * compact + weight * tall).round() as i32;
    }

    let topo = topo_order(node_count, edges);
    let mut outgoing = vec![Vec::<usize>::new(); node_count];
    for (edge_idx, edge) in edges.iter().enumerate() {
        outgoing[edge.tail].push(edge_idx);
    }
    relax_forward(&mut rank, &topo, &outgoing, edges);
    rank
}

fn topo_order(node_count: usize, edges: &[DirectedEdge]) -> Vec<usize> {
    let mut indegree = vec![0usize; node_count];
    let mut outgoing = vec![Vec::<usize>::new(); node_count];
    for (edge_id, edge) in edges.iter().enumerate() {
        if edge.flat {
            continue;
        }
        indegree[edge.head] += 1;
        outgoing[edge.tail].push(edge_id);
    }

    let mut queue = VecDeque::new();
    for (node, &deg) in indegree.iter().enumerate() {
        if deg == 0 {
            queue.push_back(node);
        }
    }

    let mut order = Vec::with_capacity(node_count);
    while let Some(node) = queue.pop_front() {
        order.push(node);
        for &edge_id in &outgoing[node] {
            let head = edges[edge_id].head;
            if indegree[head] > 0 {
                indegree[head] -= 1;
                if indegree[head] == 0 {
                    queue.push_back(head);
                }
            }
        }
    }

    if order.len() < node_count {
        let mut seen = vec![false; node_count];
        for &node in &order {
            seen[node] = true;
        }
        for (node, was_seen) in seen.into_iter().enumerate() {
            if !was_seen {
                order.push(node);
            }
        }
    }

    order
}

fn relax_forward(
    rank: &mut [i32],
    topo: &[usize],
    outgoing: &[Vec<usize>],
    edges: &[DirectedEdge],
) {
    for &node in topo {
        let base = rank[node];
        for &edge_idx in &outgoing[node] {
            let edge = &edges[edge_idx];
            let candidate = base + edge.min_len;
            if candidate > rank[edge.head] {
                rank[edge.head] = candidate;
            }
        }
    }
}

fn balance_sources(
    rank: &mut [i32],
    sources: &HashSet<usize>,
    topo: &[usize],
    outgoing: &[Vec<usize>],
    edges: &[DirectedEdge],
) {
    if sources.is_empty() {
        return;
    }
    let min_source_rank = sources
        .iter()
        .map(|&source| rank[source])
        .min()
        .unwrap_or(0);
    if min_source_rank == 0 {
        return;
    }
    for value in rank.iter_mut() {
        *value -= min_source_rank;
    }
    for &source in sources {
        rank[source] = 0;
    }
    relax_forward(rank, topo, outgoing, edges);
}

fn network_simplex_refine(
    rank: &mut [i32],
    edges: &[DirectedEdge],
    topo: &[usize],
    pinned_sources: &HashSet<usize>,
    incoming: &[Vec<usize>],
    outgoing: &[Vec<usize>],
) {
    let node_count = rank.len();
    if node_count <= 1 {
        return;
    }

    let mut reverse_topo = topo.to_vec();
    reverse_topo.reverse();
    let iterations = (node_count * 4).max(12);

    for pass in 0..iterations {
        let scan = if pass % 2 == 0 { topo } else { &reverse_topo };
        let mut changed = false;
        for &node in scan {
            if pinned_sources.contains(&node) {
                continue;
            }

            let lower_bound = incoming[node]
                .iter()
                .map(|&edge_idx| {
                    let edge = &edges[edge_idx];
                    rank[edge.tail] + edge.min_len
                })
                .max()
                .unwrap_or(i32::MIN / 4);
            let upper_bound = outgoing[node]
                .iter()
                .map(|&edge_idx| {
                    let edge = &edges[edge_idx];
                    rank[edge.head] - edge.min_len
                })
                .min()
                .unwrap_or(i32::MAX / 4);
            if lower_bound > upper_bound {
                continue;
            }

            let in_w = incoming[node]
                .iter()
                .map(|&edge_idx| {
                    let edge = &edges[edge_idx];
                    edge.weight + (edge.min_len - edge.base_min_len).max(0)
                })
                .sum::<i32>();
            let out_w = outgoing[node]
                .iter()
                .map(|&edge_idx| {
                    let edge = &edges[edge_idx];
                    edge.weight + (edge.min_len - edge.base_min_len).max(0)
                })
                .sum::<i32>();
            let weighted_neighbor_center = {
                let mut num = 0.0;
                let mut den = 0.0;
                for &edge_idx in &incoming[node] {
                    let edge = &edges[edge_idx];
                    let w = edge.weight.max(1) as f64;
                    num += (rank[edge.tail] + edge.min_len) as f64 * w;
                    den += w;
                }
                for &edge_idx in &outgoing[node] {
                    let edge = &edges[edge_idx];
                    let w = edge.weight.max(1) as f64;
                    num += (rank[edge.head] - edge.min_len) as f64 * w;
                    den += w;
                }
                if den > 0.0 {
                    Some((num / den).round() as i32)
                } else {
                    None
                }
            };
            let target = match in_w.cmp(&out_w) {
                Ordering::Greater => lower_bound,
                Ordering::Less => upper_bound,
                Ordering::Equal => rank[node].clamp(lower_bound, upper_bound),
            };
            let target = weighted_neighbor_center
                .unwrap_or(target)
                .clamp(lower_bound, upper_bound);
            if target != rank[node] {
                rank[node] = target;
                changed = true;
            }
        }

        for &source in pinned_sources {
            rank[source] = 0;
        }
        relax_forward(rank, topo, outgoing, edges);
        if !changed && pass >= 3 {
            break;
        }
    }

    // Final local pivots reduce total weighted slack while keeping feasibility.
    for _ in 0..(node_count * 2).max(8) {
        let mut changed = false;
        for &node in topo {
            if pinned_sources.contains(&node) {
                continue;
            }
            let lower_bound = incoming[node]
                .iter()
                .map(|&edge_idx| {
                    let edge = &edges[edge_idx];
                    rank[edge.tail] + edge.min_len
                })
                .max()
                .unwrap_or(i32::MIN / 4);
            let upper_bound = outgoing[node]
                .iter()
                .map(|&edge_idx| {
                    let edge = &edges[edge_idx];
                    rank[edge.head] - edge.min_len
                })
                .min()
                .unwrap_or(i32::MAX / 4);
            if lower_bound > upper_bound {
                continue;
            }

            let current = rank[node].clamp(lower_bound, upper_bound);
            let lower_cost = local_slack_cost(node, lower_bound, rank, edges, incoming, outgoing);
            let current_cost = local_slack_cost(node, current, rank, edges, incoming, outgoing);
            let upper_cost = local_slack_cost(node, upper_bound, rank, edges, incoming, outgoing);

            let (best_rank, _) = [
                (lower_bound, lower_cost),
                (current, current_cost),
                (upper_bound, upper_cost),
            ]
            .into_iter()
            .min_by_key(|(_, c)| *c)
            .unwrap_or((current, current_cost));
            if best_rank != rank[node] {
                rank[node] = best_rank;
                changed = true;
            }
        }
        for &source in pinned_sources {
            rank[source] = 0;
        }
        relax_forward(rank, topo, outgoing, edges);
        if !changed {
            break;
        }
    }
}

fn balance_rank_band(
    rank: &mut [i32],
    topo: &[usize],
    sources: &HashSet<usize>,
    sinks: &HashSet<usize>,
    incoming: &[Vec<usize>],
    outgoing: &[Vec<usize>],
    edges: &[DirectedEdge],
) {
    if rank.is_empty() {
        return;
    }

    // Lower bounds from current feasible rank assignment.
    let mut lower = rank.to_vec();
    relax_forward(&mut lower, topo, outgoing, edges);
    let top = lower.iter().copied().max().unwrap_or(0);

    // Upper feasible bounds by reverse longest-path from sinks.
    let mut upper = vec![top; rank.len()];
    for &sink in sinks {
        upper[sink] = top;
    }
    let mut reverse_topo = topo.to_vec();
    reverse_topo.reverse();
    for &node in &reverse_topo {
        if outgoing[node].is_empty() {
            upper[node] = upper[node].min(top);
            continue;
        }
        let mut best = i32::MAX / 4;
        for &edge_idx in &outgoing[node] {
            let edge = &edges[edge_idx];
            let candidate = upper[edge.head] - edge.min_len;
            if candidate < best {
                best = candidate;
            }
        }
        if best < i32::MAX / 8 {
            upper[node] = upper[node].min(best);
        }
    }

    // Keep feasible interval valid.
    for node in 0..rank.len() {
        if upper[node] < lower[node] {
            upper[node] = lower[node];
        }
    }

    // Choose balanced rank inside [lower, upper] using weighted neighborhood center.
    for node in 0..rank.len() {
        let mut num = 0.0;
        let mut den = 0.0;
        for &edge_idx in &incoming[node] {
            let edge = &edges[edge_idx];
            let w = edge.weight.max(1) as f64;
            num += (rank[edge.tail] + edge.min_len) as f64 * w;
            den += w;
        }
        for &edge_idx in &outgoing[node] {
            let edge = &edges[edge_idx];
            let w = edge.weight.max(1) as f64;
            num += (rank[edge.head] - edge.min_len) as f64 * w;
            den += w;
        }
        let centered = if den > 0.0 {
            (num / den).round() as i32
        } else {
            (lower[node] + upper[node]) / 2
        };
        rank[node] = centered.clamp(lower[node], upper[node]);
    }

    // Re-anchor min sources to 0 and restore feasibility.
    if !sources.is_empty() {
        let min_source_rank = sources
            .iter()
            .map(|&source| rank[source])
            .min()
            .unwrap_or(0);
        for value in rank.iter_mut() {
            *value -= min_source_rank;
        }
        for &source in sources {
            rank[source] = 0;
        }
    }
    relax_forward(rank, topo, outgoing, edges);
}

fn local_slack_cost(
    node: usize,
    rank_candidate: i32,
    rank: &[i32],
    edges: &[DirectedEdge],
    incoming: &[Vec<usize>],
    outgoing: &[Vec<usize>],
) -> i64 {
    let mut cost = 0i64;
    for &edge_idx in &incoming[node] {
        let edge = &edges[edge_idx];
        let slack = (rank_candidate - rank[edge.tail] - edge.min_len).max(0);
        cost += slack as i64 * edge.weight.max(1) as i64;
    }
    for &edge_idx in &outgoing[node] {
        let edge = &edges[edge_idx];
        let slack = (rank[edge.head] - rank_candidate - edge.min_len).max(0);
        cost += slack as i64 * edge.weight.max(1) as i64;
    }
    cost
}

fn max_rank(rank: &[i32]) -> usize {
    rank.iter().copied().max().unwrap_or(0).max(0) as usize
}

fn normalize_ranks(rank: &mut [i32]) {
    let min_rank = rank.iter().copied().min().unwrap_or(0);
    if min_rank < 0 {
        for value in rank {
            *value -= min_rank;
        }
    }
}

fn initialize_layer_orders(layers: &mut [Vec<usize>], layer_nodes: &mut [LayerNode]) {
    for layer in layers {
        layer.sort_unstable_by_key(|&node| {
            let key = layer_nodes[node].real_node.unwrap_or(usize::MAX);
            (key, node)
        });
        for (order, &node) in layer.iter().enumerate() {
            layer_nodes[node].order = order;
        }
    }
}

fn run_mincross(
    layers: &mut [Vec<usize>],
    layer_nodes: &mut [LayerNode],
    segments: &[SegmentEdge],
    strategy: CrossingMinimization,
    transpose: bool,
    flat_constraints: &HashSet<(usize, usize)>,
) {
    if layers.len() <= 1 {
        return;
    }

    let mut incoming_segments = vec![Vec::<usize>::new(); layer_nodes.len()];
    let mut outgoing_segments = vec![Vec::<usize>::new(); layer_nodes.len()];
    for (segment_idx, segment) in segments.iter().enumerate() {
        if segment.flat {
            continue;
        }
        incoming_segments[segment.to].push(segment_idx);
        outgoing_segments[segment.from].push(segment_idx);
    }

    let mut best_layers = layers.to_vec();
    let mut best_crossings = total_crossings(layers, layer_nodes, segments);
    let mut current = best_crossings;
    let min_quit = 8usize;
    let convergence = 0.995f64;
    let coarse_iters = 4usize;
    let fine_iters = 24usize;
    let mut pass_counter = 0usize;

    for phase in 0..=2 {
        let max_this_phase = if phase <= 1 { coarse_iters } else { fine_iters };
        if phase == 2 && current > best_crossings {
            layers.clone_from_slice(&best_layers);
            refresh_orders(layers, layer_nodes);
            current = best_crossings;
        }

        let mut trying = 0usize;
        for _ in 0..max_this_phase {
            if current == 0 || trying >= min_quit {
                break;
            }
            mincross_step(
                layers,
                layer_nodes,
                segments,
                strategy,
                transpose,
                flat_constraints,
                &incoming_segments,
                &outgoing_segments,
                pass_counter,
            );
            pass_counter += 1;

            current = total_crossings(layers, layer_nodes, segments);
            if current <= best_crossings {
                if (current as f64) < convergence * (best_crossings as f64) {
                    trying = 0;
                } else {
                    trying += 1;
                }
                best_crossings = current;
                best_layers.clone_from_slice(layers);
            } else {
                trying += 1;
            }
        }
    }

    if current > best_crossings {
        layers.clone_from_slice(&best_layers);
        refresh_orders(layers, layer_nodes);
    }
    if transpose && best_crossings > 0 {
        transpose_layers(
            layers,
            layer_nodes,
            segments,
            false,
            &incoming_segments,
            &outgoing_segments,
            flat_constraints,
        );
        let polished = total_crossings(layers, layer_nodes, segments);
        if polished < best_crossings {
            best_layers.clone_from_slice(layers);
        } else if polished > best_crossings {
            layers.clone_from_slice(&best_layers);
            refresh_orders(layers, layer_nodes);
        }
    }

    layers.clone_from_slice(&best_layers);
    refresh_orders(layers, layer_nodes);
}

fn mincross_step(
    layers: &mut [Vec<usize>],
    layer_nodes: &mut [LayerNode],
    segments: &[SegmentEdge],
    strategy: CrossingMinimization,
    transpose: bool,
    flat_constraints: &HashSet<(usize, usize)>,
    incoming_segments: &[Vec<usize>],
    outgoing_segments: &[Vec<usize>],
    pass: usize,
) {
    let reverse = pass % 4 < 2;
    if pass % 2 == 0 {
        for rank in 1..layers.len() {
            let has_fixed = medians_for_layer(
                layers,
                layer_nodes,
                segments,
                rank,
                true,
                strategy,
                incoming_segments,
                flat_constraints,
            );
            reorder_layer(
                layers,
                layer_nodes,
                rank,
                reverse,
                has_fixed,
                flat_constraints,
            );
        }
    } else if layers.len() > 1 {
        for rank in (0..layers.len() - 1).rev() {
            let has_fixed = medians_for_layer(
                layers,
                layer_nodes,
                segments,
                rank,
                false,
                strategy,
                outgoing_segments,
                flat_constraints,
            );
            reorder_layer(
                layers,
                layer_nodes,
                rank,
                reverse,
                has_fixed,
                flat_constraints,
            );
        }
    }

    if transpose {
        transpose_layers(
            layers,
            layer_nodes,
            segments,
            !reverse,
            incoming_segments,
            outgoing_segments,
            flat_constraints,
        );
    }
}

fn medians_for_layer(
    layers: &[Vec<usize>],
    nodes: &mut [LayerNode],
    segments: &[SegmentEdge],
    rank: usize,
    use_incoming: bool,
    strategy: CrossingMinimization,
    adjacency: &[Vec<usize>],
    flat_constraints: &HashSet<(usize, usize)>,
) -> bool {
    let mut has_fixed = false;
    for &node in &layers[rank] {
        let mut values = Vec::new();
        for &segment_idx in &adjacency[node] {
            let segment = &segments[segment_idx];
            let other = if use_incoming {
                segment.from
            } else {
                segment.to
            };
            values.push((nodes[other].order as i32, segment_mincross_weight(segment)));
        }

        nodes[node].median_value = match values.len() {
            0 => -1.0,
            1 => values[0].0 as f64,
            2 => {
                let w0 = values[0].1 as f64;
                let w1 = values[1].1 as f64;
                ((values[0].0 as f64 * w0) + (values[1].0 as f64 * w1)) / (w0 + w1).max(1.0)
            }
            _ => match strategy {
                CrossingMinimization::Barycenter => weighted_barycenter_value(&values),
                CrossingMinimization::Median => weighted_median_value(&values),
            },
        };
    }

    for &node in &layers[rank] {
        if nodes[node].median_value >= 0.0 {
            continue;
        }
        if let Some(flat_median) = flat_constraint_median(node, nodes, flat_constraints) {
            nodes[node].median_value = flat_median;
        } else {
            has_fixed = true;
        }
    }

    has_fixed
}

fn weighted_barycenter_value(values: &[(i32, i32)]) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    for &(value, weight) in values {
        let w = weight.max(1) as f64;
        num += value as f64 * w;
        den += w;
    }
    if den > 0.0 {
        num / den
    } else {
        0.0
    }
}

fn weighted_median_value(values: &[(i32, i32)]) -> f64 {
    let mut sorted_values = values.to_vec();
    sorted_values.sort_unstable_by_key(|(order, _)| *order);
    if sorted_values.len() % 2 == 1 {
        return sorted_values[sorted_values.len() / 2].0 as f64;
    }

    let total_weight: i64 = sorted_values
        .iter()
        .map(|(_, weight)| (*weight).max(1) as i64)
        .sum();
    let halfway = total_weight as f64 / 2.0;
    let mut acc = 0.0;
    for (i, &(order, weight)) in sorted_values.iter().enumerate() {
        acc += weight.max(1) as f64;
        if acc >= halfway {
            if (acc - halfway).abs() < f64::EPSILON && i + 1 < sorted_values.len() {
                return (order + sorted_values[i + 1].0) as f64 / 2.0;
            }
            return order as f64;
        }
    }

    // Fallback to original span-based weighted median for numerical stability.
    let rm = sorted_values.len() / 2;
    let lm = rm - 1;
    let rspan = sorted_values[sorted_values.len() - 1].0 - sorted_values[rm].0;
    let lspan = sorted_values[lm].0 - sorted_values[0].0;
    if lspan == rspan {
        (sorted_values[lm].0 + sorted_values[rm].0) as f64 / 2.0
    } else {
        let left = sorted_values[lm].0 as f64 * rspan as f64;
        let right = sorted_values[rm].0 as f64 * lspan as f64;
        (left + right) / (lspan + rspan) as f64
    }
}

fn flat_constraint_median(
    node: usize,
    nodes: &[LayerNode],
    flat_constraints: &HashSet<(usize, usize)>,
) -> Option<f64> {
    let mut left_max: Option<usize> = None;
    let mut right_min: Option<usize> = None;
    for &(left, right) in flat_constraints {
        if right == node && nodes[left].rank == nodes[node].rank {
            left_max = Some(left_max.map_or(left, |cur| {
                if nodes[left].order > nodes[cur].order {
                    left
                } else {
                    cur
                }
            }));
        } else if left == node && nodes[right].rank == nodes[node].rank {
            right_min = Some(right_min.map_or(right, |cur| {
                if nodes[right].order < nodes[cur].order {
                    right
                } else {
                    cur
                }
            }));
        }
    }

    if let Some(left) = left_max {
        return Some((nodes[left].order + 1) as f64);
    }
    if let Some(right) = right_min {
        return Some((nodes[right].order.saturating_sub(1)) as f64);
    }
    None
}

fn reorder_layer(
    layers: &mut [Vec<usize>],
    nodes: &mut [LayerNode],
    rank: usize,
    reverse: bool,
    has_fixed: bool,
    flat_constraints: &HashSet<(usize, usize)>,
) {
    if layers[rank].len() <= 1 {
        return;
    }

    let n = layers[rank].len();
    let mut ep = n;
    for _ in (0..n).rev() {
        let mut lp = 0usize;
        while lp < ep {
            while lp < ep && nodes[layers[rank][lp]].median_value < 0.0 {
                lp += 1;
            }
            if lp >= ep {
                break;
            }

            let mut rp = lp + 1;
            let mut must_stay = false;
            while rp < ep {
                let left = layers[rank][lp];
                let right = layers[rank][rp];
                if left_to_right(nodes, left, right, flat_constraints) {
                    must_stay = true;
                    break;
                }
                if nodes[right].median_value >= 0.0 {
                    break;
                }
                rp += 1;
            }
            if rp >= ep {
                break;
            }

            if !must_stay {
                let left = layers[rank][lp];
                let right = layers[rank][rp];
                let p1 = nodes[left].median_value;
                let p2 = nodes[right].median_value;
                if p1 > p2 || (reverse && p1 >= p2) {
                    layers[rank].swap(lp, rp);
                    refresh_layer_order(&layers[rank], nodes);
                }
            }

            lp = rp;
        }
        if !has_fixed && !reverse {
            ep = ep.saturating_sub(1);
        }
    }
}

fn left_to_right(
    nodes: &[LayerNode],
    left: usize,
    right: usize,
    flat_constraints: &HashSet<(usize, usize)>,
) -> bool {
    if flat_constraints.contains(&(left, right)) {
        return true;
    }
    if cluster_boundary_crossings(&nodes[left].cluster_path, &nodes[right].cluster_path) > 0 {
        return nodes[left].order < nodes[right].order;
    }
    false
}

fn cluster_boundary_crossings(a: &[usize], b: &[usize]) -> usize {
    let shared = shared_prefix_len(a, b);
    a.len().saturating_sub(shared) + b.len().saturating_sub(shared)
}

fn cluster_spacing_extra(nodes: &[LayerNode], left: usize, right: usize, base_gap: f64) -> f64 {
    if base_gap <= 0.0 {
        return 0.0;
    }
    let crossings =
        cluster_boundary_crossings(&nodes[left].cluster_path, &nodes[right].cluster_path);
    if crossings == 0 {
        0.0
    } else {
        base_gap * crossings as f64
    }
}

fn segment_mincross_weight(segment: &SegmentEdge) -> i32 {
    let kind_bonus = match segment.temp_kind {
        TempEdgeKind::ClusterBridge => 2,
        TempEdgeKind::Reversed => 1,
        _ => 0,
    };
    (segment.weight + (segment.cluster_crossings as i32 * 2) + kind_bonus).max(1)
}

fn cluster_parent_map(nodes: &[LayerNode]) -> HashMap<usize, Option<usize>> {
    let mut parents = HashMap::new();
    for node in nodes {
        for (i, &cluster) in node.cluster_path.iter().enumerate() {
            let parent = if i == 0 {
                None
            } else {
                Some(node.cluster_path[i - 1])
            };
            parents.entry(cluster).or_insert(parent);
        }
    }
    parents
}

fn enforce_sibling_cluster_separation(
    rank: usize,
    layers: &[Vec<usize>],
    nodes: &mut [LayerNode],
    boundary_gap: f64,
    cluster_parents: &HashMap<usize, Option<usize>>,
) {
    if boundary_gap <= 0.0 {
        return;
    }
    let layer = &layers[rank];
    if layer.len() <= 1 {
        return;
    }

    let mut ranges = HashMap::<usize, (f64, f64)>::new();
    for &node in layer {
        let left = nodes[node].x - nodes[node].width / 2.0;
        let right = nodes[node].x + nodes[node].width / 2.0;
        for &cluster in &nodes[node].cluster_path {
            ranges
                .entry(cluster)
                .and_modify(|range| {
                    range.0 = range.0.min(left);
                    range.1 = range.1.max(right);
                })
                .or_insert((left, right));
        }
    }

    let mut siblings_by_parent = HashMap::<Option<usize>, Vec<usize>>::new();
    for (&cluster, parent) in cluster_parents {
        if ranges.contains_key(&cluster) {
            siblings_by_parent.entry(*parent).or_default().push(cluster);
        }
    }

    for siblings in siblings_by_parent.values_mut() {
        if siblings.len() <= 1 {
            continue;
        }
        siblings.sort_unstable_by(|a, b| {
            let ca = ranges
                .get(a)
                .map(|(l, r)| (l + r) / 2.0)
                .unwrap_or(f64::INFINITY);
            let cb = ranges
                .get(b)
                .map(|(l, r)| (l + r) / 2.0)
                .unwrap_or(f64::INFINITY);
            ca.partial_cmp(&cb).unwrap_or(Ordering::Equal)
        });

        for pair in siblings.windows(2) {
            let left_cluster = pair[0];
            let right_cluster = pair[1];
            let Some(&(_left_min, left_max)) = ranges.get(&left_cluster) else {
                continue;
            };
            let Some(&(right_min, _right_max)) = ranges.get(&right_cluster) else {
                continue;
            };
            let gap = right_min - left_max;
            let needed = boundary_gap - gap;
            if needed <= 0.0 {
                continue;
            }

            for &node in layer {
                if nodes[node].cluster_path.contains(&right_cluster) {
                    nodes[node].x += needed;
                }
            }

            for (&cluster, range) in ranges.iter_mut() {
                if is_cluster_descendant_or_self(cluster, right_cluster, cluster_parents) {
                    range.0 += needed;
                    range.1 += needed;
                }
            }
        }
    }
}

fn is_cluster_descendant_or_self(
    cluster: usize,
    ancestor: usize,
    parents: &HashMap<usize, Option<usize>>,
) -> bool {
    if cluster == ancestor {
        return true;
    }
    let mut cursor = Some(cluster);
    let mut guard = 0usize;
    while let Some(current) = cursor {
        if guard > parents.len() {
            break;
        }
        let parent = parents.get(&current).copied().flatten();
        if parent == Some(ancestor) {
            return true;
        }
        cursor = parent;
        guard += 1;
    }
    false
}

fn transpose_layers(
    layers: &mut [Vec<usize>],
    nodes: &mut [LayerNode],
    segments: &[SegmentEdge],
    reverse: bool,
    incoming_segments: &[Vec<usize>],
    outgoing_segments: &[Vec<usize>],
    flat_constraints: &HashSet<(usize, usize)>,
) {
    let mut candidate = vec![true; layers.len()];
    loop {
        let mut delta = 0i64;
        for rank in 0..layers.len() {
            if !candidate[rank] {
                continue;
            }
            candidate[rank] = false;
            if layers[rank].len() <= 1 {
                continue;
            }
            for i in 0..layers[rank].len() - 1 {
                let v = layers[rank][i];
                let w = layers[rank][i + 1];
                if left_to_right(nodes, v, w, flat_constraints) {
                    continue;
                }

                let mut c0 = 0i64;
                let mut c1 = 0i64;
                c0 += weighted_in_cross(v, w, incoming_segments, segments, nodes);
                c1 += weighted_in_cross(w, v, incoming_segments, segments, nodes);
                c0 += weighted_out_cross(v, w, outgoing_segments, segments, nodes);
                c1 += weighted_out_cross(w, v, outgoing_segments, segments, nodes);

                if c1 < c0 || (c0 > 0 && reverse && c1 == c0) {
                    layers[rank].swap(i, i + 1);
                    refresh_layer_order(&layers[rank], nodes);
                    delta += c0 - c1;
                    candidate[rank] = true;
                    if rank > 0 {
                        candidate[rank - 1] = true;
                    }
                    if rank + 1 < layers.len() {
                        candidate[rank + 1] = true;
                    }
                }
            }
        }
        if delta < 1 {
            break;
        }
    }
}

fn weighted_in_cross(
    left: usize,
    right: usize,
    incoming_segments: &[Vec<usize>],
    segments: &[SegmentEdge],
    nodes: &[LayerNode],
) -> i64 {
    let mut cross = 0i64;
    for &edge_b in &incoming_segments[right] {
        let eb = &segments[edge_b];
        let inv = nodes[eb.from].order as i32;
        let cnt = segment_mincross_weight(eb) as i64;
        for &edge_a in &incoming_segments[left] {
            let ea = &segments[edge_a];
            let t = nodes[ea.from].order as i32 - inv;
            if t > 0 || (t == 0 && ea.edge_index > eb.edge_index) {
                cross += segment_mincross_weight(ea) as i64 * cnt;
            }
        }
    }
    cross
}

fn weighted_out_cross(
    left: usize,
    right: usize,
    outgoing_segments: &[Vec<usize>],
    segments: &[SegmentEdge],
    nodes: &[LayerNode],
) -> i64 {
    let mut cross = 0i64;
    for &edge_b in &outgoing_segments[right] {
        let eb = &segments[edge_b];
        let inv = nodes[eb.to].order as i32;
        let cnt = segment_mincross_weight(eb) as i64;
        for &edge_a in &outgoing_segments[left] {
            let ea = &segments[edge_a];
            let t = nodes[ea.to].order as i32 - inv;
            if t > 0 || (t == 0 && ea.edge_index > eb.edge_index) {
                cross += segment_mincross_weight(ea) as i64 * cnt;
            }
        }
    }
    cross
}

fn refresh_orders(layers: &[Vec<usize>], nodes: &mut [LayerNode]) {
    for layer in layers {
        refresh_layer_order(layer, nodes);
    }
}

fn refresh_layer_order(layer: &[usize], nodes: &mut [LayerNode]) {
    for (order, &node) in layer.iter().enumerate() {
        nodes[node].order = order;
    }
}

fn total_crossings(layers: &[Vec<usize>], nodes: &[LayerNode], segments: &[SegmentEdge]) -> i64 {
    if layers.len() <= 1 {
        return 0;
    }
    (0..layers.len() - 1)
        .map(|rank| pair_crossings(rank, nodes, segments))
        .sum()
}

fn pair_crossings(rank: usize, nodes: &[LayerNode], segments: &[SegmentEdge]) -> i64 {
    let mut pairs = Vec::<(usize, usize, i64)>::new();
    let mut max_target_order = 0usize;
    for edge in segments {
        if edge.flat {
            continue;
        }
        if nodes[edge.from].rank == rank && nodes[edge.to].rank == rank + 1 {
            let from_order = nodes[edge.from].order;
            let to_order = nodes[edge.to].order;
            max_target_order = max_target_order.max(to_order);
            pairs.push((from_order, to_order, segment_mincross_weight(edge) as i64));
        }
    }

    if pairs.len() <= 1 {
        return 0;
    }
    pairs.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    weighted_inversion_count(&pairs, max_target_order + 2)
}

fn weighted_inversion_count(values: &[(usize, usize, i64)], size: usize) -> i64 {
    if values.len() <= 1 {
        return 0;
    }
    let mut fenwick = vec![0i64; size + 2];
    let mut seen = 0i64;
    let mut inversions = 0i64;
    for &(_, to, weight) in values {
        let idx = to + 1;
        let prefix = fenwick_sum_i64(&fenwick, idx);
        inversions += weight * (seen - prefix);
        fenwick_add_i64(&mut fenwick, idx, weight);
        seen += weight;
    }
    inversions
}

fn fenwick_add_i64(tree: &mut [i64], mut index: usize, delta: i64) {
    while index < tree.len() {
        tree[index] += delta;
        index += index & index.wrapping_neg();
    }
}

fn fenwick_sum_i64(tree: &[i64], mut index: usize) -> i64 {
    let mut sum = 0i64;
    while index > 0 {
        sum += tree[index];
        index &= index - 1;
    }
    sum
}

fn assign_coordinates(
    layers: &[Vec<usize>],
    nodes: &mut [LayerNode],
    segments: &[SegmentEdge],
    config: &Config,
    render_config: &RenderConfig,
    edges: &[DirectedEdge],
    real_layer_node: &[usize],
) -> f64 {
    if layers.is_empty() {
        return 0.0;
    }

    let mut rank_heights = vec![DEFAULT_NODE_SIZE.1; layers.len()];
    for (rank, layer) in layers.iter().enumerate() {
        for &node in layer {
            rank_heights[rank] = rank_heights[rank].max(nodes[node].height);
        }
    }

    let mut current_y = config.vertex_spacing + rank_heights[0] / 2.0;
    for (rank, layer) in layers.iter().enumerate() {
        if rank > 0 {
            current_y += rank_heights[rank - 1] / 2.0 + rank_heights[rank] / 2.0;
            current_y += config.vertex_spacing * 2.0;
        }
        for &node in layer {
            nodes[node].y = current_y;
        }
    }

    for layer in layers {
        let mut x = config.vertex_spacing;
        for &node in layer {
            x += nodes[node].width / 2.0;
            nodes[node].x = x;
            x += nodes[node].width / 2.0 + config.vertex_spacing;
        }
        if let (Some(&first), Some(&last)) = (layer.first(), layer.last()) {
            let left = nodes[first].x - nodes[first].width / 2.0;
            let right = nodes[last].x + nodes[last].width / 2.0;
            let center = (left + right) / 2.0;
            for &node in layer {
                nodes[node].x -= center;
            }
        }
    }

    let mut prev_neighbors = vec![Vec::<(usize, i32)>::new(); nodes.len()];
    let mut next_neighbors = vec![Vec::<(usize, i32)>::new(); nodes.len()];
    let mut flat_in = vec![Vec::<usize>::new(); nodes.len()];
    let mut flat_out = vec![Vec::<usize>::new(); nodes.len()];
    for edge in segments {
        if edge.flat {
            flat_out[edge.from].push(edge.to);
            flat_in[edge.to].push(edge.from);
            continue;
        }
        let weight = segment_mincross_weight(edge);
        next_neighbors[edge.from].push((edge.to, weight));
        prev_neighbors[edge.to].push((edge.from, weight));
    }

    let flat_constraints =
        collect_flat_constraints(edges, real_layer_node, nodes, config.vertex_spacing);
    let cluster_parents = cluster_parent_map(nodes);
    let iterations = 64usize;
    for pass in 0..iterations {
        let downward = pass % 2 == 0;
        let progress = pass as f64 / (iterations.saturating_sub(1).max(1) as f64);
        let damping = 1.0 - progress;

        if downward {
            for rank in 0..layers.len() {
                for &node in &layers[rank] {
                    let primary = weighted_neighbor_target(&prev_neighbors[node], nodes);
                    let secondary = weighted_neighbor_target(&next_neighbors[node], nodes);
                    let target = primary.or(secondary);
                    let Some(target) = target else {
                        continue;
                    };
                    let alpha = node_relaxation_alpha(nodes[node].kind, damping);
                    nodes[node].x = (1.0 - alpha) * nodes[node].x + alpha * target;
                }
                apply_flat_constraints(&flat_constraints, rank, layers, nodes);
                enforce_flat_virtual_nodes(rank, layers, nodes, &flat_in, &flat_out);
                for _ in 0..render_config.cluster_constraint_iterations.max(1) {
                    enforce_cluster_rank_blocks(
                        rank,
                        layers,
                        nodes,
                        render_config.cluster_boundary_gap,
                    );
                    enforce_sibling_cluster_separation(
                        rank,
                        layers,
                        nodes,
                        render_config.cluster_boundary_gap,
                        &cluster_parents,
                    );
                    enforce_layer_spacing(
                        rank,
                        layers,
                        nodes,
                        config.vertex_spacing,
                        render_config.cluster_boundary_gap,
                    );
                }
                recenter_rank(rank, layers, nodes);
            }
        } else {
            for rank in (0..layers.len()).rev() {
                for &node in &layers[rank] {
                    let primary = weighted_neighbor_target(&next_neighbors[node], nodes);
                    let secondary = weighted_neighbor_target(&prev_neighbors[node], nodes);
                    let target = primary.or(secondary);
                    let Some(target) = target else {
                        continue;
                    };
                    let alpha = node_relaxation_alpha(nodes[node].kind, damping);
                    nodes[node].x = (1.0 - alpha) * nodes[node].x + alpha * target;
                }
                apply_flat_constraints(&flat_constraints, rank, layers, nodes);
                enforce_flat_virtual_nodes(rank, layers, nodes, &flat_in, &flat_out);
                for _ in 0..render_config.cluster_constraint_iterations.max(1) {
                    enforce_cluster_rank_blocks(
                        rank,
                        layers,
                        nodes,
                        render_config.cluster_boundary_gap,
                    );
                    enforce_sibling_cluster_separation(
                        rank,
                        layers,
                        nodes,
                        render_config.cluster_boundary_gap,
                        &cluster_parents,
                    );
                    enforce_layer_spacing(
                        rank,
                        layers,
                        nodes,
                        config.vertex_spacing,
                        render_config.cluster_boundary_gap,
                    );
                }
                recenter_rank(rank, layers, nodes);
            }
        }

        if pass % 8 == 7 {
            for rank in 0..layers.len() {
                enforce_layer_spacing(
                    rank,
                    layers,
                    nodes,
                    config.vertex_spacing,
                    render_config.cluster_boundary_gap,
                );
            }
        }
    }

    let min_x = nodes
        .iter()
        .map(|node| node.x - node.width / 2.0)
        .fold(f64::INFINITY, f64::min);
    let shift = if min_x.is_finite() && min_x < config.vertex_spacing {
        config.vertex_spacing - min_x
    } else {
        0.0
    };
    for node in nodes.iter_mut() {
        node.x += shift;
    }

    nodes
        .iter()
        .map(|node| node.x + node.width / 2.0)
        .fold(0.0, f64::max)
}

fn node_relaxation_alpha(kind: LayerNodeKind, damping: f64) -> f64 {
    match kind {
        LayerNodeKind::Real => (0.16 + 0.20 * damping).clamp(0.12, 0.40),
        LayerNodeKind::Dummy { .. } => (0.45 + 0.18 * damping).clamp(0.28, 0.72),
        LayerNodeKind::FlatVirtual { .. } => (0.58 + 0.20 * damping).clamp(0.35, 0.85),
    }
}

fn weighted_neighbor_target(neighbors: &[(usize, i32)], nodes: &[LayerNode]) -> Option<f64> {
    if neighbors.is_empty() {
        return None;
    }
    let mut num = 0.0;
    let mut den = 0.0;
    for &(neighbor, weight) in neighbors {
        let w = weight.max(1) as f64;
        num += nodes[neighbor].x * w;
        den += w;
    }
    if den > 0.0 {
        Some(num / den)
    } else {
        None
    }
}

fn recenter_rank(rank: usize, layers: &[Vec<usize>], nodes: &mut [LayerNode]) {
    let layer = &layers[rank];
    if layer.is_empty() {
        return;
    }
    let mut left = f64::INFINITY;
    let mut right = f64::NEG_INFINITY;
    for &node in layer {
        left = left.min(nodes[node].x - nodes[node].width / 2.0);
        right = right.max(nodes[node].x + nodes[node].width / 2.0);
    }
    if !left.is_finite() || !right.is_finite() {
        return;
    }
    let center = (left + right) / 2.0;
    for &node in layer {
        nodes[node].x -= center;
    }
}

fn enforce_flat_virtual_nodes(
    rank: usize,
    layers: &[Vec<usize>],
    nodes: &mut [LayerNode],
    flat_in: &[Vec<usize>],
    flat_out: &[Vec<usize>],
) {
    for &node in &layers[rank] {
        let LayerNodeKind::FlatVirtual { .. } = nodes[node].kind else {
            continue;
        };
        if flat_in[node].is_empty() && flat_out[node].is_empty() {
            continue;
        }

        let mut targets = Vec::new();
        for &src in &flat_in[node] {
            targets.push(nodes[src].x);
        }
        for &dst in &flat_out[node] {
            targets.push(nodes[dst].x);
        }
        if targets.is_empty() {
            continue;
        }
        let target = targets.iter().sum::<f64>() / targets.len() as f64;
        nodes[node].x = (nodes[node].x * 0.35) + (target * 0.65);
    }
}

fn enforce_layer_spacing(
    rank: usize,
    layers: &[Vec<usize>],
    nodes: &mut [LayerNode],
    spacing: f64,
    cluster_boundary_gap: f64,
) {
    let layer = &layers[rank];
    if layer.len() <= 1 {
        return;
    }

    for window in layer.windows(2) {
        let left = window[0];
        let right = window[1];
        let extra = cluster_spacing_extra(nodes, left, right, cluster_boundary_gap);
        let min_right =
            nodes[left].x + nodes[left].width / 2.0 + spacing + extra + nodes[right].width / 2.0;
        if nodes[right].x < min_right {
            nodes[right].x = min_right;
        }
    }

    for window in layer.windows(2).rev() {
        let left = window[0];
        let right = window[1];
        let extra = cluster_spacing_extra(nodes, left, right, cluster_boundary_gap);
        let max_left =
            nodes[right].x - nodes[right].width / 2.0 - spacing - extra - nodes[left].width / 2.0;
        if nodes[left].x > max_left {
            nodes[left].x = max_left;
        }
    }
}

fn enforce_cluster_rank_blocks(
    rank: usize,
    layers: &[Vec<usize>],
    nodes: &mut [LayerNode],
    cluster_boundary_gap: f64,
) {
    let layer = &layers[rank];
    if layer.len() <= 1 || cluster_boundary_gap <= 0.0 {
        return;
    }

    let mut i = 0usize;
    while i < layer.len() {
        let cluster = nodes[layer[i]].cluster_path.as_slice();
        let mut j = i + 1;
        while j < layer.len() && nodes[layer[j]].cluster_path.as_slice() == cluster {
            j += 1;
        }
        if j < layer.len() {
            let left = layer[j - 1];
            let right = layer[j];
            let left_edge = nodes[left].x + nodes[left].width / 2.0;
            let right_edge = nodes[right].x - nodes[right].width / 2.0;
            let min_gap = cluster_spacing_extra(nodes, left, right, cluster_boundary_gap);
            let needed = min_gap - (right_edge - left_edge);
            if needed > 0.0 {
                for &node in &layer[j..] {
                    nodes[node].x += needed;
                }
            }
        }
        i = j;
    }
}

fn collect_flat_order_constraints(
    edges: &[DirectedEdge],
    real_layer_node: &[usize],
    layer_nodes: &[LayerNode],
) -> HashSet<(usize, usize)> {
    let mut constraints = HashSet::new();
    for edge in edges {
        if !edge.flat {
            continue;
        }
        let a = real_layer_node[edge.tail];
        let b = real_layer_node[edge.head];
        if a == b || layer_nodes[a].rank != layer_nodes[b].rank {
            continue;
        }
        if layer_nodes[a].order <= layer_nodes[b].order {
            constraints.insert((a, b));
        } else {
            constraints.insert((b, a));
        }
    }
    constraints
}

fn collect_flat_constraints(
    edges: &[DirectedEdge],
    real_layer_node: &[usize],
    layer_nodes: &[LayerNode],
    spacing: f64,
) -> Vec<FlatConstraint> {
    let mut constraints = Vec::new();
    for edge in edges {
        if !edge.flat {
            continue;
        }
        let a = real_layer_node[edge.tail];
        let b = real_layer_node[edge.head];
        if a == b || layer_nodes[a].rank != layer_nodes[b].rank {
            continue;
        }
        let (left, right) = if layer_nodes[a].order <= layer_nodes[b].order {
            (a, b)
        } else {
            (b, a)
        };
        let base_gap =
            layer_nodes[left].width / 2.0 + layer_nodes[right].width / 2.0 + spacing.max(0.0);
        let label_gap = edge
            .label
            .as_ref()
            .map(|label| estimate_label_width(label) * 0.7)
            .unwrap_or(0.0);
        constraints.push(FlatConstraint {
            left,
            right,
            min_gap: base_gap.max(label_gap),
        });
    }
    constraints
}

fn apply_flat_constraints(
    flat_constraints: &[FlatConstraint],
    rank: usize,
    layers: &[Vec<usize>],
    nodes: &mut [LayerNode],
) {
    if flat_constraints.is_empty() || layers[rank].is_empty() {
        return;
    }
    for constraint in flat_constraints {
        let left = constraint.left;
        let right = constraint.right;
        if nodes[left].rank != rank || nodes[right].rank != rank {
            continue;
        }
        let current_gap = nodes[right].x - nodes[left].x;
        if current_gap >= constraint.min_gap {
            continue;
        }
        let push = (constraint.min_gap - current_gap) / 2.0;
        nodes[left].x -= push;
        nodes[right].x += push;
    }
}

fn estimate_label_width(label: &str) -> f64 {
    let count = label.chars().count() as f64;
    (count * 7.0 + 8.0).max(0.0)
}

fn flat_edge_points(chain: &[usize], nodes: &[LayerNode], routing_padding: f64) -> Vec<(f64, f64)> {
    if chain.is_empty() {
        return Vec::new();
    }
    let cluster_cross = flat_chain_cluster_crossings(chain, nodes) as f64;
    if chain.len() == 1 {
        let id = chain[0];
        let n = &nodes[id];
        let r = (n.width.max(n.height) / 2.0 + routing_padding + cluster_cross * routing_padding)
            .max(8.0);
        return vec![
            (n.x + r, n.y),
            (n.x + r, n.y + r),
            (n.x - r, n.y + r),
            (n.x - r, n.y),
        ];
    }
    let a = &nodes[chain[0]];
    let b = &nodes[chain[chain.len() - 1]];
    if chain.len() >= 3 {
        let mut mx = 0.0;
        let mut my = 0.0;
        let mut count = 0.0;
        for &id in &chain[1..chain.len() - 1] {
            mx += nodes[id].x;
            my += nodes[id].y;
            count += 1.0;
        }
        if count > 0.0 {
            let mid = (mx / count, my / count);
            if (mid.1 - a.y).abs() > 1e-6 || (mid.1 - b.y).abs() > 1e-6 {
                return vec![
                    (a.x, a.y),
                    (a.x, mid.1),
                    (mid.0, mid.1),
                    (b.x, mid.1),
                    (b.x, b.y),
                ];
            }
        }
    }
    let lift =
        (routing_padding * (2.0 + cluster_cross * 0.75) + (a.height.max(b.height) * 0.5)).max(12.0);
    let horizontal_pad = cluster_cross * routing_padding.max(1.0);
    let mid_x = (a.x + b.x) * 0.5;
    vec![
        (a.x, a.y),
        (a.x, a.y + lift),
        (mid_x + horizontal_pad.copysign(b.x - a.x), a.y + lift),
        (b.x, b.y + lift),
        (b.x, b.y),
    ]
}

fn flat_chain_cluster_crossings(chain: &[usize], nodes: &[LayerNode]) -> usize {
    if chain.len() < 2 {
        return 0;
    }
    let first = chain[0];
    let last = chain[chain.len() - 1];
    cluster_boundary_crossings(&nodes[first].cluster_path, &nodes[last].cluster_path)
}

fn flat_edge_label_position(points: &[(f64, f64)], routing_padding: f64) -> Option<(f64, f64)> {
    if points.is_empty() {
        return None;
    }
    let (sum_x, max_y) = points.iter().fold((0.0, f64::NEG_INFINITY), |(sx, my), p| {
        (sx + p.0, my.max(p.1))
    });
    Some((
        sum_x / points.len() as f64,
        max_y + routing_padding.max(2.0) * 1.5,
    ))
}

fn flat_virtual_label_position(
    chain: &[usize],
    nodes: &[LayerNode],
    routing_padding: f64,
) -> Option<(f64, f64)> {
    for &node_id in chain {
        if let LayerNodeKind::FlatVirtual { .. } = nodes[node_id].kind {
            return Some((
                nodes[node_id].x,
                nodes[node_id].y + routing_padding.max(2.0),
            ));
        }
    }
    None
}

fn offset_polyline(points: &mut [(f64, f64)], amount: f64) {
    if points.len() < 2 || amount == 0.0 {
        return;
    }
    let start = points[0];
    let end = points[points.len() - 1];
    let dx = end.0 - start.0;
    let dy = end.1 - start.1;
    let length = (dx * dx + dy * dy).sqrt();
    let (nx, ny) = if length > 1e-6 {
        (-dy / length, dx / length)
    } else {
        (1.0, 0.0)
    };
    for point in points {
        point.0 += nx * amount;
        point.1 += ny * amount;
    }
}

fn polyline_to_curve(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    if points.len() <= 1 {
        return points.to_vec();
    }
    let mut curve = Vec::with_capacity(points.len() * 3);
    curve.push(points[0]);
    for window in points.windows(2) {
        let a = window[0];
        let b = window[1];
        let c1 = ((2.0 * a.0 + b.0) / 3.0, (2.0 * a.1 + b.1) / 3.0);
        let c2 = ((a.0 + 2.0 * b.0) / 3.0, (a.1 + 2.0 * b.1) / 3.0);
        curve.push(c1);
        curve.push(c2);
        curve.push(b);
    }
    curve
}

fn point_at_fraction(points: &[(f64, f64)], t: f64) -> Option<(f64, f64)> {
    if points.is_empty() {
        return None;
    }
    if points.len() == 1 {
        return Some(points[0]);
    }
    let target = t.clamp(0.0, 1.0);
    let mut lengths = Vec::with_capacity(points.len() - 1);
    let mut total = 0.0;
    for window in points.windows(2) {
        let dx = window[1].0 - window[0].0;
        let dy = window[1].1 - window[0].1;
        let len = (dx * dx + dy * dy).sqrt();
        lengths.push(len);
        total += len;
    }
    if total <= 1e-9 {
        return Some(points[0]);
    }
    let wanted = total * target;
    let mut walked = 0.0;
    for (i, len) in lengths.into_iter().enumerate() {
        if walked + len >= wanted {
            let local = if len <= 1e-9 {
                0.0
            } else {
                (wanted - walked) / len
            };
            let a = points[i];
            let b = points[i + 1];
            return Some((a.0 + (b.0 - a.0) * local, a.1 + (b.1 - a.1) * local));
        }
        walked += len;
    }
    points.last().copied()
}

fn compute_node_cluster_paths(
    clusters: &[Cluster],
    node_by_id: &HashMap<u32, usize>,
    node_count: usize,
) -> Vec<Vec<usize>> {
    if clusters.is_empty() {
        return vec![Vec::new(); node_count];
    }

    let depth = cluster_depths(clusters);
    let mut memberships = vec![Vec::<usize>::new(); node_count];
    for (cluster_idx, cluster) in clusters.iter().enumerate() {
        for &node_id in cluster.nodes() {
            let Some(&position) = node_by_id.get(&node_id) else {
                continue;
            };
            memberships[position].push(cluster_idx);
        }
    }

    let mut paths = vec![Vec::<usize>::new(); node_count];
    for (node, owned) in memberships.iter().enumerate() {
        let Some(&deepest) = owned
            .iter()
            .max_by_key(|&&cluster_idx| depth.get(cluster_idx).copied().unwrap_or(0))
        else {
            continue;
        };
        paths[node] = cluster_lineage(clusters, deepest);
    }
    paths
}

fn cluster_lineage(clusters: &[Cluster], start: usize) -> Vec<usize> {
    let mut lineage = Vec::new();
    let mut cursor = Some(start);
    let mut guard = 0usize;
    while let Some(current) = cursor {
        if current >= clusters.len() || guard > clusters.len() {
            break;
        }
        lineage.push(current);
        cursor = clusters[current].parent_index();
        guard += 1;
    }
    lineage.reverse();
    lineage
}

fn shared_prefix_len(a: &[usize], b: &[usize]) -> usize {
    let mut i = 0usize;
    while i < a.len() && i < b.len() && a[i] == b[i] {
        i += 1;
    }
    i
}

fn cluster_crossings_between(a: &[usize], b: &[usize]) -> usize {
    let k = shared_prefix_len(a, b);
    let tail_out = a.len().saturating_sub(k);
    let head_in = b.len().saturating_sub(k);
    tail_out + head_in
}

fn classify_cluster_edge(
    tail_path: &[usize],
    head_path: &[usize],
    flat: bool,
) -> (usize, i32, i32, TempEdgeKind) {
    if flat {
        return (0, 0, 0, TempEdgeKind::Flat);
    }
    let crossings = cluster_crossings_between(tail_path, head_path);
    if crossings == 0 {
        return (0, 0, 0, TempEdgeKind::Original);
    }

    // Dotgen-style cluster bridge weighting: crossings increase the edge's
    // pressure during ranking/mincross more than they increase pure minlen.
    let shared = shared_prefix_len(tail_path, head_path);
    let top_level_bridge_boost = if shared == 0 { 2 } else { 0 };
    let weight_penalty = (crossings as i32 * 2) + top_level_bridge_boost;
    let minlen_penalty = crossings as i32;
    (
        crossings,
        weight_penalty,
        minlen_penalty,
        TempEdgeKind::ClusterBridge,
    )
}

fn dummy_cluster_path_between(
    tail_path: &[usize],
    head_path: &[usize],
    step: usize,
    diff: usize,
) -> Vec<usize> {
    if diff <= 1 {
        return tail_path.to_vec();
    }
    let shared = shared_prefix_len(tail_path, head_path);
    let tail_out = tail_path.len().saturating_sub(shared);
    let head_in = head_path.len().saturating_sub(shared);
    let transitions = tail_out + head_in;
    if transitions == 0 {
        return tail_path[..shared].to_vec();
    }

    // Map dummy step position to cluster transition phase.
    let phase = ((step as f64 / diff as f64) * transitions as f64).floor() as usize;
    let leaves = phase.min(tail_out);
    let enters = phase.saturating_sub(tail_out).min(head_in);

    let mut path = tail_path[..tail_path.len().saturating_sub(leaves)].to_vec();
    if enters > 0 {
        let end = (shared + enters).min(head_path.len());
        path.extend_from_slice(&head_path[shared..end]);
    }
    path
}

fn compute_cluster_layouts(
    clusters: &[Cluster],
    node_by_id: &HashMap<u32, usize>,
    coords: &BTreeMap<usize, (f64, f64)>,
    nodes: &[SizedNode],
    render_config: &RenderConfig,
) -> Vec<ClusterLayout> {
    if clusters.is_empty() {
        return Vec::new();
    }

    let mut children = vec![Vec::<usize>::new(); clusters.len()];
    for (child, cluster) in clusters.iter().enumerate() {
        if let Some(parent) = cluster.parent_index() {
            if parent < clusters.len() {
                children[parent].push(child);
            }
        }
    }

    let mut base_bounds = vec![None::<Bounds>; clusters.len()];
    for (cluster_idx, cluster) in clusters.iter().enumerate() {
        let mut bound = None::<Bounds>;
        for &node_id in cluster.nodes() {
            let Some(&position) = node_by_id.get(&node_id) else {
                continue;
            };
            let Some(&(x, y)) = coords.get(&position) else {
                continue;
            };
            let size = &nodes[position];
            let node_bound = Bounds {
                min_x: x - size.width / 2.0,
                min_y: y - size.height / 2.0,
                max_x: x + size.width / 2.0,
                max_y: y + size.height / 2.0,
            };
            bound = Some(merge_bounds(bound, node_bound));
        }
        base_bounds[cluster_idx] = bound;
    }

    let mut memo = vec![None::<Option<Bounds>>; clusters.len()];
    let mut visiting = HashSet::new();
    let mut layouts = Vec::new();
    for index in 0..clusters.len() {
        if let Some(bound) = cluster_final_bounds(
            index,
            clusters,
            &children,
            &base_bounds,
            &mut memo,
            &mut visiting,
            render_config,
        ) {
            layouts.push(ClusterLayout {
                index,
                parent: clusters[index].parent_index(),
                min_x: bound.min_x,
                min_y: bound.min_y,
                max_x: bound.max_x,
                max_y: bound.max_y,
            });
        }
    }
    layouts
}

fn cluster_final_bounds(
    index: usize,
    clusters: &[Cluster],
    children: &[Vec<usize>],
    base_bounds: &[Option<Bounds>],
    memo: &mut [Option<Option<Bounds>>],
    visiting: &mut HashSet<usize>,
    render_config: &RenderConfig,
) -> Option<Bounds> {
    if let Some(bound) = memo[index] {
        return bound;
    }
    if !visiting.insert(index) {
        return None;
    }

    let mut bound = base_bounds[index];
    for &child in &children[index] {
        if let Some(child_bound) = cluster_final_bounds(
            child,
            clusters,
            children,
            base_bounds,
            memo,
            visiting,
            render_config,
        ) {
            // Preserve explicit gap between parent and child cluster boxes.
            let expanded_child = expand_bounds(child_bound, render_config.cluster_boundary_gap);
            bound = Some(merge_bounds(bound, expanded_child));
        }
    }
    visiting.remove(&index);

    let padding = clusters[index]
        .padding_value()
        .unwrap_or(render_config.cluster_padding)
        .max(0.0);
    let bound = bound.map(|mut b| {
        b.min_x -= padding;
        b.min_y -= padding;
        b.max_x += padding;
        b.max_y += padding;
        // Keep clusters non-degenerate for stable clipping and nested boundary
        // handling in later phases.
        let min_size = (render_config.cluster_boundary_gap * 2.0).max(1.0);
        if b.max_x - b.min_x < min_size {
            let c = (b.min_x + b.max_x) / 2.0;
            b.min_x = c - min_size / 2.0;
            b.max_x = c + min_size / 2.0;
        }
        if b.max_y - b.min_y < min_size {
            let c = (b.min_y + b.max_y) / 2.0;
            b.min_y = c - min_size / 2.0;
            b.max_y = c + min_size / 2.0;
        }
        b
    });
    memo[index] = Some(bound);
    bound
}

fn expand_bounds(bounds: Bounds, amount: f64) -> Bounds {
    if amount <= 0.0 {
        return bounds;
    }
    Bounds {
        min_x: bounds.min_x - amount,
        min_y: bounds.min_y - amount,
        max_x: bounds.max_x + amount,
        max_y: bounds.max_y + amount,
    }
}

fn merge_bounds(current: Option<Bounds>, next: Bounds) -> Bounds {
    if let Some(current) = current {
        Bounds {
            min_x: current.min_x.min(next.min_x),
            min_y: current.min_y.min(next.min_y),
            max_x: current.max_x.max(next.max_x),
            max_y: current.max_y.max(next.max_y),
        }
    } else {
        next
    }
}

fn clip_edges_to_clusters(
    routed_edges: &mut [RoutedEdge],
    source_edges: &[(u32, u32)],
    node_by_id: &HashMap<u32, usize>,
    clusters: &[Cluster],
    cluster_layouts: &[ClusterLayout],
    render_config: &RenderConfig,
    node_cluster_paths: &[Vec<usize>],
) {
    if routed_edges.is_empty() || cluster_layouts.is_empty() || clusters.is_empty() {
        return;
    }

    info!(target: "layout", "Starting phase 5 [dot_compoundEdges]");

    let mut cluster_bounds = HashMap::<usize, Bounds>::new();
    for cluster in cluster_layouts {
        let expanded = Bounds {
            min_x: cluster.min_x() - render_config.cluster_boundary_gap,
            min_y: cluster.min_y() - render_config.cluster_boundary_gap,
            max_x: cluster.max_x() + render_config.cluster_boundary_gap,
            max_y: cluster.max_y() + render_config.cluster_boundary_gap,
        };
        cluster_bounds.insert(cluster.index(), expanded);
    }

    for edge in routed_edges {
        let Some(&(tail_id, head_id)) = source_edges.get(edge.index) else {
            continue;
        };
        let (Some(&tail_pos), Some(&head_pos)) =
            (node_by_id.get(&tail_id), node_by_id.get(&head_id))
        else {
            continue;
        };

        let tail_path = node_cluster_paths
            .get(tail_pos)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let head_path = node_cluster_paths
            .get(head_pos)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let shared = shared_prefix_len(tail_path, head_path);

        for &cluster_idx in tail_path[shared..].iter().rev() {
            if let Some(bound) = cluster_bounds.get(&cluster_idx).copied() {
                trim_polyline_start_outside(&mut edge.points, bound);
            }
        }
        for &cluster_idx in head_path[shared..].iter().rev() {
            if let Some(bound) = cluster_bounds.get(&cluster_idx).copied() {
                trim_polyline_end_outside(&mut edge.points, bound);
            }
        }

        if edge.label.is_some() {
            edge.label_position = point_at_fraction(&edge.points, 0.5);
        }
    }
}

fn trim_polyline_start_outside(points: &mut Vec<(f64, f64)>, bounds: Bounds) {
    for _ in 0..points.len().saturating_mul(2).max(1) {
        let before_len = points.len();
        let before_start = points.first().copied();
        clip_polyline_start(points, bounds);
        if points.len() == before_len && points.first().copied() == before_start {
            break;
        }
        if points.first().is_some_and(|&p| !point_in_bounds(p, bounds)) {
            break;
        }
    }
}

fn trim_polyline_end_outside(points: &mut Vec<(f64, f64)>, bounds: Bounds) {
    for _ in 0..points.len().saturating_mul(2).max(1) {
        let before_len = points.len();
        let before_end = points.last().copied();
        clip_polyline_end(points, bounds);
        if points.len() == before_len && points.last().copied() == before_end {
            break;
        }
        if points.last().is_some_and(|&p| !point_in_bounds(p, bounds)) {
            break;
        }
    }
}

fn cluster_depths(clusters: &[Cluster]) -> Vec<usize> {
    fn depth_of(
        idx: usize,
        clusters: &[Cluster],
        memo: &mut [Option<usize>],
        visiting: &mut HashSet<usize>,
    ) -> usize {
        if let Some(depth) = memo[idx] {
            return depth;
        }
        if !visiting.insert(idx) {
            return 0;
        }
        let depth = match clusters[idx].parent_index() {
            Some(parent) if parent < clusters.len() => {
                depth_of(parent, clusters, memo, visiting) + 1
            }
            _ => 0,
        };
        visiting.remove(&idx);
        memo[idx] = Some(depth);
        depth
    }

    let mut memo = vec![None; clusters.len()];
    let mut visiting = HashSet::new();
    (0..clusters.len())
        .map(|idx| depth_of(idx, clusters, &mut memo, &mut visiting))
        .collect()
}

fn clip_polyline_start(points: &mut Vec<(f64, f64)>, bounds: Bounds) {
    if points.len() < 2 {
        return;
    }
    if !point_in_bounds(points[0], bounds) {
        return;
    }
    for i in 0..points.len() - 1 {
        let a = points[i];
        let b = points[i + 1];
        if point_in_bounds(a, bounds) && !point_in_bounds(b, bounds) {
            if let Some(intersection) = box_intersectf(a, b, bounds) {
                let mut clipped = Vec::with_capacity(points.len() - i);
                clipped.push(intersection);
                clipped.extend_from_slice(&points[i + 1..]);
                *points = clipped;
            }
            return;
        }
    }
}

fn clip_polyline_end(points: &mut Vec<(f64, f64)>, bounds: Bounds) {
    if points.len() < 2 {
        return;
    }
    let last = *points.last().unwrap_or(&(0.0, 0.0));
    if !point_in_bounds(last, bounds) {
        return;
    }
    for i in (1..points.len()).rev() {
        let a = points[i - 1];
        let b = points[i];
        if !point_in_bounds(a, bounds) && point_in_bounds(b, bounds) {
            if let Some(intersection) = spline_intersectf(a, b, bounds) {
                let mut clipped = Vec::with_capacity(i + 1);
                clipped.extend_from_slice(&points[..i]);
                clipped.push(intersection);
                *points = clipped;
            }
            return;
        }
    }
}

fn point_in_bounds(point: (f64, f64), bounds: Bounds) -> bool {
    point.0 >= bounds.min_x
        && point.0 <= bounds.max_x
        && point.1 >= bounds.min_y
        && point.1 <= bounds.max_y
}

fn box_intersectf(a: (f64, f64), b: (f64, f64), bounds: Bounds) -> Option<(f64, f64)> {
    // Mirror compound.c::boxIntersectf semantics: first endpoint is on/inside
    // the box, second endpoint is outside. Prefer boundary chosen by outside
    // endpoint direction.
    let (ppx, ppy) = a;
    let (cpx, cpy) = b;

    if cpx < bounds.min_x && (ppx - cpx).abs() > 1e-9 {
        let x = bounds.min_x;
        let y = ppy + ((x - ppx) * (ppy - cpy) / (ppx - cpx)).round();
        if y >= bounds.min_y - 1e-6 && y <= bounds.max_y + 1e-6 {
            return Some((x, y));
        }
    }
    if cpx > bounds.max_x && (ppx - cpx).abs() > 1e-9 {
        let x = bounds.max_x;
        let y = ppy + ((x - ppx) * (ppy - cpy) / (ppx - cpx)).round();
        if y >= bounds.min_y - 1e-6 && y <= bounds.max_y + 1e-6 {
            return Some((x, y));
        }
    }
    if cpy < bounds.min_y && (ppy - cpy).abs() > 1e-9 {
        let y = bounds.min_y;
        let x = ppx + ((y - ppy) * (ppx - cpx) / (ppy - cpy)).round();
        if x >= bounds.min_x - 1e-6 && x <= bounds.max_x + 1e-6 {
            return Some((x, y));
        }
    }
    if cpy > bounds.max_y && (ppy - cpy).abs() > 1e-9 {
        let y = bounds.max_y;
        let x = ppx + ((y - ppy) * (ppx - cpx) / (ppy - cpy)).round();
        if x >= bounds.min_x - 1e-6 && x <= bounds.max_x + 1e-6 {
            return Some((x, y));
        }
    }

    segment_box_intersection(a, b, bounds)
}

fn spline_intersectf(a: (f64, f64), b: (f64, f64), bounds: Bounds) -> Option<(f64, f64)> {
    // For the polyline approximation, reuse directional box intersection by
    // reversing the segment so the first endpoint is inside.
    if point_in_bounds(b, bounds) {
        box_intersectf(b, a, bounds)
    } else {
        segment_box_intersection(a, b, bounds)
    }
}

fn segment_box_intersection(a: (f64, f64), b: (f64, f64), bounds: Bounds) -> Option<(f64, f64)> {
    let dx = b.0 - a.0;
    let dy = b.1 - a.1;
    let mut candidates = Vec::<(f64, (f64, f64))>::new();

    let mut add_candidate = |t: f64, point: (f64, f64)| {
        if (0.0..=1.0).contains(&t)
            && point.0 >= bounds.min_x - 1e-6
            && point.0 <= bounds.max_x + 1e-6
            && point.1 >= bounds.min_y - 1e-6
            && point.1 <= bounds.max_y + 1e-6
        {
            candidates.push((t, point));
        }
    };

    if dx.abs() > 1e-9 {
        let t0 = (bounds.min_x - a.0) / dx;
        add_candidate(t0, (bounds.min_x, a.1 + dy * t0));
        let t1 = (bounds.max_x - a.0) / dx;
        add_candidate(t1, (bounds.max_x, a.1 + dy * t1));
    }
    if dy.abs() > 1e-9 {
        let t2 = (bounds.min_y - a.1) / dy;
        add_candidate(t2, (a.0 + dx * t2, bounds.min_y));
        let t3 = (bounds.max_y - a.1) / dy;
        add_candidate(t3, (a.0 + dx * t3, bounds.max_y));
    }

    candidates
        .into_iter()
        .filter(|(t, _)| *t > 0.0 && *t < 1.0)
        .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal))
        .map(|(_, point)| point)
}
