use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::env;

use log::{error, info};

const DEFAULT_NODE_SIZE: (f64, f64) = (56.0, 32.0);
const COMPONENT_GAP: f64 = 80.0;
const MINIMUM_LENGTH_DEFAULT: u32 = 1;
const VERTEX_SPACING_DEFAULT: f64 = 10.0;
const DUMMY_VERTICES_DEFAULT: bool = true;
const RANKING_TYPE_DEFAULT: RankingType = RankingType::MinimizeEdgeLength;
const C_MINIMIZATION_DEFAULT: CrossingMinimization = CrossingMinimization::Median;
const TRANSPOSE_DEFAULT: bool = true;
const DUMMY_SIZE_DEFAULT: f64 = 1.0;

const ENV_MINIMUM_LENGTH: &str = "RUST_GRAPH_MIN_LEN";
const ENV_VERTEX_SPACING: &str = "RUST_GRAPH_V_SPACING";
const ENV_DUMMY_VERTICES: &str = "RUST_GRAPH_DUMMIES";
const ENV_RANKING_TYPE: &str = "RUST_GRAPH_R_TYPE";
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
    Up,
    Down,
}

impl TryFrom<String> for RankingType {
    type Error = String;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        match value.as_str() {
            "original" => Ok(Self::Original),
            "minimize" => Ok(Self::MinimizeEdgeLength),
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

    info!(target: "layout", "Starting phase 0 [decompose]");
    let mut node_by_id: HashMap<u32, usize> = HashMap::new();
    let mut sized_nodes = Vec::with_capacity(nodes.len());
    for (position, node_id) in nodes.iter().copied().enumerate() {
        node_by_id.entry(node_id).or_insert(position);
        let (width, height) = sanitize_node_size(node_size(node_id));
        sized_nodes.push(SizedNode { width, height });
    }

    let mut input_edges = Vec::with_capacity(edges.len());
    for (index, &(tail, head)) in edges.iter().enumerate() {
        let (Some(&tail_pos), Some(&head_pos)) = (node_by_id.get(&tail), node_by_id.get(&head))
        else {
            continue;
        };
        input_edges.push(InputEdge {
            index,
            tail: tail_pos,
            head: head_pos,
            min_len: config.minimum_length.max(1) as i32,
            label: edge_label(index, (tail, head)),
        });
    }

    let components = connected_components(nodes.len(), &input_edges);
    let mut global_positions = BTreeMap::new();
    let mut routed_edges = Vec::<RoutedEdge>::new();
    let mut x_offset = 0.0;
    let mut max_y = 0.0;
    let mut max_x = 0.0;

    for component_nodes in components {
        let component_set: HashSet<usize> = component_nodes.iter().copied().collect();
        let component_edges: Vec<_> = input_edges
            .iter()
            .filter(|edge| component_set.contains(&edge.tail) && component_set.contains(&edge.head))
            .cloned()
            .collect();

        if component_nodes.is_empty() {
            continue;
        }

        let component = layout_component(
            &sized_nodes,
            &component_nodes,
            &component_edges,
            config,
            render_config,
        );

        for (position, (x, y)) in component.node_positions {
            let gx = x + x_offset;
            global_positions.insert(position, (gx, y));
            if gx > max_x {
                max_x = gx;
            }
            if y > max_y {
                max_y = y;
            }
        }

        for mut edge in component.edges {
            for point in &mut edge.points {
                point.0 += x_offset;
            }
            routed_edges.push(edge);
        }

        if component.width > 0.0 {
            x_offset += component.width + COMPONENT_GAP;
        }
    }

    let cluster_layouts = compute_cluster_layouts(
        clusters,
        &node_by_id,
        &global_positions,
        &sized_nodes,
        render_config,
    );
    clip_edges_to_clusters(
        &mut routed_edges,
        edges,
        &node_by_id,
        clusters,
        &cluster_layouts,
        render_config,
    );

    let mut edge_layouts = Vec::with_capacity(routed_edges.len());
    for route in routed_edges {
        let curve = polyline_to_curve(&route.points);
        let label_position = route
            .label
            .as_ref()
            .and_then(|_| point_at_fraction(&route.points, 0.5));
        edge_layouts.push(EdgeLayout {
            index: route.index,
            points: route.points,
            curve_points: curve,
            label: route.label,
            label_position,
        });
    }

    for cluster in &cluster_layouts {
        max_x = max_x.max(cluster.max_x);
        max_y = max_y.max(cluster.max_y);
    }

    GraphLayout {
        max_x: (max_x + config.vertex_spacing).max(1.0),
        max_y: (max_y + config.vertex_spacing).max(1.0),
        coords: global_positions,
        edges: edge_layouts,
        clusters: cluster_layouts,
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
    label: Option<String>,
    original_tail: usize,
    original_head: usize,
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
}

#[derive(Debug, Clone, Copy)]
struct SegmentEdge {
    from: usize,
    to: usize,
}

#[derive(Debug, Clone)]
struct RoutedEdge {
    index: usize,
    points: Vec<(f64, f64)>,
    label: Option<String>,
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

fn connected_components(node_count: usize, edges: &[InputEdge]) -> Vec<Vec<usize>> {
    let mut adjacency = vec![Vec::<usize>::new(); node_count];
    for edge in edges {
        adjacency[edge.tail].push(edge.head);
        adjacency[edge.head].push(edge.tail);
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
        acyclic_edges.push(DirectedEdge {
            index: edge.index,
            tail,
            head,
            min_len: edge.min_len.max(1),
            label: edge.label.clone(),
            original_tail: tail,
            original_head: head,
        });
    }
    break_cycles_greedy(local_count, &mut acyclic_edges);

    let oriented_edges: Vec<DirectedEdge> = acyclic_edges
        .iter()
        .map(|edge| {
            if matches!(config.ranking_type, RankingType::Up) {
                DirectedEdge {
                    index: edge.index,
                    tail: edge.head,
                    head: edge.tail,
                    min_len: edge.min_len,
                    label: edge.label.clone(),
                    original_tail: edge.original_tail,
                    original_head: edge.original_head,
                }
            } else {
                edge.clone()
            }
        })
        .collect();

    info!(target: "layout", "Starting phase 1 [dot_rank]");
    let mut ranks = assign_ranks(local_count, &oriented_edges, config.ranking_type);
    normalize_ranks(&mut ranks);

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
        layer_nodes.push(LayerNode {
            rank,
            order: 0,
            width: nodes[global].width,
            height: nodes[global].height,
            x: 0.0,
            y: 0.0,
            real_node: Some(local),
        });
        layers[rank].push(id);
        real_layer_node[local] = id;
    }

    let mut segment_edges = Vec::<SegmentEdge>::new();
    let mut chains: HashMap<usize, Vec<usize>> = HashMap::new();
    for edge in &oriented_edges {
        let mut tail_rank = ranks[edge.tail];
        let mut head_rank = ranks[edge.head];
        if head_rank <= tail_rank {
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
            layer_nodes.push(LayerNode {
                rank,
                order: 0,
                width: size,
                height: size,
                x: 0.0,
                y: 0.0,
                real_node: None,
            });
            layers[rank].push(id);
            segment_edges.push(SegmentEdge {
                from: current,
                to: id,
            });
            chain.push(id);
            current = id;
        }
        segment_edges.push(SegmentEdge {
            from: current,
            to: head_layer,
        });
        chain.push(head_layer);
        chains.insert(edge.index, chain);
    }

    initialize_layer_orders(&mut layers, &mut layer_nodes);
    run_mincross(
        &mut layers,
        &mut layer_nodes,
        &segment_edges,
        config.c_minimization,
        config.transpose,
    );

    info!(target: "layout", "Starting phase 3 [dot_position]");
    let width = assign_coordinates(&layers, &mut layer_nodes, &segment_edges, config);

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
            let offset = (i as f64 - center) * offset_step;
            edge_offsets.insert(edge.index, offset);
        }
    }

    let mut routed_edges = Vec::new();
    for edge in &oriented_edges {
        let Some(chain) = chains.get(&edge.index) else {
            continue;
        };
        let mut points: Vec<(f64, f64)> = chain
            .iter()
            .map(|&id| (layer_nodes[id].x, layer_nodes[id].y))
            .collect();
        let reverse_output = edge.tail != edge.original_tail || edge.head != edge.original_head;
        if reverse_output {
            points.reverse();
        }

        if let Some(offset) = edge_offsets.get(&edge.index).copied() {
            offset_polyline(&mut points, offset);
        }

        routed_edges.push(RoutedEdge {
            index: edge.index,
            points,
            label: edge.label.clone(),
        });
    }

    ComponentLayout {
        node_positions,
        edges: routed_edges,
        width,
    }
}

fn break_cycles_greedy(node_count: usize, edges: &mut [DirectedEdge]) {
    if edges.is_empty() || node_count <= 1 {
        return;
    }

    let mut guard = 0usize;
    loop {
        guard += 1;
        if guard > edges.len().saturating_mul(4).max(8) {
            break;
        }

        let mut indegree = vec![0usize; node_count];
        let mut outgoing = vec![Vec::<usize>::new(); node_count];
        for (edge_id, edge) in edges.iter().enumerate() {
            indegree[edge.head] += 1;
            outgoing[edge.tail].push(edge_id);
        }

        let mut queue = VecDeque::new();
        for (node, &deg) in indegree.iter().enumerate() {
            if deg == 0 {
                queue.push_back(node);
            }
        }

        let mut removed = vec![false; node_count];
        let mut removed_count = 0usize;
        while let Some(node) = queue.pop_front() {
            if removed[node] {
                continue;
            }
            removed[node] = true;
            removed_count += 1;
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

        if removed_count == node_count {
            break;
        }

        let Some(edge) = edges
            .iter_mut()
            .find(|edge| !removed[edge.tail] && !removed[edge.head])
        else {
            break;
        };
        std::mem::swap(&mut edge.tail, &mut edge.head);
    }
}

fn assign_ranks(node_count: usize, edges: &[DirectedEdge], ranking_type: RankingType) -> Vec<i32> {
    if node_count == 0 {
        return Vec::new();
    }

    let topo = topo_order(node_count, edges);
    let mut rank = vec![0i32; node_count];
    let mut outgoing = vec![Vec::<&DirectedEdge>::new(); node_count];
    let mut incoming = vec![Vec::<&DirectedEdge>::new(); node_count];
    for edge in edges {
        outgoing[edge.tail].push(edge);
        incoming[edge.head].push(edge);
    }

    for &node in &topo {
        let base = rank[node];
        for edge in &outgoing[node] {
            let candidate = base + edge.min_len;
            if candidate > rank[edge.head] {
                rank[edge.head] = candidate;
            }
        }
    }

    let sources: HashSet<usize> = incoming
        .iter()
        .enumerate()
        .filter_map(|(node, inputs)| if inputs.is_empty() { Some(node) } else { None })
        .collect();
    for &source in &sources {
        rank[source] = 0;
    }
    relax_forward(&mut rank, &topo, &outgoing);

    if matches!(ranking_type, RankingType::MinimizeEdgeLength) {
        network_simplex_refine(&mut rank, edges, &topo, &sources);
    }

    let sinks: Vec<usize> = outgoing
        .iter()
        .enumerate()
        .filter_map(|(node, outputs)| if outputs.is_empty() { Some(node) } else { None })
        .collect();
    if !sinks.is_empty() {
        let sink_rank = sinks.iter().map(|&node| rank[node]).max().unwrap_or(0);
        for sink in sinks {
            rank[sink] = sink_rank;
        }
    }

    relax_forward(&mut rank, &topo, &outgoing);
    rank
}

fn topo_order(node_count: usize, edges: &[DirectedEdge]) -> Vec<usize> {
    let mut indegree = vec![0usize; node_count];
    let mut outgoing = vec![Vec::<usize>::new(); node_count];
    for (edge_id, edge) in edges.iter().enumerate() {
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
        let missing: HashSet<_> = order.iter().copied().collect();
        for node in 0..node_count {
            if !missing.contains(&node) {
                order.push(node);
            }
        }
    }

    order
}

fn relax_forward(rank: &mut [i32], topo: &[usize], outgoing: &[Vec<&DirectedEdge>]) {
    for &node in topo {
        let base = rank[node];
        for edge in &outgoing[node] {
            let candidate = base + edge.min_len;
            if candidate > rank[edge.head] {
                rank[edge.head] = candidate;
            }
        }
    }
}

fn network_simplex_refine(
    rank: &mut [i32],
    edges: &[DirectedEdge],
    topo: &[usize],
    pinned_sources: &HashSet<usize>,
) {
    let node_count = rank.len();
    if node_count <= 1 {
        return;
    }

    let mut outgoing = vec![Vec::<&DirectedEdge>::new(); node_count];
    let mut incoming = vec![Vec::<&DirectedEdge>::new(); node_count];
    for edge in edges {
        outgoing[edge.tail].push(edge);
        incoming[edge.head].push(edge);
    }

    let mut reverse_topo = topo.to_vec();
    reverse_topo.reverse();
    let iterations = (node_count * 4).max(12);

    for pass in 0..iterations {
        let scan = if pass % 2 == 0 { topo } else { &reverse_topo };
        for &node in scan {
            if pinned_sources.contains(&node) {
                continue;
            }

            let lower_bound = incoming[node]
                .iter()
                .map(|edge| rank[edge.tail] + edge.min_len)
                .max()
                .unwrap_or(i32::MIN / 4);
            let upper_bound = outgoing[node]
                .iter()
                .map(|edge| rank[edge.head] - edge.min_len)
                .min()
                .unwrap_or(i32::MAX / 4);
            if lower_bound > upper_bound {
                continue;
            }

            let in_w = incoming[node].len() as i32;
            let out_w = outgoing[node].len() as i32;
            let target = match in_w.cmp(&out_w) {
                Ordering::Greater => lower_bound,
                Ordering::Less => upper_bound,
                Ordering::Equal => rank[node].clamp(lower_bound, upper_bound),
            };
            rank[node] = target;
        }

        for &source in pinned_sources {
            rank[source] = 0;
        }
        relax_forward(rank, topo, &outgoing);
    }
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
) {
    if layers.len() <= 1 {
        return;
    }

    let mut incoming = vec![Vec::<usize>::new(); layer_nodes.len()];
    let mut outgoing = vec![Vec::<usize>::new(); layer_nodes.len()];
    for edge in segments {
        incoming[edge.to].push(edge.from);
        outgoing[edge.from].push(edge.to);
    }

    let mut best_layers = layers.to_vec();
    let mut best_crossings = total_crossings(layers, layer_nodes, segments);
    let mut stale_passes = 0usize;

    for _ in 0..24 {
        for rank in 1..layers.len() {
            order_layer(layers, layer_nodes, rank, &incoming, strategy);
        }
        for rank in (0..layers.len().saturating_sub(1)).rev() {
            order_layer(layers, layer_nodes, rank, &outgoing, strategy);
        }

        if transpose {
            for rank in 0..layers.len() {
                transpose_layer(layers, layer_nodes, segments, rank);
            }
        }

        let current = total_crossings(layers, layer_nodes, segments);
        if current < best_crossings {
            best_crossings = current;
            best_layers.clone_from_slice(layers);
            stale_passes = 0;
        } else {
            stale_passes += 1;
            if stale_passes >= 4 {
                break;
            }
        }
    }

    layers.clone_from_slice(&best_layers);
    for layer in layers.iter() {
        for (order, &node) in layer.iter().enumerate() {
            layer_nodes[node].order = order;
        }
    }
}

fn order_layer(
    layers: &mut [Vec<usize>],
    nodes: &mut [LayerNode],
    rank: usize,
    neighbors: &[Vec<usize>],
    strategy: CrossingMinimization,
) {
    let mut entries: Vec<(usize, f64, usize)> = layers[rank]
        .iter()
        .map(|&node| {
            let adj = &neighbors[node];
            let key = if adj.is_empty() {
                nodes[node].order as f64
            } else {
                match strategy {
                    CrossingMinimization::Barycenter => {
                        adj.iter()
                            .map(|&other| nodes[other].order as f64)
                            .sum::<f64>()
                            / adj.len() as f64
                    }
                    CrossingMinimization::Median => {
                        let mut values: Vec<usize> =
                            adj.iter().map(|&other| nodes[other].order).collect();
                        values.sort_unstable();
                        if values.len() % 2 == 1 {
                            values[values.len() / 2] as f64
                        } else {
                            (values[values.len() / 2 - 1] + values[values.len() / 2]) as f64 / 2.0
                        }
                    }
                }
            };
            (node, key, nodes[node].order)
        })
        .collect();

    entries.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(Ordering::Equal)
            .then(a.2.cmp(&b.2))
    });
    layers[rank] = entries.iter().map(|entry| entry.0).collect();
    for (order, &node) in layers[rank].iter().enumerate() {
        nodes[node].order = order;
    }
}

fn transpose_layer(
    layers: &mut [Vec<usize>],
    nodes: &mut [LayerNode],
    segments: &[SegmentEdge],
    rank: usize,
) {
    if layers[rank].len() <= 1 {
        return;
    }

    let mut improved = true;
    while improved {
        improved = false;
        let mut i = 0usize;
        while i + 1 < layers[rank].len() {
            let before = local_crossings(layers, nodes, segments, rank);
            layers[rank].swap(i, i + 1);
            nodes[layers[rank][i]].order = i;
            nodes[layers[rank][i + 1]].order = i + 1;
            let after = local_crossings(layers, nodes, segments, rank);
            if after < before {
                improved = true;
                i += 1;
            } else {
                layers[rank].swap(i, i + 1);
                nodes[layers[rank][i]].order = i;
                nodes[layers[rank][i + 1]].order = i + 1;
            }
            i += 1;
        }
    }
}

fn local_crossings(
    layers: &[Vec<usize>],
    nodes: &[LayerNode],
    segments: &[SegmentEdge],
    rank: usize,
) -> usize {
    let mut total = 0usize;
    if rank > 0 {
        total += pair_crossings(rank - 1, nodes, segments);
    }
    if rank + 1 < layers.len() {
        total += pair_crossings(rank, nodes, segments);
    }
    total
}

fn total_crossings(layers: &[Vec<usize>], nodes: &[LayerNode], segments: &[SegmentEdge]) -> usize {
    if layers.len() <= 1 {
        return 0;
    }
    (0..layers.len() - 1)
        .map(|rank| pair_crossings(rank, nodes, segments))
        .sum()
}

fn pair_crossings(rank: usize, nodes: &[LayerNode], segments: &[SegmentEdge]) -> usize {
    let mut pairs = Vec::<(usize, usize)>::new();
    let mut max_target_order = 0usize;
    for edge in segments {
        if nodes[edge.from].rank == rank && nodes[edge.to].rank == rank + 1 {
            let from_order = nodes[edge.from].order;
            let to_order = nodes[edge.to].order;
            max_target_order = max_target_order.max(to_order);
            pairs.push((from_order, to_order));
        }
    }

    if pairs.len() <= 1 {
        return 0;
    }
    pairs.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    inversion_count(
        &pairs.into_iter().map(|(_, to)| to).collect::<Vec<_>>(),
        max_target_order + 2,
    )
}

fn inversion_count(values: &[usize], size: usize) -> usize {
    if values.len() <= 1 {
        return 0;
    }
    let mut fenwick = vec![0usize; size + 2];
    let mut seen = 0usize;
    let mut inversions = 0usize;
    for &value in values {
        let idx = value + 1;
        let prefix = fenwick_sum(&fenwick, idx);
        inversions += seen - prefix;
        fenwick_add(&mut fenwick, idx, 1);
        seen += 1;
    }
    inversions
}

fn fenwick_add(tree: &mut [usize], mut index: usize, delta: usize) {
    while index < tree.len() {
        tree[index] += delta;
        index += index & (!index + 1);
    }
}

fn fenwick_sum(tree: &[usize], mut index: usize) -> usize {
    let mut sum = 0usize;
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

    let mut neighbors = vec![Vec::<usize>::new(); nodes.len()];
    for edge in segments {
        neighbors[edge.from].push(edge.to);
        neighbors[edge.to].push(edge.from);
    }

    for pass in 0..48 {
        let forward = pass % 2 == 0;
        let ranks: Vec<usize> = if forward {
            (0..layers.len()).collect()
        } else {
            (0..layers.len()).rev().collect()
        };

        for rank in ranks {
            for &node in &layers[rank] {
                if neighbors[node].is_empty() {
                    continue;
                }
                let target = neighbors[node]
                    .iter()
                    .map(|&neighbor| nodes[neighbor].x)
                    .sum::<f64>()
                    / neighbors[node].len() as f64;
                let alpha = if nodes[node].real_node.is_some() {
                    0.35
                } else {
                    0.65
                };
                nodes[node].x = (1.0 - alpha) * nodes[node].x + alpha * target;
            }
            enforce_layer_spacing(rank, layers, nodes, config.vertex_spacing);
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

fn enforce_layer_spacing(
    rank: usize,
    layers: &[Vec<usize>],
    nodes: &mut [LayerNode],
    spacing: f64,
) {
    let layer = &layers[rank];
    if layer.len() <= 1 {
        return;
    }

    for window in layer.windows(2) {
        let left = window[0];
        let right = window[1];
        let min_right =
            nodes[left].x + nodes[left].width / 2.0 + spacing + nodes[right].width / 2.0;
        if nodes[right].x < min_right {
            nodes[right].x = min_right;
        }
    }

    for window in layer.windows(2).rev() {
        let left = window[0];
        let right = window[1];
        let max_left =
            nodes[right].x - nodes[right].width / 2.0 - spacing - nodes[left].width / 2.0;
        if nodes[left].x > max_left {
            nodes[left].x = max_left;
        }
    }
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
            bound = Some(merge_bounds(bound, child_bound));
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
        b
    });
    memo[index] = Some(bound);
    bound
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

    let depth = cluster_depths(clusters);
    let mut memberships = vec![Vec::<usize>::new(); node_by_id.len()];
    for (cluster_index, cluster) in clusters.iter().enumerate() {
        for &node in cluster.nodes() {
            let Some(&position) = node_by_id.get(&node) else {
                continue;
            };
            if position < memberships.len() {
                memberships[position].push(cluster_index);
            }
        }
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
        let tail_cluster = deepest_cluster(&memberships, &depth, tail_pos);
        let head_cluster = deepest_cluster(&memberships, &depth, head_pos);
        if tail_cluster.is_some() && tail_cluster != head_cluster {
            if let Some(bound) = tail_cluster.and_then(|idx| cluster_bounds.get(&idx).copied()) {
                clip_polyline_start(&mut edge.points, bound);
            }
        }
        if head_cluster.is_some() && head_cluster != tail_cluster {
            if let Some(bound) = head_cluster.and_then(|idx| cluster_bounds.get(&idx).copied()) {
                clip_polyline_end(&mut edge.points, bound);
            }
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

fn deepest_cluster(memberships: &[Vec<usize>], depth: &[usize], node_pos: usize) -> Option<usize> {
    memberships.get(node_pos).and_then(|members| {
        members
            .iter()
            .max_by_key(|&&cluster| depth.get(cluster).copied().unwrap_or(0))
            .copied()
    })
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
    segment_box_intersection(a, b, bounds)
}

fn spline_intersectf(a: (f64, f64), b: (f64, f64), bounds: Bounds) -> Option<(f64, f64)> {
    segment_box_intersection(a, b, bounds)
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
