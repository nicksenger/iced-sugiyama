use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use petgraph::stable_graph::{EdgeIndex, NodeIndex, StableDiGraph};
use petgraph::visit::{EdgeRef, IntoEdgeReferences};

use crate::{configure::Config, from_graph};

const EPSILON: f64 = 1e-6;
const EDGE_LABEL_OBSTACLE_WIDTH: f64 = 56.0;
const EDGE_LABEL_OBSTACLE_HEIGHT: f64 = 20.0;
const EDGE_LABEL_OBSTACLE_PADDING: f64 = 2.0;
const CLUSTER_BORDER_LABEL_OBSTACLE_THICKNESS: f64 = 3.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    pub min_x: f64,
    pub min_y: f64,
    pub max_x: f64,
    pub max_y: f64,
}

impl Rect {
    fn from_center_size(center: (f64, f64), size: (f64, f64)) -> Self {
        let half_w = size.0 * 0.5;
        let half_h = size.1 * 0.5;
        Self {
            min_x: center.0 - half_w,
            min_y: center.1 - half_h,
            max_x: center.0 + half_w,
            max_y: center.1 + half_h,
        }
    }

    fn expand(self, padding: f64) -> Self {
        Self {
            min_x: self.min_x - padding,
            min_y: self.min_y - padding,
            max_x: self.max_x + padding,
            max_y: self.max_y + padding,
        }
    }

    fn translate(self, dx: f64, dy: f64) -> Self {
        Self {
            min_x: self.min_x + dx,
            min_y: self.min_y + dy,
            max_x: self.max_x + dx,
            max_y: self.max_y + dy,
        }
    }

    fn union(self, other: Self) -> Self {
        Self {
            min_x: self.min_x.min(other.min_x),
            min_y: self.min_y.min(other.min_y),
            max_x: self.max_x.max(other.max_x),
            max_y: self.max_y.max(other.max_y),
        }
    }
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

#[derive(Debug, Clone)]
struct RoutedEdgeDraft<L> {
    id: EdgeIndex,
    tail: NodeIndex,
    head: NodeIndex,
    points: Vec<(f64, f64)>,
    label_value: Option<L>,
    label_position: Option<(f64, f64)>,
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
    let base_layouts = from_graph(graph, vertex_size, config);

    let mut node_sizes = HashMap::new();
    for id in graph.node_indices() {
        if let Some(weight) = graph.node_weight(id) {
            node_sizes.insert(id, vertex_size(id, weight));
        }
    }

    let mut result = Vec::new();

    for (layout, _, _) in base_layouts {
        let mut positions = HashMap::new();
        for (id, center) in layout {
            positions.insert(id, center);
        }

        let component_nodes = positions.keys().copied().collect::<HashSet<_>>();

        let mut nodes = Vec::new();
        for id in &component_nodes {
            if let (Some(center), Some(size)) = (positions.get(id), node_sizes.get(id)) {
                nodes.push(RoutedNode {
                    id: *id,
                    center: *center,
                    size: *size,
                    bounds: Rect::from_center_size(*center, *size),
                });
            }
        }
        nodes.sort_by_key(|node| node.id.index());

        apply_graphviz_cluster_constraints(&mut nodes, clusters, config, render_config);

        let node_bounds = nodes
            .iter()
            .map(|node| (node.id, node.bounds))
            .collect::<HashMap<_, _>>();

        let mut component_clusters =
            build_cluster_layouts(clusters, &component_nodes, &node_bounds, render_config);

        let mut obstacles = Vec::new();
        for (id, bounds) in &node_bounds {
            obstacles.push((*id, bounds.expand(render_config.routing_padding)));
        }

        let mut edge_drafts = Vec::new();
        for edge in graph.edge_references() {
            let tail = edge.source();
            let head = edge.target();
            if !(component_nodes.contains(&tail) && component_nodes.contains(&head)) {
                continue;
            }
            edge_drafts.push(RoutedEdgeDraft {
                id: edge.id(),
                tail,
                head,
                points: Vec::new(),
                label_value: edge_label(edge.id(), edge.weight()),
                label_position: None,
            });
        }
        edge_drafts.sort_by_key(|edge| edge.id.index());

        let node_label_obstacles = node_bounds.values().copied().collect::<Vec<_>>();
        let cluster_label_obstacles = cluster_border_label_obstacles(&component_clusters);

        let mut placed_label_obstacles = Vec::new();
        for edge in &mut edge_drafts {
            let tail_rect = match node_bounds.get(&edge.tail) {
                Some(rect) => *rect,
                None => continue,
            };
            let head_rect = match node_bounds.get(&edge.head) {
                Some(rect) => *rect,
                None => continue,
            };

            let (start, end) = anchor_points(tail_rect, head_rect);
            edge.points = route_polyline_with_context(
                start,
                end,
                &obstacles,
                &placed_label_obstacles,
                Some((edge.tail, edge.head)),
                render_config.bend_penalty,
            );
            edge.label_position = edge
                .label_value
                .as_ref()
                .and_then(|_| {
                    choose_edge_label_position(
                        &edge.points,
                        &node_label_obstacles,
                        &cluster_label_obstacles,
                        &placed_label_obstacles,
                    )
                })
                .or_else(|| {
                    edge.label_value
                        .as_ref()
                        .map(|_| polyline_midpoint(&edge.points))
                });

            if let Some(position) = edge.label_position {
                if let Some(obstacle) = label_obstacle_rect(position) {
                    placed_label_obstacles.push(obstacle);
                }
            }
        }

        let mut edges = Vec::with_capacity(edge_drafts.len());
        for edge in edge_drafts {
            let label = edge.label_value.map(|value| {
                let position = match edge.label_position {
                    Some(position) => position,
                    None => polyline_midpoint(&edge.points),
                };
                EdgeLabel { value, position }
            });
            edges.push(RoutedEdge {
                id: edge.id,
                tail: edge.tail,
                head: edge.head,
                points: edge.points,
                label,
            });
        }

        let (min_x, min_y, max_x, max_y) =
            extents(&nodes, &edges, &component_clusters).unwrap_or((0.0, 0.0, 0.0, 0.0));

        let shift_x = -min_x;
        let shift_y = -min_y;

        for node in &mut nodes {
            node.center = (node.center.0 + shift_x, node.center.1 + shift_y);
            node.bounds = node.bounds.translate(shift_x, shift_y);
        }
        for edge in &mut edges {
            for point in &mut edge.points {
                point.0 += shift_x;
                point.1 += shift_y;
            }
            if let Some(label) = &mut edge.label {
                label.position.0 += shift_x;
                label.position.1 += shift_y;
            }
        }
        for cluster in &mut component_clusters {
            cluster.bounds = cluster.bounds.translate(shift_x, shift_y);
        }

        result.push(DetailedLayout {
            nodes,
            edges,
            clusters: component_clusters,
            width: (max_x - min_x).max(0.0),
            height: (max_y - min_y).max(0.0),
        });
    }

    result
}

#[derive(Debug, Clone)]
struct ClusterInternal {
    node_ids: HashSet<NodeIndex>,
    node_indices: Vec<usize>,
    padding: f64,
    parent: Option<usize>,
}

fn apply_graphviz_cluster_constraints<C>(
    nodes: &mut [RoutedNode<NodeIndex>],
    clusters: &[ClusterSpec<C>],
    config: &Config,
    render_config: &RenderConfig,
) {
    if nodes.is_empty() || clusters.is_empty() {
        return;
    }

    let node_index_by_id = nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.id, index))
        .collect::<HashMap<_, _>>();
    let mut cluster_internals = clusters
        .iter()
        .enumerate()
        .map(|(cluster_index, cluster)| {
            let mut node_ids = HashSet::new();
            let mut node_indices = Vec::new();
            for node in &cluster.nodes {
                let Some(node_index) = node_index_by_id.get(node).copied() else {
                    return None;
                };
                if node_ids.insert(*node) {
                    node_indices.push(node_index);
                }
            }

            let parent = cluster
                .parent
                .filter(|parent| *parent < clusters.len() && *parent != cluster_index);

            Some(ClusterInternal {
                node_ids,
                node_indices,
                padding: cluster.padding.unwrap_or(render_config.cluster_padding),
                parent,
            })
        })
        .collect::<Vec<_>>();

    for cluster in &mut cluster_internals {
        let Some(cluster) = cluster.as_mut() else {
            continue;
        };
        cluster.node_indices.sort_unstable();
        cluster.node_indices.dedup();
    }

    let cluster_count = cluster_internals.len();
    for _ in 0..cluster_count {
        let mut changed = false;
        for cluster_index in 0..cluster_count {
            let Some(parent_index) = cluster_internals
                .get(cluster_index)
                .and_then(|cluster| cluster.as_ref())
                .and_then(|cluster| cluster.parent)
            else {
                continue;
            };

            let Some((parent, child)) =
                parent_child_clusters_mut(&mut cluster_internals, parent_index, cluster_index)
            else {
                continue;
            };

            if merge_cluster_nodes(parent, child) {
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    let mut cluster_internals = cluster_internals
        .into_iter()
        .flatten()
        .filter(|cluster| !cluster.node_ids.is_empty())
        .collect::<Vec<_>>();
    if cluster_internals.is_empty() {
        return;
    }
    for cluster in &mut cluster_internals {
        cluster.node_indices.sort_unstable();
        cluster.node_indices.dedup();
    }

    let original_x = nodes.iter().map(|node| node.center.0).collect::<Vec<_>>();
    let node_cluster_memberships = build_node_cluster_memberships(nodes, &cluster_internals);

    let mut ranks = group_nodes_into_ranks(nodes);
    if ranks.is_empty() {
        return;
    }

    let mut cluster_order = (0..cluster_internals.len()).collect::<Vec<_>>();
    cluster_order.sort_by(|a, b| {
        cluster_internals[*b]
            .node_ids
            .len()
            .cmp(&cluster_internals[*a].node_ids.len())
    });

    let iterations = render_config.cluster_constraint_iterations.max(1);
    for _ in 0..iterations {
        for cluster_index in &cluster_order {
            for rank in &mut ranks {
                enforce_cluster_contiguity(rank, &node_cluster_memberships, *cluster_index);
            }
        }

        assign_rank_x_positions(
            &ranks,
            nodes,
            &original_x,
            &node_cluster_memberships,
            config.vertex_spacing,
            render_config.cluster_boundary_gap,
        );
    }

    enforce_cluster_non_overlap(
        nodes,
        &cluster_internals,
        render_config.cluster_boundary_gap,
        iterations.saturating_mul(cluster_internals.len().max(1)),
    );

    for node in nodes {
        node.bounds = Rect::from_center_size(node.center, node.size);
    }
}

fn parent_child_clusters_mut(
    cluster_internals: &mut [Option<ClusterInternal>],
    parent_index: usize,
    child_index: usize,
) -> Option<(&mut ClusterInternal, &ClusterInternal)> {
    if parent_index == child_index {
        return None;
    }

    if parent_index < child_index {
        let (left, right) = cluster_internals.split_at_mut(child_index);
        let parent = left
            .get_mut(parent_index)
            .and_then(|cluster| cluster.as_mut())?;
        let child = right.first().and_then(|cluster| cluster.as_ref())?;
        Some((parent, child))
    } else {
        let (left, right) = cluster_internals.split_at_mut(parent_index);
        let child = left.get(child_index).and_then(|cluster| cluster.as_ref())?;
        let parent = right.first_mut().and_then(|cluster| cluster.as_mut())?;
        Some((parent, child))
    }
}

fn merge_cluster_nodes(parent: &mut ClusterInternal, child: &ClusterInternal) -> bool {
    let mut changed = false;

    let previous_node_id_count = parent.node_ids.len();
    parent.node_ids.extend(child.node_ids.iter().copied());
    if parent.node_ids.len() != previous_node_id_count {
        changed = true;
    }

    if merge_sorted_unique_usize(&mut parent.node_indices, &child.node_indices) {
        changed = true;
    }

    changed
}

fn merge_sorted_unique_usize(target: &mut Vec<usize>, source: &[usize]) -> bool {
    if source.is_empty() {
        return false;
    }
    if target.is_empty() {
        target.extend_from_slice(source);
        return true;
    }

    let mut merged = Vec::with_capacity(target.len() + source.len());
    let mut i = 0;
    let mut j = 0;

    while i < target.len() && j < source.len() {
        match target[i].cmp(&source[j]) {
            Ordering::Less => {
                merged.push(target[i]);
                i += 1;
            }
            Ordering::Greater => {
                merged.push(source[j]);
                j += 1;
            }
            Ordering::Equal => {
                merged.push(target[i]);
                i += 1;
                j += 1;
            }
        }
    }

    if i < target.len() {
        merged.extend_from_slice(&target[i..]);
    }
    if j < source.len() {
        merged.extend_from_slice(&source[j..]);
    }

    if merged == *target {
        return false;
    }

    *target = merged;
    true
}

fn build_node_cluster_memberships(
    nodes: &[RoutedNode<NodeIndex>],
    clusters: &[ClusterInternal],
) -> Vec<Vec<usize>> {
    let mut memberships = vec![Vec::new(); nodes.len()];
    for (cluster_index, cluster) in clusters.iter().enumerate() {
        for node_index in &cluster.node_indices {
            if let Some(node_memberships) = memberships.get_mut(*node_index) {
                node_memberships.push(cluster_index);
            }
        }
    }
    memberships
}

fn group_nodes_into_ranks(nodes: &[RoutedNode<NodeIndex>]) -> Vec<Vec<usize>> {
    let mut ranks = Vec::<(f64, Vec<usize>)>::new();
    for (node_index, node) in nodes.iter().enumerate() {
        if let Some((_, rank_nodes)) = ranks
            .iter_mut()
            .find(|(rank_y, _)| almost_equal(*rank_y, node.center.1))
        {
            rank_nodes.push(node_index);
        } else {
            ranks.push((node.center.1, vec![node_index]));
        }
    }

    ranks.sort_by(|a, b| a.0.total_cmp(&b.0));
    for (_, rank_nodes) in &mut ranks {
        rank_nodes.sort_by(|a, b| nodes[*a].center.0.total_cmp(&nodes[*b].center.0));
    }

    ranks
        .into_iter()
        .map(|(_, rank_nodes)| rank_nodes)
        .collect()
}

fn enforce_cluster_contiguity(
    rank_nodes: &mut Vec<usize>,
    node_cluster_memberships: &[Vec<usize>],
    cluster_index: usize,
) {
    let member_positions = rank_nodes
        .iter()
        .enumerate()
        .filter_map(|(position, node_index)| {
            node_cluster_memberships
                .get(*node_index)
                .and_then(|memberships| memberships.binary_search(&cluster_index).ok())
                .map(|_| position)
        })
        .collect::<Vec<_>>();

    if member_positions.len() <= 1 {
        return;
    }

    let first_member_position = match member_positions.first().copied() {
        Some(position) => position,
        None => return,
    };

    let members = rank_nodes
        .iter()
        .copied()
        .filter(|node_index| {
            node_cluster_memberships
                .get(*node_index)
                .and_then(|memberships| memberships.binary_search(&cluster_index).ok())
                .is_some()
        })
        .collect::<Vec<_>>();
    let non_members = rank_nodes
        .iter()
        .copied()
        .filter(|node_index| {
            node_cluster_memberships
                .get(*node_index)
                .and_then(|memberships| memberships.binary_search(&cluster_index).ok())
                .is_none()
        })
        .collect::<Vec<_>>();

    let insert_at = first_member_position.min(non_members.len());
    let mut merged = Vec::with_capacity(rank_nodes.len());
    merged.extend(non_members[..insert_at].iter().copied());
    merged.extend(members);
    merged.extend(non_members[insert_at..].iter().copied());
    *rank_nodes = merged;
}

fn assign_rank_x_positions(
    ranks: &[Vec<usize>],
    nodes: &mut [RoutedNode<NodeIndex>],
    original_x: &[f64],
    node_cluster_memberships: &[Vec<usize>],
    vertex_spacing: f64,
    cluster_boundary_gap: f64,
) {
    for rank in ranks {
        if rank.is_empty() {
            continue;
        }

        for (position, node_index) in rank.iter().copied().enumerate() {
            let target_x = match position {
                0 => *original_x
                    .get(node_index)
                    .unwrap_or(&nodes[node_index].center.0),
                _ => {
                    let previous_index = rank[position - 1];
                    let min_gap = minimum_gap(
                        &nodes[previous_index],
                        &nodes[node_index],
                        &node_cluster_memberships[previous_index],
                        &node_cluster_memberships[node_index],
                        vertex_spacing,
                        cluster_boundary_gap,
                    );
                    let min_x = nodes[previous_index].center.0 + min_gap;
                    let original = *original_x
                        .get(node_index)
                        .unwrap_or(&nodes[node_index].center.0);
                    original.max(min_x)
                }
            };
            nodes[node_index].center.0 = target_x;
        }

        for reverse_position in (0..rank.len().saturating_sub(1)).rev() {
            let node_index = rank[reverse_position];
            let next_index = rank[reverse_position + 1];
            let min_gap = minimum_gap(
                &nodes[node_index],
                &nodes[next_index],
                &node_cluster_memberships[node_index],
                &node_cluster_memberships[next_index],
                vertex_spacing,
                cluster_boundary_gap,
            );
            let max_x = nodes[next_index].center.0 - min_gap;
            if nodes[node_index].center.0 > max_x {
                nodes[node_index].center.0 = max_x;
            }
        }

        for position in 1..rank.len() {
            let node_index = rank[position];
            let previous_index = rank[position - 1];
            let min_gap = minimum_gap(
                &nodes[previous_index],
                &nodes[node_index],
                &node_cluster_memberships[previous_index],
                &node_cluster_memberships[node_index],
                vertex_spacing,
                cluster_boundary_gap,
            );
            let min_x = nodes[previous_index].center.0 + min_gap;
            if nodes[node_index].center.0 < min_x {
                nodes[node_index].center.0 = min_x;
            }
        }
    }
}

fn minimum_gap(
    left_node: &RoutedNode<NodeIndex>,
    right_node: &RoutedNode<NodeIndex>,
    left_memberships: &[usize],
    right_memberships: &[usize],
    vertex_spacing: f64,
    cluster_boundary_gap: f64,
) -> f64 {
    let base_gap = (left_node.size.0 + right_node.size.0) * 0.5 + vertex_spacing;
    let transitions = cluster_membership_transitions(left_memberships, right_memberships);
    base_gap + transitions as f64 * cluster_boundary_gap * 0.5
}

fn enforce_cluster_non_overlap(
    nodes: &mut [RoutedNode<NodeIndex>],
    clusters: &[ClusterInternal],
    cluster_boundary_gap: f64,
    iterations: usize,
) {
    if nodes.is_empty() || clusters.len() < 2 || iterations == 0 {
        return;
    }

    let boundary_gap = cluster_boundary_gap.max(0.0);

    for _ in 0..iterations {
        let cluster_bounds = clusters
            .iter()
            .map(|cluster| cluster_bounds(cluster, nodes))
            .collect::<Vec<_>>();

        let mut order = cluster_bounds
            .iter()
            .enumerate()
            .filter_map(|(index, bounds)| bounds.map(|_| index))
            .collect::<Vec<_>>();
        order.sort_by(|a, b| {
            let center_a = cluster_bounds[*a]
                .map(|bounds| bounds.min_x + bounds.max_x)
                .unwrap_or(0.0);
            let center_b = cluster_bounds[*b]
                .map(|bounds| bounds.min_x + bounds.max_x)
                .unwrap_or(0.0);
            center_a.total_cmp(&center_b).then_with(|| a.cmp(b))
        });

        let mut moved = false;
        'search: for left_position in 0..order.len() {
            for right_position in (left_position + 1)..order.len() {
                let left_index = order[left_position];
                let right_index = order[right_position];

                let (Some(left_bounds), Some(right_bounds)) =
                    (cluster_bounds[left_index], cluster_bounds[right_index])
                else {
                    continue;
                };

                if !clusters[left_index]
                    .node_ids
                    .is_disjoint(&clusters[right_index].node_ids)
                {
                    continue;
                }

                let overlap_x = left_bounds.max_x.min(right_bounds.max_x)
                    - left_bounds.min_x.max(right_bounds.min_x);
                if overlap_x <= EPSILON {
                    continue;
                }

                let overlap_y = left_bounds.max_y.min(right_bounds.max_y)
                    - left_bounds.min_y.max(right_bounds.min_y);
                if overlap_y <= EPSILON {
                    continue;
                }

                let shift = overlap_x + boundary_gap;
                if shift <= EPSILON {
                    continue;
                }

                shift_cluster_nodes_x(nodes, &clusters[right_index].node_indices, shift);
                moved = true;
                break 'search;
            }
        }

        if !moved {
            break;
        }
    }
}

fn cluster_bounds(cluster: &ClusterInternal, nodes: &[RoutedNode<NodeIndex>]) -> Option<Rect> {
    let mut bounds: Option<Rect> = None;
    for node_index in &cluster.node_indices {
        let Some(node) = nodes.get(*node_index) else {
            continue;
        };
        let node_bounds = Rect::from_center_size(node.center, node.size);
        bounds = Some(match bounds {
            Some(current) => current.union(node_bounds),
            None => node_bounds,
        });
    }
    bounds.map(|rect| rect.expand(cluster.padding))
}

fn shift_cluster_nodes_x(nodes: &mut [RoutedNode<NodeIndex>], node_indices: &[usize], shift: f64) {
    for node_index in node_indices {
        let Some(node) = nodes.get_mut(*node_index) else {
            continue;
        };
        node.center.0 += shift;
        node.bounds = Rect::from_center_size(node.center, node.size);
    }
}

fn cluster_membership_transitions(left_ids: &[usize], right_ids: &[usize]) -> usize {
    let mut i = 0;
    let mut j = 0;
    let mut transitions = 0;
    while i < left_ids.len() && j < right_ids.len() {
        match left_ids[i].cmp(&right_ids[j]) {
            Ordering::Equal => {
                i += 1;
                j += 1;
            }
            Ordering::Less => {
                transitions += 1;
                i += 1;
            }
            Ordering::Greater => {
                transitions += 1;
                j += 1;
            }
        }
    }

    transitions + (left_ids.len() - i) + (right_ids.len() - j)
}

fn build_cluster_layouts<C: Clone>(
    clusters: &[ClusterSpec<C>],
    component_nodes: &HashSet<NodeIndex>,
    node_bounds: &HashMap<NodeIndex, Rect>,
    render_config: &RenderConfig,
) -> Vec<ClusterLayout<C>> {
    #[derive(Clone)]
    struct ClusterBounds<C> {
        id: C,
        parent: Option<usize>,
        padding: f64,
        content_bounds: Option<Rect>,
        final_bounds: Option<Rect>,
    }

    let mut cluster_bounds = clusters
        .iter()
        .enumerate()
        .map(|(cluster_index, cluster)| {
            if !cluster
                .nodes
                .iter()
                .all(|node| component_nodes.contains(node))
            {
                return None;
            }

            let mut content_bounds: Option<Rect> = None;
            for node in &cluster.nodes {
                if let Some(node_bounds) = node_bounds.get(node) {
                    content_bounds = Some(match content_bounds {
                        Some(current) => current.union(*node_bounds),
                        None => *node_bounds,
                    });
                }
            }

            let padding = cluster.padding.unwrap_or(render_config.cluster_padding);
            let parent = cluster
                .parent
                .filter(|parent| *parent < clusters.len() && *parent != cluster_index);

            Some(ClusterBounds {
                id: cluster.id.clone(),
                parent,
                padding,
                content_bounds,
                final_bounds: content_bounds.map(|bounds| bounds.expand(padding)),
            })
        })
        .collect::<Vec<_>>();

    let cluster_count = cluster_bounds.len();
    for _ in 0..cluster_count {
        let mut changed = false;
        for cluster_index in 0..cluster_count {
            let Some(cluster) = cluster_bounds[cluster_index].as_ref() else {
                continue;
            };
            let Some(parent_index) = cluster.parent else {
                continue;
            };
            let Some(child_bounds) = cluster.final_bounds else {
                continue;
            };
            let Some(parent) = cluster_bounds[parent_index].as_mut() else {
                continue;
            };

            let next_content = Some(match parent.content_bounds {
                Some(content) => content.union(child_bounds),
                None => child_bounds,
            });
            if parent.content_bounds != next_content {
                parent.content_bounds = next_content;
                parent.final_bounds = parent
                    .content_bounds
                    .map(|bounds| bounds.expand(parent.padding));
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    let mut component_clusters = Vec::new();
    for cluster in cluster_bounds.into_iter().flatten() {
        if let Some(bounds) = cluster.final_bounds {
            component_clusters.push(ClusterLayout {
                id: cluster.id,
                bounds,
                parent: cluster.parent,
            });
        }
    }
    component_clusters
}

fn extents<L, C>(
    nodes: &[RoutedNode<NodeIndex>],
    edges: &[RoutedEdge<NodeIndex, L>],
    clusters: &[ClusterLayout<C>],
) -> Option<(f64, f64, f64, f64)> {
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    let mut update = |x: f64, y: f64| {
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    };

    for node in nodes {
        update(node.bounds.min_x, node.bounds.min_y);
        update(node.bounds.max_x, node.bounds.max_y);
    }

    for edge in edges {
        for point in &edge.points {
            update(point.0, point.1);
        }
        if let Some(label) = &edge.label {
            update(label.position.0, label.position.1);
        }
    }

    for cluster in clusters {
        update(cluster.bounds.min_x, cluster.bounds.min_y);
        update(cluster.bounds.max_x, cluster.bounds.max_y);
    }

    if min_x.is_finite() && min_y.is_finite() && max_x.is_finite() && max_y.is_finite() {
        Some((min_x, min_y, max_x, max_y))
    } else {
        None
    }
}

fn anchor_points(tail_rect: Rect, head_rect: Rect) -> ((f64, f64), (f64, f64)) {
    let tail_center = (
        (tail_rect.min_x + tail_rect.max_x) * 0.5,
        (tail_rect.min_y + tail_rect.max_y) * 0.5,
    );
    let head_center = (
        (head_rect.min_x + head_rect.max_x) * 0.5,
        (head_rect.min_y + head_rect.max_y) * 0.5,
    );

    (tail_center, head_center)
}

fn label_obstacle_rect(position: (f64, f64)) -> Option<Rect> {
    if EDGE_LABEL_OBSTACLE_WIDTH <= EPSILON || EDGE_LABEL_OBSTACLE_HEIGHT <= EPSILON {
        return None;
    }

    Some(
        Rect::from_center_size(
            position,
            (EDGE_LABEL_OBSTACLE_WIDTH, EDGE_LABEL_OBSTACLE_HEIGHT),
        )
        .expand(EDGE_LABEL_OBSTACLE_PADDING.max(0.0)),
    )
}

fn cluster_border_label_obstacles<C>(clusters: &[ClusterLayout<C>]) -> Vec<Rect> {
    let mut obstacles = Vec::new();
    let half_thickness = CLUSTER_BORDER_LABEL_OBSTACLE_THICKNESS * 0.5;
    for cluster in clusters {
        let bounds = cluster.bounds;
        if bounds.max_x - bounds.min_x <= EPSILON || bounds.max_y - bounds.min_y <= EPSILON {
            continue;
        }

        obstacles.push(Rect {
            min_x: bounds.min_x - half_thickness,
            max_x: bounds.max_x + half_thickness,
            min_y: bounds.min_y - half_thickness,
            max_y: bounds.min_y + half_thickness,
        });
        obstacles.push(Rect {
            min_x: bounds.min_x - half_thickness,
            max_x: bounds.max_x + half_thickness,
            min_y: bounds.max_y - half_thickness,
            max_y: bounds.max_y + half_thickness,
        });
        obstacles.push(Rect {
            min_x: bounds.min_x - half_thickness,
            max_x: bounds.min_x + half_thickness,
            min_y: bounds.min_y - half_thickness,
            max_y: bounds.max_y + half_thickness,
        });
        obstacles.push(Rect {
            min_x: bounds.max_x - half_thickness,
            max_x: bounds.max_x + half_thickness,
            min_y: bounds.min_y - half_thickness,
            max_y: bounds.max_y + half_thickness,
        });
    }
    obstacles
}

fn choose_edge_label_position(
    points: &[(f64, f64)],
    node_obstacles: &[Rect],
    cluster_obstacles: &[Rect],
    placed_label_obstacles: &[Rect],
) -> Option<(f64, f64)> {
    let mut best_relaxed: Option<((f64, f64), f64)> = None;

    for position in label_position_candidates(points) {
        let Some(label_rect) = label_obstacle_rect(position) else {
            continue;
        };
        if !label_rect_clear_of_structures(label_rect, node_obstacles, cluster_obstacles) {
            continue;
        }

        let overlap = total_label_overlap_area(label_rect, placed_label_obstacles);
        if overlap <= EPSILON {
            return Some(position);
        }

        match best_relaxed {
            Some((_, best_overlap)) if overlap + EPSILON >= best_overlap => {}
            _ => best_relaxed = Some((position, overlap)),
        }
    }

    best_relaxed.map(|(position, _)| position)
}

fn label_position_candidates(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    if points.is_empty() {
        return Vec::new();
    }

    let total_length = polyline_length(points);
    if total_length <= EPSILON {
        return vec![points[0]];
    }

    let mut candidates = Vec::new();
    let midpoint_distance = total_length * 0.5;
    let step = (EDGE_LABEL_OBSTACLE_WIDTH * 0.25).max(4.0);
    let max_offset = midpoint_distance;

    if let Some(midpoint) = polyline_point_at_fraction(points, 0.5) {
        candidates.push(midpoint);
    }

    let mut offset = step;
    while offset <= max_offset + EPSILON {
        let left_distance = (midpoint_distance - offset).max(0.0);
        let right_distance = (midpoint_distance + offset).min(total_length);
        let left_fraction = left_distance / total_length;
        let right_fraction = right_distance / total_length;

        if let Some(left) = polyline_point_at_fraction(points, left_fraction) {
            if !candidates
                .iter()
                .any(|candidate| same_point(*candidate, left))
            {
                candidates.push(left);
            }
        }
        if let Some(right) = polyline_point_at_fraction(points, right_fraction) {
            if !candidates
                .iter()
                .any(|candidate| same_point(*candidate, right))
            {
                candidates.push(right);
            }
        }
        offset += step;
    }

    candidates
}

fn label_rect_clear_of_structures(
    label_rect: Rect,
    node_obstacles: &[Rect],
    cluster_obstacles: &[Rect],
) -> bool {
    !node_obstacles
        .iter()
        .any(|obstacle| rects_intersect(label_rect, *obstacle))
        && !cluster_obstacles
            .iter()
            .any(|obstacle| rects_intersect(label_rect, *obstacle))
}

fn total_label_overlap_area(label_rect: Rect, placed_label_obstacles: &[Rect]) -> f64 {
    placed_label_obstacles
        .iter()
        .map(|obstacle| rect_overlap_area(label_rect, *obstacle))
        .sum()
}

fn rect_overlap_area(a: Rect, b: Rect) -> f64 {
    let overlap_x = (a.max_x.min(b.max_x) - a.min_x.max(b.min_x)).max(0.0);
    let overlap_y = (a.max_y.min(b.max_y) - a.min_y.max(b.min_y)).max(0.0);
    overlap_x * overlap_y
}

fn rects_intersect(a: Rect, b: Rect) -> bool {
    let overlap_x = a.max_x.min(b.max_x) - a.min_x.max(b.min_x);
    let overlap_y = a.max_y.min(b.max_y) - a.min_y.max(b.min_y);
    overlap_x > EPSILON && overlap_y > EPSILON
}

#[cfg(any())]
fn route_polyline(
    start: (f64, f64),
    end: (f64, f64),
    obstacles: &[Rect],
    bend_penalty: f64,
) -> Vec<(f64, f64)> {
    route_polyline_with_context(start, end, &[], obstacles, None, bend_penalty)
}

fn route_polyline_with_context(
    start: (f64, f64),
    end: (f64, f64),
    node_obstacles: &[(NodeIndex, Rect)],
    label_obstacles: &[Rect],
    excluded_node_obstacles: Option<(NodeIndex, NodeIndex)>,
    bend_penalty: f64,
) -> Vec<(f64, f64)> {
    let mut candidates = Vec::new();

    if almost_equal(start.0, end.0) || almost_equal(start.1, end.1) {
        let straight = vec![start, end];
        if polyline_clear_with_context(
            &straight,
            node_obstacles,
            label_obstacles,
            excluded_node_obstacles,
        ) {
            candidates.push(straight);
        }
    }

    let candidate_a = vec![start, (start.0, end.1), end];
    if polyline_clear_with_context(
        &candidate_a,
        node_obstacles,
        label_obstacles,
        excluded_node_obstacles,
    ) {
        candidates.push(simplify_polyline(&candidate_a));
    }

    let candidate_b = vec![start, (end.0, start.1), end];
    if polyline_clear_with_context(
        &candidate_b,
        node_obstacles,
        label_obstacles,
        excluded_node_obstacles,
    ) {
        candidates.push(simplify_polyline(&candidate_b));
    }

    if let Some(best) = choose_shortest(&candidates) {
        return best;
    }

    if let Some(grid_route) = route_via_grid(
        start,
        end,
        node_obstacles,
        label_obstacles,
        excluded_node_obstacles,
        bend_penalty,
    ) {
        return grid_route;
    }

    vec![start, end]
}

fn choose_shortest(candidates: &[Vec<(f64, f64)>]) -> Option<Vec<(f64, f64)>> {
    let mut best_index: Option<usize> = None;
    let mut best_len = f64::INFINITY;

    for (index, candidate) in candidates.iter().enumerate() {
        let length = polyline_length(candidate);
        if length < best_len {
            best_len = length;
            best_index = Some(index);
        }
    }

    best_index.and_then(|index| candidates.get(index).cloned())
}

fn route_via_grid(
    start: (f64, f64),
    end: (f64, f64),
    node_obstacles: &[(NodeIndex, Rect)],
    label_obstacles: &[Rect],
    excluded_node_obstacles: Option<(NodeIndex, NodeIndex)>,
    bend_penalty: f64,
) -> Option<Vec<(f64, f64)>> {
    let mut xs = vec![start.0, end.0];
    let mut ys = vec![start.1, end.1];
    for (node_id, obstacle) in node_obstacles {
        if is_excluded_node_obstacle(*node_id, excluded_node_obstacles) {
            continue;
        }
        xs.push(obstacle.min_x);
        xs.push(obstacle.max_x);
        ys.push(obstacle.min_y);
        ys.push(obstacle.max_y);
    }
    for obstacle in label_obstacles {
        xs.push(obstacle.min_x);
        xs.push(obstacle.max_x);
        ys.push(obstacle.min_y);
        ys.push(obstacle.max_y);
    }

    xs = dedup_sorted(xs);
    ys = dedup_sorted(ys);

    let start_ix = find_axis_index(&xs, start.0)?;
    let end_ix = find_axis_index(&xs, end.0)?;
    let start_iy = find_axis_index(&ys, start.1)?;
    let end_iy = find_axis_index(&ys, end.1)?;

    let mut points = Vec::new();
    let y_len = ys.len();
    let point_id_grid_len = xs.len().checked_mul(y_len)?;
    let mut point_id_grid = vec![None::<usize>; point_id_grid_len];
    let grid_offset = |ix: usize, iy: usize| ix * y_len + iy;

    for (ix, x) in xs.iter().enumerate() {
        for (iy, y) in ys.iter().enumerate() {
            let point = (*x, *y);
            if point_allowed_with_context(
                point,
                start,
                end,
                node_obstacles,
                label_obstacles,
                excluded_node_obstacles,
            ) {
                let idx = points.len();
                points.push(point);
                point_id_grid[grid_offset(ix, iy)] = Some(idx);
            }
        }
    }

    let start_id = point_id_grid
        .get(grid_offset(start_ix, start_iy))
        .and_then(|id| *id)?;
    let end_id = point_id_grid
        .get(grid_offset(end_ix, end_iy))
        .and_then(|id| *id)?;

    let mut adjacency = vec![Vec::new(); points.len()];

    for iy in 0..ys.len() {
        for ix in 0..xs.len().saturating_sub(1) {
            let left = match point_id_grid.get(grid_offset(ix, iy)).and_then(|id| *id) {
                Some(id) => id,
                None => continue,
            };
            let right = match point_id_grid
                .get(grid_offset(ix + 1, iy))
                .and_then(|id| *id)
            {
                Some(id) => id,
                None => continue,
            };
            let p1 = points[left];
            let p2 = points[right];
            if segment_clear_with_context(
                p1,
                p2,
                node_obstacles,
                label_obstacles,
                excluded_node_obstacles,
            ) {
                let length = manhattan_distance(p1, p2);
                adjacency[left].push((right, length, MoveDir::Horizontal));
                adjacency[right].push((left, length, MoveDir::Horizontal));
            }
        }
    }

    for ix in 0..xs.len() {
        for iy in 0..ys.len().saturating_sub(1) {
            let top = match point_id_grid.get(grid_offset(ix, iy)).and_then(|id| *id) {
                Some(id) => id,
                None => continue,
            };
            let bottom = match point_id_grid
                .get(grid_offset(ix, iy + 1))
                .and_then(|id| *id)
            {
                Some(id) => id,
                None => continue,
            };
            let p1 = points[top];
            let p2 = points[bottom];
            if segment_clear_with_context(
                p1,
                p2,
                node_obstacles,
                label_obstacles,
                excluded_node_obstacles,
            ) {
                let length = manhattan_distance(p1, p2);
                adjacency[top].push((bottom, length, MoveDir::Vertical));
                adjacency[bottom].push((top, length, MoveDir::Vertical));
            }
        }
    }

    let route = shortest_route(start_id, end_id, &adjacency, bend_penalty, &points)?;
    Some(simplify_polyline(&route))
}

fn shortest_route(
    start: usize,
    end: usize,
    adjacency: &[Vec<(usize, f64, MoveDir)>],
    bend_penalty: f64,
    points: &[(f64, f64)],
) -> Option<Vec<(f64, f64)>> {
    let mut distances = vec![[f64::INFINITY; 3]; adjacency.len()];
    let mut previous = vec![[None::<(usize, MoveDir)>; 3]; adjacency.len()];

    let mut heap = BinaryHeap::new();
    distances[start][MoveDir::None.index()] = 0.0;
    heap.push(RouteState {
        cost: 0.0,
        node: start,
        direction: MoveDir::None,
    });

    while let Some(state) = heap.pop() {
        let dir_index = state.direction.index();
        if state.cost > distances[state.node][dir_index] + EPSILON {
            continue;
        }

        for (next, length, move_dir) in &adjacency[state.node] {
            let bend = if state.direction == MoveDir::None || state.direction == *move_dir {
                0.0
            } else {
                bend_penalty
            };

            let next_cost = state.cost + *length + bend;
            let next_dir_idx = move_dir.index();
            if next_cost + EPSILON < distances[*next][next_dir_idx] {
                distances[*next][next_dir_idx] = next_cost;
                previous[*next][next_dir_idx] = Some((state.node, state.direction));
                heap.push(RouteState {
                    cost: next_cost,
                    node: *next,
                    direction: *move_dir,
                });
            }
        }
    }

    let mut best_direction = MoveDir::None;
    let mut best_cost = distances[end][MoveDir::None.index()];
    for direction in [MoveDir::Horizontal, MoveDir::Vertical] {
        let current = distances[end][direction.index()];
        if current < best_cost {
            best_cost = current;
            best_direction = direction;
        }
    }

    if !best_cost.is_finite() {
        return None;
    }

    let mut route_points = Vec::new();
    let mut current_node = end;
    let mut current_dir = best_direction;

    route_points.push(points[current_node]);
    while current_node != start {
        let prev = previous[current_node][current_dir.index()]?;
        current_node = prev.0;
        current_dir = prev.1;
        route_points.push(points[current_node]);
    }

    route_points.reverse();
    Some(route_points)
}

fn point_allowed_with_context(
    point: (f64, f64),
    start: (f64, f64),
    end: (f64, f64),
    node_obstacles: &[(NodeIndex, Rect)],
    label_obstacles: &[Rect],
    excluded_node_obstacles: Option<(NodeIndex, NodeIndex)>,
) -> bool {
    if same_point(point, start) || same_point(point, end) {
        return true;
    }

    !node_obstacles.iter().any(|(node_id, obstacle)| {
        !is_excluded_node_obstacle(*node_id, excluded_node_obstacles)
            && point_inside_rect(point, *obstacle)
    }) && !label_obstacles
        .iter()
        .any(|obstacle| point_inside_rect(point, *obstacle))
}

fn point_inside_rect(point: (f64, f64), rect: Rect) -> bool {
    point.0 > rect.min_x + EPSILON
        && point.0 < rect.max_x - EPSILON
        && point.1 > rect.min_y + EPSILON
        && point.1 < rect.max_y - EPSILON
}

fn dedup_sorted(mut values: Vec<f64>) -> Vec<f64> {
    values.sort_by(|a, b| a.total_cmp(b));
    values.dedup_by(|a, b| almost_equal(*a, *b));
    values
}

fn find_axis_index(values: &[f64], value: f64) -> Option<usize> {
    values
        .iter()
        .enumerate()
        .find_map(|(idx, v)| almost_equal(*v, value).then_some(idx))
}

#[cfg(any())]
fn polyline_clear(points: &[(f64, f64)], obstacles: &[Rect]) -> bool {
    polyline_clear_with_context(points, &[], obstacles, None)
}

fn polyline_clear_with_context(
    points: &[(f64, f64)],
    node_obstacles: &[(NodeIndex, Rect)],
    label_obstacles: &[Rect],
    excluded_node_obstacles: Option<(NodeIndex, NodeIndex)>,
) -> bool {
    if points.len() < 2 {
        return true;
    }

    for segment in points.windows(2) {
        if !segment_clear_with_context(
            segment[0],
            segment[1],
            node_obstacles,
            label_obstacles,
            excluded_node_obstacles,
        ) {
            return false;
        }
    }

    true
}

fn segment_clear_with_context(
    start: (f64, f64),
    end: (f64, f64),
    node_obstacles: &[(NodeIndex, Rect)],
    label_obstacles: &[Rect],
    excluded_node_obstacles: Option<(NodeIndex, NodeIndex)>,
) -> bool {
    if same_point(start, end) {
        return true;
    }

    if almost_equal(start.1, end.1) {
        let y = start.1;
        let min_x = start.0.min(end.0);
        let max_x = start.0.max(end.0);
        for (node_id, obstacle) in node_obstacles {
            if is_excluded_node_obstacle(*node_id, excluded_node_obstacles) {
                continue;
            }
            if y > obstacle.min_y + EPSILON
                && y < obstacle.max_y - EPSILON
                && ranges_overlap(min_x, max_x, obstacle.min_x, obstacle.max_x)
            {
                return false;
            }
        }
        for obstacle in label_obstacles {
            if y > obstacle.min_y + EPSILON
                && y < obstacle.max_y - EPSILON
                && ranges_overlap(min_x, max_x, obstacle.min_x, obstacle.max_x)
            {
                return false;
            }
        }
        true
    } else if almost_equal(start.0, end.0) {
        let x = start.0;
        let min_y = start.1.min(end.1);
        let max_y = start.1.max(end.1);
        for (node_id, obstacle) in node_obstacles {
            if is_excluded_node_obstacle(*node_id, excluded_node_obstacles) {
                continue;
            }
            if x > obstacle.min_x + EPSILON
                && x < obstacle.max_x - EPSILON
                && ranges_overlap(min_y, max_y, obstacle.min_y, obstacle.max_y)
            {
                return false;
            }
        }
        for obstacle in label_obstacles {
            if x > obstacle.min_x + EPSILON
                && x < obstacle.max_x - EPSILON
                && ranges_overlap(min_y, max_y, obstacle.min_y, obstacle.max_y)
            {
                return false;
            }
        }
        true
    } else {
        false
    }
}

fn is_excluded_node_obstacle(
    node_id: NodeIndex,
    excluded_node_obstacles: Option<(NodeIndex, NodeIndex)>,
) -> bool {
    match excluded_node_obstacles {
        Some((tail, head)) => node_id == tail || node_id == head,
        None => false,
    }
}

fn ranges_overlap(a1: f64, a2: f64, b1: f64, b2: f64) -> bool {
    let left = a1.max(b1);
    let right = a2.min(b2);
    right > left + EPSILON
}

fn simplify_polyline(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    if points.len() <= 2 {
        return points.to_vec();
    }

    let mut simplified = Vec::new();
    simplified.push(points[0]);

    for i in 1..points.len() - 1 {
        let prev = points[i - 1];
        let current = points[i];
        let next = points[i + 1];

        let is_vertical = almost_equal(prev.0, current.0) && almost_equal(current.0, next.0);
        let is_horizontal = almost_equal(prev.1, current.1) && almost_equal(current.1, next.1);

        if !(is_vertical || is_horizontal) {
            simplified.push(current);
        }
    }

    if let Some(last) = points.last() {
        simplified.push(*last);
    }

    simplified
}

fn polyline_length(points: &[(f64, f64)]) -> f64 {
    points
        .windows(2)
        .map(|segment| manhattan_distance(segment[0], segment[1]))
        .sum()
}

fn polyline_midpoint(points: &[(f64, f64)]) -> (f64, f64) {
    if points.is_empty() {
        return (0.0, 0.0);
    }

    if points.len() == 1 {
        return points[0];
    }

    let total_length = polyline_length(points);
    if total_length <= EPSILON {
        return points[0];
    }

    let mut traversed = 0.0;
    let midpoint = total_length * 0.5;

    for segment in points.windows(2) {
        let start = segment[0];
        let end = segment[1];
        let length = manhattan_distance(start, end);
        if traversed + length >= midpoint {
            let remaining = midpoint - traversed;
            let t = if length > EPSILON {
                remaining / length
            } else {
                0.0
            };
            return (
                start.0 + (end.0 - start.0) * t,
                start.1 + (end.1 - start.1) * t,
            );
        }
        traversed += length;
    }

    points[points.len() - 1]
}

fn polyline_point_at_fraction(points: &[(f64, f64)], fraction: f64) -> Option<(f64, f64)> {
    if points.is_empty() {
        return None;
    }
    if points.len() == 1 {
        return Some(points[0]);
    }

    let total_length = polyline_length(points);
    if total_length <= EPSILON {
        return Some(points[0]);
    }

    let clamped = fraction.clamp(0.0, 1.0);
    let target_length = total_length * clamped;
    let mut traversed = 0.0;

    for segment in points.windows(2) {
        let start = segment[0];
        let end = segment[1];
        let length = manhattan_distance(start, end);
        if traversed + length >= target_length {
            if length <= EPSILON {
                return Some(start);
            }
            let remaining = target_length - traversed;
            let t = remaining / length;
            return Some((
                start.0 + (end.0 - start.0) * t,
                start.1 + (end.1 - start.1) * t,
            ));
        }
        traversed += length;
    }

    points.last().copied()
}

fn manhattan_distance(a: (f64, f64), b: (f64, f64)) -> f64 {
    (a.0 - b.0).abs() + (a.1 - b.1).abs()
}

fn almost_equal(a: f64, b: f64) -> bool {
    (a - b).abs() <= EPSILON
}

fn same_point(a: (f64, f64), b: (f64, f64)) -> bool {
    almost_equal(a.0, b.0) && almost_equal(a.1, b.1)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MoveDir {
    None,
    Horizontal,
    Vertical,
}

impl MoveDir {
    fn index(self) -> usize {
        match self {
            Self::None => 0,
            Self::Horizontal => 1,
            Self::Vertical => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct RouteState {
    cost: f64,
    node: usize,
    direction: MoveDir,
}

impl Eq for RouteState {}

impl Ord for RouteState {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost
            .total_cmp(&self.cost)
            .then_with(|| self.node.cmp(&other.node))
    }
}

impl PartialOrd for RouteState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(any())]
mod tests {
    use petgraph::stable_graph::{NodeIndex, StableDiGraph};

    use super::{
        apply_graphviz_cluster_constraints, choose_edge_label_position, from_graph_with_features,
        label_obstacle_rect, polyline_clear, polyline_midpoint, rects_intersect, route_polyline,
        ClusterSpec, Rect, RenderConfig, RoutedNode, EPSILON,
    };
    use crate::configure::Config;

    #[test]
    fn routing_avoids_obstacle() {
        let start = (0.0, 0.0);
        let end = (10.0, 0.0);
        let obstacle = Rect {
            min_x: 4.0,
            min_y: -1.0,
            max_x: 6.0,
            max_y: 1.0,
        };

        let path = route_polyline(start, end, &[obstacle], 5.0);
        assert!(!path.is_empty());
        assert_eq!(path[0], start);
        assert_eq!(path[path.len() - 1], end);
        assert!(polyline_clear(&path, &[obstacle]));
    }

    #[test]
    fn midpoint_for_orthogonal_polyline() {
        let points = vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)];
        let midpoint = polyline_midpoint(&points);
        assert_eq!(midpoint, (10.0, 0.0));
    }

    #[test]
    fn features_include_labels_and_clusters() {
        let mut graph = StableDiGraph::<&str, &str>::new();
        let a = graph.add_node("A");
        let b = graph.add_node("B");
        let c = graph.add_node("C");

        graph.add_edge(a, b, "a->b");
        graph.add_edge(b, c, "b->c");

        let clusters = vec![ClusterSpec {
            id: "group-1",
            nodes: vec![a, b],
            padding: Some(4.0),
            parent: None,
        }];

        let layouts = from_graph_with_features(
            &graph,
            &|_, _| (20.0, 10.0),
            &|_, edge| Some((*edge).to_string()),
            &clusters,
            &Config::default(),
            &RenderConfig::default(),
        );

        assert_eq!(layouts.len(), 1);
        let layout = &layouts[0];
        assert_eq!(layout.nodes.len(), 3);
        assert_eq!(layout.edges.len(), 2);
        assert_eq!(layout.clusters.len(), 1);
        assert!(layout.width > 0.0);
        assert!(layout.height > 0.0);
        assert!(layout.edges.iter().all(|edge| edge.label.is_some()));
    }

    #[test]
    fn nested_clusters_include_child_bounds() {
        let mut graph = StableDiGraph::<&str, &str>::new();
        let a = graph.add_node("A");
        let b = graph.add_node("B");
        graph.add_edge(a, b, "a->b");

        let clusters = vec![
            ClusterSpec {
                id: "parent",
                nodes: Vec::new(),
                padding: Some(3.0),
                parent: None,
            },
            ClusterSpec {
                id: "child",
                nodes: vec![a, b],
                padding: Some(5.0),
                parent: Some(0),
            },
        ];

        let layouts = from_graph_with_features(
            &graph,
            &|_, _| (20.0, 10.0),
            &|_, _| None::<String>,
            &clusters,
            &Config::default(),
            &RenderConfig::default(),
        );
        assert_eq!(layouts.len(), 1);

        let mut parent_bounds = None;
        let mut child_bounds = None;
        for cluster in &layouts[0].clusters {
            if cluster.id == "parent" {
                parent_bounds = Some(cluster.bounds);
            }
            if cluster.id == "child" {
                child_bounds = Some(cluster.bounds);
            }
        }

        if let (Some(parent), Some(child)) = (parent_bounds, child_bounds) {
            assert!(parent.min_x <= child.min_x + EPSILON);
            assert!(parent.min_y <= child.min_y + EPSILON);
            assert!(parent.max_x >= child.max_x - EPSILON);
            assert!(parent.max_y >= child.max_y - EPSILON);
        } else {
            assert!(parent_bounds.is_some() && child_bounds.is_some());
        }
    }

    #[test]
    fn cluster_constraints_make_members_contiguous_in_rank() {
        let mut nodes = vec![
            RoutedNode {
                id: NodeIndex::new(0),
                center: (0.0, 0.0),
                size: (10.0, 10.0),
                bounds: Rect::from_center_size((0.0, 0.0), (10.0, 10.0)),
            },
            RoutedNode {
                id: NodeIndex::new(1),
                center: (20.0, 0.0),
                size: (10.0, 10.0),
                bounds: Rect::from_center_size((20.0, 0.0), (10.0, 10.0)),
            },
            RoutedNode {
                id: NodeIndex::new(2),
                center: (40.0, 0.0),
                size: (10.0, 10.0),
                bounds: Rect::from_center_size((40.0, 0.0), (10.0, 10.0)),
            },
            RoutedNode {
                id: NodeIndex::new(3),
                center: (60.0, 0.0),
                size: (10.0, 10.0),
                bounds: Rect::from_center_size((60.0, 0.0), (10.0, 10.0)),
            },
        ];

        let clusters = vec![ClusterSpec {
            id: "cluster",
            nodes: vec![NodeIndex::new(0), NodeIndex::new(2)],
            padding: None,
            parent: None,
        }];
        let config = Config {
            vertex_spacing: 2.0,
            ..Default::default()
        };
        let render_config = RenderConfig {
            cluster_constraint_iterations: 2,
            cluster_boundary_gap: 4.0,
            ..Default::default()
        };

        apply_graphviz_cluster_constraints(&mut nodes, &clusters, &config, &render_config);

        let mut order = nodes
            .iter()
            .map(|node| (node.id, node.center.0))
            .collect::<Vec<_>>();
        order.sort_by(|a, b| a.1.total_cmp(&b.1));
        let ordered_ids = order.into_iter().map(|(id, _)| id).collect::<Vec<_>>();

        let pos_0 = ordered_ids.iter().position(|id| *id == NodeIndex::new(0));
        let pos_2 = ordered_ids.iter().position(|id| *id == NodeIndex::new(2));

        assert!(matches!(
            (pos_0, pos_2),
            (Some(a), Some(b)) if a.abs_diff(b) == 1
        ));
    }

    #[test]
    fn cluster_constraints_prevent_disjoint_cluster_overlap() {
        let mut nodes = vec![
            RoutedNode {
                id: NodeIndex::new(0),
                center: (0.0, 0.0),
                size: (10.0, 10.0),
                bounds: Rect::from_center_size((0.0, 0.0), (10.0, 10.0)),
            },
            RoutedNode {
                id: NodeIndex::new(1),
                center: (30.0, 0.0),
                size: (10.0, 10.0),
                bounds: Rect::from_center_size((30.0, 0.0), (10.0, 10.0)),
            },
            RoutedNode {
                id: NodeIndex::new(2),
                center: (0.0, 40.0),
                size: (10.0, 10.0),
                bounds: Rect::from_center_size((0.0, 40.0), (10.0, 10.0)),
            },
            RoutedNode {
                id: NodeIndex::new(3),
                center: (30.0, 40.0),
                size: (10.0, 10.0),
                bounds: Rect::from_center_size((30.0, 40.0), (10.0, 10.0)),
            },
        ];

        let clusters = vec![
            ClusterSpec {
                id: "left-right",
                nodes: vec![NodeIndex::new(0), NodeIndex::new(3)],
                padding: Some(0.0),
                parent: None,
            },
            ClusterSpec {
                id: "right-left",
                nodes: vec![NodeIndex::new(1), NodeIndex::new(2)],
                padding: Some(0.0),
                parent: None,
            },
        ];
        let config = Config {
            vertex_spacing: 2.0,
            ..Default::default()
        };
        let render_config = RenderConfig {
            cluster_constraint_iterations: 4,
            cluster_boundary_gap: 4.0,
            cluster_padding: 0.0,
            ..Default::default()
        };

        apply_graphviz_cluster_constraints(&mut nodes, &clusters, &config, &render_config);

        let cluster_a_bounds = nodes
            .iter()
            .filter(|node| node.id == NodeIndex::new(0) || node.id == NodeIndex::new(3))
            .fold(None::<Rect>, |current, node| {
                let bounds = Rect::from_center_size(node.center, node.size);
                Some(match current {
                    Some(acc) => acc.union(bounds),
                    None => bounds,
                })
            });
        let cluster_b_bounds = nodes
            .iter()
            .filter(|node| node.id == NodeIndex::new(1) || node.id == NodeIndex::new(2))
            .fold(None::<Rect>, |current, node| {
                let bounds = Rect::from_center_size(node.center, node.size);
                Some(match current {
                    Some(acc) => acc.union(bounds),
                    None => bounds,
                })
            });

        if let (Some(a), Some(b)) = (cluster_a_bounds, cluster_b_bounds) {
            let overlap_x = a.max_x.min(b.max_x) - a.min_x.max(b.min_x);
            let overlap_y = a.max_y.min(b.max_y) - a.min_y.max(b.min_y);
            assert!(overlap_x <= EPSILON || overlap_y <= EPSILON);
        } else {
            assert!(cluster_a_bounds.is_some() && cluster_b_bounds.is_some());
        }
    }

    #[test]
    fn routed_edges_avoid_previously_placed_labels() {
        let mut graph = StableDiGraph::<(), ()>::new();
        let source = graph.add_node(());
        let mid_a = graph.add_node(());
        let mid_b = graph.add_node(());
        let target = graph.add_node(());

        graph.add_edge(source, target, ());
        graph.add_edge(source, target, ());
        graph.add_edge(source, mid_a, ());
        graph.add_edge(mid_a, mid_b, ());
        graph.add_edge(mid_b, target, ());

        let layouts = from_graph_with_features(
            &graph,
            &|_, _| (20.0, 12.0),
            &|_, _| Some("label"),
            &[] as &[ClusterSpec<()>],
            &Config::default(),
            &RenderConfig::default(),
        );

        assert_eq!(layouts.len(), 1);
        let long_edges = layouts[0]
            .edges
            .iter()
            .filter(|edge| edge.tail == source && edge.head == target)
            .collect::<Vec<_>>();
        assert_eq!(long_edges.len(), 2);

        let first_label_rect = long_edges
            .first()
            .and_then(|edge| edge.label.as_ref())
            .and_then(|label| label_obstacle_rect(label.position));
        let second_edge = long_edges.get(1).copied();

        if let (Some(rect), Some(edge)) = (first_label_rect, second_edge) {
            assert!(polyline_clear(&edge.points, &[rect]));
        } else {
            assert!(first_label_rect.is_some() && second_edge.is_some());
        }
    }

    #[test]
    fn label_position_avoids_node_and_cluster_obstacles() {
        let points = vec![(0.0, 0.0), (300.0, 0.0)];
        let node_obstacles = vec![Rect {
            min_x: 130.0,
            min_y: -15.0,
            max_x: 170.0,
            max_y: 15.0,
        }];
        let cluster_obstacles = vec![Rect {
            min_x: 80.0,
            min_y: -15.0,
            max_x: 120.0,
            max_y: 15.0,
        }];

        let position =
            choose_edge_label_position(&points, &node_obstacles, &cluster_obstacles, &[]);
        assert!(position.is_some());

        let rect = position.and_then(label_obstacle_rect);
        assert!(rect.is_some());

        if let Some(label_rect) = rect {
            assert!(!rects_intersect(label_rect, node_obstacles[0]));
            assert!(!rects_intersect(label_rect, cluster_obstacles[0]));
        } else {
            assert!(rect.is_some());
        }
    }

    #[test]
    fn label_position_dodges_previously_placed_labels() {
        let points = vec![(0.0, 0.0), (300.0, 0.0)];

        let first_label_rect = label_obstacle_rect((150.0, 0.0));
        let placed = match first_label_rect {
            Some(rect) => vec![rect],
            None => Vec::new(),
        };

        let second_position = choose_edge_label_position(&points, &[], &[], &placed);
        assert!(second_position.is_some());

        let second_rect = second_position.and_then(label_obstacle_rect);
        assert!(second_rect.is_some());

        if let (Some(first), Some(second)) = (first_label_rect, second_rect) {
            assert!(!rects_intersect(first, second));
        } else {
            assert!(first_label_rect.is_none() || second_rect.is_some());
        }
    }
}
