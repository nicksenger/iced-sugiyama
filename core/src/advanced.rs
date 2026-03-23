use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet};

use petgraph::stable_graph::{EdgeIndex, NodeIndex, StableDiGraph};
use petgraph::visit::{EdgeRef, IntoEdgeReferences};
use petgraph::Direction::{Incoming, Outgoing};

use crate::{from_graph, Config, RenderConfig};

const EPSILON: f64 = 1e-6;
const EDGE_LABEL_OBSTACLE_WIDTH: f64 = 78.0;
const EDGE_LABEL_OBSTACLE_HEIGHT: f64 = 24.0;
const EDGE_LABEL_OBSTACLE_PADDING: f64 = 6.0;
const CLUSTER_BORDER_LABEL_OBSTACLE_THICKNESS: f64 = 3.0;
const CLUSTER_LABEL_RESERVED_HEIGHT: f64 = 28.0;
const CLUSTER_LABEL_VERTICAL_GAP: f64 = 6.0;
const CLUSTER_LABEL_HORIZONTAL_INSET: f64 = 8.0;
const CLUSTER_LABEL_OBSTACLE_HEIGHT: f64 = 22.0;
const CLUSTER_LABEL_OBSTACLE_MIN_WIDTH: f64 = 84.0;
const CLUSTER_LABEL_OBSTACLE_MAX_WIDTH: f64 = 180.0;
const EDGE_LABEL_OFFSET: f64 = 14.0;
const ROUTING_GUIDE_CLEARANCE: f64 = 8.0;
const EDGE_LABEL_NODE_CLEARANCE: f64 = 10.0;
const EDGE_LABEL_CLUSTER_CLEARANCE: f64 = 6.0;
const ROUTE_JOG_CLEARANCE: f64 = 12.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Rect {
    pub min_x: f64,
    pub min_y: f64,
    pub max_x: f64,
    pub max_y: f64,
}

impl Rect {
    pub(crate) fn from_center_size(center: (f64, f64), size: (f64, f64)) -> Self {
        let half_w = size.0 * 0.5;
        let half_h = size.1 * 0.5;
        Self {
            min_x: center.0 - half_w,
            min_y: center.1 - half_h,
            max_x: center.0 + half_w,
            max_y: center.1 + half_h,
        }
    }

    pub(crate) fn expand(self, padding: f64) -> Self {
        Self {
            min_x: self.min_x - padding,
            min_y: self.min_y - padding,
            max_x: self.max_x + padding,
            max_y: self.max_y + padding,
        }
    }

    pub(crate) fn translate(self, dx: f64, dy: f64) -> Self {
        Self {
            min_x: self.min_x + dx,
            min_y: self.min_y + dy,
            max_x: self.max_x + dx,
            max_y: self.max_y + dy,
        }
    }

    pub(crate) fn union(self, other: Self) -> Self {
        Self {
            min_x: self.min_x.min(other.min_x),
            min_y: self.min_y.min(other.min_y),
            max_x: self.max_x.max(other.max_x),
            max_y: self.max_y.max(other.max_y),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RoutedNode<T> {
    pub id: T,
    pub center: (f64, f64),
    pub size: (f64, f64),
    pub bounds: Rect,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct EdgeLabel<L> {
    pub value: L,
    pub position: (f64, f64),
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RoutedEdge<T, L> {
    pub id: EdgeIndex,
    pub tail: T,
    pub head: T,
    pub points: Vec<(f64, f64)>,
    pub curve_points: Vec<(f64, f64)>,
    pub label: Option<EdgeLabel<L>>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ClusterSpec<C> {
    pub id: C,
    pub nodes: Vec<NodeIndex>,
    pub padding: Option<f64>,
    pub parent: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ClusterLayout<C> {
    pub id: C,
    pub bounds: Rect,
    pub parent: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DetailedLayout<T, L, C> {
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

#[derive(Debug, Clone)]
struct ComputedCluster<C> {
    source_index: usize,
    id: C,
    bounds: Rect,
    parent: Option<usize>,
    node_ids: HashSet<NodeIndex>,
}

/// Creates layouts with routed edges, optional edge label placements and
/// optional cluster rectangles.
///
/// This keeps the current layered node placement and adds post-processing
/// features that are commonly needed by renderers.
pub(crate) fn from_graph_with_features<V, E, L, C>(
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
    let edge_labels = graph
        .edge_references()
        .map(|edge| (edge.id(), edge_label(edge.id(), edge.weight())))
        .collect::<HashMap<_, _>>();
    let has_edge_labels = edge_labels.values().any(|label| label.is_some());
    let rank_min_length = graphviz_rank_min_length(config, has_edge_labels);
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

        apply_graphviz_cluster_constraints(
            &mut nodes,
            graph,
            clusters,
            config,
            render_config,
            rank_min_length,
        );

        let mut node_bounds = nodes
            .iter()
            .map(|node| (node.id, node.bounds))
            .collect::<HashMap<_, _>>();

        let mut computed_clusters =
            build_cluster_layouts(clusters, &component_nodes, &node_bounds, render_config);
        compact_cluster_top_fanouts(&mut nodes, &computed_clusters, config.vertex_spacing);
        align_cluster_bottom_singletons(&mut nodes, graph, &computed_clusters);
        node_bounds = nodes
            .iter()
            .map(|node| (node.id, node.bounds))
            .collect::<HashMap<_, _>>();
        computed_clusters =
            build_cluster_layouts(clusters, &component_nodes, &node_bounds, render_config);
        enforce_labeled_edge_rank_clearance(&mut nodes, graph, &edge_labels, render_config);
        node_bounds = nodes
            .iter()
            .map(|node| (node.id, node.bounds))
            .collect::<HashMap<_, _>>();
        computed_clusters =
            build_cluster_layouts(clusters, &component_nodes, &node_bounds, render_config);
        let final_ranks = group_nodes_into_ranks(&nodes);
        center_extreme_single_node_ranks(&final_ranks, &mut nodes);
        center_top_singleton_from_successors(&mut nodes, graph);
        enforce_labeled_edge_rank_clearance(&mut nodes, graph, &edge_labels, render_config);
        node_bounds = nodes
            .iter()
            .map(|node| (node.id, node.bounds))
            .collect::<HashMap<_, _>>();
        computed_clusters =
            build_cluster_layouts(clusters, &component_nodes, &node_bounds, render_config);
        separate_nodes_from_foreign_clusters(
            &mut nodes,
            &computed_clusters,
            render_config
                .cluster_boundary_gap
                .max(render_config.cluster_padding * 0.35),
            render_config.cluster_constraint_iterations.max(1) * 2,
        );
        refine_final_cluster_rank_slots(
            &mut nodes,
            graph,
            clusters,
            render_config.cluster_padding,
            render_config.cluster_boundary_gap,
            render_config.cluster_constraint_iterations.max(1),
        );
        node_bounds = nodes
            .iter()
            .map(|node| (node.id, node.bounds))
            .collect::<HashMap<_, _>>();
        computed_clusters =
            build_cluster_layouts(clusters, &component_nodes, &node_bounds, render_config);
        let mut component_clusters = computed_clusters
            .iter()
            .map(|cluster| ClusterLayout {
                id: cluster.id.clone(),
                bounds: cluster.bounds,
                parent: cluster.parent,
            })
            .collect::<Vec<_>>();

        let mut obstacles = Vec::new();
        for (id, bounds) in &node_bounds {
            obstacles.push((*id, bounds.expand(render_config.routing_padding)));
        }

        let cluster_memberships = computed_clusters
            .iter()
            .map(|cluster| {
                (
                    cluster.source_index,
                    cluster.bounds.expand(render_config.routing_padding * 0.5),
                    cluster.node_ids.clone(),
                )
            })
            .collect::<Vec<_>>();

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
                label_value: edge_labels.get(&edge.id()).cloned().flatten(),
                label_position: None,
            });
        }
        edge_drafts.sort_by_key(|edge| edge.id.index());
        let edge_anchors = compute_edge_anchors(&edge_drafts, &node_bounds, &cluster_memberships);

        let node_label_obstacles = node_bounds
            .values()
            .map(|bounds| bounds.expand(EDGE_LABEL_NODE_CLEARANCE))
            .collect::<Vec<_>>();
        let cluster_route_obstacles = cluster_label_obstacles(&component_clusters);
        let cluster_label_obstacles = cluster_label_obstacles(&component_clusters)
            .into_iter()
            .map(|bounds| bounds.expand(EDGE_LABEL_CLUSTER_CLEARANCE))
            .collect::<Vec<_>>();

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

            let cluster_routing_obstacles = cluster_memberships
                .iter()
                .map(|(cluster_index, bounds, _)| (*cluster_index, *bounds))
                .collect::<Vec<_>>();

            let (start, end) = edge_anchors
                .get(&edge.id)
                .copied()
                .unwrap_or_else(|| anchor_points(tail_rect, head_rect));
            edge.points = route_edge_with_cluster_entries(
                start,
                end,
                edge.tail,
                edge.head,
                &obstacles,
                &cluster_routing_obstacles,
                &cluster_route_obstacles,
                &cluster_memberships,
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
                        .and_then(|_| default_label_position(&edge.points))
                });

            if let Some(position) = edge.label_position {
                if let Some(obstacle) = label_obstacle_rect(position) {
                    placed_label_obstacles.push(obstacle);
                }
            }
        }
        let refined_edge_anchors = compute_edge_anchors_from_routes(&edge_drafts, &node_bounds);
        for edge in &mut edge_drafts {
            let tail_rect = match node_bounds.get(&edge.tail) {
                Some(rect) => *rect,
                None => continue,
            };
            let head_rect = match node_bounds.get(&edge.head) {
                Some(rect) => *rect,
                None => continue,
            };

            let cluster_routing_obstacles = cluster_memberships
                .iter()
                .map(|(cluster_index, bounds, _)| (*cluster_index, *bounds))
                .collect::<Vec<_>>();

            let (start, end) = refined_edge_anchors
                .get(&edge.id)
                .copied()
                .unwrap_or_else(|| anchor_points(tail_rect, head_rect));
            edge.points = route_edge_with_cluster_entries(
                start,
                end,
                edge.tail,
                edge.head,
                &obstacles,
                &cluster_routing_obstacles,
                &cluster_route_obstacles,
                &cluster_memberships,
                render_config.bend_penalty,
            );
            edge.label_position = None;
        }
        let mut placed_label_obstacles = Vec::new();
        for edge in &mut edge_drafts {
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
                        .and_then(|_| default_label_position(&edge.points))
                });

            if let Some(position) = edge.label_position {
                if let Some(obstacle) = label_obstacle_rect(position) {
                    placed_label_obstacles.push(obstacle);
                }
            }
        }
        let mut edges = Vec::with_capacity(edge_drafts.len());
        for edge in edge_drafts {
            let tail_rect = node_bounds.get(&edge.tail).copied().unwrap_or_else(|| {
                Rect::from_center_size(
                    edge.points.first().copied().unwrap_or((0.0, 0.0)),
                    (0.0, 0.0),
                )
            });
            let head_rect = node_bounds.get(&edge.head).copied().unwrap_or_else(|| {
                Rect::from_center_size(
                    edge.points.last().copied().unwrap_or((0.0, 0.0)),
                    (0.0, 0.0),
                )
            });
            let curve_obstacles = node_bounds
                .iter()
                .filter_map(|(node_id, rect)| {
                    (*node_id != edge.tail && *node_id != edge.head).then_some(*rect)
                })
                .chain(computed_clusters.iter().filter_map(|cluster| {
                    (!cluster.node_ids.contains(&edge.tail)
                        && !cluster.node_ids.contains(&edge.head))
                    .then_some(cluster.bounds.expand(3.0))
                }))
                .collect::<Vec<_>>();
            let curve_points = bezier_curve_points_from_polyline(
                &edge.points,
                tail_rect,
                head_rect,
                &curve_obstacles,
            );
            let label = edge.label_value.map(|value| {
                let position = match edge.label_position {
                    Some(position) => position,
                    None => default_label_position(&edge.points)
                        .unwrap_or_else(|| polyline_midpoint(&edge.points)),
                };
                EdgeLabel { value, position }
            });
            edges.push(RoutedEdge {
                id: edge.id,
                tail: edge.tail,
                head: edge.head,
                points: edge.points,
                curve_points,
                label,
            });
        }
        resolve_curve_label_positions(&mut edges, &node_label_obstacles, &cluster_label_obstacles);

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
            for point in &mut edge.curve_points {
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

fn graphviz_rank_min_length(config: &Config, has_edge_labels: bool) -> i32 {
    if has_edge_labels {
        config.minimum_length.saturating_mul(2) as i32
    } else {
        config.minimum_length as i32
    }
}

#[derive(Debug, Clone)]
struct ClusterInternal {
    declaration_index: usize,
    node_ids: HashSet<NodeIndex>,
    node_indices: Vec<usize>,
    padding: f64,
    parent: Option<usize>,
}

fn apply_graphviz_cluster_constraints<V, E, C>(
    nodes: &mut [RoutedNode<NodeIndex>],
    graph: &StableDiGraph<V, E>,
    clusters: &[ClusterSpec<C>],
    config: &Config,
    render_config: &RenderConfig,
    rank_min_length: i32,
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
                declaration_index: cluster_index,
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
    let cluster_depths = build_cluster_depths(&cluster_internals);
    let direct_child_lookup = build_direct_child_lookup(&node_cluster_memberships, &cluster_depths);
    let mut ranks = build_compound_ranks(
        nodes,
        graph,
        &cluster_internals,
        &direct_child_lookup,
        &original_x,
        rank_min_length.max(1),
    );
    if ranks.is_empty() {
        return;
    }

    let iterations = render_config.cluster_constraint_iterations.max(1);
    for _ in 0..iterations {
        sweep_cluster_ordering(
            &mut ranks,
            nodes,
            graph,
            &cluster_internals,
            &direct_child_lookup,
            SweepDirection::Down,
        );
        sweep_cluster_ordering(
            &mut ranks,
            nodes,
            graph,
            &cluster_internals,
            &direct_child_lookup,
            SweepDirection::Up,
        );
        refine_rank_order_by_adjacent_medians(&mut ranks, nodes, graph, &original_x);
        assign_rank_x_positions(
            &ranks,
            nodes,
            &original_x,
            &node_cluster_memberships,
            config.vertex_spacing,
            render_config.cluster_boundary_gap,
            graph,
        );
    }

    assign_rank_y_positions(&ranks, nodes, config.vertex_spacing);

    enforce_cluster_non_overlap(
        nodes,
        &cluster_internals,
        render_config.cluster_boundary_gap,
        iterations.saturating_mul(cluster_internals.len().max(1)),
    );
    center_extreme_single_node_ranks(&ranks, nodes);

    for node in nodes {
        node.bounds = Rect::from_center_size(node.center, node.size);
    }
}

fn restore_compound_rank_slot_order<V, E, C>(
    nodes: &mut [RoutedNode<NodeIndex>],
    graph: &StableDiGraph<V, E>,
    clusters: &[ClusterSpec<C>],
    default_cluster_padding: f64,
    iterations: usize,
) {
    if nodes.is_empty() || clusters.is_empty() {
        return;
    }

    let node_index_by_id = nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.id, index))
        .collect::<HashMap<_, _>>();
    let mut cluster_internals =
        build_cluster_internals(nodes, clusters, default_cluster_padding, &node_index_by_id);
    if cluster_internals.is_empty() {
        return;
    }

    let node_cluster_memberships = build_node_cluster_memberships(nodes, &cluster_internals);
    let cluster_depths = build_cluster_depths(&cluster_internals);
    let direct_child_lookup = build_direct_child_lookup(&node_cluster_memberships, &cluster_depths);
    let mut ranks = group_nodes_into_ranks(nodes);
    if ranks.len() < 2 {
        return;
    }

    let sweep_count = iterations.max(1);
    for _ in 0..sweep_count {
        sweep_cluster_ordering(
            &mut ranks,
            nodes,
            graph,
            &cluster_internals,
            &direct_child_lookup,
            SweepDirection::Down,
        );
        sweep_cluster_ordering(
            &mut ranks,
            nodes,
            graph,
            &cluster_internals,
            &direct_child_lookup,
            SweepDirection::Up,
        );
    }

    for rank in ranks {
        if rank.len() <= 1 {
            continue;
        }

        let mut slot_xs = rank
            .iter()
            .map(|node_index| nodes[*node_index].center.0)
            .collect::<Vec<_>>();
        slot_xs.sort_by(|left, right| left.total_cmp(right));

        for (node_index, slot_x) in rank.into_iter().zip(slot_xs.into_iter()) {
            nodes[node_index].center.0 = slot_x;
            nodes[node_index].bounds =
                Rect::from_center_size(nodes[node_index].center, nodes[node_index].size);
        }
    }
}

fn refine_final_cluster_rank_slots<V, E, C>(
    nodes: &mut [RoutedNode<NodeIndex>],
    graph: &StableDiGraph<V, E>,
    clusters: &[ClusterSpec<C>],
    default_cluster_padding: f64,
    cluster_boundary_gap: f64,
    iterations: usize,
) {
    if nodes.is_empty() || clusters.is_empty() {
        return;
    }

    let node_index_by_id = nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.id, index))
        .collect::<HashMap<_, _>>();
    let cluster_internals =
        build_cluster_internals(nodes, clusters, default_cluster_padding, &node_index_by_id);
    if cluster_internals.is_empty() {
        return;
    }

    let node_cluster_memberships = build_node_cluster_memberships(nodes, &cluster_internals);
    let sibling_cluster_groups = declaration_sibling_cluster_groups(&cluster_internals);
    apply_sibling_cluster_rank_ordering(nodes, &node_cluster_memberships, &sibling_cluster_groups);
    normalize_leaf_cluster_rank_slots_by_external_flow(nodes, graph, &cluster_internals);
    let cluster_depths = build_cluster_depths(&cluster_internals);
    let direct_child_lookup = build_direct_child_lookup(&node_cluster_memberships, &cluster_depths);
    reorder_top_level_rank_blocks(nodes, graph, &cluster_internals, &direct_child_lookup);
    enforce_sibling_cluster_positions(
        nodes,
        &cluster_internals,
        &sibling_cluster_groups,
        cluster_boundary_gap,
        iterations,
    );
}

fn declaration_sibling_cluster_groups(clusters: &[ClusterInternal]) -> Vec<Vec<usize>> {
    let mut grouped = HashMap::<Option<usize>, Vec<usize>>::new();
    for (cluster_index, cluster) in clusters.iter().enumerate() {
        grouped
            .entry(cluster.parent)
            .or_default()
            .push(cluster_index);
    }

    let mut groups = grouped
        .into_values()
        .filter(|group| group.len() > 1)
        .collect::<Vec<_>>();
    for group in &mut groups {
        group.sort_by_key(|cluster_index| clusters[*cluster_index].declaration_index);
    }
    groups.sort_by(|left, right| left[0].cmp(&right[0]));
    groups
}

fn apply_sibling_cluster_rank_ordering(
    nodes: &mut [RoutedNode<NodeIndex>],
    node_cluster_memberships: &[Vec<usize>],
    sibling_cluster_groups: &[Vec<usize>],
) {
    let mut ranks = group_nodes_into_ranks(nodes);
    for rank in &mut ranks {
        for sibling_group in sibling_cluster_groups {
            enforce_sibling_cluster_order(rank, node_cluster_memberships, sibling_group);
        }
    }

    for rank in ranks {
        let mut slot_xs = rank
            .iter()
            .map(|node_index| nodes[*node_index].center.0)
            .collect::<Vec<_>>();
        slot_xs.sort_by(|left, right| left.total_cmp(right));
        for (node_index, slot_x) in rank.into_iter().zip(slot_xs.into_iter()) {
            nodes[node_index].center.0 = slot_x;
            nodes[node_index].bounds =
                Rect::from_center_size(nodes[node_index].center, nodes[node_index].size);
        }
    }
}

fn normalize_leaf_cluster_rank_slots_by_external_flow<V, E>(
    nodes: &mut [RoutedNode<NodeIndex>],
    graph: &StableDiGraph<V, E>,
    clusters: &[ClusterInternal],
) {
    let parent_clusters = clusters
        .iter()
        .filter_map(|cluster| cluster.parent)
        .collect::<HashSet<_>>();
    let ranks = group_nodes_into_ranks(nodes);
    let rank_by_node = ranks
        .iter()
        .enumerate()
        .flat_map(|(rank_index, rank)| {
            rank.iter()
                .copied()
                .map(move |node_index| (node_index, rank_index))
        })
        .collect::<HashMap<_, _>>();
    let node_index_by_id = nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.id, index))
        .collect::<HashMap<_, _>>();

    for (cluster_index, cluster) in clusters.iter().enumerate() {
        if parent_clusters.contains(&cluster_index) {
            continue;
        }

        let mut by_rank = HashMap::<usize, Vec<usize>>::new();
        for node_id in &cluster.node_ids {
            let Some(node_index) = node_index_by_id.get(node_id).copied() else {
                continue;
            };
            let Some(rank_index) = rank_by_node.get(&node_index).copied() else {
                continue;
            };
            by_rank.entry(rank_index).or_default().push(node_index);
        }

        for mut node_indices in by_rank.into_values().filter(|members| members.len() > 1) {
            let mut slot_xs = node_indices
                .iter()
                .map(|node_index| nodes[*node_index].center.0)
                .collect::<Vec<_>>();
            slot_xs.sort_by(|left, right| left.total_cmp(right));
            let row_len = node_indices.len();
            node_indices.sort_by(|left, right| {
                let left_id = nodes[*left].id;
                let right_id = nodes[*right].id;
                let left_counts = external_degree_counts(graph, &cluster.node_ids, left_id);
                let right_counts = external_degree_counts(graph, &cluster.node_ids, right_id);

                if row_len == 2 {
                    left_counts
                        .1
                        .cmp(&right_counts.1)
                        .then_with(|| left_id.index().cmp(&right_id.index()))
                        .then_with(|| left.cmp(right))
                } else {
                    let left_source_like = left_counts.1 > left_counts.0;
                    let right_source_like = right_counts.1 > right_counts.0;
                    left_source_like
                        .cmp(&right_source_like)
                        .then_with(|| left_id.index().cmp(&right_id.index()))
                        .then_with(|| left.cmp(right))
                }
            });

            for (node_index, slot_x) in node_indices.into_iter().zip(slot_xs.into_iter()) {
                nodes[node_index].center.0 = slot_x;
                nodes[node_index].bounds =
                    Rect::from_center_size(nodes[node_index].center, nodes[node_index].size);
            }
        }
    }
}

fn reorder_top_level_rank_blocks<V, E>(
    nodes: &mut [RoutedNode<NodeIndex>],
    graph: &StableDiGraph<V, E>,
    clusters: &[ClusterInternal],
    direct_child_lookup: &HashMap<(Option<usize>, usize), usize>,
) {
    let ranks = group_nodes_into_ranks(nodes);
    if ranks.len() < 2 {
        return;
    }

    let node_index_by_id = nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.id, index))
        .collect::<HashMap<_, _>>();

    for rank_index in 0..ranks.len() {
        let rank = &ranks[rank_index];
        if rank.len() <= 1 {
            continue;
        }

        let upper_positions = rank_index
            .checked_sub(1)
            .and_then(|index| ranks.get(index))
            .map(|rank| rank_positions(rank));
        let lower_positions = ranks.get(rank_index + 1).map(|rank| rank_positions(rank));

        let mut child_members = HashMap::<usize, Vec<usize>>::new();
        let mut block_sequence = Vec::<RankBlockKind>::new();
        let mut seen_clusters = HashSet::<usize>::new();

        for node_index in rank {
            if let Some(child_cluster) = direct_child_lookup.get(&(None, *node_index)).copied() {
                child_members
                    .entry(child_cluster)
                    .or_default()
                    .push(*node_index);
                if seen_clusters.insert(child_cluster) {
                    block_sequence.push(RankBlockKind::Cluster(child_cluster));
                }
            } else {
                block_sequence.push(RankBlockKind::Node(*node_index));
            }
        }

        let mut blocks = Vec::<RankOrderBlock>::new();
        for kind in block_sequence {
            match kind {
                RankBlockKind::Cluster(cluster_index) => {
                    let members = child_members.remove(&cluster_index).unwrap_or_default();
                    if members.is_empty() {
                        continue;
                    }
                    blocks.push(RankOrderBlock {
                        current_mean: mean_current_position(&members, &rank_positions(rank)),
                        members,
                        tie_order: clusters[cluster_index].declaration_index,
                    });
                }
                RankBlockKind::Node(node_index) => {
                    blocks.push(RankOrderBlock {
                        members: vec![node_index],
                        current_mean: nodes[node_index].center.0,
                        tie_order: clusters.len() + node_index,
                    });
                }
            }
        }

        blocks.sort_by(|left, right| {
            let left_is_cluster = left.tie_order < clusters.len();
            let right_is_cluster = right.tie_order < clusters.len();
            if left_is_cluster && right_is_cluster {
                left.tie_order.cmp(&right.tie_order)
            } else {
                block_adjacent_rank_key(
                    left,
                    &upper_positions,
                    &lower_positions,
                    nodes,
                    graph,
                    &node_index_by_id,
                )
                .total_cmp(&block_adjacent_rank_key(
                    right,
                    &upper_positions,
                    &lower_positions,
                    nodes,
                    graph,
                    &node_index_by_id,
                ))
                .then_with(|| left.current_mean.total_cmp(&right.current_mean))
                .then_with(|| left.tie_order.cmp(&right.tie_order))
            }
        });

        let ordered_nodes = blocks
            .into_iter()
            .flat_map(|block| block.members.into_iter())
            .collect::<Vec<_>>();
        let mut slot_xs = rank
            .iter()
            .map(|node_index| nodes[*node_index].center.0)
            .collect::<Vec<_>>();
        slot_xs.sort_by(|left, right| left.total_cmp(right));
        for (node_index, slot_x) in ordered_nodes.into_iter().zip(slot_xs.into_iter()) {
            nodes[node_index].center.0 = slot_x;
            nodes[node_index].bounds =
                Rect::from_center_size(nodes[node_index].center, nodes[node_index].size);
        }
    }
}

fn block_adjacent_rank_key<V, E>(
    block: &RankOrderBlock,
    upper_positions: &Option<HashMap<usize, f64>>,
    lower_positions: &Option<HashMap<usize, f64>>,
    nodes: &[RoutedNode<NodeIndex>],
    graph: &StableDiGraph<V, E>,
    node_index_by_id: &HashMap<NodeIndex, usize>,
) -> f64 {
    let mut positions = Vec::new();

    for node_index in &block.members {
        let node_id = nodes[*node_index].id;
        for neighbor in graph.neighbors_directed(node_id, Incoming) {
            let Some(neighbor_index) = node_index_by_id.get(&neighbor).copied() else {
                continue;
            };
            if let Some(position) = upper_positions
                .as_ref()
                .and_then(|positions| positions.get(&neighbor_index))
                .copied()
            {
                positions.push(position);
            }
        }

        for neighbor in graph.neighbors_directed(node_id, Outgoing) {
            let Some(neighbor_index) = node_index_by_id.get(&neighbor).copied() else {
                continue;
            };
            if let Some(position) = lower_positions
                .as_ref()
                .and_then(|positions| positions.get(&neighbor_index))
                .copied()
            {
                positions.push(position);
            }
        }
    }

    if positions.is_empty() {
        return block.current_mean;
    }

    positions.sort_by(|left, right| left.total_cmp(right));
    let middle = positions.len() / 2;
    if positions.len() % 2 == 0 {
        (positions[middle - 1] + positions[middle]) * 0.5
    } else {
        positions[middle]
    }
}

fn build_cluster_internals<C>(
    nodes: &[RoutedNode<NodeIndex>],
    clusters: &[ClusterSpec<C>],
    default_cluster_padding: f64,
    node_index_by_id: &HashMap<NodeIndex, usize>,
) -> Vec<ClusterInternal> {
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
                declaration_index: cluster_index,
                node_ids,
                node_indices,
                padding: cluster.padding.unwrap_or(default_cluster_padding),
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
    for cluster in &mut cluster_internals {
        cluster.node_indices.sort_unstable();
        cluster.node_indices.dedup();
    }
    cluster_internals
}

#[derive(Clone, Copy)]
enum SweepDirection {
    Down,
    Up,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum RankBlockKind {
    Cluster(usize),
    Node(usize),
}

#[derive(Clone)]
struct RankOrderBlock {
    members: Vec<usize>,
    current_mean: f64,
    tie_order: usize,
}

#[derive(Clone)]
struct ContainerRankLayout {
    node_ranks: HashMap<usize, i32>,
    span: i32,
}

#[derive(Clone, Copy)]
struct ItemRankConstraint {
    source_item: RankBlockKind,
    target_item: RankBlockKind,
    source_local_rank: i32,
    target_local_rank: i32,
    delta: i32,
}

fn build_compound_ranks<V, E>(
    nodes: &[RoutedNode<NodeIndex>],
    graph: &StableDiGraph<V, E>,
    clusters: &[ClusterInternal],
    direct_child_lookup: &HashMap<(Option<usize>, usize), usize>,
    original_x: &[f64],
    minimum_length: i32,
) -> Vec<Vec<usize>> {
    if nodes.is_empty() {
        return Vec::new();
    }

    let node_index_by_id = nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.id, index))
        .collect::<HashMap<_, _>>();
    let children_by_parent = build_children_by_parent(clusters);
    let component_nodes = (0..nodes.len()).collect::<Vec<_>>();
    let container_layout = compute_container_rank_layout(
        None,
        &component_nodes,
        nodes,
        graph,
        clusters,
        &children_by_parent,
        direct_child_lookup,
        &node_index_by_id,
        original_x,
        minimum_length,
    );
    let mut node_ranks = container_layout.node_ranks;
    apply_sibling_cluster_rank_stagger(&mut node_ranks, graph, clusters, &node_index_by_id);

    let mut ranks = BTreeMap::<i32, Vec<usize>>::new();
    for node_index in 0..nodes.len() {
        let rank = node_ranks.get(&node_index).copied().unwrap_or(0);
        ranks.entry(rank).or_default().push(node_index);
    }

    ranks
        .into_values()
        .map(|mut rank| {
            rank.sort_by(|left, right| {
                original_x[*left]
                    .total_cmp(&original_x[*right])
                    .then_with(|| left.cmp(right))
            });
            rank
        })
        .collect()
}

fn apply_sibling_cluster_rank_stagger<V, E>(
    node_ranks: &mut HashMap<usize, i32>,
    graph: &StableDiGraph<V, E>,
    clusters: &[ClusterInternal],
    node_index_by_id: &HashMap<NodeIndex, usize>,
) {
    let mut sibling_groups = HashMap::<Option<usize>, Vec<usize>>::new();
    for (cluster_index, cluster) in clusters.iter().enumerate() {
        sibling_groups
            .entry(cluster.parent)
            .or_default()
            .push(cluster_index);
    }

    let mut groups = sibling_groups
        .into_values()
        .filter(|group| group.len() > 1)
        .collect::<Vec<_>>();
    groups.sort_by_key(|group| group[0]);

    for sibling_group in groups {
        let mut ordered_group = sibling_group;
        ordered_group.sort_by_key(|cluster_index| clusters[*cluster_index].declaration_index);

        let group_base = ordered_group
            .iter()
            .flat_map(|cluster_index| clusters[*cluster_index].node_indices.iter().copied())
            .filter_map(|node_index| node_ranks.get(&node_index).copied())
            .min()
            .unwrap_or(0);

        for (offset, cluster_index) in ordered_group.into_iter().enumerate() {
            let local_ranks =
                cluster_internal_rank_template(cluster_index, graph, clusters, node_index_by_id);
            let cluster_base = group_base + offset as i32;
            for node_index in &clusters[cluster_index].node_indices {
                let local_rank = local_ranks.get(node_index).copied().unwrap_or(0);
                node_ranks.insert(*node_index, cluster_base + local_rank);
            }
        }
    }
}

fn cluster_internal_rank_template<V, E>(
    cluster_index: usize,
    graph: &StableDiGraph<V, E>,
    clusters: &[ClusterInternal],
    node_index_by_id: &HashMap<NodeIndex, usize>,
) -> HashMap<usize, i32> {
    let cluster = &clusters[cluster_index];
    let cluster_node_ids = cluster.node_ids.clone();
    let topo = petgraph::algo::toposort(graph, None).unwrap_or_default();
    let mut local_ranks = HashMap::<usize, i32>::new();

    for node_id in topo {
        if !cluster_node_ids.contains(&node_id) {
            continue;
        }
        let Some(node_index) = node_index_by_id.get(&node_id).copied() else {
            continue;
        };
        let rank = graph
            .neighbors_directed(node_id, Incoming)
            .filter(|pred| cluster_node_ids.contains(pred))
            .filter_map(|pred| node_index_by_id.get(&pred).copied())
            .map(|pred_index| local_ranks.get(&pred_index).copied().unwrap_or(0) + 1)
            .max()
            .unwrap_or(0);
        local_ranks.insert(node_index, rank);
    }

    local_ranks
}

fn build_children_by_parent(clusters: &[ClusterInternal]) -> HashMap<Option<usize>, Vec<usize>> {
    let mut children_by_parent = HashMap::<Option<usize>, Vec<usize>>::new();
    for (cluster_index, cluster) in clusters.iter().enumerate() {
        children_by_parent
            .entry(cluster.parent)
            .or_default()
            .push(cluster_index);
    }

    for child_clusters in children_by_parent.values_mut() {
        child_clusters.sort_by(|left, right| {
            clusters[*left]
                .declaration_index
                .cmp(&clusters[*right].declaration_index)
        });
    }

    children_by_parent
}

fn compute_container_rank_layout<V, E>(
    parent: Option<usize>,
    container_nodes: &[usize],
    nodes: &[RoutedNode<NodeIndex>],
    graph: &StableDiGraph<V, E>,
    clusters: &[ClusterInternal],
    children_by_parent: &HashMap<Option<usize>, Vec<usize>>,
    direct_child_lookup: &HashMap<(Option<usize>, usize), usize>,
    node_index_by_id: &HashMap<NodeIndex, usize>,
    original_x: &[f64],
    minimum_length: i32,
) -> ContainerRankLayout {
    let child_clusters = children_by_parent
        .get(&parent)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter(|cluster_index| !clusters[*cluster_index].node_indices.is_empty())
        .collect::<Vec<_>>();

    let child_layouts = child_clusters
        .iter()
        .copied()
        .map(|cluster_index| {
            (
                cluster_index,
                compute_container_rank_layout(
                    Some(cluster_index),
                    &clusters[cluster_index].node_indices,
                    nodes,
                    graph,
                    clusters,
                    children_by_parent,
                    direct_child_lookup,
                    node_index_by_id,
                    original_x,
                    minimum_length,
                ),
            )
        })
        .collect::<HashMap<_, _>>();

    let direct_nodes = container_nodes
        .iter()
        .copied()
        .filter(|node_index| direct_child_lookup.get(&(parent, *node_index)).is_none())
        .collect::<Vec<_>>();

    let mut items = child_clusters
        .iter()
        .copied()
        .map(RankBlockKind::Cluster)
        .chain(direct_nodes.iter().copied().map(RankBlockKind::Node))
        .collect::<Vec<_>>();

    items.sort_by(|left, right| {
        initial_item_x(left, clusters, original_x)
            .total_cmp(&initial_item_x(right, clusters, original_x))
            .then_with(|| {
                initial_item_order(left, clusters).cmp(&initial_item_order(right, clusters))
            })
    });

    if items.is_empty() {
        return ContainerRankLayout {
            node_ranks: HashMap::new(),
            span: 0,
        };
    }

    let container_node_set = container_nodes.iter().copied().collect::<HashSet<_>>();
    let mut edge_constraints = Vec::<ItemRankConstraint>::new();
    for edge in graph.edge_references() {
        let Some(source_index) = node_index_by_id.get(&edge.source()).copied() else {
            continue;
        };
        let Some(target_index) = node_index_by_id.get(&edge.target()).copied() else {
            continue;
        };
        if !(container_node_set.contains(&source_index)
            && container_node_set.contains(&target_index))
        {
            continue;
        }

        let source_item = direct_child_lookup
            .get(&(parent, source_index))
            .copied()
            .map(RankBlockKind::Cluster)
            .unwrap_or(RankBlockKind::Node(source_index));
        let target_item = direct_child_lookup
            .get(&(parent, target_index))
            .copied()
            .map(RankBlockKind::Cluster)
            .unwrap_or(RankBlockKind::Node(target_index));
        if source_item == target_item {
            continue;
        }

        let source_offset = local_node_rank(source_item, source_index, &child_layouts);
        let target_offset = local_node_rank(target_item, target_index, &child_layouts);
        edge_constraints.push(ItemRankConstraint {
            source_item,
            target_item,
            source_local_rank: source_offset,
            target_local_rank: target_offset,
            delta: cross_item_rank_delta(source_item, target_item, minimum_length),
        });
    }

    let item_bases = solve_item_ranks(&items, &edge_constraints, clusters);
    let mut node_ranks = HashMap::new();
    let mut max_rank = 0;

    for item in items {
        let base_rank = item_bases.get(&item).copied().unwrap_or(0);
        match item {
            RankBlockKind::Cluster(cluster_index) => {
                if let Some(child_layout) = child_layouts.get(&cluster_index) {
                    max_rank = max_rank.max(base_rank + child_layout.span.saturating_sub(1));
                    for (node_index, local_rank) in &child_layout.node_ranks {
                        node_ranks.insert(*node_index, base_rank + *local_rank);
                    }
                }
            }
            RankBlockKind::Node(node_index) => {
                max_rank = max_rank.max(base_rank);
                node_ranks.insert(node_index, base_rank);
            }
        }
    }

    ContainerRankLayout {
        node_ranks,
        span: max_rank + 1,
    }
}

fn initial_item_x(item: &RankBlockKind, clusters: &[ClusterInternal], original_x: &[f64]) -> f64 {
    match item {
        RankBlockKind::Cluster(cluster_index) => {
            let cluster = &clusters[*cluster_index];
            let total = cluster
                .node_indices
                .iter()
                .map(|node_index| original_x[*node_index])
                .sum::<f64>();
            total / cluster.node_indices.len().max(1) as f64
        }
        RankBlockKind::Node(node_index) => original_x[*node_index],
    }
}

fn initial_item_order(item: &RankBlockKind, clusters: &[ClusterInternal]) -> usize {
    match item {
        RankBlockKind::Cluster(cluster_index) => clusters[*cluster_index].declaration_index,
        RankBlockKind::Node(node_index) => clusters.len() + *node_index,
    }
}

fn local_node_rank(
    item: RankBlockKind,
    node_index: usize,
    child_layouts: &HashMap<usize, ContainerRankLayout>,
) -> i32 {
    match item {
        RankBlockKind::Cluster(cluster_index) => child_layouts
            .get(&cluster_index)
            .and_then(|layout| layout.node_ranks.get(&node_index).copied())
            .unwrap_or(0),
        RankBlockKind::Node(_) => 0,
    }
}

fn solve_item_ranks(
    items: &[RankBlockKind],
    edge_constraints: &[ItemRankConstraint],
    clusters: &[ClusterInternal],
) -> HashMap<RankBlockKind, i32> {
    let has_direct_nodes = items
        .iter()
        .any(|item| matches!(item, RankBlockKind::Node(_)));
    let mut ranks = items
        .iter()
        .copied()
        .map(|item| {
            let seed = match item {
                RankBlockKind::Cluster(_) if has_direct_nodes => 1,
                _ => 0,
            };
            (item, seed)
        })
        .collect::<HashMap<_, _>>();

    for _ in 0..items.len().max(1) * 6 {
        let previous = ranks.clone();
        for item in items {
            let mut suggestions = edge_constraints
                .iter()
                .filter_map(|constraint| {
                    if constraint.target_item == *item {
                        let source_base =
                            previous.get(&constraint.source_item).copied().unwrap_or(0);
                        Some(
                            source_base + constraint.source_local_rank + constraint.delta
                                - constraint.target_local_rank,
                        )
                    } else if constraint.source_item == *item {
                        let target_base =
                            previous.get(&constraint.target_item).copied().unwrap_or(0);
                        Some(
                            target_base + constraint.target_local_rank
                                - constraint.delta
                                - constraint.source_local_rank,
                        )
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();

            if suggestions.is_empty() {
                continue;
            }
            suggestions.sort_unstable();
            let middle = suggestions.len() / 2;
            let mut next = if suggestions.len() % 2 == 0 {
                ((suggestions[middle - 1] + suggestions[middle]) as f64 * 0.5).round() as i32
            } else {
                suggestions[middle]
            };
            if matches!(item, RankBlockKind::Cluster(_)) && has_direct_nodes {
                next = next.max(1);
            }
            ranks.insert(*item, next);
        }
    }

    let min_rank = ranks.values().copied().min().unwrap_or(0);
    for item in items {
        if let Some(rank) = ranks.get_mut(item) {
            *rank -= min_rank;
        } else {
            ranks.insert(*item, initial_item_order(item, clusters) as i32);
        }
    }

    ranks
}

fn cross_item_rank_delta(
    source_item: RankBlockKind,
    target_item: RankBlockKind,
    minimum_length: i32,
) -> i32 {
    match (source_item, target_item) {
        (RankBlockKind::Cluster(_), RankBlockKind::Cluster(_)) => 0,
        _ => minimum_length.max(1),
    }
}

fn assign_rank_y_positions(
    ranks: &[Vec<usize>],
    nodes: &mut [RoutedNode<NodeIndex>],
    vertex_spacing: f64,
) {
    if ranks.is_empty() {
        return;
    }

    let rank_gap = vertex_spacing.max(12.0);
    let mut current_top = 0.0;
    for rank in ranks {
        if rank.is_empty() {
            continue;
        }

        let rank_height = rank
            .iter()
            .map(|node_index| nodes[*node_index].size.1)
            .max_by(|left, right| left.total_cmp(right))
            .unwrap_or(0.0);
        let center_y = current_top + rank_height * 0.5;
        for node_index in rank {
            nodes[*node_index].center.1 = center_y;
            nodes[*node_index].bounds =
                Rect::from_center_size(nodes[*node_index].center, nodes[*node_index].size);
        }
        current_top += rank_height + rank_gap;
    }
}

fn build_cluster_depths(clusters: &[ClusterInternal]) -> Vec<usize> {
    fn depth_for(
        cluster_index: usize,
        clusters: &[ClusterInternal],
        cache: &mut [Option<usize>],
    ) -> usize {
        if let Some(depth) = cache[cluster_index] {
            return depth;
        }
        let depth = clusters[cluster_index]
            .parent
            .map(|parent| depth_for(parent, clusters, cache) + 1)
            .unwrap_or(0);
        cache[cluster_index] = Some(depth);
        depth
    }

    let mut cache = vec![None; clusters.len()];
    (0..clusters.len())
        .map(|cluster_index| depth_for(cluster_index, clusters, &mut cache))
        .collect()
}

fn build_direct_child_lookup(
    node_cluster_memberships: &[Vec<usize>],
    cluster_depths: &[usize],
) -> HashMap<(Option<usize>, usize), usize> {
    let mut lookup = HashMap::new();

    for (node_index, memberships) in node_cluster_memberships.iter().enumerate() {
        let mut sorted = memberships.clone();
        sorted.sort_by(|left, right| {
            cluster_depths[*left]
                .cmp(&cluster_depths[*right])
                .then_with(|| left.cmp(right))
        });

        let mut parent = None;
        for cluster_index in sorted {
            lookup.insert((parent, node_index), cluster_index);
            parent = Some(cluster_index);
        }
    }

    lookup
}

fn sweep_cluster_ordering<V, E>(
    ranks: &mut [Vec<usize>],
    nodes: &[RoutedNode<NodeIndex>],
    graph: &StableDiGraph<V, E>,
    clusters: &[ClusterInternal],
    direct_child_lookup: &HashMap<(Option<usize>, usize), usize>,
    direction: SweepDirection,
) {
    if ranks.len() < 2 {
        return;
    }

    let node_index_by_id = nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.id, index))
        .collect::<HashMap<_, _>>();

    match direction {
        SweepDirection::Down => {
            for rank_index in 1..ranks.len() {
                let other_positions = rank_positions(&ranks[rank_index - 1]);
                let current_positions = rank_positions(&ranks[rank_index]);
                ranks[rank_index] = reorder_parent_rank(
                    None,
                    &ranks[rank_index],
                    &other_positions,
                    &current_positions,
                    nodes,
                    graph,
                    clusters,
                    direct_child_lookup,
                    &node_index_by_id,
                );
            }
        }
        SweepDirection::Up => {
            for rank_index in (0..ranks.len() - 1).rev() {
                let other_positions = rank_positions(&ranks[rank_index + 1]);
                let current_positions = rank_positions(&ranks[rank_index]);
                ranks[rank_index] = reorder_parent_rank(
                    None,
                    &ranks[rank_index],
                    &other_positions,
                    &current_positions,
                    nodes,
                    graph,
                    clusters,
                    direct_child_lookup,
                    &node_index_by_id,
                );
            }
        }
    }
}

fn rank_positions(rank: &[usize]) -> HashMap<usize, f64> {
    rank.iter()
        .enumerate()
        .map(|(position, node_index)| (*node_index, position as f64))
        .collect()
}

fn refine_rank_order_by_adjacent_medians<V, E>(
    ranks: &mut [Vec<usize>],
    nodes: &[RoutedNode<NodeIndex>],
    graph: &StableDiGraph<V, E>,
    original_x: &[f64],
) {
    if ranks.len() < 2 {
        return;
    }

    let node_index_by_id = nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.id, index))
        .collect::<HashMap<_, _>>();

    for rank_index in 0..ranks.len() {
        if ranks[rank_index].len() <= 1 {
            continue;
        }

        let upper_positions = rank_index
            .checked_sub(1)
            .and_then(|index| ranks.get(index))
            .map(|rank| rank_positions(rank));
        let lower_positions = ranks.get(rank_index + 1).map(|rank| rank_positions(rank));

        ranks[rank_index].sort_by(|left, right| {
            adjacent_rank_median_key(
                *left,
                &upper_positions,
                &lower_positions,
                nodes,
                graph,
                &node_index_by_id,
                original_x,
            )
            .total_cmp(&adjacent_rank_median_key(
                *right,
                &upper_positions,
                &lower_positions,
                nodes,
                graph,
                &node_index_by_id,
                original_x,
            ))
            .then_with(|| {
                original_x[*left]
                    .total_cmp(&original_x[*right])
                    .then_with(|| left.cmp(right))
            })
        });
    }
}

fn adjacent_rank_median_key<V, E>(
    node_index: usize,
    upper_positions: &Option<HashMap<usize, f64>>,
    lower_positions: &Option<HashMap<usize, f64>>,
    nodes: &[RoutedNode<NodeIndex>],
    graph: &StableDiGraph<V, E>,
    node_index_by_id: &HashMap<NodeIndex, usize>,
    original_x: &[f64],
) -> f64 {
    let node_id = nodes[node_index].id;
    let mut positions = Vec::new();

    for neighbor in graph.neighbors_directed(node_id, Incoming) {
        let Some(neighbor_index) = node_index_by_id.get(&neighbor).copied() else {
            continue;
        };
        if let Some(position) = upper_positions
            .as_ref()
            .and_then(|positions| positions.get(&neighbor_index))
            .copied()
        {
            positions.push(position);
        }
    }

    for neighbor in graph.neighbors_directed(node_id, Outgoing) {
        let Some(neighbor_index) = node_index_by_id.get(&neighbor).copied() else {
            continue;
        };
        if let Some(position) = lower_positions
            .as_ref()
            .and_then(|positions| positions.get(&neighbor_index))
            .copied()
        {
            positions.push(position);
        }
    }

    if positions.is_empty() {
        return *original_x
            .get(node_index)
            .unwrap_or(&nodes[node_index].center.0);
    }

    positions.sort_by(|left, right| left.total_cmp(right));
    let middle = positions.len() / 2;
    if positions.len() % 2 == 0 {
        (positions[middle - 1] + positions[middle]) * 0.5
    } else {
        positions[middle]
    }
}

fn reorder_parent_rank<V, E>(
    parent: Option<usize>,
    rank_nodes: &[usize],
    other_positions: &HashMap<usize, f64>,
    current_positions: &HashMap<usize, f64>,
    nodes: &[RoutedNode<NodeIndex>],
    graph: &StableDiGraph<V, E>,
    clusters: &[ClusterInternal],
    direct_child_lookup: &HashMap<(Option<usize>, usize), usize>,
    node_index_by_id: &HashMap<NodeIndex, usize>,
) -> Vec<usize> {
    if rank_nodes.len() <= 1 {
        return rank_nodes.to_vec();
    }

    let mut child_members = HashMap::<usize, Vec<usize>>::new();
    let mut block_sequence = Vec::<RankBlockKind>::new();
    let mut seen_clusters = HashSet::<usize>::new();

    for node_index in rank_nodes {
        if let Some(child_cluster) = direct_child_lookup.get(&(parent, *node_index)).copied() {
            child_members
                .entry(child_cluster)
                .or_default()
                .push(*node_index);
            if seen_clusters.insert(child_cluster) {
                block_sequence.push(RankBlockKind::Cluster(child_cluster));
            }
        } else {
            block_sequence.push(RankBlockKind::Node(*node_index));
        }
    }

    let mut blocks = Vec::<RankOrderBlock>::new();
    for kind in block_sequence {
        match kind {
            RankBlockKind::Cluster(cluster_index) => {
                let members = child_members
                    .remove(&cluster_index)
                    .map(|members| {
                        reorder_parent_rank(
                            Some(cluster_index),
                            &members,
                            other_positions,
                            current_positions,
                            nodes,
                            graph,
                            clusters,
                            direct_child_lookup,
                            node_index_by_id,
                        )
                    })
                    .unwrap_or_default();
                if members.is_empty() {
                    continue;
                }
                let current_mean = mean_current_position(&members, current_positions);
                blocks.push(RankOrderBlock {
                    members,
                    current_mean,
                    tie_order: clusters[cluster_index].declaration_index,
                });
            }
            RankBlockKind::Node(node_index) => {
                let current_mean = *current_positions.get(&node_index).unwrap_or(&0.0);
                blocks.push(RankOrderBlock {
                    members: vec![node_index],
                    current_mean,
                    tie_order: current_mean as usize + clusters.len(),
                });
            }
        }
    }

    blocks.sort_by(|left, right| {
        ordering_key(left, other_positions, nodes, graph, node_index_by_id)
            .total_cmp(&ordering_key(
                right,
                other_positions,
                nodes,
                graph,
                node_index_by_id,
            ))
            .then_with(|| left.current_mean.total_cmp(&right.current_mean))
            .then_with(|| left.tie_order.cmp(&right.tie_order))
    });

    blocks
        .into_iter()
        .flat_map(|block| block.members.into_iter())
        .collect()
}

fn mean_current_position(members: &[usize], current_positions: &HashMap<usize, f64>) -> f64 {
    let (total, count) = members
        .iter()
        .fold((0.0, 0usize), |(total, count), node_index| {
            (
                total + current_positions.get(node_index).copied().unwrap_or(0.0),
                count + 1,
            )
        });

    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}

fn ordering_key<V, E>(
    block: &RankOrderBlock,
    other_positions: &HashMap<usize, f64>,
    nodes: &[RoutedNode<NodeIndex>],
    graph: &StableDiGraph<V, E>,
    node_index_by_id: &HashMap<NodeIndex, usize>,
) -> f64 {
    let mut positions = Vec::new();
    for node_index in &block.members {
        let node_id = nodes[*node_index].id;
        for neighbor in graph.neighbors_directed(node_id, Incoming) {
            if let Some(neighbor_index) = node_index_by_id.get(&neighbor).copied() {
                if let Some(position) = other_positions.get(&neighbor_index).copied() {
                    positions.push(position);
                }
            }
        }
        for neighbor in graph.neighbors_directed(node_id, Outgoing) {
            if let Some(neighbor_index) = node_index_by_id.get(&neighbor).copied() {
                if let Some(position) = other_positions.get(&neighbor_index).copied() {
                    positions.push(position);
                }
            }
        }
    }

    if positions.is_empty() {
        return block.current_mean;
    }

    positions.sort_by(|left, right| left.total_cmp(right));
    let middle = positions.len() / 2;
    if positions.len() % 2 == 0 {
        (positions[middle - 1] + positions[middle]) * 0.5
    } else {
        positions[middle]
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

fn build_sibling_cluster_groups(
    clusters: &[ClusterInternal],
    original_x: &[f64],
) -> Vec<Vec<usize>> {
    let mut grouped = HashMap::<Option<usize>, Vec<usize>>::new();
    for (cluster_index, cluster) in clusters.iter().enumerate() {
        grouped
            .entry(cluster.parent)
            .or_default()
            .push(cluster_index);
    }

    let mut groups = grouped
        .into_values()
        .filter(|group| group.len() > 1)
        .collect::<Vec<_>>();
    for group in &mut groups {
        group.sort_by(|left, right| {
            cluster_mean_original_x(&clusters[*left], original_x)
                .total_cmp(&cluster_mean_original_x(&clusters[*right], original_x))
                .then_with(|| {
                    clusters[*left]
                        .declaration_index
                        .cmp(&clusters[*right].declaration_index)
                })
        });
    }
    groups.sort_by(|left, right| left[0].cmp(&right[0]));
    groups
}

fn cluster_mean_original_x(cluster: &ClusterInternal, original_x: &[f64]) -> f64 {
    let total = cluster
        .node_indices
        .iter()
        .map(|node_index| original_x[*node_index])
        .sum::<f64>();
    total / cluster.node_indices.len().max(1) as f64
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

fn enforce_sibling_cluster_order(
    rank_nodes: &mut [usize],
    node_cluster_memberships: &[Vec<usize>],
    sibling_group: &[usize],
) {
    if sibling_group.len() < 2 {
        return;
    }

    let sibling_order = sibling_group
        .iter()
        .enumerate()
        .map(|(position, cluster_index)| (*cluster_index, position))
        .collect::<HashMap<_, _>>();

    let mut grouped_nodes = rank_nodes
        .iter()
        .copied()
        .filter_map(|node_index| {
            node_cluster_memberships
                .get(node_index)
                .and_then(|memberships| {
                    memberships.iter().find_map(|cluster_index| {
                        sibling_order
                            .get(cluster_index)
                            .copied()
                            .map(|order| (order, node_index))
                    })
                })
        })
        .collect::<Vec<_>>();

    if grouped_nodes.len() < 2 {
        return;
    }

    grouped_nodes.sort_by_key(|(order, _)| *order);
    let ordered_nodes = grouped_nodes
        .into_iter()
        .map(|(_, node_index)| node_index)
        .collect::<Vec<_>>();

    let mut replacement = ordered_nodes.into_iter();
    for node_index in rank_nodes.iter_mut() {
        let should_replace = node_cluster_memberships
            .get(*node_index)
            .map(|memberships| {
                memberships
                    .iter()
                    .any(|cluster_index| sibling_order.contains_key(cluster_index))
            })
            .unwrap_or(false);
        if should_replace {
            if let Some(next_node) = replacement.next() {
                *node_index = next_node;
            }
        }
    }
}

fn assign_rank_x_positions<V, E>(
    ranks: &[Vec<usize>],
    nodes: &mut [RoutedNode<NodeIndex>],
    original_x: &[f64],
    node_cluster_memberships: &[Vec<usize>],
    vertex_spacing: f64,
    cluster_boundary_gap: f64,
    graph: &StableDiGraph<V, E>,
) {
    let node_index_by_id = nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.id, index))
        .collect::<HashMap<_, _>>();

    for (rank_index, rank) in ranks.iter().enumerate() {
        if rank.is_empty() {
            continue;
        }

        let upper_rank = rank_index.checked_sub(1).and_then(|index| ranks.get(index));
        let lower_rank = ranks.get(rank_index + 1);
        let desired_positions = rank
            .iter()
            .copied()
            .map(|node_index| {
                adjacent_rank_target_x(
                    node_index,
                    upper_rank,
                    lower_rank,
                    nodes,
                    graph,
                    &node_index_by_id,
                    original_x,
                    node_cluster_memberships,
                )
            })
            .collect::<Vec<_>>();

        for (position, node_index) in rank.iter().copied().enumerate() {
            let target_x = match position {
                0 => desired_positions.get(position).copied().unwrap_or_else(|| {
                    *original_x
                        .get(node_index)
                        .unwrap_or(&nodes[node_index].center.0)
                }),
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
                    desired_positions
                        .get(position)
                        .copied()
                        .unwrap_or_else(|| {
                            *original_x
                                .get(node_index)
                                .unwrap_or(&nodes[node_index].center.0)
                        })
                        .max(min_x)
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

        if !desired_positions.is_empty() {
            let desired_center =
                desired_positions.iter().sum::<f64>() / desired_positions.len() as f64;
            let actual_center = rank
                .iter()
                .map(|node_index| nodes[*node_index].center.0)
                .sum::<f64>()
                / rank.len() as f64;
            let shift = desired_center - actual_center;
            if shift.abs() > EPSILON {
                for node_index in rank {
                    nodes[*node_index].center.0 += shift;
                }
            }
        }
    }
}

fn adjacent_rank_target_x<V, E>(
    node_index: usize,
    upper_rank: Option<&Vec<usize>>,
    lower_rank: Option<&Vec<usize>>,
    nodes: &[RoutedNode<NodeIndex>],
    graph: &StableDiGraph<V, E>,
    node_index_by_id: &HashMap<NodeIndex, usize>,
    original_x: &[f64],
    node_cluster_memberships: &[Vec<usize>],
) -> f64 {
    let node_id = nodes[node_index].id;
    let current_memberships = node_cluster_memberships
        .get(node_index)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let upper_rank_nodes = upper_rank
        .map(|rank| rank.iter().copied().collect::<HashSet<_>>())
        .unwrap_or_default();
    let lower_rank_nodes = lower_rank
        .map(|rank| rank.iter().copied().collect::<HashSet<_>>())
        .unwrap_or_default();
    let incoming_neighbors = graph
        .neighbors_directed(node_id, Incoming)
        .filter_map(|neighbor| node_index_by_id.get(&neighbor).copied())
        .filter(|neighbor_index| upper_rank_nodes.contains(neighbor_index))
        .collect::<Vec<_>>();
    let outgoing_neighbors = graph
        .neighbors_directed(node_id, Outgoing)
        .filter_map(|neighbor| node_index_by_id.get(&neighbor).copied())
        .filter(|neighbor_index| lower_rank_nodes.contains(neighbor_index))
        .collect::<Vec<_>>();

    let fallback_x = *original_x
        .get(node_index)
        .unwrap_or(&nodes[node_index].center.0);
    let (overlap, mut neighbor_xs) = preferred_neighbor_xs(
        &incoming_neighbors,
        current_memberships,
        nodes,
        node_cluster_memberships,
    )
    .or_else(|| {
        preferred_neighbor_xs(
            &outgoing_neighbors,
            current_memberships,
            nodes,
            node_cluster_memberships,
        )
    })
    .or_else(|| {
        (!incoming_neighbors.is_empty()).then(|| {
            (
                0,
                incoming_neighbors
                    .iter()
                    .map(|neighbor_index| nodes[*neighbor_index].center.0)
                    .collect::<Vec<_>>(),
            )
        })
    })
    .or_else(|| {
        (!outgoing_neighbors.is_empty()).then(|| {
            (
                0,
                outgoing_neighbors
                    .iter()
                    .map(|neighbor_index| nodes[*neighbor_index].center.0)
                    .collect::<Vec<_>>(),
            )
        })
    })
    .unwrap_or_default();

    if neighbor_xs.is_empty() {
        return *original_x
            .get(node_index)
            .unwrap_or(&nodes[node_index].center.0);
    }

    neighbor_xs.sort_by(|left, right| left.total_cmp(right));
    let middle = neighbor_xs.len() / 2;
    let target_x = if neighbor_xs.len() % 2 == 0 {
        (neighbor_xs[middle - 1] + neighbor_xs[middle]) * 0.5
    } else {
        neighbor_xs[middle]
    };

    if current_memberships.len() > 1 && overlap < current_memberships.len() {
        let trust = (overlap as f64 / current_memberships.len() as f64).clamp(0.0, 1.0);
        let neighbor_weight = if overlap == 0 {
            0.2
        } else {
            0.25 + trust * 0.5
        };
        fallback_x * (1.0 - neighbor_weight) + target_x * neighbor_weight
    } else {
        target_x
    }
}

fn preferred_neighbor_xs(
    neighbor_indices: &[usize],
    current_memberships: &[usize],
    nodes: &[RoutedNode<NodeIndex>],
    node_cluster_memberships: &[Vec<usize>],
) -> Option<(usize, Vec<f64>)> {
    let mut best_overlap = 0usize;
    let mut best_neighbors = Vec::new();

    for neighbor_index in neighbor_indices {
        let neighbor_memberships = node_cluster_memberships
            .get(*neighbor_index)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let overlap = cluster_membership_overlap(current_memberships, neighbor_memberships);
        if overlap == 0 {
            continue;
        }
        if overlap > best_overlap {
            best_overlap = overlap;
            best_neighbors.clear();
        }
        if overlap == best_overlap {
            best_neighbors.push(nodes[*neighbor_index].center.0);
        }
    }

    (!best_neighbors.is_empty()).then_some((best_overlap, best_neighbors))
}

fn cluster_membership_overlap(left_ids: &[usize], right_ids: &[usize]) -> usize {
    let mut i = 0;
    let mut j = 0;
    let mut overlap = 0;

    while i < left_ids.len() && j < right_ids.len() {
        match left_ids[i].cmp(&right_ids[j]) {
            Ordering::Equal => {
                overlap += 1;
                i += 1;
                j += 1;
            }
            Ordering::Less => i += 1,
            Ordering::Greater => j += 1,
        }
    }

    overlap
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

fn enforce_sibling_cluster_positions(
    nodes: &mut [RoutedNode<NodeIndex>],
    clusters: &[ClusterInternal],
    sibling_cluster_groups: &[Vec<usize>],
    cluster_boundary_gap: f64,
    iterations: usize,
) {
    if nodes.is_empty() || sibling_cluster_groups.is_empty() || iterations == 0 {
        return;
    }

    let boundary_gap = cluster_boundary_gap.max(0.0);
    let desired_gap = boundary_gap.max(8.0);
    for _ in 0..iterations {
        let mut moved = false;

        for sibling_group in sibling_cluster_groups {
            for sibling_pair in sibling_group.windows(2) {
                let [left_index, right_index] = match sibling_pair {
                    [left_index, right_index] => [*left_index, *right_index],
                    _ => continue,
                };

                let (Some(left_bounds), Some(right_bounds)) = (
                    cluster_bounds(&clusters[left_index], nodes),
                    cluster_bounds(&clusters[right_index], nodes),
                ) else {
                    continue;
                };
                let vertical_overlap = left_bounds.max_y.min(right_bounds.max_y)
                    - left_bounds.min_y.max(right_bounds.min_y);
                if vertical_overlap <= EPSILON {
                    continue;
                }

                let current_gap = right_bounds.min_x - left_bounds.max_x;
                if (current_gap - desired_gap).abs() <= EPSILON {
                    continue;
                }

                shift_cluster_nodes_x(
                    nodes,
                    &clusters[right_index].node_indices,
                    desired_gap - current_gap,
                );
                moved = true;
            }
        }

        if !moved {
            break;
        }
    }
}

fn center_extreme_single_node_ranks(ranks: &[Vec<usize>], nodes: &mut [RoutedNode<NodeIndex>]) {
    if ranks.len() < 2 {
        return;
    }

    center_single_rank_from_neighbor(ranks.first(), ranks.get(1), nodes);
}

fn center_top_singleton_from_successors<V, E>(
    nodes: &mut [RoutedNode<NodeIndex>],
    graph: &StableDiGraph<V, E>,
) {
    let ranks = group_nodes_into_ranks(nodes);
    let Some(top_rank) = ranks.first() else {
        return;
    };
    if top_rank.len() != 1 {
        return;
    }

    let node_index = top_rank[0];
    let node_id = nodes[node_index].id;
    let node_index_by_id = nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.id, index))
        .collect::<HashMap<_, _>>();
    let successor_xs = graph
        .neighbors_directed(node_id, Outgoing)
        .filter_map(|neighbor| node_index_by_id.get(&neighbor).copied())
        .map(|index| nodes[index].center.0)
        .collect::<Vec<_>>();
    if successor_xs.is_empty() {
        return;
    }

    let target_x = successor_xs.iter().sum::<f64>() / successor_xs.len() as f64;

    nodes[node_index].center.0 = target_x;
    nodes[node_index].bounds =
        Rect::from_center_size(nodes[node_index].center, nodes[node_index].size);
}

fn enforce_labeled_edge_rank_clearance<V, E, L>(
    nodes: &mut [RoutedNode<NodeIndex>],
    graph: &StableDiGraph<V, E>,
    edge_labels: &HashMap<EdgeIndex, Option<L>>,
    render_config: &RenderConfig,
) {
    if nodes.len() < 2 {
        return;
    }

    let ranks = group_nodes_into_ranks(nodes);
    if ranks.len() < 2 {
        return;
    }

    let node_index_by_id = nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.id, index))
        .collect::<HashMap<_, _>>();
    let rank_by_node = ranks
        .iter()
        .enumerate()
        .flat_map(|(rank_index, rank)| {
            rank.iter()
                .copied()
                .map(move |node_index| (node_index, rank_index))
        })
        .collect::<HashMap<_, _>>();

    for edge in graph.edge_references() {
        let Some(tail_index) = node_index_by_id.get(&edge.source()).copied() else {
            continue;
        };
        let Some(head_index) = node_index_by_id.get(&edge.target()).copied() else {
            continue;
        };
        let Some(tail_rank) = rank_by_node.get(&tail_index).copied() else {
            continue;
        };
        let Some(head_rank) = rank_by_node.get(&head_index).copied() else {
            continue;
        };
        if tail_rank == head_rank {
            continue;
        }

        let (upper_index, lower_index, lower_rank) =
            if nodes[tail_index].center.1 <= nodes[head_index].center.1 {
                (tail_index, head_index, head_rank)
            } else {
                (head_index, tail_index, tail_rank)
            };

        let has_label = edge_labels
            .get(&edge.id())
            .map(|label| label.is_some())
            .unwrap_or(false);
        let horizontal_delta = (nodes[upper_index].center.0 - nodes[lower_index].center.0).abs();
        let node_width = nodes[upper_index].size.0.max(nodes[lower_index].size.0);
        let tight_vertical_stack = horizontal_delta <= node_width * 0.45;
        let base_gap = render_config.routing_padding * 2.0 + ROUTE_JOG_CLEARANCE;
        let mut required_gap = base_gap;
        if has_label {
            required_gap = required_gap.max(
                EDGE_LABEL_OBSTACLE_HEIGHT
                    + EDGE_LABEL_OBSTACLE_PADDING * 2.0
                    + if tight_vertical_stack { 28.0 } else { 10.0 },
            );
        }

        let current_gap = nodes[lower_index].bounds.min_y - nodes[upper_index].bounds.max_y;
        if current_gap + EPSILON >= required_gap {
            continue;
        }

        let shift = required_gap - current_gap;
        for rank in ranks.iter().skip(lower_rank) {
            for node_index in rank {
                nodes[*node_index].center.1 += shift;
                nodes[*node_index].bounds =
                    Rect::from_center_size(nodes[*node_index].center, nodes[*node_index].size);
            }
        }
    }
}

fn center_single_rank_from_neighbor(
    rank: Option<&Vec<usize>>,
    neighbor_rank: Option<&Vec<usize>>,
    nodes: &mut [RoutedNode<NodeIndex>],
) {
    let Some(rank) = rank else {
        return;
    };
    if rank.len() != 1 {
        return;
    }

    let Some(neighbor_rank) = neighbor_rank else {
        return;
    };
    if neighbor_rank.is_empty() {
        return;
    }

    let min_x = neighbor_rank
        .iter()
        .map(|node_index| nodes[*node_index].center.0)
        .min_by(|left, right| left.total_cmp(right));
    let max_x = neighbor_rank
        .iter()
        .map(|node_index| nodes[*node_index].center.0)
        .max_by(|left, right| left.total_cmp(right));
    let Some((min_x, max_x)) = min_x.zip(max_x) else {
        return;
    };

    let node_index = rank[0];
    nodes[node_index].center.0 = (min_x + max_x) * 0.5;
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

fn cluster_final_bounds(content_bounds: Rect, padding: f64) -> Rect {
    Rect {
        min_x: content_bounds.min_x - padding,
        max_x: content_bounds.max_x + padding,
        min_y: content_bounds.min_y
            - padding
            - CLUSTER_LABEL_RESERVED_HEIGHT
            - CLUSTER_LABEL_VERTICAL_GAP,
        max_y: content_bounds.max_y + padding,
    }
}

fn separate_nodes_from_foreign_clusters<C>(
    nodes: &mut [RoutedNode<NodeIndex>],
    clusters: &[ComputedCluster<C>],
    gap: f64,
    iterations: usize,
) {
    if nodes.is_empty() || clusters.is_empty() || iterations == 0 {
        return;
    }

    let separation_gap = gap.max(0.0);
    for _ in 0..iterations {
        let mut moved = false;

        for cluster in clusters {
            let separation_bounds = cluster.bounds.expand(separation_gap);
            let cluster_center_x = (separation_bounds.min_x + separation_bounds.max_x) * 0.5;
            let cluster_center_y = (separation_bounds.min_y + separation_bounds.max_y) * 0.5;

            for node in nodes.iter_mut() {
                if cluster.node_ids.contains(&node.id)
                    || !rects_intersect(node.bounds, separation_bounds)
                {
                    continue;
                }

                let overlap_x = node.bounds.max_x.min(separation_bounds.max_x)
                    - node.bounds.min_x.max(separation_bounds.min_x);
                let overlap_y = node.bounds.max_y.min(separation_bounds.max_y)
                    - node.bounds.min_y.max(separation_bounds.min_y);
                if overlap_x <= EPSILON || overlap_y <= EPSILON {
                    continue;
                }

                if node.bounds.min_y < separation_bounds.min_y && node.center.1 <= cluster_center_y
                {
                    let shift = overlap_y;
                    node.center.1 -= shift;
                } else if node.bounds.max_y > separation_bounds.max_y
                    && node.center.1 >= cluster_center_y
                {
                    let shift = overlap_y;
                    node.center.1 += shift;
                } else {
                    let delta_x = node.center.0 - cluster_center_x;
                    let delta_y = node.center.1 - cluster_center_y;
                    if delta_y.abs() >= delta_x.abs() {
                        let shift = overlap_y;
                        if delta_y <= 0.0 {
                            node.center.1 -= shift;
                        } else {
                            node.center.1 += shift;
                        }
                    } else {
                        let shift = overlap_x;
                        if delta_x <= 0.0 {
                            node.center.0 -= shift;
                        } else {
                            node.center.0 += shift;
                        }
                    }
                }
                node.bounds = Rect::from_center_size(node.center, node.size);
                moved = true;
            }
        }

        if !moved {
            break;
        }
    }
}

fn align_sibling_cluster_top_members<C>(
    nodes: &mut [RoutedNode<NodeIndex>],
    clusters: &[ComputedCluster<C>],
) {
    let mut sibling_groups = HashMap::<Option<usize>, Vec<usize>>::new();
    for (index, cluster) in clusters.iter().enumerate() {
        sibling_groups
            .entry(cluster.parent)
            .or_default()
            .push(index);
    }

    for sibling_group in sibling_groups.into_values().filter(|group| group.len() > 1) {
        let mut target_top: Option<f64> = None;
        let mut cluster_tops = Vec::new();
        let mut cluster_group_bounds = Vec::new();

        for cluster_index in sibling_group {
            let Some(cluster) = clusters.get(cluster_index) else {
                continue;
            };
            let Some(top) = nodes
                .iter()
                .filter(|node| cluster.node_ids.contains(&node.id))
                .map(|node| node.bounds.min_y)
                .min_by(|left, right| left.total_cmp(right))
            else {
                continue;
            };

            target_top = Some(match target_top {
                Some(current) => current.min(top),
                None => top,
            });
            cluster_tops.push((cluster, top));
            cluster_group_bounds.push(cluster.bounds);
        }

        let Some(target_top) = target_top else {
            continue;
        };
        let has_vertical_overlap = cluster_group_bounds
            .iter()
            .enumerate()
            .any(|(index, left)| {
                cluster_group_bounds.iter().skip(index + 1).any(|right| {
                    left.max_y.min(right.max_y) - left.min_y.max(right.min_y) > EPSILON
                })
            });
        if !has_vertical_overlap {
            continue;
        }

        for (cluster, top) in cluster_tops {
            let shift = target_top - top;
            if shift.abs() <= EPSILON {
                continue;
            }

            for node in nodes.iter_mut() {
                if cluster.node_ids.contains(&node.id) {
                    node.center.1 += shift;
                    node.bounds = Rect::from_center_size(node.center, node.size);
                }
            }
        }
    }
}

fn reorder_cluster_rank_slots_by_flow<V, E, C>(
    nodes: &mut [RoutedNode<NodeIndex>],
    graph: &StableDiGraph<V, E>,
    clusters: &[ComputedCluster<C>],
) {
    let ranks = group_nodes_into_ranks(nodes);
    let mut rank_by_node = HashMap::new();
    for (rank_index, rank) in ranks.iter().enumerate() {
        for node_index in rank {
            rank_by_node.insert(*node_index, rank_index);
        }
    }
    let node_index_by_id = nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.id, index))
        .collect::<HashMap<_, _>>();

    for cluster in clusters.iter().filter(|cluster| cluster.parent.is_some()) {
        let mut by_rank = HashMap::<usize, Vec<usize>>::new();
        for node_id in &cluster.node_ids {
            let Some(node_index) = node_index_by_id.get(node_id).copied() else {
                continue;
            };
            let Some(rank_index) = rank_by_node.get(&node_index).copied() else {
                continue;
            };
            by_rank.entry(rank_index).or_default().push(node_index);
        }

        for node_indices in by_rank.into_values().filter(|nodes| nodes.len() > 1) {
            let mut slots = node_indices
                .iter()
                .map(|node_index| nodes[*node_index].center.0)
                .collect::<Vec<_>>();
            slots.sort_by(|left, right| left.total_cmp(right));

            let mut ordered = node_indices.clone();
            ordered.sort_by(|left, right| {
                flow_sort_key(nodes, graph, *left)
                    .cmp(&flow_sort_key(nodes, graph, *right))
                    .then_with(|| nodes[*left].center.0.total_cmp(&nodes[*right].center.0))
            });

            for (node_index, slot_x) in ordered.into_iter().zip(slots.into_iter()) {
                nodes[node_index].center.0 = slot_x;
                nodes[node_index].bounds =
                    Rect::from_center_size(nodes[node_index].center, nodes[node_index].size);
            }
        }
    }
}

fn reorder_rank_slots_by_adjacent_neighbor_x<V, E>(
    nodes: &mut [RoutedNode<NodeIndex>],
    graph: &StableDiGraph<V, E>,
) {
    let ranks = group_nodes_into_ranks(nodes);
    if ranks.len() < 2 {
        return;
    }

    let node_index_by_id = nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.id, index))
        .collect::<HashMap<_, _>>();
    let rank_by_node = ranks
        .iter()
        .enumerate()
        .flat_map(|(rank_index, rank)| {
            rank.iter()
                .copied()
                .map(move |node_index| (node_index, rank_index))
        })
        .collect::<HashMap<_, _>>();

    for rank in ranks.into_iter().filter(|rank| rank.len() > 1) {
        let mut slots = rank
            .iter()
            .map(|node_index| nodes[*node_index].center.0)
            .collect::<Vec<_>>();
        slots.sort_by(|left, right| left.total_cmp(right));

        let mut ordered = rank.clone();
        ordered.sort_by(|left, right| {
            adjacent_neighbor_x_median(*left, nodes, graph, &node_index_by_id, &rank_by_node)
                .total_cmp(&adjacent_neighbor_x_median(
                    *right,
                    nodes,
                    graph,
                    &node_index_by_id,
                    &rank_by_node,
                ))
                .then_with(|| nodes[*left].center.0.total_cmp(&nodes[*right].center.0))
                .then_with(|| left.cmp(right))
        });

        for (node_index, slot_x) in ordered.into_iter().zip(slots.into_iter()) {
            nodes[node_index].center.0 = slot_x;
            nodes[node_index].bounds =
                Rect::from_center_size(nodes[node_index].center, nodes[node_index].size);
        }
    }
}

fn adjacent_neighbor_x_median<V, E>(
    node_index: usize,
    nodes: &[RoutedNode<NodeIndex>],
    graph: &StableDiGraph<V, E>,
    node_index_by_id: &HashMap<NodeIndex, usize>,
    rank_by_node: &HashMap<usize, usize>,
) -> f64 {
    let node_id = nodes[node_index].id;
    let Some(rank_index) = rank_by_node.get(&node_index).copied() else {
        return nodes[node_index].center.0;
    };
    let mut positions = Vec::new();

    for neighbor in graph.neighbors_directed(node_id, Incoming) {
        let Some(neighbor_index) = node_index_by_id.get(&neighbor).copied() else {
            continue;
        };
        if rank_by_node
            .get(&neighbor_index)
            .copied()
            .is_some_and(|neighbor_rank| neighbor_rank + 1 == rank_index)
        {
            positions.push(nodes[neighbor_index].center.0);
        }
    }

    for neighbor in graph.neighbors_directed(node_id, Outgoing) {
        let Some(neighbor_index) = node_index_by_id.get(&neighbor).copied() else {
            continue;
        };
        if rank_by_node
            .get(&neighbor_index)
            .copied()
            .is_some_and(|neighbor_rank| neighbor_rank == rank_index + 1)
        {
            positions.push(nodes[neighbor_index].center.0);
        }
    }

    if positions.is_empty() {
        return nodes[node_index].center.0;
    }

    positions.sort_by(|left, right| left.total_cmp(right));
    let middle = positions.len() / 2;
    if positions.len() % 2 == 0 {
        (positions[middle - 1] + positions[middle]) * 0.5
    } else {
        positions[middle]
    }
}

fn flow_sort_key<V, E>(
    nodes: &[RoutedNode<NodeIndex>],
    graph: &StableDiGraph<V, E>,
    node_index: usize,
) -> (bool, i64) {
    let mut total_x = 0.0;
    let mut count = 0usize;

    for neighbor in graph.neighbors_directed(nodes[node_index].id, Outgoing) {
        if let Some(position) = nodes.iter().position(|node| node.id == neighbor) {
            total_x += nodes[position].center.0;
            count += 1;
        }
    }

    if count == 0 {
        return (false, i64::MIN);
    }

    (true, (total_x / count as f64 * 1000.0) as i64)
}

fn sink_cluster_top_singletons_toward_external_targets<V, E, C>(
    nodes: &mut [RoutedNode<NodeIndex>],
    graph: &StableDiGraph<V, E>,
    clusters: &[ComputedCluster<C>],
) {
    let ranks = group_nodes_into_ranks(nodes);
    if ranks.len() < 2 {
        return;
    }
    let node_index_by_id = nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.id, index))
        .collect::<HashMap<_, _>>();
    let rank_by_node = ranks
        .iter()
        .enumerate()
        .flat_map(|(rank_index, rank)| {
            rank.iter()
                .copied()
                .map(move |node_index| (node_index, rank_index))
        })
        .collect::<HashMap<_, _>>();

    for cluster in clusters.iter().filter(|cluster| cluster.parent.is_some()) {
        let mut by_rank = HashMap::<usize, Vec<usize>>::new();
        for node_id in &cluster.node_ids {
            let Some(node_index) = node_index_by_id.get(node_id).copied() else {
                continue;
            };
            let Some(rank_index) = rank_by_node.get(&node_index).copied() else {
                continue;
            };
            by_rank.entry(rank_index).or_default().push(node_index);
        }

        let Some((top_rank_index, top_members)) =
            by_rank.iter().min_by_key(|(rank_index, _)| *rank_index)
        else {
            continue;
        };
        if top_members.len() != 1 {
            continue;
        }

        let top_node_index = top_members[0];
        let target_rank_index = graph
            .neighbors_directed(nodes[top_node_index].id, Outgoing)
            .filter(|neighbor| !cluster.node_ids.contains(neighbor))
            .filter_map(|neighbor| node_index_by_id.get(&neighbor).copied())
            .filter_map(|neighbor_index| rank_by_node.get(&neighbor_index).copied())
            .filter(|rank_index| *rank_index > *top_rank_index)
            .max();

        let Some(target_rank_index) = target_rank_index else {
            continue;
        };
        let Some(target_rank) = ranks.get(target_rank_index) else {
            continue;
        };
        let Some(target_y) = target_rank
            .first()
            .map(|node_index| nodes[*node_index].center.1)
        else {
            continue;
        };

        nodes[top_node_index].center.1 = target_y;
        nodes[top_node_index].bounds =
            Rect::from_center_size(nodes[top_node_index].center, nodes[top_node_index].size);
    }
}

fn expand_descending_cluster_edges<V, E, C>(
    nodes: &mut [RoutedNode<NodeIndex>],
    graph: &StableDiGraph<V, E>,
    clusters: &[ComputedCluster<C>],
) {
    let mut rank_levels = nodes.iter().map(|node| node.center.1).collect::<Vec<_>>();
    rank_levels.sort_by(|left, right| left.total_cmp(right));
    rank_levels.dedup_by(|left, right| almost_equal(*left, *right));
    if rank_levels.is_empty() {
        return;
    }

    let typical_gap = rank_levels
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .filter(|gap| *gap > EPSILON)
        .min_by(|left, right| left.total_cmp(right))
        .unwrap_or(60.0);
    let node_index_by_id = nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.id, index))
        .collect::<HashMap<_, _>>();

    for _ in 0..4 {
        let mut moved = false;

        for cluster in clusters.iter().filter(|cluster| cluster.parent.is_some()) {
            for source in &cluster.node_ids {
                for target in graph.neighbors_directed(*source, Outgoing) {
                    if !cluster.node_ids.contains(&target) {
                        continue;
                    }

                    let Some(source_index) = node_index_by_id.get(source).copied() else {
                        continue;
                    };
                    let Some(target_index) = node_index_by_id.get(&target).copied() else {
                        continue;
                    };

                    let source_y = nodes[source_index].center.1;
                    let target_y = nodes[target_index].center.1;
                    if target_y > source_y + EPSILON {
                        continue;
                    }

                    let next_lower_y = rank_levels
                        .iter()
                        .copied()
                        .find(|level| *level > source_y + EPSILON)
                        .unwrap_or(source_y + typical_gap);
                    if next_lower_y <= target_y + EPSILON {
                        continue;
                    }

                    nodes[target_index].center.1 = next_lower_y;
                    nodes[target_index].bounds = Rect::from_center_size(
                        nodes[target_index].center,
                        nodes[target_index].size,
                    );
                    moved = true;
                }
            }
        }

        if !moved {
            break;
        }

        rank_levels = nodes.iter().map(|node| node.center.1).collect::<Vec<_>>();
        rank_levels.sort_by(|left, right| left.total_cmp(right));
        rank_levels.dedup_by(|left, right| almost_equal(*left, *right));
    }
}

fn align_cluster_bottom_singletons<V, E, C>(
    nodes: &mut [RoutedNode<NodeIndex>],
    graph: &StableDiGraph<V, E>,
    clusters: &[ComputedCluster<C>],
) {
    let ranks = group_nodes_into_ranks(nodes);
    let mut rank_by_node = HashMap::new();
    for (rank_index, rank) in ranks.iter().enumerate() {
        for node_index in rank {
            rank_by_node.insert(*node_index, rank_index);
        }
    }

    let node_index_by_id = nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.id, index))
        .collect::<HashMap<_, _>>();

    for cluster in clusters.iter().filter(|cluster| cluster.parent.is_some()) {
        let mut by_rank = HashMap::<usize, Vec<usize>>::new();
        for node_id in &cluster.node_ids {
            let Some(node_index) = node_index_by_id.get(node_id).copied() else {
                continue;
            };
            let Some(rank_index) = rank_by_node.get(&node_index).copied() else {
                continue;
            };
            by_rank.entry(rank_index).or_default().push(node_index);
        }

        let mut occupied_ranks = by_rank.into_iter().collect::<Vec<_>>();
        occupied_ranks.sort_by_key(|(rank_index, _)| *rank_index);
        if occupied_ranks.len() < 2 {
            continue;
        }

        let Some((_, bottom_members)) = occupied_ranks.last() else {
            continue;
        };
        if bottom_members.len() != 1 {
            continue;
        }
        let bottom_node_index = bottom_members[0];

        let Some((_, upper_members)) = occupied_ranks.get(occupied_ranks.len() - 2) else {
            continue;
        };

        let predecessor_positions = graph
            .neighbors_directed(nodes[bottom_node_index].id, Incoming)
            .filter(|neighbor| cluster.node_ids.contains(neighbor))
            .filter_map(|neighbor| node_index_by_id.get(&neighbor).copied())
            .filter_map(|node_index| {
                rank_by_node
                    .get(&node_index)
                    .copied()
                    .zip(Some(nodes[node_index].center.0))
            })
            .filter(|(rank_index, _)| *rank_index + 1 == occupied_ranks[occupied_ranks.len() - 1].0)
            .map(|(_, x)| x)
            .collect::<Vec<_>>();

        let target_x = if predecessor_positions.is_empty() {
            upper_members
                .iter()
                .map(|node_index| nodes[*node_index].center.0)
                .sum::<f64>()
                / upper_members.len() as f64
        } else {
            predecessor_positions.iter().sum::<f64>() / predecessor_positions.len() as f64
        };

        nodes[bottom_node_index].center.0 = target_x;
        nodes[bottom_node_index].bounds = Rect::from_center_size(
            nodes[bottom_node_index].center,
            nodes[bottom_node_index].size,
        );
    }
}

fn compact_cluster_top_fanouts<C>(
    nodes: &mut [RoutedNode<NodeIndex>],
    clusters: &[ComputedCluster<C>],
    vertex_spacing: f64,
) {
    let ranks = group_nodes_into_ranks(nodes);
    let rank_by_node = ranks
        .iter()
        .enumerate()
        .flat_map(|(rank_index, rank)| {
            rank.iter()
                .copied()
                .map(move |node_index| (node_index, rank_index))
        })
        .collect::<HashMap<_, _>>();
    let node_index_by_id = nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.id, index))
        .collect::<HashMap<_, _>>();

    for cluster in clusters.iter().filter(|cluster| cluster.parent.is_some()) {
        let mut occupied_ranks = cluster
            .node_ids
            .iter()
            .filter_map(|node_id| node_index_by_id.get(node_id).copied())
            .filter_map(|node_index| {
                rank_by_node
                    .get(&node_index)
                    .copied()
                    .map(|rank_index| (rank_index, node_index))
            })
            .fold(
                HashMap::<usize, Vec<usize>>::new(),
                |mut acc, (rank_index, node_index)| {
                    acc.entry(rank_index).or_default().push(node_index);
                    acc
                },
            )
            .into_iter()
            .collect::<Vec<_>>();
        occupied_ranks.sort_by_key(|(rank_index, _)| *rank_index);
        if occupied_ranks.len() < 2 {
            continue;
        }

        let Some((_, top_members)) = occupied_ranks.first() else {
            continue;
        };
        if top_members.len() != 1 {
            continue;
        }
        let Some((_, next_members)) = occupied_ranks.get(1) else {
            continue;
        };
        if next_members.len() < 2 {
            continue;
        }

        let top_x = nodes[top_members[0]].center.0;
        let mut ordered_members = next_members.clone();
        ordered_members.sort_by(|left, right| {
            nodes[*left]
                .center
                .0
                .total_cmp(&nodes[*right].center.0)
                .then_with(|| left.cmp(right))
        });

        let widths = ordered_members
            .iter()
            .map(|node_index| nodes[*node_index].size.0)
            .collect::<Vec<_>>();
        let total_width = widths.iter().sum::<f64>()
            + vertex_spacing.max(12.0) * ordered_members.len().saturating_sub(1) as f64;
        let mut current_left = top_x - total_width * 0.5;

        for (position, node_index) in ordered_members.into_iter().enumerate() {
            let width = widths[position];
            nodes[node_index].center.0 = current_left + width * 0.5;
            nodes[node_index].bounds =
                Rect::from_center_size(nodes[node_index].center, nodes[node_index].size);
            current_left += width + vertex_spacing.max(12.0);
        }
    }
}

fn spread_singleton_cluster_chains<V, E, C>(
    nodes: &mut [RoutedNode<NodeIndex>],
    graph: &StableDiGraph<V, E>,
    clusters: &[ComputedCluster<C>],
) {
    let ranks = group_nodes_into_ranks(nodes);
    let mut rank_by_node = HashMap::new();
    for (rank_index, rank) in ranks.iter().enumerate() {
        for node_index in rank {
            rank_by_node.insert(*node_index, rank_index);
        }
    }

    let node_index_by_id = nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.id, index))
        .collect::<HashMap<_, _>>();
    let cluster_by_source = clusters
        .iter()
        .enumerate()
        .map(|(index, cluster)| (cluster.source_index, index))
        .collect::<HashMap<_, _>>();

    for cluster in clusters.iter().filter(|cluster| cluster.parent.is_some()) {
        let Some(parent_index) = cluster.parent else {
            continue;
        };
        let Some(parent_cluster_index) = cluster_by_source.get(&parent_index).copied() else {
            continue;
        };
        let parent_center = (clusters[parent_cluster_index].bounds.min_x
            + clusters[parent_cluster_index].bounds.max_x)
            * 0.5;
        let cluster_center = (cluster.bounds.min_x + cluster.bounds.max_x) * 0.5;
        let direction = if cluster_center >= parent_center {
            1.0
        } else {
            -1.0
        };

        let mut occupied_ranks = cluster
            .node_ids
            .iter()
            .filter_map(|node_id| node_index_by_id.get(node_id).copied())
            .filter_map(|node_index| {
                rank_by_node
                    .get(&node_index)
                    .copied()
                    .map(|rank| (rank, node_index))
            })
            .fold(
                HashMap::<usize, Vec<usize>>::new(),
                |mut acc, (rank, node_index)| {
                    acc.entry(rank).or_default().push(node_index);
                    acc
                },
            )
            .into_iter()
            .collect::<Vec<_>>();
        occupied_ranks.sort_by_key(|(rank_index, _)| *rank_index);

        let singleton_chain = occupied_ranks
            .iter()
            .filter_map(|(_, members)| (members.len() == 1).then_some(members[0]))
            .collect::<Vec<_>>();
        if singleton_chain.len() < 2 {
            continue;
        }

        if let Some(target_x_positions) = singleton_chain_targets_from_external_flow(
            nodes,
            graph,
            cluster,
            clusters,
            &node_index_by_id,
        ) {
            for (node_index, target_x) in singleton_chain
                .iter()
                .copied()
                .zip(target_x_positions.into_iter())
            {
                nodes[node_index].center.0 = target_x;
                nodes[node_index].bounds =
                    Rect::from_center_size(nodes[node_index].center, nodes[node_index].size);
            }
            continue;
        }

        let mean_x = singleton_chain
            .iter()
            .map(|node_index| nodes[*node_index].center.0)
            .sum::<f64>()
            / singleton_chain.len() as f64;
        let step = singleton_chain
            .iter()
            .map(|node_index| nodes[*node_index].size.0)
            .min_by(|left, right| left.total_cmp(right))
            .map(|width| (width * 0.16).max(10.0))
            .unwrap_or(10.0);
        let midpoint = (singleton_chain.len() as f64 - 1.0) * 0.5;

        for (index, node_index) in singleton_chain.into_iter().enumerate() {
            nodes[node_index].center.0 = mean_x + direction * step * (index as f64 - midpoint);
            nodes[node_index].bounds =
                Rect::from_center_size(nodes[node_index].center, nodes[node_index].size);
        }
    }
}

fn align_cluster_top_singletons_toward_sibling_corridors<C>(
    nodes: &mut [RoutedNode<NodeIndex>],
    clusters: &[ComputedCluster<C>],
) {
    let ranks = group_nodes_into_ranks(nodes);
    let mut rank_by_node = HashMap::new();
    for (rank_index, rank) in ranks.iter().enumerate() {
        for node_index in rank {
            rank_by_node.insert(*node_index, rank_index);
        }
    }

    let node_index_by_id = nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.id, index))
        .collect::<HashMap<_, _>>();

    for cluster in clusters.iter().filter(|cluster| cluster.parent.is_some()) {
        let mut occupied_ranks = cluster
            .node_ids
            .iter()
            .filter_map(|node_id| node_index_by_id.get(node_id).copied())
            .filter_map(|node_index| {
                rank_by_node
                    .get(&node_index)
                    .copied()
                    .map(|rank| (rank, node_index))
            })
            .collect::<Vec<_>>();
        occupied_ranks.sort_by_key(|(rank_index, _)| *rank_index);

        let Some((_, top_node_index)) = occupied_ranks.first().copied() else {
            continue;
        };
        if occupied_ranks
            .iter()
            .filter(|(rank_index, _)| *rank_index == occupied_ranks[0].0)
            .count()
            != 1
        {
            continue;
        }

        let top_rank_index = occupied_ranks[0].0;
        let next_rank_index = occupied_ranks
            .iter()
            .filter(|(rank_index, _)| *rank_index > top_rank_index)
            .map(|(rank_index, _)| *rank_index)
            .min();
        let target_x = next_rank_index.and_then(|next_rank_index| {
            let next_rank_members = occupied_ranks
                .iter()
                .filter(|(rank_index, _)| *rank_index == next_rank_index)
                .map(|(_, node_index)| *node_index)
                .collect::<Vec<_>>();
            (!next_rank_members.is_empty()).then(|| {
                next_rank_members
                    .iter()
                    .map(|node_index| nodes[*node_index].center.0)
                    .sum::<f64>()
                    / next_rank_members.len() as f64
            })
        });
        let Some(target_x) = target_x else {
            continue;
        };

        nodes[top_node_index].center.0 = target_x;
        nodes[top_node_index].bounds =
            Rect::from_center_size(nodes[top_node_index].center, nodes[top_node_index].size);
    }
}

fn resolve_node_overlaps(nodes: &mut [RoutedNode<NodeIndex>], clearance: f64) {
    for _ in 0..16 {
        let mut moved = false;

        for lower_index in 0..nodes.len() {
            for upper_index in 0..nodes.len() {
                if upper_index == lower_index {
                    continue;
                }

                let upper = nodes[upper_index].bounds;
                let lower = nodes[lower_index].bounds;
                let overlap_x = upper.max_x.min(lower.max_x) - upper.min_x.max(lower.min_x);
                let overlap_y = upper.max_y.min(lower.max_y) - upper.min_y.max(lower.min_y);
                if overlap_x <= EPSILON || overlap_y <= EPSILON {
                    continue;
                }
                if nodes[lower_index].center.1 <= nodes[upper_index].center.1 + EPSILON {
                    continue;
                }

                let shift_y = overlap_y + clearance;
                nodes[lower_index].center.1 += shift_y;
                nodes[lower_index].bounds =
                    Rect::from_center_size(nodes[lower_index].center, nodes[lower_index].size);
                moved = true;
            }
        }

        if !moved {
            break;
        }
    }
}

fn singleton_chain_targets_from_external_flow<V, E, C>(
    nodes: &[RoutedNode<NodeIndex>],
    graph: &StableDiGraph<V, E>,
    cluster: &ComputedCluster<C>,
    clusters: &[ComputedCluster<C>],
    node_index_by_id: &HashMap<NodeIndex, usize>,
) -> Option<Vec<f64>> {
    let mut rank_members = cluster
        .node_ids
        .iter()
        .filter_map(|node_id| node_index_by_id.get(node_id).copied())
        .fold(
            HashMap::<i64, Vec<usize>>::new(),
            |mut grouped, node_index| {
                grouped
                    .entry((nodes[node_index].center.1 * 1000.0).round() as i64)
                    .or_default()
                    .push(node_index);
                grouped
            },
        )
        .into_values()
        .collect::<Vec<_>>();
    rank_members
        .sort_by(|left, right| nodes[left[0]].center.1.total_cmp(&nodes[right[0]].center.1));

    let singleton_chain = rank_members
        .iter()
        .filter_map(|members| (members.len() == 1).then_some(members[0]))
        .collect::<Vec<_>>();
    if singleton_chain.len() < 3 {
        return None;
    }

    let cluster_center = (cluster.bounds.min_x + cluster.bounds.max_x) * 0.5;
    let half_width = singleton_chain
        .iter()
        .map(|node_index| nodes[*node_index].size.0 * 0.5)
        .max_by(|left, right| left.total_cmp(right))
        .unwrap_or(36.0);
    let sibling_gap = 48.0;
    let lane_inset = half_width + 10.0;
    let left_sibling_max_x = sibling_bounds(clusters, cluster)
        .0
        .map(|bounds| bounds.max_x);
    let right_sibling_min_x = sibling_bounds(clusters, cluster)
        .1
        .map(|bounds| bounds.min_x);
    let left_x = left_sibling_max_x
        .map(|max_x| max_x + sibling_gap + half_width)
        .unwrap_or(cluster.bounds.min_x + lane_inset)
        .min(cluster_center);
    let center_x = cluster_center;
    let right_x = right_sibling_min_x
        .map(|min_x| min_x - sibling_gap - half_width)
        .unwrap_or(cluster.bounds.max_x - lane_inset)
        .max(cluster_center);

    let pulls = singleton_chain
        .iter()
        .map(|node_index| {
            external_neighbor_average_x(nodes, graph, cluster, node_index_by_id, *node_index)
        })
        .collect::<Vec<_>>();

    let has_left_pull = pulls
        .iter()
        .flatten()
        .any(|pull_x| *pull_x < cluster_center - EPSILON);
    let has_right_pull = pulls
        .iter()
        .flatten()
        .any(|pull_x| *pull_x > cluster_center + EPSILON);
    let has_left_sibling = left_sibling_max_x.is_some();
    let has_right_sibling = right_sibling_min_x.is_some();

    if !(has_left_pull && has_right_pull) {
        if has_left_sibling && !has_right_sibling {
            let mut targets = vec![left_x; singleton_chain.len()];
            if let Some(last) = targets.last_mut() {
                *last = right_x;
            }
            return Some(targets);
        }

        if !has_left_sibling && has_right_sibling {
            let mut targets = vec![right_x; singleton_chain.len()];
            if let Some(last) = targets.last_mut() {
                *last = left_x;
            }
            return Some(targets);
        }

        return None;
    }

    let targets = pulls
        .iter()
        .map(|pull_x| match pull_x {
            Some(value) if *value < cluster_center - EPSILON => left_x,
            Some(value) if *value > cluster_center + EPSILON => right_x,
            _ => center_x,
        })
        .collect::<Vec<_>>();

    Some(targets)
}

fn sibling_bounds<C>(
    clusters: &[ComputedCluster<C>],
    cluster: &ComputedCluster<C>,
) -> (Option<Rect>, Option<Rect>) {
    let mut siblings = clusters
        .iter()
        .filter(|candidate| candidate.parent == cluster.parent)
        .collect::<Vec<_>>();
    siblings.sort_by_key(|candidate| candidate.source_index);

    let Some(position) = siblings
        .iter()
        .position(|candidate| candidate.source_index == cluster.source_index)
    else {
        return (None, None);
    };

    let left = position
        .checked_sub(1)
        .and_then(|index| siblings.get(index))
        .map(|sibling| sibling.bounds);
    let right = siblings.get(position + 1).map(|sibling| sibling.bounds);

    (left, right)
}

fn external_neighbor_average_x<V, E, C>(
    nodes: &[RoutedNode<NodeIndex>],
    graph: &StableDiGraph<V, E>,
    cluster: &ComputedCluster<C>,
    node_index_by_id: &HashMap<NodeIndex, usize>,
    node_index: usize,
) -> Option<f64> {
    let node_id = nodes.get(node_index)?.id;
    let mut total_x = 0.0;
    let mut count = 0usize;

    for neighbor in graph.neighbors_directed(node_id, Incoming) {
        if cluster.node_ids.contains(&neighbor) {
            continue;
        }
        let Some(neighbor_index) = node_index_by_id.get(&neighbor).copied() else {
            continue;
        };
        total_x += nodes[neighbor_index].center.0;
        count += 1;
    }

    for neighbor in graph.neighbors_directed(node_id, Outgoing) {
        if cluster.node_ids.contains(&neighbor) {
            continue;
        }
        let Some(neighbor_index) = node_index_by_id.get(&neighbor).copied() else {
            continue;
        };
        total_x += nodes[neighbor_index].center.0;
        count += 1;
    }

    (count > 0).then_some(total_x / count as f64)
}

fn build_cluster_layouts<C: Clone>(
    clusters: &[ClusterSpec<C>],
    component_nodes: &HashSet<NodeIndex>,
    node_bounds: &HashMap<NodeIndex, Rect>,
    render_config: &RenderConfig,
) -> Vec<ComputedCluster<C>> {
    #[derive(Clone)]
    struct ClusterBounds<C> {
        source_index: usize,
        id: C,
        parent: Option<usize>,
        padding: f64,
        node_ids: HashSet<NodeIndex>,
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
            let mut node_ids = HashSet::new();
            for node in &cluster.nodes {
                if let Some(node_bounds) = node_bounds.get(node) {
                    node_ids.insert(*node);
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
                source_index: cluster_index,
                id: cluster.id.clone(),
                parent,
                padding,
                node_ids,
                content_bounds,
                final_bounds: content_bounds.map(|bounds| cluster_final_bounds(bounds, padding)),
            })
        })
        .collect::<Vec<_>>();

    let cluster_count = cluster_bounds.len();
    for _ in 0..cluster_count {
        let mut changed = false;
        for cluster_index in 0..cluster_count {
            let Some((parent_index, child_bounds, child_node_ids)) =
                cluster_bounds[cluster_index].as_ref().and_then(|cluster| {
                    cluster
                        .parent
                        .zip(cluster.final_bounds)
                        .map(|(parent_index, child_bounds)| {
                            (parent_index, child_bounds, cluster.node_ids.clone())
                        })
                })
            else {
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
                    .map(|bounds| cluster_final_bounds(bounds, parent.padding));
                changed = true;
            }
            let previous_count = parent.node_ids.len();
            parent.node_ids.extend(child_node_ids);
            if parent.node_ids.len() != previous_count {
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
            component_clusters.push(ComputedCluster {
                source_index: cluster.source_index,
                id: cluster.id,
                bounds,
                parent: cluster.parent,
                node_ids: cluster.node_ids,
            });
        }
    }
    component_clusters
}

fn apply_child_cluster_local_layout<V, E, C>(
    nodes: &mut [RoutedNode<NodeIndex>],
    graph: &StableDiGraph<V, E>,
    clusters: &[ClusterSpec<C>],
    computed_clusters: &[ComputedCluster<C>],
    vertex_spacing: f64,
) {
    if nodes.is_empty() || clusters.is_empty() || computed_clusters.is_empty() {
        return;
    }

    let ranks = group_nodes_into_ranks(nodes);
    if ranks.is_empty() {
        return;
    }

    let node_index_by_id = nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.id, index))
        .collect::<HashMap<_, _>>();
    let rank_by_node = ranks
        .iter()
        .enumerate()
        .flat_map(|(rank_index, rank)| {
            rank.iter()
                .copied()
                .map(move |node_index| (node_index, rank_index))
        })
        .collect::<HashMap<_, _>>();
    let rank_y_values = ranks
        .iter()
        .filter_map(|rank| rank.first().map(|node_index| nodes[*node_index].center.1))
        .collect::<Vec<_>>();
    let mut positive_rank_gaps = rank_y_values
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .filter(|gap| *gap > EPSILON)
        .collect::<Vec<_>>();
    positive_rank_gaps.sort_by(|left, right| left.total_cmp(right));
    let typical_rank_gap = if positive_rank_gaps.is_empty() {
        72.0
    } else {
        positive_rank_gaps.iter().sum::<f64>() / positive_rank_gaps.len() as f64
    };
    let cluster_centers = computed_clusters
        .iter()
        .map(|cluster| {
            (
                cluster.source_index,
                (cluster.bounds.min_x + cluster.bounds.max_x) * 0.5,
            )
        })
        .collect::<HashMap<_, _>>();
    let node_gap = vertex_spacing.max(12.0);

    #[derive(Clone)]
    struct LocalClusterData {
        cluster_index: usize,
        parent: usize,
        center_x: f64,
        ordered_nodes: Vec<NodeIndex>,
        local_ranks: HashMap<NodeIndex, usize>,
        max_local_rank: usize,
    }

    let mut local_clusters = Vec::<LocalClusterData>::new();
    for (cluster_index, cluster) in clusters.iter().enumerate() {
        let Some(parent) = cluster.parent else {
            continue;
        };
        let Some(center_x) = cluster_centers.get(&cluster_index).copied() else {
            continue;
        };
        let ordered_nodes = cluster
            .nodes
            .iter()
            .copied()
            .filter(|node_id| node_index_by_id.contains_key(node_id))
            .collect::<Vec<_>>();
        if ordered_nodes.is_empty() {
            continue;
        }
        let local_ranks = local_cluster_ranks(graph, &ordered_nodes);
        let max_local_rank = local_ranks.values().copied().max().unwrap_or(0);
        local_clusters.push(LocalClusterData {
            cluster_index,
            parent,
            center_x,
            ordered_nodes,
            local_ranks,
            max_local_rank,
        });
    }

    let mut sibling_groups = HashMap::<usize, Vec<LocalClusterData>>::new();
    for cluster in local_clusters {
        sibling_groups
            .entry(cluster.parent)
            .or_default()
            .push(cluster);
    }

    let mut base_rank_by_cluster = HashMap::<usize, usize>::new();
    let mut group_base_rank_by_cluster = HashMap::<usize, usize>::new();
    let mut group_base_y_by_cluster = HashMap::<usize, f64>::new();
    let mut effective_center_x_by_cluster = HashMap::<usize, f64>::new();
    for mut siblings in sibling_groups.into_values() {
        siblings.sort_by_key(|cluster| cluster.cluster_index);
        let group_base_rank = siblings
            .iter()
            .flat_map(|cluster| cluster.ordered_nodes.iter())
            .filter_map(|node_id| node_index_by_id.get(node_id).copied())
            .filter_map(|node_index| rank_by_node.get(&node_index).copied())
            .min()
            .unwrap_or(0);
        let group_base_y = rank_y_values
            .get(group_base_rank)
            .copied()
            .unwrap_or_else(|| rank_y_values.last().copied().unwrap_or(0.0));
        let mut next_rank = group_base_rank;
        let mut assigned_ranges = Vec::with_capacity(siblings.len());
        for cluster in &siblings {
            base_rank_by_cluster.insert(cluster.cluster_index, next_rank);
            group_base_rank_by_cluster.insert(cluster.cluster_index, group_base_rank);
            group_base_y_by_cluster.insert(cluster.cluster_index, group_base_y);
            assigned_ranges.push((
                cluster.cluster_index,
                next_rank,
                next_rank + cluster.max_local_rank,
            ));
            next_rank += 1;
        }

        let parent_center_x = cluster_centers
            .get(&siblings[0].parent)
            .copied()
            .unwrap_or_else(|| {
                siblings.iter().map(|cluster| cluster.center_x).sum::<f64>() / siblings.len() as f64
            });
        let has_vertical_overlap = assigned_ranges.iter().enumerate().any(|(index, left)| {
            assigned_ranges
                .iter()
                .skip(index + 1)
                .any(|right| left.1 <= right.2 && right.1 <= left.2)
        });

        for cluster in siblings {
            effective_center_x_by_cluster.insert(
                cluster.cluster_index,
                if has_vertical_overlap {
                    cluster.center_x
                } else {
                    parent_center_x
                },
            );
        }
    }

    for cluster in clusters
        .iter()
        .enumerate()
        .filter_map(|(cluster_index, cluster)| cluster.parent.map(|_| cluster_index))
    {
        let Some(center_x) = effective_center_x_by_cluster
            .get(&cluster)
            .copied()
            .or_else(|| cluster_centers.get(&cluster).copied())
        else {
            continue;
        };
        let Some(base_rank) = base_rank_by_cluster.get(&cluster).copied() else {
            continue;
        };
        let ordered_nodes = clusters[cluster]
            .nodes
            .iter()
            .copied()
            .filter(|node_id| node_index_by_id.contains_key(node_id))
            .collect::<Vec<_>>();
        if ordered_nodes.is_empty() {
            continue;
        }
        let declaration_order = ordered_nodes
            .iter()
            .enumerate()
            .map(|(index, node_id)| (*node_id, index))
            .collect::<HashMap<_, _>>();
        let local_ranks = local_cluster_ranks(graph, &ordered_nodes);
        let cluster_node_ids = ordered_nodes.iter().copied().collect::<HashSet<_>>();

        let mut nodes_by_local_rank = HashMap::<usize, Vec<NodeIndex>>::new();
        for node_id in &ordered_nodes {
            let local_rank = *local_ranks.get(node_id).unwrap_or(&0);
            nodes_by_local_rank
                .entry(local_rank)
                .or_default()
                .push(*node_id);
        }

        let mut sorted_local_ranks = nodes_by_local_rank.into_iter().collect::<Vec<_>>();
        sorted_local_ranks.sort_by_key(|(local_rank, _)| *local_rank);

        for (local_rank, mut rank_nodes) in sorted_local_ranks {
            let row_len = rank_nodes.len();
            rank_nodes.sort_by(|left, right| {
                let left_declaration = declaration_order.get(left).copied().unwrap_or(usize::MAX);
                let right_declaration = declaration_order.get(right).copied().unwrap_or(usize::MAX);
                let left_counts = external_degree_counts(graph, &cluster_node_ids, *left);
                let right_counts = external_degree_counts(graph, &cluster_node_ids, *right);

                if row_len == 2 {
                    left_counts
                        .1
                        .cmp(&right_counts.1)
                        .then_with(|| left_declaration.cmp(&right_declaration))
                } else {
                    let left_source_like = left_counts.1 > left_counts.0;
                    let right_source_like = right_counts.1 > right_counts.0;
                    left_source_like
                        .cmp(&right_source_like)
                        .then_with(|| left_declaration.cmp(&right_declaration))
                }
            });
            let target_rank = base_rank + local_rank;
            let group_base_rank = group_base_rank_by_cluster
                .get(&cluster)
                .copied()
                .unwrap_or(base_rank);
            let group_base_y = group_base_y_by_cluster
                .get(&cluster)
                .copied()
                .unwrap_or_else(|| rank_y_values.get(group_base_rank).copied().unwrap_or(0.0));
            let target_y = group_base_y
                + typical_rank_gap * target_rank.saturating_sub(group_base_rank) as f64;

            let total_width = rank_nodes
                .iter()
                .filter_map(|node_id| node_index_by_id.get(node_id).copied())
                .map(|node_index| nodes[node_index].size.0)
                .sum::<f64>()
                + node_gap * rank_nodes.len().saturating_sub(1) as f64;
            let mut cursor_x = center_x - total_width * 0.5;

            for node_id in rank_nodes {
                let Some(node_index) = node_index_by_id.get(&node_id).copied() else {
                    continue;
                };
                cursor_x += nodes[node_index].size.0 * 0.5;
                nodes[node_index].center = (cursor_x, target_y);
                nodes[node_index].bounds =
                    Rect::from_center_size(nodes[node_index].center, nodes[node_index].size);
                cursor_x += nodes[node_index].size.0 * 0.5 + node_gap;
            }
        }
    }
}

fn local_cluster_ranks<V, E>(
    graph: &StableDiGraph<V, E>,
    ordered_nodes: &[NodeIndex],
) -> HashMap<NodeIndex, usize> {
    let cluster_nodes = ordered_nodes.iter().copied().collect::<HashSet<_>>();
    let mut indegree = ordered_nodes
        .iter()
        .copied()
        .map(|node| {
            let count = graph
                .neighbors_directed(node, Incoming)
                .filter(|neighbor| cluster_nodes.contains(neighbor))
                .count();
            (node, count)
        })
        .collect::<HashMap<_, _>>();
    let mut ranks = ordered_nodes
        .iter()
        .copied()
        .map(|node| (node, 0usize))
        .collect::<HashMap<_, _>>();
    let mut queue = ordered_nodes
        .iter()
        .copied()
        .filter(|node| indegree.get(node).copied().unwrap_or(0) == 0)
        .collect::<Vec<_>>();
    let mut queue_index = 0usize;

    while let Some(node) = queue.get(queue_index).copied() {
        queue_index += 1;
        let current_rank = *ranks.get(&node).unwrap_or(&0);
        for neighbor in graph.neighbors_directed(node, Outgoing) {
            if !cluster_nodes.contains(&neighbor) {
                continue;
            }
            let next_rank = current_rank + 1;
            if next_rank > *ranks.get(&neighbor).unwrap_or(&0) {
                ranks.insert(neighbor, next_rank);
            }
            if let Some(entry) = indegree.get_mut(&neighbor) {
                *entry = entry.saturating_sub(1);
                if *entry == 0 {
                    queue.push(neighbor);
                }
            }
        }
    }

    ranks
}

fn average_external_neighbor_x<V, E>(
    nodes: &[RoutedNode<NodeIndex>],
    graph: &StableDiGraph<V, E>,
    cluster_node_ids: &HashSet<NodeIndex>,
    node_index_by_id: &HashMap<NodeIndex, usize>,
    node_id: NodeIndex,
) -> f64 {
    let mut total_x = 0.0;
    let mut count = 0usize;

    for neighbor in graph.neighbors_directed(node_id, Incoming) {
        if cluster_node_ids.contains(&neighbor) {
            continue;
        }
        if let Some(node_index) = node_index_by_id.get(&neighbor).copied() {
            total_x += nodes[node_index].center.0;
            count += 1;
        }
    }

    for neighbor in graph.neighbors_directed(node_id, Outgoing) {
        if cluster_node_ids.contains(&neighbor) {
            continue;
        }
        if let Some(node_index) = node_index_by_id.get(&neighbor).copied() {
            total_x += nodes[node_index].center.0;
            count += 1;
        }
    }

    if count > 0 {
        total_x / count as f64
    } else {
        f64::NEG_INFINITY
    }
}

fn external_degree_counts<V, E>(
    graph: &StableDiGraph<V, E>,
    cluster_node_ids: &HashSet<NodeIndex>,
    node_id: NodeIndex,
) -> (usize, usize) {
    let incoming = graph
        .neighbors_directed(node_id, Incoming)
        .filter(|neighbor| !cluster_node_ids.contains(neighbor))
        .count();
    let outgoing = graph
        .neighbors_directed(node_id, Outgoing)
        .filter(|neighbor| !cluster_node_ids.contains(neighbor))
        .count();
    (incoming, outgoing)
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

    let delta_x = head_center.0 - tail_center.0;
    let delta_y = head_center.1 - tail_center.1;
    let tail_inset = anchor_inset(tail_rect);
    let head_inset = anchor_inset(head_rect);

    if delta_y.abs() >= delta_x.abs() {
        let tail = (
            head_center
                .0
                .clamp(tail_rect.min_x + tail_inset, tail_rect.max_x - tail_inset),
            if delta_y >= 0.0 {
                tail_rect.max_y
            } else {
                tail_rect.min_y
            },
        );
        let head = (
            tail_center
                .0
                .clamp(head_rect.min_x + head_inset, head_rect.max_x - head_inset),
            if delta_y >= 0.0 {
                head_rect.min_y
            } else {
                head_rect.max_y
            },
        );
        (tail, head)
    } else {
        let tail = (
            if delta_x >= 0.0 {
                tail_rect.max_x
            } else {
                tail_rect.min_x
            },
            head_center
                .1
                .clamp(tail_rect.min_y + tail_inset, tail_rect.max_y - tail_inset),
        );
        let head = (
            if delta_x >= 0.0 {
                head_rect.min_x
            } else {
                head_rect.max_x
            },
            tail_center
                .1
                .clamp(head_rect.min_y + head_inset, head_rect.max_y - head_inset),
        );
        (tail, head)
    }
}

fn compute_edge_anchors<L>(
    edge_drafts: &[RoutedEdgeDraft<L>],
    node_bounds: &HashMap<NodeIndex, Rect>,
    cluster_memberships: &[(usize, Rect, HashSet<NodeIndex>)],
) -> HashMap<EdgeIndex, ((f64, f64), (f64, f64))> {
    let mut edge_sides = HashMap::new();
    let mut source_groups = HashMap::<(NodeIndex, AnchorSide), Vec<(EdgeIndex, f64)>>::new();
    let mut target_groups = HashMap::<(NodeIndex, AnchorSide), Vec<(EdgeIndex, f64)>>::new();

    for edge in edge_drafts {
        let Some(tail_rect) = node_bounds.get(&edge.tail).copied() else {
            continue;
        };
        let Some(head_rect) = node_bounds.get(&edge.head).copied() else {
            continue;
        };

        let tail_center = rect_center(tail_rect);
        let head_center = rect_center(head_rect);
        let (tail_side, head_side) = preferred_anchor_sides_with_context(
            edge.tail,
            edge.head,
            tail_rect,
            head_rect,
            node_bounds,
            cluster_memberships,
        );
        edge_sides.insert(edge.id, (tail_side, head_side));
        source_groups
            .entry((edge.tail, tail_side))
            .or_default()
            .push((edge.id, anchor_sort_key(tail_side, head_center)));
        target_groups
            .entry((edge.head, head_side))
            .or_default()
            .push((edge.id, anchor_sort_key(head_side, tail_center)));
    }

    let mut anchors = HashMap::new();
    for ((node_id, side), mut entries) in source_groups {
        let Some(rect) = node_bounds.get(&node_id).copied() else {
            continue;
        };
        entries.sort_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        let points = distributed_side_points(rect, side, entries.len());
        for ((edge_id, _), point) in entries.into_iter().zip(points.into_iter()) {
            anchors.entry(edge_id).or_insert((point, point)).0 = point;
        }
    }

    for ((node_id, side), mut entries) in target_groups {
        let Some(rect) = node_bounds.get(&node_id).copied() else {
            continue;
        };
        entries.sort_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        let points = distributed_side_points(rect, side, entries.len());
        for ((edge_id, _), point) in entries.into_iter().zip(points.into_iter()) {
            anchors.entry(edge_id).or_insert((point, point)).1 = point;
        }
    }

    for edge in edge_drafts {
        if let (Some(tail_rect), Some(head_rect)) = (
            node_bounds.get(&edge.tail).copied(),
            node_bounds.get(&edge.head).copied(),
        ) {
            anchors
                .entry(edge.id)
                .or_insert_with(|| anchor_points(tail_rect, head_rect));
        }
    }

    anchors
}

fn rect_center(rect: Rect) -> (f64, f64) {
    (
        (rect.min_x + rect.max_x) * 0.5,
        (rect.min_y + rect.max_y) * 0.5,
    )
}

fn compute_edge_anchors_from_routes<L>(
    edge_drafts: &[RoutedEdgeDraft<L>],
    node_bounds: &HashMap<NodeIndex, Rect>,
) -> HashMap<EdgeIndex, ((f64, f64), (f64, f64))> {
    let mut source_groups = HashMap::<(NodeIndex, AnchorSide), Vec<(EdgeIndex, f64)>>::new();
    let mut target_groups = HashMap::<(NodeIndex, AnchorSide), Vec<(EdgeIndex, f64)>>::new();

    for edge in edge_drafts {
        let Some(tail_rect) = node_bounds.get(&edge.tail).copied() else {
            continue;
        };
        let Some(head_rect) = node_bounds.get(&edge.head).copied() else {
            continue;
        };
        let route_points = simplify_polyline(&edge.points);
        if route_points.len() < 2 {
            continue;
        }

        let tail_anchor = route_points[0];
        let head_anchor = route_points[route_points.len() - 1];
        let tail_side = anchor_side_for_point(tail_rect, tail_anchor);
        let head_side = anchor_side_for_point(head_rect, head_anchor);

        source_groups
            .entry((edge.tail, tail_side))
            .or_default()
            .push((
                edge.id,
                route_anchor_order_key(&route_points, tail_side, true),
            ));
        target_groups
            .entry((edge.head, head_side))
            .or_default()
            .push((
                edge.id,
                route_anchor_order_key(&route_points, head_side, false),
            ));
    }

    let mut anchors = HashMap::new();
    for ((node_id, side), mut entries) in source_groups {
        let Some(rect) = node_bounds.get(&node_id).copied() else {
            continue;
        };
        entries.sort_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        let points = distributed_side_points(rect, side, entries.len());
        for ((edge_id, _), point) in entries.into_iter().zip(points.into_iter()) {
            anchors.entry(edge_id).or_insert((point, point)).0 = point;
        }
    }

    for ((node_id, side), mut entries) in target_groups {
        let Some(rect) = node_bounds.get(&node_id).copied() else {
            continue;
        };
        entries.sort_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        let points = distributed_side_points(rect, side, entries.len());
        for ((edge_id, _), point) in entries.into_iter().zip(points.into_iter()) {
            anchors.entry(edge_id).or_insert((point, point)).1 = point;
        }
    }

    for edge in edge_drafts {
        if let (Some(tail_rect), Some(head_rect)) = (
            node_bounds.get(&edge.tail).copied(),
            node_bounds.get(&edge.head).copied(),
        ) {
            anchors
                .entry(edge.id)
                .or_insert_with(|| anchor_points(tail_rect, head_rect));
        }
    }

    anchors
}

fn route_anchor_order_key(points: &[(f64, f64)], side: AnchorSide, from_tail: bool) -> f64 {
    let anchor = if from_tail {
        points[0]
    } else {
        points[points.len() - 1]
    };
    let anchor_value = match side {
        AnchorSide::Top | AnchorSide::Bottom => anchor.0,
        AnchorSide::Left | AnchorSide::Right => anchor.1,
    };

    let iter: Box<dyn Iterator<Item = &(f64, f64)>> = if from_tail {
        Box::new(points.iter().skip(1))
    } else {
        Box::new(points[..points.len() - 1].iter().rev())
    };

    for point in iter {
        let candidate_value = match side {
            AnchorSide::Top | AnchorSide::Bottom => point.0,
            AnchorSide::Left | AnchorSide::Right => point.1,
        };
        if (candidate_value - anchor_value).abs() > EPSILON {
            return (anchor_value + candidate_value) * 0.5;
        }
    }

    anchor_value
}

fn anchor_side_for_point(rect: Rect, point: (f64, f64)) -> AnchorSide {
    let distances = [
        (AnchorSide::Top, (point.1 - rect.min_y).abs()),
        (AnchorSide::Bottom, (point.1 - rect.max_y).abs()),
        (AnchorSide::Left, (point.0 - rect.min_x).abs()),
        (AnchorSide::Right, (point.0 - rect.max_x).abs()),
    ];
    distances
        .into_iter()
        .min_by(|left, right| left.1.total_cmp(&right.1))
        .map(|(side, _)| side)
        .unwrap_or(AnchorSide::Bottom)
}

fn anchor_side_vector(side: AnchorSide) -> (f64, f64) {
    match side {
        AnchorSide::Top => (0.0, -1.0),
        AnchorSide::Bottom => (0.0, 1.0),
        AnchorSide::Left => (-1.0, 0.0),
        AnchorSide::Right => (1.0, 0.0),
    }
}

fn preferred_anchor_sides_with_context(
    tail: NodeIndex,
    head: NodeIndex,
    tail_rect: Rect,
    head_rect: Rect,
    node_bounds: &HashMap<NodeIndex, Rect>,
    cluster_memberships: &[(usize, Rect, HashSet<NodeIndex>)],
) -> (AnchorSide, AnchorSide) {
    let tail_center = rect_center(tail_rect);
    let head_center = rect_center(head_rect);
    let delta_x = head_center.0 - tail_center.0;
    let delta_y = head_center.1 - tail_center.1;
    let same_rank_threshold =
        ((tail_rect.max_y - tail_rect.min_y).min(head_rect.max_y - head_rect.min_y) * 0.3)
            .max(12.0);

    if delta_y.abs() <= same_rank_threshold
        && same_rank_intermediate_blocker(tail, head, tail_rect, head_rect, node_bounds)
    {
        return if delta_x < 0.0 {
            (AnchorSide::Top, AnchorSide::Top)
        } else {
            (AnchorSide::Bottom, AnchorSide::Bottom)
        };
    }

    let tail_clusters = node_clusters_by_area(tail, cluster_memberships);
    let head_clusters = node_clusters_by_area(head, cluster_memberships);
    if let (Some((tail_leaf_index, _)), Some((head_leaf_index, _)), Some(_)) = (
        tail_clusters.first().copied(),
        head_clusters.first().copied(),
        smallest_common_cluster(tail, head, cluster_memberships),
    ) {
        if tail_leaf_index != head_leaf_index && delta_y.abs() > same_rank_threshold {
            return (
                if delta_y >= 0.0 {
                    AnchorSide::Bottom
                } else {
                    AnchorSide::Top
                },
                if delta_y >= 0.0 {
                    AnchorSide::Top
                } else {
                    AnchorSide::Bottom
                },
            );
        }
    }

    if delta_y.abs() > same_rank_threshold {
        return (
            if delta_y >= 0.0 {
                AnchorSide::Bottom
            } else {
                AnchorSide::Top
            },
            if delta_y >= 0.0 {
                AnchorSide::Top
            } else {
                AnchorSide::Bottom
            },
        );
    }

    if delta_y.abs() >= delta_x.abs() {
        (
            if delta_y >= 0.0 {
                AnchorSide::Bottom
            } else {
                AnchorSide::Top
            },
            if delta_y >= 0.0 {
                AnchorSide::Top
            } else {
                AnchorSide::Bottom
            },
        )
    } else {
        (
            if delta_x >= 0.0 {
                AnchorSide::Right
            } else {
                AnchorSide::Left
            },
            if delta_x >= 0.0 {
                AnchorSide::Left
            } else {
                AnchorSide::Right
            },
        )
    }
}

fn same_rank_intermediate_blocker(
    tail: NodeIndex,
    head: NodeIndex,
    tail_rect: Rect,
    head_rect: Rect,
    node_bounds: &HashMap<NodeIndex, Rect>,
) -> bool {
    let tail_center = rect_center(tail_rect);
    let head_center = rect_center(head_rect);
    let min_x = tail_center.0.min(head_center.0);
    let max_x = tail_center.0.max(head_center.0);
    let row_center_y = (tail_center.1 + head_center.1) * 0.5;
    let rank_band = ((tail_rect.max_y - tail_rect.min_y).min(head_rect.max_y - head_rect.min_y)
        * 0.35)
        .max(16.0);

    node_bounds.iter().any(|(node_id, rect)| {
        if *node_id == tail || *node_id == head {
            return false;
        }
        let center = rect_center(*rect);
        center.0 > min_x + EPSILON
            && center.0 < max_x - EPSILON
            && (center.1 - row_center_y).abs() <= rank_band
    })
}

fn anchor_sort_key(side: AnchorSide, other_center: (f64, f64)) -> f64 {
    match side {
        AnchorSide::Top | AnchorSide::Bottom => other_center.0,
        AnchorSide::Left | AnchorSide::Right => other_center.1,
    }
}

fn distributed_side_points(rect: Rect, side: AnchorSide, count: usize) -> Vec<(f64, f64)> {
    if count == 0 {
        return Vec::new();
    }

    let inset = anchor_inset(rect);
    let span = match side {
        AnchorSide::Top | AnchorSide::Bottom => (rect.min_x + inset, rect.max_x - inset),
        AnchorSide::Left | AnchorSide::Right => (rect.min_y + inset, rect.max_y - inset),
    };

    let positions = if count == 1 || span.1 <= span.0 + EPSILON {
        vec![(span.0 + span.1) * 0.5]
    } else {
        (0..count)
            .map(|index| {
                let t = index as f64 / (count as f64 - 1.0);
                let eased = 0.5 + (t - 0.5) * 0.8;
                let clamped = eased.clamp(0.0, 1.0);
                span.0 + (span.1 - span.0) * clamped
            })
            .collect()
    };

    positions
        .into_iter()
        .map(|position| match side {
            AnchorSide::Top => (position, rect.min_y),
            AnchorSide::Bottom => (position, rect.max_y),
            AnchorSide::Left => (rect.min_x, position),
            AnchorSide::Right => (rect.max_x, position),
        })
        .collect()
}

fn anchor_inset(rect: Rect) -> f64 {
    let width = (rect.max_x - rect.min_x).max(0.0);
    let height = (rect.max_y - rect.min_y).max(0.0);
    (width.min(height) * 0.12).clamp(6.0, 12.0)
}

fn route_edge_with_cluster_entries(
    start: (f64, f64),
    end: (f64, f64),
    tail: NodeIndex,
    head: NodeIndex,
    node_obstacles: &[(NodeIndex, Rect)],
    cluster_routing_obstacles: &[(usize, Rect)],
    label_obstacles: &[Rect],
    cluster_memberships: &[(usize, Rect, HashSet<NodeIndex>)],
    bend_penalty: f64,
) -> Vec<(f64, f64)> {
    if let Some(forced_route) = forced_top_descendant_route(
        start,
        end,
        tail,
        head,
        node_obstacles,
        cluster_routing_obstacles,
        label_obstacles,
        cluster_memberships,
    ) {
        return forced_route;
    }

    if let Some(forced_route) = forced_outer_descendant_route(
        start,
        end,
        tail,
        head,
        node_obstacles,
        cluster_routing_obstacles,
        label_obstacles,
        cluster_memberships,
    ) {
        return forced_route;
    }

    if let Some(forced_route) = forced_inner_descendant_route(
        start,
        end,
        tail,
        head,
        node_obstacles,
        cluster_routing_obstacles,
        label_obstacles,
        cluster_memberships,
    ) {
        return forced_route;
    }

    if let Some(forced_route) = forced_same_rank_backward_route(
        start,
        end,
        tail,
        head,
        node_obstacles,
        cluster_routing_obstacles,
        label_obstacles,
        cluster_memberships,
    ) {
        return forced_route;
    }

    let shared_clusters = cluster_memberships
        .iter()
        .filter_map(|(cluster_index, _, node_ids)| {
            (node_ids.contains(&tail) && node_ids.contains(&head)).then_some(*cluster_index)
        })
        .collect::<HashSet<_>>();
    let mut tail_only_clusters = cluster_memberships
        .iter()
        .filter_map(|(cluster_index, bounds, node_ids)| {
            (node_ids.contains(&tail) && !node_ids.contains(&head)).then_some((
                *cluster_index,
                *bounds,
                rect_area(*bounds),
            ))
        })
        .collect::<Vec<_>>();
    tail_only_clusters.sort_by(|left, right| left.2.total_cmp(&right.2));

    let mut head_only_clusters = cluster_memberships
        .iter()
        .filter_map(|(cluster_index, bounds, node_ids)| {
            (node_ids.contains(&head) && !node_ids.contains(&tail)).then_some((
                *cluster_index,
                *bounds,
                rect_area(*bounds),
            ))
        })
        .collect::<Vec<_>>();
    head_only_clusters.sort_by(|left, right| right.2.total_cmp(&left.2));

    let mut excluded_clusters = shared_clusters.iter().copied().collect::<Vec<_>>();
    let mut current = start;
    let mut path = Vec::new();

    for guide in preferred_route_guides(start, end, tail, head, cluster_memberships) {
        let segment = route_polyline_with_context(
            current,
            guide,
            node_obstacles,
            cluster_routing_obstacles,
            label_obstacles,
            Some((tail, head)),
            &excluded_clusters,
            bend_penalty,
        );
        append_polyline_segment(&mut path, &segment);
        current = guide;
    }

    for (cluster_index, bounds, _) in tail_only_clusters {
        let portal = cluster_exit_portal(bounds, current, end);
        let segment = route_polyline_with_context(
            current,
            portal,
            node_obstacles,
            cluster_routing_obstacles,
            label_obstacles,
            Some((tail, head)),
            &excluded_clusters,
            bend_penalty,
        );
        append_polyline_segment(&mut path, &segment);
        current = portal;
        excluded_clusters.push(cluster_index);
    }

    for (cluster_index, bounds, _) in head_only_clusters {
        let portal = cluster_entry_portal(bounds, current, end);
        let segment = route_polyline_with_context(
            current,
            portal,
            node_obstacles,
            cluster_routing_obstacles,
            label_obstacles,
            Some((tail, head)),
            &excluded_clusters,
            bend_penalty,
        );
        append_polyline_segment(&mut path, &segment);
        current = portal;
        excluded_clusters.push(cluster_index);
    }

    let segment = route_polyline_with_context(
        current,
        end,
        node_obstacles,
        cluster_routing_obstacles,
        label_obstacles,
        Some((tail, head)),
        &excluded_clusters,
        bend_penalty,
    );
    append_polyline_segment(&mut path, &segment);

    let path = if path.is_empty() {
        vec![start, end]
    } else {
        path
    };
    compact_route_polyline_with_context(
        &path,
        node_obstacles,
        cluster_routing_obstacles,
        label_obstacles,
        Some((tail, head)),
        &excluded_clusters,
    )
}

fn forced_top_descendant_route(
    start: (f64, f64),
    end: (f64, f64),
    tail: NodeIndex,
    head: NodeIndex,
    node_obstacles: &[(NodeIndex, Rect)],
    cluster_routing_obstacles: &[(usize, Rect)],
    label_obstacles: &[Rect],
    cluster_memberships: &[(usize, Rect, HashSet<NodeIndex>)],
) -> Option<Vec<(f64, f64)>> {
    if !node_clusters_by_area(tail, cluster_memberships).is_empty() {
        return None;
    }

    let head_clusters = node_clusters_by_area(head, cluster_memberships);
    let (_, leaf_bounds) = head_clusters.first().copied()?;
    let (_, outer_bounds) = head_clusters.last().copied()?;
    let outer_height = (outer_bounds.max_y - outer_bounds.min_y).max(EPSILON);
    if end.1 > outer_bounds.min_y + outer_height * 0.34 {
        return None;
    }

    let outer_center_x = (outer_bounds.min_x + outer_bounds.max_x) * 0.5;
    let leaf_center_x = (leaf_bounds.min_x + leaf_bounds.max_x) * 0.5;
    if leaf_center_x >= outer_center_x - EDGE_LABEL_OFFSET * 0.2 {
        return None;
    }

    let depth_ratio = ((end.1 - outer_bounds.min_y) / outer_height).clamp(0.0, 1.0);
    let top_y = outer_bounds.min_y - CLUSTER_LABEL_RESERVED_HEIGHT.max(EDGE_LABEL_OFFSET);
    let outer_lane_y = top_y + depth_ratio * EDGE_LABEL_OFFSET * 0.35;
    let outer_x = if leaf_center_x < outer_center_x {
        outer_bounds.min_x - EDGE_LABEL_OBSTACLE_WIDTH * (0.18 + depth_ratio * 0.24)
    } else {
        outer_bounds.max_x + EDGE_LABEL_OBSTACLE_WIDTH * (0.18 + depth_ratio * 0.24)
    };
    let forced = simplify_polyline(&[
        start,
        (start.0, outer_lane_y),
        (outer_x, outer_lane_y),
        (outer_x, end.1),
        end,
    ]);
    if !polyline_clear_with_context(
        &forced,
        node_obstacles,
        cluster_routing_obstacles,
        label_obstacles,
        Some((tail, head)),
        &[],
    ) {
        return None;
    }

    Some(forced)
}

fn forced_outer_descendant_route(
    start: (f64, f64),
    end: (f64, f64),
    tail: NodeIndex,
    head: NodeIndex,
    node_obstacles: &[(NodeIndex, Rect)],
    cluster_routing_obstacles: &[(usize, Rect)],
    label_obstacles: &[Rect],
    cluster_memberships: &[(usize, Rect, HashSet<NodeIndex>)],
) -> Option<Vec<(f64, f64)>> {
    let tail_clusters = node_clusters_by_area(tail, cluster_memberships);
    if !tail_clusters.is_empty() {
        return None;
    }
    let head_clusters = node_clusters_by_area(head, cluster_memberships);
    let (leaf_index, leaf_bounds) = head_clusters.first().copied()?;
    let (_, outer_bounds) = head_clusters.last().copied()?;
    let outer_center_x = (outer_bounds.min_x + outer_bounds.max_x) * 0.5;
    let leaf_center_x = (leaf_bounds.min_x + leaf_bounds.max_x) * 0.5;
    let leaf_center_y = (leaf_bounds.min_y + leaf_bounds.max_y) * 0.5;
    let leaf_height = (leaf_bounds.max_y - leaf_bounds.min_y).max(EPSILON);
    let leaf_depth_ratio = ((end.1 - leaf_bounds.min_y) / leaf_height).clamp(0.0, 1.0);
    if leaf_center_x <= outer_center_x + EDGE_LABEL_OFFSET * 0.2
        || end.1 <= leaf_center_y + EPSILON
        || leaf_depth_ratio < 0.62
    {
        return None;
    }

    let outer_x = outer_bounds.max_x + EDGE_LABEL_OBSTACLE_WIDTH * 0.35;
    let forced = vec![start, (outer_x, start.1), (outer_x, end.1), end];
    Some(compact_route_polyline_with_context(
        &forced,
        node_obstacles,
        cluster_routing_obstacles,
        label_obstacles,
        Some((tail, head)),
        &[leaf_index],
    ))
}

fn forced_inner_descendant_route(
    start: (f64, f64),
    end: (f64, f64),
    tail: NodeIndex,
    head: NodeIndex,
    node_obstacles: &[(NodeIndex, Rect)],
    cluster_routing_obstacles: &[(usize, Rect)],
    label_obstacles: &[Rect],
    cluster_memberships: &[(usize, Rect, HashSet<NodeIndex>)],
) -> Option<Vec<(f64, f64)>> {
    let tail_clusters = node_clusters_by_area(tail, cluster_memberships);
    if !tail_clusters.is_empty() {
        return None;
    }
    let head_clusters = node_clusters_by_area(head, cluster_memberships);
    let (_, leaf_bounds) = head_clusters.first().copied()?;
    let (_, outer_bounds) = head_clusters.last().copied()?;
    let outer_center_x = (outer_bounds.min_x + outer_bounds.max_x) * 0.5;
    let leaf_center_x = (leaf_bounds.min_x + leaf_bounds.max_x) * 0.5;
    let leaf_center_y = (leaf_bounds.min_y + leaf_bounds.max_y) * 0.5;
    if leaf_center_x >= outer_center_x - EDGE_LABEL_OFFSET * 0.2 || end.1 <= leaf_center_y + EPSILON
    {
        return None;
    }

    let corridor_x = cluster_memberships
        .iter()
        .filter_map(|(_, bounds, node_ids)| {
            if node_ids.contains(&head) {
                return None;
            }
            let inside_outer = bounds.min_x >= outer_bounds.min_x - EPSILON
                && bounds.max_x <= outer_bounds.max_x + EPSILON
                && bounds.min_y >= outer_bounds.min_y - EPSILON
                && bounds.max_y <= outer_bounds.max_y + EPSILON;
            if !inside_outer {
                return None;
            }
            let sibling_center_x = (bounds.min_x + bounds.max_x) * 0.5;
            if leaf_center_x < outer_center_x && sibling_center_x > leaf_center_x + EPSILON {
                Some((leaf_bounds.max_x + bounds.min_x) * 0.5)
            } else {
                None
            }
        })
        .min_by(|left, right| {
            (left - outer_center_x)
                .abs()
                .total_cmp(&(right - outer_center_x).abs())
        })?;

    let lane_y = ((start.1 + end.1) * 0.5).clamp(
        outer_bounds.min_y + ROUTING_GUIDE_CLEARANCE,
        outer_bounds.max_y - ROUTING_GUIDE_CLEARANCE,
    );
    let forced = vec![
        start,
        (corridor_x, start.1),
        (corridor_x, lane_y),
        (corridor_x, end.1),
        end,
    ];
    Some(compact_route_polyline_with_context(
        &forced,
        node_obstacles,
        cluster_routing_obstacles,
        label_obstacles,
        Some((tail, head)),
        &[],
    ))
}

fn forced_same_rank_backward_route(
    start: (f64, f64),
    end: (f64, f64),
    tail: NodeIndex,
    head: NodeIndex,
    node_obstacles: &[(NodeIndex, Rect)],
    cluster_routing_obstacles: &[(usize, Rect)],
    label_obstacles: &[Rect],
    cluster_memberships: &[(usize, Rect, HashSet<NodeIndex>)],
) -> Option<Vec<(f64, f64)>> {
    if end.0 >= start.0 - EPSILON || (start.1 - end.1).abs() > EDGE_LABEL_OBSTACLE_HEIGHT {
        return None;
    }
    if !same_rank_obstacle_between_points(start, end, tail, head, node_obstacles) {
        return None;
    }

    let (_, common_bounds) = smallest_common_cluster(tail, head, cluster_memberships)?;
    let lane_y = (start.1.min(end.1) - EDGE_LABEL_OFFSET * 2.4).clamp(
        common_bounds.min_y + ROUTING_GUIDE_CLEARANCE,
        common_bounds.max_y - ROUTING_GUIDE_CLEARANCE,
    );
    let forced = vec![start, (start.0, lane_y), (end.0, lane_y), end];
    Some(compact_route_polyline_with_context(
        &forced,
        node_obstacles,
        cluster_routing_obstacles,
        label_obstacles,
        Some((tail, head)),
        &[],
    ))
}

fn same_rank_obstacle_between_points(
    start: (f64, f64),
    end: (f64, f64),
    tail: NodeIndex,
    head: NodeIndex,
    node_obstacles: &[(NodeIndex, Rect)],
) -> bool {
    let min_x = start.0.min(end.0);
    let max_x = start.0.max(end.0);
    let row_y = (start.1 + end.1) * 0.5;

    node_obstacles.iter().any(|(node_id, rect)| {
        if *node_id == tail || *node_id == head {
            return false;
        }
        let center = rect_center(*rect);
        center.0 > min_x + EPSILON
            && center.0 < max_x - EPSILON
            && (center.1 - row_y).abs() <= (rect.max_y - rect.min_y) * 0.5 + EDGE_LABEL_OFFSET
    })
}

fn preferred_route_guides(
    start: (f64, f64),
    end: (f64, f64),
    tail: NodeIndex,
    head: NodeIndex,
    cluster_memberships: &[(usize, Rect, HashSet<NodeIndex>)],
) -> Vec<(f64, f64)> {
    let tail_clusters = node_clusters_by_area(tail, cluster_memberships);
    let head_clusters = node_clusters_by_area(head, cluster_memberships);
    let mut guides = Vec::new();

    if tail_clusters.is_empty() {
        if let (Some((_, leaf_bounds)), Some((_, outer_bounds))) = (
            head_clusters.first().copied(),
            head_clusters.last().copied(),
        ) {
            let top_y = outer_bounds.min_y - CLUSTER_LABEL_RESERVED_HEIGHT.max(EDGE_LABEL_OFFSET);
            let outer_center_x = (outer_bounds.min_x + outer_bounds.max_x) * 0.5;
            let leaf_center_x = (leaf_bounds.min_x + leaf_bounds.max_x) * 0.5;
            let leaf_center_y = (leaf_bounds.min_y + leaf_bounds.max_y) * 0.5;
            let leaf_height = (leaf_bounds.max_y - leaf_bounds.min_y).max(EPSILON);
            let leaf_depth_ratio = ((end.1 - leaf_bounds.min_y) / leaf_height).clamp(0.0, 1.0);
            let outer_height = (outer_bounds.max_y - outer_bounds.min_y).max(EPSILON);
            let depth_ratio = ((end.1 - outer_bounds.min_y) / outer_height).clamp(0.0, 1.0);
            let is_top_descendant = end.1 <= outer_bounds.min_y + outer_height * 0.34;
            let sibling_gap_x = cluster_memberships
                .iter()
                .filter_map(|(_, bounds, node_ids)| {
                    if node_ids.contains(&head) {
                        return None;
                    }
                    let inside_outer = bounds.min_x >= outer_bounds.min_x - EPSILON
                        && bounds.max_x <= outer_bounds.max_x + EPSILON
                        && bounds.min_y >= outer_bounds.min_y - EPSILON
                        && bounds.max_y <= outer_bounds.max_y + EPSILON;
                    if !inside_outer {
                        return None;
                    }
                    let sibling_center_x = (bounds.min_x + bounds.max_x) * 0.5;
                    if leaf_center_x < outer_center_x && sibling_center_x > leaf_center_x + EPSILON
                    {
                        Some((leaf_bounds.max_x + bounds.min_x) * 0.5)
                    } else if leaf_center_x > outer_center_x
                        && sibling_center_x < leaf_center_x - EPSILON
                    {
                        Some((bounds.max_x + leaf_bounds.min_x) * 0.5)
                    } else {
                        None
                    }
                })
                .min_by(|left, right| {
                    (left - outer_center_x)
                        .abs()
                        .total_cmp(&(right - outer_center_x).abs())
                });

            if is_top_descendant && leaf_center_x < outer_center_x - EDGE_LABEL_OFFSET * 0.2 {
                let outer_x = if leaf_center_x < outer_center_x {
                    outer_bounds.min_x - EDGE_LABEL_OBSTACLE_WIDTH * (0.18 + depth_ratio * 0.24)
                } else {
                    outer_bounds.max_x + EDGE_LABEL_OBSTACLE_WIDTH * (0.18 + depth_ratio * 0.24)
                };
                let outer_lane_y = top_y + depth_ratio * EDGE_LABEL_OFFSET * 0.35;
                if !almost_equal(start.1, outer_lane_y) {
                    guides.push((start.0, outer_lane_y));
                }
                guides.push((outer_x, outer_lane_y));
                guides.push((outer_x, end.1));
            } else if leaf_center_x < outer_center_x - EDGE_LABEL_OFFSET * 0.2
                && end.1 > leaf_center_y + EPSILON
                && sibling_gap_x.is_some()
            {
                let corridor_x = sibling_gap_x.unwrap();
                let lane_y = ((start.1 + end.1) * 0.5).clamp(
                    outer_bounds.min_y + ROUTING_GUIDE_CLEARANCE,
                    outer_bounds.max_y - ROUTING_GUIDE_CLEARANCE,
                );
                if !almost_equal(start.1, lane_y) {
                    guides.push((start.0, lane_y));
                }
                if !almost_equal(start.0, corridor_x) {
                    guides.push((corridor_x, lane_y));
                }
                if !almost_equal(end.1, lane_y) {
                    guides.push((corridor_x, end.1));
                }
            } else if leaf_center_x > outer_center_x + EDGE_LABEL_OFFSET * 0.2
                && end.1 > leaf_center_y + EPSILON
                && leaf_depth_ratio >= 0.62
            {
                let outer_x =
                    outer_bounds.max_x + EDGE_LABEL_OBSTACLE_WIDTH * (0.2 + depth_ratio * 0.35);
                let outer_lane_y = top_y + depth_ratio * EDGE_LABEL_OFFSET * 0.5;
                if !almost_equal(start.1, outer_lane_y) {
                    guides.push((start.0, outer_lane_y));
                }
                guides.push((outer_x, outer_lane_y));
                guides.push((outer_x, end.1));
            } else if end.0 > start.0 + EDGE_LABEL_OFFSET * 0.25 {
                let inner_x = start.0 + (end.0 - start.0) * (0.84 + depth_ratio * 0.08);
                if !almost_equal(end.1, start.1) {
                    guides.push((start.0, end.1));
                }
                if !almost_equal(start.0, inner_x) {
                    guides.push((inner_x, end.1));
                }
            }
        }
    }

    let common_cluster = smallest_common_cluster(tail, head, cluster_memberships);
    if let (
        Some((tail_leaf_index, tail_leaf_bounds)),
        Some((head_leaf_index, head_leaf_bounds)),
        Some((_, common_bounds)),
    ) = (
        tail_clusters.first().copied(),
        head_clusters.first().copied(),
        common_cluster,
    ) {
        if tail_leaf_index != head_leaf_index {
            let gap_x = if tail_leaf_bounds.min_x <= head_leaf_bounds.min_x {
                (tail_leaf_bounds.max_x + head_leaf_bounds.min_x) * 0.5
            } else {
                (head_leaf_bounds.max_x + tail_leaf_bounds.min_x) * 0.5
            };
            let clamped_gap_x = gap_x.clamp(common_bounds.min_x, common_bounds.max_x);
            let span_y = (start.1 - end.1).abs();
            let delta_x = end.0 - start.0;
            let min_lane_y = common_bounds.min_y + ROUTING_GUIDE_CLEARANCE;
            let max_lane_y = common_bounds.max_y - ROUTING_GUIDE_CLEARANCE;

            if span_y <= EDGE_LABEL_OBSTACLE_HEIGHT && delta_x < -EPSILON {
                let lane_y = (start.1 - EDGE_LABEL_OFFSET * 2.4).clamp(min_lane_y, max_lane_y);
                guides.push((start.0, lane_y));
                guides.push((clamped_gap_x, lane_y));
                guides.push((end.0, lane_y));
            } else {
                let lane_y = if span_y <= EDGE_LABEL_OBSTACLE_HEIGHT {
                    if delta_x < -EPSILON {
                        start.1
                    } else {
                        start.1.min(end.1) - EDGE_LABEL_OFFSET
                    }
                } else {
                    (start.1 + end.1) * 0.5
                }
                .clamp(min_lane_y, max_lane_y);

                if !almost_equal(start.1, lane_y) {
                    guides.push((start.0, lane_y));
                }
                guides.push((clamped_gap_x, lane_y));
                if !almost_equal(end.1, lane_y) {
                    guides.push((clamped_gap_x, end.1));
                }
            }
        }
    }

    dedup_route_guides(guides)
}

fn node_clusters_by_area(
    node: NodeIndex,
    cluster_memberships: &[(usize, Rect, HashSet<NodeIndex>)],
) -> Vec<(usize, Rect)> {
    let mut clusters = cluster_memberships
        .iter()
        .filter_map(|(cluster_index, bounds, node_ids)| {
            node_ids
                .contains(&node)
                .then_some((*cluster_index, *bounds))
        })
        .collect::<Vec<_>>();
    clusters.sort_by(|left, right| rect_area(left.1).total_cmp(&rect_area(right.1)));
    clusters
}

fn smallest_common_cluster(
    tail: NodeIndex,
    head: NodeIndex,
    cluster_memberships: &[(usize, Rect, HashSet<NodeIndex>)],
) -> Option<(usize, Rect)> {
    let mut clusters = cluster_memberships
        .iter()
        .filter_map(|(cluster_index, bounds, node_ids)| {
            (node_ids.contains(&tail) && node_ids.contains(&head))
                .then_some((*cluster_index, *bounds))
        })
        .collect::<Vec<_>>();
    clusters.sort_by(|left, right| rect_area(left.1).total_cmp(&rect_area(right.1)));
    clusters.into_iter().next()
}

fn dedup_route_guides(guides: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    let mut deduped = Vec::new();
    for guide in guides {
        if deduped
            .last()
            .copied()
            .map(|last| same_point(last, guide))
            .unwrap_or(false)
        {
            continue;
        }
        deduped.push(guide);
    }
    deduped
}

fn rect_area(rect: Rect) -> f64 {
    (rect.max_x - rect.min_x).max(0.0) * (rect.max_y - rect.min_y).max(0.0)
}

fn cluster_entry_portal(bounds: Rect, from: (f64, f64), target: (f64, f64)) -> (f64, f64) {
    let center = (
        (bounds.min_x + bounds.max_x) * 0.5,
        (bounds.min_y + bounds.max_y) * 0.5,
    );
    let delta_x = from.0 - center.0;
    let delta_y = from.1 - center.1;

    if delta_y.abs() >= delta_x.abs() {
        let blended_x = target.0 + (from.0 - target.0) * 0.55;
        (
            blended_x.clamp(bounds.min_x, bounds.max_x),
            if delta_y <= 0.0 {
                bounds.min_y
            } else {
                bounds.max_y
            },
        )
    } else {
        (
            if delta_x <= 0.0 {
                bounds.min_x
            } else {
                bounds.max_x
            },
            target.1.clamp(bounds.min_y, bounds.max_y),
        )
    }
}

fn cluster_exit_portal(bounds: Rect, from: (f64, f64), target: (f64, f64)) -> (f64, f64) {
    let center = (
        (bounds.min_x + bounds.max_x) * 0.5,
        (bounds.min_y + bounds.max_y) * 0.5,
    );
    let delta_x = target.0 - center.0;
    let delta_y = target.1 - center.1;

    if delta_y.abs() >= delta_x.abs() {
        let blended_x = from.0 + (target.0 - from.0) * 0.55;
        (
            blended_x.clamp(bounds.min_x, bounds.max_x),
            if delta_y <= 0.0 {
                bounds.min_y
            } else {
                bounds.max_y
            },
        )
    } else {
        (
            if delta_x <= 0.0 {
                bounds.min_x
            } else {
                bounds.max_x
            },
            from.1.clamp(bounds.min_y, bounds.max_y),
        )
    }
}

fn append_polyline_segment(path: &mut Vec<(f64, f64)>, segment: &[(f64, f64)]) {
    for point in segment {
        if path.last().copied() != Some(*point) {
            path.push(*point);
        }
    }
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

fn cluster_label_obstacles<C>(clusters: &[ClusterLayout<C>]) -> Vec<Rect> {
    let mut obstacles = Vec::new();
    for cluster in clusters {
        let bounds = cluster.bounds;
        if bounds.max_x - bounds.min_x <= EPSILON || bounds.max_y - bounds.min_y <= EPSILON {
            continue;
        }

        if let Some(label_rect) = cluster_label_rect(bounds) {
            obstacles.push(label_rect);
        }

        let half_thickness = CLUSTER_BORDER_LABEL_OBSTACLE_THICKNESS * 0.5;
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

fn cluster_label_rect(bounds: Rect) -> Option<Rect> {
    let available_width = (bounds.max_x - bounds.min_x) - CLUSTER_LABEL_HORIZONTAL_INSET * 2.0;
    if available_width <= EPSILON {
        return None;
    }

    let label_width = available_width
        .min(CLUSTER_LABEL_OBSTACLE_MAX_WIDTH)
        .max(CLUSTER_LABEL_OBSTACLE_MIN_WIDTH)
        .min(available_width);
    if label_width <= EPSILON {
        return None;
    }

    Some(Rect {
        min_x: bounds.min_x + CLUSTER_LABEL_HORIZONTAL_INSET,
        max_x: bounds.min_x + CLUSTER_LABEL_HORIZONTAL_INSET + label_width,
        min_y: bounds.min_y + CLUSTER_LABEL_VERTICAL_GAP,
        max_y: bounds.min_y + CLUSTER_LABEL_VERTICAL_GAP + CLUSTER_LABEL_OBSTACLE_HEIGHT,
    })
}

fn reroute_edges_around_labels<L: Clone>(
    edge_drafts: &mut [RoutedEdgeDraft<L>],
    node_bounds: &HashMap<NodeIndex, Rect>,
    _cluster_memberships: &[(usize, Rect, HashSet<NodeIndex>)],
    node_label_obstacles: &[Rect],
    cluster_label_obstacles: &[Rect],
    render_config: &RenderConfig,
) {
    if edge_drafts.len() < 2 {
        return;
    }

    let node_obstacles = node_bounds
        .iter()
        .map(|(node_id, bounds)| (*node_id, bounds.expand(render_config.routing_padding)))
        .collect::<Vec<_>>();
    let mut structural_obstacles = node_label_obstacles.to_vec();
    structural_obstacles.extend_from_slice(cluster_label_obstacles);

    for _ in 0..3 {
        let label_rects = edge_drafts
            .iter()
            .map(|edge| edge.label_position.and_then(label_obstacle_rect))
            .collect::<Vec<_>>();
        let mut changed = false;

        for edge_index in 0..edge_drafts.len() {
            let all_label_obstacles = label_rects
                .iter()
                .filter_map(|rect| *rect)
                .collect::<Vec<_>>();
            let rerouted = locally_detour_polyline_around_labels(
                &edge_drafts[edge_index].points,
                &all_label_obstacles,
                &structural_obstacles,
                &node_obstacles,
                Some((edge_drafts[edge_index].tail, edge_drafts[edge_index].head)),
                render_config.routing_padding.max(6.0),
            );
            let rerouted = compact_route_polyline_with_context(
                &rerouted,
                &node_obstacles,
                &[],
                &structural_obstacles,
                Some((edge_drafts[edge_index].tail, edge_drafts[edge_index].head)),
                &[],
            );

            if rerouted != edge_drafts[edge_index].points {
                edge_drafts[edge_index].points = rerouted;
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }
}

fn locally_detour_polyline_around_labels(
    points: &[(f64, f64)],
    label_obstacles: &[Rect],
    structural_obstacles: &[Rect],
    node_obstacles: &[(NodeIndex, Rect)],
    excluded_node_obstacles: Option<(NodeIndex, NodeIndex)>,
    clearance: f64,
) -> Vec<(f64, f64)> {
    if points.len() < 2 || label_obstacles.is_empty() {
        return points.to_vec();
    }

    let mut routed = simplify_polyline(points);
    let safe_clearance = clearance.max(4.0);

    for _ in 0..12 {
        let Some((segment_index, obstacle)) = first_label_intersection(&routed, label_obstacles)
        else {
            break;
        };

        let start = routed[segment_index];
        let end = routed[segment_index + 1];
        let all_obstacles = structural_obstacles
            .iter()
            .copied()
            .chain(label_obstacles.iter().copied())
            .collect::<Vec<_>>();
        let Some(detour) = detour_segment_around_rect(
            start,
            end,
            obstacle,
            &all_obstacles,
            node_obstacles,
            excluded_node_obstacles,
            safe_clearance,
        ) else {
            break;
        };

        let mut replacement = Vec::new();
        replacement.extend_from_slice(&routed[..segment_index]);
        replacement.extend(detour);
        replacement.extend_from_slice(&routed[segment_index + 2..]);
        routed = simplify_polyline(&replacement);
    }

    routed
}

fn first_label_intersection(
    points: &[(f64, f64)],
    label_obstacles: &[Rect],
) -> Option<(usize, Rect)> {
    for (segment_index, segment) in points.windows(2).enumerate() {
        let start = segment[0];
        let end = segment[1];
        for obstacle in label_obstacles {
            if segment_intersects_rect(start, end, *obstacle) {
                return Some((segment_index, *obstacle));
            }
        }
    }

    None
}

fn segment_intersects_rect(start: (f64, f64), end: (f64, f64), rect: Rect) -> bool {
    if almost_equal(start.1, end.1) {
        let y = start.1;
        let min_x = start.0.min(end.0);
        let max_x = start.0.max(end.0);
        y > rect.min_y + EPSILON
            && y < rect.max_y - EPSILON
            && ranges_overlap(min_x, max_x, rect.min_x, rect.max_x)
    } else if almost_equal(start.0, end.0) {
        let x = start.0;
        let min_y = start.1.min(end.1);
        let max_y = start.1.max(end.1);
        x > rect.min_x + EPSILON
            && x < rect.max_x - EPSILON
            && ranges_overlap(min_y, max_y, rect.min_y, rect.max_y)
    } else {
        diagonal_segment_intersects_rect(start, end, rect)
    }
}

fn detour_segment_around_rect(
    start: (f64, f64),
    end: (f64, f64),
    rect: Rect,
    all_obstacles: &[Rect],
    node_obstacles: &[(NodeIndex, Rect)],
    excluded_node_obstacles: Option<(NodeIndex, NodeIndex)>,
    clearance: f64,
) -> Option<Vec<(f64, f64)>> {
    let mut candidates = Vec::new();

    if almost_equal(start.1, end.1) {
        let y = start.1;
        let going_right = end.0 >= start.0;
        let before_x = if going_right {
            rect.min_x - clearance
        } else {
            rect.max_x + clearance
        };
        let after_x = if going_right {
            rect.max_x + clearance
        } else {
            rect.min_x - clearance
        };

        for detour_y in [rect.min_y - clearance, rect.max_y + clearance] {
            let candidate = simplify_polyline(&[
                start,
                (before_x, y),
                (before_x, detour_y),
                (after_x, detour_y),
                (after_x, y),
                end,
            ]);
            if polyline_clear_against_rects(&candidate, all_obstacles, rect)
                && polyline_clear_against_nodes(&candidate, node_obstacles, excluded_node_obstacles)
            {
                candidates.push(candidate);
            }
        }
    } else if almost_equal(start.0, end.0) {
        let x = start.0;
        let going_down = end.1 >= start.1;
        let before_y = if going_down {
            rect.min_y - clearance
        } else {
            rect.max_y + clearance
        };
        let after_y = if going_down {
            rect.max_y + clearance
        } else {
            rect.min_y - clearance
        };

        for detour_x in [rect.min_x - clearance, rect.max_x + clearance] {
            let candidate = simplify_polyline(&[
                start,
                (x, before_y),
                (detour_x, before_y),
                (detour_x, after_y),
                (x, after_y),
                end,
            ]);
            if polyline_clear_against_rects(&candidate, all_obstacles, rect)
                && polyline_clear_against_nodes(&candidate, node_obstacles, excluded_node_obstacles)
            {
                candidates.push(candidate);
            }
        }
    }

    choose_best_detour(
        &candidates,
        all_obstacles,
        node_obstacles,
        excluded_node_obstacles,
    )
}

fn polyline_clear_against_rects(
    points: &[(f64, f64)],
    obstacles: &[Rect],
    ignored_obstacle: Rect,
) -> bool {
    for segment in points.windows(2) {
        for obstacle in obstacles {
            if *obstacle == ignored_obstacle {
                continue;
            }
            if segment_intersects_rect(segment[0], segment[1], *obstacle) {
                return false;
            }
        }
    }

    true
}

fn polyline_clear_against_nodes(
    points: &[(f64, f64)],
    node_obstacles: &[(NodeIndex, Rect)],
    excluded_node_obstacles: Option<(NodeIndex, NodeIndex)>,
) -> bool {
    for segment in points.windows(2) {
        for (node_id, obstacle) in node_obstacles {
            if is_excluded_node_obstacle(*node_id, excluded_node_obstacles) {
                continue;
            }
            if segment_intersects_rect(segment[0], segment[1], *obstacle) {
                return false;
            }
        }
    }

    true
}

fn choose_best_detour(
    candidates: &[Vec<(f64, f64)>],
    obstacles: &[Rect],
    node_obstacles: &[(NodeIndex, Rect)],
    excluded_node_obstacles: Option<(NodeIndex, NodeIndex)>,
) -> Option<Vec<(f64, f64)>> {
    candidates
        .iter()
        .max_by(|left, right| {
            polyline_min_clearance(left, obstacles, node_obstacles, excluded_node_obstacles)
                .total_cmp(&polyline_min_clearance(
                    right,
                    obstacles,
                    node_obstacles,
                    excluded_node_obstacles,
                ))
                .then_with(|| polyline_length(right).total_cmp(&polyline_length(left)))
        })
        .cloned()
}

fn polyline_min_clearance(
    points: &[(f64, f64)],
    obstacles: &[Rect],
    node_obstacles: &[(NodeIndex, Rect)],
    excluded_node_obstacles: Option<(NodeIndex, NodeIndex)>,
) -> f64 {
    let mut minimum = f64::INFINITY;

    for point in points {
        for obstacle in obstacles {
            minimum = minimum.min(point_rect_clearance(*point, *obstacle));
        }
        for (node_id, obstacle) in node_obstacles {
            if is_excluded_node_obstacle(*node_id, excluded_node_obstacles) {
                continue;
            }
            minimum = minimum.min(point_rect_clearance(*point, *obstacle));
        }
    }

    minimum
}

fn point_rect_clearance(point: (f64, f64), rect: Rect) -> f64 {
    let dx = if point.0 < rect.min_x {
        rect.min_x - point.0
    } else if point.0 > rect.max_x {
        point.0 - rect.max_x
    } else {
        0.0
    };
    let dy = if point.1 < rect.min_y {
        rect.min_y - point.1
    } else if point.1 > rect.max_y {
        point.1 - rect.max_y
    } else {
        0.0
    };

    if dx <= EPSILON {
        dy
    } else if dy <= EPSILON {
        dx
    } else {
        (dx * dx + dy * dy).sqrt()
    }
}

fn choose_edge_label_position(
    points: &[(f64, f64)],
    node_obstacles: &[Rect],
    cluster_obstacles: &[Rect],
    placed_label_obstacles: &[Rect],
) -> Option<(f64, f64)> {
    let mut best: Option<((f64, f64), f64, f64, f64)> = None;

    for position in label_position_candidates(points) {
        let Some(label_rect) = label_obstacle_rect(position) else {
            continue;
        };
        if !label_rect_clear_of_structures(label_rect, node_obstacles, cluster_obstacles) {
            continue;
        }

        let overlap = total_label_overlap_area(label_rect, placed_label_obstacles);
        let structure_clearance = label_rect_min_clearance(
            label_rect,
            node_obstacles,
            cluster_obstacles,
            placed_label_obstacles,
        );
        let route_clearance = label_rect_route_clearance(label_rect, points);
        let effective_clearance = structure_clearance.min(route_clearance);

        match best {
            Some((_, best_overlap, best_effective_clearance, best_route_clearance))
                if overlap > best_overlap + EPSILON
                    || (almost_equal(overlap, best_overlap)
                        && (effective_clearance + EPSILON < best_effective_clearance
                            || (almost_equal(effective_clearance, best_effective_clearance)
                                && route_clearance + EPSILON < best_route_clearance))) => {}
            _ => {
                best = Some((position, overlap, effective_clearance, route_clearance));
            }
        }
    }

    best.map(|(position, _, _, _)| position)
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
    for candidate in segment_label_candidates(points) {
        if !candidates
            .iter()
            .any(|existing| same_point(*existing, candidate))
        {
            candidates.push(candidate);
        }
    }

    let minimum_fraction = 0.2;
    let maximum_fraction = 0.8;
    let midpoint_distance = total_length * 0.5;
    let step = (EDGE_LABEL_OBSTACLE_WIDTH * 0.25).max(4.0);
    let max_offset = (total_length * (maximum_fraction - minimum_fraction) * 0.5).max(0.0);

    append_label_candidates(points, 0.5, &mut candidates);

    let mut offset = step;
    while offset <= max_offset + EPSILON {
        let left_distance = (midpoint_distance - offset).max(total_length * minimum_fraction);
        let right_distance = (midpoint_distance + offset).min(total_length * maximum_fraction);
        let left_fraction = left_distance / total_length;
        let right_fraction = right_distance / total_length;

        append_label_candidates(points, left_fraction, &mut candidates);
        append_label_candidates(points, right_fraction, &mut candidates);
        offset += step;
    }

    candidates
}

fn segment_label_candidates(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    if points.len() < 2 {
        return Vec::new();
    }

    let total_length = polyline_length(points);
    if total_length <= EPSILON {
        return Vec::new();
    }

    let mut traversed = 0.0;
    let mut segments = Vec::new();

    for segment in points.windows(2) {
        let start = segment[0];
        let end = segment[1];
        let length = euclidean_distance(start, end);
        if length <= EDGE_LABEL_OBSTACLE_WIDTH * 0.6 {
            traversed += length;
            continue;
        }

        let midpoint_fraction = (traversed + length * 0.5) / total_length;
        traversed += length;
        if !(0.15..=0.85).contains(&midpoint_fraction) {
            continue;
        }

        let midpoint = ((start.0 + end.0) * 0.5, (start.1 + end.1) * 0.5);
        let direction = if almost_equal(start.1, end.1) {
            MoveDir::Horizontal
        } else if almost_equal(start.0, end.0) {
            MoveDir::Vertical
        } else {
            MoveDir::None
        };
        let midpoint_bias = (1.0 - (midpoint_fraction - 0.6).abs()).clamp(0.0, 1.0);
        let direction_bias = match direction {
            MoveDir::Vertical => 1.25,
            MoveDir::Horizontal => 0.9,
            MoveDir::None => 1.0,
        };
        let score = length * (0.7 + midpoint_bias * 0.6) * direction_bias;
        segments.push((score, length, midpoint_fraction, midpoint, direction));
    }

    segments.sort_by(|left, right| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| (left.2 - 0.6).abs().total_cmp(&(right.2 - 0.6).abs()))
            .then_with(|| right.1.total_cmp(&left.1))
    });

    let mut candidates = Vec::new();
    for (_, _, _, midpoint, direction) in segments {
        match direction {
            MoveDir::Horizontal => {
                for multiplier in [1.0, 1.6, 2.2] {
                    let offset = EDGE_LABEL_OFFSET * multiplier;
                    candidates.push((midpoint.0, midpoint.1 - offset));
                    candidates.push((midpoint.0, midpoint.1 + offset));
                }
            }
            MoveDir::Vertical => {
                for multiplier in [1.0, 1.6, 2.2] {
                    let offset = EDGE_LABEL_OFFSET * multiplier;
                    candidates.push((midpoint.0 + offset, midpoint.1));
                    candidates.push((midpoint.0 - offset, midpoint.1));
                }
            }
            MoveDir::None => candidates.push(midpoint),
        }
    }

    candidates
}

fn append_label_candidates(points: &[(f64, f64)], fraction: f64, candidates: &mut Vec<(f64, f64)>) {
    for candidate in offset_label_positions(points, fraction) {
        if !candidates
            .iter()
            .any(|existing| same_point(*existing, candidate))
        {
            candidates.push(candidate);
        }
    }
}

fn offset_label_positions(points: &[(f64, f64)], fraction: f64) -> Vec<(f64, f64)> {
    let Some((point, direction)) = polyline_point_and_direction_at_fraction(points, fraction)
    else {
        return Vec::new();
    };

    let mut candidates = Vec::new();
    match direction {
        MoveDir::Horizontal => {
            for multiplier in [1.0, 1.6, 2.2] {
                let offset = EDGE_LABEL_OFFSET * multiplier;
                candidates.push((point.0, point.1 - offset));
                candidates.push((point.0, point.1 + offset));
            }
        }
        MoveDir::Vertical => {
            for multiplier in [1.0, 1.6, 2.2] {
                let offset = EDGE_LABEL_OFFSET * multiplier;
                candidates.push((point.0 + offset, point.1));
                candidates.push((point.0 - offset, point.1));
            }
        }
        MoveDir::None => {
            candidates.push(point);
        }
    }
    candidates
}

fn default_label_position(points: &[(f64, f64)]) -> Option<(f64, f64)> {
    label_position_candidates(points).into_iter().next()
}

fn resolve_curve_label_positions<T, L>(
    edges: &mut [RoutedEdge<T, L>],
    node_obstacles: &[Rect],
    cluster_obstacles: &[Rect],
) {
    let curve_routes = edges
        .iter()
        .map(|edge| sampled_curve_route(edge))
        .collect::<Vec<_>>();

    for pass in 0..3 {
        let reverse = pass % 2 == 1;
        let mut placed_label_obstacles = Vec::new();
        let indices = if reverse {
            (0..edges.len()).rev().collect::<Vec<_>>()
        } else {
            (0..edges.len()).collect::<Vec<_>>()
        };

        for index in indices {
            let Some(label) = edges[index].label.as_mut() else {
                continue;
            };
            let route = &curve_routes[index];
            let position = choose_curve_label_position(
                route,
                Some(label.position),
                &curve_routes,
                index,
                node_obstacles,
                cluster_obstacles,
                &placed_label_obstacles,
            )
            .or(Some(label.position));

            if let Some(position) = position {
                label.position = position;
                if let Some(obstacle) = label_obstacle_rect(position) {
                    placed_label_obstacles.push(obstacle);
                }
            }
        }
    }
}

fn sampled_curve_route<T, L>(edge: &RoutedEdge<T, L>) -> Vec<(f64, f64)> {
    if edge.curve_points.len() >= 4 && (edge.curve_points.len() - 1) % 3 == 0 {
        sample_bezier_curve_points(&edge.curve_points, 12)
    } else {
        simplify_polyline(&edge.points)
    }
}

fn sample_bezier_curve_points(
    curve_points: &[(f64, f64)],
    samples_per_segment: usize,
) -> Vec<(f64, f64)> {
    let mut samples = Vec::new();
    let steps = samples_per_segment.max(2);

    if let Some(&start) = curve_points.first() {
        samples.push(start);
    }

    for chunk in curve_points[1..].chunks(3) {
        if chunk.len() != 3 {
            break;
        }
        let p0 = *samples.last().unwrap_or(&curve_points[0]);
        let p1 = chunk[0];
        let p2 = chunk[1];
        let p3 = chunk[2];

        for step in 1..=steps {
            let t = step as f64 / steps as f64;
            let mt = 1.0 - t;
            let point = (
                mt * mt * mt * p0.0
                    + 3.0 * mt * mt * t * p1.0
                    + 3.0 * mt * t * t * p2.0
                    + t * t * t * p3.0,
                mt * mt * mt * p0.1
                    + 3.0 * mt * mt * t * p1.1
                    + 3.0 * mt * t * t * p2.1
                    + t * t * t * p3.1,
            );
            push_unique_point(&mut samples, point);
        }
    }

    simplify_polyline(&samples)
}

fn choose_curve_label_position(
    points: &[(f64, f64)],
    existing_position: Option<(f64, f64)>,
    all_routes: &[Vec<(f64, f64)>],
    edge_index: usize,
    node_obstacles: &[Rect],
    cluster_obstacles: &[Rect],
    placed_label_obstacles: &[Rect],
) -> Option<(f64, f64)> {
    let mut best: Option<((f64, f64), f64, f64, f64)> = None;

    for position in curve_label_candidates_with_existing(points, existing_position) {
        let Some(label_rect) = label_obstacle_rect(position) else {
            continue;
        };
        if !label_rect_clear_of_structures(label_rect, node_obstacles, cluster_obstacles) {
            continue;
        }

        let overlap = total_label_overlap_area(label_rect, placed_label_obstacles);
        let structure_clearance = label_rect_min_clearance(
            label_rect,
            node_obstacles,
            cluster_obstacles,
            placed_label_obstacles,
        );
        let route_clearance = all_routes
            .iter()
            .enumerate()
            .filter_map(|(index, route)| (index != edge_index).then_some(route))
            .map(|route| label_rect_route_clearance(label_rect, route))
            .fold(label_rect_route_clearance(label_rect, points), f64::min);
        let effective_clearance = structure_clearance.min(route_clearance);

        match best {
            Some((_, best_overlap, best_effective_clearance, best_route_clearance))
                if overlap > best_overlap + EPSILON
                    || (almost_equal(overlap, best_overlap)
                        && (effective_clearance + EPSILON < best_effective_clearance
                            || (almost_equal(effective_clearance, best_effective_clearance)
                                && route_clearance + EPSILON < best_route_clearance))) => {}
            _ => best = Some((position, overlap, effective_clearance, route_clearance)),
        }
    }

    best.map(|(position, _, _, _)| position)
}

fn curve_label_candidates(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    curve_label_candidates_with_existing(points, None)
}

fn curve_label_candidates_with_existing(
    points: &[(f64, f64)],
    existing_position: Option<(f64, f64)>,
) -> Vec<(f64, f64)> {
    if points.len() < 2 {
        return points.first().copied().into_iter().collect();
    }

    let mut candidates = Vec::new();
    if let Some(existing_position) = existing_position {
        candidates.push(existing_position);
    }
    for candidate in curve_segment_label_candidates(points) {
        if !candidates
            .iter()
            .any(|existing| same_point(*existing, candidate))
        {
            candidates.push(candidate);
        }
    }
    for fraction in [0.5, 0.42, 0.58, 0.34, 0.66, 0.26, 0.74] {
        let Some((point, tangent)) = polyline_point_and_tangent_at_fraction(points, fraction)
        else {
            continue;
        };
        let normal = (-tangent.1, tangent.0);
        for multiplier in [1.0, 1.6, 2.2, 2.8] {
            let offset = EDGE_LABEL_OFFSET * multiplier;
            for sign in [-1.0, 1.0] {
                let candidate = (
                    point.0 + normal.0 * offset * sign,
                    point.1 + normal.1 * offset * sign,
                );
                if !candidates
                    .iter()
                    .any(|existing| same_point(*existing, candidate))
                {
                    candidates.push(candidate);
                }
            }
        }
    }
    candidates
}

fn curve_segment_label_candidates(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    if points.len() < 2 {
        return Vec::new();
    }

    let total_length = polyline_length(points);
    if total_length <= EPSILON {
        return Vec::new();
    }

    let mut traversed = 0.0;
    let mut segments = Vec::new();

    for segment in points.windows(2) {
        let start = segment[0];
        let end = segment[1];
        let length = euclidean_distance(start, end);
        if length <= EDGE_LABEL_OBSTACLE_WIDTH * 0.4 {
            traversed += length;
            continue;
        }

        let midpoint_fraction = (traversed + length * 0.5) / total_length;
        traversed += length;
        if !(0.12..=0.88).contains(&midpoint_fraction) {
            continue;
        }

        let midpoint = ((start.0 + end.0) * 0.5, (start.1 + end.1) * 0.5);
        let tangent = unit_direction(start, end);
        if tangent == (0.0, 0.0) {
            continue;
        }
        let midpoint_bias = (1.0 - (midpoint_fraction - 0.55).abs()).clamp(0.0, 1.0);
        let score = length * (0.8 + midpoint_bias * 0.5);
        segments.push((score, midpoint, tangent));
    }

    segments.sort_by(|left, right| right.0.total_cmp(&left.0));

    let mut candidates = Vec::new();
    for (_, midpoint, tangent) in segments {
        let normal = (-tangent.1, tangent.0);
        for multiplier in [1.0, 1.5, 2.0, 2.6] {
            let offset = EDGE_LABEL_OFFSET * multiplier;
            for sign in [-1.0, 1.0] {
                let candidate = (
                    midpoint.0 + normal.0 * offset * sign,
                    midpoint.1 + normal.1 * offset * sign,
                );
                if !candidates
                    .iter()
                    .any(|existing| same_point(*existing, candidate))
                {
                    candidates.push(candidate);
                }
            }
        }
    }

    candidates
}

fn polyline_point_and_tangent_at_fraction(
    points: &[(f64, f64)],
    fraction: f64,
) -> Option<((f64, f64), (f64, f64))> {
    if points.len() < 2 {
        return None;
    }

    let total_length = polyline_length(points);
    if total_length <= EPSILON {
        return None;
    }

    let clamped = fraction.clamp(0.0, 1.0);
    let target_length = total_length * clamped;
    let mut traversed = 0.0;

    for segment in points.windows(2) {
        let start = segment[0];
        let end = segment[1];
        let length = euclidean_distance(start, end);
        if traversed + length >= target_length {
            if length <= EPSILON {
                break;
            }
            let t = (target_length - traversed) / length;
            let tangent = unit_direction(start, end);
            if tangent == (0.0, 0.0) {
                break;
            }
            return Some((
                (
                    start.0 + (end.0 - start.0) * t,
                    start.1 + (end.1 - start.1) * t,
                ),
                tangent,
            ));
        }
        traversed += length;
    }

    None
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

fn label_rect_min_clearance(
    label_rect: Rect,
    node_obstacles: &[Rect],
    cluster_obstacles: &[Rect],
    placed_label_obstacles: &[Rect],
) -> f64 {
    node_obstacles
        .iter()
        .chain(cluster_obstacles.iter())
        .chain(placed_label_obstacles.iter())
        .map(|obstacle| rect_rect_clearance(label_rect, *obstacle))
        .fold(f64::INFINITY, f64::min)
}

fn label_rect_route_clearance(label_rect: Rect, points: &[(f64, f64)]) -> f64 {
    points
        .windows(2)
        .map(|segment| segment_rect_clearance(segment[0], segment[1], label_rect))
        .fold(f64::INFINITY, f64::min)
}

fn rect_rect_clearance(a: Rect, b: Rect) -> f64 {
    let dx = if a.max_x < b.min_x {
        b.min_x - a.max_x
    } else if b.max_x < a.min_x {
        a.min_x - b.max_x
    } else {
        0.0
    };
    let dy = if a.max_y < b.min_y {
        b.min_y - a.max_y
    } else if b.max_y < a.min_y {
        a.min_y - b.max_y
    } else {
        0.0
    };

    if dx <= EPSILON {
        dy
    } else if dy <= EPSILON {
        dx
    } else {
        (dx * dx + dy * dy).sqrt()
    }
}

fn segment_rect_clearance(start: (f64, f64), end: (f64, f64), rect: Rect) -> f64 {
    if point_in_rect(start, rect) || point_in_rect(end, rect) {
        return 0.0;
    }

    if segment_intersects_rect(start, end, rect) {
        return 0.0;
    }

    let mut minimum = point_rect_clearance(start, rect).min(point_rect_clearance(end, rect));
    let rect_corners = [
        (rect.min_x, rect.min_y),
        (rect.max_x, rect.min_y),
        (rect.max_x, rect.max_y),
        (rect.min_x, rect.max_y),
    ];

    for corner in rect_corners {
        minimum = minimum.min(point_segment_distance(corner, start, end));
    }

    minimum
}

fn point_in_rect(point: (f64, f64), rect: Rect) -> bool {
    point.0 >= rect.min_x - EPSILON
        && point.0 <= rect.max_x + EPSILON
        && point.1 >= rect.min_y - EPSILON
        && point.1 <= rect.max_y + EPSILON
}

fn point_segment_distance(
    point: (f64, f64),
    segment_start: (f64, f64),
    segment_end: (f64, f64),
) -> f64 {
    let dx = segment_end.0 - segment_start.0;
    let dy = segment_end.1 - segment_start.1;
    let length_squared = dx * dx + dy * dy;
    if length_squared <= EPSILON {
        return euclidean_distance(point, segment_start);
    }

    let t = (((point.0 - segment_start.0) * dx + (point.1 - segment_start.1) * dy)
        / length_squared)
        .clamp(0.0, 1.0);
    euclidean_distance(point, (segment_start.0 + dx * t, segment_start.1 + dy * t))
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

#[cfg(test)]
fn route_polyline(
    start: (f64, f64),
    end: (f64, f64),
    obstacles: &[Rect],
    bend_penalty: f64,
) -> Vec<(f64, f64)> {
    route_polyline_with_context(start, end, &[], &[], obstacles, None, &[], bend_penalty)
}

fn route_polyline_with_context(
    start: (f64, f64),
    end: (f64, f64),
    node_obstacles: &[(NodeIndex, Rect)],
    cluster_obstacles: &[(usize, Rect)],
    label_obstacles: &[Rect],
    excluded_node_obstacles: Option<(NodeIndex, NodeIndex)>,
    excluded_cluster_obstacles: &[usize],
    bend_penalty: f64,
) -> Vec<(f64, f64)> {
    let mut candidates = Vec::new();

    let straight = vec![start, end];
    if polyline_clear_with_context(
        &straight,
        node_obstacles,
        cluster_obstacles,
        label_obstacles,
        excluded_node_obstacles,
        excluded_cluster_obstacles,
    ) {
        candidates.push(straight);
    }

    let candidate_a = vec![start, (start.0, end.1), end];
    if polyline_clear_with_context(
        &candidate_a,
        node_obstacles,
        cluster_obstacles,
        label_obstacles,
        excluded_node_obstacles,
        excluded_cluster_obstacles,
    ) {
        candidates.push(simplify_polyline(&candidate_a));
    }

    let candidate_b = vec![start, (end.0, start.1), end];
    if polyline_clear_with_context(
        &candidate_b,
        node_obstacles,
        cluster_obstacles,
        label_obstacles,
        excluded_node_obstacles,
        excluded_cluster_obstacles,
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
        cluster_obstacles,
        label_obstacles,
        excluded_node_obstacles,
        excluded_cluster_obstacles,
        bend_penalty,
    ) {
        return grid_route;
    }

    vec![start, end]
}

fn choose_shortest(candidates: &[Vec<(f64, f64)>]) -> Option<Vec<(f64, f64)>> {
    let mut best_index: Option<usize> = None;
    let mut best_len = f64::INFINITY;
    let mut best_points = usize::MAX;

    for (index, candidate) in candidates.iter().enumerate() {
        let length = polyline_length(candidate);
        if length < best_len - EPSILON
            || (almost_equal(length, best_len) && candidate.len() < best_points)
        {
            best_len = length;
            best_points = candidate.len();
            best_index = Some(index);
        }
    }

    best_index.and_then(|index| candidates.get(index).cloned())
}

fn route_via_grid(
    start: (f64, f64),
    end: (f64, f64),
    node_obstacles: &[(NodeIndex, Rect)],
    cluster_obstacles: &[(usize, Rect)],
    label_obstacles: &[Rect],
    excluded_node_obstacles: Option<(NodeIndex, NodeIndex)>,
    excluded_cluster_obstacles: &[usize],
    bend_penalty: f64,
) -> Option<Vec<(f64, f64)>> {
    let preferred_bounds = preferred_route_bounds(start, end);
    let mut xs = vec![start.0, end.0];
    let mut ys = vec![start.1, end.1];
    let search_bounds = routing_search_bounds(start, end);
    xs.push(search_bounds.min_x);
    xs.push(search_bounds.max_x);
    ys.push(search_bounds.min_y);
    ys.push(search_bounds.max_y);

    for (node_id, obstacle) in node_obstacles {
        if is_excluded_node_obstacle(*node_id, excluded_node_obstacles) {
            continue;
        }
        append_rect_routing_guides(
            &mut xs,
            &mut ys,
            *obstacle,
            ROUTING_GUIDE_CLEARANCE,
            search_bounds,
        );
    }
    for obstacle in label_obstacles {
        append_rect_routing_guides(
            &mut xs,
            &mut ys,
            *obstacle,
            ROUTING_GUIDE_CLEARANCE,
            search_bounds,
        );
    }
    for (cluster_index, obstacle) in cluster_obstacles {
        if is_excluded_cluster_obstacle(*cluster_index, excluded_cluster_obstacles) {
            continue;
        }
        append_rect_routing_guides(
            &mut xs,
            &mut ys,
            *obstacle,
            ROUTING_GUIDE_CLEARANCE,
            search_bounds,
        );
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
                cluster_obstacles,
                label_obstacles,
                excluded_node_obstacles,
                excluded_cluster_obstacles,
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
                cluster_obstacles,
                label_obstacles,
                excluded_node_obstacles,
                excluded_cluster_obstacles,
            ) {
                let length = euclidean_distance(p1, p2)
                    + route_preference_penalty(p1, p2, preferred_bounds)
                    + directional_route_penalty(start, end, p1, p2, MoveDir::Horizontal);
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
                cluster_obstacles,
                label_obstacles,
                excluded_node_obstacles,
                excluded_cluster_obstacles,
            ) {
                let length = euclidean_distance(p1, p2)
                    + route_preference_penalty(p1, p2, preferred_bounds)
                    + directional_route_penalty(start, end, p1, p2, MoveDir::Vertical);
                adjacency[top].push((bottom, length, MoveDir::Vertical));
                adjacency[bottom].push((top, length, MoveDir::Vertical));
            }
        }
    }

    for ix in 0..xs.len().saturating_sub(1) {
        for iy in 0..ys.len().saturating_sub(1) {
            let Some(top_left) = point_id_grid.get(grid_offset(ix, iy)).and_then(|id| *id) else {
                continue;
            };
            let Some(top_right) = point_id_grid
                .get(grid_offset(ix + 1, iy))
                .and_then(|id| *id)
            else {
                continue;
            };
            let Some(bottom_left) = point_id_grid
                .get(grid_offset(ix, iy + 1))
                .and_then(|id| *id)
            else {
                continue;
            };
            let Some(bottom_right) = point_id_grid
                .get(grid_offset(ix + 1, iy + 1))
                .and_then(|id| *id)
            else {
                continue;
            };

            for (from, to) in [(top_left, bottom_right), (top_right, bottom_left)] {
                let p1 = points[from];
                let p2 = points[to];
                if segment_clear_with_context(
                    p1,
                    p2,
                    node_obstacles,
                    cluster_obstacles,
                    label_obstacles,
                    excluded_node_obstacles,
                    excluded_cluster_obstacles,
                ) {
                    let length = euclidean_distance(p1, p2)
                        + route_preference_penalty(p1, p2, preferred_bounds)
                        + directional_route_penalty(start, end, p1, p2, MoveDir::None);
                    adjacency[from].push((to, length, MoveDir::None));
                    adjacency[to].push((from, length, MoveDir::None));
                }
            }
        }
    }

    let route = shortest_route(start_id, end_id, &adjacency, bend_penalty, &points)?;
    Some(route)
}

fn preferred_route_bounds(start: (f64, f64), end: (f64, f64)) -> Rect {
    Rect {
        min_x: start.0.min(end.0) - EDGE_LABEL_OBSTACLE_WIDTH,
        max_x: start.0.max(end.0) + EDGE_LABEL_OBSTACLE_WIDTH,
        min_y: start.1.min(end.1) - EDGE_LABEL_OBSTACLE_HEIGHT * 2.0,
        max_y: start.1.max(end.1) + EDGE_LABEL_OBSTACLE_HEIGHT * 2.0,
    }
}

fn route_preference_penalty(start: (f64, f64), end: (f64, f64), preferred_bounds: Rect) -> f64 {
    let midpoint = ((start.0 + end.0) * 0.5, (start.1 + end.1) * 0.5);
    let dx = if midpoint.0 < preferred_bounds.min_x {
        preferred_bounds.min_x - midpoint.0
    } else if midpoint.0 > preferred_bounds.max_x {
        midpoint.0 - preferred_bounds.max_x
    } else {
        0.0
    };
    let dy = if midpoint.1 < preferred_bounds.min_y {
        preferred_bounds.min_y - midpoint.1
    } else if midpoint.1 > preferred_bounds.max_y {
        midpoint.1 - preferred_bounds.max_y
    } else {
        0.0
    };

    (dx + dy) * 12.0
}

fn directional_route_penalty(
    route_start: (f64, f64),
    route_end: (f64, f64),
    start: (f64, f64),
    end: (f64, f64),
    move_dir: MoveDir,
) -> f64 {
    let total_dy = route_end.1 - route_start.1;
    let total_dx = route_end.0 - route_start.0;
    let midpoint = ((start.0 + end.0) * 0.5, (start.1 + end.1) * 0.5);
    let segment_length = euclidean_distance(start, end);
    let mut penalty = 0.0;

    if total_dy.abs() > EDGE_LABEL_OFFSET {
        let preferred_progress_y = if total_dy > 0.0 {
            route_start.1 + total_dy * 0.32
        } else {
            route_start.1 + total_dy * 0.32
        };
        let wrong_way =
            (end.1 - start.1).signum() != total_dy.signum() && (end.1 - start.1).abs() > EPSILON;
        if wrong_way {
            penalty += segment_length * 2.8;
        }

        if move_dir == MoveDir::Horizontal {
            let before_progress = if total_dy > 0.0 {
                midpoint.1 < preferred_progress_y - EPSILON
            } else {
                midpoint.1 > preferred_progress_y + EPSILON
            };
            if before_progress {
                penalty += segment_length * 0.9;
                penalty += (midpoint.1 - preferred_progress_y).abs() * 1.2;
            }
        }
    }

    if total_dx.abs() > EDGE_LABEL_OFFSET && move_dir == MoveDir::Vertical {
        let preferred_progress_x = route_start.0 + total_dx * 0.18;
        let stalled_laterally = if total_dx > 0.0 {
            midpoint.0 < preferred_progress_x - EPSILON
        } else {
            midpoint.0 > preferred_progress_x + EPSILON
        };
        if stalled_laterally && total_dy.abs() <= total_dx.abs() * 0.7 {
            penalty += segment_length * 0.18;
        }
    }

    penalty
}

fn routing_search_bounds(start: (f64, f64), end: (f64, f64)) -> Rect {
    let horizontal_margin = EDGE_LABEL_OBSTACLE_WIDTH * 2.0 + ROUTING_GUIDE_CLEARANCE * 2.0;
    let vertical_margin = EDGE_LABEL_OBSTACLE_HEIGHT * 4.0 + ROUTING_GUIDE_CLEARANCE * 2.0;
    Rect {
        min_x: start.0.min(end.0) - horizontal_margin,
        max_x: start.0.max(end.0) + horizontal_margin,
        min_y: start.1.min(end.1) - vertical_margin,
        max_y: start.1.max(end.1) + vertical_margin,
    }
}

fn append_rect_routing_guides(
    xs: &mut Vec<f64>,
    ys: &mut Vec<f64>,
    rect: Rect,
    clearance: f64,
    search_bounds: Rect,
) {
    if rect.max_x < search_bounds.min_x - clearance
        || rect.min_x > search_bounds.max_x + clearance
        || rect.max_y < search_bounds.min_y - clearance
        || rect.min_y > search_bounds.max_y + clearance
    {
        return;
    }

    xs.push((rect.min_x - clearance).clamp(search_bounds.min_x, search_bounds.max_x));
    xs.push(rect.min_x.clamp(search_bounds.min_x, search_bounds.max_x));
    xs.push(rect.max_x.clamp(search_bounds.min_x, search_bounds.max_x));
    xs.push((rect.max_x + clearance).clamp(search_bounds.min_x, search_bounds.max_x));
    ys.push((rect.min_y - clearance).clamp(search_bounds.min_y, search_bounds.max_y));
    ys.push(rect.min_y.clamp(search_bounds.min_y, search_bounds.max_y));
    ys.push(rect.max_y.clamp(search_bounds.min_y, search_bounds.max_y));
    ys.push((rect.max_y + clearance).clamp(search_bounds.min_y, search_bounds.max_y));
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
    Some(simplify_polyline(&route_points))
}

fn point_allowed_with_context(
    point: (f64, f64),
    start: (f64, f64),
    end: (f64, f64),
    node_obstacles: &[(NodeIndex, Rect)],
    cluster_obstacles: &[(usize, Rect)],
    label_obstacles: &[Rect],
    excluded_node_obstacles: Option<(NodeIndex, NodeIndex)>,
    excluded_cluster_obstacles: &[usize],
) -> bool {
    if same_point(point, start) || same_point(point, end) {
        return true;
    }

    !node_obstacles.iter().any(|(node_id, obstacle)| {
        !is_excluded_node_obstacle(*node_id, excluded_node_obstacles)
            && point_inside_rect(point, *obstacle)
    }) && !cluster_obstacles.iter().any(|(cluster_index, obstacle)| {
        !is_excluded_cluster_obstacle(*cluster_index, excluded_cluster_obstacles)
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

#[cfg(test)]
fn polyline_clear(points: &[(f64, f64)], obstacles: &[Rect]) -> bool {
    polyline_clear_with_context(points, &[], &[], obstacles, None, &[])
}

fn polyline_clear_with_context(
    points: &[(f64, f64)],
    node_obstacles: &[(NodeIndex, Rect)],
    cluster_obstacles: &[(usize, Rect)],
    label_obstacles: &[Rect],
    excluded_node_obstacles: Option<(NodeIndex, NodeIndex)>,
    excluded_cluster_obstacles: &[usize],
) -> bool {
    if points.len() < 2 {
        return true;
    }

    for segment in points.windows(2) {
        if !segment_clear_with_context(
            segment[0],
            segment[1],
            node_obstacles,
            cluster_obstacles,
            label_obstacles,
            excluded_node_obstacles,
            excluded_cluster_obstacles,
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
    cluster_obstacles: &[(usize, Rect)],
    label_obstacles: &[Rect],
    excluded_node_obstacles: Option<(NodeIndex, NodeIndex)>,
    excluded_cluster_obstacles: &[usize],
) -> bool {
    if same_point(start, end) {
        return true;
    }

    !node_obstacles.iter().any(|(node_id, obstacle)| {
        !is_excluded_node_obstacle(*node_id, excluded_node_obstacles)
            && segment_intersects_rect(start, end, *obstacle)
    }) && !label_obstacles
        .iter()
        .any(|obstacle| segment_intersects_rect(start, end, *obstacle))
        && !cluster_obstacles.iter().any(|(cluster_index, obstacle)| {
            !is_excluded_cluster_obstacle(*cluster_index, excluded_cluster_obstacles)
                && segment_intersects_rect(start, end, *obstacle)
        })
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

fn is_excluded_cluster_obstacle(
    cluster_index: usize,
    excluded_cluster_obstacles: &[usize],
) -> bool {
    excluded_cluster_obstacles.contains(&cluster_index)
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

fn compact_route_polyline_with_context(
    points: &[(f64, f64)],
    node_obstacles: &[(NodeIndex, Rect)],
    cluster_obstacles: &[(usize, Rect)],
    label_obstacles: &[Rect],
    excluded_node_obstacles: Option<(NodeIndex, NodeIndex)>,
    excluded_cluster_obstacles: &[usize],
) -> Vec<(f64, f64)> {
    let mut compacted = simplify_polyline(points);

    for _ in 0..12 {
        let mut changed = false;

        let mut index = 0;
        while index + 2 < compacted.len() {
            let start = compacted[index];
            let end = compacted[index + 2];
            if segment_clear_with_context(
                start,
                end,
                node_obstacles,
                cluster_obstacles,
                label_obstacles,
                excluded_node_obstacles,
                excluded_cluster_obstacles,
            ) {
                compacted.remove(index + 1);
                changed = true;
                continue;
            }
            index += 1;
        }

        index = 0;
        while index + 3 < compacted.len() {
            let start = compacted[index];
            let end = compacted[index + 3];
            let window = &compacted[index..=index + 3];
            let existing_length = polyline_length(window);

            let candidates = [
                vec![start, (start.0, end.1), end],
                vec![start, (end.0, start.1), end],
            ];

            let best = candidates
                .into_iter()
                .map(|candidate| simplify_polyline(&candidate))
                .filter(|candidate| candidate.len() >= 2)
                .filter(|candidate| {
                    polyline_clear_with_context(
                        candidate,
                        node_obstacles,
                        cluster_obstacles,
                        label_obstacles,
                        excluded_node_obstacles,
                        excluded_cluster_obstacles,
                    )
                })
                .filter(|candidate| {
                    polyline_length(candidate) <= existing_length + ROUTE_JOG_CLEARANCE + EPSILON
                })
                .min_by(|left, right| {
                    polyline_length(left)
                        .total_cmp(&polyline_length(right))
                        .then_with(|| left.len().cmp(&right.len()))
                });

            if let Some(candidate) = best {
                let mut replacement = Vec::with_capacity(compacted.len());
                replacement.extend_from_slice(&compacted[..index]);
                replacement.extend(candidate);
                replacement.extend_from_slice(&compacted[index + 4..]);
                compacted = simplify_polyline(&replacement);
                changed = true;
                continue;
            }

            index += 1;
        }

        if !changed {
            break;
        }
    }

    compacted
}

fn polyline_length(points: &[(f64, f64)]) -> f64 {
    points
        .windows(2)
        .map(|segment| euclidean_distance(segment[0], segment[1]))
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
        let length = euclidean_distance(start, end);
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
    polyline_point_and_direction_at_fraction(points, fraction).map(|(point, _)| point)
}

fn polyline_point_at_distance(points: &[(f64, f64)], distance: f64) -> Option<(f64, f64)> {
    let total_length = polyline_length(points);
    if total_length <= EPSILON {
        return points.first().copied();
    }
    polyline_point_at_fraction(points, (distance / total_length).clamp(0.0, 1.0))
}

fn polyline_point_at_distance_from_end(points: &[(f64, f64)], distance: f64) -> Option<(f64, f64)> {
    let total_length = polyline_length(points);
    if total_length <= EPSILON {
        return points.last().copied();
    }
    polyline_point_at_fraction(points, (1.0 - distance / total_length).clamp(0.0, 1.0))
}

fn bezier_curve_points_from_polyline(
    points: &[(f64, f64)],
    tail_rect: Rect,
    head_rect: Rect,
    node_obstacles: &[Rect],
) -> Vec<(f64, f64)> {
    if points.len() < 2 {
        return points.to_vec();
    }

    let route_points = simplify_polyline(points);
    let route_points = simplify_curve_guide_points(&route_points, node_obstacles);
    let tail_anchor = route_points[0];
    let head_anchor = route_points[route_points.len() - 1];
    let tail_side = anchor_side_for_point(tail_rect, tail_anchor);
    let head_side = anchor_side_for_point(head_rect, head_anchor);

    if route_points.len() == 2 {
        return direct_bezier_between_anchors(tail_anchor, head_anchor, tail_side, head_side);
    }

    if route_points.len() <= 4 {
        let direct_curve =
            direct_bezier_between_anchors(tail_anchor, head_anchor, tail_side, head_side);
        if bezier_curve_is_clear(&direct_curve, node_obstacles) {
            return direct_curve;
        }
    }

    let spline_points = build_port_extended_route_points(&route_points, tail_rect, head_rect);
    let rounded_curve = clip_bezier_endpoints_to_rects(
        rounded_bezier_curve_points_from_polyline(&spline_points),
        tail_rect,
        head_rect,
        tail_anchor,
        head_anchor,
        anchor_side_vector(tail_side),
        anchor_side_vector(head_side),
    );
    if bezier_curve_is_clear(&rounded_curve, node_obstacles) {
        return rounded_curve;
    }

    let guide_points = build_spline_guide_points(&spline_points);
    let guide_points = simplify_curve_guide_points(&guide_points, node_obstacles);
    let guide_points = curve_fit_polyline(&guide_points);
    if guide_points.len() == 2 {
        return direct_bezier_between_anchors(
            guide_points[0],
            guide_points[1],
            tail_side,
            head_side,
        );
    }

    fit_dot_like_spline_section(
        &guide_points,
        Some(tail_side),
        Some(head_side),
        node_obstacles,
        0,
    )
}

fn bezier_curve_is_clear(control_points: &[(f64, f64)], node_obstacles: &[Rect]) -> bool {
    if control_points.len() < 4 || (control_points.len() - 1) % 3 != 0 {
        return false;
    }

    let sampled = sample_bezier_curve_points(control_points, 16);
    sampled
        .windows(2)
        .all(|segment| segment_clear_of_curve_obstacles(segment[0], segment[1], node_obstacles))
}

fn fit_dot_like_spline_section(
    points: &[(f64, f64)],
    start_side: Option<AnchorSide>,
    end_side: Option<AnchorSide>,
    node_obstacles: &[Rect],
    depth: usize,
) -> Vec<(f64, f64)> {
    const MAX_SPLINE_RECURSION: usize = 8;

    if points.len() < 2 {
        return points.to_vec();
    }

    if points.len() == 2 {
        let start = points[0];
        let end = points[1];
        let start_direction = endpoint_direction(points, true, start_side);
        let end_direction = endpoint_direction(points, false, end_side);
        return cubic_from_endpoints_and_directions(start, end, start_direction, end_direction);
    }

    let start = points[0];
    let end = points[points.len() - 1];
    let candidate = cubic_from_endpoints_and_directions(
        start,
        end,
        endpoint_direction(points, true, start_side),
        endpoint_direction(points, false, end_side),
    );

    if depth >= MAX_SPLINE_RECURSION || spline_section_fits(&candidate, points, node_obstacles) {
        return candidate;
    }

    let split_index = best_spline_split_index(points);
    if depth < MAX_SPLINE_RECURSION
        && has_significant_orthogonal_turn(points)
        && split_index > 0
        && split_index + 1 < points.len()
    {
        let left = fit_dot_like_spline_section(
            &points[..=split_index],
            start_side,
            None,
            node_obstacles,
            depth + 1,
        );
        let right = fit_dot_like_spline_section(
            &points[split_index..],
            None,
            end_side,
            node_obstacles,
            depth + 1,
        );

        let mut combined = left;
        if right.len() > 1 {
            combined.extend_from_slice(&right[1..]);
        }
        return combined;
    }
    if split_index == 0 || split_index + 1 >= points.len() {
        return candidate;
    }

    let left = fit_dot_like_spline_section(
        &points[..=split_index],
        start_side,
        None,
        node_obstacles,
        depth + 1,
    );
    let right = fit_dot_like_spline_section(
        &points[split_index..],
        None,
        end_side,
        node_obstacles,
        depth + 1,
    );

    let mut combined = left;
    if right.len() > 1 {
        combined.extend_from_slice(&right[1..]);
    }
    combined
}

fn has_significant_orthogonal_turn(points: &[(f64, f64)]) -> bool {
    points.windows(3).any(|window| {
        let first = (window[1].0 - window[0].0, window[1].1 - window[0].1);
        let second = (window[2].0 - window[1].0, window[2].1 - window[1].1);
        let first_horizontal = first.0.abs() > EPSILON && first.1.abs() <= EPSILON;
        let first_vertical = first.1.abs() > EPSILON && first.0.abs() <= EPSILON;
        let second_horizontal = second.0.abs() > EPSILON && second.1.abs() <= EPSILON;
        let second_vertical = second.1.abs() > EPSILON && second.0.abs() <= EPSILON;
        let first_len = first.0.abs() + first.1.abs();
        let second_len = second.0.abs() + second.1.abs();
        first_len > EDGE_LABEL_OFFSET * 1.6
            && second_len > EDGE_LABEL_OFFSET * 1.6
            && ((first_horizontal && second_vertical) || (first_vertical && second_horizontal))
    })
}

fn endpoint_direction(
    points: &[(f64, f64)],
    is_start: bool,
    side: Option<AnchorSide>,
) -> (f64, f64) {
    let anchor = if is_start {
        points[0]
    } else {
        points[points.len() - 1]
    };
    let route_direction = if is_start {
        polyline_point_at_distance(points, 56.0)
            .filter(|point| !same_point(*point, anchor))
            .map(|point| unit_direction(anchor, point))
            .filter(|direction| *direction != (0.0, 0.0))
            .unwrap_or_else(|| unit_direction(anchor, points[1]))
    } else {
        polyline_point_at_distance_from_end(points, 56.0)
            .filter(|point| !same_point(*point, anchor))
            .map(|point| unit_direction(point, anchor))
            .filter(|direction| *direction != (0.0, 0.0))
            .unwrap_or_else(|| unit_direction(points[points.len() - 2], anchor))
    };

    let side_direction = side.map(anchor_side_vector).unwrap_or((0.0, 0.0));
    if is_start {
        blend_weighted_directions(side_direction, 1.2, route_direction, 1.0)
            .or_else(|| (route_direction != (0.0, 0.0)).then_some(route_direction))
            .or_else(|| (side_direction != (0.0, 0.0)).then_some(side_direction))
            .unwrap_or((0.0, 0.0))
    } else {
        let inward = (-side_direction.0, -side_direction.1);
        blend_weighted_directions(route_direction, 1.0, inward, 1.2)
            .or_else(|| (route_direction != (0.0, 0.0)).then_some(route_direction))
            .or_else(|| (inward != (0.0, 0.0)).then_some(inward))
            .unwrap_or((0.0, 0.0))
    }
}

fn cubic_from_endpoints_and_directions(
    start: (f64, f64),
    end: (f64, f64),
    start_direction: (f64, f64),
    end_direction: (f64, f64),
) -> Vec<(f64, f64)> {
    let length = euclidean_distance(start, end);
    let handle_length = (length * 0.5).clamp(28.0, 96.0);
    vec![
        start,
        (
            start.0 + start_direction.0 * handle_length,
            start.1 + start_direction.1 * handle_length,
        ),
        (
            end.0 - end_direction.0 * handle_length,
            end.1 - end_direction.1 * handle_length,
        ),
        end,
    ]
}

fn spline_section_fits(
    candidate: &[(f64, f64)],
    guide_points: &[(f64, f64)],
    node_obstacles: &[Rect],
) -> bool {
    const MAX_DEVIATION: f64 = 40.0;

    let sampled = sample_bezier_curve_points(candidate, 24);
    if sampled.len() < 2 {
        return false;
    }

    if sampled
        .windows(2)
        .any(|segment| !segment_clear_of_curve_obstacles(segment[0], segment[1], node_obstacles))
    {
        return false;
    }

    guide_points
        .iter()
        .copied()
        .all(|point| point_to_polyline_distance(point, &sampled) <= MAX_DEVIATION)
}

fn best_spline_split_index(points: &[(f64, f64)]) -> usize {
    if points.len() <= 2 {
        return 0;
    }

    let start = points[0];
    let end = points[points.len() - 1];
    points
        .iter()
        .enumerate()
        .skip(1)
        .take(points.len().saturating_sub(2))
        .max_by(|(_, left), (_, right)| {
            point_to_segment_distance(**left, start, end)
                .total_cmp(&point_to_segment_distance(**right, start, end))
        })
        .map(|(index, _)| index)
        .unwrap_or(points.len() / 2)
}

fn point_to_polyline_distance(point: (f64, f64), polyline: &[(f64, f64)]) -> f64 {
    polyline
        .windows(2)
        .map(|segment| point_to_segment_distance(point, segment[0], segment[1]))
        .fold(f64::INFINITY, f64::min)
}

fn point_to_segment_distance(point: (f64, f64), start: (f64, f64), end: (f64, f64)) -> f64 {
    let segment = (end.0 - start.0, end.1 - start.1);
    let segment_length_squared = segment.0 * segment.0 + segment.1 * segment.1;
    if segment_length_squared <= EPSILON {
        return euclidean_distance(point, start);
    }

    let t = (((point.0 - start.0) * segment.0) + ((point.1 - start.1) * segment.1))
        / segment_length_squared;
    let t = t.clamp(0.0, 1.0);
    let projection = (start.0 + segment.0 * t, start.1 + segment.1 * t);
    euclidean_distance(point, projection)
}

fn direct_bezier_between_anchors(
    start: (f64, f64),
    end: (f64, f64),
    tail_side: AnchorSide,
    head_side: AnchorSide,
) -> Vec<(f64, f64)> {
    let direct = unit_direction(start, end);
    if direct == (0.0, 0.0) {
        return vec![start, start, end, end];
    }

    let tail_outward = anchor_side_vector(tail_side);
    let head_inward = {
        let outward = anchor_side_vector(head_side);
        (-outward.0, -outward.1)
    };
    let is_near_axis_aligned = direct.0.abs() >= 0.92 || direct.1.abs() >= 0.92;
    let start_direction = if is_near_axis_aligned {
        direct
    } else {
        blend_weighted_directions(tail_outward, 1.2, direct, 0.85).unwrap_or(direct)
    };
    let end_direction = if is_near_axis_aligned {
        direct
    } else {
        blend_weighted_directions(head_inward, 1.2, direct, 0.85).unwrap_or(direct)
    };
    let handle_length = if is_near_axis_aligned {
        (euclidean_distance(start, end) * 0.22).clamp(12.0, 28.0)
    } else {
        (euclidean_distance(start, end) * 0.35).clamp(18.0, 52.0)
    };

    vec![
        start,
        (
            start.0 + start_direction.0 * handle_length,
            start.1 + start_direction.1 * handle_length,
        ),
        (
            end.0 - end_direction.0 * handle_length,
            end.1 - end_direction.1 * handle_length,
        ),
        end,
    ]
}

fn curve_fit_polyline(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut fitted = simplify_polyline(points);

    for _ in 0..8 {
        if fitted.len() < 4 {
            break;
        }

        let mut changed = false;
        let mut index = 1;
        while index + 1 < fitted.len() {
            let previous = fitted[index - 1];
            let current = fitted[index];
            let next = fitted[index + 1];
            let incoming = euclidean_distance(previous, current);
            let outgoing = euclidean_distance(current, next);

            if incoming.min(outgoing) <= 32.0
                || point_to_segment_distance(current, previous, next) <= 10.0
            {
                fitted.remove(index);
                changed = true;
                continue;
            }

            index += 1;
        }

        if !changed {
            break;
        }
    }

    fitted
}

fn unit_direction(start: (f64, f64), end: (f64, f64)) -> (f64, f64) {
    let dx = end.0 - start.0;
    let dy = end.1 - start.1;
    let length = (dx * dx + dy * dy).sqrt();
    if length <= EPSILON {
        (0.0, 0.0)
    } else {
        (dx / length, dy / length)
    }
}

fn blend_unit_directions(a: (f64, f64), b: (f64, f64)) -> Option<(f64, f64)> {
    let blended = (a.0 + b.0, a.1 + b.1);
    let length = (blended.0 * blended.0 + blended.1 * blended.1).sqrt();
    if length <= EPSILON {
        None
    } else {
        Some((blended.0 / length, blended.1 / length))
    }
}

fn blend_weighted_directions(
    a: (f64, f64),
    a_weight: f64,
    b: (f64, f64),
    b_weight: f64,
) -> Option<(f64, f64)> {
    let blended = (
        a.0 * a_weight + b.0 * b_weight,
        a.1 * a_weight + b.1 * b_weight,
    );
    let length = (blended.0 * blended.0 + blended.1 * blended.1).sqrt();
    if length <= EPSILON {
        None
    } else {
        Some((blended.0 / length, blended.1 / length))
    }
}

fn build_spline_guide_points(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    if points.len() <= 2 {
        return points.to_vec();
    }

    const TURN_OFFSET_SCALE: f64 = 0.42;
    const MAX_TURN_OFFSET: f64 = 40.0;

    let mut guides = Vec::with_capacity(points.len());
    guides.push(points[0]);

    for index in 1..points.len() - 1 {
        let previous = points[index - 1];
        let current = points[index];
        let next = points[index + 1];

        let incoming_direction = unit_direction(current, previous);
        let outgoing_direction = unit_direction(current, next);

        if incoming_direction == (0.0, 0.0)
            || outgoing_direction == (0.0, 0.0)
            || vectors_are_collinear(incoming_direction, outgoing_direction)
        {
            push_unique_point(&mut guides, current);
            continue;
        }

        let offset = (euclidean_distance(previous, current).min(euclidean_distance(current, next))
            * TURN_OFFSET_SCALE)
            .min(MAX_TURN_OFFSET);

        let bisector = blend_unit_directions(incoming_direction, outgoing_direction)
            .unwrap_or(incoming_direction);
        let apex = (
            current.0 + bisector.0 * offset,
            current.1 + bisector.1 * offset,
        );

        push_unique_point(&mut guides, apex);
    }

    push_unique_point(&mut guides, *points.last().unwrap_or(&points[0]));
    guides
}

fn build_port_extended_route_points(
    route_points: &[(f64, f64)],
    tail_rect: Rect,
    head_rect: Rect,
) -> Vec<(f64, f64)> {
    if route_points.len() < 2 {
        return route_points.to_vec();
    }

    let mut extended = Vec::with_capacity(route_points.len() + 2);
    let tail_anchor = route_points[0];
    let head_anchor = route_points[route_points.len() - 1];
    let tail_side = anchor_side_for_point(tail_rect, tail_anchor);
    let head_side = anchor_side_for_point(head_rect, head_anchor);
    let tail_vector = anchor_side_vector(tail_side);
    let head_vector = anchor_side_vector(head_side);

    extended.push(tail_anchor);
    push_unique_point(
        &mut extended,
        (
            tail_anchor.0 + tail_vector.0 * port_extension_length(tail_rect),
            tail_anchor.1 + tail_vector.1 * port_extension_length(tail_rect),
        ),
    );

    if route_points.len() > 2 {
        for &point in &route_points[1..route_points.len() - 1] {
            push_unique_point(&mut extended, point);
        }
    }

    push_unique_point(
        &mut extended,
        (
            head_anchor.0 + head_vector.0 * port_extension_length(head_rect),
            head_anchor.1 + head_vector.1 * port_extension_length(head_rect),
        ),
    );
    push_unique_point(&mut extended, head_anchor);

    simplify_polyline(&extended)
}

fn simplify_curve_guide_points(points: &[(f64, f64)], node_obstacles: &[Rect]) -> Vec<(f64, f64)> {
    let mut simplified = simplify_polyline(points);
    if simplified.len() < 3 {
        return simplified;
    }

    let mut index = 0;
    while index + 2 < simplified.len() {
        let start = simplified[index];
        let end = simplified[index + 2];
        if segment_clear_of_curve_obstacles(start, end, node_obstacles) {
            simplified.remove(index + 1);
            index = index.saturating_sub(1);
            continue;
        }
        index += 1;
    }

    simplified
}

fn segment_clear_of_curve_obstacles(
    start: (f64, f64),
    end: (f64, f64),
    obstacles: &[Rect],
) -> bool {
    let expanded_clearance = 3.0;
    !obstacles.iter().any(|obstacle| {
        diagonal_segment_intersects_rect(start, end, obstacle.expand(expanded_clearance))
    })
}

fn diagonal_segment_intersects_rect(start: (f64, f64), end: (f64, f64), rect: Rect) -> bool {
    if point_on_or_inside_rect(start, rect) || point_on_or_inside_rect(end, rect) {
        return true;
    }

    let corners = [
        (rect.min_x, rect.min_y),
        (rect.max_x, rect.min_y),
        (rect.max_x, rect.max_y),
        (rect.min_x, rect.max_y),
    ];

    (0..4).any(|index| {
        let edge_start = corners[index];
        let edge_end = corners[(index + 1) % 4];
        segments_intersect(start, end, edge_start, edge_end)
    })
}

fn segments_intersect(
    a_start: (f64, f64),
    a_end: (f64, f64),
    b_start: (f64, f64),
    b_end: (f64, f64),
) -> bool {
    let a1 = orientation(a_start, a_end, b_start);
    let a2 = orientation(a_start, a_end, b_end);
    let b1 = orientation(b_start, b_end, a_start);
    let b2 = orientation(b_start, b_end, a_end);

    if a1 == 0 && on_segment(a_start, b_start, a_end) {
        return true;
    }
    if a2 == 0 && on_segment(a_start, b_end, a_end) {
        return true;
    }
    if b1 == 0 && on_segment(b_start, a_start, b_end) {
        return true;
    }
    if b2 == 0 && on_segment(b_start, a_end, b_end) {
        return true;
    }

    (a1 > 0) != (a2 > 0) && (b1 > 0) != (b2 > 0)
}

fn orientation(a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> i32 {
    let value = (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0);
    if value.abs() <= EPSILON {
        0
    } else if value > 0.0 {
        1
    } else {
        -1
    }
}

fn on_segment(a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> bool {
    b.0 >= a.0.min(c.0) - EPSILON
        && b.0 <= a.0.max(c.0) + EPSILON
        && b.1 >= a.1.min(c.1) - EPSILON
        && b.1 <= a.1.max(c.1) + EPSILON
}

fn port_extension_length(rect: Rect) -> f64 {
    let width = (rect.max_x - rect.min_x).max(0.0);
    let height = (rect.max_y - rect.min_y).max(0.0);
    (width.min(height) * 0.28).clamp(12.0, 20.0)
}

fn rounded_bezier_curve_points_from_polyline(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    if points.len() < 2 {
        return points.to_vec();
    }

    let mut control_points = Vec::with_capacity(points.len() * 3 + 1);
    control_points.push(points[0]);

    let mut current = points[0];
    for index in 1..points.len() - 1 {
        let previous = points[index - 1];
        let corner = points[index];
        let next = points[index + 1];
        let incoming = unit_direction(previous, corner);
        let outgoing = unit_direction(corner, next);

        if incoming == (0.0, 0.0)
            || outgoing == (0.0, 0.0)
            || vectors_are_collinear(incoming, outgoing)
        {
            continue;
        }

        let turn_radius = corner_rounding_radius(previous, corner, next);
        let before_corner = (
            corner.0 - incoming.0 * turn_radius,
            corner.1 - incoming.1 * turn_radius,
        );
        let after_corner = (
            corner.0 + outgoing.0 * turn_radius,
            corner.1 + outgoing.1 * turn_radius,
        );

        append_straight_cubic_segment(&mut control_points, current, before_corner);
        append_quadratic_as_cubic_segment(&mut control_points, before_corner, corner, after_corner);
        current = after_corner;
    }

    append_straight_cubic_segment(&mut control_points, current, points[points.len() - 1]);
    control_points
}

fn corner_rounding_radius(previous: (f64, f64), corner: (f64, f64), next: (f64, f64)) -> f64 {
    const TURN_RADIUS_SCALE: f64 = 0.42;
    const MIN_TURN_RADIUS: f64 = 10.0;
    const MAX_TURN_RADIUS: f64 = 44.0;

    (euclidean_distance(previous, corner).min(euclidean_distance(corner, next)) * TURN_RADIUS_SCALE)
        .clamp(MIN_TURN_RADIUS, MAX_TURN_RADIUS)
}

fn build_centerline_route_points(
    route_points: &[(f64, f64)],
    tail_center: (f64, f64),
    head_center: (f64, f64),
    tail_rect: Rect,
    head_rect: Rect,
) -> Vec<(f64, f64)> {
    let mut centerline = Vec::with_capacity(route_points.len() + 2);
    let tail_anchor = route_points.first().copied().unwrap_or(tail_center);
    let head_anchor = route_points.last().copied().unwrap_or(head_center);
    let tail_side = anchor_side_for_point(tail_rect, tail_anchor);
    let head_side = anchor_side_for_point(head_rect, head_anchor);
    let tail_vector = anchor_side_vector(tail_side);
    let head_vector = anchor_side_vector(head_side);
    let tail_extension = (
        tail_anchor.0 + tail_vector.0 * 18.0,
        tail_anchor.1 + tail_vector.1 * 18.0,
    );
    let head_extension = (
        head_anchor.0 - head_vector.0 * 18.0,
        head_anchor.1 - head_vector.1 * 18.0,
    );

    let _ = (tail_rect, head_rect);
    centerline.push(tail_center);
    push_unique_point(&mut centerline, tail_extension);

    if route_points.len() > 2 {
        for &point in &route_points[1..route_points.len() - 1] {
            push_unique_point(&mut centerline, point);
        }
    }

    push_unique_point(&mut centerline, head_extension);
    push_unique_point(&mut centerline, head_center);
    simplify_polyline(&centerline)
}

fn clip_bezier_endpoints_to_rects(
    mut control_points: Vec<(f64, f64)>,
    tail_rect: Rect,
    head_rect: Rect,
    tail_anchor: (f64, f64),
    head_anchor: (f64, f64),
    tail_port_direction: (f64, f64),
    head_port_direction: (f64, f64),
) -> Vec<(f64, f64)> {
    if control_points.len() < 4 {
        return control_points;
    }

    let tail_center = rect_center(tail_rect);
    let head_center = rect_center(head_rect);

    if let Some(source_direction) =
        non_zero_direction(tail_port_direction).or_else(|| bezier_start_direction(&control_points))
    {
        if point_on_or_inside_rect(tail_anchor, tail_rect) {
            control_points[0] = tail_anchor;
            let source_handle = control_points[1];
            let handle_distance = euclidean_distance(tail_center, source_handle).max(12.0);
            control_points[1] = (
                tail_anchor.0 + source_direction.0 * handle_distance,
                tail_anchor.1 + source_direction.1 * handle_distance,
            );
        }
    }

    if let Some(destination_direction) =
        non_zero_direction(head_port_direction).or_else(|| bezier_end_direction(&control_points))
    {
        if point_on_or_inside_rect(head_anchor, head_rect) {
            let last = control_points.len() - 1;
            control_points[last] = head_anchor;
            let handle_index = last - 1;
            let destination_handle = control_points[handle_index];
            let handle_distance = euclidean_distance(head_center, destination_handle).max(12.0);
            control_points[handle_index] = (
                head_anchor.0 - destination_direction.0 * handle_distance,
                head_anchor.1 - destination_direction.1 * handle_distance,
            );
        }
    }

    control_points
}

fn non_zero_direction(direction: (f64, f64)) -> Option<(f64, f64)> {
    (direction != (0.0, 0.0)).then_some(direction)
}

fn bezier_start_direction(control_points: &[(f64, f64)]) -> Option<(f64, f64)> {
    control_points
        .get(1)
        .copied()
        .and_then(|control| {
            let direction = unit_direction(control_points[0], control);
            (direction != (0.0, 0.0)).then_some(direction)
        })
        .or_else(|| {
            control_points.get(3).copied().and_then(|end| {
                let direction = unit_direction(control_points[0], end);
                (direction != (0.0, 0.0)).then_some(direction)
            })
        })
}

fn bezier_end_direction(control_points: &[(f64, f64)]) -> Option<(f64, f64)> {
    let last = control_points.len().checked_sub(1)?;
    control_points
        .get(last.saturating_sub(1))
        .copied()
        .and_then(|control| {
            let direction = unit_direction(control, control_points[last]);
            (direction != (0.0, 0.0)).then_some(direction)
        })
        .or_else(|| {
            control_points
                .get(last.saturating_sub(3))
                .copied()
                .and_then(|start| {
                    let direction = unit_direction(start, control_points[last]);
                    (direction != (0.0, 0.0)).then_some(direction)
                })
        })
}

fn rect_ray_boundary_intersection(
    rect: Rect,
    origin: (f64, f64),
    direction: (f64, f64),
) -> Option<(f64, f64)> {
    if direction == (0.0, 0.0) {
        return None;
    }

    let mut t_values = Vec::with_capacity(4);
    if direction.0 > EPSILON {
        t_values.push((rect.max_x - origin.0) / direction.0);
    } else if direction.0 < -EPSILON {
        t_values.push((rect.min_x - origin.0) / direction.0);
    }
    if direction.1 > EPSILON {
        t_values.push((rect.max_y - origin.1) / direction.1);
    } else if direction.1 < -EPSILON {
        t_values.push((rect.min_y - origin.1) / direction.1);
    }

    t_values
        .into_iter()
        .filter(|t| *t >= -EPSILON)
        .map(|t| (origin.0 + direction.0 * t, origin.1 + direction.1 * t, t))
        .filter(|(x, y, _)| {
            *x >= rect.min_x - 1e-3
                && *x <= rect.max_x + 1e-3
                && *y >= rect.min_y - 1e-3
                && *y <= rect.max_y + 1e-3
        })
        .min_by(|left, right| left.2.total_cmp(&right.2))
        .map(|(x, y, _)| (x, y))
}

fn point_on_or_inside_rect(point: (f64, f64), rect: Rect) -> bool {
    point.0 >= rect.min_x - EPSILON
        && point.0 <= rect.max_x + EPSILON
        && point.1 >= rect.min_y - EPSILON
        && point.1 <= rect.max_y + EPSILON
}

fn dot(a: (f64, f64), b: (f64, f64)) -> f64 {
    a.0 * b.0 + a.1 * b.1
}

fn push_unique_point(points: &mut Vec<(f64, f64)>, point: (f64, f64)) {
    if points
        .last()
        .copied()
        .map(|last| same_point(last, point))
        .unwrap_or(false)
    {
        return;
    }
    points.push(point);
}

fn vectors_are_collinear(a: (f64, f64), b: (f64, f64)) -> bool {
    (a.0 * b.1 - a.1 * b.0).abs() <= EPSILON
}

fn clamp_handle_to_length(
    origin: (f64, f64),
    handle: (f64, f64),
    maximum_length: f64,
) -> (f64, f64) {
    let direction = unit_direction(origin, handle);
    if direction == (0.0, 0.0) {
        return origin;
    }

    let length = euclidean_distance(origin, handle);
    let clamped = length.min(maximum_length.max(0.0));
    (
        origin.0 + direction.0 * clamped,
        origin.1 + direction.1 * clamped,
    )
}

fn append_straight_cubic_segment(
    control_points: &mut Vec<(f64, f64)>,
    start: (f64, f64),
    end: (f64, f64),
) {
    if manhattan_distance(start, end) <= EPSILON {
        return;
    }

    control_points.push(interpolate_point(start, end, 1.0 / 3.0));
    control_points.push(interpolate_point(start, end, 2.0 / 3.0));
    control_points.push(end);
}

fn append_quadratic_as_cubic_segment(
    control_points: &mut Vec<(f64, f64)>,
    start: (f64, f64),
    control: (f64, f64),
    end: (f64, f64),
) {
    let cubic_one = (
        start.0 + (control.0 - start.0) * (2.0 / 3.0),
        start.1 + (control.1 - start.1) * (2.0 / 3.0),
    );
    let cubic_two = (
        end.0 + (control.0 - end.0) * (2.0 / 3.0),
        end.1 + (control.1 - end.1) * (2.0 / 3.0),
    );

    control_points.push(cubic_one);
    control_points.push(cubic_two);
    control_points.push(end);
}

fn interpolate_point(start: (f64, f64), end: (f64, f64), t: f64) -> (f64, f64) {
    (
        start.0 + (end.0 - start.0) * t,
        start.1 + (end.1 - start.1) * t,
    )
}

fn polyline_point_and_direction_at_fraction(
    points: &[(f64, f64)],
    fraction: f64,
) -> Option<((f64, f64), MoveDir)> {
    if points.is_empty() {
        return None;
    }
    if points.len() == 1 {
        return Some((points[0], MoveDir::None));
    }

    let total_length = polyline_length(points);
    if total_length <= EPSILON {
        return Some((points[0], MoveDir::None));
    }

    let clamped = fraction.clamp(0.0, 1.0);
    let target_length = total_length * clamped;
    let mut traversed = 0.0;

    for segment in points.windows(2) {
        let start = segment[0];
        let end = segment[1];
        let length = euclidean_distance(start, end);
        if traversed + length >= target_length {
            if length <= EPSILON {
                return Some((start, MoveDir::None));
            }
            let remaining = target_length - traversed;
            let t = remaining / length;
            let direction = if almost_equal(start.1, end.1) {
                MoveDir::Horizontal
            } else if almost_equal(start.0, end.0) {
                MoveDir::Vertical
            } else {
                MoveDir::None
            };
            return Some((
                (
                    start.0 + (end.0 - start.0) * t,
                    start.1 + (end.1 - start.1) * t,
                ),
                direction,
            ));
        }
        traversed += length;
    }

    points.last().copied().map(|point| (point, MoveDir::None))
}

fn manhattan_distance(a: (f64, f64), b: (f64, f64)) -> f64 {
    (a.0 - b.0).abs() + (a.1 - b.1).abs()
}

fn euclidean_distance(a: (f64, f64), b: (f64, f64)) -> f64 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    (dx * dx + dy * dy).sqrt()
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum AnchorSide {
    Top,
    Bottom,
    Left,
    Right,
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
