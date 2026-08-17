//! The microdot layout algorithm.
//!
//! Ported from the `iced-sugiyama2` core (`core/src/layout.rs`): a layered
//! (Sugiyama-style) layout with cycle orientation, ranking, crossing
//! minimization, cluster-aware ordering and coordinate assignment, plus
//! bezier edge routing.

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet};

use crate::layout_engine::{Cluster, GraphLayout, LayoutInput};
use rust_sugiyama::{
    ClusterLayout, Config, CrossingMinimization, EdgeLayout, RankingType, RenderConfig,
};

const DEFAULT_NODE_SIZE: (f64, f64) = (56.0, 32.0);

/// Compute the layout using the microdot algorithm.
pub fn microdot_layout<'a>(input: &LayoutInput<'a>) -> GraphLayout {
    let mut config = input.config;
    if matches!(config.ranking_type, RankingType::Hybrid) {
        config.ranking_type = if config.hybrid_weight >= 0.5 {
            RankingType::Original
        } else {
            RankingType::MinimizeEdgeLength
        };
    }
    layout_graph(
        &input.nodes,
        &input.edges,
        &config,
        &|n| (input.node_size)(n),
        &|i, e| (input.edge_label)(i, e),
        &input.clusters,
        &input.render_config,
    )
}

fn empty_layout() -> GraphLayout {
    GraphLayout::from_parts(0.0, 1.0, BTreeMap::new(), Vec::new(), Vec::new())
}

const POINT_TO_PIXEL: f64 = 4.0 / 3.0;
const EDGE_LABEL_HEIGHT: f64 = 16.5 * POINT_TO_PIXEL;
const EDGE_LABEL_FONT_SIZE: f64 = 14.0;
const CLUSTER_LABEL_FONT_SIZE: f64 = 14.0;
const CLUSTER_LABEL_ROW: f64 = 24.5 * POINT_TO_PIXEL;
const CLUSTER_LABEL_SIDE_MARGIN: f64 = 6.5 * POINT_TO_PIXEL;
const ROOT_CLUSTER_OFFSET: f64 = 8.0 * POINT_TO_PIXEL;
const ODD_RANK_SEPARATION: f64 = 5.0 * POINT_TO_PIXEL;
const TAIL_ARROW_CLEARANCE: f64 = 4.8 * POINT_TO_PIXEL;
const HEAD_ARROW_CLEARANCE: f64 = 5.3 * POINT_TO_PIXEL;
const EPSILON: f64 = 1.0e-7;

#[derive(Clone)]
struct NodeData {
    width: f64,
    height: f64,
    rank: usize,
    path: Vec<usize>,
    owner: Option<usize>,
    declaration_order: usize,
    item: usize,
}

#[derive(Clone)]
struct ClusterData {
    parent: Option<usize>,
    children: Vec<usize>,
    depth: usize,
    members: Vec<usize>,
}

#[derive(Clone)]
struct WorkEdge {
    index: usize,
    tail: usize,
    head: usize,
    label: Option<String>,
    reversed: bool,
    chain: Vec<usize>,
    label_item: Option<usize>,
}

#[derive(Clone, Copy)]
enum ItemKind {
    Real(usize),
    Virtual { edge: usize, label: bool },
}

#[derive(Clone)]
struct LayerItem {
    kind: ItemKind,
    layer: usize,
    width: f64,
    path: Vec<usize>,
    stable_order: f64,
    x: f64,
}

impl LayerItem {
    fn is_label(&self) -> bool {
        matches!(self.kind, ItemKind::Virtual { label: true, .. })
    }

    fn route_x(&self) -> f64 {
        if self.is_label() {
            self.x - self.width / 2.0
        } else {
            self.x
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Bounds {
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
}

impl Bounds {
    fn empty() -> Self {
        Self {
            min_x: f64::INFINITY,
            min_y: f64::INFINITY,
            max_x: f64::NEG_INFINITY,
            max_y: f64::NEG_INFINITY,
        }
    }

    fn is_empty(self) -> bool {
        !self.min_x.is_finite()
    }

    fn include_point(&mut self, x: f64, y: f64) {
        self.min_x = self.min_x.min(x);
        self.min_y = self.min_y.min(y);
        self.max_x = self.max_x.max(x);
        self.max_y = self.max_y.max(y);
    }

    fn include_rect(&mut self, other: Self) {
        if other.is_empty() {
            return;
        }
        self.include_point(other.min_x, other.min_y);
        self.include_point(other.max_x, other.max_y);
    }

    fn width(self) -> f64 {
        (self.max_x - self.min_x).max(0.0)
    }

    fn center_x(self) -> f64 {
        (self.min_x + self.max_x) / 2.0
    }

    fn overlaps_y(self, other: Self) -> bool {
        self.min_y < other.max_y - EPSILON && other.min_y < self.max_y - EPSILON
    }
}

fn layout_graph<NodeSize, EdgeLabel>(
    nodes: &[u32],
    edges: &[(u32, u32)],
    config: &Config,
    node_size: NodeSize,
    edge_label: EdgeLabel,
    clusters: &[Cluster],
    render_config: &RenderConfig,
) -> GraphLayout
where
    NodeSize: Fn(u32) -> (f64, f64),
    EdgeLabel: Fn(usize, (u32, u32)) -> Option<String>,
{
    if nodes.is_empty() {
        return empty_layout();
    }

    let mut node_index = HashMap::with_capacity(nodes.len());
    let mut node_data = Vec::with_capacity(nodes.len());
    for (position, &node) in nodes.iter().enumerate() {
        node_index.entry(node).or_insert(position);
        let (width, height) = sanitize_size(node_size(node));
        node_data.push(NodeData {
            width,
            height,
            rank: 0,
            path: Vec::new(),
            owner: None,
            declaration_order: position,
            item: usize::MAX,
        });
    }

    let cluster_data = prepare_clusters(clusters, &node_index, nodes.len());
    assign_cluster_paths(&mut node_data, &cluster_data);
    assign_declaration_order(&mut node_data, &cluster_data);

    let mut work_edges = Vec::with_capacity(edges.len());
    for (index, &(tail, head)) in edges.iter().enumerate() {
        let (Some(&tail), Some(&head)) = (node_index.get(&tail), node_index.get(&head)) else {
            continue;
        };
        work_edges.push(WorkEdge {
            index,
            tail,
            head,
            label: edge_label(index, edges[index]),
            reversed: false,
            chain: Vec::new(),
            label_item: None,
        });
    }

    orient_cycles(&node_data, &mut work_edges);
    assign_ranks(&mut node_data, &work_edges, config);

    let (mut items, mut layers, segments) =
        build_layer_graph(&mut node_data, &mut work_edges, config);
    minimize_crossings(&mut layers, &items, &segments, config);
    balance_real_cluster_blocks(&mut layers, &items);
    assign_x_coordinates(
        &layers,
        &mut items,
        &segments,
        config,
        render_config,
        work_edges.iter().any(|edge| edge.label.is_some()),
    );
    align_vertical_blocks(&node_data, &work_edges, &mut items);
    order_top_level_components(&node_data, &cluster_data, &mut items, config);
    relax_virtual_items(
        &layers,
        &mut items,
        &segments,
        config,
        render_config,
        work_edges.iter().any(|edge| edge.label.is_some()),
    );
    if align_unary_cluster_stacks(
        &node_data,
        &cluster_data,
        clusters,
        &mut items,
        render_config,
    ) {
        relax_virtual_items(
            &layers,
            &mut items,
            &segments,
            config,
            render_config,
            work_edges.iter().any(|edge| edge.label.is_some()),
        );
    }

    let (rank_y, layer_y) = assign_y_coordinates(
        &node_data,
        &cluster_data,
        &work_edges,
        config,
        clusters,
        render_config,
    );

    enforce_cluster_separation(
        &node_data,
        &cluster_data,
        clusters,
        &layers,
        &rank_y,
        &mut items,
        config,
        render_config,
    );
    order_top_level_components(&node_data, &cluster_data, &mut items, config);
    for item in &mut items {
        item.x = (item.x / POINT_TO_PIXEL).round() * POINT_TO_PIXEL;
    }

    let cluster_bounds = compute_all_cluster_bounds(
        &node_data,
        &cluster_data,
        clusters,
        &items,
        &rank_y,
        render_config,
    );
    shape_long_cluster_routes(&node_data, &work_edges, &cluster_bounds, &mut items, config);

    let mut edge_layouts = route_edges(&node_data, &work_edges, &items, &rank_y, &layer_y);

    let mut graph_bounds = Bounds::empty();
    for node in &node_data {
        let x = items[node.item].x;
        let y = rank_y[node.rank];
        graph_bounds.include_rect(Bounds {
            min_x: x - node.width / 2.0,
            min_y: y - node.height / 2.0,
            max_x: x + node.width / 2.0,
            max_y: y + node.height / 2.0,
        });
    }

    for (index, bounds) in cluster_bounds.iter().copied().enumerate() {
        if cluster_data[index].parent.is_none() {
            graph_bounds.include_rect(Bounds {
                min_x: bounds.min_x - ROOT_CLUSTER_OFFSET,
                min_y: bounds.min_y - ROOT_CLUSTER_OFFSET,
                max_x: bounds.max_x + ROOT_CLUSTER_OFFSET,
                max_y: bounds.max_y + ROOT_CLUSTER_OFFSET,
            });
        }
    }

    for edge in &edge_layouts {
        for &(x, y) in edge.curve_points.iter().chain(edge.points.iter()) {
            graph_bounds.include_point(x, y);
        }
        if let (Some(label), Some((x, y))) = (&edge.label, edge.label_position) {
            let width = text_width(label, EDGE_LABEL_FONT_SIZE) * POINT_TO_PIXEL;
            graph_bounds.include_rect(Bounds {
                min_x: x - width / 2.0,
                min_y: y - EDGE_LABEL_HEIGHT / 2.0,
                max_x: x + width / 2.0,
                max_y: y + EDGE_LABEL_HEIGHT / 2.0,
            });
        }
    }

    if graph_bounds.is_empty() {
        return empty_layout();
    }

    let shift_x = -graph_bounds.min_x;
    let shift_y = -graph_bounds.min_y;
    let max_x = graph_bounds.width().max(1.0);
    let max_y = (graph_bounds.max_y - graph_bounds.min_y).max(1.0);

    let mut coords = std::collections::BTreeMap::new();
    for (position, node) in node_data.iter().enumerate() {
        coords.insert(
            position,
            (items[node.item].x + shift_x, rank_y[node.rank] + shift_y),
        );
    }

    for edge in &mut edge_layouts {
        for point in edge.points.iter_mut().chain(edge.curve_points.iter_mut()) {
            point.0 += shift_x;
            point.1 += shift_y;
        }
        if let Some((x, y)) = edge.label_position.as_mut() {
            *x += shift_x;
            *y += shift_y;
        }
    }
    edge_layouts.sort_by_key(|edge| edge.index);

    let mut cluster_layouts = cluster_bounds
        .into_iter()
        .enumerate()
        .map(|(index, bounds)| {
            ClusterLayout::new(
                index,
                cluster_data[index].parent,
                bounds.min_x + shift_x,
                bounds.min_y + shift_y,
                bounds.max_x + shift_x,
                bounds.max_y + shift_y,
            )
        })
        .collect::<Vec<_>>();
    cluster_layouts.sort_by_key(|cluster| cluster.index());

    let edge_layouts = edge_layouts
        .into_iter()
        .map(|edge| {
            EdgeLayout::new(
                edge.index,
                edge.points,
                edge.curve_points,
                edge.label,
                edge.label_position,
            )
        })
        .collect();

    GraphLayout::from_parts(max_x, max_y, coords, edge_layouts, cluster_layouts)
}

fn sanitize_size((width, height): (f64, f64)) -> (f64, f64) {
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

fn prepare_clusters(
    clusters: &[Cluster],
    node_index: &HashMap<u32, usize>,
    node_count: usize,
) -> Vec<ClusterData> {
    let mut parents = clusters
        .iter()
        .enumerate()
        .map(|(index, cluster)| {
            cluster
                .parent
                .filter(|&parent| parent < clusters.len() && parent != index)
        })
        .collect::<Vec<_>>();

    for index in 0..parents.len() {
        let mut cursor = parents[index];
        let mut seen = HashSet::new();
        seen.insert(index);
        while let Some(parent) = cursor {
            if !seen.insert(parent) {
                parents[index] = None;
                break;
            }
            cursor = parents[parent];
        }
    }

    let mut children = vec![Vec::new(); clusters.len()];
    for (index, parent) in parents.iter().copied().enumerate() {
        if let Some(parent) = parent {
            children[parent].push(index);
        }
    }

    let mut result = Vec::with_capacity(clusters.len());
    for (index, cluster) in clusters.iter().enumerate() {
        let mut members = cluster
            .nodes
            .iter()
            .filter_map(|node| node_index.get(node).copied())
            .filter(|position| *position < node_count)
            .collect::<Vec<_>>();
        members.sort_unstable();
        members.dedup();

        let mut depth = 0;
        let mut cursor = parents[index];
        while let Some(parent) = cursor {
            depth += 1;
            cursor = parents[parent];
        }

        result.push(ClusterData {
            parent: parents[index],
            children: children[index].clone(),
            depth,
            members,
        });
    }
    result
}

fn assign_cluster_paths(nodes: &mut [NodeData], clusters: &[ClusterData]) {
    for (position, node) in nodes.iter_mut().enumerate() {
        let owner = clusters
            .iter()
            .enumerate()
            .filter(|(_, cluster)| cluster.members.binary_search(&position).is_ok())
            .max_by(|(left_index, left), (right_index, right)| {
                left.depth
                    .cmp(&right.depth)
                    .then_with(|| right_index.cmp(left_index))
            })
            .map(|(index, _)| index);

        node.owner = owner;
        if let Some(owner) = owner {
            let mut path = Vec::new();
            let mut cursor = Some(owner);
            while let Some(cluster) = cursor {
                path.push(cluster);
                cursor = clusters[cluster].parent;
            }
            path.reverse();
            node.path = path;
        }
    }
}

fn assign_declaration_order(nodes: &mut [NodeData], clusters: &[ClusterData]) {
    fn declare_cluster(
        cluster: usize,
        nodes: &[NodeData],
        clusters: &[ClusterData],
        seen: &mut [bool],
        order: &mut Vec<usize>,
    ) {
        let member_set = clusters[cluster]
            .members
            .iter()
            .copied()
            .collect::<HashSet<_>>();
        for position in 0..nodes.len() {
            if !seen[position]
                && nodes[position].owner == Some(cluster)
                && member_set.contains(&position)
            {
                seen[position] = true;
                order.push(position);
            }
        }
        for &child in &clusters[cluster].children {
            declare_cluster(child, nodes, clusters, seen, order);
        }
    }

    let mut seen = vec![false; nodes.len()];
    let mut order = Vec::with_capacity(nodes.len());
    for position in 0..nodes.len() {
        if nodes[position].owner.is_none() {
            seen[position] = true;
            order.push(position);
        }
    }
    for cluster in 0..clusters.len() {
        if clusters[cluster].parent.is_none() {
            declare_cluster(cluster, nodes, clusters, &mut seen, &mut order);
        }
    }
    for position in 0..nodes.len() {
        if !seen[position] {
            order.push(position);
        }
    }
    for (declaration_order, position) in order.into_iter().enumerate() {
        nodes[position].declaration_order = declaration_order;
    }
}

fn orient_cycles(nodes: &[NodeData], edges: &mut [WorkEdge]) {
    fn visit(node: usize, colors: &mut [u8], outgoing: &[Vec<usize>], edges: &mut [WorkEdge]) {
        colors[node] = 1;
        for &edge_index in &outgoing[node] {
            if edges[edge_index].tail == edges[edge_index].head {
                continue;
            }
            let head = edges[edge_index].head;
            match colors[head] {
                0 => visit(head, colors, outgoing, edges),
                1 => edges[edge_index].reversed = true,
                _ => {}
            }
        }
        colors[node] = 2;
    }

    let mut outgoing = vec![Vec::new(); nodes.len()];
    for (edge_index, edge) in edges.iter().enumerate() {
        outgoing[edge.tail].push(edge_index);
    }
    for list in &mut outgoing {
        list.sort_by_key(|edge| {
            (
                nodes[edges[*edge].head].declaration_order,
                edges[*edge].index,
            )
        });
    }

    let mut starts = (0..nodes.len()).collect::<Vec<_>>();
    starts.sort_by_key(|node| nodes[*node].declaration_order);
    let mut colors = vec![0; nodes.len()];
    for node in starts {
        if colors[node] == 0 {
            visit(node, &mut colors, &outgoing, edges);
        }
    }
}

fn oriented_endpoints(edge: &WorkEdge) -> (usize, usize) {
    if edge.reversed {
        (edge.head, edge.tail)
    } else {
        (edge.tail, edge.head)
    }
}

fn assign_ranks(nodes: &mut [NodeData], edges: &[WorkEdge], config: &Config) {
    let minimum_length = config.minimum_length as usize;
    let mut outgoing = vec![Vec::new(); nodes.len()];
    let mut incoming = vec![Vec::new(); nodes.len()];
    let mut indegree = vec![0usize; nodes.len()];

    for (edge_index, edge) in edges.iter().enumerate() {
        if edge.tail == edge.head {
            continue;
        }
        let (tail, head) = oriented_endpoints(edge);
        outgoing[tail].push((head, edge_index));
        incoming[head].push((tail, edge_index));
        indegree[head] += 1;
    }

    let mut ready = (0..nodes.len())
        .filter(|node| indegree[*node] == 0)
        .collect::<Vec<_>>();
    ready.sort_by_key(|node| nodes[*node].declaration_order);
    let mut order = Vec::with_capacity(nodes.len());
    while !ready.is_empty() {
        let node = ready.remove(0);
        order.push(node);
        for &(head, _) in &outgoing[node] {
            indegree[head] = indegree[head].saturating_sub(1);
            if indegree[head] == 0 {
                ready.push(head);
                ready.sort_by_key(|candidate| nodes[*candidate].declaration_order);
            }
        }
    }
    if order.len() != nodes.len() {
        let in_order = order.iter().copied().collect::<HashSet<_>>();
        let mut remaining = (0..nodes.len())
            .filter(|node| !in_order.contains(node))
            .collect::<Vec<_>>();
        remaining.sort_by_key(|node| nodes[*node].declaration_order);
        order.extend(remaining);
    }

    let mut ranks = vec![0usize; nodes.len()];
    for &tail in &order {
        for &(head, _) in &outgoing[tail] {
            ranks[head] = ranks[head].max(ranks[tail].saturating_add(minimum_length));
        }
    }

    if config.ranking_type == RankingType::Up {
        let mut to_sink = vec![0usize; nodes.len()];
        for &tail in order.iter().rev() {
            for &(head, _) in &outgoing[tail] {
                to_sink[tail] = to_sink[tail].max(to_sink[head].saturating_add(minimum_length));
            }
        }
        let maximum = to_sink.iter().copied().max().unwrap_or(0);
        for node in 0..nodes.len() {
            ranks[node] = maximum.saturating_sub(to_sink[node]);
        }
    } else if config.ranking_type == RankingType::MinimizeEdgeLength {
        let maximum = ranks.iter().copied().max().unwrap_or(0);
        for _ in 0..8 {
            let mut changed = false;
            for &node in order.iter().chain(order.iter().rev()) {
                let lower = incoming[node]
                    .iter()
                    .map(|(tail, _)| ranks[*tail].saturating_add(minimum_length))
                    .max()
                    .unwrap_or(0);
                let upper = outgoing[node]
                    .iter()
                    .map(|(head, _)| ranks[*head].saturating_sub(minimum_length))
                    .min()
                    .unwrap_or(maximum)
                    .max(lower);
                let derivative = incoming[node].len() as isize - outgoing[node].len() as isize;
                let desired = match derivative.cmp(&0) {
                    Ordering::Less => upper,
                    Ordering::Greater => lower,
                    Ordering::Equal => ranks[node].clamp(lower, upper),
                };
                if desired != ranks[node] {
                    ranks[node] = desired;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        let minimum = ranks.iter().copied().min().unwrap_or(0);
        for rank in &mut ranks {
            *rank = rank.saturating_sub(minimum);
        }
    }

    for (node, rank) in nodes.iter_mut().zip(ranks) {
        node.rank = rank;
    }
}

fn build_layer_graph(
    nodes: &mut [NodeData],
    edges: &mut [WorkEdge],
    config: &Config,
) -> (Vec<LayerItem>, Vec<Vec<usize>>, Vec<(usize, usize)>) {
    let max_rank = nodes.iter().map(|node| node.rank).max().unwrap_or(0);
    let mut layers = vec![Vec::new(); max_rank.saturating_mul(2).saturating_add(1)];
    let mut items = Vec::new();

    let mut node_order = (0..nodes.len()).collect::<Vec<_>>();
    node_order.sort_by_key(|node| nodes[*node].declaration_order);
    for node in node_order {
        let item = items.len();
        nodes[node].item = item;
        let layer = nodes[node].rank * 2;
        items.push(LayerItem {
            kind: ItemKind::Real(node),
            layer,
            width: nodes[node].width,
            path: nodes[node].path.clone(),
            stable_order: nodes[node].declaration_order as f64,
            x: 0.0,
        });
        layers[layer].push(item);
    }

    let mut segments = Vec::new();
    for edge_position in 0..edges.len() {
        let (low_node, high_node) = oriented_endpoints(&edges[edge_position]);
        if low_node == high_node {
            edges[edge_position].chain = vec![nodes[low_node].item];
            continue;
        }
        let low_rank = nodes[low_node].rank;
        let high_rank = nodes[high_node].rank;
        if high_rank <= low_rank {
            edges[edge_position].chain = vec![nodes[low_node].item, nodes[high_node].item];
            continue;
        }

        let mut chain = vec![nodes[low_node].item];
        let common_path = common_prefix(&nodes[low_node].path, &nodes[high_node].path);
        let low_order = nodes[low_node].declaration_order as f64;
        let high_order = nodes[high_node].declaration_order as f64;
        let low_layer = low_rank * 2;
        let high_layer = high_rank * 2;
        let midpoint_layer = low_rank + high_rank;

        for layer in (low_layer + 1)..high_layer {
            let is_label = edges[edge_position].label.is_some() && layer == midpoint_layer;
            let width = if is_label {
                text_width(
                    edges[edge_position].label.as_deref().unwrap_or_default(),
                    EDGE_LABEL_FONT_SIZE,
                ) * POINT_TO_PIXEL
            } else if config.dummy_vertices {
                (config.vertex_spacing * config.dummy_size * 0.02).max(0.0)
            } else {
                0.0
            };
            let fraction = (layer - low_layer) as f64 / (high_layer - low_layer) as f64;
            let stable_order = low_order * (1.0 - fraction)
                + high_order * fraction
                + edges[edge_position].index as f64 * 1.0e-5;
            let item = items.len();
            items.push(LayerItem {
                kind: ItemKind::Virtual {
                    edge: edge_position,
                    label: is_label,
                },
                layer,
                width,
                path: common_path.clone(),
                stable_order,
                x: 0.0,
            });
            layers[layer].push(item);
            chain.push(item);
            if is_label {
                edges[edge_position].label_item = Some(item);
            }
        }
        chain.push(nodes[high_node].item);
        for pair in chain.windows(2) {
            segments.push((pair[0], pair[1]));
        }
        edges[edge_position].chain = chain;
    }

    let stable = items
        .iter()
        .map(|item| item.stable_order)
        .collect::<Vec<_>>();
    for layer in &mut layers {
        *layer = constrained_order(layer, &items, &stable, None);
    }

    (items, layers, segments)
}

fn common_prefix(left: &[usize], right: &[usize]) -> Vec<usize> {
    left.iter()
        .zip(right)
        .take_while(|(left, right)| left == right)
        .map(|(value, _)| *value)
        .collect()
}

#[derive(Default)]
struct OrderGroup {
    entries: Vec<OrderEntry>,
}

enum OrderEntry {
    Item(usize),
    Cluster(usize, OrderGroup),
}

impl OrderGroup {
    fn insert(&mut self, item: usize, path: &[usize]) {
        if let Some((&cluster, rest)) = path.split_first() {
            let position = self.entries.iter().position(
                |entry| matches!(entry, OrderEntry::Cluster(index, _) if *index == cluster),
            );
            let position = position.unwrap_or_else(|| {
                self.entries
                    .push(OrderEntry::Cluster(cluster, OrderGroup::default()));
                self.entries.len() - 1
            });
            if let OrderEntry::Cluster(_, group) = &mut self.entries[position] {
                group.insert(item, rest);
            }
        } else {
            self.entries.push(OrderEntry::Item(item));
        }
    }

    fn sort(&mut self, keys: &[f64], old_positions: &[usize]) {
        for entry in &mut self.entries {
            if let OrderEntry::Cluster(_, group) = entry {
                group.sort(keys, old_positions);
            }
        }
        self.entries.sort_by(|left, right| {
            let (left_key, left_old) = left.key(keys, old_positions);
            let (right_key, right_old) = right.key(keys, old_positions);
            left_key
                .partial_cmp(&right_key)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left_old.partial_cmp(&right_old).unwrap_or(Ordering::Equal))
        });
    }

    fn flatten(&self, output: &mut Vec<usize>) {
        for entry in &self.entries {
            match entry {
                OrderEntry::Item(item) => output.push(*item),
                OrderEntry::Cluster(_, group) => group.flatten(output),
            }
        }
    }

    fn balance_real_cluster_blocks(&mut self, items: &[LayerItem]) {
        for entry in &mut self.entries {
            if let OrderEntry::Cluster(_, group) = entry {
                group.balance_real_cluster_blocks(items);
            }
        }

        let cluster_entries = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| {
                matches!(entry, OrderEntry::Cluster(_, _)) && entry.contains_real(items)
            })
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        let direct_real_count = self
            .entries
            .iter()
            .filter(|entry| {
                matches!(entry, OrderEntry::Item(item) if matches!(items[*item].kind, ItemKind::Real(_)))
            })
            .count();
        if cluster_entries.len() != 1 || direct_real_count < 2 {
            return;
        }

        let cluster_index = cluster_entries[0];
        let real_entries = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| {
                entry.contains_real(items)
                    && (matches!(entry, OrderEntry::Cluster(_, _))
                        || matches!(entry, OrderEntry::Item(_)))
            })
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        let Some(cluster_position) = real_entries
            .iter()
            .position(|entry_index| *entry_index == cluster_index)
        else {
            return;
        };
        if cluster_position != 0 && cluster_position + 1 != real_entries.len() {
            return;
        }

        let cluster_entry = self.entries.remove(cluster_index);
        let direct_before = direct_real_count / 2;
        let mut seen_direct = 0usize;
        let mut insertion = self.entries.len();
        for (index, entry) in self.entries.iter().enumerate() {
            if matches!(entry, OrderEntry::Item(item) if matches!(items[*item].kind, ItemKind::Real(_)))
            {
                seen_direct += 1;
                if seen_direct == direct_before {
                    insertion = index + 1;
                    break;
                }
            }
        }
        self.entries.insert(insertion, cluster_entry);
    }
}

impl OrderEntry {
    fn key(&self, keys: &[f64], old_positions: &[usize]) -> (f64, f64) {
        let mut leaves = Vec::new();
        self.collect(&mut leaves);
        let count = leaves.len().max(1) as f64;
        let key = leaves.iter().map(|item| keys[*item]).sum::<f64>() / count;
        let old = leaves
            .iter()
            .map(|item| old_positions[*item] as f64)
            .sum::<f64>()
            / count;
        (key, old)
    }

    fn collect(&self, output: &mut Vec<usize>) {
        match self {
            OrderEntry::Item(item) => output.push(*item),
            OrderEntry::Cluster(_, group) => {
                for entry in &group.entries {
                    entry.collect(output);
                }
            }
        }
    }

    fn contains_real(&self, items: &[LayerItem]) -> bool {
        match self {
            OrderEntry::Item(item) => matches!(items[*item].kind, ItemKind::Real(_)),
            OrderEntry::Cluster(_, group) => {
                group.entries.iter().any(|entry| entry.contains_real(items))
            }
        }
    }
}

fn constrained_order(
    layer: &[usize],
    items: &[LayerItem],
    keys: &[f64],
    global_positions: Option<&[usize]>,
) -> Vec<usize> {
    let mut positions = vec![usize::MAX; items.len()];
    if let Some(global_positions) = global_positions {
        positions.copy_from_slice(global_positions);
    } else {
        for (position, item) in layer.iter().copied().enumerate() {
            positions[item] = position;
        }
    }
    let mut root = OrderGroup::default();
    for &item in layer {
        root.insert(item, &items[item].path);
    }
    root.sort(keys, &positions);
    let mut result = Vec::with_capacity(layer.len());
    root.flatten(&mut result);
    result
}

fn balance_real_cluster_blocks(layers: &mut [Vec<usize>], items: &[LayerItem]) {
    for (layer_index, layer) in layers.iter_mut().enumerate() {
        if layer_index % 2 != 0 {
            continue;
        }
        let mut root = OrderGroup::default();
        for &item in layer.iter() {
            root.insert(item, &items[item].path);
        }
        root.balance_real_cluster_blocks(items);
        let mut balanced = Vec::with_capacity(layer.len());
        root.flatten(&mut balanced);
        *layer = balanced;
    }
}

fn minimize_crossings(
    layers: &mut [Vec<usize>],
    items: &[LayerItem],
    segments: &[(usize, usize)],
    config: &Config,
) {
    if layers.len() < 2 {
        return;
    }
    let (predecessors, successors) = adjacency(items.len(), segments);
    let mut best = layers.to_vec();
    let mut best_crossings = crossing_count(layers, items, segments);
    let mut best_span = order_span(layers, items, segments);
    let iterations = 12usize.max(layers.len().min(24));

    for _ in 0..iterations {
        for layer_index in 1..layers.len() {
            reorder_layer(
                layers,
                layer_index,
                items,
                &predecessors,
                config.c_minimization,
            );
        }
        for layer_index in (0..layers.len() - 1).rev() {
            reorder_layer(
                layers,
                layer_index,
                items,
                &successors,
                config.c_minimization,
            );
        }

        if config.transpose {
            transpose_equal_groups(layers, items, segments);
        }

        let crossings = crossing_count(layers, items, segments);
        let span = order_span(layers, items, segments);
        if crossings < best_crossings || (crossings == best_crossings && span < best_span) {
            best_crossings = crossings;
            best_span = span;
            best.clone_from_slice(layers);
        }
    }
    layers.clone_from_slice(&best);
}

fn order_span(layers: &[Vec<usize>], items: &[LayerItem], segments: &[(usize, usize)]) -> usize {
    let positions = positions(layers, items.len());
    segments
        .iter()
        .map(|(tail, head)| positions[*tail].abs_diff(positions[*head]))
        .sum()
}

fn adjacency(item_count: usize, segments: &[(usize, usize)]) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let mut predecessors = vec![Vec::new(); item_count];
    let mut successors = vec![Vec::new(); item_count];
    for &(tail, head) in segments {
        successors[tail].push(head);
        predecessors[head].push(tail);
    }
    (predecessors, successors)
}

fn positions(layers: &[Vec<usize>], item_count: usize) -> Vec<usize> {
    let mut positions = vec![usize::MAX; item_count];
    for layer in layers {
        for (position, item) in layer.iter().copied().enumerate() {
            positions[item] = position;
        }
    }
    positions
}

fn reorder_layer(
    layers: &mut [Vec<usize>],
    layer_index: usize,
    items: &[LayerItem],
    neighbors: &[Vec<usize>],
    method: CrossingMinimization,
) {
    if layers[layer_index].len() < 2 {
        return;
    }
    let global_positions = positions(layers, items.len());
    let mut keys = vec![0.0; items.len()];
    for &item in &layers[layer_index] {
        let mut neighbor_positions = neighbors[item]
            .iter()
            .map(|neighbor| global_positions[*neighbor] as f64)
            .collect::<Vec<_>>();
        keys[item] = if neighbor_positions.is_empty() {
            global_positions[item] as f64
        } else {
            match method {
                CrossingMinimization::Barycenter => {
                    neighbor_positions.iter().sum::<f64>() / neighbor_positions.len() as f64
                }
                CrossingMinimization::Median => median(&mut neighbor_positions),
            }
        };
    }
    layers[layer_index] =
        constrained_order(&layers[layer_index], items, &keys, Some(&global_positions));
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    if values.len() % 2 == 1 {
        values[values.len() / 2]
    } else {
        (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
    }
}

fn crossing_count(
    layers: &[Vec<usize>],
    items: &[LayerItem],
    segments: &[(usize, usize)],
) -> usize {
    let positions = positions(layers, items.len());
    let mut by_layer = vec![Vec::new(); layers.len().saturating_sub(1)];
    for &(tail, head) in segments {
        let layer = items[tail].layer.min(items[head].layer);
        if layer < by_layer.len() {
            by_layer[layer].push((tail, head));
        }
    }

    let mut crossings = 0;
    for layer_segments in by_layer {
        for left in 0..layer_segments.len() {
            for right in (left + 1)..layer_segments.len() {
                let (a, b) = layer_segments[left];
                let (c, d) = layer_segments[right];
                if a == c || b == d {
                    continue;
                }
                let left_delta = positions[a] as isize - positions[c] as isize;
                let right_delta = positions[b] as isize - positions[d] as isize;
                if left_delta * right_delta < 0 {
                    crossings += 1;
                }
            }
        }
    }
    crossings
}

fn transpose_equal_groups(
    layers: &mut [Vec<usize>],
    items: &[LayerItem],
    segments: &[(usize, usize)],
) {
    for _ in 0..3 {
        let mut changed = false;
        for layer_index in 0..layers.len() {
            let mut position = 0;
            while position + 1 < layers[layer_index].len() {
                let left = layers[layer_index][position];
                let right = layers[layer_index][position + 1];
                if items[left].path == items[right].path {
                    let before = crossing_count(layers, items, segments);
                    let before_span = order_span(layers, items, segments);
                    layers[layer_index].swap(position, position + 1);
                    let after = crossing_count(layers, items, segments);
                    let after_span = order_span(layers, items, segments);
                    if after < before || (after == before && after_span < before_span) {
                        changed = true;
                    } else {
                        layers[layer_index].swap(position, position + 1);
                    }
                }
                position += 1;
            }
        }
        if !changed {
            break;
        }
    }
}

fn assign_x_coordinates(
    layers: &[Vec<usize>],
    items: &mut [LayerItem],
    segments: &[(usize, usize)],
    config: &Config,
    render_config: &RenderConfig,
    has_labels: bool,
) {
    let node_separation = graphviz_distance(config.vertex_spacing);
    for (layer_index, layer) in layers.iter().enumerate() {
        let mut cursor = 0.0;
        for (position, item) in layer.iter().copied().enumerate() {
            if position == 0 {
                cursor = items[item].width / 2.0;
            } else {
                let previous = layer[position - 1];
                cursor += items[previous].width / 2.0
                    + layer_separation(
                        layer_index,
                        previous,
                        item,
                        items,
                        node_separation,
                        render_config,
                        has_labels,
                    )
                    + items[item].width / 2.0;
            }
            items[item].x = cursor;
        }
        if let (Some(first), Some(last)) = (layer.first(), layer.last()) {
            let center = (items[*first].x - items[*first].width / 2.0
                + items[*last].x
                + items[*last].width / 2.0)
                / 2.0;
            for &item in layer {
                items[item].x -= center;
            }
        }
    }

    let (predecessors, successors) = adjacency(items.len(), segments);
    for _ in 0..32 {
        for layer_index in 1..layers.len() {
            project_toward_neighbors(
                layer_index,
                layers,
                items,
                &predecessors,
                node_separation,
                render_config,
                has_labels,
            );
        }
        for layer_index in (0..layers.len().saturating_sub(1)).rev() {
            project_toward_neighbors(
                layer_index,
                layers,
                items,
                &successors,
                node_separation,
                render_config,
                has_labels,
            );
        }
    }
}

fn graphviz_distance(pixel_distance: f64) -> f64 {
    if !pixel_distance.is_finite() {
        return 0.0;
    }
    (pixel_distance.max(0.0) / POINT_TO_PIXEL).round() * POINT_TO_PIXEL
}

fn layer_separation(
    layer: usize,
    left: usize,
    right: usize,
    items: &[LayerItem],
    node_separation: f64,
    render_config: &RenderConfig,
    has_labels: bool,
) -> f64 {
    let base = if has_labels && layer % 2 == 1 {
        ODD_RANK_SEPARATION
    } else {
        node_separation
    };
    let common = common_prefix_len(&items[left].path, &items[right].path);
    let boundary_count =
        items[left].path.len() + items[right].path.len() - common.saturating_mul(2);
    base + boundary_count as f64 * render_config.cluster_boundary_gap.max(0.0)
}

fn common_prefix_len(left: &[usize], right: &[usize]) -> usize {
    left.iter()
        .zip(right)
        .take_while(|(left, right)| left == right)
        .count()
}

fn project_toward_neighbors(
    layer_index: usize,
    layers: &[Vec<usize>],
    items: &mut [LayerItem],
    neighbors: &[Vec<usize>],
    node_separation: f64,
    render_config: &RenderConfig,
    has_labels: bool,
) {
    let layer = &layers[layer_index];
    if layer.is_empty() {
        return;
    }

    let mut desired = Vec::with_capacity(layer.len());
    for &item in layer {
        let mut targets = neighbors[item]
            .iter()
            .map(|neighbor| items[*neighbor].route_x())
            .collect::<Vec<_>>();
        let target_route = if targets.is_empty() {
            items[item].route_x()
        } else {
            median(&mut targets)
        };
        let own_offset = if items[item].is_label() {
            items[item].width / 2.0
        } else {
            0.0
        };
        desired.push(target_route + own_offset);
    }

    let mut prefix = vec![0.0; layer.len()];
    for position in 1..layer.len() {
        let left = layer[position - 1];
        let right = layer[position];
        prefix[position] = prefix[position - 1]
            + items[left].width / 2.0
            + layer_separation(
                layer_index,
                left,
                right,
                items,
                node_separation,
                render_config,
                has_labels,
            )
            + items[right].width / 2.0;
    }

    #[derive(Clone, Copy)]
    struct Block {
        start: usize,
        end: usize,
        sum: f64,
        weight: f64,
    }
    impl Block {
        fn average(self) -> f64 {
            self.sum / self.weight
        }
    }

    let mut blocks = Vec::<Block>::new();
    for position in 0..layer.len() {
        blocks.push(Block {
            start: position,
            end: position,
            sum: desired[position] - prefix[position],
            weight: 1.0,
        });
        while blocks.len() >= 2 {
            let last = blocks[blocks.len() - 1];
            let previous = blocks[blocks.len() - 2];
            if previous.average() <= last.average() + EPSILON {
                break;
            }
            blocks.pop();
            blocks.pop();
            blocks.push(Block {
                start: previous.start,
                end: last.end,
                sum: previous.sum + last.sum,
                weight: previous.weight + last.weight,
            });
        }
    }

    for block in blocks {
        let value = block.average();
        for position in block.start..=block.end {
            items[layer[position]].x = value + prefix[position];
        }
    }
}

fn assign_y_coordinates(
    nodes: &[NodeData],
    clusters: &[ClusterData],
    edges: &[WorkEdge],
    config: &Config,
    cluster_specs: &[Cluster],
    render_config: &RenderConfig,
) -> (Vec<f64>, Vec<f64>) {
    let max_rank = nodes.iter().map(|node| node.rank).max().unwrap_or(0);
    let has_any_label = edges.iter().any(|edge| edge.label.is_some());
    let mut half_height = vec![0.0_f64; max_rank + 1];
    for node in nodes {
        half_height[node.rank] = half_height[node.rank].max(node.height / 2.0);
    }

    let mut labels_crossing = vec![false; max_rank];
    for edge in edges {
        if edge.label.is_none() || edge.tail == edge.head {
            continue;
        }
        let low = nodes[edge.tail].rank.min(nodes[edge.head].rank);
        let high = nodes[edge.tail].rank.max(nodes[edge.head].rank);
        let midpoint_layer = low + high;
        if midpoint_layer % 2 == 0 {
            let rank = midpoint_layer / 2;
            if rank < half_height.len() {
                half_height[rank] = half_height[rank].max(EDGE_LABEL_HEIGHT / 2.0);
            }
        } else {
            let transition = midpoint_layer / 2;
            if transition < labels_crossing.len() {
                labels_crossing[transition] = true;
            }
        }
    }

    let mut min_rank = vec![usize::MAX; clusters.len()];
    let mut max_cluster_rank = vec![0usize; clusters.len()];
    let mut has_cluster_nodes = vec![false; clusters.len()];
    for (cluster, data) in clusters.iter().enumerate() {
        for &node in &data.members {
            has_cluster_nodes[cluster] = true;
            min_rank[cluster] = min_rank[cluster].min(nodes[node].rank);
            max_cluster_rank[cluster] = max_cluster_rank[cluster].max(nodes[node].rank);
        }
    }
    let mut by_depth = (0..clusters.len()).collect::<Vec<_>>();
    by_depth.sort_by_key(|cluster| std::cmp::Reverse(clusters[*cluster].depth));
    for cluster in by_depth {
        if let Some(parent) = clusters[cluster].parent {
            if has_cluster_nodes[cluster] {
                has_cluster_nodes[parent] = true;
                min_rank[parent] = min_rank[parent].min(min_rank[cluster]);
                max_cluster_rank[parent] = max_cluster_rank[parent].max(max_cluster_rank[cluster]);
            }
        }
    }
    for cluster in 0..clusters.len() {
        if min_rank[cluster] == usize::MAX {
            min_rank[cluster] = 0;
        }
    }

    let rank_separation = graphviz_distance(config.vertex_spacing);
    let mut rank_y = vec![0.0; max_rank + 1];
    for rank in 0..max_rank {
        let mut distance = half_height[rank] + half_height[rank + 1] + rank_separation;
        if labels_crossing[rank] {
            distance += EDGE_LABEL_HEIGHT;
        }

        for cluster in 0..clusters.len() {
            if has_cluster_nodes[cluster] && min_rank[cluster] == rank + 1 {
                let padding = cluster_padding_points(cluster, cluster_specs, render_config);
                let extra_points = if clusters[cluster].parent.is_some() {
                    padding + 23.5
                } else {
                    padding + 22.5
                };
                distance += extra_points * POINT_TO_PIXEL;
            }
            if has_cluster_nodes[cluster]
                && max_cluster_rank[cluster] == rank
                && min_rank[cluster] <= rank
            {
                distance +=
                    cluster_padding_points(cluster, cluster_specs, render_config) * POINT_TO_PIXEL;
            }
        }
        rank_y[rank + 1] = rank_y[rank] + distance;
    }

    let mut layer_y = vec![0.0; max_rank * 2 + 1];
    for rank in 0..=max_rank {
        layer_y[rank * 2] = rank_y[rank];
        if rank < max_rank {
            layer_y[rank * 2 + 1] = if has_any_label {
                rank_y[rank]
                    + half_height[rank]
                    + rank_separation / 2.0
                    + if labels_crossing[rank] {
                        EDGE_LABEL_HEIGHT / 2.0
                    } else {
                        0.0
                    }
            } else {
                (rank_y[rank] + rank_y[rank + 1]) / 2.0
            };
        }
    }
    (rank_y, layer_y)
}

fn cluster_padding_points(
    cluster: usize,
    clusters: &[Cluster],
    render_config: &RenderConfig,
) -> f64 {
    clusters
        .get(cluster)
        .and_then(|cluster| cluster.padding)
        .unwrap_or(render_config.cluster_padding)
        .max(0.0)
}

fn align_unary_cluster_stacks(
    nodes: &[NodeData],
    clusters: &[ClusterData],
    cluster_specs: &[Cluster],
    items: &mut [LayerItem],
    render_config: &RenderConfig,
) -> bool {
    if clusters.is_empty() {
        return false;
    }
    let max_rank = nodes.iter().map(|node| node.rank).max().unwrap_or(0);
    let provisional_y = vec![0.0; max_rank + 1];
    let bounds = compute_all_cluster_bounds(
        nodes,
        clusters,
        cluster_specs,
        items,
        &provisional_y,
        render_config,
    );
    let mut changed = false;

    for cluster in 0..clusters.len() {
        if clusters[cluster].children.len() != 1 {
            continue;
        }
        let direct_nodes = nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| node.owner == Some(cluster))
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        if direct_nodes.len() != 1 {
            continue;
        }
        let node_index = direct_nodes[0];
        let child = clusters[cluster].children[0];
        let desired_x = bounds[child].min_x + nodes[node_index].width / 2.0;
        let delta = desired_x - items[nodes[node_index].item].x;
        if delta.abs() > EPSILON {
            items[nodes[node_index].item].x += delta / 2.0;
            shift_cluster(child, -delta / 2.0, items);
            changed = true;
        }
    }
    changed
}

fn enforce_cluster_separation(
    nodes: &[NodeData],
    clusters: &[ClusterData],
    cluster_specs: &[Cluster],
    layers: &[Vec<usize>],
    rank_y: &[f64],
    items: &mut [LayerItem],
    config: &Config,
    render_config: &RenderConfig,
) {
    let layer_positions = positions(layers, items.len());
    let node_separation = graphviz_distance(config.vertex_spacing);
    let iterations = render_config.cluster_constraint_iterations.max(1);
    for _ in 0..iterations {
        let bounds = compute_all_cluster_bounds(
            nodes,
            clusters,
            cluster_specs,
            items,
            rank_y,
            render_config,
        );
        let mut changed = false;

        for parent in std::iter::once(None).chain((0..clusters.len()).map(Some)) {
            let mut siblings = (0..clusters.len())
                .filter(|cluster| clusters[*cluster].parent == parent)
                .collect::<Vec<_>>();
            siblings.sort_by(|left, right| {
                bounds[*left]
                    .center_x()
                    .partial_cmp(&bounds[*right].center_x())
                    .unwrap_or(Ordering::Equal)
            });
            for pair in siblings.windows(2) {
                let left = pair[0];
                let right = pair[1];
                if !bounds[left].overlaps_y(bounds[right]) {
                    continue;
                }
                let overlap = bounds[left].max_x + render_config.cluster_boundary_gap.max(0.0)
                    - bounds[right].min_x;
                if overlap > EPSILON {
                    shift_cluster(right, overlap, items);
                    changed = true;
                }
            }
        }

        for cluster in 0..clusters.len() {
            let parent = clusters[cluster].parent;
            for (node_index, node) in nodes.iter().enumerate() {
                if node.owner != parent {
                    continue;
                }
                let node_bounds = Bounds {
                    min_x: items[node.item].x - node.width / 2.0,
                    max_x: items[node.item].x + node.width / 2.0,
                    min_y: rank_y[node.rank] - node.height / 2.0,
                    max_y: rank_y[node.rank] + node.height / 2.0,
                };
                if !node_bounds.overlaps_y(bounds[cluster]) {
                    continue;
                }
                let gap = render_config.cluster_boundary_gap.max(0.0);
                if node_bounds.max_x > bounds[cluster].min_x - gap
                    && node_bounds.min_x < bounds[cluster].max_x + gap
                {
                    let node_position = layer_positions[node.item];
                    let same_rank_members = nodes
                        .iter()
                        .filter(|member| member.rank == node.rank && member.path.contains(&cluster))
                        .map(|member| layer_positions[member.item])
                        .filter(|position| *position != usize::MAX)
                        .collect::<Vec<_>>();
                    let member_span = if let (Some(first_member), Some(last_member)) = (
                        same_rank_members.iter().min(),
                        same_rank_members.iter().max(),
                    ) {
                        Some((*first_member, *last_member))
                    } else {
                        None
                    };
                    let place_left = if let Some((first_member, last_member)) = member_span {
                        if node_position < first_member {
                            true
                        } else if node_position > last_member {
                            false
                        } else {
                            items[node.item].x <= bounds[cluster].center_x()
                        }
                    } else {
                        items[node.item].x <= bounds[cluster].center_x()
                    };
                    let flank_offset = member_span
                        .map(|(first_member, last_member)| {
                            nodes
                                .iter()
                                .filter(|other| {
                                    other.owner == parent
                                        && other.rank == node.rank
                                        && if place_left {
                                            let position = layer_positions[other.item];
                                            position > node_position && position < first_member
                                        } else {
                                            let position = layer_positions[other.item];
                                            position < node_position && position > last_member
                                        }
                                })
                                .map(|other| other.width + node_separation)
                                .sum::<f64>()
                        })
                        .unwrap_or(0.0);
                    let delta = if place_left {
                        bounds[cluster].min_x - gap - node_bounds.max_x - flank_offset
                    } else {
                        bounds[cluster].max_x + gap - node_bounds.min_x + flank_offset
                    };
                    if delta.abs() > EPSILON {
                        items[nodes[node_index].item].x += delta;
                        changed = true;
                    }
                }
            }
        }

        if !changed {
            break;
        }
    }
}

fn shift_cluster(cluster: usize, amount: f64, items: &mut [LayerItem]) {
    for item in items {
        if item.path.contains(&cluster) {
            item.x += amount;
        }
    }
}

fn order_top_level_components(
    nodes: &[NodeData],
    clusters: &[ClusterData],
    items: &mut [LayerItem],
    config: &Config,
) {
    let roots = clusters
        .iter()
        .enumerate()
        .filter(|(index, cluster)| {
            cluster.parent.is_none() && nodes.iter().any(|node| node.path.first() == Some(index))
        })
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    if roots.len() < 2 {
        return;
    }

    let mut widths = nodes.iter().map(|node| node.width).collect::<Vec<_>>();
    let typical_width = if widths.is_empty() {
        DEFAULT_NODE_SIZE.0
    } else {
        median(&mut widths)
    };
    let node_separation = graphviz_distance(config.vertex_spacing);
    let first_component_step = typical_width + node_separation * 2.0;
    let sibling_component_step = first_component_step + node_separation / 2.0;
    let rootless_nodes = nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| node.path.is_empty())
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let mut previous_center = if rootless_nodes.is_empty() {
        None
    } else {
        let min_x = rootless_nodes
            .iter()
            .map(|node| items[nodes[*node].item].x - nodes[*node].width / 2.0)
            .fold(f64::INFINITY, f64::min);
        let max_x = rootless_nodes
            .iter()
            .map(|node| items[nodes[*node].item].x + nodes[*node].width / 2.0)
            .fold(f64::NEG_INFINITY, f64::max);
        Some((min_x + max_x) / 2.0)
    };

    for (root_position, root) in roots.into_iter().enumerate() {
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        for node in nodes.iter().filter(|node| node.path.first() == Some(&root)) {
            min_x = min_x.min(items[node.item].x - node.width / 2.0);
            max_x = max_x.max(items[node.item].x + node.width / 2.0);
        }
        if !min_x.is_finite() {
            continue;
        }
        let center = (min_x + max_x) / 2.0;
        if let Some(previous) = previous_center {
            let step = if root_position == 0 && !rootless_nodes.is_empty() {
                first_component_step
            } else {
                sibling_component_step
            };
            let desired = previous + step;
            let delta = desired - center;
            if delta.abs() > EPSILON {
                shift_cluster(root, delta, items);
            }
            previous_center = Some(center + delta);
        } else {
            previous_center = Some(center);
        }
    }
}

fn align_vertical_blocks(nodes: &[NodeData], edges: &[WorkEdge], items: &mut [LayerItem]) {
    #[derive(Clone, Copy)]
    struct Candidate {
        shift: f64,
        edge: usize,
        same_root: bool,
    }

    let max_rank = nodes.iter().map(|node| node.rank).max().unwrap_or(0);
    let multiple_cluster_roots = nodes
        .iter()
        .filter_map(|node| node.path.first().copied())
        .collect::<HashSet<_>>()
        .len()
        > 1;
    let mut rank_nodes = vec![Vec::new(); max_rank + 1];
    let mut rank_degree = vec![0usize; max_rank + 1];
    for (node_index, node) in nodes.iter().enumerate() {
        rank_nodes[node.rank].push(node_index);
    }
    for edge in edges {
        if edge.tail != edge.head {
            rank_degree[nodes[edge.tail].rank] += 1;
            rank_degree[nodes[edge.head].rank] += 1;
        }
    }

    let mut rank_shift = vec![None; max_rank + 1];
    let mut rank_anchor = vec![usize::MAX; max_rank + 1];
    let mut selected_edges = Vec::new();
    while rank_shift
        .iter()
        .enumerate()
        .any(|(rank, shift)| shift.is_none() && !rank_nodes[rank].is_empty())
    {
        let anchor = (0..=max_rank)
            .filter(|rank| rank_shift[*rank].is_none() && !rank_nodes[*rank].is_empty())
            .max_by_key(|rank| (rank_nodes[*rank].len(), rank_degree[*rank], *rank))
            .unwrap_or(0);
        rank_shift[anchor] = Some(0.0);
        rank_anchor[anchor] = anchor;
        let mut frontier = vec![anchor];

        while !frontier.is_empty() {
            let frontier_set = frontier.iter().copied().collect::<HashSet<_>>();
            let mut candidates = vec![Vec::<Candidate>::new(); max_rank + 1];

            for (edge_index, edge) in edges.iter().enumerate() {
                if edge.tail == edge.head {
                    continue;
                }
                let tail_rank = nodes[edge.tail].rank;
                let head_rank = nodes[edge.head].rank;
                if tail_rank == head_rank {
                    continue;
                }

                let pair = if frontier_set.contains(&tail_rank) && rank_shift[head_rank].is_none() {
                    Some((edge.tail, edge.head, head_rank))
                } else if frontier_set.contains(&head_rank) && rank_shift[tail_rank].is_none() {
                    Some((edge.head, edge.tail, tail_rank))
                } else {
                    None
                };
                let Some((aligned, unaligned, unaligned_rank)) = pair else {
                    continue;
                };
                let aligned_rank = nodes[aligned].rank;
                let aligned_x =
                    items[nodes[aligned].item].x + rank_shift[aligned_rank].unwrap_or(0.0);
                let shift = aligned_x - items[nodes[unaligned].item].x;
                let aligned_root = nodes[aligned].path.first().copied();
                let unaligned_root = nodes[unaligned].path.first().copied();
                candidates[unaligned_rank].push(Candidate {
                    shift,
                    edge: edge_index,
                    same_root: aligned_root == unaligned_root,
                });
            }

            let mut next_frontier = Vec::new();
            for rank in 0..=max_rank {
                if candidates[rank].is_empty() || rank_shift[rank].is_some() {
                    continue;
                }
                if !multiple_cluster_roots {
                    candidates[rank].sort_by(|left, right| {
                        left.shift
                            .abs()
                            .partial_cmp(&right.shift.abs())
                            .unwrap_or(Ordering::Equal)
                            .then_with(|| left.edge.cmp(&right.edge))
                    });
                    let candidate = candidates[rank][0];
                    rank_shift[rank] = Some(candidate.shift);
                    selected_edges.push(candidate.edge);
                } else {
                    let mut same_root = candidates[rank]
                        .iter()
                        .copied()
                        .filter(|candidate| candidate.same_root)
                        .collect::<Vec<_>>();
                    if same_root.is_empty() {
                        continue;
                    }
                    same_root.sort_by(|left, right| {
                        left.shift
                            .partial_cmp(&right.shift)
                            .unwrap_or(Ordering::Equal)
                            .then_with(|| left.edge.cmp(&right.edge))
                    });
                    let candidate = same_root[same_root.len() / 2];
                    rank_shift[rank] = Some(candidate.shift);
                    selected_edges.push(candidate.edge);
                }
                rank_anchor[rank] = anchor;
                next_frontier.push(rank);
            }
            frontier = next_frontier;
        }
    }

    for rank in 0..=max_rank {
        if !multiple_cluster_roots
            || rank_nodes[rank].len() != 1
            || rank_anchor[rank] == usize::MAX
            || rank.abs_diff(rank_anchor[rank]) <= 1
        {
            continue;
        }
        let node = rank_nodes[rank][0];
        let root = nodes[node].path.first().copied();
        let mut desired_shifts = edges
            .iter()
            .filter_map(|edge| {
                if edge.tail == edge.head {
                    return None;
                }
                let neighbor = if edge.tail == node {
                    edge.head
                } else if edge.head == node {
                    edge.tail
                } else {
                    return None;
                };
                if nodes[neighbor].path.first().copied() != root {
                    return None;
                }
                Some(
                    items[nodes[neighbor].item].x + rank_shift[nodes[neighbor].rank].unwrap_or(0.0)
                        - items[nodes[node].item].x,
                )
            })
            .collect::<Vec<_>>();
        if desired_shifts.len() >= 2 {
            rank_shift[rank] = Some(median(&mut desired_shifts));
        }
    }

    for node in nodes {
        items[node.item].x += rank_shift[node.rank].unwrap_or(0.0);
    }
    for item in 0..items.len() {
        let ItemKind::Virtual { edge, .. } = items[item].kind else {
            continue;
        };
        let (low_node, high_node) = oriented_endpoints(&edges[edge]);
        if multiple_cluster_roots && nodes[low_node].path.first() != nodes[high_node].path.first() {
            continue;
        }
        let low_layer = nodes[low_node].rank * 2;
        let high_layer = nodes[high_node].rank * 2;
        if high_layer <= low_layer {
            continue;
        }
        let fraction = (items[item].layer - low_layer) as f64 / (high_layer - low_layer) as f64;
        let low_shift = rank_shift[nodes[low_node].rank].unwrap_or(0.0);
        let high_shift = rank_shift[nodes[high_node].rank].unwrap_or(0.0);
        items[item].x += low_shift + (high_shift - low_shift) * fraction;
    }
    selected_edges.sort_unstable();
    selected_edges.dedup();
    for edge_index in selected_edges {
        straighten_edge_chain(&edges[edge_index], nodes, items);
    }
}

fn relax_virtual_items(
    layers: &[Vec<usize>],
    items: &mut [LayerItem],
    segments: &[(usize, usize)],
    config: &Config,
    render_config: &RenderConfig,
    has_labels: bool,
) {
    let (predecessors, successors) = adjacency(items.len(), segments);
    let node_separation = graphviz_distance(config.vertex_spacing);

    for _ in 0..24 {
        let mut desired = Vec::new();
        for (item, data) in items.iter().enumerate() {
            if matches!(data.kind, ItemKind::Real(_)) {
                continue;
            }
            let mut targets = predecessors[item]
                .iter()
                .chain(successors[item].iter())
                .map(|neighbor| items[*neighbor].route_x())
                .collect::<Vec<_>>();
            if targets.is_empty() {
                continue;
            }
            let target = median(&mut targets)
                + if data.is_label() {
                    data.width / 2.0
                } else {
                    0.0
                };
            desired.push((item, target));
        }
        for (item, target) in desired {
            items[item].x = items[item].x * 0.25 + target * 0.75;
        }

        for (layer_index, layer) in layers.iter().enumerate() {
            for _ in 0..3 {
                for pair in layer.windows(2) {
                    let left = pair[0];
                    let right = pair[1];
                    let minimum = items[left].width / 2.0
                        + layer_separation(
                            layer_index,
                            left,
                            right,
                            items,
                            node_separation,
                            render_config,
                            has_labels,
                        )
                        + items[right].width / 2.0;
                    let overlap = items[left].x + minimum - items[right].x;
                    if overlap <= EPSILON {
                        continue;
                    }
                    let left_real = matches!(items[left].kind, ItemKind::Real(_));
                    let right_real = matches!(items[right].kind, ItemKind::Real(_));
                    match (left_real, right_real) {
                        (false, false) => {
                            items[left].x -= overlap / 2.0;
                            items[right].x += overlap / 2.0;
                        }
                        (false, true) => items[left].x -= overlap,
                        (true, false) => items[right].x += overlap,
                        (true, true) => {}
                    }
                }
            }
        }
    }
}

fn straighten_edge_chain(edge: &WorkEdge, nodes: &[NodeData], items: &mut [LayerItem]) {
    if edge.chain.len() < 3 {
        return;
    }
    let (low_node, high_node) = oriented_endpoints(edge);
    let low_layer = nodes[low_node].rank * 2;
    let high_layer = nodes[high_node].rank * 2;
    if high_layer <= low_layer {
        return;
    }
    let low_x = items[nodes[low_node].item].x;
    let high_x = items[nodes[high_node].item].x;

    for &item in edge
        .chain
        .iter()
        .skip(1)
        .take(edge.chain.len().saturating_sub(2))
    {
        let fraction = (items[item].layer - low_layer) as f64 / (high_layer - low_layer) as f64;
        let route_x = low_x + (high_x - low_x) * fraction;
        items[item].x = route_x
            + if items[item].is_label() {
                items[item].width / 2.0
            } else {
                0.0
            };
    }
}

fn shape_long_cluster_routes(
    nodes: &[NodeData],
    edges: &[WorkEdge],
    cluster_bounds: &[Bounds],
    items: &mut [LayerItem],
    config: &Config,
) {
    let lane_step = EDGE_LABEL_HEIGHT + graphviz_distance(config.vertex_spacing);
    let boundary_inset = (graphviz_distance(config.vertex_spacing) - 2.0 * POINT_TO_PIXEL).max(0.0);
    for edge in edges {
        if edge.chain.len() < 3 || edge.tail == edge.head {
            continue;
        }
        let (low_node, high_node) = oriented_endpoints(edge);
        let low_root = nodes[low_node].path.first().copied();
        let high_root = nodes[high_node].path.first().copied();
        let low_layer = nodes[low_node].rank * 2;
        let high_layer = nodes[high_node].rank * 2;
        if high_layer <= low_layer {
            continue;
        }
        let low_x = items[nodes[low_node].item].x;
        let high_x = items[nodes[high_node].item].x;
        let rank_span = nodes[low_node].rank.abs_diff(nodes[high_node].rank);
        let different_roots = low_root.is_some() && high_root.is_some() && low_root != high_root;

        let lane = if different_roots {
            let direction = if high_x >= low_x { 1.0 } else { -1.0 };
            let outward = rank_span.saturating_sub(1) as f64 * lane_step;
            (direction, outward)
        } else {
            if rank_span < 3 {
                continue;
            }
            let common = common_prefix_len(&nodes[low_node].path, &nodes[high_node].path);
            let shorter = nodes[low_node].path.len().min(nodes[high_node].path.len());
            if common != shorter || nodes[low_node].path.len() == nodes[high_node].path.len() {
                continue;
            }
            let longer_path = if nodes[low_node].path.len() > nodes[high_node].path.len() {
                &nodes[low_node].path
            } else {
                &nodes[high_node].path
            };
            let Some(&boundary_cluster) = longer_path.get(common) else {
                continue;
            };
            let Some(boundary) = cluster_bounds.get(boundary_cluster).copied() else {
                continue;
            };

            let tail_x = items[nodes[edge.tail].item].x;
            let head_x = items[nodes[edge.head].item].x;
            let horizontal = tail_x - head_x;
            let side = if horizontal.abs() > boundary_inset / 2.0 {
                horizontal.signum()
            } else if (tail_x + head_x) / 2.0 <= boundary.center_x() {
                -1.0
            } else {
                1.0
            };
            let crosses_root_boundary = common == 0
                && (nodes[low_node].path.is_empty() || nodes[high_node].path.is_empty());
            let lane_x = if side < 0.0 {
                if crosses_root_boundary {
                    boundary.min_x - boundary_inset
                } else {
                    boundary.min_x + boundary_inset
                }
            } else if crosses_root_boundary {
                boundary.max_x + boundary_inset
            } else {
                boundary.max_x - boundary_inset
            };
            (lane_x, 0.0)
        };

        for &item in edge
            .chain
            .iter()
            .skip(1)
            .take(edge.chain.len().saturating_sub(2))
        {
            let fraction = (items[item].layer - low_layer) as f64 / (high_layer - low_layer) as f64;
            let linear_x = low_x + (high_x - low_x) * fraction;
            let route_x = if different_roots {
                let (direction, outward) = lane;
                let bulge = 4.0 * fraction * (1.0 - fraction);
                linear_x + direction * outward * bulge
            } else {
                let blend = (std::f64::consts::PI * fraction).sin().max(0.0).powf(0.75);
                linear_x + (lane.0 - linear_x) * blend
            };
            items[item].x = route_x
                + if items[item].is_label() {
                    items[item].width / 2.0
                } else {
                    0.0
                };
        }
    }
}

fn compute_all_cluster_bounds(
    nodes: &[NodeData],
    clusters: &[ClusterData],
    cluster_specs: &[Cluster],
    items: &[LayerItem],
    rank_y: &[f64],
    render_config: &RenderConfig,
) -> Vec<Bounds> {
    fn compute(
        cluster: usize,
        nodes: &[NodeData],
        clusters: &[ClusterData],
        cluster_specs: &[Cluster],
        items: &[LayerItem],
        rank_y: &[f64],
        render_config: &RenderConfig,
        symmetric_unary_labels: bool,
        memo: &mut [Option<Bounds>],
    ) -> Bounds {
        if let Some(bounds) = memo[cluster] {
            return bounds;
        }

        let mut content = Bounds::empty();
        for node in nodes {
            if node.owner == Some(cluster) {
                let x = items[node.item].x;
                let y = rank_y[node.rank];
                content.include_rect(Bounds {
                    min_x: x - node.width / 2.0,
                    min_y: y - node.height / 2.0,
                    max_x: x + node.width / 2.0,
                    max_y: y + node.height / 2.0,
                });
            }
        }
        for &child in &clusters[cluster].children {
            let child_bounds = compute(
                child,
                nodes,
                clusters,
                cluster_specs,
                items,
                rank_y,
                render_config,
                symmetric_unary_labels,
                memo,
            );
            content.include_rect(child_bounds);
        }

        if content.is_empty() {
            for &node_index in &clusters[cluster].members {
                let node = &nodes[node_index];
                let x = items[node.item].x;
                let y = rank_y[node.rank];
                content.include_rect(Bounds {
                    min_x: x - node.width / 2.0,
                    min_y: y - node.height / 2.0,
                    max_x: x + node.width / 2.0,
                    max_y: y + node.height / 2.0,
                });
            }
        }
        if content.is_empty() {
            content = Bounds {
                min_x: 0.0,
                min_y: 0.0,
                max_x: 0.0,
                max_y: 0.0,
            };
        }

        let padding =
            cluster_padding_points(cluster, cluster_specs, render_config) * POINT_TO_PIXEL;
        let mut bounds = Bounds {
            min_x: content.min_x - padding,
            min_y: content.min_y - padding - CLUSTER_LABEL_ROW,
            max_x: content.max_x + padding,
            max_y: content.max_y + padding,
        };

        let label = cluster_label(cluster, clusters[cluster].parent);
        let minimum_width = text_width(&label, CLUSTER_LABEL_FONT_SIZE) * POINT_TO_PIXEL
            + CLUSTER_LABEL_SIDE_MARGIN * 2.0;
        let direct_node_count = nodes
            .iter()
            .filter(|node| node.owner == Some(cluster))
            .count();
        if bounds.width() < minimum_width {
            let extra = minimum_width - bounds.width();
            if symmetric_unary_labels && direct_node_count + clusters[cluster].children.len() == 1 {
                bounds.min_x -= extra / 2.0;
                bounds.max_x += extra / 2.0;
            } else {
                bounds.min_x -= extra * 0.19;
                bounds.max_x += extra * 0.81;
            }
        }
        if clusters[cluster].parent.is_none()
            && direct_node_count > 1
            && !clusters[cluster].children.is_empty()
        {
            bounds.min_x -= (cluster_padding_points(cluster, cluster_specs, render_config) + 1.0)
                * POINT_TO_PIXEL;
        }
        bounds.min_x = (bounds.min_x / POINT_TO_PIXEL).round() * POINT_TO_PIXEL;
        bounds.max_x = (bounds.max_x / POINT_TO_PIXEL).round() * POINT_TO_PIXEL;

        memo[cluster] = Some(bounds);
        bounds
    }

    let mut memo = vec![None; clusters.len()];
    let symmetric_unary_labels = clusters
        .iter()
        .filter(|cluster| cluster.parent.is_none())
        .count()
        > 1;
    for cluster in 0..clusters.len() {
        compute(
            cluster,
            nodes,
            clusters,
            cluster_specs,
            items,
            rank_y,
            render_config,
            symmetric_unary_labels,
            &mut memo,
        );
    }
    memo.into_iter()
        .map(|bounds| bounds.unwrap_or_else(Bounds::empty))
        .collect()
}

fn cluster_label(index: usize, parent: Option<usize>) -> String {
    match parent {
        Some(parent) => format!("cluster {index} (child of {parent})"),
        None => format!("cluster {index}"),
    }
}

/// A routed edge before it is wrapped in a [`EdgeLayout`].
#[derive(Clone)]
struct RoutedEdge {
    index: usize,
    points: Vec<(f64, f64)>,
    curve_points: Vec<(f64, f64)>,
    label: Option<String>,
    label_position: Option<(f64, f64)>,
}

fn route_edges(
    nodes: &[NodeData],
    edges: &[WorkEdge],
    items: &[LayerItem],
    rank_y: &[f64],
    layer_y: &[f64],
) -> Vec<RoutedEdge> {
    let mut result = Vec::with_capacity(edges.len());
    for edge in edges {
        let label_position = edge.label_item.map(|item| {
            (
                items[item].x,
                layer_y.get(items[item].layer).copied().unwrap_or_default(),
            )
        });

        let (points, curve_points) = if edge.tail == edge.head {
            route_self_edge(&nodes[edge.tail], items, rank_y)
        } else if nodes[edge.tail].rank == nodes[edge.head].rank {
            route_flat_edge(&nodes[edge.tail], &nodes[edge.head], items, rank_y)
        } else {
            route_ranked_edge(edge, nodes, items, rank_y, layer_y)
        };

        let label_position = label_position.or_else(|| {
            edge.label.as_ref().map(|label| {
                let midpoint = point_along_polyline(
                    if curve_points.len() > 2 {
                        &curve_points[2..]
                    } else {
                        &points
                    },
                    0.5,
                );
                (
                    midpoint.0 + text_width(label, EDGE_LABEL_FONT_SIZE) * POINT_TO_PIXEL / 2.0,
                    midpoint.1,
                )
            })
        });

        result.push(RoutedEdge {
            index: edge.index,
            points,
            curve_points,
            label: edge.label.clone(),
            label_position,
        });
    }
    result
}

fn route_ranked_edge(
    edge: &WorkEdge,
    nodes: &[NodeData],
    items: &[LayerItem],
    rank_y: &[f64],
    layer_y: &[f64],
) -> (Vec<(f64, f64)>, Vec<(f64, f64)>) {
    let (low_node, _) = oriented_endpoints(edge);
    let mut chain = edge.chain.clone();
    if edge.tail != low_node {
        chain.reverse();
    }

    let mut centerline = Vec::with_capacity(chain.len());
    for item in chain {
        let y = match items[item].kind {
            ItemKind::Real(node) => rank_y[nodes[node].rank],
            ItemKind::Virtual { edge: owner, .. } => {
                let _ = owner;
                layer_y[items[item].layer]
            }
        };
        centerline.push((items[item].route_x(), y));
    }

    let tail = &nodes[edge.tail];
    let head = &nodes[edge.head];
    let tail_center = (items[tail.item].x, rank_y[tail.rank]);
    let head_center = (items[head.item].x, rank_y[head.rank]);
    if let Some(first) = centerline.first_mut() {
        *first = tail_center;
    }
    if let Some(last) = centerline.last_mut() {
        *last = head_center;
    }

    let first_target = centerline
        .iter()
        .copied()
        .skip(1)
        .find(|point| distance(*point, tail_center) > EPSILON)
        .unwrap_or(head_center);
    let last_source = centerline
        .iter()
        .copied()
        .rev()
        .skip(1)
        .find(|point| distance(*point, head_center) > EPSILON)
        .unwrap_or(tail_center);

    let start = clip_to_rectangle(
        tail_center,
        first_target,
        tail.width / 2.0,
        tail.height / 2.0,
    );
    let end = clip_to_rectangle(
        head_center,
        last_source,
        head.width / 2.0,
        head.height / 2.0,
    );
    let start_direction = unit_vector(tail_center, first_target);
    let end_direction = unit_vector(last_source, head_center);
    let path_start = advance(start, start_direction, TAIL_ARROW_CLEARANCE);
    let path_end = advance(end, end_direction, -HEAD_ARROW_CLEARANCE);

    if let Some(first) = centerline.first_mut() {
        *first = path_start;
    }
    if let Some(last) = centerline.last_mut() {
        *last = path_end;
    }
    let anchors = simplify_polyline(centerline);
    let bezier = catmull_rom_bezier(&anchors);

    let mut curve_points = vec![start, end];
    curve_points.extend(bezier);

    let mut points = vec![start];
    points.extend(
        anchors
            .iter()
            .copied()
            .skip(1)
            .take(anchors.len().saturating_sub(2)),
    );
    points.push(end);
    (simplify_polyline(points), curve_points)
}

fn route_flat_edge(
    tail: &NodeData,
    head: &NodeData,
    items: &[LayerItem],
    rank_y: &[f64],
) -> (Vec<(f64, f64)>, Vec<(f64, f64)>) {
    let tail_center = (items[tail.item].x, rank_y[tail.rank]);
    let head_center = (items[head.item].x, rank_y[head.rank]);
    let direction = if head_center.0 >= tail_center.0 {
        1.0
    } else {
        -1.0
    };
    let start = (tail_center.0 + direction * tail.width / 2.0, tail_center.1);
    let end = (head_center.0 - direction * head.width / 2.0, head_center.1);
    let rise = 24.0 + (head_center.0 - tail_center.0).abs() * 0.12;
    let anchors = vec![
        advance(start, (direction, 0.0), TAIL_ARROW_CLEARANCE),
        ((start.0 + end.0) / 2.0, start.1 - rise),
        advance(end, (direction, 0.0), -HEAD_ARROW_CLEARANCE),
    ];
    let mut curve = vec![start, end];
    curve.extend(catmull_rom_bezier(&anchors));
    (vec![start, anchors[1], end], curve)
}

fn route_self_edge(
    node: &NodeData,
    items: &[LayerItem],
    rank_y: &[f64],
) -> (Vec<(f64, f64)>, Vec<(f64, f64)>) {
    let center = (items[node.item].x, rank_y[node.rank]);
    let start = (center.0 + node.width / 2.0, center.1 - node.height * 0.2);
    let end = (center.0 + node.width / 2.0, center.1 + node.height * 0.2);
    let reach = node.width / 2.0 + 30.0;
    let anchors = vec![
        (start.0 + TAIL_ARROW_CLEARANCE, start.1),
        (center.0 + reach, center.1 - node.height / 2.0),
        (center.0 + reach, center.1 + node.height / 2.0),
        (end.0 + HEAD_ARROW_CLEARANCE, end.1),
    ];
    let mut curve = vec![start, end];
    curve.extend(catmull_rom_bezier(&anchors));
    (vec![start, anchors[1], anchors[2], end], curve)
}

fn clip_to_rectangle(
    center: (f64, f64),
    toward: (f64, f64),
    half_width: f64,
    half_height: f64,
) -> (f64, f64) {
    let dx = toward.0 - center.0;
    let dy = toward.1 - center.1;
    if dx.abs() < EPSILON && dy.abs() < EPSILON {
        return (center.0, center.1 + half_height);
    }
    let tx = if dx.abs() < EPSILON {
        f64::INFINITY
    } else {
        half_width / dx.abs()
    };
    let ty = if dy.abs() < EPSILON {
        f64::INFINITY
    } else {
        half_height / dy.abs()
    };
    let scale = tx.min(ty);
    (center.0 + dx * scale, center.1 + dy * scale)
}

fn unit_vector(from: (f64, f64), to: (f64, f64)) -> (f64, f64) {
    let dx = to.0 - from.0;
    let dy = to.1 - from.1;
    let length = dx.hypot(dy);
    if length <= EPSILON {
        (0.0, 1.0)
    } else {
        (dx / length, dy / length)
    }
}

fn advance(point: (f64, f64), direction: (f64, f64), amount: f64) -> (f64, f64) {
    (
        point.0 + direction.0 * amount,
        point.1 + direction.1 * amount,
    )
}

fn simplify_polyline(points: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    let mut result = Vec::with_capacity(points.len());
    for point in points {
        if result
            .last()
            .is_some_and(|previous| distance(*previous, point) < 0.5)
        {
            continue;
        }
        while result.len() >= 2 {
            let a = result[result.len() - 2];
            let b = result[result.len() - 1];
            if point_line_distance(b, a, point) < 1.0
                && (b.0 - a.0) * (point.0 - b.0) + (b.1 - a.1) * (point.1 - b.1) >= -EPSILON
            {
                result.pop();
            } else {
                break;
            }
        }
        result.push(point);
    }
    result
}

fn point_line_distance(point: (f64, f64), start: (f64, f64), end: (f64, f64)) -> f64 {
    let dx = end.0 - start.0;
    let dy = end.1 - start.1;
    let length_squared = dx * dx + dy * dy;
    if length_squared <= EPSILON {
        return distance(point, start);
    }
    let t =
        (((point.0 - start.0) * dx + (point.1 - start.1) * dy) / length_squared).clamp(0.0, 1.0);
    distance(point, (start.0 + t * dx, start.1 + t * dy))
}

fn catmull_rom_bezier(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    if points.len() < 2 {
        return points.to_vec();
    }
    let mut result = Vec::with_capacity(1 + (points.len() - 1) * 3);
    result.push(points[0]);
    for index in 0..points.len() - 1 {
        let previous = if index == 0 {
            points[index]
        } else {
            points[index - 1]
        };
        let current = points[index];
        let next = points[index + 1];
        let after = if index + 2 < points.len() {
            points[index + 2]
        } else {
            next
        };
        let control_1 = if index == 0 {
            (
                current.0 + (next.0 - current.0) / 3.0,
                current.1 + (next.1 - current.1) / 3.0,
            )
        } else {
            (
                current.0 + (next.0 - previous.0) / 6.0,
                current.1 + (next.1 - previous.1) / 6.0,
            )
        };
        let control_2 = if index + 1 == points.len() - 1 {
            (
                next.0 - (next.0 - current.0) / 3.0,
                next.1 - (next.1 - current.1) / 3.0,
            )
        } else {
            (
                next.0 - (after.0 - current.0) / 6.0,
                next.1 - (after.1 - current.1) / 6.0,
            )
        };
        result.push(control_1);
        result.push(control_2);
        result.push(next);
    }
    result
}

fn point_along_polyline(points: &[(f64, f64)], fraction: f64) -> (f64, f64) {
    if points.is_empty() {
        return (0.0, 0.0);
    }
    let total = points
        .windows(2)
        .map(|pair| distance(pair[0], pair[1]))
        .sum::<f64>();
    if total <= EPSILON {
        return points[0];
    }
    let target = total * fraction.clamp(0.0, 1.0);
    let mut traversed = 0.0;
    for pair in points.windows(2) {
        let length = distance(pair[0], pair[1]);
        if traversed + length >= target {
            let local = (target - traversed) / length.max(EPSILON);
            return (
                pair[0].0 + (pair[1].0 - pair[0].0) * local,
                pair[0].1 + (pair[1].1 - pair[0].1) * local,
            );
        }
        traversed += length;
    }
    *points.last().unwrap_or(&(0.0, 0.0))
}

fn distance(left: (f64, f64), right: (f64, f64)) -> f64 {
    (left.0 - right.0).hypot(left.1 - right.1)
}

fn text_width(text: &str, font_size: f64) -> f64 {
    text.chars().map(times_roman_advance).sum::<f64>() * font_size / 1000.0
}

fn times_roman_advance(character: char) -> f64 {
    match character {
        ' ' => 250.0,
        '!' => 333.0,
        '"' => 408.0,
        '#' => 500.0,
        '$' => 500.0,
        '%' => 833.0,
        '&' => 778.0,
        '\'' => 180.0,
        '(' | ')' => 333.0,
        '*' => 500.0,
        '+' => 564.0,
        ',' | '.' => 250.0,
        '-' => 333.0,
        '/' => 278.0,
        '0'..='9' => 500.0,
        ':' | ';' => 278.0,
        '<' | '=' | '>' => 564.0,
        '?' => 444.0,
        '@' => 921.0,
        'A' | 'V' | 'W' | 'Y' => 722.0,
        'B' | 'R' => 667.0,
        'C' => 667.0,
        'D' | 'O' | 'Q' => 722.0,
        'E' | 'F' | 'L' | 'P' | 'S' | 'T' => 611.0,
        'G' | 'H' | 'K' | 'N' | 'U' | 'X' => 722.0,
        'I' => 333.0,
        'J' => 389.0,
        'M' => 889.0,
        'Z' => 611.0,
        '[' | ']' => 333.0,
        '\\' => 278.0,
        '^' => 469.0,
        '_' => 500.0,
        '`' => 333.0,
        'a' | 'c' | 'e' => 444.0,
        'b' | 'd' | 'g' | 'h' | 'n' | 'o' | 'p' | 'q' | 'u' => 500.0,
        'f' | 'r' => 333.0,
        'i' | 'j' | 'l' | 't' => 278.0,
        'k' | 'v' | 'x' | 'y' => 500.0,
        's' => 389.0,
        'm' => 778.0,
        'w' => 722.0,
        'z' => 444.0,
        '{' | '}' => 480.0,
        '|' => 200.0,
        '~' => 541.0,
        _ => 500.0,
    }
}
