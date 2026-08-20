use std::collections::{HashMap, VecDeque};
use std::f64::consts::{PI, TAU};

use crate::layout_engine::{GraphLayout, LayoutInput};

// ── Data structures ──────────────────────────────────────────────────────

/// A block (biconnected component) in the block-cutpoint tree.
#[derive(Clone)]
struct Block {
    /// Nodes in this block (by index into the original node list).
    nodes: Vec<usize>,
    /// Edges internal to this block as (src_idx, dst_idx) pairs.
    edges: Vec<(usize, usize)>,
    /// The articulation point (cut vertex) that connects this block to its parent,
    /// if any. This is an index into `nodes`.
    child: Option<usize>,
    /// Children in the block tree.
    children: Vec<Block>,
    /// Radius of this block's circle (including subtree radii).
    radius: f64,
    /// Radius of just this block (not including children).
    rad0: f64,
    /// Ordered list of nodes around the circle (indices into `nodes`).
    circle_list: Vec<usize>,
    /// If block has 1 node, the angle to place the parent.
    parent_pos: Option<f64>,
    /// Whether this block has been coalesced with its only child.
    coalesced: bool,
}

/// State for the circular layout algorithm.
struct CircState {
    /// Minimum distance between nodes.
    min_dist: f64,
    /// Size of each node, indexed by position in the input node list.
    node_sizes: Vec<(f64, f64)>,
}

/// The extent of a node along any direction: the larger of its width and
/// height. Degenerate (non-finite or non-positive) dimensions are treated as
/// zero so they can't poison the layout.
fn node_diameter((width, height): (f64, f64)) -> f64 {
    let width = if width.is_finite() && width > 0.0 { width } else { 0.0 };
    let height = if height.is_finite() && height > 0.0 { height } else { 0.0 };
    width.max(height)
}

// ── DFS state for block decomposition ────────────────────────────────────

struct BlockFinder {
    val: Vec<u32>,
    low: Vec<u32>,
    parent: Vec<Option<usize>>,
    block_of: Vec<Option<usize>>,
    blocks: Vec<Block>,
    edge_stack: Vec<(usize, usize)>,
    order: u32,
}

impl BlockFinder {
    fn new(n: usize) -> Self {
        Self {
            val: vec![0; n],
            low: vec![0; n],
            parent: vec![None; n],
            block_of: vec![None; n],
            blocks: Vec::new(),
            edge_stack: Vec::new(),
            order: 1,
        }
    }

    fn find_blocks(&mut self, adj: &[Vec<usize>], root: usize) {
        self.dfs(root, adj, true);
        if self.block_of[root].is_none() {
            let idx = self.blocks.len();
            self.blocks.push(Block {
                nodes: vec![root],
                edges: Vec::new(),
                child: None,
                children: Vec::new(),
                radius: 0.0,
                rad0: 0.0,
                circle_list: Vec::new(),
                parent_pos: None,
                coalesced: false,
            });
            self.block_of[root] = Some(idx);
        }
    }

    fn dfs(&mut self, u: usize, adj: &[Vec<usize>], is_root: bool) {
        self.val[u] = self.order;
        self.low[u] = self.order;
        self.order += 1;

        for &v in &adj[u] {
            if self.val[v] == 0 {
                self.parent[v] = Some(u);
                self.edge_stack.push((u, v));
                self.dfs(v, adj, false);
                self.low[u] = self.low[u].min(self.low[v]);

                if self.low[v] >= self.val[u] {
                    self.pop_block(u, v);
                }
            } else if self.parent[u] != Some(v) {
                self.low[u] = self.low[u].min(self.val[v]);
                if self.val[v] < self.val[u] {
                    self.edge_stack.push((u, v));
                }
            }
        }

        if is_root && self.block_of[u].is_none() {
            let idx = self.blocks.len();
            self.blocks.push(Block {
                nodes: vec![u],
                edges: Vec::new(),
                child: None,
                children: Vec::new(),
                radius: 0.0,
                rad0: 0.0,
                circle_list: Vec::new(),
                parent_pos: None,
                coalesced: false,
            });
            self.block_of[u] = Some(idx);
        }
    }

    fn pop_block(&mut self, u: usize, v: usize) {
        let mut block_nodes: Vec<usize> = Vec::new();
        let mut block_edges: Vec<(usize, usize)> = Vec::new();

        loop {
            let Some((a, b)) = self.edge_stack.pop() else { break };
            block_edges.push((a, b));

            for &node in &[a, b] {
                if self.block_of[node].is_none() {
                    block_nodes.push(node);
                }
            }

            if a == u && b == v {
                break;
            }
        }

        block_nodes.sort();
        block_nodes.dedup();

        if block_nodes.is_empty() {
            return;
        }

        let idx = self.blocks.len();
        let mut block = Block {
            nodes: block_nodes.clone(),
            edges: block_edges,
            child: None,
            children: Vec::new(),
            radius: 0.0,
            rad0: 0.0,
            circle_list: Vec::new(),
            parent_pos: None,
            coalesced: false,
        };

        if block.nodes.len() > 1 && !block.nodes.contains(&u) {
            block.nodes.push(u);
        }

        for &node in &block.nodes {
            self.block_of[node] = Some(idx);
        }

        self.blocks.push(block);
    }
}

// ── Block tree construction ──────────────────────────────────────────────

fn build_block_tree(adj: &[Vec<usize>]) -> Option<Block> {
    let n = adj.len();
    let mut finder = BlockFinder::new(n);

    let root = (0..n).find(|&i| !adj[i].is_empty()).unwrap_or(0);
    finder.find_blocks(adj, root);

    if finder.blocks.is_empty() {
        return None;
    }

    let blocks = finder.blocks;
    let block_of = finder.block_of;
    let val = finder.val;

    let root_block = block_of[root].unwrap_or(0);

    let mut block_parent = vec![None; blocks.len()];
    let mut block_child_node = vec![None; blocks.len()];

    for bi in 0..blocks.len() {
        if bi == root_block {
            continue;
        }
        let mut min_val = u32::MAX;
        let mut best_node = 0;
        for &node in &blocks[bi].nodes {
            if val[node] < min_val {
                min_val = val[node];
                best_node = node;
            }
        }
        let parent_node = finder.parent[best_node];
        if let Some(pn) = parent_node {
            if let Some(pb) = block_of[pn] {
                block_parent[bi] = Some(pb);
                block_child_node[bi] = Some(best_node);
            }
        }
    }

    let mut children_map: HashMap<usize, Vec<usize>> = HashMap::new();
    for bi in 0..blocks.len() {
        if bi == root_block {
            continue;
        }
        if let Some(pb) = block_parent[bi] {
            children_map.entry(pb).or_default().push(bi);
        } else {
            children_map.entry(root_block).or_default().push(bi);
        }
    }

    fn build_subtree(idx: usize, blocks: &[Block], children: &HashMap<usize, Vec<usize>>, child_node: &[Option<usize>]) -> Block {
        let mut block = blocks[idx].clone();
        let child_indices = children.get(&idx).cloned().unwrap_or_default();
        for &child_idx in &child_indices {
            let mut child = build_subtree(child_idx, blocks, children, child_node);
            child.child = child_node[child_idx];
            block.children.push(child);
        }
        block
    }

    Some(build_subtree(root_block, &blocks, &children_map, &block_child_node))
}

// ── Spanning tree and longest path ───────────────────────────────────────

fn spanning_tree(nodes: &[usize], edges: &[(usize, usize)]) -> Vec<Option<usize>> {
    let n = nodes.len();
    if n == 0 {
        return Vec::new();
    }

    let global_to_local: HashMap<usize, usize> = nodes.iter().copied().enumerate().map(|(i, g)| (g, i)).collect();
    let mut local_adj: Vec<Vec<usize>> = vec![Vec::new(); n];

    for &(a, b) in edges {
        if let Some(&la) = global_to_local.get(&a) {
            if let Some(&lb) = global_to_local.get(&b) {
                local_adj[la].push(lb);
                local_adj[lb].push(la);
            }
        }
    }

    let mut parent: Vec<Option<usize>> = vec![None; n];
    let mut visited = vec![false; n];
    let mut queue = VecDeque::new();

    visited[0] = true;
    queue.push_back(0);

    while let Some(cur) = queue.pop_front() {
        for &next in &local_adj[cur] {
            if !visited[next] {
                visited[next] = true;
                parent[next] = Some(cur);
                queue.push_back(next);
            }
        }
    }

    parent
}

fn longest_path_in_tree(parent: &[Option<usize>]) -> Vec<usize> {
    let n = parent.len();
    if n == 0 {
        return Vec::new();
    }

    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut root = 0;
    for (i, &p) in parent.iter().enumerate() {
        if let Some(par) = p {
            children[par].push(i);
        } else {
            root = i;
        }
    }

    fn dfs_farthest(u: usize, children: &[Vec<usize>]) -> (usize, usize) {
        let mut farthest = u;
        let mut max_depth = 0;
        let mut stack = vec![(u, 0, 0)];
        while let Some((node, par, depth)) = stack.pop() {
            if depth > max_depth {
                max_depth = depth;
                farthest = node;
            }
            for &child in &children[node] {
                if child != par {
                    stack.push((child, node, depth + 1));
                }
            }
        }
        (farthest, max_depth)
    }

    let (endpoint, _) = dfs_farthest(root, &children);

    fn dfs_path(u: usize, target: usize, children: &[Vec<usize>], visited: &mut [bool], path: &mut Vec<usize>) -> bool {
        if u == target {
            path.push(u);
            return true;
        }
        visited[u] = true;
        path.push(u);
        for &child in &children[u] {
            if !visited[child] {
                if dfs_path(child, target, children, visited, path) {
                    return true;
                }
            }
        }
        path.pop();
        false
    }

    let mut visited = vec![false; n];
    let mut path = Vec::new();
    dfs_path(root, endpoint, &children, &mut visited, &mut path);

    path
}

// ── Node placement ───────────────────────────────────────────────────────

fn place_residual_nodes(block_nodes: &[usize], edges: &[(usize, usize)], path: &mut Vec<usize>) {
    let n = block_nodes.len();

    let mut on_path = vec![false; n];
    for &idx in path.iter() {
        if idx < n {
            on_path[idx] = true;
        }
    }

    let mut neighbor_of: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(a, b) in edges {
        if a < n && b < n {
            neighbor_of[a].push(b);
            neighbor_of[b].push(a);
        }
    }

    let mut remaining: Vec<usize> = (0..n).filter(|&i| !on_path[i]).collect();

    while let Some(node) = remaining.pop() {
        if on_path[node] {
            continue;
        }

        let neighbors: Vec<usize> = neighbor_of[node]
            .iter()
            .copied()
            .filter(|&nb| on_path[nb])
            .collect();

        let mut placed = false;

        if neighbors.len() >= 2 {
            for pos in 0..path.len() {
                let next = (pos + 1) % path.len();
                if neighbors.contains(&path[pos]) && neighbors.contains(&path[next]) {
                    path.insert(pos + 1, node);
                    placed = true;
                    break;
                }
            }
        }

        if !placed && !neighbors.is_empty() {
            for pos in 0..path.len() {
                if neighbors.contains(&path[pos]) {
                    path.insert(pos + 1, node);
                    placed = true;
                    break;
                }
            }
        }

        if !placed {
            path.push(node);
        }

        on_path[node] = true;
    }
}

// ── Edge crossing counting and reduction ─────────────────────────────────

fn count_crossings(list: &[usize], sub_edges: &[(usize, usize)]) -> usize {
    let mut crossings = 0;
    let n = list.len();
    if n < 2 {
        return 0;
    }

    let pos: HashMap<usize, usize> = list.iter().copied().enumerate().map(|(i, n)| (n, i)).collect();

    for i in 0..sub_edges.len() {
        let (a1, b1) = sub_edges[i];
        let p1a = pos.get(&a1).copied().unwrap_or(usize::MAX);
        let p1b = pos.get(&b1).copied().unwrap_or(usize::MAX);
        if p1a == usize::MAX || p1b == usize::MAX {
            continue;
        }
        let (lo1, hi1) = (p1a.min(p1b), p1a.max(p1b));

        for j in (i + 1)..sub_edges.len() {
            let (a2, b2) = sub_edges[j];
            let p2a = pos.get(&a2).copied().unwrap_or(usize::MAX);
            let p2b = pos.get(&b2).copied().unwrap_or(usize::MAX);
            if p2a == usize::MAX || p2b == usize::MAX {
                continue;
            }
            let (lo2, hi2) = (p2a.min(p2b), p2a.max(p2b));

            if (lo2 > lo1 && lo2 < hi1 && hi2 > hi1) || (hi2 > lo1 && hi2 < hi1 && lo2 < lo1) {
                crossings += 1;
            }
        }
    }

    crossings
}

fn reduce_crossings(list: &mut Vec<usize>, sub_edges: &[(usize, usize)]) {
    let mut crossings = count_crossings(list, sub_edges);
    if crossings == 0 {
        return;
    }

    let n = list.len();
    for _iter in 0..10 {
        let _orig = crossings;
        let mut improved = false;

        for i in 0..n {
            let node = list[i];
            let neighbors: Vec<usize> = sub_edges
                .iter()
                .filter(|(a, b)| *a == node || *b == node)
                .flat_map(|(a, b)| if *a == node { Some(*b) } else { Some(*a) })
                .collect();

            for &nb in &neighbors {
                if let Some(pos) = list.iter().position(|&x| x == nb) {
                    let mut new_list = list.clone();
                    new_list.remove(i);
                    let insert_pos = if pos < i { pos } else { pos - 1 };
                    new_list.insert(insert_pos, node);

                    let new_crossings = count_crossings(&new_list, sub_edges);
                    if new_crossings < crossings {
                        *list = new_list;
                        crossings = new_crossings;
                        improved = true;
                        break;
                    }
                }
            }
            if improved {
                break;
            }
        }

        if !improved || crossings == 0 {
            break;
        }
    }
}

// ── Block layout (place nodes on a circle) ───────────────────────────────

fn layout_block(block: &mut Block, state: &CircState) {
    let n = block.nodes.len();
    if n == 0 {
        return;
    }

    let parent = spanning_tree(&block.nodes, &block.edges);
    let mut path = longest_path_in_tree(&parent);

    place_residual_nodes(&block.nodes, &block.edges, &mut path);

    reduce_crossings(&mut path, &block.edges);

    // Account for node sizes: radius = N * (min_dist + largest_node) / (2 * PI),
    // where largest_node is the biggest max(width, height) in this block so
    // adjacent nodes on the circle never overlap.
    let largest_node = block
        .nodes
        .iter()
        .map(|&i| node_diameter(state.node_sizes[i]))
        .fold(0.0, f64::max);
    let radius = if path.len() <= 1 {
        0.0
    } else {
        let circumference = path.len() as f64 * (state.min_dist + largest_node);
        circumference / TAU
    };

    block.circle_list = path;
    block.radius = if n <= 1 { (state.min_dist + largest_node) / 2.0 } else { radius };
    block.rad0 = block.radius;
    block.parent_pos = None;
}

// ── Child block positioning ──────────────────────────────────────────────

fn get_rotation(child: &Block, _x: f64, _y: f64, theta: f64) -> f64 {
    if let Some(pp) = child.parent_pos {
        let mut angle = theta + PI - pp;
        if angle < 0.0 {
            angle += TAU;
        }
        return angle;
    }

    let count = child.circle_list.len();
    if count == 2 {
        return theta - PI / 2.0;
    }

    let _neighbor = match child.child {
        Some(idx) => idx,
        None => return 0.0,
    };

    0.0
}

fn apply_delta(_block: &mut Block, _x: f64, _y: f64, _rotate: f64) {
}

fn position_children(block: &mut Block, state: &CircState) {
    let child_count = block.children.len();
    if child_count == 0 {
        return;
    }

    let length = block.circle_list.len();
    let node_angle = if length > 0 { TAU / (length as f64) } else { TAU / child_count as f64 };

    let mut parent_nodes: Vec<(usize, usize)> = Vec::new();
    for (ci, child) in block.children.iter().enumerate() {
        if let Some(cn) = child.child {
            if let Some(pos) = block.circle_list.iter().position(|&x| x == cn) {
                parent_nodes.push((pos, ci));
            }
        }
    }

    if parent_nodes.is_empty() {
        let child_indices: Vec<usize> = (0..child_count).collect();
        for &ci in &child_indices {
            let child = &mut block.children[ci];
            let angle = ci as f64 * TAU / child_count as f64;
            let child_radius = child.radius;
            let r = block.radius + child_radius + state.min_dist;
            let dx = r * angle.cos();
            let dy = r * angle.sin();
            apply_delta(child, dx, dy, 0.0);
        }
        let max_child_r = block.children.iter().map(|c| c.radius).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(0.0);
        if child_count == 1 {
            block.radius += state.min_dist / 2.0 + max_child_r;
            block.coalesced = true;
        } else {
            block.radius += max_child_r;
        }
        return;
    }

    let mut max_radius: f64 = 0.0;
    for &(path_idx, ci) in &parent_nodes {
        let child = &mut block.children[ci];
        let theta = path_idx as f64 * node_angle;
        let child_radius = child.radius;
        let r = block.radius + child_radius + state.min_dist;
        let dx = r * theta.cos();
        let dy = r * theta.sin();
        let rotate = get_rotation(child, dx, dy, theta);
        apply_delta(child, dx, dy, rotate);
        max_radius = max_radius.max(child_radius);
    }

    if child_count == 1 {
        block.radius += state.min_dist / 2.0 + max_radius;
        block.coalesced = true;
    } else {
        block.radius += max_radius;
    }
}

// ── Main recursive layout ────────────────────────────────────────────────

fn do_block(block: &mut Block, state: &CircState) {
    for i in 0..block.children.len() {
        do_block(&mut block.children[i], state);
    }
    layout_block(block, state);
    position_children(block, state);
}

fn collect_positions(
    block: &Block,
    offset_x: f64,
    offset_y: f64,
    rotation: f64,
    coords: &mut std::collections::BTreeMap<usize, (f64, f64)>,
    all_nodes: &[u32],
    min_dist: f64,
) {
    let n = block.circle_list.len();
    if n == 0 {
        return;
    }

    let cos_r = rotation.cos();
    let sin_r = rotation.sin();

    let radius = block.radius;
    for (i, &node_local) in block.circle_list.iter().enumerate() {
        let theta = i as f64 * TAU / n as f64;
        let local_x = radius * theta.cos();
        let local_y = radius * theta.sin();

        let rx = local_x * cos_r - local_y * sin_r;
        let ry = local_x * sin_r + local_y * cos_r;

        let global_x = rx + offset_x;
        let global_y = ry + offset_y;

        coords.insert(node_local, (global_x, global_y));
    }

    for child in &block.children {
        let child_radius: f64 = block.radius + child.radius + min_dist;
        let child_theta: f64 = 0.0;
        let child_x = child_radius * child_theta.cos();
        let child_y = child_radius * child_theta.sin();

        let cx = child_x * cos_r - child_y * sin_r + offset_x;
        let cy = child_x * sin_r + child_y * cos_r + offset_y;

        collect_positions(child, cx, cy, rotation, coords, all_nodes, min_dist);
    }
}

/// Compute a circular layout using the circo algorithm.
///
/// This implements the circular layout algorithm from Graphviz's circo
/// engine. Nodes are arranged on circles representing biconnected components
/// (blocks), connected via articulation points.
///
/// Reference: Six and Tollis, "A Framework for Circular Drawings of Networks",
/// GD '99, LNCS 1731, pp. 107-116.
pub fn circo_layout<'a>(input: &LayoutInput<'a>) -> GraphLayout {
    let nodes: &[u32] = &input.nodes;
    let edges: &[(u32, u32)] = &input.edges;

    if nodes.is_empty() {
        return GraphLayout::from_parts(0.0, 1.0, std::collections::BTreeMap::new(), Vec::new(), Vec::new());
    }

    let node_count = nodes.len();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); node_count];

    let node_index: HashMap<u32, usize> = nodes.iter().copied().enumerate().map(|(i, n)| (n, i)).collect();

    for &(from, to) in edges {
        if let Some(&fi) = node_index.get(&from) {
            if let Some(&ti) = node_index.get(&to) {
                adj[fi].push(ti);
                adj[ti].push(fi);
            }
        }
    }

    let node_sizes: Vec<(f64, f64)> = nodes.iter().map(|&node| (input.node_size)(node)).collect();

    let state = CircState {
        min_dist: 120.0,
        node_sizes,
    };

    let mut root = match build_block_tree(&adj) {
        Some(r) => r,
        None => {
            return GraphLayout::from_parts(0.0, 1.0, std::collections::BTreeMap::new(), Vec::new(), Vec::new());
        }
    };

    do_block(&mut root, &state);

    let mut coords: std::collections::BTreeMap<usize, (f64, f64)> = std::collections::BTreeMap::new();
    collect_positions(&root, 0.0, 0.0, 0.0, &mut coords, nodes, state.min_dist);

    if coords.is_empty() {
        return GraphLayout::from_parts(0.0, 1.0, coords, Vec::new(), Vec::new());
    }

    // Circo places nodes on circles centered at the origin, so raw coordinates
    // span [-R, R] in each axis. The rest of the crate expects layout
    // coordinates to start at the origin with max_x/max_y describing the total
    // width/height (see the default Sugiyama and microdot layouts); autofit
    // relies on that convention. Compute the full bounds including node
    // extents, then shift everything into the positive quadrant.
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for (index, &(x, y)) in &coords {
        let (width, height) = nodes
            .get(*index)
            .map(|&node| (input.node_size)(node))
            .unwrap_or((0.0, 0.0));
        min_x = min_x.min(x - width / 2.0);
        max_x = max_x.max(x + width / 2.0);
        min_y = min_y.min(y - height / 2.0);
        max_y = max_y.max(y + height / 2.0);
    }

    for (_, (x, y)) in coords.iter_mut() {
        *x -= min_x;
        *y -= min_y;
    }

    let layout_width = (max_x - min_x).max(1.0);
    let layout_height = (max_y - min_y).max(1.0);

    let edge_layouts: Vec<_> = edges
        .iter()
        .enumerate()
        .map(|(i, &(from, to))| {
            let from_pos = coords.get(&(from as usize)).copied().unwrap_or((0.0, 0.0));
            let to_pos = coords.get(&(to as usize)).copied().unwrap_or((0.0, 0.0));
            let points = vec![from_pos, to_pos];
            let label = (input.edge_label)(i, (from, to));
            rust_sugiyama::EdgeLayout::new(i, points.clone(), points, label, None)
        })
        .collect();

    let cluster_layouts = Vec::new();

    GraphLayout::from_parts(layout_width, layout_height, coords, edge_layouts, cluster_layouts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn circo_input(nodes: Vec<u32>, edges: Vec<(u32, u32)>) -> LayoutInput<'static> {
        circo_input_with_size(nodes, edges, (100.0, 40.0))
    }

    fn circo_input_with_size(
        nodes: Vec<u32>,
        edges: Vec<(u32, u32)>,
        size: (f64, f64),
    ) -> LayoutInput<'static> {
        let clusters: Arc<[crate::layout_engine::Cluster]> = Arc::from(Vec::new());
        LayoutInput {
            nodes: Arc::from(nodes),
            edges: Arc::from(edges),
            config: rust_sugiyama::Config::default(),
            render_config: rust_sugiyama::RenderConfig::default(),
            clusters,
            node_size: Arc::new(move |_| size),
            edge_label: Arc::new(|_, _| None),
        }
    }

    #[test]
    fn bounds_cover_full_circle_not_just_positive_quadrant() {
        // A 4-cycle is a single biconnected component laid out on one circle
        // centered at the origin, so raw coordinates span [-R, R] per axis.
        let nodes = vec![0, 1, 2, 3];
        let edges = vec![(0, 1), (1, 2), (2, 3), (3, 0)];
        let layout = circo_layout(&circo_input(nodes.clone(), edges));

        // radius = n * (min_dist + largest_node) / TAU with min_dist = 120 and
        // largest_node = max(width, height) = 100 for the (100, 40) test nodes
        // (see layout_block).
        let radius = 4.0 * (120.0 + 100.0) / TAU;

        // The reported size must be the full extent (2R plus one node size per
        // axis), not just the positive quadrant (R plus half a node). This is
        // what autofit uses to compute zoom/pan.
        assert!((layout.max_x() - (2.0 * radius + 100.0)).abs() < 0.01);
        assert!((layout.max_y() - (2.0 * radius + 40.0)).abs() < 0.01);

        // All node centers must sit inside the reported box, in the positive
        // quadrant.
        for position in 0..nodes.len() {
            let (x, y) = layout.position(position).expect("node has a position");
            assert!(x >= -1e-9 && x <= layout.max_x() + 1e-9);
            assert!(y >= -1e-9 && y <= layout.max_y() + 1e-9);
        }
    }

    #[test]
    fn radius_scales_with_node_size() {
        // The circle radius must come from the actual node sizes, not a fixed
        // constant: bigger nodes need a bigger circle to avoid overlapping.
        let nodes = vec![0, 1, 2, 3];
        let edges = vec![(0, 1), (1, 2), (2, 3), (3, 0)];

        let small = circo_layout(&circo_input_with_size(
            nodes.clone(),
            edges.clone(),
            (40.0, 40.0),
        ));
        let large = circo_layout(&circo_input_with_size(nodes, edges, (160.0, 40.0)));

        // radius = n * (min_dist + largest_node) / TAU with min_dist = 120.
        let small_radius = 4.0 * (120.0 + 40.0) / TAU;
        let large_radius = 4.0 * (120.0 + 160.0) / TAU;

        assert!((small.max_x() - (2.0 * small_radius + 40.0)).abs() < 0.01);
        assert!((large.max_x() - (2.0 * large_radius + 160.0)).abs() < 0.01);
        assert!(large.max_x() > small.max_x());
    }
}
