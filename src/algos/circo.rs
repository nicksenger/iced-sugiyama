use std::collections::{HashMap, VecDeque};
use std::f64::consts::{PI, TAU};

use crate::layout_engine::{GraphLayout, LayoutInput};

// ── Data structures ──────────────────────────────────────────────────────

/// A block (biconnected component) in the block-cutpoint tree.
#[derive(Clone, Debug)]
pub struct Block {
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
    /// Ordered list of nodes around the circle (global indices into the
    /// original node list).
    circle_list: Vec<usize>,
    /// If block has 1 node, the angle to place the parent.
    parent_pos: Option<f64>,
    /// Whether this block has been coalesced with its only child.
    coalesced: bool,
    /// Translation of this block's center relative to its parent's center,
    /// expressed in the parent's coordinate frame.
    offset: (f64, f64),
    /// Rotation of this block relative to its parent, in radians.
    rotation: f64,
}

/// State for the circular layout algorithm.
struct CircState {
    /// Minimum distance between nodes.
    min_dist: f64,
}

// ── DFS state for block decomposition ────────────────────────────────────

struct BlockFinder {
    val: Vec<u32>,
    low: Vec<u32>,
    parent: Vec<Option<usize>>,
    block_of: Vec<Option<usize>>,
    /// Blocks each node has already been assigned to. Articulation points
    /// belong to every block containing them, so membership is tracked per
    /// node instead of being claimed by the first pop.
    pops: Vec<Vec<usize>>,
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
            pops: vec![Vec::new(); n],
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
                offset: (0.0, 0.0),
                rotation: 0.0,
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
                offset: (0.0, 0.0),
                rotation: 0.0,
            });
            self.block_of[u] = Some(idx);
        }
    }

    fn pop_block(&mut self, u: usize, v: usize) {
        let mut block_nodes: Vec<usize> = Vec::new();
        let mut block_edges: Vec<(usize, usize)> = Vec::new();

        loop {
            let Some((a, b)) = self.edge_stack.pop() else {
                break;
            };
            block_edges.push((a, b));

            for &node in &[a, b] {
                if !self.pops[node].contains(&self.blocks.len()) {
                    block_nodes.push(node);
                    self.pops[node].push(self.blocks.len());
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
            offset: (0.0, 0.0),
            rotation: 0.0,
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
    let pops = finder.pops;

    let root_block = block_of[root].unwrap_or(0);

    let mut block_parent = vec![None; blocks.len()];
    let mut block_child_node = vec![None; blocks.len()];

    for bi in 0..blocks.len() {
        if bi == root_block {
            continue;
        }
        // The cutpoint connecting this block to the rest of the graph is its
        // lowest-DFS-order node.
        let mut min_val = u32::MAX;
        let mut best_node = 0;
        for &node in &blocks[bi].nodes {
            if val[node] < min_val {
                min_val = val[node];
                best_node = node;
            }
        }
        // Its parent is the latest-popped block sharing that cutpoint — the
        // one closest to the DFS root. (Looking up the cutpoint's DFS parent
        // instead fails when that node is itself shared by several blocks.)
        let Some(shared) = pops.get(best_node).filter(|p| p.len() > 1) else {
            continue;
        };
        if let Some(parent) = shared.iter().copied().filter(|&b| b != bi).max() {
            block_parent[bi] = Some(parent);
            block_child_node[bi] = Some(best_node);
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

    fn build_subtree(
        idx: usize,
        blocks: &[Block],
        children: &HashMap<usize, Vec<usize>>,
        child_node: &[Option<usize>],
    ) -> Block {
        let mut block = blocks[idx].clone();
        let child_indices = children.get(&idx).cloned().unwrap_or_default();
        for &child_idx in &child_indices {
            let mut child = build_subtree(child_idx, blocks, children, child_node);
            child.child = child_node[child_idx];
            block.children.push(child);
        }
        block
    }

    Some(build_subtree(
        root_block,
        &blocks,
        &children_map,
        &block_child_node,
    ))
}

// ── Spanning tree and longest path ───────────────────────────────────────

/// BFS spanning tree over `n` nodes, with edges already expressed in local
/// indices (positions within the block's node list).
fn spanning_tree(n: usize, edges: &[(usize, usize)]) -> Vec<Option<usize>> {
    if n == 0 {
        return Vec::new();
    }

    let mut local_adj: Vec<Vec<usize>> = vec![Vec::new(); n];

    for &(a, b) in edges {
        local_adj[a].push(b);
        local_adj[b].push(a);
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

    fn dfs_path(
        u: usize,
        target: usize,
        children: &[Vec<usize>],
        visited: &mut [bool],
        path: &mut Vec<usize>,
    ) -> bool {
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

/// Appends the nodes missing from `path` (local indices over `n` nodes) so
/// that every node appears exactly once. Edges are local.
fn place_residual_nodes(n: usize, edges: &[(usize, usize)], path: &mut Vec<usize>) {
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

    let pos: HashMap<usize, usize> = list
        .iter()
        .copied()
        .enumerate()
        .map(|(i, n)| (n, i))
        .collect();

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

    // The ordering helpers below work in local indices (positions within the
    // block's node list), so express the block's global edges locally first.
    let local_index: HashMap<usize, usize> = block.nodes.iter().copied().enumerate().collect();
    let local_edges: Vec<(usize, usize)> = block
        .edges
        .iter()
        .filter_map(|&(a, b)| Some((*local_index.get(&a)?, *local_index.get(&b)?)))
        .collect();

    let parent = spanning_tree(n, &local_edges);
    let mut path = longest_path_in_tree(&parent);

    place_residual_nodes(n, &local_edges, &mut path);

    reduce_crossings(&mut path, &local_edges);

    // Account for node sizes: radius = N * (min_dist + largest_node) / (2 * PI)
    let largest_node = 72.0; // default node size from the moar example (MIN_NODE_SIDE)
    let radius = if path.len() <= 1 {
        0.0
    } else {
        let circumference = path.len() as f64 * (state.min_dist + largest_node);
        circumference / TAU
    };

    // Map the local ordering back to global node indices so that
    // `collect_positions` can address the shared coordinate map without
    // colliding across blocks.
    block.circle_list = path.iter().map(|&local| block.nodes[local]).collect();
    block.radius = if n <= 1 {
        (state.min_dist + largest_node) / 2.0
    } else {
        radius
    };
    block.rad0 = block.radius;

    // Angle at which this block's articulation point (shared with its parent
    // block) sits on the circle, so the parent can rotate the block to line
    // the shared node up.
    let mut parent_pos = None;
    if let Some(art) = block.child {
        if let Some(pos) = path.iter().position(|&local| block.nodes[local] == art) {
            parent_pos = Some(pos as f64 * TAU / n as f64);
        }
    }
    block.parent_pos = parent_pos;
}

// ── Child block positioning ──────────────────────────────────────────────

/// Rotation that lines the child's articulation node up with the parent's:
/// the shared node must point from the child's center back towards the
/// tangent point, i.e. along `theta + PI`.
fn get_rotation(child: &Block, _x: f64, _y: f64, theta: f64) -> f64 {
    match child.parent_pos {
        Some(pp) => {
            let mut angle = theta + PI - pp;
            if angle < 0.0 {
                angle += TAU;
            }
            angle
        }
        None => 0.0,
    }
}

fn position_children(block: &mut Block, state: &CircState) {
    let child_count = block.children.len();
    if child_count == 0 {
        return;
    }

    let length = block.circle_list.len();
    let node_angle = if length > 0 {
        TAU / (length as f64)
    } else {
        TAU / child_count as f64
    };

    let mut parent_nodes: Vec<(usize, usize)> = Vec::new();
    for (ci, child) in block.children.iter().enumerate() {
        if let Some(cn) = child.child {
            if let Some(pos) = block.circle_list.iter().position(|&x| x == cn) {
                parent_nodes.push((pos, ci));
            }
        }
    }

    let mut max_radius: f64 = 0.0;
    let mut placed = vec![false; child_count];
    for &(path_idx, ci) in &parent_nodes {
        let child = &mut block.children[ci];
        let theta = path_idx as f64 * node_angle;
        // The child circle is tangent to this one at the shared articulation
        // point, which sits on our own circle (`rad0`) at angle `theta`.
        let r = block.rad0 + child.rad0;
        let dx = r * theta.cos();
        let dy = r * theta.sin();
        let rotate = get_rotation(child, dx, dy, theta);
        child.offset = (dx, dy);
        child.rotation = rotate;
        placed[ci] = true;
        max_radius = max_radius.max(child.radius);
    }

    // Children whose articulation point is not on this block's circle would
    // otherwise keep the default (0, 0) offset and stack on top of their
    // parent. Spread them on a half-step grid over the full circle so they
    // land between the anchored slots.
    let unanchored: Vec<usize> = (0..child_count).filter(|&ci| !placed[ci]).collect();
    for (k, &ci) in unanchored.iter().enumerate() {
        let child = &mut block.children[ci];
        let angle = TAU * (k as f64 + 0.5) / child_count as f64;
        // No shared articulation point, so keep a minimum gap between the
        // circles.
        let r = block.rad0 + child.rad0 + state.min_dist;
        child.offset = (r * angle.cos(), r * angle.sin());
        child.rotation = 0.0;
        max_radius = max_radius.max(child.radius);
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

/// Places every node of `block` (and its subtree) into `coords`, which is
/// keyed by global node index.
///
/// `(offset_x, offset_y)` is the parent block's center in global coordinates
/// and `parent_rotation` is the parent's accumulated rotation. A block's own
/// `offset`/`rotation` are expressed in the parent's frame, so they are
/// rotated by `parent_rotation` before being applied.
fn collect_positions(
    block: &Block,
    offset_x: f64,
    offset_y: f64,
    parent_rotation: f64,
    coords: &mut std::collections::BTreeMap<usize, (f64, f64)>,
) {
    let cos_p = parent_rotation.cos();
    let sin_p = parent_rotation.sin();

    // This block's center in global coordinates.
    let cx = offset_x + block.offset.0 * cos_p - block.offset.1 * sin_p;
    let cy = offset_y + block.offset.0 * sin_p + block.offset.1 * cos_p;
    let rotation = parent_rotation + block.rotation;

    let n = block.circle_list.len();
    if n > 0 {
        let cos_r = rotation.cos();
        let sin_r = rotation.sin();

        // Nodes sit on the block's own circle (`rad0`); `radius` also
        // accounts for child subtrees and must not push the nodes out of the
        // positions their children were spaced against.
        let radius = block.rad0;
        for (i, &node) in block.circle_list.iter().enumerate() {
            let theta = i as f64 * TAU / n as f64;
            let local_x = radius * theta.cos();
            let local_y = radius * theta.sin();

            let rx = local_x * cos_r - local_y * sin_r;
            let ry = local_x * sin_r + local_y * cos_r;

            coords.insert(node, (cx + rx, cy + ry));
        }
    }

    for child in &block.children {
        collect_positions(child, cx, cy, rotation, coords);
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
        return GraphLayout::from_parts(
            0.0,
            1.0,
            std::collections::BTreeMap::new(),
            Vec::new(),
            Vec::new(),
        );
    }

    let node_count = nodes.len();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); node_count];

    let node_index: HashMap<u32, usize> = nodes
        .iter()
        .copied()
        .enumerate()
        .map(|(i, n)| (n, i))
        .collect();

    for &(from, to) in edges {
        if let Some(&fi) = node_index.get(&from) {
            if let Some(&ti) = node_index.get(&to) {
                adj[fi].push(ti);
                adj[ti].push(fi);
            }
        }
    }

    let state = CircState { min_dist: 120.0 };

    let mut root = match build_block_tree(&adj) {
        Some(r) => r,
        None => {
            return GraphLayout::from_parts(
                0.0,
                1.0,
                std::collections::BTreeMap::new(),
                Vec::new(),
                Vec::new(),
            );
        }
    };

    do_block(&mut root, &state);

    let mut coords: std::collections::BTreeMap<usize, (f64, f64)> =
        std::collections::BTreeMap::new();
    collect_positions(&root, 0.0, 0.0, 0.0, &mut coords);

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
            let from_pos = node_index
                .get(&from)
                .and_then(|&index| coords.get(&index))
                .copied()
                .unwrap_or((0.0, 0.0));
            let to_pos = node_index
                .get(&to)
                .and_then(|&index| coords.get(&index))
                .copied()
                .unwrap_or((0.0, 0.0));
            let points = vec![from_pos, to_pos];
            let label = (input.edge_label)(i, (from, to));
            iced_sugiyama_core::EdgeLayout::new(i, points.clone(), points, label, None)
        })
        .collect();

    let cluster_layouts = Vec::new();

    GraphLayout::from_parts(
        layout_width,
        layout_height,
        coords,
        edge_layouts,
        cluster_layouts,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn circo_input(nodes: Vec<u32>, edges: Vec<(u32, u32)>) -> LayoutInput<'static> {
        let clusters: Arc<[crate::layout_engine::Cluster]> = Arc::from(Vec::new());
        LayoutInput {
            nodes: Arc::from(nodes),
            edges: Arc::from(edges),
            config: iced_sugiyama_core::Config::default(),
            render_config: iced_sugiyama_core::RenderConfig::default(),
            clusters,
            node_size: Arc::new(|_| (100.0, 40.0)),
            edge_label: Arc::new(|_, _| None),
        }
    }

    #[test]
    fn bounds_cover_full_circle_not_just_positive_quadrant() {
        // A 4-cycle is a single biconnected component laid out on one circle
        // centered at the origin, so raw coordinates span [-R, R] per axis.
        let nodes = vec![0, 1, 2, 3];
        let edges = vec![(0, 1), (1, 2), (2, 3), (3, 0)];
        let count = nodes.len();
        let layout = circo_layout(&circo_input(nodes, edges));

        // radius = n * (min_dist + largest_node) / TAU with min_dist = 120 and
        // largest_node = 72 (see layout_block).
        let radius = 4.0 * (120.0 + 72.0) / TAU;

        // The reported size must be the full extent (2R plus one node size per
        // axis), not just the positive quadrant (R plus half a node). This is
        // what autofit uses to compute zoom/pan.
        assert!((layout.max_x() - (2.0 * radius + 100.0)).abs() < 0.01);
        assert!((layout.max_y() - (2.0 * radius + 40.0)).abs() < 0.01);

        // All node centers must sit inside the reported box, in the positive
        // quadrant.
        for position in 0..count {
            let (x, y) = layout.position(position).expect("node has a position");
            assert!(x >= -1e-9 && x <= layout.max_x() + 1e-9);
            assert!(y >= -1e-9 && y <= layout.max_y() + 1e-9);
        }
    }
}

#[cfg(test)]
mod multi_block_tests {
    use super::*;
    use std::sync::Arc;

    fn circo_input(nodes: Vec<u32>, edges: Vec<(u32, u32)>) -> LayoutInput<'static> {
        let clusters: Arc<[crate::layout_engine::Cluster]> = Arc::from(Vec::new());
        LayoutInput {
            nodes: Arc::from(nodes),
            edges: Arc::from(edges),
            config: iced_sugiyama_core::Config::default(),
            render_config: iced_sugiyama_core::RenderConfig::default(),
            clusters,
            node_size: Arc::new(|_| (100.0, 40.0)),
            edge_label: Arc::new(|_, _| None),
        }
    }

    /// Collects every node's position, panicking if any node is missing one.
    fn all_positions(layout: &GraphLayout, count: usize) -> Vec<(f64, f64)> {
        (0..count)
            .map(|i| {
                layout
                    .position(i)
                    .unwrap_or_else(|| panic!("node index {i} has no position"))
            })
            .collect()
    }

    fn assert_distinct(positions: &[(f64, f64)]) {
        for i in 0..positions.len() {
            for j in (i + 1)..positions.len() {
                let dx = positions[i].0 - positions[j].0;
                let dy = positions[i].1 - positions[j].1;
                assert!(
                    dx * dx + dy * dy > 1.0,
                    "nodes {i} and {j} collide: {positions:?}"
                );
            }
        }
    }

    #[test]
    fn path_graph_gives_every_node_a_distinct_position() {
        // A path is three biconnected blocks chained by articulation points.
        // Regression test for local/global index confusion: child blocks used
        // to write their node positions over the root block's coordinates,
        // leaving some nodes without any position at all.
        let nodes = vec![0u32, 1, 2, 3];
        let edges = vec![(0u32, 1), (1, 2), (2, 3)];
        let count = nodes.len();
        let layout = circo_layout(&circo_input(nodes, edges));

        let positions = all_positions(&layout, count);
        assert_distinct(&positions);
    }

    #[test]
    fn longer_path_keeps_blocks_chained() {
        // Five chained blocks: each articulation point must sit at the same
        // global position in both of its blocks (tangent attachment), so no
        // node may end up closer than the minimum distance.
        let nodes = vec![0u32, 1, 2, 3, 4, 5];
        let edges = vec![(0u32, 1), (1, 2), (2, 3), (3, 4), (4, 5)];
        let count = nodes.len();
        let layout = circo_layout(&circo_input(nodes, edges));

        let positions = all_positions(&layout, count);
        assert_distinct(&positions);
    }

    #[test]
    fn cycle_is_a_single_block() {
        // A 4-cycle is biconnected: every node lands on one circle.
        let nodes = vec![0u32, 1, 2, 3];
        let edges = vec![(0u32, 1), (1, 2), (2, 3), (3, 0)];
        let count = nodes.len();
        let layout = circo_layout(&circo_input(nodes, edges));

        let positions = all_positions(&layout, count);
        assert_distinct(&positions);
    }

    #[test]
    fn butterfly_shares_its_central_articulation_point() {
        // Two triangles glued at node 0: two blocks sharing one cutpoint.
        let nodes = vec![0u32, 1, 2, 3, 4];
        let edges = vec![(0u32, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 0)];
        let count = nodes.len();
        let layout = circo_layout(&circo_input(nodes, edges));

        let positions = all_positions(&layout, count);
        assert_distinct(&positions);
    }

    #[test]
    fn cycle_with_tail_and_branch_lays_out_cleanly() {
        // The shape of a merged beam sun: a biconnected core with a chained
        // tail and a single-node branch off the core.
        let nodes = vec![0u32, 1, 2, 3, 4, 5, 6];
        let edges = vec![(0u32, 1), (1, 2), (2, 3), (3, 0), (3, 4), (4, 5), (1, 6)];
        let count = nodes.len();
        let layout = circo_layout(&circo_input(nodes, edges));

        let positions = all_positions(&layout, count);
        assert_distinct(&positions);
    }

    #[test]
    fn sibling_child_blocks_do_not_stack() {
        // Node 0 is an articulation point with two child blocks ({0,1} and
        // {0,2}). Regression test for `apply_delta` being a no-op: every
        // child used to be placed at the same angle, stacking whole
        // sub-blocks on top of each other.
        let nodes = vec![0u32, 1, 2];
        let edges = vec![(0u32, 1), (0, 2)];
        let count = nodes.len();
        let layout = circo_layout(&circo_input(nodes, edges));

        let positions = all_positions(&layout, count);
        assert_distinct(&positions);
    }
}
