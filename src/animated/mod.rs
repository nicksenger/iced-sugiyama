#![allow(deprecated)]

use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::rc::Rc;
use std::time::Duration;

use iced::advanced::widget;
use iced::advanced::widget::{Operation, Tree, Widget, tree};
use iced::time::Instant;
use iced::widget::canvas::{self, Path};
use iced::widget::{Component, Lazy, Stack};
use iced::window::RedrawRequest;
use iced::{Color, Element, Length, Padding, Point, Size, Task, Transformation, Vector, event};

pub use crate::layout_engine::{Cluster, EdgeEndpoint, EdgeEndpointKind};
use crate::layout_engine::{GraphLayout, compute_layout};
use crate::motion::easing::Easing;

pub mod motion;
mod switch;
pub use switch::Switch;

const FRAME_RATE_HZ: u64 = 60;
const FRAME_DURATION: Duration = Duration::from_millis(1000 / FRAME_RATE_HZ);
const EDGE_FINAL_SETTLE_START: f32 = 0.82;
const MIN_ZOOM: f32 = 0.002;
const MAX_ZOOM: f32 = 400.0;
const DRAG_CAPTURE_THRESHOLD: f32 = 3.0;
const INERTIA_MIN_SPEED: f32 = 20.0;
const INERTIA_DECAY_PER_SECOND: f32 = 0.08;

#[derive(Default, Clone)]
pub struct SharedAnimation(Rc<RefCell<Animation>>);
impl SharedAnimation {
    fn get(&self) -> Animation {
        *self.0.borrow()
    }

    fn set(&self, new: Animation) {
        *self.0.borrow_mut() = new;
    }
}
impl Hash for SharedAnimation {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::ptr::hash(Rc::as_ptr(&self.0), state);
    }
}

#[derive(Debug, Clone, Copy)]
struct DragState {
    last_cursor: Point,
    last_update: Instant,
    distance: f32,
}

#[derive(Debug, Clone, Copy)]
struct ViewportState {
    pan: Vector,
    zoom: f32,
    velocity: Vector,
    drag: Option<DragState>,
    last_tick: Option<Instant>,
}

impl Default for ViewportState {
    fn default() -> Self {
        Self {
            pan: Vector::new(0.0, 0.0),
            zoom: 1.0,
            velocity: Vector::new(0.0, 0.0),
            drag: None,
            last_tick: None,
        }
    }
}

#[derive(Default, Clone)]
pub struct SharedViewport(Rc<RefCell<ViewportState>>);
impl SharedViewport {
    fn get(&self) -> ViewportState {
        *self.0.borrow()
    }

    fn begin_drag(&self, cursor: Point, now: Instant) {
        let mut state = self.0.borrow_mut();
        state.drag = Some(DragState {
            last_cursor: cursor,
            last_update: now,
            distance: 0.0,
        });
        state.velocity = Vector::new(0.0, 0.0);
        state.last_tick = Some(now);
    }

    fn drag_to(&self, cursor: Point, now: Instant) -> bool {
        let mut state = self.0.borrow_mut();
        let Some(mut drag) = state.drag else {
            return false;
        };

        let delta = cursor - drag.last_cursor;
        state.pan = Vector::new(state.pan.x + delta.x, state.pan.y + delta.y);

        let dt = now
            .saturating_duration_since(drag.last_update)
            .as_secs_f32()
            .max(1e-4);
        state.velocity = Vector::new(delta.x / dt, delta.y / dt);

        drag.last_cursor = cursor;
        drag.last_update = now;
        drag.distance += delta.x.hypot(delta.y);
        state.drag = Some(drag);
        state.last_tick = Some(now);

        drag.distance >= DRAG_CAPTURE_THRESHOLD
    }

    fn end_drag(&self, now: Instant) -> bool {
        let mut state = self.0.borrow_mut();
        let captured = state
            .drag
            .map(|drag| drag.distance >= DRAG_CAPTURE_THRESHOLD)
            .unwrap_or(false);
        state.drag = None;
        state.last_tick = Some(now);
        if !captured {
            state.velocity = Vector::new(0.0, 0.0);
        }
        captured
    }

    fn zoom_at(&self, delta_y: f32, cursor: Point, size: Size) -> bool {
        let mut state = self.0.borrow_mut();
        let old_zoom = state.zoom;
        let factor = (1.0 + delta_y / 30.0).max(0.01);
        let new_zoom = (old_zoom * factor).clamp(MIN_ZOOM, MAX_ZOOM);

        if (new_zoom - old_zoom).abs() <= f32::EPSILON {
            return false;
        }

        let center = Point::new(size.width * 0.5, size.height * 0.5);
        let world_x = (cursor.x - center.x - state.pan.x) / old_zoom;
        let world_y = (cursor.y - center.y - state.pan.y) / old_zoom;

        state.pan = Vector::new(
            cursor.x - center.x - world_x * new_zoom,
            cursor.y - center.y - world_y * new_zoom,
        );
        state.zoom = new_zoom;
        state.velocity = Vector::new(0.0, 0.0);

        true
    }

    fn tick(&self, now: Instant) -> bool {
        let mut state = self.0.borrow_mut();
        if state.drag.is_some() {
            state.last_tick = Some(now);
            return false;
        }

        let speed = state.velocity.x.hypot(state.velocity.y);
        if speed <= INERTIA_MIN_SPEED {
            if speed > 0.0 {
                state.velocity = Vector::new(0.0, 0.0);
            }
            state.last_tick = Some(now);
            return false;
        }

        let last = state.last_tick.unwrap_or(now);
        let dt = now.saturating_duration_since(last).as_secs_f32().max(1e-4);
        state.pan = Vector::new(
            state.pan.x + state.velocity.x * dt,
            state.pan.y + state.velocity.y * dt,
        );
        let decay = INERTIA_DECAY_PER_SECOND.powf(dt);
        state.velocity = Vector::new(state.velocity.x * decay, state.velocity.y * decay);
        state.last_tick = Some(now);

        true
    }

    fn is_moving(&self) -> bool {
        let state = self.0.borrow();
        state.drag.is_some() || state.velocity.x.hypot(state.velocity.y) > INERTIA_MIN_SPEED
    }

    fn is_dragging(&self) -> bool {
        self.0.borrow().drag.is_some()
    }
}

impl Hash for SharedViewport {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::ptr::hash(Rc::as_ptr(&self.0), state);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub enum Animation {
    #[default]
    Pending,
    Active {
        start: Instant,
        elapsed: Duration,
    },
    Complete,
}

impl Animation {
    fn with_elapsed(mut self, new_elapsed: Duration) -> Self {
        if let Self::Active { elapsed, .. } = &mut self {
            *elapsed = new_elapsed;
        }

        self
    }

    fn timed_transition(
        &self,
        transition_duration: Duration,
        now: Instant,
    ) -> (Self, Option<RedrawRequest>) {
        let Self::Active { start, .. } = self else {
            return (*self, None);
        };

        let elapsed = now.duration_since(*start);
        if elapsed >= transition_duration {
            let new_state = match self {
                Animation::Pending | Animation::Complete => {
                    return (*self, None);
                }
                Animation::Active { .. } => Animation::Complete,
            };

            (new_state, Some(RedrawRequest::At(now + FRAME_DURATION)))
        } else {
            (
                self.with_elapsed(elapsed),
                Some(RedrawRequest::At(now + FRAME_DURATION)),
            )
        }
    }

    fn progress(&self, easing: &Easing, duration: Duration) -> Option<f32> {
        let Self::Active { elapsed, .. } = self else {
            return None;
        };

        Some(
            easing
                .y_at_x(elapsed.as_secs_f32() / duration.as_secs_f32())
                .min(1.0),
        )
    }
}

#[derive(Clone)]
pub struct Graph {
    pub nodes: Vec<u32>,
    pub edges: Vec<(u32, u32)>,
    pub config: rust_sugiyama::Config,
}

impl<'a> From<&'a Graph> for Cow<'a, Graph> {
    fn from(graph: &'a Graph) -> Self {
        Cow::Borrowed(graph)
    }
}

impl Hash for Graph {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.nodes.hash(state);
        self.edges.hash(state);
    }
}

impl Graph {
    pub fn new(nodes: Vec<u32>, edges: Vec<(u32, u32)>) -> Self {
        Self {
            nodes,
            edges,
            config: rust_sugiyama::Config::default(),
        }
    }

    pub fn config(self, config: rust_sugiyama::Config) -> Self {
        Self { config, ..self }
    }
}

#[derive(Clone)]
pub enum Event<Message> {
    Passthrough(Message),
    Noop,
}

#[derive(Debug, Clone, Copy)]
pub struct OutgoingEdgeStyle {
    pub visible: bool,
    pub width_scale: f32,
    pub alpha: f32,
    pub color_override: Option<(Color, Color)>,
}

impl Default for OutgoingEdgeStyle {
    fn default() -> Self {
        Self {
            visible: true,
            width_scale: 1.0,
            alpha: 1.0,
            color_override: None,
        }
    }
}

impl OutgoingEdgeStyle {
    pub fn hidden() -> Self {
        Self {
            visible: false,
            width_scale: 0.0,
            alpha: 0.0,
            color_override: None,
        }
    }
}

#[derive(Default)]
#[doc(hidden)]
pub struct LayoutMemo {
    entries: Vec<(u64, GraphLayout)>,
}

impl LayoutMemo {
    fn layout_for(&mut self, signature: u64, compute: impl FnOnce() -> GraphLayout) -> GraphLayout {
        if let Some((_, layout)) = self
            .entries
            .iter()
            .find(|(cached_signature, _)| *cached_signature == signature)
        {
            return layout.clone();
        }

        let layout = compute();
        self.entries.push((signature, layout.clone()));
        if self.entries.len() > 4 {
            self.entries.remove(0);
        }
        layout
    }

    fn get_layout(&self, signature: u64) -> Option<GraphLayout> {
        self.entries
            .iter()
            .find(|(cached_signature, _)| *cached_signature == signature)
            .map(|(_, layout)| layout.clone())
    }
}

/// An iced widget which draws a layered graph of elements
pub struct Sugiyama<'a, Message, Theme, Renderer> {
    graph: Cow<'a, Graph>,
    id: Option<widget::Id>,
    view_node: Box<dyn Fn(u32) -> Element<'static, Message, Theme, Renderer> + 'a>,
    cluster_container:
        Box<dyn Fn(usize, &Cluster) -> Option<Element<'static, Message, Theme, Renderer>> + 'a>,
    stroke_width: f32,
    edge_corner_radius: f32,
    edge_endpoint_extension: f32,
    edge_color: fn(usize) -> (Color, Color),
    outgoing_edge_style: Box<dyn Fn(u32) -> OutgoingEdgeStyle + 'a>,
    edge_label: Box<dyn Fn(usize, (u32, u32)) -> Option<String> + 'a>,
    edge_label_element: Box<
        dyn Fn(
                usize,
                (u32, u32),
                Option<&str>,
            ) -> Option<Element<'static, Message, Theme, Renderer>>
            + 'a,
    >,
    edge_endpoint: Box<
        dyn Fn(
                usize,
                (u32, u32),
                EdgeEndpointKind,
                EdgeEndpoint,
            ) -> Option<Element<'static, Message, Theme, Renderer>>
            + 'a,
    >,
    node_size: Box<dyn Fn(u32) -> (f64, f64) + 'a>,
    clusters: Vec<Cluster>,
    render_config: rust_sugiyama::RenderConfig,
    cluster_color: fn(usize) -> Color,
    label_color: fn(usize) -> Color,
    padding: Padding,
    motion_easing: &'static motion::easing::Easing,
    motion_duration: Duration,
}

impl<'a, Message, Theme, Renderer> Sugiyama<'a, Message, Theme, Renderer> {
    pub fn new(
        graph: impl Into<Cow<'a, Graph>>,
        view_node: impl Fn(u32) -> Element<'static, Message, Theme, Renderer> + 'a,
    ) -> Self {
        Self {
            graph: graph.into(),
            id: None,
            view_node: Box::new(view_node),
            cluster_container: Box::new(|_, _| None),
            stroke_width: 2.0,
            edge_corner_radius: 10.0,
            edge_endpoint_extension: 0.0,
            edge_color: |_| (Color::BLACK, Color::BLACK.scale_alpha(0.5)),
            outgoing_edge_style: Box::new(|_| OutgoingEdgeStyle::default()),
            edge_label: Box::new(|_, _| None),
            edge_label_element: Box::new(|_, _, _| None),
            edge_endpoint: Box::new(|_, _, _, _| None),
            node_size: Box::new(|_| (56.0, 32.0)),
            clusters: Vec::new(),
            render_config: Default::default(),
            cluster_color: |_| Color::from_rgba8(90, 90, 90, 0.6),
            label_color: |_| Color::BLACK,
            padding: iced::Padding::ZERO,
            motion_easing: &motion::easing::EMPHASIZED,
            motion_duration: motion::duration::MEDIUM_4,
        }
    }

    pub fn id(mut self, id: impl Into<Id>) -> Self {
        self.id = Some(id.into().0);
        self
    }

    pub fn edge_color(mut self, f: fn(usize) -> (Color, Color)) -> Self {
        self.edge_color = f;
        self
    }

    pub fn outgoing_edge_style(mut self, f: impl Fn(u32) -> OutgoingEdgeStyle + 'a) -> Self {
        self.outgoing_edge_style = Box::new(f);
        self
    }

    pub fn stroke_width(mut self, width: f32) -> Self {
        self.stroke_width = width;
        self
    }

    pub fn edge_corner_radius(mut self, radius: f32) -> Self {
        self.edge_corner_radius = radius.max(0.0);
        self
    }

    pub fn edge_endpoint_extension(mut self, extension: f32) -> Self {
        self.edge_endpoint_extension = extension.max(0.0);
        self
    }

    pub fn edge_label(mut self, f: impl Fn(usize, (u32, u32)) -> Option<String> + 'a) -> Self {
        self.edge_label = Box::new(f);
        self
    }

    pub fn edge_label_element(
        mut self,
        f: impl Fn(
            usize,
            (u32, u32),
            Option<&str>,
        ) -> Option<Element<'static, Message, Theme, Renderer>>
        + 'a,
    ) -> Self {
        self.edge_label_element = Box::new(f);
        self
    }

    pub fn edge_endpoint(
        mut self,
        f: impl Fn(
            usize,
            (u32, u32),
            EdgeEndpointKind,
            EdgeEndpoint,
        ) -> Option<Element<'static, Message, Theme, Renderer>>
        + 'a,
    ) -> Self {
        self.edge_endpoint = Box::new(f);
        self
    }

    pub fn node_size(mut self, f: impl Fn(u32) -> (f64, f64) + 'a) -> Self {
        self.node_size = Box::new(f);
        self
    }

    pub fn clusters(mut self, clusters: Vec<Cluster>) -> Self {
        self.clusters = clusters;
        self
    }

    pub fn cluster_container(
        mut self,
        f: impl Fn(usize, &Cluster) -> Option<Element<'static, Message, Theme, Renderer>> + 'a,
    ) -> Self {
        self.cluster_container = Box::new(f);
        self
    }

    pub fn cluster_label(
        self,
        f: impl Fn(usize, &Cluster) -> Option<Element<'static, Message, Theme, Renderer>> + 'a,
    ) -> Self {
        self.cluster_container(f)
    }

    pub fn render_config(mut self, config: rust_sugiyama::RenderConfig) -> Self {
        self.render_config = config;
        self
    }

    pub fn cluster_color(mut self, f: fn(usize) -> Color) -> Self {
        self.cluster_color = f;
        self
    }

    pub fn label_color(mut self, f: fn(usize) -> Color) -> Self {
        self.label_color = f;
        self
    }

    pub fn padding(mut self, p: impl Into<Padding>) -> Self {
        self.padding = p.into();
        self
    }

    pub fn animation_easing(mut self, easing: &'static Easing) -> Self {
        self.motion_easing = easing;
        self
    }

    pub fn animation_duration(mut self, duration: Duration) -> Self {
        self.motion_duration = duration;
        self
    }
}

impl<Message, Theme, Renderer> Component<Message, Theme, Renderer>
    for Sugiyama<'_, Message, Theme, Renderer>
where
    Renderer: iced::advanced::Renderer + iced::advanced::graphics::geometry::Renderer + 'static,
    Theme: 'static,
    Message: Clone + 'static,
{
    type Event = Event<Message>;
    type State = SugiyamaState;

    fn view(&self, state: &Self::State) -> Element<'_, Self::Event, Theme, Renderer> {
        let switch_state = state.switch_state;
        let layout_memo = state.layout_memo.clone();
        let old_sugiyama = state
            .previous_signature
            .and_then(|signature| layout_memo.borrow().get_layout(signature));
        Lazy::new(
            (
                &self.graph,
                state.previous_graph.clone(),
                switch_state,
                state.animation.clone(),
                state.viewport.clone(),
                state.refresh_nonce,
            ),
            move |(graph, old, switch_state, animation, viewport, _refresh_nonce)| {
                let mut children = self
                    .graph
                    .nodes
                    .iter()
                    .map(|n| (self.view_node)(*n).map(Event::Passthrough))
                    .collect::<Vec<_>>();

                let node_map = self
                    .graph
                    .nodes
                    .iter()
                    .enumerate()
                    .map(|(i, n)| (*n, i))
                    .collect::<HashMap<_, _>>();
                let old_node_map = old.as_ref().map(|g| {
                    g.nodes
                        .iter()
                        .enumerate()
                        .map(|(i, n)| (*n, i))
                        .collect::<HashMap<_, _>>()
                });

                let old_sugiyama = old_sugiyama.clone();
                let signature = crate::layout_engine::layout_signature(
                    &graph.nodes,
                    &graph.edges,
                    &self.clusters,
                );
                let sugiyama = {
                    let mut memo = layout_memo.borrow_mut();
                    memo.layout_for(signature, || {
                        compute_layout(
                            &graph.nodes,
                            &graph.edges,
                            &graph.config,
                            &self.node_size,
                            &self.edge_label,
                            &self.clusters,
                            &self.render_config,
                        )
                    })
                };
                let edge_style_by_index = graph
                    .edges
                    .iter()
                    .enumerate()
                    .map(|(index, edge)| {
                        (
                            index,
                            normalize_outgoing_edge_style((self.outgoing_edge_style)(edge.0)),
                        )
                    })
                    .collect::<HashMap<_, _>>();
                let old_edge_style_by_index = old.as_ref().map(|old_graph| {
                    old_graph
                        .edges
                        .iter()
                        .enumerate()
                        .map(|(index, edge)| {
                            (
                                index,
                                normalize_outgoing_edge_style((self.outgoing_edge_style)(edge.0)),
                            )
                        })
                        .collect::<HashMap<_, _>>()
                });
                let mut cluster_container_children = Vec::new();
                for cluster in sugiyama.clusters() {
                    let Some(cluster_spec) = self.clusters.get(cluster.index()) else {
                        continue;
                    };
                    let Some(container) = (self.cluster_container)(cluster.index(), cluster_spec)
                    else {
                        continue;
                    };
                    cluster_container_children.push(ClusterContainerChild {
                        cluster_index: cluster.index(),
                    });
                    children.push(container.map(Event::Passthrough));
                }
                let mut edge_label_children = Vec::new();
                let mut edge_label_overlay_edges = HashSet::new();
                for edge in sugiyama.edges() {
                    let Some(edge_data) = graph.edges.get(edge.index()).copied() else {
                        continue;
                    };
                    let style = edge_style_by_index
                        .get(&edge.index())
                        .copied()
                        .unwrap_or_default();
                    if !style.visible || style.alpha <= f32::EPSILON {
                        continue;
                    }
                    let Some(label_element) =
                        (self.edge_label_element)(edge.index(), edge_data, edge.label())
                    else {
                        continue;
                    };
                    edge_label_overlay_edges.insert(edge.index());
                    edge_label_children.push(EdgeLabelChild {
                        edge_index: edge.index(),
                    });
                    children.push(label_element.map(Event::Passthrough));
                }
                let mut edge_endpoint_children = Vec::new();
                for edge in sugiyama.edges() {
                    let Some(edge_data) = graph.edges.get(edge.index()).copied() else {
                        continue;
                    };
                    let style = edge_style_by_index
                        .get(&edge.index())
                        .copied()
                        .unwrap_or_default();
                    if !style.visible || style.alpha <= f32::EPSILON {
                        continue;
                    }

                    for kind in [EdgeEndpointKind::Source, EdgeEndpointKind::Destination] {
                        let node_id = match kind {
                            EdgeEndpointKind::Source => edge_data.0,
                            EdgeEndpointKind::Destination => edge_data.1,
                        };
                        let Some(node_index) = node_map.get(&node_id).copied() else {
                            continue;
                        };
                        let Some((cx, cy)) = sugiyama.position(node_index) else {
                            continue;
                        };
                        let (width, height) = (self.node_size)(node_id);
                        let node_size = Size::new(width.max(1.0) as f32, height.max(1.0) as f32);
                        let node_center = Vector::new(cx as f32, cy as f32);
                        let Some(endpoint) = edge_endpoint_metadata(
                            edge,
                            kind,
                            node_center,
                            node_size,
                            self.edge_endpoint_extension,
                        ) else {
                            continue;
                        };
                        let Some(widget) =
                            (self.edge_endpoint)(edge.index(), edge_data, kind, endpoint)
                        else {
                            continue;
                        };
                        edge_endpoint_children.push(EdgeEndpointChild {
                            edge_index: edge.index(),
                            edge: edge_data,
                            kind,
                        });
                        children.push(widget.map(Event::Passthrough));
                    }
                }
                let overlay = GraphNodes::<Event<Message>, Theme, Renderer> {
                    children,
                    nodes: self.graph.nodes.clone(),
                    sugiyama: sugiyama.clone(),
                    old_sugiyama: old_sugiyama.clone(),
                    node_map: node_map.clone(),
                    old_node_map,
                    edges: self.graph.edges.clone(),
                    cluster_container_children,
                    edge_label_children,
                    edge_endpoint_children,
                    padding: self.padding,
                    edge_endpoint_extension: self.edge_endpoint_extension,
                    motion_easing: self.motion_easing,
                    motion_duration: self.motion_duration,
                    animation: animation.clone(),
                    viewport: viewport.clone(),
                };

                Switch::new(
                    *switch_state,
                    Stack::with_children(vec![
                        iced::widget::canvas(GraphCanvas::<Renderer> {
                            cache: Default::default(),
                            sugiyama: sugiyama.clone(),
                            old_sugiyama,
                            padding: self.padding,
                            stroke_width: self.stroke_width,
                            edge_corner_radius: self.edge_corner_radius,
                            edge_endpoint_extension: self.edge_endpoint_extension,
                            edge_color: self.edge_color,
                            edge_style_by_index: edge_style_by_index.clone(),
                            old_edge_style_by_index,
                            cluster_color: self.cluster_color,
                            label_color: self.label_color,
                            edge_label_overlay_edges,
                            animation: animation.clone(),
                            viewport: viewport.clone(),
                            motion_easing: self.motion_easing,
                            motion_duration: self.motion_duration,
                        })
                        .width(Length::Fill)
                        .height(Length::Fill)
                        .into(),
                        overlay.into(),
                    ]),
                )
            },
        )
        .into()
    }

    fn update(&mut self, state: &mut Self::State, event: Self::Event) -> Option<Message> {
        state.switch_state.flip();
        let signature = crate::layout_engine::layout_signature(
            &self.graph.nodes,
            &self.graph.edges,
            &self.clusters,
        );
        {
            let mut memo = state.layout_memo.borrow_mut();
            let _ = memo.layout_for(signature, || {
                compute_layout(
                    &self.graph.nodes,
                    &self.graph.edges,
                    &self.graph.config,
                    &self.node_size,
                    &self.edge_label,
                    &self.clusters,
                    &self.render_config,
                )
            });
        }
        state.previous_graph = Some(self.graph.clone().into_owned());
        state.previous_signature = Some(signature);
        state.animation = SharedAnimation::default();

        match event {
            Event::Noop => None,
            Event::Passthrough(message) => Some(message),
        }
    }

    fn operate(
        &self,
        bounds: iced::Rectangle,
        state: &mut Self::State,
        operation: &mut dyn widget::Operation,
    ) {
        let mut invalidate_request = InvalidateRequest::default();
        operation.custom(self.id.as_ref(), bounds, &mut invalidate_request);

        if invalidate_request.requested {
            state.switch_state.flip();
            let signature = crate::layout_engine::layout_signature(
                &self.graph.nodes,
                &self.graph.edges,
                &self.clusters,
            );
            {
                let mut memo = state.layout_memo.borrow_mut();
                let _ = memo.layout_for(signature, || {
                    compute_layout(
                        &self.graph.nodes,
                        &self.graph.edges,
                        &self.graph.config,
                        &self.node_size,
                        &self.edge_label,
                        &self.clusters,
                        &self.render_config,
                    )
                });
            }
            state.previous_graph = Some(self.graph.clone().into_owned());
            state.previous_signature = Some(signature);
            state.animation = SharedAnimation::default();
            state.refresh_nonce = state.refresh_nonce.saturating_add(1);
        }

        operation.custom(self.id.as_ref(), bounds, &mut state.refresh_nonce);
    }
}

#[derive(Default)]
#[doc(hidden)]
pub struct SugiyamaState {
    switch_state: switch::State,
    previous_graph: Option<Graph>,
    previous_signature: Option<u64>,
    animation: SharedAnimation,
    viewport: SharedViewport,
    layout_memo: Rc<RefCell<LayoutMemo>>,
    refresh_nonce: u64,
}

#[derive(Default)]
struct InvalidateRequest {
    requested: bool,
}

struct GraphCanvas<Renderer>
where
    Renderer: iced::advanced::graphics::geometry::Renderer,
{
    cache: canvas::Cache<Renderer>,
    sugiyama: GraphLayout,
    old_sugiyama: Option<GraphLayout>,
    padding: iced::Padding,
    stroke_width: f32,
    edge_corner_radius: f32,
    edge_endpoint_extension: f32,
    edge_color: fn(usize) -> (Color, Color),
    edge_style_by_index: HashMap<usize, OutgoingEdgeStyle>,
    old_edge_style_by_index: Option<HashMap<usize, OutgoingEdgeStyle>>,
    cluster_color: fn(usize) -> Color,
    label_color: fn(usize) -> Color,
    edge_label_overlay_edges: HashSet<usize>,
    animation: SharedAnimation,
    viewport: SharedViewport,
    motion_easing: &'static Easing,
    motion_duration: Duration,
}

impl<Message, Theme, Renderer> canvas::Program<Message, Theme, Renderer> for GraphCanvas<Renderer>
where
    Renderer: iced::advanced::Renderer + iced::advanced::graphics::geometry::Renderer,
{
    type State = ();

    fn draw(
        &self,
        _state: &Self::State,
        renderer: &Renderer,
        _theme: &Theme,
        bounds: iced::Rectangle,
        _cursor: iced::mouse::Cursor,
    ) -> Vec<canvas::Geometry<Renderer>> {
        self.cache.clear();

        let mut size = bounds.size();
        size.width = size.width - self.padding.left - self.padding.right;
        size.height = size.height - self.padding.top - self.padding.bottom;

        let graph = self.cache.draw(renderer, bounds.size(), |frame| {
            let viewport = self.viewport.get();
            let visual_scale = viewport.zoom.max(0.01);
            let scaled_stroke_width = (self.stroke_width * visual_scale).max(0.25);
            let scaled_corner_radius = self.edge_corner_radius * visual_scale;
            let scaled_endpoint_extension = self.edge_endpoint_extension * visual_scale;
            let scaled_edge_label_size = 14.0 * visual_scale;
            let project = |layout: &GraphLayout, x: f64, y: f64| {
                let offset = layout_offset(layout, size);
                let transformed = apply_view_transform(
                    Vector::new(x as f32 + offset.x, y as f32 + offset.y),
                    size,
                    viewport,
                );
                Point::new(
                    transformed.x + self.padding.left,
                    transformed.y + self.padding.top,
                )
            };
            let style_for_current = |index: usize| {
                self.edge_style_by_index
                    .get(&index)
                    .copied()
                    .unwrap_or_default()
            };
            let style_for_old = |index: usize| {
                self.old_edge_style_by_index
                    .as_ref()
                    .and_then(|styles| styles.get(&index).copied())
                    .unwrap_or_else(|| style_for_current(index))
            };
            if let Some(old_layout) = self.old_sugiyama.as_ref() {
                let animation = self.animation.get();
                let progress = animation
                    .progress(self.motion_easing, self.motion_duration)
                    .unwrap_or(if matches!(animation, Animation::Pending) {
                        0.0
                    } else {
                        1.0
                    });

                match animation {
                    Animation::Pending => {
                        for cluster in old_layout.clusters() {
                            draw_cluster_outline(
                                frame,
                                scaled_stroke_width,
                                (self.cluster_color)(cluster.index()),
                                project(old_layout, cluster.min_x(), cluster.min_y()),
                                project(old_layout, cluster.max_x(), cluster.max_y()),
                            );
                        }
                        for edge in old_layout.edges() {
                            let projected = edge
                                .points()
                                .iter()
                                .map(|(x, y)| project(old_layout, *x, *y))
                                .collect::<Vec<_>>();
                            let projected_curve = edge
                                .curve_points()
                                .iter()
                                .map(|(x, y)| project(old_layout, *x, *y))
                                .collect::<Vec<_>>();
                            let label_position = edge_canvas_label_position(
                                &self.edge_label_overlay_edges,
                                edge.index(),
                                edge.label_position()
                                    .map(|(x, y)| project(old_layout, x, y)),
                            );
                            draw_styled_edge(
                                frame,
                                scaled_stroke_width,
                                scaled_corner_radius,
                                scaled_endpoint_extension,
                                scaled_edge_label_size,
                                self.edge_color,
                                self.label_color,
                                edge,
                                &projected,
                                &projected_curve,
                                style_for_old(edge.index()),
                                1.0,
                                label_position,
                            );
                        }
                    }
                    Animation::Complete => {
                        for cluster in self.sugiyama.clusters() {
                            draw_cluster_outline(
                                frame,
                                scaled_stroke_width,
                                (self.cluster_color)(cluster.index()),
                                project(&self.sugiyama, cluster.min_x(), cluster.min_y()),
                                project(&self.sugiyama, cluster.max_x(), cluster.max_y()),
                            );
                        }
                        for edge in self.sugiyama.edges() {
                            let projected = edge
                                .points()
                                .iter()
                                .map(|(x, y)| project(&self.sugiyama, *x, *y))
                                .collect::<Vec<_>>();
                            let projected_curve = edge
                                .curve_points()
                                .iter()
                                .map(|(x, y)| project(&self.sugiyama, *x, *y))
                                .collect::<Vec<_>>();
                            let label_position = edge_canvas_label_position(
                                &self.edge_label_overlay_edges,
                                edge.index(),
                                edge.label_position()
                                    .map(|(x, y)| project(&self.sugiyama, x, y)),
                            );
                            draw_styled_edge(
                                frame,
                                scaled_stroke_width,
                                scaled_corner_radius,
                                scaled_endpoint_extension,
                                scaled_edge_label_size,
                                self.edge_color,
                                self.label_color,
                                edge,
                                &projected,
                                &projected_curve,
                                style_for_current(edge.index()),
                                1.0,
                                label_position,
                            );
                        }
                    }
                    Animation::Active { .. } => {
                        let mut old_clusters = old_layout
                            .clusters()
                            .iter()
                            .map(|cluster| (cluster.index(), cluster))
                            .collect::<HashMap<_, _>>();

                        for cluster in self.sugiyama.clusters() {
                            if let Some(old_cluster) = old_clusters.remove(&cluster.index()) {
                                let min_x = lerp(old_cluster.min_x(), cluster.min_x(), progress);
                                let min_y = lerp(old_cluster.min_y(), cluster.min_y(), progress);
                                let max_x = lerp(old_cluster.max_x(), cluster.max_x(), progress);
                                let max_y = lerp(old_cluster.max_y(), cluster.max_y(), progress);
                                draw_cluster_outline(
                                    frame,
                                    scaled_stroke_width,
                                    (self.cluster_color)(cluster.index()),
                                    project(&self.sugiyama, min_x, min_y),
                                    project(&self.sugiyama, max_x, max_y),
                                );
                            } else {
                                draw_cluster_outline(
                                    frame,
                                    scaled_stroke_width,
                                    (self.cluster_color)(cluster.index()).scale_alpha(progress),
                                    project(&self.sugiyama, cluster.min_x(), cluster.min_y()),
                                    project(&self.sugiyama, cluster.max_x(), cluster.max_y()),
                                );
                            }
                        }
                        for cluster in old_clusters.values() {
                            draw_cluster_outline(
                                frame,
                                scaled_stroke_width,
                                (self.cluster_color)(cluster.index()).scale_alpha(1.0 - progress),
                                project(old_layout, cluster.min_x(), cluster.min_y()),
                                project(old_layout, cluster.max_x(), cluster.max_y()),
                            );
                        }

                        let mut old_edges = old_layout
                            .edges()
                            .iter()
                            .map(|edge| (edge.index(), edge))
                            .collect::<HashMap<_, _>>();

                        for edge in self.sugiyama.edges() {
                            if let Some(old_edge) = old_edges.remove(&edge.index()) {
                                let projected_old = old_edge
                                    .points()
                                    .iter()
                                    .map(|(x, y)| project(old_layout, *x, *y))
                                    .collect::<Vec<_>>();
                                let projected_new = edge
                                    .points()
                                    .iter()
                                    .map(|(x, y)| project(&self.sugiyama, *x, *y))
                                    .collect::<Vec<_>>();
                                let projected_new_curve = edge
                                    .curve_points()
                                    .iter()
                                    .map(|(x, y)| project(&self.sugiyama, *x, *y))
                                    .collect::<Vec<_>>();
                                let points = interpolate_orthogonal_polylines(
                                    &projected_old,
                                    &projected_new,
                                    progress,
                                );
                                let interpolated_label_position = edge_canvas_label_position(
                                    &self.edge_label_overlay_edges,
                                    edge.index(),
                                    match (old_edge.label_position(), edge.label_position()) {
                                        (Some((ox, oy)), Some((nx, ny))) => Some(Point::new(
                                            lerp(
                                                project(old_layout, ox, oy).x as f64,
                                                project(&self.sugiyama, nx, ny).x as f64,
                                                progress,
                                            ) as f32,
                                            lerp(
                                                project(old_layout, ox, oy).y as f64,
                                                project(&self.sugiyama, nx, ny).y as f64,
                                                progress,
                                            ) as f32,
                                        )),
                                        (None, Some((nx, ny))) => {
                                            Some(project(&self.sugiyama, nx, ny))
                                        }
                                        (Some((ox, oy)), None) => Some(project(old_layout, ox, oy)),
                                        (None, None) => None,
                                    },
                                );
                                let final_label_position = edge_canvas_label_position(
                                    &self.edge_label_overlay_edges,
                                    edge.index(),
                                    edge.label_position()
                                        .map(|(x, y)| project(&self.sugiyama, x, y)),
                                );
                                let settle = ((progress - EDGE_FINAL_SETTLE_START)
                                    / (1.0 - EDGE_FINAL_SETTLE_START))
                                    .clamp(0.0, 1.0);
                                let style = style_for_current(edge.index());
                                draw_styled_edge(
                                    frame,
                                    scaled_stroke_width,
                                    scaled_corner_radius,
                                    scaled_endpoint_extension,
                                    scaled_edge_label_size,
                                    self.edge_color,
                                    self.label_color,
                                    edge,
                                    &points,
                                    &[],
                                    style,
                                    1.0 - settle,
                                    interpolated_label_position,
                                );
                                if settle > 0.0 {
                                    draw_styled_edge(
                                        frame,
                                        scaled_stroke_width,
                                        scaled_corner_radius,
                                        scaled_endpoint_extension,
                                        scaled_edge_label_size,
                                        self.edge_color,
                                        self.label_color,
                                        edge,
                                        &projected_new,
                                        &projected_new_curve,
                                        style,
                                        settle,
                                        final_label_position,
                                    );
                                }
                            } else {
                                let projected = edge
                                    .points()
                                    .iter()
                                    .map(|(x, y)| project(&self.sugiyama, *x, *y))
                                    .collect::<Vec<_>>();
                                let projected_curve = edge
                                    .curve_points()
                                    .iter()
                                    .map(|(x, y)| project(&self.sugiyama, *x, *y))
                                    .collect::<Vec<_>>();
                                let label_position = edge_canvas_label_position(
                                    &self.edge_label_overlay_edges,
                                    edge.index(),
                                    edge.label_position()
                                        .map(|(x, y)| project(&self.sugiyama, x, y)),
                                );
                                draw_styled_edge(
                                    frame,
                                    scaled_stroke_width,
                                    scaled_corner_radius,
                                    scaled_endpoint_extension,
                                    scaled_edge_label_size,
                                    self.edge_color,
                                    self.label_color,
                                    edge,
                                    &projected,
                                    &projected_curve,
                                    style_for_current(edge.index()),
                                    progress,
                                    label_position,
                                );
                            }
                        }

                        for edge in old_edges.values() {
                            let projected = edge
                                .points()
                                .iter()
                                .map(|(x, y)| project(old_layout, *x, *y))
                                .collect::<Vec<_>>();
                            let projected_curve = edge
                                .curve_points()
                                .iter()
                                .map(|(x, y)| project(old_layout, *x, *y))
                                .collect::<Vec<_>>();
                            let label_position = edge_canvas_label_position(
                                &self.edge_label_overlay_edges,
                                edge.index(),
                                edge.label_position()
                                    .map(|(x, y)| project(old_layout, x, y)),
                            );
                            draw_styled_edge(
                                frame,
                                scaled_stroke_width,
                                scaled_corner_radius,
                                scaled_endpoint_extension,
                                scaled_edge_label_size,
                                self.edge_color,
                                self.label_color,
                                edge,
                                &projected,
                                &projected_curve,
                                style_for_old(edge.index()),
                                1.0 - progress,
                                label_position,
                            );
                        }
                    }
                }
            } else {
                for cluster in self.sugiyama.clusters() {
                    draw_cluster_outline(
                        frame,
                        scaled_stroke_width,
                        (self.cluster_color)(cluster.index()),
                        project(&self.sugiyama, cluster.min_x(), cluster.min_y()),
                        project(&self.sugiyama, cluster.max_x(), cluster.max_y()),
                    );
                }
                for edge in self.sugiyama.edges() {
                    let projected = edge
                        .points()
                        .iter()
                        .map(|(x, y)| project(&self.sugiyama, *x, *y))
                        .collect::<Vec<_>>();
                    let projected_curve = edge
                        .curve_points()
                        .iter()
                        .map(|(x, y)| project(&self.sugiyama, *x, *y))
                        .collect::<Vec<_>>();
                    let label_position = edge_canvas_label_position(
                        &self.edge_label_overlay_edges,
                        edge.index(),
                        edge.label_position()
                            .map(|(x, y)| project(&self.sugiyama, x, y)),
                    );
                    draw_styled_edge(
                        frame,
                        scaled_stroke_width,
                        scaled_corner_radius,
                        scaled_endpoint_extension,
                        scaled_edge_label_size,
                        self.edge_color,
                        self.label_color,
                        edge,
                        &projected,
                        &projected_curve,
                        style_for_current(edge.index()),
                        1.0,
                        label_position,
                    );
                }
            }
        });
        vec![graph]
    }
}

impl<'a, Message, Theme, Renderer> From<Sugiyama<'a, Message, Theme, Renderer>>
    for Element<'a, Message, Theme, Renderer>
where
    Renderer: iced::advanced::Renderer + iced::advanced::graphics::geometry::Renderer + 'static,
    Theme: 'static,
    Message: Clone + 'static,
{
    fn from(x: Sugiyama<'a, Message, Theme, Renderer>) -> Element<'a, Message, Theme, Renderer> {
        iced::widget::component(x)
    }
}

/// The identifier of an animated [`Sugiyama`] widget.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Id(widget::Id);

impl Id {
    /// Creates a custom [`Id`].
    pub fn new(id: impl Into<std::borrow::Cow<'static, str>>) -> Self {
        Self(id.into().into_owned().into())
    }

    /// Creates a unique [`Id`].
    ///
    /// This function produces a different [`Id`] every time it is called.
    pub fn unique() -> Self {
        Self(widget::Id::unique())
    }
}

impl From<Id> for widget::Id {
    fn from(id: Id) -> Self {
        id.0
    }
}

/// Produces a [`Task`] that forces a rebuild of the animated [`Sugiyama`] view
/// for the widget with the given [`Id`].
pub fn force_review<Message>(id: impl Into<Id>) -> Task<Message>
where
    Message: Send + 'static,
{
    struct ForceReview {
        target: widget::Id,
    }

    impl<T> widget::Operation<T> for ForceReview {
        fn custom(
            &mut self,
            id: Option<&widget::Id>,
            _bounds: iced::Rectangle,
            state: &mut dyn std::any::Any,
        ) {
            if id != Some(&self.target) {
                return;
            }

            if let Some(refresh_nonce) = state.downcast_mut::<u64>() {
                *refresh_nonce = refresh_nonce.saturating_add(1);
            }
        }

        fn traverse(&mut self, operate: &mut dyn FnMut(&mut dyn widget::Operation<T>)) {
            operate(self);
        }
    }

    iced::advanced::widget::operate::<()>(ForceReview {
        target: id.into().0,
    })
    .discard()
}

/// Produces a [`Task`] that snapshots the currently displayed graph state,
/// primes animation, and rebuilds the animated [`Sugiyama`] view for the
/// widget with the given [`Id`].
pub fn invalidate<Message>(id: impl Into<Id>) -> Task<Message>
where
    Message: Send + 'static,
{
    struct Invalidate {
        target: widget::Id,
    }

    impl<T> widget::Operation<T> for Invalidate {
        fn custom(
            &mut self,
            id: Option<&widget::Id>,
            _bounds: iced::Rectangle,
            state: &mut dyn std::any::Any,
        ) {
            if id != Some(&self.target) {
                return;
            }

            if let Some(request) = state.downcast_mut::<InvalidateRequest>() {
                request.requested = true;
            }
        }

        fn traverse(&mut self, operate: &mut dyn FnMut(&mut dyn widget::Operation<T>)) {
            operate(self);
        }
    }

    iced::advanced::widget::operate::<()>(Invalidate {
        target: id.into().0,
    })
    .discard()
}

struct GraphNodes<Message, Theme, Renderer>
where
    Renderer: iced::advanced::Renderer,
{
    children: Vec<Element<'static, Message, Theme, Renderer>>,
    nodes: Vec<u32>,
    sugiyama: GraphLayout,
    old_sugiyama: Option<GraphLayout>,
    node_map: HashMap<u32, usize>,
    old_node_map: Option<HashMap<u32, usize>>,
    edges: Vec<(u32, u32)>,
    cluster_container_children: Vec<ClusterContainerChild>,
    edge_label_children: Vec<EdgeLabelChild>,
    edge_endpoint_children: Vec<EdgeEndpointChild>,
    padding: Padding,
    edge_endpoint_extension: f32,
    motion_easing: &'static Easing,
    motion_duration: Duration,
    animation: SharedAnimation,
    viewport: SharedViewport,
}

#[derive(Debug, Clone, Copy)]
struct EdgeLabelChild {
    edge_index: usize,
}

#[derive(Debug, Clone, Copy)]
struct ClusterContainerChild {
    cluster_index: usize,
}

#[derive(Debug, Clone, Copy)]
struct EdgeEndpointChild {
    edge_index: usize,
    edge: (u32, u32),
    kind: EdgeEndpointKind,
}

impl<Message, Theme, Renderer> Widget<Message, Theme, Renderer>
    for GraphNodes<Message, Theme, Renderer>
where
    Renderer: iced::advanced::Renderer,
{
    fn size(&self) -> iced::Size<Length> {
        iced::Size {
            width: Length::Fill,
            height: Length::Fill,
        }
    }

    fn state(&self) -> tree::State {
        tree::State::new(self.animation.clone())
    }

    fn children(&self) -> Vec<Tree> {
        self.children.iter().map(Tree::new).collect()
    }

    fn diff(&self, tree: &mut Tree) {
        tree.diff_children(&self.children)
    }

    fn layout(
        &mut self,
        tree: &mut Tree,
        renderer: &Renderer,
        limits: &iced::advanced::layout::Limits,
    ) -> iced::advanced::layout::Node {
        let state = tree.state.downcast_mut::<SharedAnimation>();
        let limits = limits.shrink(Size {
            width: self.padding.left + self.padding.right,
            height: self.padding.top + self.padding.bottom,
        });
        let size = limits.max();

        let node_count = self.node_map.len();
        let cluster_container_layouts = cluster_container_layouts(
            &self.sugiyama,
            self.old_sugiyama.as_ref(),
            &self.cluster_container_children,
            size,
            state,
            self.motion_easing,
            self.motion_duration,
        );
        let cluster_container_count = cluster_container_layouts.len();
        let layouts = self
            .children
            .iter_mut()
            .zip(&mut tree.children)
            .enumerate()
            .map(|(index, (child, tree))| {
                if (node_count..(node_count + cluster_container_count)).contains(&index) {
                    let cluster_layout = cluster_container_layouts[index - node_count];
                    let child_limits = iced::advanced::layout::Limits::new(
                        cluster_layout.size,
                        cluster_layout.size,
                    );
                    child.as_widget_mut().layout(tree, renderer, &child_limits)
                } else {
                    child.as_widget_mut().layout(tree, renderer, &limits)
                }
            })
            .collect::<Vec<_>>();
        let node_sizes = layouts
            .iter()
            .take(node_count)
            .map(|layout| layout.bounds().size())
            .collect::<Vec<_>>();

        let raw_child_positions = child_positions(
            &self.sugiyama,
            self.old_sugiyama.as_ref(),
            &self.nodes,
            &self.node_map,
            self.old_node_map.as_ref(),
            &self.edges,
            &node_sizes,
            size,
            state,
            self.motion_easing,
            self.motion_duration,
        );
        let edge_label_positions = edge_label_positions(
            &self.sugiyama,
            self.old_sugiyama.as_ref(),
            &self.edge_label_children,
            size,
            state,
            self.motion_easing,
            self.motion_duration,
        );
        let edge_endpoint_positions = edge_endpoint_positions(
            &self.sugiyama,
            self.old_sugiyama.as_ref(),
            &self.node_map,
            &raw_child_positions,
            &node_sizes,
            &self.edge_endpoint_children,
            size,
            state,
            self.motion_easing,
            self.motion_duration,
            self.edge_endpoint_extension,
        );

        let mut translated_children = Vec::with_capacity(layouts.len());
        let mut layout_iter = layouts.into_iter();

        for vector in raw_child_positions {
            let Some(node) = layout_iter.next() else {
                break;
            };
            let node_bounds = node.bounds();
            translated_children.push(node.translate(Vector {
                x: vector.x - node_bounds.width / 2.0 + self.padding.left,
                y: vector.y - node_bounds.height / 2.0 + self.padding.top,
            }));
        }
        for cluster_layout in cluster_container_layouts {
            let Some(node) = layout_iter.next() else {
                break;
            };
            let node_bounds = node.bounds();
            translated_children.push(node.translate(Vector {
                x: cluster_layout.center.x - node_bounds.width / 2.0 + self.padding.left,
                y: cluster_layout.center.y - node_bounds.height / 2.0 + self.padding.top,
            }));
        }
        for vector in edge_label_positions {
            let Some(node) = layout_iter.next() else {
                break;
            };
            let node_bounds = node.bounds();
            translated_children.push(node.translate(Vector {
                x: vector.x - node_bounds.width / 2.0 + self.padding.left,
                y: vector.y - node_bounds.height / 2.0 + self.padding.top,
            }));
        }
        for vector in edge_endpoint_positions {
            let Some(node) = layout_iter.next() else {
                break;
            };
            let node_bounds = node.bounds();
            translated_children.push(node.translate(Vector {
                x: vector.x - node_bounds.width / 2.0 + self.padding.left,
                y: vector.y - node_bounds.height / 2.0 + self.padding.top,
            }));
        }

        iced::advanced::layout::Node::with_children(size, translated_children)
    }

    fn draw(
        &self,
        state: &iced::advanced::widget::Tree,
        renderer: &mut Renderer,
        theme: &Theme,
        style: &iced::advanced::renderer::Style,
        layout: iced::advanced::Layout<'_>,
        cursor: iced::advanced::mouse::Cursor,
        viewport: &iced::Rectangle,
    ) {
        let camera = self.viewport.get();
        let transform = view_transformation(layout.bounds(), self.padding, camera);
        let transformed_cursor = transform_cursor(cursor, layout.bounds(), self.padding, camera);
        let transformed_viewport =
            transform_viewport(*viewport, layout.bounds(), self.padding, camera);

        renderer.with_transformation(transform, |renderer| {
            self.children
                .iter()
                .zip(layout.children())
                .zip(&state.children)
                .for_each(|((child, layout), state)| {
                    child.as_widget().draw(
                        state,
                        renderer,
                        theme,
                        style,
                        layout,
                        transformed_cursor,
                        &transformed_viewport,
                    )
                })
        });
    }

    fn mouse_interaction(
        &self,
        tree: &Tree,
        layout: iced::advanced::Layout<'_>,
        cursor: iced::advanced::mouse::Cursor,
        viewport: &iced::Rectangle,
        renderer: &Renderer,
    ) -> iced::mouse::Interaction {
        let camera = self.viewport.get();
        let transformed_cursor = transform_cursor(cursor, layout.bounds(), self.padding, camera);
        let transformed_viewport =
            transform_viewport(*viewport, layout.bounds(), self.padding, camera);

        self.children
            .iter()
            .zip(&tree.children)
            .zip(layout.children())
            .map(|((child, state), layout)| {
                child.as_widget().mouse_interaction(
                    state,
                    layout,
                    transformed_cursor,
                    &transformed_viewport,
                    renderer,
                )
            })
            .max()
            .unwrap_or_default()
    }

    fn operate(
        &mut self,
        tree: &mut Tree,
        layout: iced::advanced::Layout<'_>,
        renderer: &Renderer,
        operation: &mut dyn Operation,
    ) {
        operation.container(None, layout.bounds());
        operation.traverse(&mut |operation| {
            self.children
                .iter_mut()
                .zip(&mut tree.children)
                .zip(layout.children())
                .for_each(|((child, state), layout)| {
                    child
                        .as_widget_mut()
                        .operate(state, layout, renderer, operation);
                })
        });
    }

    fn update(
        &mut self,
        tree: &mut Tree,
        event: &iced::Event,
        layout: iced::advanced::Layout<'_>,
        cursor: iced::advanced::mouse::Cursor,
        renderer: &Renderer,
        clipboard: &mut dyn iced::advanced::Clipboard,
        shell: &mut iced::advanced::Shell<'_, Message>,
        viewport: &iced::Rectangle,
    ) {
        let animation = tree.state.downcast_mut::<SharedAnimation>();
        if let Animation::Pending = animation.get() {
            animation.set(Animation::Active {
                start: Instant::now(),
                elapsed: Duration::ZERO,
            });
            shell.request_redraw();
        }

        let bounds = layout.bounds();
        let inner_size = bounds.size();
        let inner_cursor = cursor
            .position_in(bounds)
            .map(|position| {
                Point::new(
                    position.x - self.padding.left,
                    position.y - self.padding.top,
                )
            })
            .filter(|position| {
                position.x >= 0.0
                    && position.y >= 0.0
                    && position.x <= inner_size.width
                    && position.y <= inner_size.height
            });

        let mut viewport_status = event::Status::Ignored;
        let mut swallow_children = false;
        match &event {
            iced::Event::Mouse(iced::mouse::Event::WheelScrolled { delta }) => {
                if let Some(position) = inner_cursor {
                    let delta_y = match delta {
                        iced::mouse::ScrollDelta::Lines { y, .. }
                        | iced::mouse::ScrollDelta::Pixels { y, .. } => *y,
                    };
                    if self.viewport.zoom_at(delta_y, position, inner_size) {
                        shell.invalidate_layout();
                        shell.request_redraw();
                        viewport_status = event::Status::Captured;
                        swallow_children = true;
                    }
                }
            }
            iced::Event::Mouse(iced::mouse::Event::ButtonPressed(iced::mouse::Button::Left)) => {
                if let Some(position) = inner_cursor {
                    self.viewport.begin_drag(position, Instant::now());
                }
            }
            iced::Event::Mouse(iced::mouse::Event::CursorMoved { .. }) => {
                if self.viewport.is_dragging() {
                    if let Some(position) = inner_cursor {
                        let captured = self.viewport.drag_to(position, Instant::now());
                        shell.invalidate_layout();
                        shell.request_redraw();
                        if captured {
                            viewport_status = event::Status::Captured;
                        }
                    }
                    swallow_children = true;
                }
            }
            iced::Event::Mouse(iced::mouse::Event::ButtonReleased(iced::mouse::Button::Left)) => {
                let captured = self.viewport.end_drag(Instant::now());
                if self.viewport.is_moving() {
                    shell.request_redraw();
                }
                if captured {
                    viewport_status = event::Status::Captured;
                    swallow_children = true;
                }
            }
            _ => {}
        }

        if let iced::Event::Window(iced::window::Event::RedrawRequested(now)) = event {
            let (new_animation, redraw) =
                animation.get().timed_transition(self.motion_duration, *now);
            let viewport_needs_layout = self.viewport.tick(*now);
            if let Some(redraw) = redraw {
                shell.invalidate_layout();
                shell.request_redraw_at(redraw);
            }
            if viewport_needs_layout {
                shell.invalidate_layout();
            }
            if self.viewport.is_moving() {
                shell.request_redraw_at(*now + FRAME_DURATION);
            }
            animation.set(new_animation);
        }

        let child_status = if swallow_children {
            event::Status::Ignored
        } else {
            let camera = self.viewport.get();
            let transformed_cursor =
                transform_cursor(cursor, layout.bounds(), self.padding, camera);
            let transformed_viewport =
                transform_viewport(*viewport, layout.bounds(), self.padding, camera);
            self.children
                .iter_mut()
                .zip(&mut tree.children)
                .zip(layout.children())
                .for_each(|((child, state), layout)| {
                    child.as_widget_mut().update(
                        state,
                        event,
                        layout,
                        transformed_cursor,
                        renderer,
                        clipboard,
                        shell,
                        &transformed_viewport,
                    );
                });
            shell.event_status()
        };

        if event::Status::merge(viewport_status, child_status) == event::Status::Captured {
            shell.capture_event();
        }
    }

    fn overlay<'b>(
        &'b mut self,
        tree: &'b mut Tree,
        layout: iced::advanced::Layout<'b>,
        renderer: &Renderer,
        viewport: &iced::Rectangle,
        translation: Vector,
    ) -> Option<iced::advanced::overlay::Element<'b, Message, Theme, Renderer>> {
        let graph_bounds = iced::Rectangle {
            x: layout.bounds().x + translation.x,
            y: layout.bounds().y + translation.y,
            width: layout.bounds().width,
            height: layout.bounds().height,
        };

        iced::advanced::overlay::from_children(
            &mut self.children,
            tree,
            layout,
            renderer,
            viewport,
            translation,
        )
        .map(|content| {
            iced::advanced::overlay::Element::new(Box::new(ViewTransformedOverlay {
                content,
                graph_bounds,
                padding: self.padding,
                viewport: self.viewport.get(),
            }))
        })
    }
}

struct ViewTransformedOverlay<'a, Message, Theme, Renderer>
where
    Renderer: iced::advanced::Renderer,
{
    content: iced::advanced::overlay::Element<'a, Message, Theme, Renderer>,
    graph_bounds: iced::Rectangle,
    padding: Padding,
    viewport: ViewportState,
}

impl<'a, Message, Theme, Renderer> iced::advanced::Overlay<Message, Theme, Renderer>
    for ViewTransformedOverlay<'a, Message, Theme, Renderer>
where
    Renderer: iced::advanced::Renderer,
{
    fn layout(&mut self, renderer: &Renderer, bounds: Size) -> iced::advanced::layout::Node {
        self.content.as_overlay_mut().layout(renderer, bounds)
    }

    fn draw(
        &self,
        renderer: &mut Renderer,
        theme: &Theme,
        style: &iced::advanced::renderer::Style,
        layout: iced::advanced::Layout<'_>,
        cursor: iced::mouse::Cursor,
    ) {
        let transform = view_transformation(self.graph_bounds, self.padding, self.viewport);
        let transformed_cursor =
            transform_cursor(cursor, self.graph_bounds, self.padding, self.viewport);
        renderer.with_transformation(transform, |renderer| {
            self.content
                .as_overlay()
                .draw(renderer, theme, style, layout, transformed_cursor);
        });
    }

    fn operate(
        &mut self,
        layout: iced::advanced::Layout<'_>,
        renderer: &Renderer,
        operation: &mut dyn iced::advanced::widget::Operation,
    ) {
        self.content
            .as_overlay_mut()
            .operate(layout, renderer, operation);
    }

    fn update(
        &mut self,
        event: &iced::Event,
        layout: iced::advanced::Layout<'_>,
        cursor: iced::mouse::Cursor,
        renderer: &Renderer,
        clipboard: &mut dyn iced::advanced::Clipboard,
        shell: &mut iced::advanced::Shell<'_, Message>,
    ) {
        let transformed_cursor =
            transform_cursor(cursor, self.graph_bounds, self.padding, self.viewport);
        self.content.as_overlay_mut().update(
            event,
            layout,
            transformed_cursor,
            renderer,
            clipboard,
            shell,
        );
    }

    fn mouse_interaction(
        &self,
        layout: iced::advanced::Layout<'_>,
        cursor: iced::mouse::Cursor,
        renderer: &Renderer,
    ) -> iced::mouse::Interaction {
        let transformed_cursor =
            transform_cursor(cursor, self.graph_bounds, self.padding, self.viewport);
        self.content
            .as_overlay()
            .mouse_interaction(layout, transformed_cursor, renderer)
    }

    fn overlay<'b>(
        &'b mut self,
        layout: iced::advanced::Layout<'b>,
        renderer: &Renderer,
    ) -> Option<iced::advanced::overlay::Element<'b, Message, Theme, Renderer>> {
        self.content
            .as_overlay_mut()
            .overlay(layout, renderer)
            .map(|content| {
            iced::advanced::overlay::Element::new(Box::new(ViewTransformedOverlay {
                content,
                graph_bounds: self.graph_bounds,
                padding: self.padding,
                viewport: self.viewport,
            }))
        })
    }
}

fn lerp(a: f64, b: f64, t: f32) -> f64 {
    a + (b - a) * t as f64
}

fn edge_canvas_label_position(
    overlay_edges: &HashSet<usize>,
    edge_index: usize,
    candidate: Option<Point>,
) -> Option<Point> {
    if overlay_edges.contains(&edge_index) {
        None
    } else {
        candidate
    }
}

fn draw_cluster_outline<Renderer>(
    frame: &mut canvas::Frame<Renderer>,
    stroke_width: f32,
    color: Color,
    a: Point,
    b: Point,
) where
    Renderer: iced::advanced::graphics::geometry::Renderer,
{
    let top_left = Point::new(a.x.min(b.x), a.y.min(b.y));
    let rect_size = Size::new((a.x - b.x).abs().max(1.0), (a.y - b.y).abs().max(1.0));
    frame.stroke(
        &Path::rectangle(top_left, rect_size),
        canvas::Stroke {
            width: stroke_width.max(1.0) * 0.5,
            style: canvas::stroke::Style::Solid(color),
            ..canvas::Stroke::default()
        },
    );
}

fn draw_styled_edge<Renderer>(
    frame: &mut canvas::Frame<Renderer>,
    scaled_stroke_width: f32,
    scaled_corner_radius: f32,
    scaled_endpoint_extension: f32,
    scaled_edge_label_size: f32,
    edge_color: fn(usize) -> (Color, Color),
    label_color: fn(usize) -> Color,
    edge: &crate::layout_engine::EdgeLayout,
    projected_points: &[Point],
    projected_curve_points: &[Point],
    style: OutgoingEdgeStyle,
    base_alpha: f32,
    label_position: Option<Point>,
) where
    Renderer: iced::advanced::graphics::geometry::Renderer,
{
    if !style.visible {
        return;
    }

    let alpha = (base_alpha * style.alpha).clamp(0.0, 1.0);
    if alpha <= f32::EPSILON {
        return;
    }

    let width_scale = style.width_scale.max(0.0);
    if width_scale <= f32::EPSILON {
        return;
    }

    let stroke_width = scaled_stroke_width * width_scale;
    let corner_radius = scaled_corner_radius * width_scale;
    let endpoint_extension = scaled_endpoint_extension * width_scale;
    let label_size = scaled_edge_label_size * width_scale.max(0.25);

    let (mut from_color, mut to_color) = edge_color(edge.index());
    if let Some((from_override, to_override)) = style.color_override {
        from_color = from_override;
        to_color = to_override;
    }

    draw_edge_with_label(
        frame,
        stroke_width,
        corner_radius,
        endpoint_extension,
        edge,
        projected_points,
        projected_curve_points,
        from_color,
        to_color,
        label_color(edge.index()),
        alpha,
        label_size,
        label_position,
    );
}

fn draw_edge_with_label<Renderer>(
    frame: &mut canvas::Frame<Renderer>,
    stroke_width: f32,
    corner_radius: f32,
    endpoint_extension: f32,
    edge: &crate::layout_engine::EdgeLayout,
    points: &[Point],
    curve_points: &[Point],
    from_color: Color,
    to_color: Color,
    label_color: Color,
    alpha: f32,
    label_text_size: f32,
    label_position: Option<Point>,
) where
    Renderer: iced::advanced::graphics::geometry::Renderer,
{
    if points.len() < 2 {
        return;
    }
    let adjusted = extend_polyline_endpoints(points, endpoint_extension);
    let start = adjusted[0];
    let end = adjusted[adjusted.len() - 1];
    frame.stroke(
        &edge_path(&adjusted, curve_points, corner_radius, endpoint_extension),
        canvas::Stroke {
            width: stroke_width,
            style: canvas::stroke::Style::Gradient(canvas::Gradient::Linear(
                canvas::gradient::Linear::new(start, end)
                    .add_stop(0.0, to_color.scale_alpha(alpha))
                    .add_stop(1.0, from_color.scale_alpha(alpha)),
            )),
            line_cap: canvas::LineCap::Round,
            ..canvas::Stroke::default()
        },
    );

    if let (Some(label), Some(position)) = (edge.label(), label_position) {
        frame.fill_text(canvas::Text {
            content: label.to_string(),
            position,
            color: label_color.scale_alpha(alpha),
            size: iced::Pixels(label_text_size.max(1.0)),
            align_x: iced::alignment::Horizontal::Center.into(),
            align_y: iced::alignment::Vertical::Center.into(),
            ..canvas::Text::default()
        });
    }
}

fn rounded_polyline_path(points: &[Point], radius: f32) -> Path {
    Path::new(|path| {
        if points.is_empty() {
            return;
        }
        if points.len() == 1 {
            path.move_to(points[0]);
            return;
        }

        path.move_to(points[0]);
        if radius <= f32::EPSILON || points.len() == 2 {
            for point in &points[1..] {
                path.line_to(*point);
            }
            return;
        }

        for i in 1..points.len() - 1 {
            let prev = points[i - 1];
            let current = points[i];
            let next = points[i + 1];

            let in_vec = Vector::new(prev.x - current.x, prev.y - current.y);
            let out_vec = Vector::new(next.x - current.x, next.y - current.y);
            let in_len = (in_vec.x * in_vec.x + in_vec.y * in_vec.y).sqrt();
            let out_len = (out_vec.x * out_vec.x + out_vec.y * out_vec.y).sqrt();

            if in_len <= f32::EPSILON || out_len <= f32::EPSILON {
                path.line_to(current);
                continue;
            }

            let corner = radius.min(in_len * 0.5).min(out_len * 0.5);
            let in_norm = Vector::new(in_vec.x / in_len, in_vec.y / in_len);
            let out_norm = Vector::new(out_vec.x / out_len, out_vec.y / out_len);

            let start = Point::new(
                current.x + in_norm.x * corner,
                current.y + in_norm.y * corner,
            );
            let end = Point::new(
                current.x + out_norm.x * corner,
                current.y + out_norm.y * corner,
            );

            path.line_to(start);
            path.quadratic_curve_to(current, end);
        }

        path.line_to(*points.last().unwrap_or(&points[0]));
    })
}

fn edge_path(
    points: &[Point],
    curve_points: &[Point],
    radius: f32,
    endpoint_extension: f32,
) -> Path {
    if endpoint_extension <= f32::EPSILON && bezier_curve_points_are_valid(curve_points) {
        return bezier_curve_path(curve_points);
    }

    rounded_polyline_path(points, radius)
}

fn bezier_curve_points_are_valid(curve_points: &[Point]) -> bool {
    curve_points.len() >= 4 && (curve_points.len() - 1) % 3 == 0
}

fn bezier_curve_path(curve_points: &[Point]) -> Path {
    Path::new(|path| {
        let Some(start) = curve_points.first().copied() else {
            return;
        };

        path.move_to(start);

        for chunk in curve_points[1..].chunks(3) {
            match chunk {
                [control_a, control_b, end] => {
                    path.bezier_curve_to(*control_a, *control_b, *end);
                }
                [end] => path.line_to(*end),
                _ => {}
            }
        }
    })
}

fn extend_polyline_endpoints(points: &[Point], extension: f32) -> Vec<Point> {
    if points.len() < 2 || extension <= f32::EPSILON {
        return points.to_vec();
    }

    let mut adjusted = points.to_vec();

    let first = adjusted[0];
    let second = adjusted[1];
    let first_vec = Vector::new(second.x - first.x, second.y - first.y);
    let first_len = (first_vec.x * first_vec.x + first_vec.y * first_vec.y).sqrt();
    if first_len > f32::EPSILON {
        let first_extension = extension.min(first_len * 0.49);
        let first_norm = Vector::new(first_vec.x / first_len, first_vec.y / first_len);
        adjusted[0] = Point::new(
            first.x - first_norm.x * first_extension,
            first.y - first_norm.y * first_extension,
        );
    }

    let last_index = adjusted.len() - 1;
    let last = adjusted[last_index];
    let penultimate = adjusted[last_index - 1];
    let last_vec = Vector::new(last.x - penultimate.x, last.y - penultimate.y);
    let last_len = (last_vec.x * last_vec.x + last_vec.y * last_vec.y).sqrt();
    if last_len > f32::EPSILON {
        let last_extension = extension.min(last_len * 0.49);
        let last_norm = Vector::new(last_vec.x / last_len, last_vec.y / last_len);
        adjusted[last_index] = Point::new(
            last.x + last_norm.x * last_extension,
            last.y + last_norm.y * last_extension,
        );
    }

    adjusted
}

fn interpolate_orthogonal_polylines(from: &[Point], to: &[Point], t: f32) -> Vec<Point> {
    let samples = from.len().max(to.len()).max(2);
    let from_resampled = resample_polyline(from, samples);
    let to_resampled = resample_polyline(to, samples);

    let blended = from_resampled
        .iter()
        .zip(to_resampled.iter())
        .map(|(a, b)| Point::new(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t))
        .collect::<Vec<_>>();

    if blended.len() < 2 {
        return blended;
    }

    let mut orthogonal = Vec::with_capacity(blended.len() * 2);
    orthogonal.push(blended[0]);

    for i in 0..blended.len().saturating_sub(1) {
        let target = blended[i + 1];
        let current = match orthogonal.last().copied() {
            Some(point) => point,
            None => target,
        };

        if point_close(current, target) {
            continue;
        }
        if almost_equal_f32(current.x, target.x) || almost_equal_f32(current.y, target.y) {
            orthogonal.push(target);
            continue;
        }

        let from_dx = (from_resampled[i + 1].x - from_resampled[i].x).abs();
        let from_dy = (from_resampled[i + 1].y - from_resampled[i].y).abs();
        let to_dx = (to_resampled[i + 1].x - to_resampled[i].x).abs();
        let to_dy = (to_resampled[i + 1].y - to_resampled[i].y).abs();
        let horizontal_score = from_dx * (1.0 - t) + to_dx * t;
        let vertical_score = from_dy * (1.0 - t) + to_dy * t;
        let horizontal_first = horizontal_score >= vertical_score;

        let corner = if horizontal_first {
            Point::new(target.x, current.y)
        } else {
            Point::new(current.x, target.y)
        };

        if !point_close(current, corner) {
            orthogonal.push(corner);
        }
        if !point_close(corner, target) {
            orthogonal.push(target);
        }
    }

    simplify_orthogonal_polyline(&orthogonal)
}

fn resample_polyline(points: &[Point], samples: usize) -> Vec<Point> {
    if points.is_empty() {
        return Vec::new();
    }
    if points.len() == 1 {
        return vec![points[0]; samples];
    }

    let mut lengths = Vec::with_capacity(points.len().saturating_sub(1));
    let mut total_length = 0.0f32;
    for segment in points.windows(2) {
        let dx = segment[1].x - segment[0].x;
        let dy = segment[1].y - segment[0].y;
        let length = (dx * dx + dy * dy).sqrt();
        lengths.push(length);
        total_length += length;
    }

    if total_length <= f32::EPSILON {
        return vec![points[0]; samples];
    }

    let mut result = Vec::with_capacity(samples);
    for sample in 0..samples {
        let distance = if samples <= 1 {
            0.0
        } else {
            total_length * sample as f32 / (samples as f32 - 1.0)
        };
        result.push(point_at_distance(points, &lengths, distance));
    }
    result
}

fn point_at_distance(points: &[Point], lengths: &[f32], mut distance: f32) -> Point {
    for (index, length) in lengths.iter().enumerate() {
        if distance <= *length || index + 1 == lengths.len() {
            let start = points[index];
            let end = points[index + 1];
            let t = if *length <= f32::EPSILON {
                0.0
            } else {
                (distance / *length).clamp(0.0, 1.0)
            };
            return Point::new(
                start.x + (end.x - start.x) * t,
                start.y + (end.y - start.y) * t,
            );
        }
        distance -= *length;
    }
    match points.last().copied() {
        Some(point) => point,
        None => Point::ORIGIN,
    }
}

fn simplify_orthogonal_polyline(points: &[Point]) -> Vec<Point> {
    if points.is_empty() {
        return Vec::new();
    }

    let mut simplified = Vec::with_capacity(points.len());
    for point in points {
        if match simplified.last().copied() {
            Some(last) => point_close(last, *point),
            None => false,
        } {
            continue;
        }

        simplified.push(*point);

        while simplified.len() >= 3 {
            let len = simplified.len();
            let a = simplified[len - 3];
            let b = simplified[len - 2];
            let c = simplified[len - 1];
            let collinear_x = almost_equal_f32(a.x, b.x) && almost_equal_f32(b.x, c.x);
            let collinear_y = almost_equal_f32(a.y, b.y) && almost_equal_f32(b.y, c.y);
            if collinear_x || collinear_y {
                simplified.remove(len - 2);
            } else {
                break;
            }
        }
    }

    simplified
}

fn point_close(a: Point, b: Point) -> bool {
    almost_equal_f32(a.x, b.x) && almost_equal_f32(a.y, b.y)
}

fn almost_equal_f32(a: f32, b: f32) -> bool {
    (a - b).abs() <= 1e-3
}

fn view_center(bounds: iced::Rectangle, padding: Padding) -> Point {
    Point::new(
        bounds.x + padding.left + bounds.width * 0.5,
        bounds.y + padding.top + bounds.height * 0.5,
    )
}

fn view_transformation(
    bounds: iced::Rectangle,
    padding: Padding,
    viewport: ViewportState,
) -> Transformation {
    let center = view_center(bounds, padding);
    Transformation::translate(center.x + viewport.pan.x, center.y + viewport.pan.y)
        * Transformation::scale(viewport.zoom)
        * Transformation::translate(-center.x, -center.y)
}

fn inverse_view_transform_point(
    position: Point,
    bounds: iced::Rectangle,
    padding: Padding,
    viewport: ViewportState,
) -> Point {
    let center = view_center(bounds, padding);
    Point::new(
        center.x + (position.x - center.x - viewport.pan.x) / viewport.zoom,
        center.y + (position.y - center.y - viewport.pan.y) / viewport.zoom,
    )
}

fn transform_viewport(
    viewport_rect: iced::Rectangle,
    bounds: iced::Rectangle,
    padding: Padding,
    viewport: ViewportState,
) -> iced::Rectangle {
    let top_left = inverse_view_transform_point(
        Point::new(viewport_rect.x, viewport_rect.y),
        bounds,
        padding,
        viewport,
    );
    let bottom_right = inverse_view_transform_point(
        Point::new(
            viewport_rect.x + viewport_rect.width,
            viewport_rect.y + viewport_rect.height,
        ),
        bounds,
        padding,
        viewport,
    );

    iced::Rectangle {
        x: top_left.x.min(bottom_right.x),
        y: top_left.y.min(bottom_right.y),
        width: (bottom_right.x - top_left.x).abs(),
        height: (bottom_right.y - top_left.y).abs(),
    }
}

fn transform_cursor(
    cursor: iced::mouse::Cursor,
    bounds: iced::Rectangle,
    padding: Padding,
    viewport: ViewportState,
) -> iced::mouse::Cursor {
    let Some(position) = cursor.position() else {
        return cursor;
    };
    let mapped = inverse_view_transform_point(position, bounds, padding, viewport);
    iced::mouse::Cursor::Available(mapped)
}

fn apply_view_transform(point: Vector, size: Size, viewport: ViewportState) -> Vector {
    let center = Vector::new(size.width * 0.5, size.height * 0.5);
    Vector::new(
        center.x + (point.x - center.x) * viewport.zoom + viewport.pan.x,
        center.y + (point.y - center.y) * viewport.zoom + viewport.pan.y,
    )
}

fn layout_offset(sugiyama: &GraphLayout, size: iced::Size) -> Vector {
    Vector {
        x: (size.width - sugiyama.max_x() as f32).max(0.0) * 0.5,
        y: (size.height - sugiyama.max_y() as f32).max(0.0) * 0.5,
    }
}

fn edge_endpoint_metadata(
    edge: &crate::layout_engine::EdgeLayout,
    kind: EdgeEndpointKind,
    node_center: Vector,
    node_size: Size,
    endpoint_extension: f32,
) -> Option<EdgeEndpoint> {
    let points = edge
        .points()
        .iter()
        .map(|(x, y)| Point::new(*x as f32, *y as f32))
        .collect::<Vec<_>>();
    let curve_points = edge
        .curve_points()
        .iter()
        .map(|(x, y)| Point::new(*x as f32, *y as f32))
        .collect::<Vec<_>>();
    endpoint_anchor_and_direction(
        &points,
        &curve_points,
        kind,
        node_center,
        node_size,
        endpoint_extension,
    )
    .map(|(_, direction)| EdgeEndpoint {
        direction_x: direction.x,
        direction_y: direction.y,
    })
}

#[allow(clippy::too_many_arguments)]
fn edge_label_positions(
    sugiyama: &GraphLayout,
    old_sugiyama: Option<&GraphLayout>,
    edge_label_children: &[EdgeLabelChild],
    size: iced::Size,
    animation: &SharedAnimation,
    motion_easing: &'static Easing,
    motion_duration: Duration,
) -> Vec<Vector> {
    if edge_label_children.is_empty() {
        return Vec::new();
    }

    if let Some(old_sugiyama) = old_sugiyama {
        let animation = animation.get();
        let old_offset = layout_offset(old_sugiyama, size);
        let new_offset = layout_offset(sugiyama, size);
        let old_edges = old_sugiyama
            .edges()
            .iter()
            .map(|edge| (edge.index(), edge))
            .collect::<HashMap<_, _>>();
        let new_edges = sugiyama
            .edges()
            .iter()
            .map(|edge| (edge.index(), edge))
            .collect::<HashMap<_, _>>();

        match animation {
            Animation::Pending => edge_label_children
                .iter()
                .map(|child| {
                    if let Some(old_edge) = old_edges.get(&child.edge_index) {
                        edge_label_anchor(old_edge, old_offset)
                            .unwrap_or(Vector::new(old_offset.x, old_offset.y))
                    } else if let Some(new_edge) = new_edges.get(&child.edge_index) {
                        edge_label_anchor(new_edge, new_offset)
                            .unwrap_or(Vector::new(new_offset.x, new_offset.y))
                    } else {
                        Vector::new(new_offset.x, new_offset.y)
                    }
                })
                .collect(),
            Animation::Active { .. } => {
                let progress = animation
                    .progress(motion_easing, motion_duration)
                    .unwrap_or(1.0);
                edge_label_children
                    .iter()
                    .map(|child| {
                        let old_anchor = old_edges
                            .get(&child.edge_index)
                            .and_then(|edge| edge_label_anchor(edge, old_offset));
                        let new_anchor = new_edges
                            .get(&child.edge_index)
                            .and_then(|edge| edge_label_anchor(edge, new_offset));

                        match (old_anchor, new_anchor) {
                            (Some(old_anchor), Some(new_anchor)) => Vector::new(
                                lerp(old_anchor.x as f64, new_anchor.x as f64, progress) as f32,
                                lerp(old_anchor.y as f64, new_anchor.y as f64, progress) as f32,
                            ),
                            (Some(old_anchor), None) => old_anchor,
                            (None, Some(new_anchor)) => new_anchor,
                            (None, None) => Vector::new(new_offset.x, new_offset.y),
                        }
                    })
                    .collect()
            }
            Animation::Complete => edge_label_children
                .iter()
                .map(|child| {
                    new_edges
                        .get(&child.edge_index)
                        .and_then(|edge| edge_label_anchor(edge, new_offset))
                        .unwrap_or(Vector::new(new_offset.x, new_offset.y))
                })
                .collect(),
        }
    } else {
        let offset = layout_offset(sugiyama, size);
        let edges = sugiyama
            .edges()
            .iter()
            .map(|edge| (edge.index(), edge))
            .collect::<HashMap<_, _>>();
        edge_label_children
            .iter()
            .map(|child| {
                edges
                    .get(&child.edge_index)
                    .and_then(|edge| edge_label_anchor(edge, offset))
                    .unwrap_or(Vector::new(offset.x, offset.y))
            })
            .collect()
    }
}

fn edge_label_anchor(edge: &crate::layout_engine::EdgeLayout, offset: Vector) -> Option<Vector> {
    if let Some((x, y)) = edge.label_position() {
        return Some(Vector::new(x as f32 + offset.x, y as f32 + offset.y));
    }

    polyline_midpoint_f64(&edge.points())
        .map(|(x, y)| Vector::new(x as f32 + offset.x, y as f32 + offset.y))
}

fn polyline_midpoint_f64(points: &[(f64, f64)]) -> Option<(f64, f64)> {
    if points.len() < 2 {
        return points.first().copied();
    }

    let total = points
        .windows(2)
        .map(|segment| {
            let dx = segment[1].0 - segment[0].0;
            let dy = segment[1].1 - segment[0].1;
            (dx * dx + dy * dy).sqrt()
        })
        .sum::<f64>();
    if total <= f64::EPSILON {
        return points.first().copied();
    }

    let target = total * 0.5;
    let mut cumulative = 0.0f64;
    for segment in points.windows(2) {
        let a = segment[0];
        let b = segment[1];
        let dx = b.0 - a.0;
        let dy = b.1 - a.1;
        let length = (dx * dx + dy * dy).sqrt();
        if length <= f64::EPSILON {
            continue;
        }
        if cumulative + length >= target {
            let t = ((target - cumulative) / length).clamp(0.0, 1.0);
            return Some((a.0 + dx * t, a.1 + dy * t));
        }
        cumulative += length;
    }

    points.last().copied()
}

#[allow(clippy::too_many_arguments)]
fn edge_endpoint_positions(
    sugiyama: &GraphLayout,
    old_sugiyama: Option<&GraphLayout>,
    node_map: &HashMap<u32, usize>,
    node_positions: &[Vector],
    node_sizes: &[Size],
    endpoint_children: &[EdgeEndpointChild],
    size: iced::Size,
    animation: &SharedAnimation,
    motion_easing: &'static Easing,
    motion_duration: Duration,
    endpoint_extension: f32,
) -> Vec<Vector> {
    if endpoint_children.is_empty() {
        return Vec::new();
    }

    if let Some(old_sugiyama) = old_sugiyama {
        let animation = animation.get();
        match animation {
            Animation::Pending => {
                let old_offset = layout_offset(old_sugiyama, size);
                let new_offset = layout_offset(sugiyama, size);
                let old_edges = old_sugiyama
                    .edges()
                    .iter()
                    .map(|edge| (edge.index(), edge))
                    .collect::<HashMap<_, _>>();
                let new_edges = sugiyama
                    .edges()
                    .iter()
                    .map(|edge| (edge.index(), edge))
                    .collect::<HashMap<_, _>>();

                endpoint_children
                    .iter()
                    .map(|child| {
                        let points = if let Some(old_edge) =
                            old_edges.get(&child.edge_index).copied()
                        {
                            old_edge
                                .points()
                                .iter()
                                .map(|(x, y)| {
                                    Point::new(*x as f32 + old_offset.x, *y as f32 + old_offset.y)
                                })
                                .collect::<Vec<_>>()
                        } else if let Some(new_edge) = new_edges.get(&child.edge_index).copied() {
                            new_edge
                                .points()
                                .iter()
                                .map(|(x, y)| {
                                    Point::new(*x as f32 + new_offset.x, *y as f32 + new_offset.y)
                                })
                                .collect::<Vec<_>>()
                        } else {
                            Vec::new()
                        };
                        let curve_points = if let Some(old_edge) =
                            old_edges.get(&child.edge_index).copied()
                        {
                            old_edge
                                .curve_points()
                                .iter()
                                .map(|(x, y)| {
                                    Point::new(*x as f32 + old_offset.x, *y as f32 + old_offset.y)
                                })
                                .collect::<Vec<_>>()
                        } else if let Some(new_edge) = new_edges.get(&child.edge_index).copied() {
                            new_edge
                                .curve_points()
                                .iter()
                                .map(|(x, y)| {
                                    Point::new(*x as f32 + new_offset.x, *y as f32 + new_offset.y)
                                })
                                .collect::<Vec<_>>()
                        } else {
                            Vec::new()
                        };

                        match (
                            (!points.is_empty()).then_some(points),
                            (!curve_points.is_empty()).then_some(curve_points),
                            node_center_and_size(node_map, node_positions, node_sizes, child),
                        ) {
                            (Some(points), curve_points, Some((center, node_size))) => {
                                endpoint_anchor_and_direction(
                                    &points,
                                    curve_points.as_deref().unwrap_or(&[]),
                                    child.kind,
                                    center,
                                    node_size,
                                    endpoint_extension,
                                )
                                .map(|(anchor, direction)| {
                                    offset_endpoint_anchor(anchor, direction, child.kind)
                                })
                                .unwrap_or(center)
                            }
                            (Some(points), _, None) => fallback_endpoint_anchor_from_points(
                                &points,
                                child.kind,
                                endpoint_extension,
                            )
                            .unwrap_or(Vector::new(new_offset.x, new_offset.y)),
                            (None, _, Some((center, _))) => center,
                            (None, _, None) => Vector::new(new_offset.x, new_offset.y),
                        }
                    })
                    .collect()
            }
            Animation::Active { .. } => {
                let progress = animation
                    .progress(motion_easing, motion_duration)
                    .unwrap_or(1.0);
                let old_offset = layout_offset(old_sugiyama, size);
                let new_offset = layout_offset(sugiyama, size);
                let old_edges = old_sugiyama
                    .edges()
                    .iter()
                    .map(|edge| (edge.index(), edge))
                    .collect::<HashMap<_, _>>();
                let new_edges = sugiyama
                    .edges()
                    .iter()
                    .map(|edge| (edge.index(), edge))
                    .collect::<HashMap<_, _>>();

                endpoint_children
                    .iter()
                    .map(|child| {
                        let old_edge = old_edges.get(&child.edge_index).copied();
                        let new_edge = new_edges.get(&child.edge_index).copied();

                        let points = match (old_edge, new_edge) {
                            (Some(old_edge), Some(new_edge)) => {
                                let projected_old = old_edge
                                    .points()
                                    .iter()
                                    .map(|(x, y)| {
                                        Point::new(
                                            *x as f32 + old_offset.x,
                                            *y as f32 + old_offset.y,
                                        )
                                    })
                                    .collect::<Vec<_>>();
                                let projected_new = new_edge
                                    .points()
                                    .iter()
                                    .map(|(x, y)| {
                                        Point::new(
                                            *x as f32 + new_offset.x,
                                            *y as f32 + new_offset.y,
                                        )
                                    })
                                    .collect::<Vec<_>>();
                                interpolate_orthogonal_polylines(
                                    &projected_old,
                                    &projected_new,
                                    progress,
                                )
                            }
                            (None, Some(new_edge)) => new_edge
                                .points()
                                .iter()
                                .map(|(x, y)| {
                                    Point::new(*x as f32 + new_offset.x, *y as f32 + new_offset.y)
                                })
                                .collect::<Vec<_>>(),
                            (Some(old_edge), None) => old_edge
                                .points()
                                .iter()
                                .map(|(x, y)| {
                                    Point::new(*x as f32 + old_offset.x, *y as f32 + old_offset.y)
                                })
                                .collect::<Vec<_>>(),
                            (None, None) => Vec::new(),
                        };

                        match (
                            (!points.is_empty()).then_some(points),
                            None::<Vec<Point>>,
                            node_center_and_size(node_map, node_positions, node_sizes, child),
                        ) {
                            (Some(points), curve_points, Some((center, node_size))) => {
                                endpoint_anchor_and_direction(
                                    &points,
                                    curve_points.as_deref().unwrap_or(&[]),
                                    child.kind,
                                    center,
                                    node_size,
                                    endpoint_extension,
                                )
                                .map(|(anchor, direction)| {
                                    offset_endpoint_anchor(anchor, direction, child.kind)
                                })
                                .unwrap_or(center)
                            }
                            (Some(points), _, None) => fallback_endpoint_anchor_from_points(
                                &points,
                                child.kind,
                                endpoint_extension,
                            )
                            .unwrap_or(Vector::new(new_offset.x, new_offset.y)),
                            (None, _, Some((center, _))) => center,
                            (None, _, None) => Vector::new(new_offset.x, new_offset.y),
                        }
                    })
                    .collect()
            }
            Animation::Complete => {
                let offset = layout_offset(sugiyama, size);
                let edges = sugiyama
                    .edges()
                    .iter()
                    .map(|edge| (edge.index(), edge))
                    .collect::<HashMap<_, _>>();
                endpoint_children
                    .iter()
                    .map(|child| {
                        let points = edges
                            .get(&child.edge_index)
                            .map(|edge| {
                                edge.points()
                                    .iter()
                                    .map(|(x, y)| {
                                        Point::new(*x as f32 + offset.x, *y as f32 + offset.y)
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .unwrap_or_default();
                        let curve_points = edges
                            .get(&child.edge_index)
                            .map(|edge| {
                                edge.curve_points()
                                    .iter()
                                    .map(|(x, y)| {
                                        Point::new(*x as f32 + offset.x, *y as f32 + offset.y)
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .unwrap_or_default();
                        match (
                            (!points.is_empty()).then_some(points),
                            (!curve_points.is_empty()).then_some(curve_points),
                            node_center_and_size(node_map, node_positions, node_sizes, child),
                        ) {
                            (Some(points), curve_points, Some((center, node_size))) => {
                                endpoint_anchor_and_direction(
                                    &points,
                                    curve_points.as_deref().unwrap_or(&[]),
                                    child.kind,
                                    center,
                                    node_size,
                                    endpoint_extension,
                                )
                                .map(|(anchor, direction)| {
                                    offset_endpoint_anchor(anchor, direction, child.kind)
                                })
                                .unwrap_or(center)
                            }
                            (Some(points), _, None) => fallback_endpoint_anchor_from_points(
                                &points,
                                child.kind,
                                endpoint_extension,
                            )
                            .unwrap_or(Vector::new(offset.x, offset.y)),
                            (None, _, Some((center, _))) => center,
                            (None, _, None) => Vector::new(offset.x, offset.y),
                        }
                    })
                    .collect()
            }
        }
    } else {
        let offset = layout_offset(sugiyama, size);
        let edges = sugiyama
            .edges()
            .iter()
            .map(|edge| (edge.index(), edge))
            .collect::<HashMap<_, _>>();
        endpoint_children
            .iter()
            .map(|child| {
                let points = edges
                    .get(&child.edge_index)
                    .map(|edge| {
                        edge.points()
                            .iter()
                            .map(|(x, y)| Point::new(*x as f32 + offset.x, *y as f32 + offset.y))
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                let curve_points = edges
                    .get(&child.edge_index)
                    .map(|edge| {
                        edge.curve_points()
                            .iter()
                            .map(|(x, y)| Point::new(*x as f32 + offset.x, *y as f32 + offset.y))
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                match (
                    (!points.is_empty()).then_some(points),
                    (!curve_points.is_empty()).then_some(curve_points),
                    node_center_and_size(node_map, node_positions, node_sizes, child),
                ) {
                    (Some(points), curve_points, Some((center, node_size))) => {
                        endpoint_anchor_and_direction(
                            &points,
                            curve_points.as_deref().unwrap_or(&[]),
                            child.kind,
                            center,
                            node_size,
                            endpoint_extension,
                        )
                        .map(|(anchor, direction)| {
                            offset_endpoint_anchor(anchor, direction, child.kind)
                        })
                        .unwrap_or(center)
                    }
                    (Some(points), _, None) => fallback_endpoint_anchor_from_points(
                        &points,
                        child.kind,
                        endpoint_extension,
                    )
                    .unwrap_or(Vector::new(offset.x, offset.y)),
                    (None, _, Some((center, _))) => center,
                    (None, _, None) => Vector::new(offset.x, offset.y),
                }
            })
            .collect()
    }
}

fn node_center_and_size(
    node_map: &HashMap<u32, usize>,
    node_positions: &[Vector],
    node_sizes: &[Size],
    endpoint_child: &EdgeEndpointChild,
) -> Option<(Vector, Size)> {
    let node_id = match endpoint_child.kind {
        EdgeEndpointKind::Source => endpoint_child.edge.0,
        EdgeEndpointKind::Destination => endpoint_child.edge.1,
    };
    let node_index = node_map.get(&node_id).copied()?;
    let center = node_positions.get(node_index).copied()?;
    let size = node_sizes.get(node_index).copied()?;
    Some((center, size))
}

fn endpoint_anchor_and_direction(
    points: &[Point],
    curve_points: &[Point],
    kind: EdgeEndpointKind,
    node_center: Vector,
    node_size: Size,
    endpoint_extension: f32,
) -> Option<(Vector, Vector)> {
    if points.len() < 2 {
        return None;
    }

    if endpoint_extension <= f32::EPSILON
        && bezier_curve_points_are_valid(curve_points)
        && curve_endpoint_on_node_boundary(curve_points, kind, node_center, node_size)
    {
        let anchor = match kind {
            EdgeEndpointKind::Source => curve_points.first().copied(),
            EdgeEndpointKind::Destination => curve_points.last().copied(),
        }
        .map(|point| Vector::new(point.x, point.y))?;
        let direction = endpoint_center_direction(anchor, node_center, kind)
            .or_else(|| endpoint_curve_direction(curve_points, kind))?;
        return Some((anchor, direction));
    }

    let adjusted = extend_polyline_endpoints(points, endpoint_extension);
    match kind {
        EdgeEndpointKind::Source => {
            for segment in adjusted.windows(2) {
                let from = segment[0];
                let to = segment[1];
                let inside_from = point_inside_node_rect(from, node_center, node_size);
                let inside_to = point_inside_node_rect(to, node_center, node_size);
                if inside_from && !inside_to {
                    let anchor = segment_boundary_intersection(from, to, node_center, node_size)
                        .unwrap_or(Vector::new(from.x, from.y));
                    let direction = endpoint_center_direction(anchor, node_center, kind)
                        .or_else(|| endpoint_curve_direction(curve_points, kind))
                        .or_else(|| normalize_vector(Vector::new(to.x - from.x, to.y - from.y)))
                        .unwrap_or(Vector::new(1.0, 0.0));
                    return Some((anchor, direction));
                }
            }
        }
        EdgeEndpointKind::Destination => {
            for segment in adjusted.windows(2).rev() {
                let from = segment[0];
                let to = segment[1];
                let inside_from = point_inside_node_rect(from, node_center, node_size);
                let inside_to = point_inside_node_rect(to, node_center, node_size);
                if !inside_from && inside_to {
                    let anchor = segment_boundary_intersection(to, from, node_center, node_size)
                        .unwrap_or(Vector::new(to.x, to.y));
                    let direction = endpoint_center_direction(anchor, node_center, kind)
                        .or_else(|| endpoint_curve_direction(curve_points, kind))
                        .or_else(|| normalize_vector(Vector::new(to.x - from.x, to.y - from.y)))
                        .unwrap_or(Vector::new(1.0, 0.0));
                    return Some((anchor, direction));
                }
            }
        }
    }

    let (from, to) = match kind {
        EdgeEndpointKind::Source => (adjusted[0], adjusted[1]),
        EdgeEndpointKind::Destination => {
            let last = adjusted.len() - 1;
            (adjusted[last - 1], adjusted[last])
        }
    };
    let anchor = segment_boundary_intersection(
        Point::new(node_center.x, node_center.y),
        Point::new(to.x, to.y),
        node_center,
        node_size,
    )
    .unwrap_or(node_center);
    let direction = endpoint_center_direction(anchor, node_center, kind)
        .or_else(|| endpoint_curve_direction(curve_points, kind))
        .or_else(|| normalize_vector(Vector::new(to.x - from.x, to.y - from.y)))?;
    Some((anchor, direction))
}

fn endpoint_center_direction(
    anchor: Vector,
    node_center: Vector,
    kind: EdgeEndpointKind,
) -> Option<Vector> {
    let direction = match kind {
        EdgeEndpointKind::Source => Vector::new(anchor.x - node_center.x, anchor.y - node_center.y),
        EdgeEndpointKind::Destination => {
            Vector::new(node_center.x - anchor.x, node_center.y - anchor.y)
        }
    };
    normalize_vector(direction)
}

fn fallback_endpoint_anchor_from_points(
    points: &[Point],
    kind: EdgeEndpointKind,
    endpoint_extension: f32,
) -> Option<Vector> {
    if points.len() < 2 {
        return None;
    }
    let adjusted = extend_polyline_endpoints(points, endpoint_extension);
    match kind {
        EdgeEndpointKind::Source => adjusted.first().map(|point| Vector::new(point.x, point.y)),
        EdgeEndpointKind::Destination => adjusted.last().map(|point| Vector::new(point.x, point.y)),
    }
}

fn offset_endpoint_anchor(anchor: Vector, direction: Vector, kind: EdgeEndpointKind) -> Vector {
    let offset_distance = 0.0;
    let sign = match kind {
        EdgeEndpointKind::Source => 1.0,
        EdgeEndpointKind::Destination => -1.0,
    };

    Vector::new(
        anchor.x + direction.x * offset_distance * sign,
        anchor.y + direction.y * offset_distance * sign,
    )
}

fn point_inside_node_rect(point: Point, node_center: Vector, node_size: Size) -> bool {
    let half_width = node_size.width * 0.5;
    let half_height = node_size.height * 0.5;
    let min_x = node_center.x - half_width;
    let max_x = node_center.x + half_width;
    let min_y = node_center.y - half_height;
    let max_y = node_center.y + half_height;
    point.x >= min_x - 1e-3
        && point.x <= max_x + 1e-3
        && point.y >= min_y - 1e-3
        && point.y <= max_y + 1e-3
}

fn segment_boundary_intersection(
    inside: Point,
    outside: Point,
    node_center: Vector,
    node_size: Size,
) -> Option<Vector> {
    let dx = outside.x - inside.x;
    let dy = outside.y - inside.y;
    if dx.abs() <= f32::EPSILON && dy.abs() <= f32::EPSILON {
        return None;
    }

    let radius = (node_size.width.min(node_size.height) * 0.25).min(18.0);
    segment_boundary_intersection_t(inside, outside, node_center, node_size)
        .and_then(|t| {
            let x = inside.x + dx * t;
            let y = inside.y + dy * t;
            rounded_rect_boundary_intersection(
                Point::new(x, y),
                normalize_vector(Vector::new(dx, dy))?,
                node_center,
                node_size,
                radius,
            )
        })
        .or_else(|| {
            segment_boundary_intersection_t(inside, outside, node_center, node_size)
                .map(|t| Vector::new(inside.x + dx * t, inside.y + dy * t))
        })
}

fn segment_boundary_intersection_t(
    inside: Point,
    outside: Point,
    node_center: Vector,
    node_size: Size,
) -> Option<f32> {
    let dx = outside.x - inside.x;
    let dy = outside.y - inside.y;
    if dx.abs() <= f32::EPSILON && dy.abs() <= f32::EPSILON {
        return None;
    }

    let half_width = node_size.width * 0.5;
    let half_height = node_size.height * 0.5;
    let min_x = node_center.x - half_width;
    let max_x = node_center.x + half_width;
    let min_y = node_center.y - half_height;
    let max_y = node_center.y + half_height;

    let mut t_values = Vec::with_capacity(2);
    if dx > f32::EPSILON {
        t_values.push((max_x - inside.x) / dx);
    } else if dx < -f32::EPSILON {
        t_values.push((min_x - inside.x) / dx);
    }
    if dy > f32::EPSILON {
        t_values.push((max_y - inside.y) / dy);
    } else if dy < -f32::EPSILON {
        t_values.push((min_y - inside.y) / dy);
    }

    t_values
        .into_iter()
        .filter(|t| *t >= -1e-4 && *t <= 1.0 + 1e-4)
        .filter_map(|t| {
            let x = inside.x + dx * t;
            let y = inside.y + dy * t;
            if x >= min_x - 1e-2 && x <= max_x + 1e-2 && y >= min_y - 1e-2 && y <= max_y + 1e-2 {
                Some(t)
            } else {
                None
            }
        })
        .min_by(|a, b| a.total_cmp(b))
}

fn rounded_rect_boundary_intersection(
    rect_intersection: Point,
    direction: Vector,
    node_center: Vector,
    node_size: Size,
    radius: f32,
) -> Option<Vector> {
    let half_width = node_size.width * 0.5;
    let half_height = node_size.height * 0.5;
    let abs_x = (rect_intersection.x - node_center.x).abs();
    let abs_y = (rect_intersection.y - node_center.y).abs();
    let inner_half_width = (half_width - radius).max(0.0);
    let inner_half_height = (half_height - radius).max(0.0);

    let corner_center = Vector::new(
        node_center.x + (rect_intersection.x - node_center.x).signum() * inner_half_width,
        node_center.y + (rect_intersection.y - node_center.y).signum() * inner_half_height,
    );

    if abs_x <= inner_half_width + 1e-3 || abs_y <= inner_half_height + 1e-3 || radius <= 1e-3 {
        return Some(Vector::new(rect_intersection.x, rect_intersection.y));
    }

    let ray_start = Vector::new(node_center.x, node_center.y);
    ray_circle_intersection(ray_start, direction, corner_center, radius)
        .or_else(|| Some(Vector::new(rect_intersection.x, rect_intersection.y)))
}

fn ray_circle_intersection(
    origin: Vector,
    direction: Vector,
    center: Vector,
    radius: f32,
) -> Option<Vector> {
    let ox = origin.x - center.x;
    let oy = origin.y - center.y;
    let a = direction.x * direction.x + direction.y * direction.y;
    let b = 2.0 * (ox * direction.x + oy * direction.y);
    let c = ox * ox + oy * oy - radius * radius;
    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        return None;
    }
    let sqrt_discriminant = discriminant.sqrt();
    let mut roots = [
        (-b - sqrt_discriminant) / (2.0 * a),
        (-b + sqrt_discriminant) / (2.0 * a),
    ];
    roots.sort_by(|left, right| left.total_cmp(right));
    roots
        .into_iter()
        .find(|t| *t >= 0.0)
        .map(|t| Vector::new(origin.x + direction.x * t, origin.y + direction.y * t))
}

fn endpoint_curve_direction(curve_points: &[Point], kind: EdgeEndpointKind) -> Option<Vector> {
    if curve_points.len() >= 4 && (curve_points.len() - 1) % 3 == 0 {
        return match kind {
            EdgeEndpointKind::Source => curve_points.get(1).and_then(|control| {
                normalize_vector(Vector::new(
                    control.x - curve_points[0].x,
                    control.y - curve_points[0].y,
                ))
            }),
            EdgeEndpointKind::Destination => {
                let end = curve_points.last().copied()?;
                let control = curve_points
                    .get(curve_points.len().saturating_sub(2))
                    .copied()?;
                normalize_vector(Vector::new(end.x - control.x, end.y - control.y))
            }
        };
    }

    None
}

fn curve_endpoint_on_node_boundary(
    curve_points: &[Point],
    kind: EdgeEndpointKind,
    node_center: Vector,
    node_size: Size,
) -> bool {
    let Some(point) = (match kind {
        EdgeEndpointKind::Source => curve_points.first().copied(),
        EdgeEndpointKind::Destination => curve_points.last().copied(),
    }) else {
        return false;
    };

    point_inside_node_rect(point, node_center, node_size)
}

fn normalize_vector(vector: Vector) -> Option<Vector> {
    let length = (vector.x * vector.x + vector.y * vector.y).sqrt();
    if length <= f32::EPSILON {
        return None;
    }
    Some(Vector::new(vector.x / length, vector.y / length))
}

#[derive(Debug, Clone, Copy)]
struct ClusterContainerLayout {
    center: Vector,
    size: Size,
}

#[allow(clippy::too_many_arguments)]
fn cluster_container_layouts(
    sugiyama: &GraphLayout,
    old_sugiyama: Option<&GraphLayout>,
    cluster_container_children: &[ClusterContainerChild],
    size: iced::Size,
    animation: &SharedAnimation,
    motion_easing: &'static Easing,
    motion_duration: Duration,
) -> Vec<ClusterContainerLayout> {
    if cluster_container_children.is_empty() {
        return Vec::new();
    }

    if let Some(old_sugiyama) = old_sugiyama {
        let animation = animation.get();
        match animation {
            Animation::Pending => {
                let old_offset = layout_offset(old_sugiyama, size);
                let new_offset = layout_offset(sugiyama, size);
                cluster_container_children
                    .iter()
                    .map(|child| {
                        if let Some(cluster) = old_sugiyama
                            .clusters()
                            .iter()
                            .find(|cluster| cluster.index() == child.cluster_index)
                        {
                            cluster_container_layout(cluster, old_offset)
                        } else if let Some(cluster) = sugiyama
                            .clusters()
                            .iter()
                            .find(|cluster| cluster.index() == child.cluster_index)
                        {
                            cluster_container_layout(cluster, new_offset)
                        } else {
                            ClusterContainerLayout {
                                center: Vector::new(new_offset.x, new_offset.y),
                                size: Size::new(1.0, 1.0),
                            }
                        }
                    })
                    .collect()
            }
            Animation::Active { .. } => {
                let progress = animation
                    .progress(motion_easing, motion_duration)
                    .unwrap_or(1.0);
                let old_offset = layout_offset(old_sugiyama, size);
                let new_offset = layout_offset(sugiyama, size);
                cluster_container_children
                    .iter()
                    .map(|child| {
                        let old_cluster = old_sugiyama
                            .clusters()
                            .iter()
                            .find(|cluster| cluster.index() == child.cluster_index);
                        let new_cluster = sugiyama
                            .clusters()
                            .iter()
                            .find(|cluster| cluster.index() == child.cluster_index);
                        match (old_cluster, new_cluster) {
                            (Some(old_cluster), Some(new_cluster)) => {
                                cluster_container_layout_from_bounds(
                                    lerp(old_cluster.min_x(), new_cluster.min_x(), progress),
                                    lerp(old_cluster.min_y(), new_cluster.min_y(), progress),
                                    lerp(old_cluster.max_x(), new_cluster.max_x(), progress),
                                    lerp(old_cluster.max_y(), new_cluster.max_y(), progress),
                                    new_offset,
                                )
                            }
                            (None, Some(new_cluster)) => {
                                cluster_container_layout(new_cluster, new_offset)
                            }
                            (Some(old_cluster), None) => {
                                cluster_container_layout(old_cluster, old_offset)
                            }
                            (None, None) => ClusterContainerLayout {
                                center: Vector::new(new_offset.x, new_offset.y),
                                size: Size::new(1.0, 1.0),
                            },
                        }
                    })
                    .collect()
            }
            Animation::Complete => {
                let offset = layout_offset(sugiyama, size);
                cluster_container_children
                    .iter()
                    .map(|child| {
                        sugiyama
                            .clusters()
                            .iter()
                            .find(|cluster| cluster.index() == child.cluster_index)
                            .map(|cluster| cluster_container_layout(cluster, offset))
                            .unwrap_or(ClusterContainerLayout {
                                center: Vector::new(offset.x, offset.y),
                                size: Size::new(1.0, 1.0),
                            })
                    })
                    .collect()
            }
        }
    } else {
        let offset = layout_offset(sugiyama, size);
        cluster_container_children
            .iter()
            .map(|child| {
                sugiyama
                    .clusters()
                    .iter()
                    .find(|cluster| cluster.index() == child.cluster_index)
                    .map(|cluster| cluster_container_layout(cluster, offset))
                    .unwrap_or(ClusterContainerLayout {
                        center: Vector::new(offset.x, offset.y),
                        size: Size::new(1.0, 1.0),
                    })
            })
            .collect()
    }
}

fn cluster_container_layout(
    cluster: &crate::layout_engine::ClusterLayout,
    offset: Vector,
) -> ClusterContainerLayout {
    cluster_container_layout_from_bounds(
        cluster.min_x(),
        cluster.min_y(),
        cluster.max_x(),
        cluster.max_y(),
        offset,
    )
}

fn cluster_container_layout_from_bounds(
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
    offset: Vector,
) -> ClusterContainerLayout {
    let width = (max_x - min_x).max(1.0) as f32;
    let height = (max_y - min_y).max(1.0) as f32;
    ClusterContainerLayout {
        center: Vector::new(
            min_x as f32 + width * 0.5 + offset.x,
            min_y as f32 + height * 0.5 + offset.y,
        ),
        size: Size::new(width, height),
    }
}

#[allow(clippy::too_many_arguments)]
fn child_positions(
    sugiyama: &GraphLayout,
    old_sugiyama: Option<&GraphLayout>,
    nodes: &[u32],
    node_map: &HashMap<u32, usize>,
    old_node_map: Option<&HashMap<u32, usize>>,
    edges: &[(u32, u32)],
    node_sizes: &[Size],
    size: iced::Size,
    animation: &SharedAnimation,
    motion_easing: &'static Easing,
    motion_duration: Duration,
) -> Vec<Vector> {
    let mut positions = vec![Vector::new(0.0, 0.0); node_map.len()];
    let new_offset = layout_offset(sugiyama, size);

    let new_position = |node_id: u32| -> Option<Vector> {
        let new_index = node_map.get(&node_id).copied()?;
        let (x, y) = sugiyama.position(new_index)?;
        Some(Vector {
            x: x as f32 + new_offset.x,
            y: y as f32 + new_offset.y,
        })
    };

    let Some(old_sugiyama) = old_sugiyama else {
        for node_id in nodes {
            let Some(index) = node_map.get(node_id).copied() else {
                continue;
            };
            let Some(new_pos) = new_position(*node_id) else {
                continue;
            };
            positions[index] = new_pos;
        }
        return positions;
    };

    let old_offset = layout_offset(old_sugiyama, size);
    let old_position = |node_id: u32| -> Option<Vector> {
        let old_index = old_node_map?.get(&node_id).copied()?;
        let (x, y) = old_sugiyama.position(old_index)?;
        Some(Vector {
            x: x as f32 + old_offset.x,
            y: y as f32 + old_offset.y,
        })
    };

    let persisted_nodes = nodes
        .iter()
        .copied()
        .filter(|node_id| old_position(*node_id).is_some())
        .collect::<HashSet<_>>();

    let hidden_persisted_nodes = nodes
        .iter()
        .copied()
        .filter(|node_id| {
            let Some(index) = node_map.get(node_id).copied() else {
                return false;
            };
            let Some(size) = node_sizes.get(index).copied() else {
                return false;
            };
            size.width <= 0.5 && size.height <= 0.5 && persisted_nodes.contains(node_id)
        })
        .collect::<Vec<_>>();

    let choose_nearest =
        |candidates: &[u32], target: Vector, position_for: &dyn Fn(u32) -> Option<Vector>| {
            candidates
                .iter()
                .copied()
                .filter_map(|candidate| {
                    let position = position_for(candidate)?;
                    let dx = position.x - target.x;
                    let dy = position.y - target.y;
                    Some((candidate, dx * dx + dy * dy))
                })
                .min_by(|(_, left), (_, right)| left.total_cmp(right))
                .map(|(node_id, _)| node_id)
        };

    let gained_source = |node_id: u32, target: Vector| -> Option<Vector> {
        let incoming_persisted = edges
            .iter()
            .filter_map(|(from, to)| {
                if *to == node_id && persisted_nodes.contains(from) {
                    Some(*from)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        if let Some(source_node) = choose_nearest(&incoming_persisted, target, &old_position) {
            return old_position(source_node);
        }

        if let Some(source_node) = choose_nearest(&hidden_persisted_nodes, target, &old_position) {
            return old_position(source_node);
        }

        let fallback_candidates = persisted_nodes.iter().copied().collect::<Vec<_>>();
        if let Some(source_node) = choose_nearest(&fallback_candidates, target, &old_position) {
            return old_position(source_node);
        }

        None
    };

    let animation = animation.get();
    let progress = match animation {
        Animation::Pending => 0.0,
        Animation::Active { .. } => animation
            .progress(motion_easing, motion_duration)
            .unwrap_or(1.0),
        Animation::Complete => 1.0,
    };

    for node_id in nodes {
        let Some(index) = node_map.get(node_id).copied() else {
            continue;
        };
        let Some(new_pos) = new_position(*node_id) else {
            continue;
        };

        let position = if let Some(old_pos) = old_position(*node_id) {
            Vector {
                x: lerp(old_pos.x as f64, new_pos.x as f64, progress) as f32,
                y: lerp(old_pos.y as f64, new_pos.y as f64, progress) as f32,
            }
        } else if let Some(source) = gained_source(*node_id, new_pos) {
            Vector {
                x: lerp(source.x as f64, new_pos.x as f64, progress) as f32,
                y: lerp(source.y as f64, new_pos.y as f64, progress) as f32,
            }
        } else {
            new_pos
        };

        positions[index] = position;
    }

    positions
}

impl<'a, Message, Theme, Renderer> From<GraphNodes<Message, Theme, Renderer>>
    for Element<'a, Message, Theme, Renderer>
where
    Renderer: iced::advanced::Renderer + 'a,
    Message: Clone + 'a,
    Theme: 'a,
{
    fn from(x: GraphNodes<Message, Theme, Renderer>) -> Self {
        Self::new(x)
    }
}

fn normalize_outgoing_edge_style(style: OutgoingEdgeStyle) -> OutgoingEdgeStyle {
    OutgoingEdgeStyle {
        visible: style.visible,
        width_scale: style.width_scale.max(0.0),
        alpha: style.alpha.clamp(0.0, 1.0),
        color_override: style.color_override,
    }
}
