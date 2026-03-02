#![allow(deprecated)]

use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::Hash;
use std::rc::Rc;
use std::time::Duration;

use iced::advanced::widget::{Operation, Tree, Widget, tree};
use iced::time::Instant;
use iced::widget::canvas::{self, Path};
use iced::widget::{Component, Lazy, Stack};
use iced::window::RedrawRequest;
use iced::{Color, Element, Length, Padding, Point, Size, Vector, event};

pub use crate::layout_engine::Cluster;
use crate::layout_engine::{GraphLayout, compute_layout};
use crate::motion::easing::Easing;

pub mod motion;
mod switch;
pub use switch::Switch;

const FRAME_RATE_HZ: u64 = 60;
const FRAME_DURATION: Duration = Duration::from_millis(1000 / FRAME_RATE_HZ);

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
    pub config: rust_sugiyama::configure::Config,
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
            config: rust_sugiyama::configure::Config::default(),
        }
    }

    pub fn config(self, config: rust_sugiyama::configure::Config) -> Self {
        Self { config, ..self }
    }
}

#[derive(Clone)]
pub enum Event<Message> {
    Passthrough(Message),
    Noop,
}

/// An iced widget which draws a layered graph of elements
pub struct Sugiyama<'a, Message, Theme, Renderer> {
    graph: Cow<'a, Graph>,
    view_node: Box<dyn Fn(u32) -> Element<'static, Message, Theme, Renderer> + 'a>,
    stroke_width: f32,
    edge_corner_radius: f32,
    edge_endpoint_extension: f32,
    edge_color: fn(usize) -> (Color, Color),
    edge_label: fn(usize, (u32, u32)) -> Option<String>,
    node_size: fn(u32) -> (f64, f64),
    clusters: Vec<Cluster>,
    render_config: rust_sugiyama::advanced::RenderConfig,
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
            view_node: Box::new(view_node),
            stroke_width: 2.0,
            edge_corner_radius: 10.0,
            edge_endpoint_extension: 8.0,
            edge_color: |_| (Color::BLACK, Color::BLACK.scale_alpha(0.5)),
            edge_label: |_, _| None,
            node_size: |_| (56.0, 32.0),
            clusters: Vec::new(),
            render_config: Default::default(),
            cluster_color: |_| Color::from_rgba8(90, 90, 90, 0.6),
            label_color: |_| Color::BLACK,
            padding: iced::Padding::ZERO,
            motion_easing: &motion::easing::EMPHASIZED,
            motion_duration: motion::duration::MEDIUM_4,
        }
    }

    pub fn edge_color(mut self, f: fn(usize) -> (Color, Color)) -> Self {
        self.edge_color = f;
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

    pub fn edge_label(mut self, f: fn(usize, (u32, u32)) -> Option<String>) -> Self {
        self.edge_label = f;
        self
    }

    pub fn node_size(mut self, f: fn(u32) -> (f64, f64)) -> Self {
        self.node_size = f;
        self
    }

    pub fn clusters(mut self, clusters: Vec<Cluster>) -> Self {
        self.clusters = clusters;
        self
    }

    pub fn render_config(mut self, config: rust_sugiyama::advanced::RenderConfig) -> Self {
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
    type State = (switch::State, Option<Graph>, SharedAnimation);

    fn view(&self, state: &Self::State) -> Element<'_, Self::Event, Theme, Renderer> {
        let switch_state = state.0;
        Lazy::new(
            (&self.graph, state.1.clone(), switch_state, state.2.clone()),
            |(graph, old, switch_state, animation)| {
                let children = self
                    .graph
                    .nodes
                    .iter()
                    .map(|n| (self.view_node)(*n).map(Event::Passthrough))
                    .collect();

                let node_map = self
                    .graph
                    .nodes
                    .iter()
                    .enumerate()
                    .map(|(i, n)| (*n, i))
                    .collect::<HashMap<_, _>>();

                let old_sugiyama = old.as_ref().map(|g| {
                    compute_layout(
                        &g.nodes,
                        &g.edges,
                        &g.config,
                        self.node_size,
                        self.edge_label,
                        &self.clusters,
                        &self.render_config,
                    )
                });

                let sugiyama = compute_layout(
                    &graph.nodes,
                    &graph.edges,
                    &graph.config,
                    self.node_size,
                    self.edge_label,
                    &self.clusters,
                    &self.render_config,
                );
                let overlay = GraphNodes::<Event<Message>, Theme, Renderer> {
                    children,
                    sugiyama: sugiyama.clone(),
                    old_sugiyama: old_sugiyama.clone(),
                    node_map: node_map.clone(),
                    edges: self.graph.edges.clone(),
                    padding: self.padding,
                    motion_easing: self.motion_easing,
                    motion_duration: self.motion_duration,
                    animation: animation.clone(),
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
                            cluster_color: self.cluster_color,
                            label_color: self.label_color,
                            animation: animation.clone(),
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
        state.0.flip();
        *state = (
            state.0,
            Some(self.graph.clone().into_owned()),
            SharedAnimation::default(),
        );

        match event {
            Event::Noop => None,
            Event::Passthrough(message) => Some(message),
        }
    }
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
    cluster_color: fn(usize) -> Color,
    label_color: fn(usize) -> Color,
    animation: SharedAnimation,
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
            let project = |layout: &GraphLayout, x: f64, y: f64| {
                let offset = layout_offset(layout, size);
                Point::new(
                    x as f32 + offset.x + self.padding.left,
                    y as f32 + offset.y + self.padding.top,
                )
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
                        for cluster in &old_layout.clusters {
                            draw_cluster_outline(
                                frame,
                                self.stroke_width,
                                (self.cluster_color)(cluster.index),
                                project(old_layout, cluster.min_x, cluster.min_y),
                                project(old_layout, cluster.max_x, cluster.max_y),
                            );
                        }
                        for edge in &old_layout.edges {
                            let projected = edge
                                .points
                                .iter()
                                .map(|(x, y)| project(old_layout, *x, *y))
                                .collect::<Vec<_>>();
                            let label_position =
                                edge.label_position.map(|(x, y)| project(old_layout, x, y));
                            let (from_color, to_color) = (self.edge_color)(edge.index);
                            draw_edge_with_label(
                                frame,
                                self.stroke_width,
                                self.edge_corner_radius,
                                self.edge_endpoint_extension,
                                edge,
                                &projected,
                                from_color,
                                to_color,
                                (self.label_color)(edge.index),
                                1.0,
                                label_position,
                            );
                        }
                    }
                    Animation::Complete => {
                        for cluster in &self.sugiyama.clusters {
                            draw_cluster_outline(
                                frame,
                                self.stroke_width,
                                (self.cluster_color)(cluster.index),
                                project(&self.sugiyama, cluster.min_x, cluster.min_y),
                                project(&self.sugiyama, cluster.max_x, cluster.max_y),
                            );
                        }
                        for edge in &self.sugiyama.edges {
                            let projected = edge
                                .points
                                .iter()
                                .map(|(x, y)| project(&self.sugiyama, *x, *y))
                                .collect::<Vec<_>>();
                            let label_position = edge
                                .label_position
                                .map(|(x, y)| project(&self.sugiyama, x, y));
                            let (from_color, to_color) = (self.edge_color)(edge.index);
                            draw_edge_with_label(
                                frame,
                                self.stroke_width,
                                self.edge_corner_radius,
                                self.edge_endpoint_extension,
                                edge,
                                &projected,
                                from_color,
                                to_color,
                                (self.label_color)(edge.index),
                                1.0,
                                label_position,
                            );
                        }
                    }
                    Animation::Active { .. } => {
                        let mut old_clusters = old_layout
                            .clusters
                            .iter()
                            .map(|cluster| (cluster.index, cluster))
                            .collect::<HashMap<_, _>>();

                        for cluster in &self.sugiyama.clusters {
                            if let Some(old_cluster) = old_clusters.remove(&cluster.index) {
                                let interpolated = crate::layout_engine::ClusterLayout {
                                    index: cluster.index,
                                    min_x: lerp(old_cluster.min_x, cluster.min_x, progress),
                                    min_y: lerp(old_cluster.min_y, cluster.min_y, progress),
                                    max_x: lerp(old_cluster.max_x, cluster.max_x, progress),
                                    max_y: lerp(old_cluster.max_y, cluster.max_y, progress),
                                };
                                draw_cluster_outline(
                                    frame,
                                    self.stroke_width,
                                    (self.cluster_color)(cluster.index),
                                    project(&self.sugiyama, interpolated.min_x, interpolated.min_y),
                                    project(&self.sugiyama, interpolated.max_x, interpolated.max_y),
                                );
                            } else {
                                draw_cluster_outline(
                                    frame,
                                    self.stroke_width,
                                    (self.cluster_color)(cluster.index).scale_alpha(progress),
                                    project(&self.sugiyama, cluster.min_x, cluster.min_y),
                                    project(&self.sugiyama, cluster.max_x, cluster.max_y),
                                );
                            }
                        }
                        for cluster in old_clusters.values() {
                            draw_cluster_outline(
                                frame,
                                self.stroke_width,
                                (self.cluster_color)(cluster.index).scale_alpha(1.0 - progress),
                                project(old_layout, cluster.min_x, cluster.min_y),
                                project(old_layout, cluster.max_x, cluster.max_y),
                            );
                        }

                        let mut old_edges = old_layout
                            .edges
                            .iter()
                            .map(|edge| (edge.index, edge))
                            .collect::<HashMap<_, _>>();

                        for edge in &self.sugiyama.edges {
                            if let Some(old_edge) = old_edges.remove(&edge.index) {
                                let projected_old = old_edge
                                    .points
                                    .iter()
                                    .map(|(x, y)| project(old_layout, *x, *y))
                                    .collect::<Vec<_>>();
                                let projected_new = edge
                                    .points
                                    .iter()
                                    .map(|(x, y)| project(&self.sugiyama, *x, *y))
                                    .collect::<Vec<_>>();
                                let (from_color, to_color) = (self.edge_color)(edge.index);
                                draw_edge_with_label(
                                    frame,
                                    self.stroke_width,
                                    self.edge_corner_radius,
                                    self.edge_endpoint_extension,
                                    old_edge,
                                    &projected_old,
                                    from_color,
                                    to_color,
                                    (self.label_color)(edge.index),
                                    1.0 - progress,
                                    old_edge
                                        .label_position
                                        .map(|(x, y)| project(old_layout, x, y)),
                                );
                                draw_edge_with_label(
                                    frame,
                                    self.stroke_width,
                                    self.edge_corner_radius,
                                    self.edge_endpoint_extension,
                                    edge,
                                    &projected_new,
                                    from_color,
                                    to_color,
                                    (self.label_color)(edge.index),
                                    progress,
                                    edge.label_position
                                        .map(|(x, y)| project(&self.sugiyama, x, y)),
                                );
                            } else {
                                let projected = edge
                                    .points
                                    .iter()
                                    .map(|(x, y)| project(&self.sugiyama, *x, *y))
                                    .collect::<Vec<_>>();
                                let label_position = edge
                                    .label_position
                                    .map(|(x, y)| project(&self.sugiyama, x, y));
                                let (from_color, to_color) = (self.edge_color)(edge.index);
                                draw_edge_with_label(
                                    frame,
                                    self.stroke_width,
                                    self.edge_corner_radius,
                                    self.edge_endpoint_extension,
                                    edge,
                                    &projected,
                                    from_color,
                                    to_color,
                                    (self.label_color)(edge.index),
                                    progress,
                                    label_position,
                                );
                            }
                        }

                        for edge in old_edges.values() {
                            let projected = edge
                                .points
                                .iter()
                                .map(|(x, y)| project(old_layout, *x, *y))
                                .collect::<Vec<_>>();
                            let label_position =
                                edge.label_position.map(|(x, y)| project(old_layout, x, y));
                            let (from_color, to_color) = (self.edge_color)(edge.index);
                            draw_edge_with_label(
                                frame,
                                self.stroke_width,
                                self.edge_corner_radius,
                                self.edge_endpoint_extension,
                                edge,
                                &projected,
                                from_color,
                                to_color,
                                (self.label_color)(edge.index),
                                1.0 - progress,
                                label_position,
                            );
                        }
                    }
                }
            } else {
                for cluster in &self.sugiyama.clusters {
                    draw_cluster_outline(
                        frame,
                        self.stroke_width,
                        (self.cluster_color)(cluster.index),
                        project(&self.sugiyama, cluster.min_x, cluster.min_y),
                        project(&self.sugiyama, cluster.max_x, cluster.max_y),
                    );
                }
                for edge in &self.sugiyama.edges {
                    let projected = edge
                        .points
                        .iter()
                        .map(|(x, y)| project(&self.sugiyama, *x, *y))
                        .collect::<Vec<_>>();
                    let label_position = edge
                        .label_position
                        .map(|(x, y)| project(&self.sugiyama, x, y));
                    let (from_color, to_color) = (self.edge_color)(edge.index);
                    draw_edge_with_label(
                        frame,
                        self.stroke_width,
                        self.edge_corner_radius,
                        self.edge_endpoint_extension,
                        edge,
                        &projected,
                        from_color,
                        to_color,
                        (self.label_color)(edge.index),
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

struct GraphNodes<Message, Theme, Renderer>
where
    Renderer: iced::advanced::Renderer,
{
    children: Vec<Element<'static, Message, Theme, Renderer>>,
    sugiyama: GraphLayout,
    old_sugiyama: Option<GraphLayout>,
    node_map: HashMap<u32, usize>,
    edges: Vec<(u32, u32)>,
    padding: Padding,
    motion_easing: &'static Easing,
    motion_duration: Duration,
    animation: SharedAnimation,
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
        &self,
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

        let layouts = self
            .children
            .iter()
            .zip(&mut tree.children)
            .map(|(child, tree)| child.as_widget().layout(tree, renderer, &limits))
            .collect::<Vec<_>>();

        let child_positions = child_positions(
            &self.sugiyama,
            self.old_sugiyama.as_ref(),
            &self.node_map,
            &self.edges,
            size,
            state,
            self.motion_easing,
            self.motion_duration,
        );

        let children = child_positions
            .into_iter()
            .zip(layouts)
            .map(|(vector, node)| {
                let node_bounds = node.bounds();
                node.translate(Vector {
                    x: vector.x - node_bounds.width / 2.0 + self.padding.left,
                    y: vector.y - node_bounds.height / 2.0 + self.padding.top,
                })
            })
            .collect::<Vec<_>>();

        iced::advanced::layout::Node::with_children(size, children)
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
        self.children
            .iter()
            .zip(layout.children())
            .zip(&state.children)
            .for_each(|((child, layout), state)| {
                child
                    .as_widget()
                    .draw(state, renderer, theme, style, layout, cursor, viewport)
            })
    }

    fn mouse_interaction(
        &self,
        tree: &Tree,
        layout: iced::advanced::Layout<'_>,
        cursor: iced::advanced::mouse::Cursor,
        viewport: &iced::Rectangle,
        renderer: &Renderer,
    ) -> iced::mouse::Interaction {
        self.children
            .iter()
            .zip(&tree.children)
            .zip(layout.children())
            .map(|((child, state), layout)| {
                child
                    .as_widget()
                    .mouse_interaction(state, layout, cursor, viewport, renderer)
            })
            .max()
            .unwrap_or_default()
    }

    fn operate(
        &self,
        tree: &mut Tree,
        layout: iced::advanced::Layout<'_>,
        renderer: &Renderer,
        operation: &mut dyn Operation,
    ) {
        operation.container(None, layout.bounds(), &mut |operation| {
            self.children
                .iter()
                .zip(&mut tree.children)
                .zip(layout.children())
                .for_each(|((child, state), layout)| {
                    child
                        .as_widget()
                        .operate(state, layout, renderer, operation);
                })
        });
    }

    fn on_event(
        &mut self,
        tree: &mut Tree,
        event: iced::Event,
        layout: iced::advanced::Layout<'_>,
        cursor: iced::advanced::mouse::Cursor,
        renderer: &Renderer,
        clipboard: &mut dyn iced::advanced::Clipboard,
        shell: &mut iced::advanced::Shell<'_, Message>,
        viewport: &iced::Rectangle,
    ) -> iced::event::Status {
        let state = tree.state.downcast_mut::<SharedAnimation>();
        if let Animation::Pending = state.get() {
            state.set(Animation::Active {
                start: Instant::now(),
                elapsed: Duration::ZERO,
            });
            shell.request_redraw(RedrawRequest::NextFrame);
        }

        if let iced::Event::Window(iced::window::Event::RedrawRequested(now)) = event {
            let (new_animation, redraw) = state.get().timed_transition(self.motion_duration, now);
            if let Some(redraw) = redraw {
                shell.invalidate_layout();
                shell.request_redraw(redraw);
            }
            state.set(new_animation);
        }

        self.children
            .iter_mut()
            .zip(&mut tree.children)
            .zip(layout.children())
            .map(|((child, state), layout)| {
                child.as_widget_mut().on_event(
                    state,
                    event.clone(),
                    layout,
                    cursor,
                    renderer,
                    clipboard,
                    shell,
                    viewport,
                )
            })
            .fold(event::Status::Ignored, event::Status::merge)
    }
}

fn lerp(a: f64, b: f64, t: f32) -> f64 {
    a + (b - a) * t as f64
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

fn draw_edge_with_label<Renderer>(
    frame: &mut canvas::Frame<Renderer>,
    stroke_width: f32,
    corner_radius: f32,
    endpoint_extension: f32,
    edge: &crate::layout_engine::EdgeLayout,
    points: &[Point],
    from_color: Color,
    to_color: Color,
    label_color: Color,
    alpha: f32,
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
        &rounded_polyline_path(&adjusted, corner_radius),
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

    if let (Some(label), Some(position)) = (&edge.label, label_position) {
        frame.fill_text(canvas::Text {
            content: label.clone(),
            position,
            color: label_color.scale_alpha(alpha),
            size: iced::Pixels(14.0),
            horizontal_alignment: iced::alignment::Horizontal::Center,
            vertical_alignment: iced::alignment::Vertical::Center,
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
        let first_norm = Vector::new(first_vec.x / first_len, first_vec.y / first_len);
        adjusted[0] = Point::new(
            first.x - first_norm.x * extension,
            first.y - first_norm.y * extension,
        );
    }

    let last_index = adjusted.len() - 1;
    let last = adjusted[last_index];
    let penultimate = adjusted[last_index - 1];
    let last_vec = Vector::new(last.x - penultimate.x, last.y - penultimate.y);
    let last_len = (last_vec.x * last_vec.x + last_vec.y * last_vec.y).sqrt();
    if last_len > f32::EPSILON {
        let last_norm = Vector::new(last_vec.x / last_len, last_vec.y / last_len);
        adjusted[last_index] = Point::new(
            last.x + last_norm.x * extension,
            last.y + last_norm.y * extension,
        );
    }

    adjusted
}

fn layout_offset(sugiyama: &GraphLayout, size: iced::Size) -> Vector {
    Vector {
        x: (size.width - sugiyama.max_x as f32).max(0.0) * 0.5,
        y: (size.height - sugiyama.max_y as f32).max(0.0) * 0.5,
    }
}

#[allow(clippy::too_many_arguments)]
fn child_positions(
    sugiyama: &GraphLayout,
    old_sugiyama: Option<&GraphLayout>,
    node_map: &HashMap<u32, usize>,
    edges: &[(u32, u32)],
    size: iced::Size,
    animation: &SharedAnimation,
    motion_easing: &'static Easing,
    motion_duration: Duration,
) -> Vec<Vector> {
    if let Some(old_sugiyama) = old_sugiyama {
        let animation = animation.get();

        match &animation {
            Animation::Pending => {
                let offset = layout_offset(old_sugiyama, size);
                old_sugiyama
                    .coords
                    .values()
                    .map(|(x, y)| Vector {
                        x: *x as f32 + offset.x,
                        y: *y as f32 + offset.y,
                    })
                    .collect()
            }
            Animation::Active { .. } => {
                let progress = animation
                    .progress(motion_easing, motion_duration)
                    .unwrap_or(1.0) as f64;
                let (mut sug, lost, gained) = old_sugiyama.avg(sugiyama, progress);

                let mut gained_sources = HashMap::new();
                let mut lost_sources = HashMap::new();
                for (from, to) in edges {
                    // This edge terminates in a gained node
                    let (Some(to), Some(from)) = (node_map.get(to), node_map.get(from)) else {
                        continue;
                    };
                    if gained.contains(to) {
                        let entry = gained_sources.entry(to).or_insert(from);
                        if from < *entry {
                            *entry = from;
                        }
                    }
                    if lost.contains(to) {
                        let entry = lost_sources.entry(to).or_insert(from);
                        if from < *entry {
                            *entry = from;
                        }
                    }
                }

                // Animate gained nodes as emerging from source
                for (gained_to, from) in gained_sources {
                    let Some((from_x, from_y)) = sug.coords.get(from).copied() else {
                        continue;
                    };
                    let Some((to_x, to_y)) = sug.coords.get_mut(gained_to) else {
                        continue;
                    };
                    *to_x = from_x + (*to_x - from_x) * progress;
                    *to_y = from_y + (*to_y - from_y) * progress;
                }
                // Animate lost nodes as returning to source
                for (gained_to, from) in lost_sources {
                    let Some((from_x, from_y)) = sug.coords.get(from).copied() else {
                        continue;
                    };
                    let Some((to_x, to_y)) = sug.coords.get_mut(gained_to) else {
                        continue;
                    };
                    *to_x = from_x + (*to_x - from_x) * (1. - progress);
                    *to_y = from_y + (*to_y - from_y) * (1. - progress);
                }

                let offset = layout_offset(&sug, size);
                sug.coords
                    .values()
                    .map(|(x, y)| Vector {
                        x: *x as f32 + offset.x,
                        y: *y as f32 + offset.y,
                    })
                    .collect()
            }
            Animation::Complete => {
                let offset = layout_offset(sugiyama, size);
                sugiyama
                    .coords
                    .values()
                    .map(|(x, y)| Vector {
                        x: *x as f32 + offset.x,
                        y: *y as f32 + offset.y,
                    })
                    .collect()
            }
        }
    } else {
        let offset = layout_offset(sugiyama, size);
        sugiyama
            .coords
            .values()
            .map(|(x, y)| Vector {
                x: *x as f32 + offset.x,
                y: *y as f32 + offset.y,
            })
            .collect()
    }
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
