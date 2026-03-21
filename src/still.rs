#![allow(deprecated)]
// If anyone wants to help refactor this not to use Component,
// that would be great!

use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::rc::Rc;

use iced::advanced::widget::{Operation, Tree, Widget};
use iced::widget::canvas::{self, Path};
use iced::widget::{Component, Lazy, Stack};
use iced::{Color, Element, Length, Padding, Point, Size, Vector, event};

pub use crate::layout_engine::{Cluster, EdgeEndpoint, EdgeEndpointKind};
use crate::layout_engine::{GraphLayout, compute_layout, layout_signature};

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
}

/// An iced widget which draws a layered graph of elements
pub struct Sugiyama<'a, Message, Theme, Renderer> {
    graph: Cow<'a, Graph>,
    view_node: Box<dyn Fn(u32) -> Element<'static, Message, Theme, Renderer> + 'a>,
    cluster_container:
        Box<dyn Fn(usize, &Cluster) -> Option<Element<'static, Message, Theme, Renderer>> + 'a>,
    stroke_width: f32,
    edge_corner_radius: f32,
    edge_endpoint_extension: f32,
    edge_color: fn(usize) -> (Color, Color),
    outgoing_edge_style: Box<dyn Fn(u32) -> OutgoingEdgeStyle + 'a>,
    edge_label: fn(usize, (u32, u32)) -> Option<String>,
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
    node_size: fn(u32) -> (f64, f64),
    clusters: Vec<Cluster>,
    render_config: rust_sugiyama::advanced::RenderConfig,
    cluster_color: fn(usize) -> Color,
    label_color: fn(usize) -> Color,
    padding: Padding,
}

impl<'a, Message, Theme, Renderer> Sugiyama<'a, Message, Theme, Renderer> {
    pub fn new(
        graph: impl Into<Cow<'a, Graph>>,
        view_node: impl Fn(u32) -> Element<'static, Message, Theme, Renderer> + 'a,
    ) -> Self {
        Self {
            graph: graph.into(),
            view_node: Box::new(view_node),
            cluster_container: Box::new(|_, _| None),
            stroke_width: 4.0,
            edge_corner_radius: 10.0,
            edge_endpoint_extension: 0.0,
            edge_color: |_| (Color::BLACK, Color::BLACK.scale_alpha(0.5)),
            outgoing_edge_style: Box::new(|_| OutgoingEdgeStyle::default()),
            edge_label: |_, _| None,
            edge_label_element: Box::new(|_, _, _| None),
            edge_endpoint: Box::new(|_, _, _, _| None),
            node_size: |_| (56.0, 32.0),
            clusters: Vec::new(),
            render_config: Default::default(),
            cluster_color: |_| Color::from_rgba8(90, 90, 90, 0.6),
            label_color: |_| Color::BLACK,
            padding: iced::Padding::ZERO,
        }
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

    pub fn edge_label(mut self, f: fn(usize, (u32, u32)) -> Option<String>) -> Self {
        self.edge_label = f;
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

    pub fn node_size(mut self, f: fn(u32) -> (f64, f64)) -> Self {
        self.node_size = f;
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
}

impl<Message, Theme, Renderer> Component<Message, Theme, Renderer>
    for Sugiyama<'_, Message, Theme, Renderer>
where
    Renderer: iced::advanced::Renderer + iced::advanced::graphics::geometry::Renderer + 'static,
    Theme: 'static,
    Message: Clone + 'static,
{
    type Event = Event<Message>;
    type State = Rc<RefCell<LayoutMemo>>;

    fn view(&self, state: &Self::State) -> Element<'_, Self::Event, Theme, Renderer> {
        let layout_memo = state.clone();
        Lazy::new(&self.graph, move |graph| {
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
                .map(|(index, node)| (*node, index))
                .collect::<HashMap<_, _>>();

            let signature = layout_signature(&graph.nodes, &graph.edges, &self.clusters);
            let sugiyama = {
                let mut memo = layout_memo.borrow_mut();
                memo.layout_for(signature, || {
                    compute_layout(
                        &graph.nodes,
                        &graph.edges,
                        &graph.config,
                        self.node_size,
                        self.edge_label,
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
            let mut cluster_container_children = Vec::new();
            for cluster in &sugiyama.clusters {
                let Some(cluster_spec) = self.clusters.get(cluster.index) else {
                    continue;
                };
                let Some(container) = (self.cluster_container)(cluster.index, cluster_spec) else {
                    continue;
                };
                cluster_container_children.push(ClusterContainerChild {
                    cluster_index: cluster.index,
                });
                children.push(container.map(Event::Passthrough));
            }
            let mut edge_label_children = Vec::new();
            let mut edge_label_overlay_edges = HashSet::new();
            for edge in &sugiyama.edges {
                let Some(edge_data) = graph.edges.get(edge.index).copied() else {
                    continue;
                };
                let style = edge_style_by_index
                    .get(&edge.index)
                    .copied()
                    .unwrap_or_default();
                if !style.visible || style.alpha <= f32::EPSILON {
                    continue;
                }
                let Some(label_element) =
                    (self.edge_label_element)(edge.index, edge_data, edge.label.as_deref())
                else {
                    continue;
                };
                edge_label_overlay_edges.insert(edge.index);
                edge_label_children.push(EdgeLabelChild {
                    edge_index: edge.index,
                });
                children.push(label_element.map(Event::Passthrough));
            }
            let mut edge_endpoint_children = Vec::new();
            for edge in &sugiyama.edges {
                let Some(edge_data) = graph.edges.get(edge.index).copied() else {
                    continue;
                };
                let style = edge_style_by_index
                    .get(&edge.index)
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
                    let Some((cx, cy)) = sugiyama.coords.get(&node_index).copied() else {
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
                    let Some(widget) = (self.edge_endpoint)(edge.index, edge_data, kind, endpoint)
                    else {
                        continue;
                    };
                    edge_endpoint_children.push(EdgeEndpointChild {
                        edge_index: edge.index,
                        edge: edge_data,
                        kind,
                    });
                    children.push(widget.map(Event::Passthrough));
                }
            }
            let overlay = GraphNodes::<Event<Message>, Theme, Renderer> {
                children,
                node_map,
                cluster_container_children,
                edge_label_children,
                edge_endpoint_children,
                sugiyama: sugiyama.clone(),
                padding: self.padding,
                edge_endpoint_extension: self.edge_endpoint_extension,
            };

            Stack::with_children(vec![
                iced::widget::canvas(GraphCanvas::<Renderer> {
                    cache: Default::default(),
                    sugiyama,
                    padding: self.padding,
                    stroke_width: self.stroke_width,
                    edge_corner_radius: self.edge_corner_radius,
                    edge_endpoint_extension: self.edge_endpoint_extension,
                    edge_color: self.edge_color,
                    edge_style_by_index,
                    cluster_color: self.cluster_color,
                    label_color: self.label_color,
                    edge_label_overlay_edges,
                })
                .width(Length::Fill)
                .height(Length::Fill)
                .into(),
                overlay.into(),
            ])
        })
        .into()
    }

    fn update(&mut self, _state: &mut Self::State, event: Self::Event) -> Option<Message> {
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
    padding: iced::Padding,
    stroke_width: f32,
    edge_corner_radius: f32,
    edge_endpoint_extension: f32,
    edge_color: fn(usize) -> (Color, Color),
    edge_style_by_index: HashMap<usize, OutgoingEdgeStyle>,
    cluster_color: fn(usize) -> Color,
    label_color: fn(usize) -> Color,
    edge_label_overlay_edges: HashSet<usize>,
}

// Draw the edges (these will be in a layer under the nodes)
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
        let mut size = bounds.size();
        size.width = size.width - self.padding.left - self.padding.right;
        size.height = size.height - self.padding.top - self.padding.bottom;

        let graph = self.cache.draw(renderer, bounds.size(), |frame| {
            let offset = layout_offset(&self.sugiyama, size);
            let project = |x: f64, y: f64| {
                Point::new(
                    x as f32 + offset.x + self.padding.left,
                    y as f32 + offset.y + self.padding.top,
                )
            };

            for cluster in &self.sugiyama.clusters {
                let a = project(cluster.min_x, cluster.min_y);
                let b = project(cluster.max_x, cluster.max_y);
                let top_left = Point::new(a.x.min(b.x), a.y.min(b.y));
                let rect_size = Size::new((a.x - b.x).abs().max(1.0), (a.y - b.y).abs().max(1.0));
                frame.stroke(
                    &Path::rectangle(top_left, rect_size),
                    canvas::Stroke {
                        width: self.stroke_width.max(1.0) * 0.5,
                        style: canvas::stroke::Style::Solid((self.cluster_color)(cluster.index)),
                        ..canvas::Stroke::default()
                    },
                );
            }

            for edge in &self.sugiyama.edges {
                if edge.points.len() < 2 {
                    continue;
                }

                let style = self
                    .edge_style_by_index
                    .get(&edge.index)
                    .copied()
                    .unwrap_or_default();
                if !style.visible || style.alpha <= f32::EPSILON {
                    continue;
                }
                let width_scale = style.width_scale.max(0.0);
                if width_scale <= f32::EPSILON {
                    continue;
                }

                let projected = edge
                    .points
                    .iter()
                    .map(|(x, y)| project(*x, *y))
                    .collect::<Vec<_>>();
                let projected_curve = edge
                    .curve_points
                    .iter()
                    .map(|(x, y)| project(*x, *y))
                    .collect::<Vec<_>>();
                let adjusted = extend_polyline_endpoints(
                    &projected,
                    self.edge_endpoint_extension * width_scale,
                );
                let start = match adjusted.first().copied() {
                    Some(point) => point,
                    None => continue,
                };
                let end = match adjusted.last().copied() {
                    Some(point) => point,
                    None => continue,
                };

                let (mut from_color, mut to_color) = (self.edge_color)(edge.index);
                if let Some((from_override, to_override)) = style.color_override {
                    from_color = from_override;
                    to_color = to_override;
                }
                frame.stroke(
                    &edge_path(
                        &adjusted,
                        &projected_curve,
                        self.edge_corner_radius * width_scale,
                        self.edge_endpoint_extension * width_scale,
                    ),
                    canvas::Stroke {
                        width: self.stroke_width * width_scale,
                        style: canvas::stroke::Style::Gradient(canvas::Gradient::Linear(
                            canvas::gradient::Linear::new(start, end)
                                .add_stop(0.0, to_color.scale_alpha(style.alpha))
                                .add_stop(1.0, from_color.scale_alpha(style.alpha)),
                        )),
                        line_cap: canvas::LineCap::Round,
                        ..canvas::Stroke::default()
                    },
                );

                if self.edge_label_overlay_edges.contains(&edge.index) {
                    continue;
                }
                if let (Some(label), Some((label_x, label_y))) = (&edge.label, edge.label_position)
                {
                    frame.fill_text(canvas::Text {
                        content: label.clone(),
                        position: project(label_x, label_y),
                        color: (self.label_color)(edge.index).scale_alpha(style.alpha),
                        size: iced::Pixels((14.0 * width_scale.max(0.25)).max(1.0)),
                        horizontal_alignment: iced::alignment::Horizontal::Center,
                        vertical_alignment: iced::alignment::Vertical::Center,
                        ..canvas::Text::default()
                    });
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
    node_map: HashMap<u32, usize>,
    cluster_container_children: Vec<ClusterContainerChild>,
    edge_label_children: Vec<EdgeLabelChild>,
    edge_endpoint_children: Vec<EdgeEndpointChild>,
    sugiyama: GraphLayout,
    padding: Padding,
    edge_endpoint_extension: f32,
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
        let limits = limits.shrink(Size {
            width: self.padding.left + self.padding.right,
            height: self.padding.top + self.padding.bottom,
        });
        let size = limits.max();

        let node_positions = child_positions(&self.sugiyama, size);
        let node_count = node_positions.len();
        let cluster_container_layouts =
            cluster_container_layouts(&self.sugiyama, size, &self.cluster_container_children);
        let cluster_container_count = cluster_container_layouts.len();
        let layouts = self
            .children
            .iter()
            .zip(&mut tree.children)
            .enumerate()
            .map(|(index, (child, tree))| {
                if (node_count..(node_count + cluster_container_count)).contains(&index) {
                    let cluster_layout = cluster_container_layouts[index - node_count];
                    let child_limits = iced::advanced::layout::Limits::new(
                        cluster_layout.size,
                        cluster_layout.size,
                    );
                    child.as_widget().layout(tree, renderer, &child_limits)
                } else {
                    child.as_widget().layout(tree, renderer, &limits)
                }
            })
            .collect::<Vec<_>>();

        let node_sizes = layouts
            .iter()
            .take(node_count)
            .map(|layout| layout.bounds().size())
            .collect::<Vec<_>>();
        let edge_label_positions =
            edge_label_positions(&self.sugiyama, size, &self.edge_label_children);
        let edge_endpoint_positions = edge_endpoint_positions(
            &self.sugiyama,
            &self.node_map,
            &node_positions,
            &node_sizes,
            size,
            &self.edge_endpoint_children,
            self.edge_endpoint_extension,
        );
        let mut translated_children = Vec::with_capacity(layouts.len());
        let mut layout_iter = layouts.into_iter();

        for vector in node_positions {
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

    fn overlay<'b>(
        &'b mut self,
        tree: &'b mut Tree,
        layout: iced::advanced::Layout<'_>,
        renderer: &Renderer,
        translation: Vector,
    ) -> Option<iced::advanced::overlay::Element<'b, Message, Theme, Renderer>> {
        iced::advanced::overlay::from_children(
            &mut self.children,
            tree,
            layout,
            renderer,
            translation,
        )
    }
}

fn child_positions(sugiyama: &GraphLayout, size: iced::Size) -> Vec<Vector> {
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

#[derive(Debug, Clone, Copy)]
struct ClusterContainerLayout {
    center: Vector,
    size: Size,
}

fn cluster_container_layouts(
    sugiyama: &GraphLayout,
    size: iced::Size,
    cluster_container_children: &[ClusterContainerChild],
) -> Vec<ClusterContainerLayout> {
    let offset = layout_offset(sugiyama, size);
    cluster_container_children
        .iter()
        .map(|child| {
            let cluster = sugiyama
                .clusters
                .iter()
                .find(|cluster| cluster.index == child.cluster_index);
            match cluster {
                Some(cluster) => {
                    let width = (cluster.max_x - cluster.min_x).max(1.0) as f32;
                    let height = (cluster.max_y - cluster.min_y).max(1.0) as f32;
                    ClusterContainerLayout {
                        center: Vector::new(
                            cluster.min_x as f32 + width * 0.5 + offset.x,
                            cluster.min_y as f32 + height * 0.5 + offset.y,
                        ),
                        size: Size::new(width, height),
                    }
                }
                None => ClusterContainerLayout {
                    center: Vector::new(offset.x, offset.y),
                    size: Size::new(1.0, 1.0),
                },
            }
        })
        .collect()
}

fn edge_label_positions(
    sugiyama: &GraphLayout,
    size: iced::Size,
    edge_label_children: &[EdgeLabelChild],
) -> Vec<Vector> {
    if edge_label_children.is_empty() {
        return Vec::new();
    }

    let offset = layout_offset(sugiyama, size);
    let edges_by_index = sugiyama
        .edges
        .iter()
        .map(|edge| (edge.index, edge))
        .collect::<HashMap<_, _>>();

    edge_label_children
        .iter()
        .map(|child| {
            edges_by_index
                .get(&child.edge_index)
                .and_then(|edge| edge_label_anchor(edge, offset))
                .unwrap_or(Vector::new(offset.x, offset.y))
        })
        .collect()
}

fn edge_label_anchor(edge: &crate::layout_engine::EdgeLayout, offset: Vector) -> Option<Vector> {
    if let Some((x, y)) = edge.label_position {
        return Some(Vector::new(x as f32 + offset.x, y as f32 + offset.y));
    }

    polyline_midpoint(&edge.points)
        .map(|(x, y)| Vector::new(x as f32 + offset.x, y as f32 + offset.y))
}

fn polyline_midpoint(points: &[(f64, f64)]) -> Option<(f64, f64)> {
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

fn edge_endpoint_positions(
    sugiyama: &GraphLayout,
    node_map: &HashMap<u32, usize>,
    node_positions: &[Vector],
    node_sizes: &[Size],
    size: iced::Size,
    endpoint_children: &[EdgeEndpointChild],
    endpoint_extension: f32,
) -> Vec<Vector> {
    if endpoint_children.is_empty() {
        return Vec::new();
    }

    let offset = layout_offset(sugiyama, size);
    let edges_by_index = sugiyama
        .edges
        .iter()
        .map(|edge| (edge.index, edge))
        .collect::<HashMap<_, _>>();

    endpoint_children
        .iter()
        .map(|child| {
            match (
                edges_by_index.get(&child.edge_index).copied(),
                node_center_and_size(node_map, node_positions, node_sizes, child),
            ) {
                (Some(edge), Some((center, node_size))) => {
                    let projected = edge
                        .points
                        .iter()
                        .map(|(x, y)| Point::new(*x as f32 + offset.x, *y as f32 + offset.y))
                        .collect::<Vec<_>>();
                    let projected_curve = edge
                        .curve_points
                        .iter()
                        .map(|(x, y)| Point::new(*x as f32 + offset.x, *y as f32 + offset.y))
                        .collect::<Vec<_>>();
                    endpoint_anchor_and_direction(
                        &projected,
                        &projected_curve,
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
                (Some(edge), None) => {
                    let projected = edge
                        .points
                        .iter()
                        .map(|(x, y)| Point::new(*x as f32 + offset.x, *y as f32 + offset.y))
                        .collect::<Vec<_>>();
                    fallback_endpoint_anchor_from_points(&projected, child.kind, endpoint_extension)
                        .unwrap_or(Vector::new(offset.x, offset.y))
                }
                (None, Some((center, _))) => center,
                (None, None) => Vector::new(offset.x, offset.y),
            }
        })
        .collect()
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

fn edge_endpoint_metadata(
    edge: &crate::layout_engine::EdgeLayout,
    kind: EdgeEndpointKind,
    node_center: Vector,
    node_size: Size,
    endpoint_extension: f32,
) -> Option<EdgeEndpoint> {
    let points = edge
        .points
        .iter()
        .map(|(x, y)| Point::new(*x as f32, *y as f32))
        .collect::<Vec<_>>();
    let curve_points = edge
        .curve_points
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

fn endpoint_center_direction(anchor: Vector, node_center: Vector, kind: EdgeEndpointKind) -> Option<Vector> {
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

    let half_width = node_size.width * 0.5;
    let half_height = node_size.height * 0.5;
    let min_x = node_center.x - half_width;
    let max_x = node_center.x + half_width;
    let min_y = node_center.y - half_height;
    let max_y = node_center.y + half_height;

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

fn layout_offset(sugiyama: &GraphLayout, size: iced::Size) -> Vector {
    Vector {
        x: ((size.width - sugiyama.max_x as f32).max(0.0)) * 0.5,
        y: ((size.height - sugiyama.max_y as f32).max(0.0)) * 0.5,
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
