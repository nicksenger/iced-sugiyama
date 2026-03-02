#![allow(deprecated)]
// If anyone wants to help refactor this not to use Component,
// that would be great!

use std::borrow::Cow;
use std::hash::Hash;

use iced::advanced::widget::{Operation, Tree, Widget};
use iced::widget::canvas::{self, Path};
use iced::widget::{Component, Lazy, Stack};
use iced::{Color, Element, Length, Padding, Point, Size, Vector, event};

pub use crate::layout_engine::Cluster;
use crate::layout_engine::{GraphLayout, compute_layout};

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
    edge_color: fn(usize) -> (Color, Color),
    edge_label: fn(usize, (u32, u32)) -> Option<String>,
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
            stroke_width: 4.0,
            edge_color: |_| (Color::BLACK, Color::BLACK.scale_alpha(0.5)),
            edge_label: |_, _| None,
            node_size: |_| (80.0, 40.0),
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

    pub fn stroke_width(mut self, width: f32) -> Self {
        self.stroke_width = width;
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
}

impl<Message, Theme, Renderer> Component<Message, Theme, Renderer>
    for Sugiyama<'_, Message, Theme, Renderer>
where
    Renderer: iced::advanced::Renderer + iced::advanced::graphics::geometry::Renderer + 'static,
    Theme: 'static,
    Message: Clone + 'static,
{
    type Event = Event<Message>;
    type State = ();

    fn view(&self, _state: &Self::State) -> Element<'_, Self::Event, Theme, Renderer> {
        Lazy::new(&self.graph, |graph| {
            let children = self
                .graph
                .nodes
                .iter()
                .map(|n| (self.view_node)(*n).map(Event::Passthrough))
                .collect();

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
                padding: self.padding,
            };

            Stack::with_children(vec![
                iced::widget::canvas(GraphCanvas::<Renderer> {
                    cache: Default::default(),
                    sugiyama,
                    padding: self.padding,
                    stroke_width: self.stroke_width,
                    edge_color: self.edge_color,
                    cluster_color: self.cluster_color,
                    label_color: self.label_color,
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
    edge_color: fn(usize) -> (Color, Color),
    cluster_color: fn(usize) -> Color,
    label_color: fn(usize) -> Color,
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
            let stroke_gradient = |(ap, ac), (bp, bc)| canvas::Stroke {
                width: self.stroke_width,
                style: canvas::stroke::Style::Gradient(canvas::Gradient::Linear(
                    canvas::gradient::Linear::new(ap, bp)
                        .add_stop(0.0, bc)
                        .add_stop(1.0, ac),
                )),
                line_cap: canvas::LineCap::Round,
                ..canvas::Stroke::default()
            };

            let (max_x, max_y) = (self.sugiyama.max_x, self.sugiyama.max_y);
            let project = |x: f64, y: f64| {
                Point::new(
                    if max_x <= 0.0 {
                        0.5 * size.width
                    } else {
                        (x as f32 / max_x as f32) * size.width
                    } + self.padding.left,
                    if max_y <= 0.0 {
                        0.5 * size.height
                    } else {
                        (y as f32 / max_y as f32) * size.height
                    } + self.padding.top,
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

                let mut projected = edge
                    .points
                    .iter()
                    .map(|(x, y)| project(*x, *y))
                    .collect::<Vec<_>>();
                let start = match projected.first().copied() {
                    Some(point) => point,
                    None => continue,
                };
                let end = match projected.last().copied() {
                    Some(point) => point,
                    None => continue,
                };

                let (from_color, to_color) = (self.edge_color)(edge.index);
                frame.stroke(
                    &Path::new(|path| {
                        path.move_to(start);
                        for point in projected.drain(1..) {
                            path.line_to(point);
                        }
                    }),
                    stroke_gradient((start, from_color), (end, to_color)),
                );

                if let (Some(label), Some((label_x, label_y))) = (&edge.label, edge.label_position)
                {
                    frame.fill_text(canvas::Text {
                        content: label.clone(),
                        position: project(label_x, label_y),
                        color: (self.label_color)(edge.index),
                        size: iced::Pixels(14.0),
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
    sugiyama: GraphLayout,
    padding: Padding,
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

        let layouts = self
            .children
            .iter()
            .zip(&mut tree.children)
            .map(|(child, tree)| child.as_widget().layout(tree, renderer, &limits))
            .collect::<Vec<_>>();

        let child_positions = child_positions(&self.sugiyama, size);
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

fn child_positions(sugiyama: &GraphLayout, size: iced::Size) -> Vec<Vector> {
    sugiyama
        .coords
        .values()
        .map(|(x, y)| Vector {
            x: if sugiyama.max_x == 0. {
                0.5
            } else {
                *x as f32 / sugiyama.max_x as f32
            } * size.width,
            y: (*y as f32 / sugiyama.max_y as f32) * size.height,
        })
        .collect()
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
