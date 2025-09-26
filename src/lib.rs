#![allow(deprecated)]
// If anyone wants to help refactor this not to use Component,
// that would be great!

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::hash::Hash;

use iced::advanced::widget::{Operation, Tree, Widget};
use iced::widget::canvas::{self, Path};
use iced::widget::{Component, Lazy, Stack};
use iced::{Color, Element, Length, Padding, Point, Size, Vector, event};

struct GraphLayout {
    max_x: f64,
    max_y: f64,
    coords: BTreeMap<usize, (f64, f64)>,
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

    fn sugiyama(&self) -> GraphLayout {
        let (mut max_x, mut max_y) = (0f64, 1f64);
        let mut coords = BTreeMap::new();
        for (layers, _, _) in
            rust_sugiyama::from_edges(&self.edges, &rust_sugiyama::configure::Config::default())
        {
            for (i, (x, y)) in layers {
                max_x = max_x.max(x);
                max_y = max_y.max(y);
                coords.insert(i, (x, y));
            }
        }

        GraphLayout {
            max_x,
            max_y,
            coords,
        }
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
    edge_color: (Color, Color),
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
            edge_color: (Color::BLACK, Color::BLACK.scale_alpha(0.5)),
            padding: iced::Padding::ZERO,
        }
    }

    pub fn stroke_width(mut self, width: f32) -> Self {
        self.stroke_width = width;
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

            let node_map = self
                .graph
                .nodes
                .iter()
                .enumerate()
                .map(|(i, n)| (*n, i))
                .collect::<HashMap<_, _>>();

            let sugiyama = if graph.nodes.len() == 1 {
                GraphLayout {
                    coords: [(0usize, (1., 1.))].into_iter().collect(),
                    max_x: 0.,
                    max_y: 1.,
                }
            } else {
                graph.sugiyama()
            };
            let overlay = GraphNodes::<Event<Message>, Theme, Renderer> {
                children,
                sugiyama,
                padding: self.padding,
            };

            Stack::with_children(vec![
                iced::widget::canvas(GraphCanvas::<Renderer> {
                    cache: Default::default(),
                    sugiyama: graph.sugiyama(),
                    node_map,
                    edges: self.graph.edges.clone(),
                    padding: self.padding,
                    stroke_width: self.stroke_width,
                    edge_color: self.edge_color,
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
    node_map: HashMap<u32, usize>,
    edges: Vec<(u32, u32)>,
    padding: iced::Padding,
    stroke_width: f32,
    edge_color: (Color, Color),
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
            for (from, to) in &self.edges {
                let Some((from_x, from_y)) = self
                    .node_map
                    .get(from)
                    .and_then(|i| self.sugiyama.coords.get(i))
                else {
                    continue;
                };
                let Some((to_x, to_y)) = self
                    .node_map
                    .get(to)
                    .and_then(|i| self.sugiyama.coords.get(i))
                else {
                    continue;
                };

                let a = Point::new(
                    if max_x == 0. {
                        0.5
                    } else {
                        *from_x as f32 / max_x as f32
                    } * size.width
                        + self.padding.left,
                    (*from_y as f32 / max_y as f32) * size.height + self.padding.top,
                );
                let b = Point::new(
                    if max_x == 0. {
                        0.5
                    } else {
                        *to_x as f32 / max_x as f32
                    } * size.width
                        + self.padding.left,
                    (*to_y as f32 / max_y as f32) * size.height + self.padding.top,
                );

                frame.stroke(
                    &Path::line(a, b),
                    stroke_gradient((a, self.edge_color.0), (b, self.edge_color.1)),
                );
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
