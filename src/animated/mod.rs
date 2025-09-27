#![allow(deprecated)]

use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::f32::consts::PI;
use std::hash::Hash;
use std::rc::Rc;
use std::time::Duration;

use iced::advanced::widget::{Operation, Tree, Widget, tree};
use iced::time::Instant;
use iced::widget::canvas::{self, Path};
use iced::widget::{Component, Lazy, Stack};
use iced::window::RedrawRequest;
use iced::{Color, Element, Length, Padding, Point, Size, Vector, event};

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
        self.0.borrow().hash(state);
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
struct GraphLayout {
    max_x: f64,
    max_y: f64,
    coords: BTreeMap<usize, (f64, f64)>,
}

impl GraphLayout {
    fn avg(&self, other: &Self, weight_other: f64) -> (Self, HashSet<usize>, HashSet<usize>) {
        let own_keys = self.coords.keys().collect::<HashSet<_>>();
        let other_keys = other.coords.keys().collect::<HashSet<_>>();

        let lost = own_keys
            .difference(&other_keys)
            .copied()
            .copied()
            .collect::<HashSet<usize>>();
        let gained = other_keys
            .difference(&own_keys)
            .copied()
            .copied()
            .collect::<HashSet<usize>>();

        let coords = own_keys
            .intersection(&other_keys)
            .filter_map(|k| {
                let a = self.coords.get(k)?;
                let b = other.coords.get(k)?;

                Some((
                    **k,
                    (
                        (b.0 - a.0) * weight_other + a.0,
                        (b.1 - a.1) * weight_other + a.1,
                    ),
                ))
            })
            .chain(
                lost.iter()
                    .filter_map(|k| self.coords.get(k).copied().map(|pos| (*k, pos))),
            )
            .chain(
                gained
                    .iter()
                    .filter_map(|k| other.coords.get(k).copied().map(|pos| (*k, pos))),
            )
            .collect();

        (
            Self {
                max_x: self.max_x + (other.max_x - self.max_x) * weight_other,
                max_y: self.max_y + (other.max_y - self.max_y) * weight_other,
                coords,
            },
            lost,
            gained,
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
    edge_color: fn(usize) -> (Color, Color),
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
            edge_color: |_| (Color::BLACK, Color::BLACK.scale_alpha(0.5)),
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
                    if g.nodes.len() == 1 {
                        GraphLayout {
                            coords: [(0usize, (1., 1.))].into_iter().collect(),
                            max_x: 0.,
                            max_y: 1.,
                        }
                    } else {
                        g.sugiyama()
                    }
                });

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
                    old_sugiyama,
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
                            sugiyama: graph.sugiyama(),
                            old_sugiyama: old.as_ref().map(|g| g.sugiyama()),
                            old_edges: old.as_ref().map(|g| g.edges.clone()),
                            node_map,
                            edges: self.graph.edges.clone(),
                            padding: self.padding,
                            stroke_width: self.stroke_width,
                            edge_color: self.edge_color,
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
    old_edges: Option<Vec<(u32, u32)>>,
    node_map: HashMap<u32, usize>,
    edges: Vec<(u32, u32)>,
    padding: iced::Padding,
    stroke_width: f32,
    edge_color: fn(usize) -> (Color, Color),
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

            if let (Some(old_sugiyama), Some(old_edges)) =
                (self.old_sugiyama.as_ref(), self.old_edges.as_ref())
            {
                let animation = self.animation.get();

                let (mut lost, mut gained) = (HashSet::<usize>::new(), HashSet::<usize>::new());
                let sugiyama = match animation {
                    Animation::Pending => old_sugiyama.clone(),
                    Animation::Active { .. } => {
                        let (sug, l, g) = old_sugiyama.avg(
                            &self.sugiyama,
                            animation
                                .progress(self.motion_easing, self.motion_duration)
                                .unwrap() as f64,
                        );
                        lost.extend(l);
                        gained.extend(g);
                        sug
                    }
                    Animation::Complete => self.sugiyama.clone(),
                };

                let progress = animation
                    .progress(self.motion_easing, self.motion_duration)
                    .unwrap_or(if matches!(animation, Animation::Pending) {
                        0.
                    } else {
                        1.
                    });
                for (idx, (from, to)) in self.edges.iter().enumerate() {
                    let is_new = !old_edges.iter().any(|x| (&x.0, &x.1) == (from, to));

                    let Some(((from_x, from_y), to_lost)) = self
                        .node_map
                        .get(from)
                        .and_then(|i| sugiyama.coords.get(i).map(|c| (c, lost.contains(i))))
                    else {
                        continue;
                    };
                    let Some(((to_x, to_y), to_gained)) = self
                        .node_map
                        .get(to)
                        .and_then(|i| sugiyama.coords.get(i).map(|c| (c, gained.contains(i))))
                    else {
                        continue;
                    };

                    let (max_x, max_y) = (sugiyama.max_x, sugiyama.max_y);
                    let a = Point::new(
                        if max_x == 0. {
                            0.5
                        } else {
                            *from_x as f32 / max_x as f32
                        } * size.width
                            + self.padding.left,
                        (*from_y as f32 / max_y as f32) * size.height + self.padding.top,
                    );
                    let mut b = Point::new(
                        if max_x == 0. {
                            0.5
                        } else {
                            *to_x as f32 / max_x as f32
                        } * size.width
                            + self.padding.left,
                        (*to_y as f32 / max_y as f32) * size.height + self.padding.top,
                    );

                    if to_gained {
                        b = a + (b - a) * progress;
                    }
                    if to_lost {
                        b = a + (b - a) * (1. - progress);
                    }

                    let (from_color, to_color) = (self.edge_color)(idx);
                    if is_new {
                        frame.stroke(
                            &Path::new(|p| {
                                p.move_to(a);
                                p.bezier_curve_to(
                                    Point::new(
                                        a.x + (b.x - a.x) / 1.618,
                                        b.y + (a.y - b.y) / 1.618,
                                    ),
                                    Point::new(
                                        b.x + (a.x - b.x) / 1.618,
                                        a.y + (b.y - a.y) / 1.618,
                                    ),
                                    b,
                                );
                            }),
                            stroke_gradient(
                                (a, from_color.scale_alpha(progress)),
                                (b, to_color.scale_alpha(progress)),
                            ),
                        );
                    } else {
                        let (from_color, to_color) = (self.edge_color)(idx);
                        frame.stroke(
                            &Path::new(|p| {
                                p.move_to(a);
                                p.bezier_curve_to(
                                    Point::new(a.x + (b.x - a.x) / PI, b.y + (a.y - b.y) / PI),
                                    Point::new(b.x + (a.x - b.x) / PI, a.y + (b.y - a.y) / PI),
                                    b,
                                );
                            }),
                            stroke_gradient((a, from_color), (b, to_color)),
                        );
                    }
                }
            } else {
                let (max_x, max_y) = (self.sugiyama.max_x, self.sugiyama.max_y);
                for (idx, (from, to)) in self.edges.iter().enumerate() {
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

                    let (from_color, to_color) = (self.edge_color)(idx);
                    frame.stroke(
                        &Path::new(|p| {
                            p.move_to(a);
                            p.bezier_curve_to(
                                Point::new(a.x + (b.x - a.x) / PI, b.y + (a.y - b.y) / PI),
                                Point::new(b.x + (a.x - b.x) / PI, a.y + (b.y - a.y) / PI),
                                b,
                            );
                        }),
                        stroke_gradient((a, from_color), (b, to_color)),
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
            Animation::Pending => old_sugiyama
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
                .collect(),
            Animation::Active { .. } => {
                let progress = animation.progress(motion_easing, motion_duration).unwrap() as f64;
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

                sug.coords
                    .values()
                    .map(|(x, y)| Vector {
                        x: if sug.max_x == 0. {
                            0.5
                        } else {
                            *x as f32 / sug.max_x as f32
                        } * size.width,
                        y: (*y as f32 / sug.max_y as f32) * size.height,
                    })
                    .collect()
            }
            Animation::Complete => sugiyama
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
                .collect(),
        }
    } else {
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
