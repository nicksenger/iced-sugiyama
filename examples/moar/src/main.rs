use std::collections::HashSet;

use iced::application::Title;
use iced::widget::{Container, button, text};
use iced_sugiyama::{Cluster, EdgeEndpoint, EdgeEndpointKind, Graph, Sugiyama};

pub fn main() -> iced::Result {
    iced::application(
        Moarificator::default(),
        Moarificator::update,
        Moarificator::view,
    )
    .theme(|_| iced::Theme::Light)
    .window_size((800., 600.))
    .run()
}

impl<S> Title<S> for Moarificator {
    fn title(&self, _state: &S) -> String {
        "iced-sugiyama".to_string()
    }
}

struct Moarificator(Graph);
impl Default for Moarificator {
    fn default() -> Self {
        Self(Graph {
            nodes: vec![0, 1, 2],
            edges: vec![(0, 1), (0, 2)],
            config: Default::default(),
        })
    }
}

#[derive(Debug, Clone, Copy)]
enum Message {
    Moar(u32),
}

impl Moarificator {
    fn update(&mut self, message: Message) {
        match message {
            Message::Moar(n) => match n {
                0 => {
                    let from = fastrand::choice(self.0.nodes.iter()).copied().unwrap_or(0);
                    let to = *self.0.nodes.iter().max().unwrap_or(&0) + 1;
                    self.0.nodes.push(to);
                    self.0.edges.push((from, to));
                }

                n => {
                    let from = n;
                    let existing_connections = self
                        .0
                        .edges
                        .iter()
                        .filter_map(|(n, m)| {
                            (*n == from).then_some(m).or((*m == from).then_some(n))
                        })
                        .collect::<HashSet<_>>();
                    let potential_connections = self
                        .0
                        .nodes
                        .iter()
                        .filter(|n| **n > 0 && **n != from && !existing_connections.contains(n))
                        .copied()
                        .collect::<Vec<_>>();

                    if potential_connections.is_empty() {
                        let to = *self.0.nodes.iter().max().unwrap_or(&0) + 1;
                        self.0.nodes.push(to);
                        self.0.edges.push((from, to));
                    }

                    let Some(to) = fastrand::choice(potential_connections.iter()) else {
                        return;
                    };
                    self.0.edges.push((from, *to));
                }
            },
        }
    }

    fn view(&self) -> Container<'_, Message> {
        let even_cluster_nodes = self
            .0
            .nodes
            .iter()
            .copied()
            .filter(|node| *node != 0 && node % 2 == 0)
            .collect::<Vec<_>>();
        let odd_cluster_nodes = self
            .0
            .nodes
            .iter()
            .copied()
            .filter(|node| *node % 2 == 1)
            .collect::<Vec<_>>();
        let all_cluster_nodes = self
            .0
            .nodes
            .iter()
            .copied()
            .filter(|node| *node != 0)
            .collect::<Vec<_>>();
        let mut clusters = Vec::new();
        let mut parent_cluster_index = None;
        if all_cluster_nodes.len() > 2 {
            parent_cluster_index = Some(clusters.len());
            clusters.push(Cluster::new(all_cluster_nodes).padding(18.0));
        }
        if odd_cluster_nodes.len() > 1 {
            let cluster = Cluster::new(odd_cluster_nodes).padding(12.0);
            clusters.push(match parent_cluster_index {
                Some(parent) => cluster.parent(parent),
                None => cluster,
            });
        }
        if even_cluster_nodes.len() > 1 {
            let cluster = Cluster::new(even_cluster_nodes).padding(12.0);
            clusters.push(match parent_cluster_index {
                Some(parent) => cluster.parent(parent),
                None => cluster,
            });
        }

        Container::new(
            Sugiyama::<Message, iced::Theme, iced::Renderer>::new(&self.0, |n| {
                button(text(if n == 0 {
                    "Moar".to_string()
                } else {
                    n.to_string()
                }))
                .on_press(Message::Moar(n))
                .width(if n == 0 { 100. } else { 10. * (n as f32) })
                .height(if n == 0 { 100. } else { 10. * (n as f32) })
                .into()
            })
            .edge_color(|i| {
                let blue = match i % 9 {
                    0 => iced::Color::from_rgb8(115, 147, 179),
                    1 => iced::Color::from_rgb8(20, 52, 164),
                    2 => iced::Color::from_rgb8(63, 0, 255),
                    3 => iced::Color::from_rgb8(31, 81, 255),
                    4 => iced::Color::from_rgb8(70, 130, 180),
                    5 => iced::Color::from_rgb8(8, 143, 143),
                    6 => iced::Color::from_rgb8(0, 163, 108),
                    7 => iced::Color::from_rgb8(0, 128, 128),
                    _ => iced::Color::from_rgb8(64, 181, 173),
                };
                (blue, blue.scale_alpha(0.5))
            })
            .edge_label(|idx, (from, to)| Some(format!("{idx}: {from}->{to}")))
            .edge_label_element(|_idx, (from, to), _s| Some(text(format!("[{from}->{to}]")).into()))
            .edge_endpoint(|_, _, kind, endpoint| {
                let marker = match kind {
                    EdgeEndpointKind::Source => "o",
                    EdgeEndpointKind::Destination => directional_marker(endpoint),
                };
                Some(iced::widget::text(marker).size(14).into())
            })
            .node_size(|n| {
                let x = if n == 0 { 100. } else { 10. * (n as f64) };
                (x, x)
            })
            .edge_corner_radius(20.0)
            .edge_endpoint_extension(0.0)
            .clusters(clusters)
            .cluster_label(|idx, cluster| {
                let text = match cluster.parent {
                    Some(parent) => format!("cluster {idx} (child of {parent})"),
                    None => format!("cluster {idx}"),
                };
                Some(iced::widget::text(text).size(14).into())
            })
            .cluster_color(|idx| match idx % 2 {
                0 => iced::Color::from_rgba8(255, 112, 67, 0.9),
                _ => iced::Color::from_rgba8(46, 125, 50, 0.9),
            })
            .padding(50),
        )
        .width(800)
        .height(600)
    }
}

fn directional_marker(endpoint: EdgeEndpoint) -> &'static str {
    let angle = endpoint.angle_radians();
    if (-std::f32::consts::FRAC_PI_4..std::f32::consts::FRAC_PI_4).contains(&angle) {
        ">"
    } else if (std::f32::consts::FRAC_PI_4..3.0 * std::f32::consts::FRAC_PI_4).contains(&angle) {
        "v"
    } else if (-3.0 * std::f32::consts::FRAC_PI_4..-std::f32::consts::FRAC_PI_4).contains(&angle) {
        "^"
    } else {
        "<"
    }
}
