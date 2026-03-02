use std::collections::HashSet;

use iced::application::Title;
use iced::widget::{Container, button, text};
use iced_sugiyama::{Cluster, Graph, Sugiyama};

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
        let mut clusters = Vec::new();
        if even_cluster_nodes.len() > 1 {
            clusters.push(Cluster::new(even_cluster_nodes).padding(14.0));
        }
        if odd_cluster_nodes.len() > 1 {
            clusters.push(Cluster::new(odd_cluster_nodes).padding(14.0));
        }

        Container::new(
            Sugiyama::<Message, iced::Theme, iced::Renderer>::new(&self.0, |n| {
                button(text(if n == 0 {
                    "Moar".to_string()
                } else {
                    n.to_string()
                }))
                .on_press(Message::Moar(n))
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
            .clusters(clusters)
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
