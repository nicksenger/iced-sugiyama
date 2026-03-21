use std::collections::HashSet;
use std::env;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Duration;

use iced::alignment::{Horizontal, Vertical};
use iced::application::Title;
use iced::widget::{Column, Container, button, container, text};
use iced::window;
use iced::window::Screenshot;
use iced::{Alignment, Background, Color, Element, Font, Length, Task, Theme, border};
use iced_sugiyama::{Cluster, EdgeEndpoint, EdgeEndpointKind, Graph, Sugiyama};
use thiserror::Error;

const GRAPH_FONT: Font = Font::with_name("Times New Roman");
const NODE_BORDER_RADIUS: f32 = 16.0;
const CLUSTER_BORDER_RADIUS: f32 = 18.0;
const MIN_NODE_SIDE: f64 = 72.0;

const WINDOW_WIDTH: f32 = 2600.0;
const WINDOW_HEIGHT: f32 = 1600.0;

pub fn main() -> iced::Result {
    let options = AppOptions::from_env();

    iced::application(
        |state: &Moarificator| state.title(&()),
        Moarificator::update,
        Moarificator::view,
    )
    .theme(|_| iced::Theme::Light)
    .window(window::Settings {
        size: iced::Size::new(WINDOW_WIDTH, WINDOW_HEIGHT),
        ..Default::default()
    })
    .run_with(move || {
        let task = if options.headless {
            Task::done(Message::RenderMergedPng)
        } else {
            Task::none()
        };

        (Moarificator::new(options.headless), task)
    })
}

impl<S> Title<S> for Moarificator {
    fn title(&self, _state: &S) -> String {
        "iced-sugiyama".to_string()
    }
}

struct Moarificator {
    headless: bool,
    export_in_progress: bool,
    graph: Graph,
    pending_graphviz_png: Option<Vec<u8>>,
    pending_iced_screenshot: Option<Screenshot>,
}
impl Moarificator {
    fn new(headless: bool) -> Self {
        Self {
            headless,
            export_in_progress: false,
            graph: initial_graph(),
            pending_graphviz_png: None,
            pending_iced_screenshot: None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct AppOptions {
    headless: bool,
}

impl AppOptions {
    fn from_env() -> Self {
        Self {
            headless: env::args().skip(1).any(|arg| arg == "--headless"),
        }
    }
}

#[derive(Debug, Clone)]
enum Message {
    Moar(u32),
    RenderMergedPng,
    CaptureIcedView,
    GraphvizPngReady(Result<Vec<u8>, String>),
    IcedViewCaptured(Screenshot),
    MergedPngSaved(Result<PathBuf, String>),
}

#[derive(Debug, Error)]
enum GraphvizPngError {
    #[error("failed to spawn graphviz dot")]
    SpawnDot(#[source] std::io::Error),
    #[error("failed to open dot stdin")]
    MissingDotStdin,
    #[error("failed to write dot input to graphviz")]
    WriteDot(#[source] std::io::Error),
    #[error("failed to wait for graphviz")]
    WaitForDot(#[source] std::io::Error),
    #[error("graphviz dot exited with status {status}: {stderr}")]
    DotFailed { status: i32, stderr: String },
}

#[derive(Debug, Error)]
enum MergePngError {
    #[error("failed to determine current working directory")]
    CurrentDirectory(#[source] std::io::Error),
    #[error("failed to decode graphviz png")]
    DecodeGraphviz(#[source] image::ImageError),
    #[error("failed to create iced screenshot image buffer")]
    InvalidIcedScreenshotBuffer,
    #[error("failed to save merged png")]
    SavePng(#[source] image::ImageError),
}

impl Moarificator {
    fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::RenderMergedPng => {
                self.export_in_progress = true;
                self.pending_graphviz_png = None;
                self.pending_iced_screenshot = None;

                return Task::batch([
                    Task::perform(
                        render_graphviz_png(self.graph.clone()),
                        Message::GraphvizPngReady,
                    ),
                    Task::done(Message::CaptureIcedView),
                ]);
            }
            Message::CaptureIcedView => {
                return window::get_latest()
                    .and_then(window::screenshot)
                    .map(Message::IcedViewCaptured);
            }
            Message::GraphvizPngReady(result) => match result {
                Ok(bytes) => {
                    self.pending_graphviz_png = Some(bytes);
                    return self.maybe_merge_pngs();
                }
                Err(error) => {
                    self.export_in_progress = false;
                    self.pending_graphviz_png = None;
                    self.pending_iced_screenshot = None;
                    eprintln!("Failed to render Graphviz PNG: {error}");
                    if self.headless {
                        return close_latest_window();
                    }
                }
            },
            Message::IcedViewCaptured(screenshot) => {
                self.pending_iced_screenshot = Some(screenshot);
                return self.maybe_merge_pngs();
            }
            Message::MergedPngSaved(result) => {
                self.export_in_progress = false;
                match result {
                    Ok(path) => println!("Wrote {}", path.display()),
                    Err(error) => eprintln!("Failed to save merged PNG: {error}"),
                }

                if self.headless {
                    return close_latest_window();
                }
            }
            Message::Moar(n) => match n {
                0 => {
                    let from = fastrand::choice(self.graph.nodes.iter())
                        .copied()
                        .unwrap_or(0);
                    let to = *self.graph.nodes.iter().max().unwrap_or(&0) + 1;
                    self.graph.nodes.push(to);
                    self.graph.edges.push((from, to));
                }

                n => {
                    let from = n;
                    let existing_connections = self
                        .graph
                        .edges
                        .iter()
                        .filter_map(|(n, m)| {
                            (*n == from).then_some(m).or((*m == from).then_some(n))
                        })
                        .collect::<HashSet<_>>();
                    let potential_connections = self
                        .graph
                        .nodes
                        .iter()
                        .filter(|n| **n > 0 && **n != from && !existing_connections.contains(n))
                        .copied()
                        .collect::<Vec<_>>();

                    if potential_connections.is_empty() {
                        let to = *self.graph.nodes.iter().max().unwrap_or(&0) + 1;
                        self.graph.nodes.push(to);
                        self.graph.edges.push((from, to));
                    }

                    let Some(to) = fastrand::choice(potential_connections.iter()) else {
                        return Task::none();
                    };
                    self.graph.edges.push((from, *to));
                }
            },
        }

        Task::none()
    }

    fn maybe_merge_pngs(&mut self) -> Task<Message> {
        let Some(graphviz_png) = self.pending_graphviz_png.take() else {
            return Task::none();
        };
        let Some(iced_screenshot) = self.pending_iced_screenshot.take() else {
            self.pending_graphviz_png = Some(graphviz_png);
            return Task::none();
        };

        Task::perform(
            save_merged_png(graphviz_png, iced_screenshot),
            Message::MergedPngSaved,
        )
    }

    fn view(&self) -> Container<'_, Message> {
        let clusters = build_clusters(&self.graph);

        let graph = Container::new({
            let graph = Sugiyama::<Message, iced::Theme, iced::Renderer>::new(&self.graph, |n| {
                button(
                    container(text(node_label(n)).font(GRAPH_FONT))
                        .width(Length::Fill)
                        .height(Length::Fill)
                        .center_x(Length::Fill)
                        .center_y(Length::Fill),
                )
                .padding(0)
                .style(transparent_button_style)
                .on_press(Message::Moar(n))
                .width(node_size_f32(n))
                .height(node_size_f32(n))
                .into()
            })
            .edge_color(edge_colors)
            .edge_label(edge_label)
            .edge_label_element(|idx, edge, _s| {
                edge_label(idx, edge).map(|label| {
                    text(label)
                        .font(GRAPH_FONT)
                        .color(edge_colors(idx).0)
                        .into()
                })
            })
            .edge_endpoint(|_, _, kind, endpoint| {
                let marker = match kind {
                    EdgeEndpointKind::Source => "o",
                    EdgeEndpointKind::Destination => directional_marker(endpoint),
                };
                Some(iced::widget::text(marker).size(14).font(GRAPH_FONT).into())
            })
            .node_size(node_size)
            .edge_corner_radius(8.0)
            .edge_endpoint_extension(0.0)
            .clusters(clusters)
            .cluster_container(|idx, cluster| {
                Some(
                    container(
                        container(
                            iced::widget::text(cluster_label(idx, cluster))
                                .font(GRAPH_FONT)
                                .size(14),
                        )
                        .padding([4, 8])
                        .style(move |_| cluster_label_style(idx)),
                    )
                    .width(Length::Fill)
                    .height(Length::Fill)
                    .padding(12)
                    .align_x(Horizontal::Left)
                    .align_y(Vertical::Top)
                    .style(move |_| cluster_container_style(idx))
                    .into(),
                )
            })
            .cluster_color(|_| Color::TRANSPARENT)
            .padding(50);

            #[cfg(feature = "animated")]
            let graph = {
                let animation_duration = if self.export_in_progress || self.headless {
                    Duration::ZERO
                } else {
                    Duration::from_millis(400)
                };
                graph.animation_duration(animation_duration)
            };

            graph
        })
        .width(Length::Fill)
        .height(Length::Fill);

        let export_button: Element<'_, Message> = if self.export_in_progress {
            container(
                text("Render PNG")
                    .font(GRAPH_FONT)
                    .color(Color::TRANSPARENT),
            )
            .padding([8, 16])
            .into()
        } else {
            button(text("Render PNG").font(GRAPH_FONT))
                .style(transparent_button_style)
                .on_press(Message::RenderMergedPng)
                .padding([8, 16])
                .into()
        };

        Container::new(
            Column::new()
                //.push(export_button)
                .push(graph)
                .spacing(16)
                .align_x(Alignment::Start)
                .width(Length::Fill)
                .height(Length::Fill),
        )
        .width(Length::Fill)
        .height(Length::Fill)
        .padding(16)
    }
}

fn graph_to_dot(graph: &Graph) -> String {
    let clusters = build_clusters(graph);
    let mut dot = String::from("digraph G {\n");
    dot.push_str("    graph [\n");
    dot.push_str("        rankdir=TB,\n");
    dot.push_str("        compound=true,\n");
    dot.push_str("        dpi=96,\n");
    dot.push_str("        pad=0.520833,\n");
    dot.push_str(&format!(
        "        nodesep={:.6},\n",
        graph.config.vertex_spacing / 96.0
    ));
    dot.push_str(&format!(
        "        ranksep={:.6}\n",
        graph.config.vertex_spacing / 96.0
    ));
    dot.push_str("    ];\n");
    dot.push_str("    node [shape=box, style=\"rounded,filled\", fillcolor=\"#f5f5f5\", color=\"#444444\", fixedsize=true, fontname=\"Times-Roman\"];\n");
    dot.push_str(
        "    edge [dir=both, arrowtail=odot, arrowhead=normal, fontname=\"Times-Roman\"];\n",
    );

    let clustered_nodes = clusters
        .iter()
        .flat_map(|cluster| cluster.nodes.iter().copied())
        .collect::<HashSet<_>>();

    for node in graph
        .nodes
        .iter()
        .copied()
        .filter(|node| !clustered_nodes.contains(node))
    {
        push_node_dot(&mut dot, 1, node);
    }

    for cluster_index in 0..clusters.len() {
        if clusters[cluster_index].parent.is_none() {
            push_cluster_dot(&mut dot, &clusters, cluster_index, 1);
        }
    }

    for (index, edge) in graph.edges.iter().copied().enumerate() {
        let (stroke, _) = edge_colors(index);
        dot.push_str(&format!(
            "    {} -> {} [label=\"{}\", color=\"{}\", fontcolor=\"{}\", minlen={}];\n",
            edge.0,
            edge.1,
            dot_escape(&edge_label(index, edge).unwrap_or_default()),
            color_to_hex(stroke),
            color_to_hex(stroke),
            graph.config.minimum_length
        ));
    }

    dot.push_str("}\n");
    dot
}

fn initial_graph() -> Graph {
    let mut rng = fastrand::Rng::with_seed(0x5EED_5EED);
    let mut nodes = vec![0_u32];
    let mut edges = Vec::new();

    for to in 1_u32..=6 {
        let max_edges = usize::min(3, to as usize);
        let edge_count = rng.usize(1..=max_edges);
        let mut connected_from = HashSet::new();

        while connected_from.len() < edge_count {
            let from = rng.u32(0..to);
            if connected_from.insert(from) {
                edges.push((from, to));
            }
        }

        nodes.push(to);
    }

    Graph {
        nodes,
        edges,
        config: Default::default(),
    }
}

fn close_latest_window() -> Task<Message> {
    window::get_latest().then(|id| match id {
        Some(id) => window::close(id),
        None => Task::none(),
    })
}

async fn render_graphviz_png(graph: Graph) -> Result<Vec<u8>, String> {
    let dot = graph_to_dot(&graph);

    let mut child = Command::new("dot")
        .arg("-Tpng")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(GraphvizPngError::SpawnDot)
        .map_err(|error| error.to_string())?;

    let mut stdin = child
        .stdin
        .take()
        .ok_or(GraphvizPngError::MissingDotStdin)
        .map_err(|error| error.to_string())?;
    stdin
        .write_all(dot.as_bytes())
        .map_err(GraphvizPngError::WriteDot)
        .map_err(|error| error.to_string())?;
    drop(stdin);

    let output = child
        .wait_with_output()
        .map_err(GraphvizPngError::WaitForDot)
        .map_err(|error| error.to_string())?;

    if output.status.success() {
        Ok(output.stdout)
    } else {
        Err(GraphvizPngError::DotFailed {
            status: output.status.code().unwrap_or(-1),
            stderr: String::from_utf8_lossy(&output.stderr).trim().to_string(),
        })
        .map_err(|error| error.to_string())
    }
}

async fn save_merged_png(graphviz_png: Vec<u8>, screenshot: Screenshot) -> Result<PathBuf, String> {
    let output_path = env::current_dir()
        .map_err(MergePngError::CurrentDirectory)
        .map_err(|error| error.to_string())?
        .join("moar-render.png");

    let graphviz_image =
        image::load_from_memory_with_format(&graphviz_png, image::ImageFormat::Png)
            .map_err(MergePngError::DecodeGraphviz)
            .map_err(|error| error.to_string())?
            .to_rgba8();
    let iced_image = image::RgbaImage::from_raw(
        screenshot.size.width,
        screenshot.size.height,
        screenshot.bytes.to_vec(),
    )
    .ok_or(MergePngError::InvalidIcedScreenshotBuffer)
    .map_err(|error| error.to_string())?;

    let merged_width = graphviz_image.width() + iced_image.width();
    let merged_height = graphviz_image.height().max(iced_image.height());
    let mut merged = image::RgbaImage::from_pixel(
        merged_width,
        merged_height,
        image::Rgba([255, 255, 255, 255]),
    );
    image::imageops::overlay(&mut merged, &graphviz_image, 0, 0);
    image::imageops::overlay(
        &mut merged,
        &iced_image,
        i64::from(graphviz_image.width()),
        0,
    );

    merged
        .save(&output_path)
        .map_err(MergePngError::SavePng)
        .map_err(|error| error.to_string())?;

    Ok(output_path)
}

fn build_clusters(graph: &Graph) -> Vec<Cluster> {
    let even_cluster_nodes = graph
        .nodes
        .iter()
        .copied()
        .filter(|node| *node != 0 && node % 2 == 0)
        .collect::<Vec<_>>();
    let odd_cluster_nodes = graph
        .nodes
        .iter()
        .copied()
        .filter(|node| *node % 2 == 1)
        .collect::<Vec<_>>();
    let all_cluster_nodes = graph
        .nodes
        .iter()
        .copied()
        .filter(|node| *node != 0)
        .collect::<Vec<_>>();
    let mut clusters = Vec::new();
    let mut parent_cluster_index = None;
    if all_cluster_nodes.len() > 2 {
        parent_cluster_index = Some(clusters.len());
        clusters.push(Cluster::new(all_cluster_nodes).padding(10.0));
    }
    if odd_cluster_nodes.len() > 1 {
        let cluster = Cluster::new(odd_cluster_nodes).padding(10.0);
        clusters.push(match parent_cluster_index {
            Some(parent) => cluster.parent(parent),
            None => cluster,
        });
    }
    if even_cluster_nodes.len() > 1 {
        let cluster = Cluster::new(even_cluster_nodes).padding(10.0);
        clusters.push(match parent_cluster_index {
            Some(parent) => cluster.parent(parent),
            None => cluster,
        });
    }
    clusters
}

fn node_label(node: u32) -> String {
    if node == 0 {
        "Moar".to_string()
    } else {
        node.to_string()
    }
}

fn edge_label(index: usize, (from, to): (u32, u32)) -> Option<String> {
    let _ = index;
    Some(format!("{from} -> {to}"))
}

fn cluster_label(index: usize, cluster: &Cluster) -> String {
    match cluster.parent {
        Some(parent) => format!("cluster {index} (child of {parent})"),
        None => format!("cluster {index}"),
    }
}

fn node_size(node: u32) -> (f64, f64) {
    let side = if node == 0 {
        100.0
    } else {
        10.0 * f64::from(node)
    }
    .max(MIN_NODE_SIDE);
    (side, side)
}

fn node_size_f32(node: u32) -> f32 {
    node_size(node).0 as f32
}

fn edge_colors(index: usize) -> (iced::Color, iced::Color) {
    let blue = match index % 9 {
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
}

fn cluster_color(index: usize) -> iced::Color {
    match index % 2 {
        0 => iced::Color::from_rgba8(255, 112, 67, 0.9),
        _ => iced::Color::from_rgba8(46, 125, 50, 0.9),
    }
}

fn transparent_button_style(_theme: &Theme, _status: button::Status) -> button::Style {
    button::Style {
        background: Some(Background::Color(Color::WHITE)),
        text_color: Color::BLACK,
        border: border::rounded(NODE_BORDER_RADIUS)
            .width(2.0)
            .color(Color::BLACK),
        shadow: Default::default(),
    }
}

fn cluster_container_style(index: usize) -> container::Style {
    let color = cluster_color(index);
    container::Style::default()
        .color(color)
        .background(Background::Color(Color::TRANSPARENT))
        .border(
            border::rounded(CLUSTER_BORDER_RADIUS)
                .width(2.0)
                .color(color),
        )
}

fn cluster_label_style(index: usize) -> container::Style {
    container::Style::default()
        .color(cluster_color(index))
        .background(Background::Color(Color::WHITE))
}

fn push_cluster_dot(dot: &mut String, clusters: &[Cluster], index: usize, indent: usize) {
    let indent_str = "    ".repeat(indent);
    let cluster = &clusters[index];
    let color = color_to_hex(cluster_color(index));
    dot.push_str(&format!("{indent_str}subgraph cluster_{index} {{\n"));
    dot.push_str(&format!(
        "{indent_str}    label=\"{}\";\n",
        dot_escape(&cluster_label(index, cluster))
    ));
    dot.push_str(&format!("{indent_str}    fontname=\"Times-Roman\";\n"));
    dot.push_str(&format!("{indent_str}    color=\"{color}\";\n"));
    dot.push_str(&format!("{indent_str}    fontcolor=\"{color}\";\n"));
    dot.push_str(&format!("{indent_str}    pencolor=\"{color}\";\n"));
    dot.push_str(&format!("{indent_str}    style=\"rounded\";\n"));
    if let Some(padding) = cluster.padding {
        dot.push_str(&format!("{indent_str}    margin={padding:.6};\n"));
    }

    let mut descendant_nodes = HashSet::new();
    for child_index in 0..clusters.len() {
        if clusters[child_index].parent == Some(index) {
            descendant_nodes.extend(clusters[child_index].nodes.iter().copied());
        }
    }

    for node in cluster
        .nodes
        .iter()
        .copied()
        .filter(|node| !descendant_nodes.contains(node))
    {
        push_node_dot(dot, indent + 1, node);
    }

    for child_index in 0..clusters.len() {
        if clusters[child_index].parent == Some(index) {
            push_cluster_dot(dot, clusters, child_index, indent + 1);
        }
    }

    dot.push_str(&format!("{indent_str}}}\n"));
}

fn push_node_dot(dot: &mut String, indent: usize, node: u32) {
    let indent_str = "    ".repeat(indent);
    let (width, height) = node_size(node);
    dot.push_str(&format!(
        "{indent_str}{node} [label=\"{}\", width={:.6}, height={:.6}];\n",
        dot_escape(&node_label(node)),
        width / 96.0,
        height / 96.0
    ));
}

fn dot_escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

fn color_to_hex(color: iced::Color) -> String {
    let [r, g, b, a] = color.into_rgba8();
    format!("#{r:02X}{g:02X}{b:02X}{a:02X}")
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
