use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Duration;

use iced::alignment::{Horizontal, Vertical};
use iced::application::Title;
use iced::mouse;
use iced::widget::canvas::{self, Path};
use iced::widget::{Column, Container, button, container, text};
use iced::window;
use iced::window::Screenshot;
use iced::{Alignment, Background, Color, Element, Font, Length, Task, Theme, border};
use iced::{Point, Rectangle, Vector};
use iced_sugiyama::{Cluster, EdgeEndpointKind, Graph, Sugiyama};
use serde_json::Value;
use thiserror::Error;

const GRAPH_FONT: Font = Font::with_name("Times New Roman");
const NODE_BORDER_RADIUS: f32 = 16.0;
const CLUSTER_BORDER_RADIUS: f32 = 18.0;
const MIN_NODE_SIDE: f64 = 72.0;

const WINDOW_WIDTH: f32 = 800.0;
const WINDOW_HEIGHT: f32 = 800.0;
const DEFAULT_GRAPH_SEED: u64 = 0x5EED_5EED;
const DEFAULT_GRAPH_NODE_COUNT: u32 = 6;
const DEFAULT_GRAPH_CLUSTER_COUNT: usize = 3;
const GRAPHVIZ_PNG_SCALE: f64 = 0.7;
const MERGED_PNG_OUTPUT_PATH: &str = "/tmp/iced-sugiyama-comp.png";
const GRAPHVIZ_PLAIN_OUTPUT_PATH: &str = "/tmp/iced-sugiyama-gviz.txt";
const CUSTOM_LAYOUT_PLAIN_OUTPUT_PATH: &str = "/tmp/iced-sugiyama-out.txt";
const LAYOUT_SCORES_OUTPUT_PATH: &str = "/tmp/iced-sugiyama-scores.txt";

#[derive(Debug, Clone, Copy)]
enum GraphvizEndpointGlyphKind {
    OpenDot,
    NormalArrow,
}

#[derive(Debug, Clone, Copy)]
struct GraphvizEndpointGlyph {
    kind: GraphvizEndpointGlyphKind,
    color: Color,
    angle_radians: f32,
}

impl GraphvizEndpointGlyph {
    fn size(self) -> f32 {
        match self.kind {
            GraphvizEndpointGlyphKind::OpenDot => 16.0,
            GraphvizEndpointGlyphKind::NormalArrow => 20.0,
        }
    }
}

impl<Message, Theme, Renderer> canvas::Program<Message, Theme, Renderer> for GraphvizEndpointGlyph
where
    Renderer: iced::advanced::graphics::geometry::Renderer,
{
    type State = ();

    fn draw(
        &self,
        _state: &Self::State,
        renderer: &Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<canvas::Geometry<Renderer>> {
        let mut frame = canvas::Frame::new(renderer, bounds.size());
        let anchor = frame.center();

        match self.kind {
            GraphvizEndpointGlyphKind::OpenDot => {
                let radius = 4.0;
                let direction = Vector::new(self.angle_radians.cos(), self.angle_radians.sin());
                let center = Point::new(
                    anchor.x + direction.x * radius,
                    anchor.y + direction.y * radius,
                );
                let circle = Path::circle(center, radius);

                frame.fill(&circle, Color::WHITE);
                frame.stroke(
                    &circle,
                    canvas::Stroke {
                        width: 1.5,
                        style: canvas::stroke::Style::Solid(self.color),
                        ..canvas::Stroke::default()
                    },
                );
            }
            GraphvizEndpointGlyphKind::NormalArrow => {
                let arrow = Path::new(|path| {
                    path.move_to(Point::new(0.0, 0.0));
                    path.line_to(Point::new(-10.0, 4.0));
                    path.line_to(Point::new(-7.25, 0.0));
                    path.line_to(Point::new(-10.0, -4.0));
                    path.close();
                });

                frame.with_save(|frame| {
                    frame.translate(Vector::new(anchor.x, anchor.y));
                    frame.rotate(self.angle_radians);
                    frame.fill(&arrow, self.color);
                });
            }
        }

        vec![frame.into_geometry()]
    }
}

pub fn main() -> iced::Result {
    let options = AppOptions::from_env();

    if options.headless && options.noimg {
        match render_graphviz_artifacts(
            initial_graph(options.seed, options.node_count),
            options.seed,
            options.cluster_count,
            false,
        ) {
            Ok(artifacts) => {
                println!("Wrote {}", artifacts.graphviz_plain_path.display());
                println!("Wrote {}", artifacts.custom_plain_path.display());
                println!("Wrote {}", artifacts.scores_path.display());
            }
            Err(error) => eprintln!("Failed to render Graphviz artifacts: {error}"),
        }

        return Ok(());
    }

    iced::application(
        |state: &Moarificator| state.title(&()),
        Moarificator::update,
        Moarificator::view,
    )
    .theme(|_| iced::Theme::Light)
    .scale_factor(|_| 0.8)
    .window(window::Settings {
        size: iced::Size::new(WINDOW_WIDTH, WINDOW_HEIGHT),
        ..Default::default()
    })
    .run_with(move || {
        let task = Task::done(Message::AppStarted);

        (
            Moarificator::new(
                options.headless,
                options.noimg,
                options.seed,
                options.node_count,
                options.cluster_count,
            ),
            task,
        )
    })
}

impl<S> Title<S> for Moarificator {
    fn title(&self, _state: &S) -> String {
        "iced-sugiyama".to_string()
    }
}

struct Moarificator {
    headless: bool,
    noimg: bool,
    export_in_progress: bool,
    graph: Graph,
    cluster_seed: u64,
    cluster_count: usize,
    pending_graphviz_png: Option<Vec<u8>>,
    pending_iced_screenshot: Option<Screenshot>,
}
impl Moarificator {
    fn new(headless: bool, noimg: bool, seed: u64, node_count: u32, cluster_count: usize) -> Self {
        Self {
            headless,
            noimg,
            export_in_progress: false,
            graph: initial_graph(seed, node_count),
            cluster_seed: seed,
            cluster_count,
            pending_graphviz_png: None,
            pending_iced_screenshot: None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct AppOptions {
    headless: bool,
    noimg: bool,
    seed: u64,
    node_count: u32,
    cluster_count: usize,
}

impl AppOptions {
    fn from_env() -> Self {
        let mut headless = false;
        let mut noimg = false;
        let mut seed = DEFAULT_GRAPH_SEED;
        let mut node_count = DEFAULT_GRAPH_NODE_COUNT;
        let mut cluster_count = DEFAULT_GRAPH_CLUSTER_COUNT;
        let mut args = env::args().skip(1);

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--headless" => headless = true,
                "--noimg" => noimg = true,
                "--seed" => {
                    let raw_seed = args
                        .next()
                        .unwrap_or_else(|| panic!("missing value for --seed"));
                    seed = parse_seed_arg(&raw_seed);
                }
                _ if arg.starts_with("--seed=") => {
                    seed = parse_seed_arg(&arg["--seed=".len()..]);
                }
                "--nodes" => {
                    let raw_node_count = args
                        .next()
                        .unwrap_or_else(|| panic!("missing value for --nodes"));
                    node_count = parse_node_count_arg(&raw_node_count);
                }
                _ if arg.starts_with("--nodes=") => {
                    node_count = parse_node_count_arg(&arg["--nodes=".len()..]);
                }
                "--clusters" => {
                    let raw_cluster_count = args
                        .next()
                        .unwrap_or_else(|| panic!("missing value for --clusters"));
                    cluster_count = parse_cluster_count_arg(&raw_cluster_count);
                }
                _ if arg.starts_with("--clusters=") => {
                    cluster_count = parse_cluster_count_arg(&arg["--clusters=".len()..]);
                }
                _ => {}
            }
        }

        Self {
            headless,
            noimg,
            seed,
            node_count,
            cluster_count,
        }
    }
}

fn parse_seed_arg(raw_seed: &str) -> u64 {
    if let Some(hex_seed) = raw_seed.strip_prefix("0x") {
        return u64::from_str_radix(hex_seed, 16)
            .unwrap_or_else(|_| panic!("invalid value for --seed: {raw_seed}"));
    }

    raw_seed
        .parse()
        .unwrap_or_else(|_| panic!("invalid value for --seed: {raw_seed}"))
}

fn parse_node_count_arg(raw_node_count: &str) -> u32 {
    raw_node_count
        .parse::<u32>()
        .unwrap_or_else(|_| panic!("invalid value for --nodes: {raw_node_count}"))
        .max(1)
}

fn parse_cluster_count_arg(raw_cluster_count: &str) -> usize {
    raw_cluster_count
        .parse::<usize>()
        .unwrap_or_else(|_| panic!("invalid value for --clusters: {raw_cluster_count}"))
}

#[derive(Debug, Clone)]
enum Message {
    AppStarted,
    Moar(u32),
    RenderMergedPng,
    CaptureIcedView,
    GraphvizArtifactsReady(Result<GraphvizArtifacts, String>),
    IcedViewCaptured(Screenshot),
    MergedPngSaved(Result<PathBuf, String>),
}

#[derive(Debug, Clone)]
struct GraphvizArtifacts {
    png: Option<Vec<u8>>,
    graphviz_plain_path: PathBuf,
    custom_plain_path: PathBuf,
    scores_path: PathBuf,
}

#[derive(Debug, Error)]
enum GraphvizOutputError {
    #[error("failed to spawn graphviz dot for {format} output")]
    SpawnDot {
        format: &'static str,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to open dot stdin")]
    MissingDotStdin,
    #[error("failed to write dot input to graphviz")]
    WriteDot(#[source] std::io::Error),
    #[error("failed to wait for graphviz")]
    WaitForDot(#[source] std::io::Error),
    #[error("graphviz dot exited with status {status} while rendering {format}: {stderr}")]
    DotFailed {
        format: &'static str,
        status: i32,
        stderr: String,
    },
    #[error("failed to save graphviz plain output")]
    SavePlain(#[source] std::io::Error),
    #[error("failed to save custom layout plain output")]
    SaveCustomPlain(#[source] std::io::Error),
    #[error("failed to save layout scores output")]
    SaveScores(#[source] std::io::Error),
    #[error("failed to parse graphviz json output")]
    ParseJson(#[source] serde_json::Error),
    #[error("graphviz json output missing {field}")]
    MissingJsonField { field: &'static str },
    #[error("graphviz json output has invalid {field}")]
    InvalidJsonField { field: &'static str },
    #[error("failed to parse plain layout output")]
    ParsePlainOutput,
    #[error("invalid plain layout output")]
    InvalidPlainOutput,
}

#[derive(Debug, Error)]
enum MergePngError {
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
            Message::AppStarted => {
                let resize = window::get_latest().and_then(|id| {
                    window::resize::<f32>(id, iced::Size::new(WINDOW_WIDTH, WINDOW_HEIGHT))
                        .discard()
                });

                if self.headless {
                    return Task::batch([resize, Task::done(Message::RenderMergedPng)]);
                }

                return resize;
            }
            Message::RenderMergedPng => {
                self.export_in_progress = true;
                self.pending_graphviz_png = None;
                self.pending_iced_screenshot = None;

                let include_png = !(self.headless && self.noimg);
                let graph = self.graph.clone();
                let cluster_seed = self.cluster_seed;
                let cluster_count = self.cluster_count;
                let render = Task::perform(
                    async move {
                        render_graphviz_artifacts(graph, cluster_seed, cluster_count, include_png)
                    },
                    Message::GraphvizArtifactsReady,
                );

                if include_png {
                    return Task::batch([render, Task::done(Message::CaptureIcedView)]);
                }

                return render;
            }
            Message::CaptureIcedView => {
                return window::get_latest()
                    .and_then(window::screenshot)
                    .map(Message::IcedViewCaptured);
            }
            Message::GraphvizArtifactsReady(result) => match result {
                Ok(artifacts) => {
                    println!("Wrote {}", artifacts.graphviz_plain_path.display());
                    println!("Wrote {}", artifacts.custom_plain_path.display());
                    println!("Wrote {}", artifacts.scores_path.display());
                    if let Some(png) = artifacts.png {
                        self.pending_graphviz_png = Some(png);
                        return self.maybe_merge_pngs();
                    }

                    self.export_in_progress = false;
                    if self.headless {
                        return close_latest_window();
                    }
                }
                Err(error) => {
                    self.export_in_progress = false;
                    self.pending_graphviz_png = None;
                    self.pending_iced_screenshot = None;
                    eprintln!("Failed to render Graphviz artifacts: {error}");
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
        let clusters = build_clusters(&self.graph, self.cluster_seed, self.cluster_count);

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
            .label_color(|idx| edge_colors(idx).0)
            .edge_endpoint(|idx, _, kind, endpoint| {
                let color = edge_colors(idx).0;
                let kind = match kind {
                    EdgeEndpointKind::Source => GraphvizEndpointGlyphKind::OpenDot,
                    EdgeEndpointKind::Destination => GraphvizEndpointGlyphKind::NormalArrow,
                };
                let glyph = GraphvizEndpointGlyph {
                    kind,
                    color,
                    angle_radians: endpoint.angle_radians(),
                };
                Some(
                    canvas::Canvas::new(glyph)
                        .width(glyph.size())
                        .height(glyph.size())
                        .into(),
                )
            })
            .node_size(node_size)
            .edge_corner_radius(8.0)
            .edge_endpoint_extension(0.0)
            .clusters(clusters)
            .render_config(render_config())
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
            .padding(70);

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

fn graph_to_dot(graph: &Graph, cluster_seed: u64, cluster_count: usize) -> String {
    let clusters = build_clusters(graph, cluster_seed, cluster_count);
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

fn initial_graph(seed: u64, node_count: u32) -> Graph {
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut nodes = vec![0_u32];
    let mut edges = Vec::new();

    for to in 1_u32..node_count {
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
        config: rust_sugiyama::Config {
            vertex_spacing: 26.0,
            ..Default::default()
        },
    }
}

fn close_latest_window() -> Task<Message> {
    window::get_latest().then(|id| match id {
        Some(id) => window::close(id),
        None => Task::none(),
    })
}

fn run_graphviz(dot: &str, format: &'static str) -> Result<Vec<u8>, GraphvizOutputError> {
    let mut child = Command::new("dot")
        .arg(format!("-T{format}"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|source| GraphvizOutputError::SpawnDot { format, source })?;

    let mut stdin = child
        .stdin
        .take()
        .ok_or(GraphvizOutputError::MissingDotStdin)?;
    stdin
        .write_all(dot.as_bytes())
        .map_err(GraphvizOutputError::WriteDot)?;
    drop(stdin);

    let output = child
        .wait_with_output()
        .map_err(GraphvizOutputError::WaitForDot)?;

    if output.status.success() {
        Ok(output.stdout)
    } else {
        Err(GraphvizOutputError::DotFailed {
            format,
            status: output.status.code().unwrap_or(-1),
            stderr: String::from_utf8_lossy(&output.stderr).trim().to_string(),
        })
    }
}

fn render_graphviz_artifacts(
    graph: Graph,
    cluster_seed: u64,
    cluster_count: usize,
    include_png: bool,
) -> Result<GraphvizArtifacts, String> {
    let dot = graph_to_dot(&graph, cluster_seed, cluster_count);
    let clusters = build_clusters(&graph, cluster_seed, cluster_count);
    let png = if include_png {
        Some(run_graphviz(&dot, "png").map_err(|error| error.to_string())?)
    } else {
        None
    };
    let graphviz_json = run_graphviz(&dot, "json0").map_err(|error| error.to_string())?;
    let graphviz_plain_path = PathBuf::from(GRAPHVIZ_PLAIN_OUTPUT_PATH);
    let custom_plain_path = PathBuf::from(CUSTOM_LAYOUT_PLAIN_OUTPUT_PATH);
    let scores_path = PathBuf::from(LAYOUT_SCORES_OUTPUT_PATH);
    let custom_plain = iced_sugiyama::graphviz_plain_layout(
        &graph.nodes,
        &graph.edges,
        &graph.config,
        node_size,
        node_label,
        edge_label,
        &clusters,
        &render_config(),
        cluster_label,
    );
    let graphviz_plain =
        graphviz_json_to_plain(&graphviz_json).map_err(|error| error.to_string())?;
    let scores =
        compare_plain_layouts(&graphviz_plain, &custom_plain).map_err(|error| error.to_string())?;

    fs::write(&graphviz_plain_path, graphviz_plain)
        .map_err(GraphvizOutputError::SavePlain)
        .map_err(|error| error.to_string())?;
    fs::write(&custom_plain_path, custom_plain)
        .map_err(GraphvizOutputError::SaveCustomPlain)
        .map_err(|error| error.to_string())?;
    fs::write(&scores_path, scores.to_line())
        .map_err(GraphvizOutputError::SaveScores)
        .map_err(|error| error.to_string())?;

    Ok(GraphvizArtifacts {
        png,
        graphviz_plain_path,
        custom_plain_path,
        scores_path,
    })
}

fn graphviz_json_to_plain(json_bytes: &[u8]) -> Result<String, GraphvizOutputError> {
    fn json_string<'a>(
        value: &'a Value,
        field: &'static str,
    ) -> Result<&'a str, GraphvizOutputError> {
        value
            .as_str()
            .ok_or(GraphvizOutputError::MissingJsonField { field })
    }

    fn json_u64(value: &Value, field: &'static str) -> Result<u64, GraphvizOutputError> {
        value
            .as_u64()
            .ok_or(GraphvizOutputError::MissingJsonField { field })
    }

    fn parse_point(value: &str) -> Result<(f64, f64), GraphvizOutputError> {
        let mut parts = value.split(',');
        let x = parts
            .next()
            .ok_or(GraphvizOutputError::InvalidJsonField { field: "point" })?
            .parse::<f64>()
            .map_err(|_| GraphvizOutputError::InvalidJsonField { field: "point" })?;
        let y = parts
            .next()
            .ok_or(GraphvizOutputError::InvalidJsonField { field: "point" })?
            .parse::<f64>()
            .map_err(|_| GraphvizOutputError::InvalidJsonField { field: "point" })?;
        Ok((x, y))
    }

    fn parse_bb(value: &str) -> Result<(f64, f64, f64, f64), GraphvizOutputError> {
        let mut parts = value.split(',');
        let min_x = parts
            .next()
            .ok_or(GraphvizOutputError::InvalidJsonField { field: "bb" })?
            .parse::<f64>()
            .map_err(|_| GraphvizOutputError::InvalidJsonField { field: "bb" })?;
        let min_y = parts
            .next()
            .ok_or(GraphvizOutputError::InvalidJsonField { field: "bb" })?
            .parse::<f64>()
            .map_err(|_| GraphvizOutputError::InvalidJsonField { field: "bb" })?;
        let max_x = parts
            .next()
            .ok_or(GraphvizOutputError::InvalidJsonField { field: "bb" })?
            .parse::<f64>()
            .map_err(|_| GraphvizOutputError::InvalidJsonField { field: "bb" })?;
        let max_y = parts
            .next()
            .ok_or(GraphvizOutputError::InvalidJsonField { field: "bb" })?
            .parse::<f64>()
            .map_err(|_| GraphvizOutputError::InvalidJsonField { field: "bb" })?;
        Ok((min_x, min_y, max_x, max_y))
    }

    fn parse_edge_points(value: &str) -> Result<Vec<(f64, f64)>, GraphvizOutputError> {
        value
            .split_whitespace()
            .filter_map(|token| {
                let trimmed = token
                    .strip_prefix("s,")
                    .or_else(|| token.strip_prefix("e,"))
                    .unwrap_or(token);
                trimmed.contains(',').then_some(trimmed)
            })
            .map(parse_point)
            .collect()
    }

    fn format_number(value: f64) -> String {
        let mut text = format!("{value:.5}");
        while text.contains('.') && text.ends_with('0') {
            text.pop();
        }
        if text.ends_with('.') {
            text.pop();
        }
        text
    }

    fn format_label(value: &str) -> String {
        let simple = !value.is_empty()
            && value
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.' | '-'));
        if simple {
            value.to_string()
        } else {
            format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""))
        }
    }

    #[derive(Clone)]
    struct GraphvizCluster {
        index: usize,
        gvid: u64,
        label: String,
        bounds: (f64, f64, f64, f64),
        child_gvids: Vec<u64>,
        parent: Option<usize>,
    }

    #[derive(Clone)]
    struct GraphvizNode {
        id: u32,
        center: (f64, f64),
        size: (f64, f64),
        label: String,
    }

    #[derive(Clone)]
    struct GraphvizEdge {
        tail: u32,
        head: u32,
        points: Vec<(f64, f64)>,
        label: Option<String>,
        label_position: Option<(f64, f64)>,
    }

    const POINTS_PER_INCH: f64 = 72.0;

    let json: Value = serde_json::from_slice(json_bytes).map_err(GraphvizOutputError::ParseJson)?;
    let (_, _, max_x, max_y) = parse_bb(json_string(&json["bb"], "bb")?)?;
    let objects = json["objects"]
        .as_array()
        .ok_or(GraphvizOutputError::MissingJsonField { field: "objects" })?;
    let mut clusters = Vec::new();
    let mut nodes = Vec::new();
    let mut node_ids_by_gvid = std::collections::HashMap::new();

    for object in objects {
        let name = json_string(&object["name"], "objects[].name")?;
        let gvid = json_u64(&object["_gvid"], "objects[]._gvid")?;

        if let Some(suffix) = name.strip_prefix("cluster_") {
            let index =
                suffix
                    .parse::<usize>()
                    .map_err(|_| GraphvizOutputError::InvalidJsonField {
                        field: "objects[].name",
                    })?;
            let bounds = parse_bb(json_string(&object["bb"], "objects[].bb")?)?;
            let child_gvids = object["subgraphs"]
                .as_array()
                .map(|subgraphs| {
                    subgraphs
                        .iter()
                        .map(|child| json_u64(child, "objects[].subgraphs[]"))
                        .collect::<Result<Vec<_>, _>>()
                })
                .transpose()?
                .unwrap_or_default();
            let label = object["label"].as_str().unwrap_or_default().to_string();
            clusters.push(GraphvizCluster {
                index,
                gvid,
                label,
                bounds,
                child_gvids,
                parent: None,
            });
            continue;
        }

        let Ok(id) = name.parse::<u32>() else {
            continue;
        };
        let center = parse_point(json_string(&object["pos"], "objects[].pos")?)?;
        let width = json_string(&object["width"], "objects[].width")?
            .parse::<f64>()
            .map_err(|_| GraphvizOutputError::InvalidJsonField {
                field: "objects[].width",
            })?;
        let height = json_string(&object["height"], "objects[].height")?
            .parse::<f64>()
            .map_err(|_| GraphvizOutputError::InvalidJsonField {
                field: "objects[].height",
            })?;
        node_ids_by_gvid.insert(gvid, id);
        nodes.push(GraphvizNode {
            id,
            center: (center.0, center.1),
            size: (width, height),
            label: object["label"].as_str().unwrap_or(name).to_string(),
        });
    }

    let cluster_index_by_gvid = clusters
        .iter()
        .enumerate()
        .map(|(position, cluster)| (cluster.gvid, position))
        .collect::<std::collections::HashMap<_, _>>();
    for parent_position in 0..clusters.len() {
        let parent_index = clusters[parent_position].index;
        let child_positions = clusters[parent_position]
            .child_gvids
            .iter()
            .filter_map(|gvid| cluster_index_by_gvid.get(gvid).copied())
            .collect::<Vec<_>>();
        for child_position in child_positions {
            clusters[child_position].parent = Some(parent_index);
        }
    }

    let mut edges = Vec::new();
    for edge in json["edges"]
        .as_array()
        .ok_or(GraphvizOutputError::MissingJsonField { field: "edges" })?
    {
        let tail_gvid = json_u64(&edge["tail"], "edges[].tail")?;
        let head_gvid = json_u64(&edge["head"], "edges[].head")?;
        let tail = node_ids_by_gvid.get(&tail_gvid).copied().ok_or(
            GraphvizOutputError::InvalidJsonField {
                field: "edges[].tail",
            },
        )?;
        let head = node_ids_by_gvid.get(&head_gvid).copied().ok_or(
            GraphvizOutputError::InvalidJsonField {
                field: "edges[].head",
            },
        )?;
        let points = parse_edge_points(json_string(&edge["pos"], "edges[].pos")?)?;
        let label = edge["label"].as_str().map(str::to_string);
        let label_position = edge["lp"].as_str().map(parse_point).transpose()?;
        edges.push(GraphvizEdge {
            tail,
            head,
            points,
            label,
            label_position,
        });
    }

    let mut plain = String::new();
    plain.push_str(&format!(
        "graph 1 {} {}\n",
        format_number(max_x / POINTS_PER_INCH),
        format_number(max_y / POINTS_PER_INCH)
    ));

    clusters.sort_by_key(|cluster| cluster.index);
    for cluster in clusters {
        plain.push_str(&format!(
            "cluster {} {} {} {} {} {} {}\n",
            cluster.index,
            cluster
                .parent
                .map(|parent| parent.to_string())
                .unwrap_or_else(|| "_".to_string()),
            format_number(cluster.bounds.0 / POINTS_PER_INCH),
            format_number(cluster.bounds.1 / POINTS_PER_INCH),
            format_number(cluster.bounds.2 / POINTS_PER_INCH),
            format_number(cluster.bounds.3 / POINTS_PER_INCH),
            format_label(&cluster.label)
        ));
    }

    for node in nodes {
        plain.push_str(&format!(
            "node {} {} {} {} {} {}\n",
            node.id,
            format_number(node.center.0 / POINTS_PER_INCH),
            format_number(node.center.1 / POINTS_PER_INCH),
            format_number(node.size.0),
            format_number(node.size.1),
            format_label(&node.label)
        ));
    }

    for edge in edges {
        plain.push_str(&format!(
            "edge {} {} {}",
            edge.tail,
            edge.head,
            edge.points.len()
        ));
        for point in edge.points {
            plain.push_str(&format!(
                " {} {}",
                format_number(point.0 / POINTS_PER_INCH),
                format_number(point.1 / POINTS_PER_INCH)
            ));
        }
        if let (Some(label), Some((x, y))) = (edge.label, edge.label_position) {
            plain.push_str(&format!(
                " {} {} {}",
                format_label(&label),
                format_number(x / POINTS_PER_INCH),
                format_number(y / POINTS_PER_INCH)
            ));
        }
        plain.push('\n');
    }

    plain.push_str("stop\n");
    Ok(plain)
}

#[derive(Debug, Clone, Copy)]
struct SimilarityScores {
    nodes: f64,
    edges: f64,
    clusters: f64,
    overall: f64,
}

impl SimilarityScores {
    fn to_line(self) -> String {
        format!(
            "nodes: {:.2}, edges: {:.2}, clusters: {:.2}, overall: {:.2}\n",
            self.nodes, self.edges, self.clusters, self.overall
        )
    }
}

#[derive(Debug, Clone)]
struct ParsedPlainLayout {
    width: f64,
    height: f64,
    nodes: HashMap<u32, PlainNode>,
    edges: HashMap<(u32, u32), PlainEdge>,
    clusters: HashMap<usize, PlainCluster>,
}

#[derive(Debug, Clone, Copy)]
struct PlainNode {
    center: (f64, f64),
    size: (f64, f64),
}

#[derive(Debug, Clone)]
struct PlainEdge {
    points: Vec<(f64, f64)>,
    label_position: Option<(f64, f64)>,
}

#[derive(Debug, Clone, Copy)]
struct PlainCluster {
    parent: Option<usize>,
    bounds: (f64, f64, f64, f64),
}

fn compare_plain_layouts(left: &str, right: &str) -> Result<SimilarityScores, GraphvizOutputError> {
    let left = parse_plain_layout(left)?;
    let right = parse_plain_layout(right)?;
    let diagonal = left
        .width
        .max(right.width)
        .hypot(left.height.max(right.height))
        .max(1.0);

    let nodes = node_similarity(&left, &right, diagonal);
    let edges = edge_similarity(&left, &right, diagonal);
    let clusters = cluster_similarity(&left, &right);
    let overall = (nodes + edges + clusters) / 3.0;

    Ok(SimilarityScores {
        nodes,
        edges,
        clusters,
        overall,
    })
}

fn parse_plain_layout(input: &str) -> Result<ParsedPlainLayout, GraphvizOutputError> {
    let mut layout = ParsedPlainLayout {
        width: 0.0,
        height: 0.0,
        nodes: HashMap::new(),
        edges: HashMap::new(),
        clusters: HashMap::new(),
    };

    for line in input.lines().map(str::trim).filter(|line| !line.is_empty()) {
        let tokens = tokenize_plain_line(line)?;
        if tokens.is_empty() {
            continue;
        }

        match tokens[0].as_str() {
            "graph" => {
                if tokens.len() < 4 {
                    return Err(GraphvizOutputError::InvalidPlainOutput);
                }
                layout.width = parse_plain_f64(&tokens[2])?;
                layout.height = parse_plain_f64(&tokens[3])?;
            }
            "cluster" => {
                if tokens.len() < 7 {
                    return Err(GraphvizOutputError::InvalidPlainOutput);
                }
                let index = parse_plain_usize(&tokens[1])?;
                let parent = match tokens[2].as_str() {
                    "_" => None,
                    value => Some(parse_plain_usize(value)?),
                };
                layout.clusters.insert(
                    index,
                    PlainCluster {
                        parent,
                        bounds: (
                            parse_plain_f64(&tokens[3])?,
                            parse_plain_f64(&tokens[4])?,
                            parse_plain_f64(&tokens[5])?,
                            parse_plain_f64(&tokens[6])?,
                        ),
                    },
                );
            }
            "node" => {
                if tokens.len() < 6 {
                    return Err(GraphvizOutputError::InvalidPlainOutput);
                }
                let id = parse_plain_u32(&tokens[1])?;
                layout.nodes.insert(
                    id,
                    PlainNode {
                        center: (parse_plain_f64(&tokens[2])?, parse_plain_f64(&tokens[3])?),
                        size: (parse_plain_f64(&tokens[4])?, parse_plain_f64(&tokens[5])?),
                    },
                );
            }
            "edge" => {
                if tokens.len() < 4 {
                    return Err(GraphvizOutputError::InvalidPlainOutput);
                }
                let tail = parse_plain_u32(&tokens[1])?;
                let head = parse_plain_u32(&tokens[2])?;
                let count = parse_plain_usize(&tokens[3])?;
                let coords_end = 4 + count.saturating_mul(2);
                if tokens.len() < coords_end {
                    return Err(GraphvizOutputError::InvalidPlainOutput);
                }
                let mut points = Vec::with_capacity(count);
                for index in 0..count {
                    let token_index = 4 + index * 2;
                    points.push((
                        parse_plain_f64(&tokens[token_index])?,
                        parse_plain_f64(&tokens[token_index + 1])?,
                    ));
                }
                let remaining = &tokens[coords_end..];
                let label_position = match remaining.len() {
                    0 => None,
                    3 => Some((
                        parse_plain_f64(&remaining[1])?,
                        parse_plain_f64(&remaining[2])?,
                    )),
                    _ => return Err(GraphvizOutputError::InvalidPlainOutput),
                };
                layout.edges.insert(
                    (tail, head),
                    PlainEdge {
                        points,
                        label_position,
                    },
                );
            }
            "stop" => break,
            _ => {}
        }
    }

    Ok(layout)
}

fn tokenize_plain_line(line: &str) -> Result<Vec<String>, GraphvizOutputError> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut chars = line.chars().peekable();
    let mut in_quotes = false;
    let mut escaping = false;

    while let Some(ch) = chars.next() {
        if in_quotes {
            if escaping {
                current.push(ch);
                escaping = false;
                continue;
            }
            match ch {
                '\\' => escaping = true,
                '"' => {
                    tokens.push(current.clone());
                    current.clear();
                    in_quotes = false;
                }
                _ => current.push(ch),
            }
            continue;
        }

        match ch {
            '"' => {
                if !current.is_empty() {
                    tokens.push(current.clone());
                    current.clear();
                }
                in_quotes = true;
            }
            ch if ch.is_whitespace() => {
                if !current.is_empty() {
                    tokens.push(current.clone());
                    current.clear();
                }
            }
            _ => current.push(ch),
        }
    }

    if escaping || in_quotes {
        return Err(GraphvizOutputError::ParsePlainOutput);
    }
    if !current.is_empty() {
        tokens.push(current);
    }

    Ok(tokens)
}

fn parse_plain_u32(value: &str) -> Result<u32, GraphvizOutputError> {
    value
        .parse::<u32>()
        .map_err(|_| GraphvizOutputError::ParsePlainOutput)
}

fn parse_plain_usize(value: &str) -> Result<usize, GraphvizOutputError> {
    value
        .parse::<usize>()
        .map_err(|_| GraphvizOutputError::ParsePlainOutput)
}

fn parse_plain_f64(value: &str) -> Result<f64, GraphvizOutputError> {
    value
        .parse::<f64>()
        .map_err(|_| GraphvizOutputError::ParsePlainOutput)
}

fn node_similarity(left: &ParsedPlainLayout, right: &ParsedPlainLayout, diagonal: f64) -> f64 {
    let node_ids = left
        .nodes
        .keys()
        .chain(right.nodes.keys())
        .copied()
        .collect::<HashSet<_>>();

    if node_ids.is_empty() {
        return 1.0;
    }

    let total = node_ids
        .into_iter()
        .map(
            |node_id| match (left.nodes.get(&node_id), right.nodes.get(&node_id)) {
                (Some(left_node), Some(right_node)) => {
                    let center_distance = point_distance(left_node.center, right_node.center);
                    clamp01(1.0 - center_distance / diagonal)
                }
                _ => 0.0,
            },
        )
        .sum::<f64>();

    total
        / (left
            .nodes
            .keys()
            .chain(right.nodes.keys())
            .collect::<HashSet<_>>()
            .len()
            .max(1) as f64)
}

fn edge_similarity(left: &ParsedPlainLayout, right: &ParsedPlainLayout, diagonal: f64) -> f64 {
    let edge_ids = left
        .edges
        .keys()
        .chain(right.edges.keys())
        .copied()
        .collect::<HashSet<_>>();

    if edge_ids.is_empty() {
        return 1.0;
    }

    let total = edge_ids
        .into_iter()
        .map(
            |edge_id| match (left.edges.get(&edge_id), right.edges.get(&edge_id)) {
                (Some(left_edge), Some(right_edge)) => {
                    let path_score =
                        polyline_similarity(&left_edge.points, &right_edge.points, diagonal);
                    let label_score = match (left_edge.label_position, right_edge.label_position) {
                        (Some(left_pos), Some(right_pos)) => {
                            clamp01(1.0 - point_distance(left_pos, right_pos) / diagonal)
                        }
                        (None, None) => 1.0,
                        _ => 0.0,
                    };
                    (path_score + label_score) * 0.5
                }
                _ => 0.0,
            },
        )
        .sum::<f64>();

    total
        / (left
            .edges
            .keys()
            .chain(right.edges.keys())
            .collect::<HashSet<_>>()
            .len()
            .max(1) as f64)
}

fn cluster_similarity(left: &ParsedPlainLayout, right: &ParsedPlainLayout) -> f64 {
    let cluster_ids = left
        .clusters
        .keys()
        .chain(right.clusters.keys())
        .copied()
        .collect::<HashSet<_>>();

    if cluster_ids.is_empty() {
        return 1.0;
    }

    let total = cluster_ids
        .into_iter()
        .map(|cluster_id| {
            match (
                left.clusters.get(&cluster_id),
                right.clusters.get(&cluster_id),
            ) {
                (Some(left_cluster), Some(right_cluster)) => {
                    let bounds_score = rect_iou(left_cluster.bounds, right_cluster.bounds);
                    let parent_score = if left_cluster.parent == right_cluster.parent {
                        1.0
                    } else {
                        0.0
                    };
                    bounds_score * 0.8 + parent_score * 0.2
                }
                _ => 0.0,
            }
        })
        .sum::<f64>();

    total
        / (left
            .clusters
            .keys()
            .chain(right.clusters.keys())
            .collect::<HashSet<_>>()
            .len()
            .max(1) as f64)
}

fn polyline_similarity(left: &[(f64, f64)], right: &[(f64, f64)], diagonal: f64) -> f64 {
    if left.is_empty() && right.is_empty() {
        return 1.0;
    }
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }

    let sample_count = 24;
    let left_samples = resample_polyline(left, sample_count);
    let right_samples = resample_polyline(right, sample_count);
    let mean_distance = left_samples
        .iter()
        .zip(right_samples.iter())
        .map(|(left_point, right_point)| point_distance(*left_point, *right_point))
        .sum::<f64>()
        / sample_count as f64;
    let shape_score = clamp01(1.0 - mean_distance / diagonal);
    let length_score = relative_similarity(polyline_length(left), polyline_length(right));
    shape_score * 0.8 + length_score * 0.2
}

fn resample_polyline(points: &[(f64, f64)], samples: usize) -> Vec<(f64, f64)> {
    if points.is_empty() {
        return vec![(0.0, 0.0); samples];
    }
    if points.len() == 1 {
        return vec![points[0]; samples];
    }

    let total_length = polyline_length(points);
    if total_length <= f64::EPSILON {
        return vec![points[0]; samples];
    }

    let mut result = Vec::with_capacity(samples);
    for sample_index in 0..samples {
        let target = if samples == 1 {
            0.0
        } else {
            total_length * sample_index as f64 / (samples - 1) as f64
        };
        result.push(point_at_distance(points, target));
    }
    result
}

fn point_at_distance(points: &[(f64, f64)], target: f64) -> (f64, f64) {
    let mut traversed = 0.0;

    for segment in points.windows(2) {
        let start = segment[0];
        let end = segment[1];
        let length = point_distance(start, end);
        if length <= f64::EPSILON {
            continue;
        }
        if traversed + length >= target {
            let t = (target - traversed) / length;
            return (
                start.0 + (end.0 - start.0) * t,
                start.1 + (end.1 - start.1) * t,
            );
        }
        traversed += length;
    }

    points.last().copied().unwrap_or((0.0, 0.0))
}

fn polyline_length(points: &[(f64, f64)]) -> f64 {
    points
        .windows(2)
        .map(|segment| point_distance(segment[0], segment[1]))
        .sum()
}

fn rect_iou(left: (f64, f64, f64, f64), right: (f64, f64, f64, f64)) -> f64 {
    let overlap_min_x = left.0.max(right.0);
    let overlap_min_y = left.1.max(right.1);
    let overlap_max_x = left.2.min(right.2);
    let overlap_max_y = left.3.min(right.3);
    let overlap_width = (overlap_max_x - overlap_min_x).max(0.0);
    let overlap_height = (overlap_max_y - overlap_min_y).max(0.0);
    let intersection = overlap_width * overlap_height;
    let left_area = (left.2 - left.0).max(0.0) * (left.3 - left.1).max(0.0);
    let right_area = (right.2 - right.0).max(0.0) * (right.3 - right.1).max(0.0);
    let union = left_area + right_area - intersection;

    if union <= f64::EPSILON {
        1.0
    } else {
        clamp01(intersection / union)
    }
}

fn point_distance(left: (f64, f64), right: (f64, f64)) -> f64 {
    (left.0 - right.0).hypot(left.1 - right.1)
}

fn relative_similarity(left: f64, right: f64) -> f64 {
    let denom = left.abs().max(right.abs()).max(1.0);
    clamp01(1.0 - (left - right).abs() / denom)
}

fn clamp01(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}

async fn save_merged_png(graphviz_png: Vec<u8>, screenshot: Screenshot) -> Result<PathBuf, String> {
    let output_path = PathBuf::from(MERGED_PNG_OUTPUT_PATH);

    let graphviz_image =
        image::load_from_memory_with_format(&graphviz_png, image::ImageFormat::Png)
            .map_err(MergePngError::DecodeGraphviz)
            .map_err(|error| error.to_string())?
            .to_rgba8();
    let graphviz_image = image::imageops::resize(
        &graphviz_image,
        scaled_dimension(graphviz_image.width(), GRAPHVIZ_PNG_SCALE),
        scaled_dimension(graphviz_image.height(), GRAPHVIZ_PNG_SCALE),
        image::imageops::FilterType::Lanczos3,
    );
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

fn scaled_dimension(value: u32, factor: f64) -> u32 {
    let scaled = (f64::from(value) * factor).round();

    if scaled < 1.0 {
        1
    } else if scaled > f64::from(u32::MAX) {
        u32::MAX
    } else {
        scaled as u32
    }
}

fn build_clusters(graph: &Graph, cluster_seed: u64, cluster_count: usize) -> Vec<Cluster> {
    #[derive(Clone, Copy)]
    enum ClusterItem {
        Node(u32),
        Cluster(usize),
    }

    struct ClusterSpec {
        parent: Option<usize>,
        items: Vec<ClusterItem>,
    }

    fn collect_cluster_nodes(
        index: usize,
        specs: &[ClusterSpec],
        cache: &mut [Option<Vec<u32>>],
    ) -> Vec<u32> {
        if let Some(nodes) = &cache[index] {
            return nodes.clone();
        }

        let mut nodes = Vec::new();
        for item in &specs[index].items {
            match *item {
                ClusterItem::Node(node) => nodes.push(node),
                ClusterItem::Cluster(child_index) => {
                    nodes.extend(collect_cluster_nodes(child_index, specs, cache));
                }
            }
        }
        nodes.sort_unstable();
        nodes.dedup();
        cache[index] = Some(nodes.clone());
        nodes
    }

    fn pick_items(items: &mut Vec<ClusterItem>, rng: &mut fastrand::Rng) -> Vec<ClusterItem> {
        let selected_count = rng.usize(1..=items.len());
        let mut indices = (0..items.len()).collect::<Vec<_>>();
        rng.shuffle(&mut indices);
        indices.truncate(selected_count);
        indices.sort_unstable();

        let mut selected = Vec::with_capacity(selected_count);
        for index in indices.into_iter().rev() {
            selected.push(items.remove(index));
        }
        selected.reverse();
        selected
    }

    fn count_nodes(items: &[ClusterItem]) -> usize {
        items
            .iter()
            .filter(|item| matches!(item, ClusterItem::Node(_)))
            .count()
    }

    fn count_clusters(items: &[ClusterItem]) -> usize {
        items
            .iter()
            .filter(|item| matches!(item, ClusterItem::Cluster(_)))
            .count()
    }

    fn pick_cluster_contents(
        items: &mut Vec<ClusterItem>,
        rng: &mut fastrand::Rng,
    ) -> Vec<ClusterItem> {
        let node_indices = items
            .iter()
            .enumerate()
            .filter_map(|(index, item)| matches!(item, ClusterItem::Node(_)).then_some(index))
            .collect::<Vec<_>>();
        let cluster_indices = items
            .iter()
            .enumerate()
            .filter_map(|(index, item)| matches!(item, ClusterItem::Cluster(_)).then_some(index))
            .collect::<Vec<_>>();

        if node_indices.is_empty() {
            return pick_items(items, rng);
        }

        let selected_node_count = rng.usize(1..=node_indices.len());
        let mut shuffled_node_indices = node_indices;
        rng.shuffle(&mut shuffled_node_indices);
        shuffled_node_indices.truncate(selected_node_count);

        let selected_cluster_count = if cluster_indices.is_empty() {
            0
        } else if cluster_indices.len() == 1 {
            usize::from(rng.bool())
        } else {
            rng.usize(0..=cluster_indices.len())
        };
        let mut shuffled_cluster_indices = cluster_indices;
        rng.shuffle(&mut shuffled_cluster_indices);
        shuffled_cluster_indices.truncate(selected_cluster_count);

        let mut selected_indices = shuffled_node_indices;
        selected_indices.extend(shuffled_cluster_indices);
        selected_indices.sort_unstable();

        let mut selected = Vec::with_capacity(selected_indices.len());
        for index in selected_indices.into_iter().rev() {
            selected.push(items.remove(index));
        }
        selected.reverse();
        selected
    }

    let clustered_nodes = graph
        .nodes
        .iter()
        .copied()
        .filter(|node| *node != 0)
        .collect::<Vec<_>>();

    if cluster_count == 0 || clustered_nodes.is_empty() {
        return Vec::new();
    }

    let mut rng = fastrand::Rng::with_seed(
        cluster_seed ^ ((cluster_count as u64) << 32) ^ (clustered_nodes.len() as u64),
    );
    let mut root_items = clustered_nodes
        .into_iter()
        .map(ClusterItem::Node)
        .collect::<Vec<_>>();
    let mut specs: Vec<ClusterSpec> = Vec::with_capacity(cluster_count);

    for cluster_index in 0..cluster_count {
        let mut candidates = Vec::with_capacity(cluster_index + 1);
        let root_node_count = count_nodes(&root_items);
        let root_cluster_count = count_clusters(&root_items);
        if root_node_count > 0 || root_cluster_count > 1 {
            candidates.push(None);
        }
        for existing_index in 0..specs.len() {
            let node_count = count_nodes(&specs[existing_index].items);
            let child_count = count_clusters(&specs[existing_index].items);
            if node_count > 0 || child_count > 1 {
                candidates.push(Some(existing_index));
            }
        }

        if candidates.is_empty() {
            if !root_items.is_empty() {
                candidates.push(None);
            }
            for existing_index in 0..specs.len() {
                if !specs[existing_index].items.is_empty() {
                    candidates.push(Some(existing_index));
                }
            }
        }

        let parent = candidates[rng.usize(0..candidates.len())];
        let items = match parent {
            Some(parent_index) => pick_cluster_contents(&mut specs[parent_index].items, &mut rng),
            None => pick_cluster_contents(&mut root_items, &mut rng),
        };

        specs.push(ClusterSpec { parent, items });

        let child_indices = specs[cluster_index]
            .items
            .iter()
            .filter_map(|item| match *item {
                ClusterItem::Cluster(child_index) => Some(child_index),
                ClusterItem::Node(_) => None,
            })
            .collect::<Vec<_>>();
        for child_index in child_indices {
            specs[child_index].parent = Some(cluster_index);
        }

        match parent {
            Some(parent_index) => specs[parent_index]
                .items
                .push(ClusterItem::Cluster(cluster_index)),
            None => root_items.push(ClusterItem::Cluster(cluster_index)),
        }
    }

    let mut cache = vec![None; specs.len()];
    specs
        .iter()
        .enumerate()
        .map(|(index, spec)| {
            let cluster =
                Cluster::new(collect_cluster_nodes(index, &specs, &mut cache)).padding(10.0);
            match spec.parent {
                Some(parent) => cluster.parent(parent),
                None => cluster,
            }
        })
        .collect()
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

fn render_config() -> rust_sugiyama::RenderConfig {
    rust_sugiyama::RenderConfig {
        routing_padding: 4.0,
        bend_penalty: 6.0,
        cluster_padding: 10.0,
        cluster_constraint_iterations: 4,
        cluster_boundary_gap: 8.0,
    }
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

#[cfg(test)]
mod tests {
    use super::compare_plain_layouts;

    const BASE_LAYOUT: &str = r#"graph 1 4 4
cluster 0 _ 0.5 0.5 3.5 3.5 "cluster 0"
node 0 1 3 1 1 A
node 1 3 1 1 1 B
edge 0 1 3 1 3 2 2 3 1 "0 -> 1" 2 2
stop
"#;

    const PERTURBED_LAYOUT: &str = r#"graph 1 4 4
cluster 0 _ 0.55 0.5 3.45 3.45 "cluster 0"
node 0 1 3 1 1 A
node 1 2.85 1.1 1 1 B
edge 0 1 3 1 3 2.1 2.05 2.95 1.05 "0 -> 1" 2.05 1.95
stop
"#;

    #[test]
    fn identical_layouts_score_perfectly() {
        let scores = compare_plain_layouts(BASE_LAYOUT, BASE_LAYOUT).expect("scores");

        assert!((scores.nodes - 1.0).abs() < 1e-9);
        assert!((scores.edges - 1.0).abs() < 1e-9);
        assert!((scores.clusters - 1.0).abs() < 1e-9);
        assert!((scores.overall - 1.0).abs() < 1e-9);
    }

    #[test]
    fn small_perturbations_keep_scores_high() {
        let scores = compare_plain_layouts(BASE_LAYOUT, PERTURBED_LAYOUT).expect("scores");

        assert!(
            scores.nodes < 1.0 && scores.nodes > 0.9,
            "nodes={}",
            scores.nodes
        );
        assert!(
            scores.edges < 1.0 && scores.edges > 0.9,
            "edges={}",
            scores.edges
        );
        assert!(
            scores.clusters < 1.0 && scores.clusters > 0.9,
            "clusters={}",
            scores.clusters
        );
        assert!(
            scores.overall < 1.0 && scores.overall > 0.9,
            "overall={}",
            scores.overall
        );
    }
}
