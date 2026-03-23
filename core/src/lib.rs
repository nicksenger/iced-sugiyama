use std::collections::{BTreeMap, HashMap};
use std::env;

use log::{error, info};
use petgraph::{graph::NodeIndex, stable_graph::StableDiGraph};

type Layout = (Vec<(usize, (f64, f64))>, f64, f64);
type Layouts<T> = Vec<(Vec<(T, (f64, f64))>, f64, f64)>;

const DEFAULT_NODE_SIZE: (f64, f64) = (56.0, 32.0);
const COMPONENT_GAP: f64 = 80.0;
const MINIMUM_LENGTH_DEFAULT: u32 = 1;
const VERTEX_SPACING_DEFAULT: f64 = 10.0;
const DUMMY_VERTICES_DEFAULT: bool = true;
const RANKING_TYPE_DEFAULT: RankingType = RankingType::MinimizeEdgeLength;
const C_MINIMIZATION_DEFAULT: CrossingMinimization = CrossingMinimization::Median;
const TRANSPOSE_DEFAULT: bool = true;
const DUMMY_SIZE_DEFAULT: f64 = 1.0;

const ENV_MINIMUM_LENGTH: &str = "RUST_GRAPH_MIN_LEN";
const ENV_VERTEX_SPACING: &str = "RUST_GRAPH_V_SPACING";
const ENV_DUMMY_VERTICES: &str = "RUST_GRAPH_DUMMIES";
const ENV_RANKING_TYPE: &str = "RUST_GRAPH_R_TYPE";
const ENV_CROSSING_MINIMIZATION: &str = "RUST_GRAPH_CROSS_MIN";
const ENV_TRANSPOSE: &str = "RUST_GRAPH_TRANSPOSE";
const ENV_DUMMY_SIZE: &str = "RUST_GRAPH_DUMMY_SIZE";

macro_rules! read_env {
    ($field:expr, $cb:tt, $env:ident) => {
        #[allow(unused_parens)]
        match env::var($env).map($cb) {
            Ok(Ok(v)) => $field = v,
            Ok(Err(e)) => {
                error!(target: "initialization", "{e}");
            }
            _ => (),
        }
    };
}

/// Used to configure parameters of the graph layout.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// Length between layers.
    pub minimum_length: u32,
    /// The minimum spacing between vertices on the same layer and between
    /// layers.
    pub vertex_spacing: f64,
    /// Whether to include dummy vertices when calculating the layout.
    pub dummy_vertices: bool,
    /// How much space a dummy should take up, as a multiplier of the
    /// [`Self::vertex_spacing`].
    pub dummy_size: f64,
    /// Defines how vertices are placed vertically.
    pub ranking_type: RankingType,
    /// Which heuristic to use when minimizing edge crossings.
    pub c_minimization: CrossingMinimization,
    /// Whether to attempt to further reduce crossings by swapping vertices in a
    /// layer. This may increase runtime significantly.
    pub transpose: bool,
}

impl Config {
    /// Read in configuration values from environment variables.
    pub fn new_from_env() -> Self {
        let mut config = Self::default();

        let parse_bool = |x: String| match x.as_str() {
            "y" => Ok(true),
            "n" => Ok(false),
            v => Err(format!("Invalid argument for dummy vertex env: {v}")),
        };

        read_env!(
            config.minimum_length,
            (|x| x.parse::<u32>()),
            ENV_MINIMUM_LENGTH
        );
        read_env!(
            config.c_minimization,
            (TryFrom::try_from),
            ENV_CROSSING_MINIMIZATION
        );
        read_env!(config.ranking_type, (TryFrom::try_from), ENV_RANKING_TYPE);
        read_env!(
            config.vertex_spacing,
            (|x| x.parse::<f64>()),
            ENV_VERTEX_SPACING
        );
        read_env!(config.dummy_vertices, parse_bool, ENV_DUMMY_VERTICES);
        read_env!(config.dummy_size, (|x| x.parse::<f64>()), ENV_DUMMY_SIZE);
        read_env!(config.transpose, parse_bool, ENV_TRANSPOSE);

        config
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            minimum_length: MINIMUM_LENGTH_DEFAULT,
            vertex_spacing: VERTEX_SPACING_DEFAULT,
            dummy_vertices: DUMMY_VERTICES_DEFAULT,
            dummy_size: DUMMY_SIZE_DEFAULT,
            ranking_type: RANKING_TYPE_DEFAULT,
            c_minimization: C_MINIMIZATION_DEFAULT,
            transpose: TRANSPOSE_DEFAULT,
        }
    }
}

/// Defines the Ranking type, i.e. how vertices are placed on each layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RankingType {
    Original,
    MinimizeEdgeLength,
    Up,
    Down,
}

impl TryFrom<String> for RankingType {
    type Error = String;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        match value.as_str() {
            "original" => Ok(Self::Original),
            "minimize" => Ok(Self::MinimizeEdgeLength),
            "up" => Ok(Self::Up),
            "down" => Ok(Self::Down),
            s => Err(format!("invalid value for ranking type: {s}")),
        }
    }
}

impl From<RankingType> for &'static str {
    fn from(value: RankingType) -> Self {
        match value {
            RankingType::Up => "up",
            RankingType::Down => "down",
            RankingType::Original => "original",
            RankingType::MinimizeEdgeLength => "minimize",
        }
    }
}

/// Defines the heuristic used for crossing minimization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CrossingMinimization {
    Barycenter,
    Median,
}

impl TryFrom<String> for CrossingMinimization {
    type Error = String;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        match value.as_str() {
            "barycenter" => Ok(Self::Barycenter),
            "median" => Ok(Self::Median),
            s => Err(format!("invalid value for crossing minimization: {s}")),
        }
    }
}

impl From<CrossingMinimization> for &'static str {
    fn from(value: CrossingMinimization) -> Self {
        match value {
            CrossingMinimization::Median => "median",
            CrossingMinimization::Barycenter => "barycenter",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderConfig {
    pub routing_padding: f64,
    pub bend_penalty: f64,
    pub cluster_padding: f64,
    pub cluster_constraint_iterations: usize,
    pub cluster_boundary_gap: f64,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            routing_padding: 2.0,
            bend_penalty: 8.0,
            cluster_padding: 8.0,
            cluster_constraint_iterations: 4,
            cluster_boundary_gap: 6.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Cluster {
    nodes: Vec<u32>,
    padding: Option<f64>,
    parent: Option<usize>,
}

impl Cluster {
    pub fn new(nodes: Vec<u32>) -> Self {
        Self {
            nodes,
            padding: None,
            parent: None,
        }
    }

    pub fn padding(self, padding: f64) -> Self {
        Self {
            padding: Some(padding),
            ..self
        }
    }

    pub fn parent(self, parent: usize) -> Self {
        Self {
            parent: Some(parent),
            ..self
        }
    }

    pub fn nodes(&self) -> &[u32] {
        &self.nodes
    }

    pub fn padding_value(&self) -> Option<f64> {
        self.padding
    }

    pub fn parent_index(&self) -> Option<usize> {
        self.parent
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EdgeLayout {
    index: usize,
    points: Vec<(f64, f64)>,
    curve_points: Vec<(f64, f64)>,
    label: Option<String>,
    label_position: Option<(f64, f64)>,
}

impl EdgeLayout {
    pub fn index(&self) -> usize {
        self.index
    }

    pub fn points(&self) -> &[(f64, f64)] {
        &self.points
    }

    pub fn curve_points(&self) -> &[(f64, f64)] {
        &self.curve_points
    }

    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    pub fn label_position(&self) -> Option<(f64, f64)> {
        self.label_position
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClusterLayout {
    index: usize,
    parent: Option<usize>,
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
}

impl ClusterLayout {
    pub fn index(&self) -> usize {
        self.index
    }

    pub fn parent(&self) -> Option<usize> {
        self.parent
    }

    pub fn min_x(&self) -> f64 {
        self.min_x
    }

    pub fn min_y(&self) -> f64 {
        self.min_y
    }

    pub fn max_x(&self) -> f64 {
        self.max_x
    }

    pub fn max_y(&self) -> f64 {
        self.max_y
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GraphLayout {
    max_x: f64,
    max_y: f64,
    coords: BTreeMap<usize, (f64, f64)>,
    edges: Vec<EdgeLayout>,
    clusters: Vec<ClusterLayout>,
}

impl GraphLayout {
    fn empty() -> Self {
        Self {
            max_x: 0.0,
            max_y: 1.0,
            coords: BTreeMap::new(),
            edges: Vec::new(),
            clusters: Vec::new(),
        }
    }

    pub fn max_x(&self) -> f64 {
        self.max_x
    }

    pub fn max_y(&self) -> f64 {
        self.max_y
    }

    pub fn position(&self, index: usize) -> Option<(f64, f64)> {
        self.coords.get(&index).copied()
    }

    pub fn edges(&self) -> &[EdgeLayout] {
        &self.edges
    }

    pub fn clusters(&self) -> &[ClusterLayout] {
        &self.clusters
    }
}

pub fn layout_graph(
    nodes: &[u32],
    edges: &[(u32, u32)],
    config: &Config,
    node_size: impl Fn(u32) -> (f64, f64),
    edge_label: impl Fn(usize, (u32, u32)) -> Option<String>,
    clusters: &[Cluster],
    render_config: &RenderConfig,
) -> GraphLayout {
    GraphLayout::empty() // TODO
}
