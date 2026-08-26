//! `divsel` — a native Rust implementation of GIST, a max-min diversification
//! algorithm over a submodular utility (arXiv:2405.18754v3, NeurIPS 2025), selecting
//! a subset `S` maximizing `g(S) + lambda * min-pairwise-distance(S)` with a provable
//! approximation guarantee, in place of heuristics like MMR. Licensed under Apache-2.0.
//!
//! # Example
//!
//! ```
//! use divsel::{gist, GistConfig, Linear, Metric, Points};
//!
//! // Six points on a line, all equally useful, so `f` is decided by diversity.
//! let pts = Points::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 8.0], 1, Metric::Euclidean)?;
//! let mut util = Linear::uniform(pts.n());
//! let cfg = GistConfig { k: 3, ..Default::default() };
//!
//! let out = gist(&pts, &mut util, &cfg)?;
//! assert!(out.selected.len() <= 3);
//! // The identity holds for every `lambda > 0`. At `lambda == 0` the core adds
//! // a literal `0.0` instead of forming the product, which differs from
//! // `0.0 * div` exactly when `div` is infinite; see `GistResult::f_value` and
//! // `docs/CONFORMANCE.md` rule 18.
//! assert_eq!(out.f_value, out.g_value + cfg.lambda * f64::from(out.div));
//! # Ok::<(), divsel::DivselError>(())
//! ```

#![deny(missing_docs)]
#![warn(clippy::all)]

pub mod error;
pub mod gist;
pub mod greedy;
pub mod metric;
pub mod points;
pub mod utility;

// Deterministic test-data helpers, shared by this crate's unit tests, its
// integration tests and its benches. Public so those targets can reach it, but
// not part of the supported API.
#[doc(hidden)]
pub mod testutil;

pub use error::DivselError;
pub use gist::{div, eval_g, gist, thresholds, DiameterMode, GistConfig, GistResult, Stage};
pub use greedy::greedy_independent_set;
pub use metric::Metric;
pub use points::Points;
pub use utility::{Coverage, FacilityLocation, Linear, Utility};
