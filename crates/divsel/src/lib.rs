//! `divsel` — a native Rust implementation of GIST, a max-min diversification
//! algorithm over a submodular utility (arXiv:2405.18754v3, NeurIPS 2025), selecting
//! a subset `S` maximizing `g(S) + lambda * min-pairwise-distance(S)` with a provable
//! approximation guarantee, in place of heuristics like MMR. Licensed under Apache-2.0.

#![deny(missing_docs)]
#![warn(clippy::all)]

pub mod error;
pub mod metric;
pub mod points;

pub use error::DivselError;
pub use metric::Metric;
pub use points::Points;
