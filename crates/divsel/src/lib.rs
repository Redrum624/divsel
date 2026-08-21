//! `divsel` — a native Rust implementation of GIST, a max-min diversification
//! algorithm over a submodular utility (arXiv:2405.18754v3, NeurIPS 2025), selecting
//! a subset `S` maximizing `g(S) + lambda * min-pairwise-distance(S)` with a provable
//! approximation guarantee, in place of heuristics like MMR. Licensed under Apache-2.0.

#![deny(missing_docs)]
#![warn(clippy::all)]

pub mod error;
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
pub use greedy::greedy_independent_set;
pub use metric::Metric;
pub use points::Points;
pub use utility::{Coverage, FacilityLocation, Linear, Utility};
