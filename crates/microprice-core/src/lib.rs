//! # microprice-core
//!
//! Core primitive types for MicroPrice-Rust: integer-tick prices, resting
//! quantities, a validated top-of-book type, and queue imbalance.
//!
//! This crate is Phase 1 of the project (see the repository root README and
//! `docs/model-spec.md`): it deliberately implements *only* the primitives
//! an L1 order book needs to be described unambiguously. State encoding
//! (Phase 3), transition estimation (Phase 5), and the micro-price solver
//! itself live in other crates and other phases — nothing here should be
//! read as a preview of those.
//!
//! No heap allocation happens in any of these primitive calculations, and
//! `unsafe` is forbidden crate-wide.

#![forbid(unsafe_code)]

pub mod book;
pub mod error;
pub mod imbalance;
pub mod price;
pub mod quantity;
pub mod state;

pub use book::{BookValidationPolicy, SpreadTicks, TopOfBook};
pub use error::MicroPriceError;
pub use imbalance::Imbalance;
pub use price::PriceTicks;
pub use quantity::Quantity;
pub use state::StateId;
