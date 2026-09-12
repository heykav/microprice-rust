//! # microprice-core
//!
//! Core primitive types for MicroPrice-Rust: integer-tick prices, resting
//! quantities, a validated top-of-book type, queue imbalance, and (as of
//! Phase 3 / Prompt 2) the V1 state discretization engine.
//!
//! Transition estimation and the micro-price solver itself (Phase 5+) live
//! in `microprice-calibration`, not implemented yet — nothing here should
//! be read as a preview of that.
//!
//! No heap allocation happens in any of these primitive calculations, and
//! `unsafe` is forbidden crate-wide.

#![forbid(unsafe_code)]

pub mod book;
pub mod error;
pub mod event;
pub mod imbalance;
pub mod price;
pub mod quantity;
pub mod state;

pub use book::{BookValidationPolicy, SpreadTicks, TopOfBook};
pub use error::MicroPriceError;
pub use event::{BookEvent, SymbolId};
pub use imbalance::Imbalance;
pub use price::PriceTicks;
pub use quantity::Quantity;
pub use state::{ImbalanceBucketing, SpreadBucketing, StateDescription, StateId, StateSpaceConfig};
