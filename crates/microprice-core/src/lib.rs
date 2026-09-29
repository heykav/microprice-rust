//! # microprice-core
//!
//! Core primitive types for MicroPrice-Rust: integer-tick prices, resting
//! quantities, a validated top-of-book type, queue imbalance, and the state
//! discretization engine. Transition estimation and the micro-price solver
//! live in `microprice-calibration`.
//!
//! ```
//! use microprice_core::{
//!     BookValidationPolicy, Imbalance, ImbalanceBucketing, PriceTicks, Quantity,
//!     SpreadBucketing, StateSpaceConfig, TopOfBook,
//! };
//!
//! let book = TopOfBook::new(
//!     PriceTicks(10_000),
//!     Quantity(300),
//!     PriceTicks(10_001),
//!     Quantity(100),
//!     BookValidationPolicy::RejectCrossedAndLocked,
//! )?;
//! // I = Qb / (Qb + Qa) = 300 / 400
//! let imbalance = Imbalance::compute(book.bid_qty, book.ask_qty)?;
//! assert_eq!(imbalance.value(), 0.75);
//!
//! // 4 imbalance buckets x spread buckets {1}, {2}, {3+} ticks = 12 states.
//! let space = StateSpaceConfig::new(
//!     ImbalanceBucketing::new(4)?,
//!     SpreadBucketing::new(vec![1, 2])?,
//! );
//! let state = space.encode(&book)?;
//! assert_eq!(state.0, 3); // spread bucket 0, imbalance bucket floor(0.75 * 4) = 3
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
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
