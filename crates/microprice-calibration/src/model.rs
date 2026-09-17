//! The trained model artifact: a calibrated `G*` vector plus enough
//! metadata to reconstruct the `StateSpaceConfig` it was calibrated
//! against, serializable, and an allocation-free `predict()`.
//!
//! Serialization deliberately lives here, not in `microprice-core`: core's
//! whole design principle since Phase 1 is "only the primitives, minimal
//! dependencies" (it has exactly one dependency, `thiserror`), so a
//! `MicroPriceModel` reconstructs its `StateSpaceConfig` from plain
//! primitive fields (`num_imbalance_buckets: u32`,
//! `spread_bucket_bounds_ticks: Vec<i64>`) rather than requiring core's
//! types to derive `Serialize`/`Deserialize` themselves.

use std::fs;
use std::path::Path;

use microprice_core::{
    Imbalance, ImbalanceBucketing, SpreadBucketing, StateSpaceConfig, TopOfBook,
};
use serde::{Deserialize, Serialize};

use crate::error::CalibrationError;

/// Bumped 1 -> 2 when [`MicroPriceModel`] gained its per-state `p_up`
/// vector. A v1 artifact has no `P(up)` in it to deserialize, and this
/// type's own contract is that a model file is never silently misread, so
/// v1 artifacts fail `load` with a clear message rather than loading with
/// a fabricated or defaulted `p_up`.
pub const SCHEMA_VERSION: u32 = 2;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelMetadata {
    pub schema_version: u32,
    pub symbol_id: u32,
    pub num_imbalance_buckets: u32,
    pub spread_bucket_bounds_ticks: Vec<i64>,
    pub smoothing_alpha: f64,
    pub training_observations: u64,
}

/// A calibrated, immutable micro-price model. `Send + Sync` (all fields
/// are plain owned data with no interior mutability), so it can be shared
/// across threads behind an `Arc` with no locking on the inference path.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MicroPriceModel {
    metadata: ModelMetadata,
    g_star: Vec<f64>,
    /// Per-state `P(next move is up | next move is directional)`; `None`
    /// where training never observed a directional move out of that state.
    /// Carried alongside `g_star` because it is the one quantity `g_star`
    /// provably cannot express — `g_star` is a signed tick-magnitude
    /// expectation, and no rearrangement of a sum of signed deltas
    /// recovers the up/down split. See `crate::estimator`'s module docs.
    p_up: Vec<Option<f64>>,
    visits: Vec<u64>,
}

/// A single prediction's full detail — more than one number, per the
/// project brief's explicit instruction not to return just a price.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MicroPriceEstimate {
    pub mid_ticks: f64,
    pub weighted_mid_ticks: f64,
    pub microprice_ticks: f64,
    pub adjustment_ticks: f64,
    pub state_id: u32,
    pub state_observations: u64,
    /// `P(next move is up | next move is directional)` for this book's
    /// state, or `None` where training never saw a directional move out of
    /// it. The only genuinely *probabilistic* field here — `adjustment_ticks`
    /// is a signed tick magnitude and is not one, which is why the Brier
    /// score has to read this and not that.
    pub p_up: Option<f64>,
}

impl MicroPriceModel {
    pub fn new(
        metadata: ModelMetadata,
        g_star: Vec<f64>,
        p_up: Vec<Option<f64>>,
        visits: Vec<u64>,
    ) -> Self {
        MicroPriceModel {
            metadata,
            g_star,
            p_up,
            visits,
        }
    }

    pub fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    /// The calibrated `G*` adjustment vector, one entry per state — for
    /// inspection/reporting (e.g. `microprice-cli`'s `inspect`/`benchmark`
    /// subcommands). Not used on the `predict` hot path itself, which
    /// indexes `self.g_star` directly.
    pub fn g_star(&self) -> &[f64] {
        &self.g_star
    }

    /// Per-state directional-up probabilities — for inspection/reporting,
    /// mirroring [`Self::g_star`]. Not used on the `predict` hot path,
    /// which indexes `self.p_up` directly.
    pub fn p_up(&self) -> &[Option<f64>] {
        &self.p_up
    }

    /// Per-state training observation counts — for inspection/reporting.
    pub fn visits(&self) -> &[u64] {
        &self.visits
    }

    /// Reconstructs the [`StateSpaceConfig`] this model was calibrated
    /// against, from its own metadata — public so callers that need to
    /// `decode` a state (e.g. `microprice-cli`'s `visualize` subcommand,
    /// mapping each `g_star` entry back to an imbalance/spread range for
    /// plotting) don't have to duplicate the bucket-reconstruction logic
    /// `predict`/`predict_batch` already contain.
    pub fn state_space(&self) -> Result<StateSpaceConfig, CalibrationError> {
        let imbalance = ImbalanceBucketing::new(self.metadata.num_imbalance_buckets)
            .map_err(CalibrationError::from)?;
        let spread = SpreadBucketing::new(self.metadata.spread_bucket_bounds_ticks.clone())
            .map_err(CalibrationError::from)?;
        Ok(StateSpaceConfig::new(imbalance, spread))
    }

    /// Allocation-free (aside from the one-time `StateSpaceConfig`
    /// reconstruction — see the note on `state_space()` above the real
    /// hot-path concern is `encode` + array lookups, both `O(1)`/alloc-free):
    /// encode the book's state, look up the calibrated adjustment, and
    /// compute the micro-price.
    pub fn predict(&self, book: &TopOfBook) -> Result<MicroPriceEstimate, CalibrationError> {
        let state_space = self.state_space()?;
        let state = state_space.encode(book)?;
        let mid_ticks = book.mid_price_ticks() as f64;

        let imbalance = Imbalance::compute(book.bid_qty, book.ask_qty)?;
        let i = imbalance.value();
        let weighted_mid_ticks = book.ask_price.0 as f64 * i + book.bid_price.0 as f64 * (1.0 - i);

        let idx = state.0 as usize;
        let adjustment_ticks = self.g_star[idx];
        Ok(MicroPriceEstimate {
            mid_ticks,
            weighted_mid_ticks,
            microprice_ticks: mid_ticks + adjustment_ticks,
            adjustment_ticks,
            state_id: state.0,
            state_observations: self.visits[idx],
            p_up: self.p_up[idx],
        })
    }

    /// Batch prediction into a caller-provided output slice — no
    /// per-element allocation.
    pub fn predict_batch(
        &self,
        books: &[TopOfBook],
        output: &mut [MicroPriceEstimate],
    ) -> Result<(), CalibrationError> {
        if books.len() != output.len() {
            return Err(CalibrationError::DimensionMismatch {
                expected: books.len(),
                actual: output.len(),
            });
        }
        let state_space = self.state_space()?;
        for (book, out) in books.iter().zip(output.iter_mut()) {
            let state = state_space.encode(book)?;
            let mid_ticks = book.mid_price_ticks() as f64;
            let imbalance = Imbalance::compute(book.bid_qty, book.ask_qty)?;
            let i = imbalance.value();
            let weighted_mid_ticks =
                book.ask_price.0 as f64 * i + book.bid_price.0 as f64 * (1.0 - i);
            let idx = state.0 as usize;
            let adjustment_ticks = self.g_star[idx];
            *out = MicroPriceEstimate {
                mid_ticks,
                weighted_mid_ticks,
                microprice_ticks: mid_ticks + adjustment_ticks,
                adjustment_ticks,
                state_id: state.0,
                state_observations: self.visits[idx],
                p_up: self.p_up[idx],
            };
        }
        Ok(())
    }

    /// Saves the model as bincode (`<path>`) plus a human-readable JSON
    /// metadata dump (`<path>.json`) — matching the project brief's
    /// `model.bin` + `model.json` pattern.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), CalibrationError> {
        let path = path.as_ref();
        let bytes = bincode::serde::encode_to_vec(self, bincode::config::standard())
            .map_err(|e| CalibrationError::Serialization(e.to_string()))?;
        fs::write(path, bytes).map_err(|e| CalibrationError::Io(e.to_string()))?;

        let json = serde_json::to_string_pretty(&self.metadata)
            .map_err(|e| CalibrationError::Serialization(e.to_string()))?;
        let json_path = path.with_extension(match path.extension() {
            Some(ext) => format!("{}.json", ext.to_string_lossy()),
            None => "json".to_string(),
        });
        fs::write(json_path, json).map_err(|e| CalibrationError::Io(e.to_string()))?;
        Ok(())
    }

    /// Loads and **validates** a model artifact — dimensions must be
    /// self-consistent, every `g_star` value finite, every `p_up` a real
    /// probability in `[0, 1]`, and the schema version recognized, per the
    /// project brief's explicit "model files must not blindly trust
    /// serialized content" requirement.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, CalibrationError> {
        let bytes = fs::read(path.as_ref()).map_err(|e| CalibrationError::Io(e.to_string()))?;
        let (model, _): (MicroPriceModel, usize) =
            bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
                .map_err(|e| CalibrationError::Serialization(e.to_string()))?;
        model.validate()?;
        Ok(model)
    }

    fn validate(&self) -> Result<(), CalibrationError> {
        if self.metadata.schema_version != SCHEMA_VERSION {
            return Err(CalibrationError::InvalidModelArtifact {
                reason: format!(
                    "unsupported schema_version {} (this build supports {})",
                    self.metadata.schema_version, SCHEMA_VERSION
                ),
            });
        }
        let state_space = self.state_space()?;
        let expected = state_space.state_count() as usize;
        if self.g_star.len() != expected {
            return Err(CalibrationError::InvalidModelArtifact {
                reason: format!(
                    "g_star has {} entries, expected {expected} for the declared state space",
                    self.g_star.len()
                ),
            });
        }
        if self.visits.len() != expected {
            return Err(CalibrationError::InvalidModelArtifact {
                reason: format!(
                    "visits has {} entries, expected {expected}",
                    self.visits.len()
                ),
            });
        }
        if self.p_up.len() != expected {
            return Err(CalibrationError::InvalidModelArtifact {
                reason: format!("p_up has {} entries, expected {expected}", self.p_up.len()),
            });
        }
        if let Some(bad) = self.g_star.iter().find(|v| !v.is_finite()) {
            return Err(CalibrationError::InvalidModelArtifact {
                reason: format!("g_star contains a non-finite value: {bad}"),
            });
        }
        // `!v.is_finite()` also catches NaN, which would otherwise slip
        // through both range comparisons below (every NaN comparison is
        // false, so neither `< 0.0` nor `> 1.0` would fire).
        if let Some(bad) = self
            .p_up
            .iter()
            .flatten()
            .find(|v| !v.is_finite() || **v < 0.0 || **v > 1.0)
        {
            return Err(CalibrationError::InvalidModelArtifact {
                reason: format!("p_up contains a value outside [0, 1]: {bad}"),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use microprice_core::{BookValidationPolicy, PriceTicks, Quantity};

    fn toy_model() -> MicroPriceModel {
        MicroPriceModel::new(
            ModelMetadata {
                schema_version: SCHEMA_VERSION,
                symbol_id: 1,
                num_imbalance_buckets: 2,
                spread_bucket_bounds_ticks: vec![],
                smoothing_alpha: 0.0,
                training_observations: 100,
            },
            vec![0.2, 0.225],
            vec![Some(0.6), None],
            vec![100, 50],
        )
    }

    /// Saves a deliberately-corrupted `toy_model()` to a unique temp path
    /// and returns what `load` says about it. Keeps the corruption
    /// one-line at each call site, so the test names read as the property
    /// under test.
    fn load_corrupted(tag: &str, corrupt: impl FnOnce(&mut MicroPriceModel)) -> CalibrationError {
        let mut model = toy_model();
        corrupt(&mut model);
        let dir =
            std::env::temp_dir().join(format!("microprice-test-{tag}-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad.bin");
        model.save(&path).unwrap();
        let err = MicroPriceModel::load(&path).unwrap_err();
        fs::remove_dir_all(&dir).ok();
        err
    }

    fn book(bid: i64, bid_qty: u64, ask: i64, ask_qty: u64) -> TopOfBook {
        TopOfBook::new(
            PriceTicks(bid),
            Quantity(bid_qty),
            PriceTicks(ask),
            Quantity(ask_qty),
            BookValidationPolicy::RejectCrossedAndLocked,
        )
        .unwrap()
    }

    #[test]
    fn predict_computes_microprice_as_mid_plus_adjustment() {
        let model = toy_model();
        let b = book(10000, 100, 10002, 100); // balanced -> imbalance 0.5 -> bucket 1
        let est = model.predict(&b).unwrap();
        assert_eq!(est.mid_ticks, 10001.0);
        assert_eq!(est.adjustment_ticks, 0.225); // state 1
        assert_eq!(est.microprice_ticks, 10001.225);
        assert_eq!(est.state_observations, 50);
        // State 1 is the one trained with no directional evidence, so its
        // probability is absent rather than invented.
        assert_eq!(est.p_up, None);
    }

    #[test]
    fn predict_carries_the_states_directional_probability() {
        let model = toy_model();
        // Qb=100, Qa=900 -> I=0.1 -> bucket 0, which does have evidence.
        let est = model.predict(&book(10000, 100, 10002, 900)).unwrap();
        assert_eq!(est.state_id, 0);
        assert_eq!(est.p_up, Some(0.6));
        assert_eq!(model.p_up(), &[Some(0.6), None]);
    }

    #[test]
    fn predict_weighted_mid_matches_the_documented_formula() {
        let model = toy_model();
        let b = book(10000, 300, 10002, 100); // Qb=300, Qa=100
        let est = model.predict(&b).unwrap();
        // WeightedMid = Pa*Qb/(Qb+Qa) + Pb*Qa/(Qb+Qa)
        let expected = 10002.0 * 300.0 / 400.0 + 10000.0 * 100.0 / 400.0;
        assert!((est.weighted_mid_ticks - expected).abs() < 1e-9);
    }

    #[test]
    fn predict_batch_matches_scalar_predict_for_every_book() {
        let model = toy_model();
        let books = vec![
            book(10000, 900, 10002, 100),
            book(10000, 100, 10002, 900),
            book(10000, 500, 10002, 500),
        ];
        let mut out = vec![
            MicroPriceEstimate {
                mid_ticks: 0.0,
                weighted_mid_ticks: 0.0,
                microprice_ticks: 0.0,
                adjustment_ticks: 0.0,
                state_id: 0,
                state_observations: 0,
                p_up: None,
            };
            3
        ];
        model.predict_batch(&books, &mut out).unwrap();
        for (book, expected) in books.iter().zip(out.iter()) {
            let scalar = model.predict(book).unwrap();
            assert_eq!(scalar, *expected);
        }
    }

    #[test]
    fn predict_batch_rejects_mismatched_lengths() {
        let model = toy_model();
        let books = vec![book(10000, 500, 10002, 500)];
        let mut out = vec![];
        assert!(matches!(
            model.predict_batch(&books, &mut out),
            Err(CalibrationError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn save_and_load_round_trips_exactly() {
        let model = toy_model();
        let dir = std::env::temp_dir().join(format!("microprice-test-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.bin");
        model.save(&path).unwrap();
        let loaded = MicroPriceModel::load(&path).unwrap();
        assert_eq!(model, loaded);

        let json_path = dir.join("model.bin.json");
        assert!(json_path.exists());
        let json = fs::read_to_string(&json_path).unwrap();
        assert!(json.contains("\"schema_version\""));

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_rejects_a_dimension_mismatched_artifact() {
        let mut model = toy_model();
        model.g_star.push(0.0); // now 3 entries but state space declares 2
        let dir = std::env::temp_dir().join(format!("microprice-test-bad-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad.bin");
        model.save(&path).unwrap();
        let result = MicroPriceModel::load(&path);
        assert!(matches!(
            result,
            Err(CalibrationError::InvalidModelArtifact { .. })
        ));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_rejects_an_unsupported_schema_version() {
        let mut model = toy_model();
        model.metadata.schema_version = 999;
        let dir =
            std::env::temp_dir().join(format!("microprice-test-schema-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad_schema.bin");
        model.save(&path).unwrap();
        let result = MicroPriceModel::load(&path);
        assert!(matches!(
            result,
            Err(CalibrationError::InvalidModelArtifact { .. })
        ));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_rejects_a_non_finite_g_star_entry() {
        let mut model = toy_model();
        model.g_star[0] = f64::NAN;
        let dir = std::env::temp_dir().join(format!("microprice-test-nan-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad_nan.bin");
        model.save(&path).unwrap();
        let result = MicroPriceModel::load(&path);
        assert!(matches!(
            result,
            Err(CalibrationError::InvalidModelArtifact { .. })
        ));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_rejects_a_p_up_outside_the_unit_interval() {
        assert!(matches!(
            load_corrupted("pup-range", |m| m.p_up[0] = Some(1.5)),
            CalibrationError::InvalidModelArtifact { .. }
        ));
        assert!(matches!(
            load_corrupted("pup-negative", |m| m.p_up[0] = Some(-0.01)),
            CalibrationError::InvalidModelArtifact { .. }
        ));
    }

    #[test]
    fn load_rejects_a_nan_p_up() {
        // NaN is the trap here: every comparison against it is false, so a
        // naive `v < 0.0 || v > 1.0` range check would wave it straight
        // through. The finiteness check has to come first.
        assert!(matches!(
            load_corrupted("pup-nan", |m| m.p_up[0] = Some(f64::NAN)),
            CalibrationError::InvalidModelArtifact { .. }
        ));
    }

    #[test]
    fn load_rejects_a_dimension_mismatched_p_up() {
        assert!(matches!(
            load_corrupted("pup-dim", |m| m.p_up.push(None)),
            CalibrationError::InvalidModelArtifact { .. }
        ));
    }

    #[test]
    fn load_rejects_a_model_written_before_p_up_existed() {
        // A schema_version-1 artifact: correct dimensions, finite g_star,
        // but no p_up at all. It must be refused, not loaded with a
        // guess at the directional probability.
        assert!(matches!(
            load_corrupted("schema1", |m| {
                m.metadata.schema_version = 1;
                m.p_up = vec![];
            }),
            CalibrationError::InvalidModelArtifact { .. }
        ));
    }
}
