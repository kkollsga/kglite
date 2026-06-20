//! Sodir FactMaps REST catalog: dataset stem → (base_url, layer_id).
//!
//! Vendored from `kkollsga/factpages-py` (`datasets.py`, ~v0.1.31).
//! Sodir's FactMaps publishes ~150 datasets under three FeatureServer
//! endpoints; the endpoint is determined by which table the stem
//! lives in. Ported verbatim from the Python `catalog.py`.

use crate::datasets::sodir::error::{Result, SodirError};

/// DataService FeatureServer — geometry layers + non-spatial tables.
pub const DATASERVICE_URL: &str =
    "https://factmaps.sodir.no/api/rest/services/DataService/Data/FeatureServer";
/// FactMaps FeatureServer — pre-filtered / styled display layers.
pub const FACTMAPS_URL: &str =
    "https://factmaps.sodir.no/api/rest/services/Factmaps/FactMapsWGS84/FeatureServer";
/// Which FeatureServer + table family a dataset belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataKind {
    /// DataService geometry layer.
    Layer,
    /// DataService non-spatial table.
    Table,
    /// FactMaps styled display layer.
    Factmaps,
}

impl DataKind {
    pub fn as_str(self) -> &'static str {
        match self {
            DataKind::Layer => "layer",
            DataKind::Table => "table",
            DataKind::Factmaps => "factmaps",
        }
    }
}

/// DataService geometry layers (39 entries).
pub const LAYERS: &[(&str, u32)] = &[
    ("block", 1001),
    ("quadrant", 1002),
    ("sub_area", 1003),
    ("sbm_block", 1004),
    ("sbm_quadrant", 1005),
    ("areastatus", 1100),
    ("structural_elements", 2000),
    ("domes", 2001),
    ("faults_boundaries", 2002),
    ("sediment_boundaries", 2004),
    ("licence", 3000),
    ("licence_area_poly_hst", 3002),
    ("licence_document_area", 3006),
    ("licence_area_count", 3011),
    ("apa_gross", 3102),
    ("apa_open", 3103),
    ("announced_blocks_history", 3104),
    ("announced_history", 3106),
    ("apa_gross_history", 3107),
    ("afex_area", 3200),
    ("afex_area_history", 3201),
    ("business_arrangement_area", 3300),
    ("business_arrangement_history", 3301),
    ("seismic_acquisition", 4000),
    ("seismic_acquisition_poly", 4008),
    ("sbm_sample_point", 4501),
    ("sbm_survey_area", 4502),
    ("sbm_survey_line", 4503),
    ("sbm_survey_sub_area", 4504),
    ("wellbore", 5000),
    ("facility", 6000),
    ("pipeline", 6100),
    ("discovery", 7000),
    ("discovery_map_reference", 7004),
    ("discovery_poly_hst", 7005),
    ("field", 7100),
    ("play", 7800),
    ("sbm_occurrence", 8001),
    ("sbm_play_resource_estimate", 8002),
];

/// DataService non-spatial tables (71 entries).
pub const TABLES: &[(&str, u32)] = &[
    ("company", 1200),
    ("strat_litho", 2100),
    ("strat_litho_wellbore", 2101),
    ("strat_litho_wellbore_core", 2102),
    ("strat_chrono", 2200),
    ("licence_additional_area", 3001),
    ("licence_transfer_hst", 3003),
    ("licence_document", 3005),
    ("licence_licensee_hst", 3007),
    ("licence_operator_hst", 3008),
    ("licence_phase_hst", 3009),
    ("licence_task", 3010),
    ("licensing_activity", 3100),
    ("business_arrangement_operator", 3302),
    ("business_arrangement_licensee_hst", 3304),
    ("business_arrangement_transfer_hst", 3305),
    ("petreg_licence", 3400),
    ("petreg_licence_licensee", 3401),
    ("petreg_licence_message", 3402),
    ("petreg_licence_operator", 3403),
    ("seismic_acquisition_company", 4001),
    ("seismic_acquisition_format", 4002),
    ("seismic_acquisition_fishery", 4003),
    ("seismic_acquisition_for_company", 4004),
    ("seismic_acquisition_licence", 4005),
    ("seismic_acquisition_licences", 4006),
    ("seismic_acquisition_polygon", 4009),
    ("seismic_acquisition_progress", 4011),
    ("seismic_acquisition_scientific", 4012),
    ("seismic_acquisition_vessel", 4013),
    ("seismic_acquisition_weekly_done", 4014),
    ("seismic_acquisition_weekly_plan", 4015),
    ("wellbore_casing", 5001),
    ("wellbore_co2", 5002),
    ("wellbore_core", 5003),
    ("wellbore_core_photo", 5004),
    ("wellbore_core_photo_aggr", 5005),
    ("wellbore_cutting", 5006),
    ("wellbore_document", 5007),
    ("wellbore_dst", 5008),
    ("wellbore_formation_top", 5009),
    ("wellbore_log", 5011),
    ("wellbore_mud", 5012),
    ("wellbore_oil_sample", 5013),
    ("wellbore_paly_slide", 5014),
    ("wellbore_thin_section", 5015),
    ("wellbore_history", 5050),
    ("facility_function", 6001),
    ("tuf", 6200),
    ("tuf_operator_hst", 6201),
    ("tuf_owner_hst", 6202),
    ("discovery_description", 7001),
    ("discovery_extends_into", 7002),
    ("discovery_licensee_hst", 7003),
    ("discovery_operator_hst", 7006),
    ("discovery_owner_hst", 7007),
    ("discovery_reserves", 7008),
    ("field_activity_status_hst", 7101),
    ("field_description", 7102),
    ("field_discoveries_incl_hst", 7103),
    ("field_extends_into", 7104),
    ("field_image", 7106),
    ("field_investment_expected", 7107),
    ("field_licensee_hst", 7108),
    ("field_operator_hst", 7110),
    ("field_owner_hst", 7111),
    ("field_pdo_hst", 7112),
    ("field_reserves", 7113),
    ("field_reserves_company", 7114),
    ("profiles", 7300),
    ("csd_injection", 9001),
];

/// FactMaps styled display layers (39 entries) — pre-filtered views,
/// served from the separate FactMaps FeatureServer.
pub const FACTMAPS_LAYERS: &[(&str, u32)] = &[
    ("wellbore_all", 201),
    ("wellbore_exploration_active", 203),
    ("wellbore_exploration", 204),
    ("wellbore_development", 205),
    ("wellbore_other", 206),
    ("wellbore_co2_factmaps", 207),
    ("facility_in_place", 304),
    ("facility_not_in_place", 306),
    ("facility_all", 307),
    ("pipeline_factmaps", 311),
    ("seismic_pending", 403),
    ("seismic_planned", 404),
    ("seismic_ongoing", 405),
    ("seismic_paused", 406),
    ("seismic_cancelled", 407),
    ("seismic_finished", 421),
    ("em_pending", 409),
    ("em_planned", 410),
    ("em_ongoing", 411),
    ("em_paused", 412),
    ("em_cancelled", 413),
    ("em_finished", 422),
    ("survey_all", 420),
    ("other_survey_pending", 415),
    ("other_survey_planned", 416),
    ("other_survey_ongoing", 417),
    ("other_survey_paused", 418),
    ("other_survey_cancelled", 419),
    ("other_survey_finished", 423),
    ("field_by_status", 502),
    ("discovery_active", 503),
    ("discovery_all", 504),
    ("discovery_history", 505),
    ("play_factmaps", 540),
    ("apa_gross_factmaps", 603),
    ("apa_open_factmaps", 604),
    ("blocks_factmaps", 802),
    ("quadrants_factmaps", 803),
    ("sub_areas_factmaps", 804),
];

/// Look up a dataset stem across all three catalog tables, in the
/// resolution order LAYERS → TABLES → FACTMAPS_LAYERS. Returns the
/// FeatureServer base URL, the layer id, and which family it is.
pub fn lookup(stem: &str) -> Option<(&'static str, u32, DataKind)> {
    if let Some(&(_, id)) = LAYERS.iter().find(|(s, _)| *s == stem) {
        return Some((DATASERVICE_URL, id, DataKind::Layer));
    }
    if let Some(&(_, id)) = TABLES.iter().find(|(s, _)| *s == stem) {
        return Some((DATASERVICE_URL, id, DataKind::Table));
    }
    if let Some(&(_, id)) = FACTMAPS_LAYERS.iter().find(|(s, _)| *s == stem) {
        return Some((FACTMAPS_URL, id, DataKind::Factmaps));
    }
    None
}

/// Map a dataset stem to its `(base_url, layer_id)`.
pub fn resolve(stem: &str) -> Result<(&'static str, u32)> {
    lookup(stem)
        .map(|(url, id, _)| (url, id))
        .ok_or_else(|| SodirError::UnknownStem(stem.to_string()))
}

/// True if the stem is in any catalog table.
pub fn is_known(stem: &str) -> bool {
    lookup(stem).is_some()
}

/// Which family the stem belongs to (`layer` / `table` / `factmaps`).
pub fn kind_of(stem: &str) -> Result<DataKind> {
    lookup(stem)
        .map(|(_, _, k)| k)
        .ok_or_else(|| SodirError::UnknownStem(stem.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn entry_counts() {
        assert_eq!(LAYERS.len(), 39);
        assert_eq!(TABLES.len(), 71);
        assert_eq!(FACTMAPS_LAYERS.len(), 39);
    }

    #[test]
    fn no_stem_appears_in_two_tables() {
        let mut seen = HashSet::new();
        for (s, _) in LAYERS.iter().chain(TABLES).chain(FACTMAPS_LAYERS) {
            assert!(seen.insert(*s), "duplicate stem across catalog tables: {s}");
        }
        assert_eq!(seen.len(), 149);
    }

    #[test]
    fn resolve_routes_to_the_right_endpoint() {
        assert_eq!(resolve("field").unwrap(), (DATASERVICE_URL, 7100));
        assert_eq!(resolve("company").unwrap(), (DATASERVICE_URL, 1200));
        assert_eq!(resolve("wellbore_all").unwrap(), (FACTMAPS_URL, 201));
    }

    #[test]
    fn kind_of_classifies() {
        assert_eq!(kind_of("wellbore").unwrap(), DataKind::Layer);
        assert_eq!(kind_of("company").unwrap(), DataKind::Table);
        assert_eq!(kind_of("discovery_all").unwrap(), DataKind::Factmaps);
    }

    #[test]
    fn unknown_stem_errors() {
        assert!(resolve("not_a_dataset").is_err());
        assert!(!is_known("not_a_dataset"));
        assert!(kind_of("not_a_dataset").is_err());
    }
}
