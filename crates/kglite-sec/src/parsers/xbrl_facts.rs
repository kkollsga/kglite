//! Parser for SEC's XBRL company-facts JSON.
//!
//! `data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json` returns every
//! tagged XBRL fact a company has reported, grouped by taxonomy
//! (`us-gaap`, `dei`, `srt`, …) → concept → unit → list of facts.
//!
//! JSON shape:
//!
//! ```json
//! {
//!   "cik": 320193,
//!   "entityName": "Apple Inc.",
//!   "facts": {
//!     "us-gaap": {
//!       "Revenues": {
//!         "label": "Revenues",
//!         "units": {
//!           "USD": [
//!             {"end": "2023-09-30", "val": 383285000000,
//!              "fy": 2023, "fp": "FY", "form": "10-K",
//!              "accn": "0000320193-23-000106", "frame": "CY2023"},
//!             ...
//!           ]
//!         }
//!       }
//!     }
//!   }
//! }
//! ```
//!
//! Each leaf object is one `XbrlFact`. The `frame` field, when
//! present, is SEC's canonical period label (`CY2023`, `CY2023Q3I`).

use serde::Deserialize;

/// One XBRL fact — a single tagged numeric value for one period.
#[derive(Debug, Clone, PartialEq)]
pub struct XbrlFact {
    /// Taxonomy + concept, e.g. "us-gaap:Revenues".
    pub tag: String,
    /// Reporting unit ("USD", "shares", "USD/shares", "pure", …).
    pub unit: String,
    /// The numeric value.
    pub value: f64,
    /// Period start (empty for instant facts like balance-sheet items).
    pub period_start: String,
    /// Period end (always present).
    pub period_end: String,
    /// Fiscal year.
    pub fiscal_year: i64,
    /// Fiscal period ("FY", "Q1", "Q2", "Q3").
    pub fiscal_period: String,
    /// Source form ("10-K", "10-Q", "8-K", …).
    pub form: String,
    /// Accession number of the filing this fact came from.
    pub accession: String,
    /// SEC's canonical frame label, e.g. "CY2023" or "CY2023Q3I".
    /// Empty when the fact isn't frame-aligned.
    pub frame: String,
}

/// `facts` → taxonomy name → concept name → `Concept`.
type FactsMap = std::collections::HashMap<String, std::collections::HashMap<String, Concept>>;

#[derive(Deserialize)]
struct CompanyFacts {
    #[serde(default)]
    facts: FactsMap,
}

#[derive(Deserialize)]
struct Concept {
    #[serde(default)]
    units: std::collections::HashMap<String, Vec<RawFact>>,
}

#[derive(Deserialize)]
struct RawFact {
    #[serde(default)]
    start: String,
    #[serde(default)]
    end: String,
    #[serde(default)]
    val: f64,
    // `fy` and `fp` are explicitly `null` for some facts in the
    // real company-facts feed — Option tolerates both missing and
    // null (a bare `#[serde(default)] i64` rejects explicit null).
    #[serde(default)]
    fy: Option<i64>,
    #[serde(default)]
    fp: Option<String>,
    #[serde(default)]
    form: String,
    #[serde(default)]
    accn: String,
    #[serde(default)]
    frame: String,
}

/// Parse a company-facts JSON document into a flat list of facts.
///
/// `tag_whitelist`: if non-empty, only concepts whose name is in the
/// set are kept (e.g. ["Revenues", "NetIncomeLoss", "Assets"]).
/// Empty whitelist keeps every concept — but a typical company-facts
/// document has 500-2000 concepts, so callers usually pass a
/// whitelist to keep `metric_fact.csv` to the headline line-items.
pub fn parse_company_facts(json: &str, tag_whitelist: &[&str]) -> Result<Vec<XbrlFact>, String> {
    let parsed: CompanyFacts =
        serde_json::from_str(json).map_err(|e| format!("companyfacts JSON: {e}"))?;

    let mut out = Vec::new();
    for (taxonomy, concepts) in &parsed.facts {
        for (concept, c) in concepts {
            if !tag_whitelist.is_empty() && !tag_whitelist.contains(&concept.as_str()) {
                continue;
            }
            let tag = format!("{taxonomy}:{concept}");
            for (unit, facts) in &c.units {
                for f in facts {
                    out.push(XbrlFact {
                        tag: tag.clone(),
                        unit: unit.clone(),
                        value: f.val,
                        period_start: f.start.clone(),
                        period_end: f.end.clone(),
                        fiscal_year: f.fy.unwrap_or(0),
                        fiscal_period: f.fp.clone().unwrap_or_default(),
                        form: f.form.clone(),
                        accession: f.accn.clone(),
                        frame: f.frame.clone(),
                    });
                }
            }
        }
    }
    Ok(out)
}

/// The headline income-statement / balance-sheet / cash-flow concepts
/// a typical analyst wants. Used as the default whitelist so
/// `metric_fact.csv` stays focused (a full company-facts document has
/// thousands of niche concepts).
pub const DEFAULT_FINANCIAL_TAGS: &[&str] = &[
    // Income statement
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "CostOfRevenue",
    "GrossProfit",
    "OperatingIncomeLoss",
    "NetIncomeLoss",
    "EarningsPerShareBasic",
    "EarningsPerShareDiluted",
    "ResearchAndDevelopmentExpense",
    "SellingGeneralAndAdministrativeExpense",
    // Balance sheet
    "Assets",
    "AssetsCurrent",
    "Liabilities",
    "LiabilitiesCurrent",
    "StockholdersEquity",
    "CashAndCashEquivalentsAtCarryingValue",
    "LongTermDebtNoncurrent",
    "RetainedEarningsAccumulatedDeficit",
    // Cash flow
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInInvestingActivities",
    "NetCashProvidedByUsedInFinancingActivities",
    // Shares
    "CommonStockSharesOutstanding",
    "WeightedAverageNumberOfSharesOutstandingBasic",
    "WeightedAverageNumberOfDilutedSharesOutstanding",
];

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"{
        "cik": 320193,
        "entityName": "Apple Inc.",
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "label": "Revenues",
                    "units": {
                        "USD": [
                            {"start": "2022-10-01", "end": "2023-09-30", "val": 383285000000,
                             "fy": 2023, "fp": "FY", "form": "10-K",
                             "accn": "0000320193-23-000106", "frame": "CY2023"},
                            {"start": "2023-07-01", "end": "2023-09-30", "val": 89498000000,
                             "fy": 2023, "fp": "Q4", "form": "10-K",
                             "accn": "0000320193-23-000106"}
                        ]
                    }
                },
                "Assets": {
                    "label": "Assets",
                    "units": {
                        "USD": [
                            {"end": "2023-09-30", "val": 352583000000,
                             "fy": 2023, "fp": "FY", "form": "10-K",
                             "accn": "0000320193-23-000106", "frame": "CY2023Q3I"}
                        ]
                    }
                },
                "SomeNicheConcept": {
                    "label": "Niche",
                    "units": {"USD": [{"end": "2023-09-30", "val": 1, "fy": 2023, "fp": "FY"}]}
                }
            },
            "dei": {
                "EntityCommonStockSharesOutstanding": {
                    "units": {"shares": [{"end": "2023-10-20", "val": 15552752000, "fy": 2023, "fp": "FY"}]}
                }
            }
        }
    }"#;

    #[test]
    fn parses_all_facts_with_empty_whitelist() {
        let facts = parse_company_facts(SAMPLE, &[]).unwrap();
        // Revenues x2 + Assets x1 + SomeNicheConcept x1 + dei x1 = 5.
        assert_eq!(facts.len(), 5);
    }

    #[test]
    fn whitelist_filters_to_headline_concepts() {
        let facts = parse_company_facts(SAMPLE, &["Revenues", "Assets"]).unwrap();
        // SomeNicheConcept + dei concept dropped → 2 Revenues + 1 Assets.
        assert_eq!(facts.len(), 3);
        assert!(facts
            .iter()
            .all(|f| f.tag.ends_with("Revenues") || f.tag.ends_with("Assets")));
    }

    #[test]
    fn captures_period_and_provenance_fields() {
        let facts = parse_company_facts(SAMPLE, &["Revenues"]).unwrap();
        let annual = facts
            .iter()
            .find(|f| f.fiscal_period == "FY")
            .expect("FY revenue fact");
        assert_eq!(annual.value, 383285000000.0);
        assert_eq!(annual.period_start, "2022-10-01");
        assert_eq!(annual.period_end, "2023-09-30");
        assert_eq!(annual.form, "10-K");
        assert_eq!(annual.accession, "0000320193-23-000106");
        assert_eq!(annual.frame, "CY2023");
        assert_eq!(annual.tag, "us-gaap:Revenues");
    }

    #[test]
    fn instant_facts_have_empty_start() {
        let facts = parse_company_facts(SAMPLE, &["Assets"]).unwrap();
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].period_start, "");
        assert_eq!(facts[0].period_end, "2023-09-30");
    }

    #[test]
    fn malformed_json_errors() {
        assert!(parse_company_facts("{not json", &[]).is_err());
    }
}
