//! Form D — Notice of Exempt Offering of Securities (Reg D private
//! placement).
//!
//! Filed as structured XML. Issuer reports per-raise terms: total
//! offering amount, amount sold, type of securities, exemption
//! claimed, number of investors, sales-compensation recipients.

use quick_xml::events::Event;
use quick_xml::Reader;
use std::io::BufRead;

use crate::datasets::sec::error::{Result, SecError};

use super::{append_unescaped_text, append_xml_reference};

#[derive(Debug, Clone, Default, PartialEq)]
pub struct FormD {
    pub issuer_cik: String,
    pub issuer_name: String,
    pub entity_type: String,
    pub state_of_incorporation: String,
    pub year_of_incorporation: String,
    pub industry_group_type: String,
    /// Total offering amount declared (may be unlimited).
    pub total_offering_amount: f64,
    pub total_amount_sold: f64,
    pub total_remaining: f64,
    /// "Y" / "N" — does this offering raise > $1MM?
    pub is_indefinite: String,
    /// Securities offered — equity / debt / option / warrant / units.
    pub securities_offered: Vec<String>,
    /// Number of non-accredited investors.
    pub non_accredited_investors: u64,
    /// Total number of investors who have purchased.
    pub total_investors: u64,
    /// Date of first sale.
    pub first_sale_date: String,
    /// Sales commission paid.
    pub sales_commission: f64,
    /// Finders fees paid.
    pub finders_fees: f64,
    /// Gross proceeds to be used for executive officers (use of
    /// proceeds — short text).
    pub use_of_proceeds_summary: String,
}

pub fn parse_formd<R: BufRead>(reader: R) -> Result<FormD> {
    let mut xml = Reader::from_reader(reader);
    xml.config_mut().trim_text(false);

    let mut out = FormD::default();
    let mut path: Vec<String> = Vec::new();
    let mut text = String::new();

    let mut buf = Vec::new();
    loop {
        match xml.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => {
                let name = std::str::from_utf8(e.name().as_ref())
                    .map_err(|err| SecError::Decode(format!("Form D tag: {err}")))?
                    .to_string();
                path.push(name);
                text.clear();
            }
            Ok(Event::Text(t)) => {
                append_unescaped_text(&mut text, &t, "Form D text")?;
            }
            Ok(Event::GeneralRef(reference)) => {
                append_xml_reference(&mut text, &reference, "Form D text")?;
            }
            Ok(Event::End(e)) => {
                let name = std::str::from_utf8(e.name().as_ref())
                    .map_err(|err| SecError::Decode(format!("Form D end tag: {err}")))?
                    .to_string();
                apply(&name, &path, text.trim(), &mut out);
                path.pop();
                text.clear();
            }
            Ok(Event::Eof) => break,
            Err(e) => return Err(SecError::Decode(format!("Form D XML parse: {e}"))),
            _ => {}
        }
        buf.clear();
    }
    Ok(out)
}

fn apply(leaf: &str, _path: &[String], text: &str, out: &mut FormD) {
    if text.is_empty() {
        return;
    }
    match leaf {
        // Issuer info.
        "cik" if out.issuer_cik.is_empty() => out.issuer_cik = text.to_string(),
        "entityName" => out.issuer_name = text.to_string(),
        "entityType" => out.entity_type = text.to_string(),
        "jurisdictionOfInc" | "stateOfIncorp" => out.state_of_incorporation = text.to_string(),
        "yearOfInc" => out.year_of_incorporation = text.to_string(),
        "industryGroupType" => out.industry_group_type = text.to_string(),

        // Offering economics.
        "totalOfferingAmount" => out.total_offering_amount = parse_float(text),
        "totalAmountSold" => out.total_amount_sold = parse_float(text),
        "totalRemaining" => out.total_remaining = parse_float(text),
        "isIndefiniteAmount" => out.is_indefinite = text.to_string(),

        // Securities offered.
        "isEquityType" if text == "true" => {
            out.securities_offered.push("equity".to_string());
        }
        "isDebtType" if text == "true" => {
            out.securities_offered.push("debt".to_string());
        }
        "isOptionToAcquireType" if text == "true" => {
            out.securities_offered.push("option".to_string());
        }
        "isSecurityToBeAcquiredType" if text == "true" => {
            out.securities_offered.push("acquirable".to_string());
        }
        "isPooledInvestmentFundType" if text == "true" => {
            out.securities_offered.push("pooled_fund".to_string());
        }
        "isTenantInCommonType" if text == "true" => {
            out.securities_offered.push("tic".to_string());
        }
        "isMineralPropertyType" if text == "true" => {
            out.securities_offered.push("mineral".to_string());
        }
        "isOtherType" if text == "true" => {
            out.securities_offered.push("other".to_string());
        }

        // Investors.
        "totalNumberAlreadyInvested" => out.total_investors = parse_int(text),
        "nonAccreditedInvestorsCount" => out.non_accredited_investors = parse_int(text),
        "firstSale" => out.first_sale_date = text.to_string(),
        // Sales compensation.
        "totalSalesCommission" => out.sales_commission = parse_float(text),
        "totalFindersFees" => out.finders_fees = parse_float(text),
        // Use of proceeds — usually a free-text block in the filing
        // narrative; SEC's Form D XML rarely has it structured. Pick
        // up "useOfProceedsSummary" when present.
        "useOfProceedsSummary" => out.use_of_proceeds_summary = text.to_string(),
        _ => {}
    }
}

fn parse_float(s: &str) -> f64 {
    let cleaned: String = s
        .trim()
        .chars()
        .filter(|c| *c != ',' && *c != '$')
        .collect();
    cleaned.parse::<f64>().unwrap_or(0.0)
}

fn parse_int(s: &str) -> u64 {
    let cleaned: String = s.trim().chars().filter(|c| c.is_ascii_digit()).collect();
    cleaned.parse::<u64>().unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    const SAMPLE: &str = r#"<?xml version="1.0"?>
<edgarSubmission>
  <primaryIssuer>
    <cik>0001318605</cik>
    <entityName>SpaceX Series E LLC</entityName>
    <entityType>Limited Liability Company</entityType>
    <jurisdictionOfInc>Delaware</jurisdictionOfInc>
    <yearOfInc>2020</yearOfInc>
    <industryGroupType>Aerospace</industryGroupType>
  </primaryIssuer>
  <offeringData>
    <typesOfSecuritiesOffered>
      <isEquityType>true</isEquityType>
      <isOtherType>false</isOtherType>
    </typesOfSecuritiesOffered>
    <offeringSalesAmounts>
      <totalOfferingAmount>250000000</totalOfferingAmount>
      <totalAmountSold>200000000</totalAmountSold>
      <totalRemaining>50000000</totalRemaining>
      <isIndefiniteAmount>false</isIndefiniteAmount>
    </offeringSalesAmounts>
    <investors>
      <totalNumberAlreadyInvested>22</totalNumberAlreadyInvested>
      <nonAccreditedInvestorsCount>0</nonAccreditedInvestorsCount>
    </investors>
    <salesCommissionsFindersFees>
      <totalSalesCommission>500000</totalSalesCommission>
      <totalFindersFees>0</totalFindersFees>
    </salesCommissionsFindersFees>
  </offeringData>
</edgarSubmission>"#;

    #[test]
    fn parses_typical_form_d() {
        let parsed = parse_formd(Cursor::new(SAMPLE)).unwrap();
        assert!(parsed.issuer_name.contains("SpaceX"));
        assert_eq!(parsed.total_offering_amount, 250_000_000.0);
        assert_eq!(parsed.total_amount_sold, 200_000_000.0);
        assert_eq!(parsed.total_investors, 22);
        assert_eq!(parsed.sales_commission, 500_000.0);
        assert!(parsed.securities_offered.iter().any(|s| s == "equity"));
    }
}
