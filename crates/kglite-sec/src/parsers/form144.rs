//! Form 144 (post-2016 XML) parser — Notice of Proposed Sale of
//! Restricted Securities.
//!
//! SEC mandates XML submission from 2016Q4. Older filings are HTML;
//! we cover the XML path here, leaving HTML as a TODO for legacy
//! coverage.
//!
//! Three feature blocks:
//!
//! - `securitiesInformation` — what's being sold (class, CUSIP,
//!   broker, approx sale date, market value, payment date)
//! - `securitiesToBeSold` — repeatable; planned-sale rows
//! - `securitiesSoldPast3Months` — repeatable; historical sales for
//!   context (Rule 144's 3-month volume limit)

use quick_xml::events::Event;
use quick_xml::Reader;
use std::io::BufRead;

use crate::error::{Result, SecError};

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Form144 {
    pub filer_name: String,
    pub filer_cik: String,
    pub issuer_name: String,
    pub issuer_cik: String,
    pub security_class: String,
    pub broker_name: String,
    pub broker_address: String,
    /// Date of acquisition of the restricted securities being sold.
    pub securities_acquired_date: String,
    pub nature_of_acquisition: String,
    /// "cash" / "stock" / "services" — how the filer paid.
    pub payment_kind: String,
    /// Aggregate market value of all securities to be sold.
    pub aggregate_market_value: f64,
    /// Date the broker is approximately going to sell.
    pub approx_sale_date: String,
    pub shares_to_be_sold: f64,
    pub planned_sales: Vec<PlannedSale>,
    pub historical_sales: Vec<HistoricalSale>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct PlannedSale {
    pub security_class: String,
    pub shares: f64,
    pub approx_sale_date: String,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct HistoricalSale {
    pub seller_name: String,
    pub security_class: String,
    pub sale_date: String,
    pub shares: f64,
    pub gross_proceeds: f64,
}

/// Parse one Form 144 XML document.
pub fn parse_form144<R: BufRead>(reader: R) -> Result<Form144> {
    let mut xml = Reader::from_reader(reader);
    xml.config_mut().trim_text(true);

    let mut out = Form144::default();
    let mut path: Vec<String> = Vec::new();
    let mut text = String::new();
    let mut current_planned: Option<PlannedSale> = None;
    let mut current_hist: Option<HistoricalSale> = None;

    let mut buf = Vec::new();
    loop {
        match xml.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => {
                let name = std::str::from_utf8(e.name().as_ref())
                    .map_err(|err| SecError::Decode(format!("Form 144 tag: {err}")))?
                    .to_string();
                path.push(name.clone());
                text.clear();
                match name.as_str() {
                    "securitiesToBeSoldInfo" => current_planned = Some(PlannedSale::default()),
                    "sellerDetails" => current_hist = Some(HistoricalSale::default()),
                    _ => {}
                }
            }
            Ok(Event::Text(t)) => {
                let s = t
                    .unescape()
                    .map_err(|err| SecError::Decode(format!("Form 144 text: {err}")))?;
                text.push_str(&s);
            }
            Ok(Event::End(e)) => {
                let name = std::str::from_utf8(e.name().as_ref())
                    .map_err(|err| SecError::Decode(format!("Form 144 end tag: {err}")))?
                    .to_string();
                apply(
                    &name,
                    &path,
                    &text,
                    &mut out,
                    current_planned.as_mut(),
                    current_hist.as_mut(),
                );
                if name == "securitiesToBeSoldInfo" {
                    if let Some(p) = current_planned.take() {
                        out.planned_sales.push(p);
                    }
                }
                if name == "sellerDetails" {
                    if let Some(h) = current_hist.take() {
                        out.historical_sales.push(h);
                    }
                }
                path.pop();
                text.clear();
            }
            Ok(Event::Eof) => break,
            Err(e) => return Err(SecError::Decode(format!("Form 144 XML parse: {e}"))),
            _ => {}
        }
        buf.clear();
    }
    Ok(out)
}

fn apply(
    leaf: &str,
    _path: &[String],
    text: &str,
    out: &mut Form144,
    planned: Option<&mut PlannedSale>,
    hist: Option<&mut HistoricalSale>,
) {
    if text.is_empty() {
        return;
    }
    // Top-level identity / filing fields.
    match leaf {
        "personName" | "filerName" => out.filer_name = text.to_string(),
        "filerCik" | "personCik" => out.filer_cik = text.to_string(),
        "issuerName" => out.issuer_name = text.to_string(),
        "issuerCik" => out.issuer_cik = text.to_string(),
        "brokerName" | "brokerOrDealerName" => out.broker_name = text.to_string(),
        "brokerAddress" | "brokerOrDealerAddress" => out.broker_address = text.to_string(),
        "securitiesClassTitle" if planned.is_none() && hist.is_none() => {
            out.security_class = text.to_string()
        }
        "securitiesAcquiredDate" => out.securities_acquired_date = text.to_string(),
        "natureOfAcquisitionTransaction" => out.nature_of_acquisition = text.to_string(),
        "natureOfPayment" => out.payment_kind = text.to_string(),
        "aggregateMarketValue" => out.aggregate_market_value = parse_float(text),
        "approxSaleDate" if planned.is_none() => out.approx_sale_date = text.to_string(),
        "noOfUnitsToBeSold" | "amountOfSecuritiesToBeSold" if planned.is_none() => {
            out.shares_to_be_sold = parse_float(text)
        }
        _ => {}
    }

    // Planned-sale block scope.
    if let Some(p) = planned {
        match leaf {
            "securitiesClassTitle" => p.security_class = text.to_string(),
            "noOfUnitsToBeSold" | "amountOfSecuritiesToBeSold" => p.shares = parse_float(text),
            "approxSaleDate" => p.approx_sale_date = text.to_string(),
            _ => {}
        }
    }

    // Historical-sale block scope.
    if let Some(h) = hist {
        match leaf {
            "personName" | "sellerName" => h.seller_name = text.to_string(),
            "securitiesClassTitle" => h.security_class = text.to_string(),
            "saleDate" => h.sale_date = text.to_string(),
            "amountSold" | "noOfUnitsSold" => h.shares = parse_float(text),
            "grossProceeds" => h.gross_proceeds = parse_float(text),
            _ => {}
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    const SAMPLE: &str = r#"<?xml version="1.0"?>
<edgarSubmission>
    <issuerInfo>
        <issuerCik>0000320193</issuerCik>
        <issuerName>Apple Inc.</issuerName>
    </issuerInfo>
    <filerInfo>
        <filerCik>0001214156</filerCik>
        <filerName>COOK TIMOTHY D</filerName>
    </filerInfo>
    <securitiesInformation>
        <securitiesClassTitle>Common Stock</securitiesClassTitle>
        <brokerOrDealerName>Charles Schwab</brokerOrDealerName>
        <securitiesAcquiredDate>2021-09-30</securitiesAcquiredDate>
        <natureOfAcquisitionTransaction>RSU vesting</natureOfAcquisitionTransaction>
        <natureOfPayment>services</natureOfPayment>
        <aggregateMarketValue>22500000</aggregateMarketValue>
        <approxSaleDate>2024-10-15</approxSaleDate>
        <amountOfSecuritiesToBeSold>100000</amountOfSecuritiesToBeSold>
    </securitiesInformation>
    <securitiesToBeSoldInfo>
        <securitiesClassTitle>Common Stock</securitiesClassTitle>
        <amountOfSecuritiesToBeSold>100000</amountOfSecuritiesToBeSold>
        <approxSaleDate>2024-10-15</approxSaleDate>
    </securitiesToBeSoldInfo>
    <securitiesSoldInPast3Months>
        <sellerDetails>
            <personName>COOK TIMOTHY D</personName>
            <securitiesClassTitle>Common Stock</securitiesClassTitle>
            <saleDate>2024-08-15</saleDate>
            <amountSold>50000</amountSold>
            <grossProceeds>11250000</grossProceeds>
        </sellerDetails>
    </securitiesSoldInPast3Months>
</edgarSubmission>"#;

    #[test]
    fn parses_typical_form144_xml() {
        let parsed = parse_form144(Cursor::new(SAMPLE)).unwrap();
        assert_eq!(parsed.issuer_name, "Apple Inc.");
        assert_eq!(parsed.filer_name, "COOK TIMOTHY D");
        assert_eq!(parsed.broker_name, "Charles Schwab");
        assert_eq!(parsed.security_class, "Common Stock");
        assert_eq!(parsed.aggregate_market_value, 22500000.0);
        assert_eq!(parsed.shares_to_be_sold, 100000.0);
        assert_eq!(parsed.planned_sales.len(), 1);
        assert_eq!(parsed.planned_sales[0].shares, 100000.0);
        assert_eq!(parsed.historical_sales.len(), 1);
        assert_eq!(parsed.historical_sales[0].shares, 50000.0);
        assert_eq!(parsed.historical_sales[0].gross_proceeds, 11250000.0);
    }
}
