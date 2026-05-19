//! Streaming parser for SEC Form 13F-HR information tables (institutional holdings).
//!
//! 13F-HR information table XML structure:
//!
//! ```xml
//! <informationTable xmlns="http://www.sec.gov/edgar/document/thirteenf/informationtable">
//!   <infoTable>
//!     <nameOfIssuer>APPLE INC</nameOfIssuer>
//!     <titleOfClass>COM</titleOfClass>
//!     <cusip>037833100</cusip>
//!     <value>1234567</value>
//!     <shrsOrPrnAmt>
//!       <sshPrnamt>100000</sshPrnamt>
//!       <sshPrnamtType>SH</sshPrnamtType>
//!     </shrsOrPrnAmt>
//!     <investmentDiscretion>SOLE</investmentDiscretion>
//!     <votingAuthority>
//!       <Sole>100000</Sole>
//!       <Shared>0</Shared>
//!       <None>0</None>
//!     </votingAuthority>
//!   </infoTable>
//! </informationTable>
//! ```
//!
//! The `value` field is the position's market value. Pre-2023 Q4 it
//! was in $1000s; from 2023 Q4 onward it's in $1. We store raw —
//! callers must know the period to interpret.

use quick_xml::events::Event;
use quick_xml::Reader;
use std::io::BufRead;

use crate::error::{Result, SecError};

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Holding {
    pub name_of_issuer: String,
    pub title_of_class: String,
    pub cusip: String,
    /// Position value (USD thousands pre-2023Q4, USD from 2023Q4).
    pub value: f64,
    /// Share count (or principal amount for bonds).
    pub shares: f64,
    /// "SH" (shares) or "PRN" (principal amount).
    pub shares_type: String,
    /// "SOLE" / "DFND" / "OTR".
    pub investment_discretion: String,
    pub voting_sole: f64,
    pub voting_shared: f64,
    pub voting_none: f64,
}

/// Parse a 13F-HR information table XML stream → list of holdings.
pub fn parse_13f_info_table<R: BufRead>(reader: R) -> Result<Vec<Holding>> {
    let mut xml = Reader::from_reader(reader);
    xml.config_mut().trim_text(true);

    let mut holdings = Vec::new();
    let mut current: Option<Holding> = None;
    let mut path: Vec<String> = Vec::new();
    let mut text = String::new();

    let mut buf = Vec::new();
    loop {
        match xml.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => {
                let name =
                    local_name(&e.name()).ok_or_else(|| SecError::Decode("13F tag".into()))?;
                path.push(name.clone());
                text.clear();
                if name == "infoTable" {
                    current = Some(Holding::default());
                }
            }
            Ok(Event::Text(t)) => {
                let s = t
                    .unescape()
                    .map_err(|err| SecError::Decode(format!("13F text: {err}")))?;
                text.push_str(&s);
            }
            Ok(Event::End(e)) => {
                let name =
                    local_name(&e.name()).ok_or_else(|| SecError::Decode("13F end tag".into()))?;
                if let Some(h) = current.as_mut() {
                    apply(h, &name, &path, &text);
                }
                if name == "infoTable" {
                    if let Some(h) = current.take() {
                        holdings.push(h);
                    }
                }
                path.pop();
                text.clear();
            }
            Ok(Event::Eof) => break,
            Err(e) => return Err(SecError::Decode(format!("13F XML parse: {e}"))),
            _ => {}
        }
        buf.clear();
    }
    Ok(holdings)
}

fn local_name(name: &quick_xml::name::QName) -> Option<String> {
    // local_name() strips the namespace prefix; we copy out as owned
    // String so the borrow doesn't escape the event lifetime.
    std::str::from_utf8(name.local_name().as_ref())
        .ok()
        .map(|s| s.to_string())
}

fn apply(h: &mut Holding, leaf: &str, path: &[String], text: &str) {
    if text.is_empty() {
        return;
    }
    match leaf {
        "nameOfIssuer" => h.name_of_issuer = text.to_string(),
        "titleOfClass" => h.title_of_class = text.to_string(),
        "cusip" => h.cusip = text.trim().to_string(),
        "value" => h.value = parse_float(text),
        "sshPrnamt" => h.shares = parse_float(text),
        "sshPrnamtType" => h.shares_type = text.to_string(),
        "investmentDiscretion" => h.investment_discretion = text.to_string(),
        "Sole" if in_voting_authority(path) => h.voting_sole = parse_float(text),
        "Shared" if in_voting_authority(path) => h.voting_shared = parse_float(text),
        "None" if in_voting_authority(path) => h.voting_none = parse_float(text),
        _ => {}
    }
}

fn in_voting_authority(path: &[String]) -> bool {
    path.iter().any(|p| p == "votingAuthority")
}

fn parse_float(s: &str) -> f64 {
    s.trim().replace(',', "").parse::<f64>().unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    const SAMPLE: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<informationTable xmlns="http://www.sec.gov/edgar/document/thirteenf/informationtable">
    <infoTable>
        <nameOfIssuer>APPLE INC</nameOfIssuer>
        <titleOfClass>COM</titleOfClass>
        <cusip>037833100</cusip>
        <value>1234567</value>
        <shrsOrPrnAmt>
            <sshPrnamt>5500000</sshPrnamt>
            <sshPrnamtType>SH</sshPrnamtType>
        </shrsOrPrnAmt>
        <investmentDiscretion>SOLE</investmentDiscretion>
        <votingAuthority>
            <Sole>5500000</Sole>
            <Shared>0</Shared>
            <None>0</None>
        </votingAuthority>
    </infoTable>
    <infoTable>
        <nameOfIssuer>MICROSOFT CORP</nameOfIssuer>
        <titleOfClass>COM</titleOfClass>
        <cusip>594918104</cusip>
        <value>987654</value>
        <shrsOrPrnAmt>
            <sshPrnamt>2400000</sshPrnamt>
            <sshPrnamtType>SH</sshPrnamtType>
        </shrsOrPrnAmt>
        <investmentDiscretion>SOLE</investmentDiscretion>
        <votingAuthority>
            <Sole>2400000</Sole>
            <Shared>0</Shared>
            <None>0</None>
        </votingAuthority>
    </infoTable>
</informationTable>"#;

    #[test]
    fn parses_two_holdings() {
        let h = parse_13f_info_table(Cursor::new(SAMPLE)).unwrap();
        assert_eq!(h.len(), 2);
        assert_eq!(h[0].name_of_issuer, "APPLE INC");
        assert_eq!(h[0].cusip, "037833100");
        assert_eq!(h[0].value, 1_234_567.0);
        assert_eq!(h[0].shares, 5_500_000.0);
        assert_eq!(h[0].shares_type, "SH");
        assert_eq!(h[0].investment_discretion, "SOLE");
        assert_eq!(h[0].voting_sole, 5_500_000.0);
        assert_eq!(h[1].name_of_issuer, "MICROSOFT CORP");
        assert_eq!(h[1].cusip, "594918104");
    }

    #[test]
    fn empty_table_yields_no_holdings() {
        let h = parse_13f_info_table(Cursor::new(r#"<?xml version="1.0"?><informationTable/>"#))
            .unwrap();
        assert_eq!(h.len(), 0);
    }
}
