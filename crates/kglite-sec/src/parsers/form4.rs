//! Streaming parser for SEC Form 4 / 4/A XML (insider transactions).
//!
//! Form 4 XSD schemaVersion X0508 structure:
//!
//! ```xml
//! <ownershipDocument>
//!   <documentType>4</documentType>
//!   <periodOfReport>2024-10-29</periodOfReport>
//!   <issuer>
//!     <issuerCik>0000320193</issuerCik>
//!     <issuerName>Apple Inc.</issuerName>
//!     <issuerTradingSymbol>AAPL</issuerTradingSymbol>
//!   </issuer>
//!   <reportingOwner>
//!     <reportingOwnerId>
//!       <rptOwnerCik>0001214156</rptOwnerCik>
//!       <rptOwnerName>COOK TIMOTHY D</rptOwnerName>
//!     </reportingOwnerId>
//!     <reportingOwnerRelationship>
//!       <isDirector>0</isDirector>
//!       <isOfficer>1</isOfficer>
//!       <isTenPercentOwner>0</isTenPercentOwner>
//!       <isOther>0</isOther>
//!       <officerTitle>CEO</officerTitle>
//!     </reportingOwnerRelationship>
//!   </reportingOwner>
//!   <nonDerivativeTable>
//!     <nonDerivativeTransaction>
//!       <securityTitle><value>Common Stock</value></securityTitle>
//!       <transactionDate><value>2024-10-15</value></transactionDate>
//!       <transactionCoding>
//!         <transactionCode>S</transactionCode>
//!       </transactionCoding>
//!       <transactionAmounts>
//!         <transactionShares><value>100000</value></transactionShares>
//!         <transactionPricePerShare><value>225.50</value></transactionPricePerShare>
//!         <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
//!       </transactionAmounts>
//!       <postTransactionAmounts>
//!         <sharesOwnedFollowingTransaction>
//!           <value>3000000</value>
//!         </sharesOwnedFollowingTransaction>
//!       </postTransactionAmounts>
//!       <ownershipNature>
//!         <directOrIndirectOwnership><value>D</value></directOrIndirectOwnership>
//!       </ownershipNature>
//!     </nonDerivativeTransaction>
//!   </nonDerivativeTable>
//!   <!-- derivativeTable has the same shape with extra option-pricing fields -->
//! </ownershipDocument>
//! ```
//!
//! Many fields are wrapped in `<value>...</value>` (a quirk of the
//! Form 4 XSD — they carry optional footnote refs as siblings).
//! Empty / missing optionals are tolerated and emitted as `None` /
//! empty string.

use quick_xml::events::Event;
use quick_xml::Reader;
use std::io::BufRead;

use crate::error::{Result, SecError};

/// One insider's relationship to the issuer + their non-derivative
/// + derivative transactions on a single filing.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Form4 {
    pub period_of_report: String,
    pub issuer_cik: String,
    pub issuer_name: String,
    pub issuer_trading_symbol: String,
    pub reporter_cik: String,
    pub reporter_name: String,
    pub is_director: bool,
    pub is_officer: bool,
    pub is_ten_percent_owner: bool,
    pub is_other: bool,
    pub officer_title: String,
    pub transactions: Vec<InsiderTransaction>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct InsiderTransaction {
    /// "Common Stock", "Restricted Stock Unit", "Option to Purchase ...", etc.
    pub security_title: String,
    pub transaction_date: String,
    /// SEC transaction code (P/S/A/D/M/F/G/J/V/X/...).
    /// See https://www.sec.gov/about/forms/form4data.pdf for the full list.
    pub transaction_code: String,
    pub shares: f64,
    pub price_per_share: f64,
    /// "A" (acquired) or "D" (disposed). Together with shares this gives the signed delta.
    pub acquired_disposed: String,
    pub shares_owned_after: f64,
    /// "D" (direct) or "I" (indirect; e.g. through a trust).
    pub direct_indirect: String,
    /// True if this came from `<derivativeTable>` (options, warrants); false from `<nonDerivativeTable>`.
    pub is_derivative: bool,
}

/// Parse one Form 4 XML document from a streaming reader. Tolerates
/// missing fields (older Form 4 schema variants); raises only on
/// malformed XML.
pub fn parse_form4<R: BufRead>(reader: R) -> Result<Form4> {
    let mut xml = Reader::from_reader(reader);
    xml.config_mut().trim_text(true);

    let mut out = Form4::default();
    let mut path: Vec<String> = Vec::new();
    let mut current_text = String::new();

    // Working transaction being filled in; pushed onto out.transactions
    // when </nonDerivativeTransaction> or </derivativeTransaction> closes.
    let mut current_txn: Option<InsiderTransaction> = None;

    let mut buf = Vec::new();
    loop {
        match xml.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => {
                let name = std::str::from_utf8(e.name().as_ref())
                    .map_err(|err| SecError::Decode(format!("Form 4 tag: {err}")))?
                    .to_string();
                path.push(name.clone());
                current_text.clear();
                match name.as_str() {
                    "nonDerivativeTransaction" => {
                        current_txn = Some(InsiderTransaction {
                            is_derivative: false,
                            ..Default::default()
                        });
                    }
                    "derivativeTransaction" => {
                        current_txn = Some(InsiderTransaction {
                            is_derivative: true,
                            ..Default::default()
                        });
                    }
                    _ => {}
                }
            }
            Ok(Event::Text(t)) => {
                let s = t
                    .unescape()
                    .map_err(|err| SecError::Decode(format!("Form 4 text: {err}")))?;
                current_text.push_str(&s);
            }
            Ok(Event::End(e)) => {
                let name = std::str::from_utf8(e.name().as_ref())
                    .map_err(|err| SecError::Decode(format!("Form 4 end tag: {err}")))?
                    .to_string();
                handle_end(&name, &path, &current_text, &mut out, &mut current_txn);
                if name == "nonDerivativeTransaction" || name == "derivativeTransaction" {
                    if let Some(txn) = current_txn.take() {
                        out.transactions.push(txn);
                    }
                }
                path.pop();
                current_text.clear();
            }
            Ok(Event::Eof) => break,
            Err(e) => {
                return Err(SecError::Decode(format!("Form 4 XML parse: {e}")));
            }
            _ => {}
        }
        buf.clear();
    }
    Ok(out)
}

/// Apply the text content of `<tag>` to the right output field based
/// on the surrounding XML path. The `<value>` wrapper inside many
/// Form 4 leaves means we need to look at the second-deepest tag to
/// know which field this text belongs to (e.g.
/// path = `[..., transactionShares, value]`, leaf = "value", parent =
/// "transactionShares" → set `shares`).
fn handle_end(
    leaf: &str,
    path: &[String],
    text: &str,
    out: &mut Form4,
    txn: &mut Option<InsiderTransaction>,
) {
    if text.is_empty() {
        return;
    }
    let parent = if path.len() >= 2 {
        path[path.len() - 2].as_str()
    } else {
        ""
    };
    // For `<value>` leaves, the parent is what carries the field name.
    let field = if leaf == "value" { parent } else { leaf };

    // Issuer / reporter / relationship fields go on the top level.
    match field {
        "periodOfReport" => out.period_of_report = text.to_string(),
        "issuerCik" => out.issuer_cik = strip_leading_zeros(text),
        "issuerName" => out.issuer_name = text.to_string(),
        "issuerTradingSymbol" => out.issuer_trading_symbol = text.to_string(),
        "rptOwnerCik" => out.reporter_cik = strip_leading_zeros(text),
        "rptOwnerName" => out.reporter_name = text.to_string(),
        "isDirector" => out.is_director = parse_bool(text),
        "isOfficer" => out.is_officer = parse_bool(text),
        "isTenPercentOwner" => out.is_ten_percent_owner = parse_bool(text),
        "isOther" => out.is_other = parse_bool(text),
        "officerTitle" => out.officer_title = text.to_string(),
        _ => {}
    }

    // Transaction-scoped fields only apply when we're inside a transaction.
    if let Some(t) = txn.as_mut() {
        match field {
            "securityTitle" => t.security_title = text.to_string(),
            "transactionDate" => t.transaction_date = text.to_string(),
            "transactionCode" => t.transaction_code = text.to_string(),
            "transactionShares" => t.shares = parse_float(text),
            "transactionPricePerShare" => t.price_per_share = parse_float(text),
            "transactionAcquiredDisposedCode" => t.acquired_disposed = text.to_string(),
            "sharesOwnedFollowingTransaction" => t.shares_owned_after = parse_float(text),
            "directOrIndirectOwnership" => t.direct_indirect = text.to_string(),
            _ => {}
        }
    }
}

fn parse_bool(s: &str) -> bool {
    let t = s.trim();
    t == "1" || t.eq_ignore_ascii_case("true")
}

fn parse_float(s: &str) -> f64 {
    s.trim().parse::<f64>().unwrap_or(0.0)
}

fn strip_leading_zeros(s: &str) -> String {
    let t = s.trim().trim_start_matches('0');
    if t.is_empty() {
        "0".to_string()
    } else {
        t.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    const COOK_FORM4: &str = r#"<?xml version="1.0"?>
<ownershipDocument>
    <schemaVersion>X0508</schemaVersion>
    <documentType>4</documentType>
    <periodOfReport>2024-10-29</periodOfReport>
    <issuer>
        <issuerCik>0000320193</issuerCik>
        <issuerName>Apple Inc.</issuerName>
        <issuerTradingSymbol>AAPL</issuerTradingSymbol>
    </issuer>
    <reportingOwner>
        <reportingOwnerId>
            <rptOwnerCik>0001214156</rptOwnerCik>
            <rptOwnerName>COOK TIMOTHY D</rptOwnerName>
        </reportingOwnerId>
        <reportingOwnerAddress>
            <rptOwnerStreet1>ONE APPLE PARK WAY</rptOwnerStreet1>
            <rptOwnerCity>CUPERTINO</rptOwnerCity>
            <rptOwnerState>CA</rptOwnerState>
            <rptOwnerZipCode>95014</rptOwnerZipCode>
        </reportingOwnerAddress>
        <reportingOwnerRelationship>
            <isDirector>0</isDirector>
            <isOfficer>1</isOfficer>
            <isTenPercentOwner>0</isTenPercentOwner>
            <isOther>0</isOther>
            <officerTitle>CEO</officerTitle>
        </reportingOwnerRelationship>
    </reportingOwner>
    <nonDerivativeTable>
        <nonDerivativeTransaction>
            <securityTitle><value>Common Stock</value></securityTitle>
            <transactionDate><value>2024-10-15</value></transactionDate>
            <transactionCoding>
                <transactionFormType>4</transactionFormType>
                <transactionCode>S</transactionCode>
                <equitySwapInvolved>0</equitySwapInvolved>
            </transactionCoding>
            <transactionAmounts>
                <transactionShares><value>100000</value></transactionShares>
                <transactionPricePerShare><value>225.50</value></transactionPricePerShare>
                <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
            </transactionAmounts>
            <postTransactionAmounts>
                <sharesOwnedFollowingTransaction><value>3000000</value></sharesOwnedFollowingTransaction>
            </postTransactionAmounts>
            <ownershipNature>
                <directOrIndirectOwnership><value>D</value></directOrIndirectOwnership>
            </ownershipNature>
        </nonDerivativeTransaction>
        <nonDerivativeTransaction>
            <securityTitle><value>Restricted Stock Unit</value></securityTitle>
            <transactionDate><value>2024-04-01</value></transactionDate>
            <transactionCoding>
                <transactionCode>M</transactionCode>
            </transactionCoding>
            <transactionAmounts>
                <transactionShares><value>50000</value></transactionShares>
                <transactionPricePerShare><value>0</value></transactionPricePerShare>
                <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
            </transactionAmounts>
            <postTransactionAmounts>
                <sharesOwnedFollowingTransaction><value>3050000</value></sharesOwnedFollowingTransaction>
            </postTransactionAmounts>
            <ownershipNature>
                <directOrIndirectOwnership><value>D</value></directOrIndirectOwnership>
            </ownershipNature>
        </nonDerivativeTransaction>
    </nonDerivativeTable>
</ownershipDocument>
"#;

    #[test]
    fn parses_cook_form4_fixture() {
        let f4 = parse_form4(Cursor::new(COOK_FORM4)).unwrap();
        assert_eq!(f4.period_of_report, "2024-10-29");
        assert_eq!(f4.issuer_cik, "320193");
        assert_eq!(f4.issuer_name, "Apple Inc.");
        assert_eq!(f4.issuer_trading_symbol, "AAPL");
        assert_eq!(f4.reporter_cik, "1214156");
        assert_eq!(f4.reporter_name, "COOK TIMOTHY D");
        assert!(!f4.is_director);
        assert!(f4.is_officer);
        assert!(!f4.is_ten_percent_owner);
        assert!(!f4.is_other);
        assert_eq!(f4.officer_title, "CEO");
        assert_eq!(f4.transactions.len(), 2);

        let sale = &f4.transactions[0];
        assert_eq!(sale.transaction_code, "S");
        assert_eq!(sale.transaction_date, "2024-10-15");
        assert_eq!(sale.shares, 100000.0);
        assert_eq!(sale.price_per_share, 225.50);
        assert_eq!(sale.acquired_disposed, "D");
        assert_eq!(sale.shares_owned_after, 3000000.0);
        assert_eq!(sale.direct_indirect, "D");
        assert_eq!(sale.security_title, "Common Stock");
        assert!(!sale.is_derivative);

        let vest = &f4.transactions[1];
        assert_eq!(vest.transaction_code, "M");
        assert_eq!(vest.acquired_disposed, "A");
        assert_eq!(vest.security_title, "Restricted Stock Unit");
    }

    #[test]
    fn handles_empty_xml() {
        let r = parse_form4(Cursor::new(r#"<?xml version="1.0"?><ownershipDocument/>"#));
        let f4 = r.unwrap();
        assert!(f4.issuer_cik.is_empty());
        assert_eq!(f4.transactions.len(), 0);
    }

    #[test]
    fn parses_derivative_transaction() {
        let xml = r#"<?xml version="1.0"?>
<ownershipDocument>
    <documentType>4</documentType>
    <issuer><issuerCik>123</issuerCik><issuerName>Test</issuerName></issuer>
    <reportingOwner>
        <reportingOwnerId><rptOwnerCik>456</rptOwnerCik><rptOwnerName>Smith</rptOwnerName></reportingOwnerId>
        <reportingOwnerRelationship><isOfficer>1</isOfficer></reportingOwnerRelationship>
    </reportingOwner>
    <derivativeTable>
        <derivativeTransaction>
            <securityTitle><value>Option to Purchase Common Stock</value></securityTitle>
            <transactionDate><value>2024-01-15</value></transactionDate>
            <transactionCoding><transactionCode>A</transactionCode></transactionCoding>
            <transactionAmounts>
                <transactionShares><value>1000</value></transactionShares>
                <transactionPricePerShare><value>0</value></transactionPricePerShare>
                <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
            </transactionAmounts>
        </derivativeTransaction>
    </derivativeTable>
</ownershipDocument>"#;
        let f4 = parse_form4(Cursor::new(xml)).unwrap();
        assert_eq!(f4.transactions.len(), 1);
        assert!(f4.transactions[0].is_derivative);
        assert_eq!(
            f4.transactions[0].security_title,
            "Option to Purchase Common Stock"
        );
    }

    #[test]
    fn rejects_severely_malformed_xml() {
        // quick-xml is permissive about unclosed tags (it just hits
        // EOF cleanly). It does reject malformed tags like missing `>`.
        let r = parse_form4(Cursor::new("<ownershipDocument <bad"));
        assert!(r.is_err());
    }
}
