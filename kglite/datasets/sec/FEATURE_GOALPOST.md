# SEC feature-extraction goalpost (v0.9.46-WIP)

Inventory of every semantic feature `edgartools` extracts from SEC filings — the
target for our clean-room rebuild. Source: open-source survey of
`dgunning/edgartools` (1652 classes, 12693 functions, MIT-licensed) on
2026-05-20. We **do not** copy code; this is purely a capabilities checklist.

Status legend:
- ✅ already implemented in our extractor
- 🟡 partial (e.g. headers + lot index but missing fields)
- ⬜ not yet implemented
- ➖ out of scope for v0.9.46 (defer to later)

Provenance footer (every info-row CSV gets these 8 columns):
`source_form, source_accession, source_document, source_url, source_lot,
source_page, source_paragraph, source_extracted_at`.

---

## 1. Insider ownership — Forms 3, 4, 4/A, 5, 5/A, 144

### Form 4 / 4/A / 5 (transactions)
*edgartools `Form4` / `Form5` → `NonDerivativeTransaction` + `DerivativeTransaction`*

Per-transaction fields (one row per lot):
- `security` (security_title)
- `date` (transaction_date)
- `shares` (int — number traded)
- `remaining` (shares_owned_after — running balance per ledger)
- `price` (price_per_share)
- `acquired_disposed` ("A" or "D")
- `direct_indirect` ("D" or "I")
- `transaction_code` (P/S/A/D/M/F/G/J/W/X/…)
- `transaction_type` (interpretation of code)
- `equity_swap` flag
- `footnotes` (text of any cited footnotes; e.g. weighted-avg price disclosures)

Status per field: 🟡 — we have everything except `equity_swap` and `transaction_type` interpretation; we DO have footnote-aware price correction (J7).

Plus per-filing fields (apply to every transaction):
- `issuer` (issuer_cik, issuer_name, issuer_trading_symbol) ✅
- `reporting_owner` (cik, name, address, role flags) ✅ (role flags split to typed edges in J4)
- `period_of_report` ⬜ (we don't currently capture)
- `signature` (date, name) ⬜

Output info-CSVs: `purchase.csv` (codes P, A, M-acquisition-side, G-acquisition, J-acquired) + `sale.csv` (codes S, D, F-disposal-side, G-disposed, J-disposed, X) + `holding.csv` (every lot's `shares_owned_after`).

### Form 3 (initial ownership)
*edgartools `Form3` → `NonDerivativeHolding` + `DerivativeHolding` + `InitialOwnershipSummary`*

Per-holding fields (one row per security held at insider-start):
- `security` (security_title)
- `shares` (shares held)
- `direct_indirect`
- `nature_of_ownership` (free text)
- `footnotes`

For derivatives also: `conversion_price`, `exercise_date`, `expiration_date`, `underlying_security`, `underlying_shares`.

Status: ⬜ — we have Form 3 filings in the index but no extractor.

Output info-CSVs: `holding.csv` (initial state, with `source_form = "3"`).

### Form 144 (notice of proposed restricted sale)
*edgartools `Form144` → `SecuritiesInformation`, `SecuritiesToBeSold`, `SecuritiesSoldPast3Months`, `NoticeSignature`*

Three feature blocks:

**`SecuritiesInformation`** (what's being sold):
- `securitiesClassTitle`
- `cusip`
- `broker_name`, `broker_address`
- `approx_sale_date`
- `securities_acquired_date`, `nature_of_acquisition` (how the filer got them)
- `aggregate_market_value`, `shares_or_units_to_be_sold`
- `payment_date` (when filer paid for the restricted shares)

**`SecuritiesToBeSold`** (planned sale):
- `securitiesClassTitle`, `shares`, `approx_sale_date`

**`SecuritiesSoldPast3Months`** (historical sales context):
- `sellerName`, `class`, `date_of_sale`, `shares`, `gross_proceeds`

Status: ⬜.

Output info-CSVs: `planned_sale.csv` + `sale.csv` (the past-3-months rows go to general sale.csv).

---

## 2. Beneficial-ownership filings — Schedule 13D, 13D/A, 13G, 13G/A

*edgartools `Schedule13D` / `Schedule13G` → `ReportingPerson` + `IssuerInfo` + `SecurityInfo` + `Schedule13DItems` (narrative) + `AmendmentInfo`*

### `ReportingPerson` (one row per filer; joint filers have multiple)
- `cik`, `name`
- `citizenship` (country / state of org)
- `sole_voting_power`, `shared_voting_power`
- `sole_dispositive_power`, `shared_dispositive_power`
- `aggregate_amount` (total beneficially owned)
- `percent_of_class`
- `type_of_reporting_person` (IN=individual, CO=corporation, BD=broker-dealer, …)
- `fund_type` (Optional)
- `member_of_group` ("a" if joint filer, "b" if separate)
- `is_aggregate_exclude_shares`
- `no_cik` flag

### `IssuerInfo`
- `name`, `cik`, `address`
- `class_of_security`

### `SecurityInfo`
- `class_title`
- `cusip`
- `total_outstanding` (denominator for percent calc)

### `Schedule13DItems` (narrative, items 1-7)
- Item 1: Security and Issuer
- Item 2: Identity and Background
- Item 3: Source and Amount of Funds
- Item 4: Purpose of Transaction (the activist's intent — gold)
- Item 5: Interest in Securities (tx in last 60 days)
- Item 6: Contracts, Arrangements
- Item 7: Material to be Filed as Exhibits

### `Schedule13GItems` (10 items, simpler/structured)

### `AmendmentInfo`
- `is_amendment`, `amendment_number`, `original_filing_accession`

Status: 🟡 — we have `Stake` with `percent_owned` + `purpose_text`. Missing: per-filer breakout, voting/dispositive split, group membership, full item-7 narrative, amendment linkage.

Output info-CSVs:
- `holding.csv` (5%+ snapshot row per filer, `source_form = "SC 13D"` or `"SC 13G"`)
- `activist_filing.csv` (per-filer narrative items + source-of-funds + purpose)
- `holder_group.csv` (joint-filer relationships when present)

---

## 3. Institutional holdings — 13F-HR, 13F-HR/A, 13F-NT

*edgartools `ThirteenF` → `CoverPage` + `FilingManager` + `OtherManager` + `SummaryPage` + `Holding` (info table rows) + `Signature`*

### Per-holding row (one per security per manager per quarter)
- `name_of_issuer`, `title_of_class`
- `cusip`
- `figi` (Bloomberg Open Symbology)
- `value` (USD value of holding)
- `shares` (shares or principal amount), `shares_type` (SH / PRN)
- `put_call` (PUT, CALL, or empty)
- `investment_discretion` (SOLE / DFND / OTR)
- `other_managers` (numbers from OtherManager list)
- `voting_authority_sole`, `voting_authority_shared`, `voting_authority_none`

### `FilingManager` / `OtherManager`
- Name, address, form-13F-file-number
- (OtherManager is the cross-manager list — when one fund's holdings include shares another fund discretionary-manages)

### Comparison/history models
- `HoldingsComparison` (Q-over-Q delta) — derived
- `HoldingsHistory` (multi-quarter trend) — derived

Status: 🟡 — we have manager + security + HOLDS edges with value/shares/discretion/voting/quarter. Missing: figi, put_call, shares_type explicit, other_managers list, comparison/history (those are derivable downstream).

Output info-CSVs:
- `institutional_holding.csv` (one row per manager × security × quarter)
- `institutional_manager.csv` (manager identity) ✅
- `security.csv` (CUSIP + class) ✅

---

## 4. Proxy filings — DEF 14A, DEFA14A, PRE 14A, N-PX

### DEF 14A (annual proxy statement)
*edgartools `ProxyStatement` + `html_extractor.{BeneficialOwner, DirectorCompEntry, ExecutiveCompEntry, CEOPayRatio, AuditFees, VotingProposal}` + `models.{ExecutiveCompensation, NamedExecutive, PayVsPerformance}`*

**`BeneficialOwner`** — the ownership table (the snapshot we want most):
- `name`
- `holder_type` ('5pct_holder', 'director_officer', 'group')
- `shares`
- `percent_of_class`

**`VotingProposal`** — meeting agenda:
- `number`
- `description`
- `board_recommendation` (FOR/AGAINST/ABSTAIN)
- `proposal_type` (company_proposal / shareholder_proposal)

**`ExecutiveCompEntry`** (Summary Compensation Table, one row per exec per year):
- `name`, `title`, `year`
- `salary`, `bonus`, `stock_awards`, `option_awards`
- `non_equity_incentive`, `pension_change`, `other_compensation`
- `total`

**`DirectorCompEntry`** (Director Compensation Table):
- `name`, fees, stock awards, option awards, non-equity, all other, total

**`CEOPayRatio`** — `ceo_total_comp`, `median_employee_comp`, `ratio`

**`AuditFees`** — audit, audit-related, tax, other fees by year + auditor identity

**`PayVsPerformance`** — exec comp vs TSR comparison (multi-year table)

**`NamedExecutive`** — full bio for named execs (age, principal positions, business experience)

Status: 🟡 — we have a broken `extract_directors` that grabs text fragments. Everything else is unimplemented.

Output info-CSVs:
- `holding.csv` (from BeneficialOwner; `source_form = "DEF 14A"`)
- `role.csv` (director/officer roles)
- `compensation.csv` (ExecutiveCompEntry + DirectorCompEntry)
- `proposal.csv` (VotingProposal)
- `pay_vs_performance.csv`
- `audit_fees.csv`

### N-PX (fund proxy votes)
*edgartools `ProxyVoteTable` → `ProxyTable` × N (per vote)*

Per-vote-record fields:
- `issuer_name`, `meeting_date`
- `vote_description`, `other_vote_description`
- `cusip`, `isin`, `figi`
- `shares_voted`, `shares_on_loan`
- `vote_source` (manager/recommendation provider)
- `vote_series` (the fund series that voted)
- `vote_categories[]` — categorisation (election, M&A, comp, etc.)
- `vote_records[]` — per-proposal vote (FOR/AGAINST/ABSTAIN/WITHHOLD + management recommendation)
- `other_managers[]` — joint voters

Plus per-filing: `IncludedManager`, `SeriesReport`, `ClassInfo`, `ReportSeriesClassInfo`.

Status: ⬜.

Output info-CSVs: `fund_vote.csv` (one row per (fund, security, meeting, proposal)) + cross-references to `proposal.csv`.

---

## 5. Periodic reports — 10-K, 10-Q, 20-F, 6-K, 40-F

*edgartools `TenK`, `TenQ`, `TwentyF`, `SixK`, `FortyF` (`CompanyReport` base) + `SubsidiaryList` + `AuditorInfo` + `PressReleases`*

### Cross-form features
- **Document items** — `FilingStructure` parses item-coded sections (Item 1 Business, 1A Risks, 2 Properties, 3 Legal, 7 MD&A, 7A Market Risk, 8 Financials, 10 Officers, 11 Comp, 12 Ownership, 13 Related Party, 14 Auditor, 15 Exhibits)
- **Exhibit 21 — Subsidiaries**: `Subsidiary` (name, jurisdiction) + `SubsidiaryList` ✅
- **Auditor**: `AuditorInfo` (name, location, PCAOB ID)
- **Press Release attachments** (8-K only): `PressRelease` extracts the actual press release text from EX-99 exhibits

### Item-12 ownership table (same shape as DEF 14A `BeneficialOwner`)
- 5% holders, directors, officers, group totals
- shares + percent_of_class
Status: ⬜ — would emit to `holding.csv` with `source_form = "10-K"` (Item 12 specifically).

### Item-13 related-party transactions
- Counterparty name, relationship, transaction type, amount, year
Status: ⬜.

### Item-7 MD&A free text + Item-1A risk factors
Status: ➖ (NLP-heavy, defer)

---

## 6. Current reports — 8-K

*edgartools `CurrentReport` + per-item handling via `FilingStructure` + `EarningsRelease` (Item 7.01 / 2.02 specifically)*

### `EightKItem` (per item code) ✅
- Item 1.01 Material Agreement
- Item 1.02 Termination
- Item 2.01 Completed Acquisition/Disposition
- Item 2.02 Earnings Release (often attaches Press Release exhibit)
- Item 2.03 Material Off-Balance Obligation
- Item 2.04 Triggering Event
- Item 3.01 Listing/Delisting
- Item 3.02 Unregistered Sale
- Item 4.01 Auditor Change
- Item 4.02 Restatement
- Item 5.02 Officer/Director Change (departure, election, comp)
- Item 5.03 Charter Amendment
- Item 5.07 Vote Results (from annual meeting)
- Item 7.01 Reg FD Disclosure
- Item 8.01 Other Material Event

Status: ✅ item codes captured. Missing: typed NER per item.

### Typed event extractors (edgartools doesn't fully do these yet either — opportunity)
- `officer_change.csv` — Item 5.02 parsed for person name + role change type (departure/appointment) + effective date
- `ma_event.csv` — Items 1.01 / 2.01 parsed for target / acquirer / consideration
- `vote_result.csv` — Item 5.07 parsed for proposal × (for, against, abstain, broker non-votes)
- `auditor_change.csv` — Item 4.01 parsed for old + new auditor
- `earnings_release.csv` — EarningsRelease income statement + balance sheet + cash flow tables

Status: ⬜ for the typed extractors.

### `EarningsRelease` (8-K Item 2.02 + EX-99 press releases)
- `income_statement.dataframe` (parsed financial table)
- `balance_sheet.dataframe`
- `cash_flow.dataframe`
- `FinancialTable` model — parsed multi-column financial table from earnings press release HTML

Status: ⬜.

---

## 7. Offerings — S-1, S-3, S-4, 424B(2/3/5), DRS, Form D, Form C

*edgartools `RegistrationS1`, `RegistrationS3`, `Prospectus424B`, `FormD`, `FormC`, `DraftRegistrationStatement`*

### Selling stockholders (S-1, 424B)
**`SellingStockholderEntry`** (one row per selling holder):
- `name`
- `shares_before_offering`, `pct_before_offering`
- `shares_offered`
- `shares_after_offering`, `pct_after_offering`
- `warrants_or_convertible`

### Offering economics (424B `Prospectus424B`)
- `CoverPageData`, `PricingData`, `PricingColumnData`, `OfferingTerms`, `DilutionData`, `CapitalizationData`
- `Deal` (synthesised summary: shares, price, gross proceeds, net proceeds, use of proceeds, underwriting discount)
- `RegistrationFeeTable` + `FilingFeesRow` (the EX-FILING FEE exhibit)
- `ShelfLifecycle` (where in the shelf process this 424B sits)
- `StructuredNoteTerms` (for structured product offerings)
- `UnderwriterEntry`, `UnderwritingInfo`

### Form D (Reg D private placement)
**`FormD`** → `Filer`, `OfferingData`, `OfferingSalesAmounts`, `UseOfProceeds`, `SalesCommissionFindersFees`, `Investors`, `BusinessCombinationTransaction`, `IndustryGroup`, `InvestmentFundInfo`, `SalesCompensationRecipient`

Per filing: total offering amount, amount sold, type of securities (equity, debt, option, warrant), industry, related M&A.

### Form C (crowdfunding)
**`FormC`** → `FundingPortal`, `OfferingInformation`, `AnnualReportDisclosure`, `IssuerCompany`, `IssuerSignature`, etc.

Status: ⬜ across the board (none of these forms are parsed today).

Output info-CSVs:
- `offering.csv` (S-1/S-3/424B/Form D/Form C — all offerings with raise size, security type, price)
- `sale.csv` (SellingStockholderEntry rows; `source_form = "S-1"` etc.)
- `holding.csv` (shares-before-offering snapshot)
- `underwriter.csv` (per-offering underwriter line; with discount + over-allotment option)
- `use_of_proceeds.csv`

---

## 8. M&A registration — S-4

*edgartools `Prospectus424B` covers some S-4 derivatives; `BusinessCombinationTransaction` in Form D*

Conceptually: target company, acquirer, consideration (cash + stock + earn-out), exchange ratio, fairness opinion provider, expected close date. edgartools' S-4 coverage is shallower than S-1.

Status: ⬜.
Output info-CSVs: `merger.csv` (target + acquirer + consideration + close date).

---

## 9. Fund-specific filings — N-CEN, N-CSR, N-MFP3, 497K, 24F-2NT

### N-CEN (Annual census)
*edgartools `FundCensus` → `Director`, `Accountant`, `BrokerDealer`, `AuthorizedParticipant`, `LiquidityProvider`, `LineOfCredit`, `PrincipalTransaction`, `ServiceProvider`, `SecuritiesLending`, `FundSeriesInfo`, `ETFInfo`, `RegistrantInfo`, `SignatureInfo`*

### N-CSR (shareholder report)
`FundShareholderReport` → `Holding` (top holdings), `AnnualReturn`, `ShareClassInfo`

### N-MFP3 (money market fund holdings, monthly)
`MoneyMarketFund` → `PortfolioSecurity`, `RepurchaseAgreement`, `CollateralIssuer`, `CreditRating`, `GeneralInfo`, `SeriesLevelInfo`, `ShareClassInfo`

### 497K (summary prospectus)
`Prospectus497K` → `PerformanceReturn`, `ShareClassFees`

### 24F-2NT (annual fee notice)
`FundFeeNotice` → `SeriesInfo`, `FundClassFee`

Status: ➖ all of these (defer until v0.9.47+, fund tracking is a separate use case from public-company analytics).

---

## 10. Specialized — ABS-EE, 10-D, ATS-N, BDC

### ABS-EE / 10-D (asset-backed securities)
*edgartools `TenD`, `DistributionReport`, `AutoLeaseAssetData`, `CMBSAssetData`, `DistributionMetrics`*

Per loan-pool: distribution amount, principal balance, delinquency, default. ABS-EE EX-102 has asset-level data.

Status: ➖.

### ATS-N (Alternative Trading System operator)
*edgartools `AlternativeTradingSystem` + 6 sub-models (ATSIdentifyingInfo, ATSOperations, ATSOperatorActivities, etc.)*

Status: ➖ (very narrow).

### BDC (Business Development Company)
*edgartools `PortfolioInvestment` (Schedule of Investments line) + `NonAccrualInvestment`*

Per investment: target company, investment type (senior secured, subordinated, equity), fair value, cost, interest rate, maturity, non-accrual flag.

Status: ⬜ but possibly high-value — BDC portfolios are large private-credit datasets.

### Municipal advisors
`MunicipalAdvisorForm` + 14 disclosure models (CivilDisclosure, CriminalDisclosure, etc.)

Status: ➖.

---

## 11. Cross-cutting capabilities

### XBRL financial facts
*edgartools `EntityFacts` + `FactsView` + `FactQuery` + `XBRL` model family*

Standardised financial facts from 10-K/10-Q XBRL tagging:
- `Fact` (concept, value, unit, period, context, dimensions, decimals)
- `Context` (entity, period, dimensions) — what the fact refers to
- `ElementCatalog` (the company's tagged concepts)
- `CalculationTree` (additivity relationships — how line items roll up)
- `PresentationTree` (display order)
- `Notes` to the financial statements

Status: 🟡 — we have a FSNDS parser but the URL 404s recently (SEC moved files). The bulk-XBRL FSNDS approach is one source; the per-filing XBRL approach (R-files in 10-K/10-Q) is another. edgartools uses the per-filing approach which is more reliable.

Output info-CSVs:
- `metric_fact.csv` (one row per tagged fact)
- `metric_context.csv` (dimensional contexts)
- `financial_statement.csv` (rolled-up statements: income, balance, cashflow)

### TTM (trailing twelve months)
*edgartools `TTMCalculator`*: derives TTM metrics from quarterly facts. Pure aggregation, doesn't extract new info.
Status: ➖ — derive in blueprint compute, not extractor.

### Earnings release tables
*edgartools `EarningsRelease`* (covered above in §6) extracts income/balance/cashflow tables from 8-K press release HTML. Bridges 8-K → financial-statement data without waiting for XBRL.

Status: ⬜.

### Entity identity
*edgartools `Company`, `CompanyData`, `EntityFacts`*

Beyond what we have:
- Stock exchange listings (issuer_trading_symbol per exchange)
- Former names history (we have it ✅)
- Address (we have part of it ✅)
- SIC industry classification (we have it ✅)
- Auditor history
- IRS number, state of incorporation, fiscal year end ✅

Status: ✅ mostly.

---

## Goalpost summary — info-node CSVs

The processor should produce these CSVs (each row = one extracted fact with the 8-column provenance footer):

| CSV | Rows from | Status |
|---|---|---|
| `purchase.csv` | Form 4 (P/A/M/G acq), Form 5, 144 history, S-1 selling, 424B | 🟡 → produces via M2 filter; needs Form 5, 144, S-1 |
| `sale.csv` | Form 4 (S/D/F/X), Form 5, 144 history, 144 planned, S-1 | 🟡 → same |
| `holding.csv` | Form 4 lots, Form 3 initial, DEF 14A ownership, 10-K Item 12, SC 13D/G total | 🟡 → only Form 4 today |
| `planned_sale.csv` | Form 144 SecuritiesToBeSold | ⬜ |
| `institutional_holding.csv` | 13F-HR | 🟡 → has core fields, missing figi/put_call/other_managers |
| `role.csv` | Form 4 flags ✅, DEF 14A noms, 10-K Item 10 | 🟡 → only Form 4 |
| `compensation.csv` | DEF 14A Summary Comp + Director Comp | ⬜ |
| `pay_vs_performance.csv` | DEF 14A | ⬜ |
| `ceo_pay_ratio.csv` | DEF 14A | ⬜ |
| `audit_fees.csv` | DEF 14A / 10-K Item 14 | ⬜ |
| `auditor.csv` | DEI XBRL (AuditorInfo) | ⬜ |
| `proposal.csv` | DEF 14A VotingProposal | ⬜ |
| `fund_vote.csv` | N-PX ProxyTable + VoteRecord | ⬜ |
| `vote_result.csv` | 8-K Item 5.07 | ⬜ |
| `corporate_event.csv` | 8-K item codes | ✅ |
| `officer_change.csv` | 8-K Item 5.02 NER | ⬜ |
| `ma_event.csv` | 8-K Items 1.01 + 2.01 + S-4 | ⬜ |
| `auditor_change.csv` | 8-K Item 4.01 | ⬜ |
| `restatement.csv` | 8-K Item 4.02 | ⬜ |
| `subsidiary.csv` | 10-K / 20-F Exhibit 21 | ✅ |
| `related_party_transaction.csv` | 10-K / DEF 14A Item 13 | ⬜ |
| `activist_filing.csv` | SC 13D narrative + group | 🟡 → percent + purpose only |
| `holder_group.csv` | SC 13D joint filer links | ⬜ |
| `offering.csv` | S-1 / S-3 / 424B / Form D / Form C | ⬜ |
| `merger.csv` | S-4 (+ Form D BusinessCombinationTransaction) | ⬜ |
| `underwriter.csv` | S-1 / 424B | ⬜ |
| `use_of_proceeds.csv` | S-1 / 424B / Form D | ⬜ |
| `selling_stockholder.csv` | S-1 / 424B (also feeds `sale.csv`) | ⬜ |
| `earnings_release.csv` | 8-K EX-99 financial tables | ⬜ |
| `metric_fact.csv` | 10-K / 10-Q XBRL | 🟡 (FSNDS feed broken) |
| `metric_context.csv` | XBRL dimensional contexts | ⬜ |
| `financial_statement.csv` | XBRL rolled-up statements | ⬜ |
| `bdc_investment.csv` | BDC Schedule of Investments | ⬜ (high-value but optional) |

Plus identity tables (unchanged shape from today):
- `company.csv` ✅
- `person.csv` ✅
- `security.csv` ✅
- `institutional_manager.csv` ✅

---

## Out of scope for v0.9.46 (deferred)

- All fund-specific filings (N-CEN, N-CSR, N-MFP3, 497K, 24F-2NT) — fund tracking is a separate workstream.
- ABS-EE / 10-D — asset-backed securities loan-level data, narrow audience.
- ATS-N — alternative trading systems, niche.
- Municipal advisors — niche.
- DRS / Form C — crowdfunding + draft registrations, low-volume.
- 6-K, 20-F detailed extraction — foreign issuers; defer until US issuers fully covered.
- 40-F — Canadian MJDS, defer.
- Press release NLP (named entity extraction from free text).
- Risk factors / MD&A NLP (huge surface, defer to ML pipeline if needed).

---

## Suggested implementation order

1. **Form 4/4/A** (✅ improve to all fields incl. equity_swap + transaction_type)
2. **Form 3** (initial holdings — same XML shape as Form 4)
3. **Form 5** (same XML — late-reported Form 4 shape)
4. **DEF 14A ownership table** (BeneficialOwner — `holding.csv` snapshot)
5. **DEF 14A nominees + comp** (`role.csv`, `compensation.csv`, `pay_vs_performance.csv`)
6. **DEF 14A proposals** (`proposal.csv`)
7. **SC 13D/G full** (per-filer breakout, voting/dispositive split, group)
8. **Form 144** (`planned_sale.csv`, `sale.csv` history)
9. **8-K Item 5.02 NER** (`officer_change.csv`)
10. **8-K Item 5.07** (`vote_result.csv`)
11. **8-K Item 4.01** (`auditor_change.csv`)
12. **EarningsRelease** (8-K Item 2.02 + EX-99 → `earnings_release.csv`)
13. **10-K Item 12** (BeneficialOwner — second source for `holding.csv`)
14. **10-K Item 13** (`related_party_transaction.csv`)
15. **S-1 + 424B** (`offering.csv`, `selling_stockholder.csv` → `sale.csv`)
16. **Form D** (`offering.csv` private placement)
17. **S-4** (`merger.csv`)
18. **XBRL per-filing approach** (replace broken FSNDS feed)
19. **N-PX** (`fund_vote.csv` — fund proxy voting records)
20. **BDC investments** (optional, high-value private-credit data)

Each filing-family commit is a self-contained unit: extractor + tests + (no blueprint change yet — blueprint comes after the processor is done).
