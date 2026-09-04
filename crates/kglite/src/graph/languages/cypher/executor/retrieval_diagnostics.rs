//! Operator-level retrieval evidence; no synchronization on the scalar hot path.
use super::*;

impl RetrievalDiagnostics {
    pub(super) fn exact(reason: &str) -> Self {
        Self {
            requested_policy: "auto".into(),
            actual_mode: "exact".into(),
            fallback_reason: Some(reason.into()),
            store: None,
        }
    }
    pub(super) fn fallback(mut self, reason: &str) -> Self {
        self.fallback_reason = Some(reason.into());
        self
    }
}

impl CypherExecutor<'_> {
    pub(super) fn record_retrieval(&self, record: RetrievalDiagnostics) {
        let mut records = self
            .runtime_retrieval
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if !records.contains(&record) {
            records.push(record);
        }
    }
    fn take_retrieval(&self) -> Vec<RetrievalDiagnostics> {
        std::mem::take(
            &mut *self
                .runtime_retrieval
                .lock()
                .unwrap_or_else(|e| e.into_inner()),
        )
    }
    pub(super) fn attach_runtime_diagnostics(&self, result: &mut CypherResult) {
        let records = self.take_retrieval();
        let warnings = self.take_runtime_warnings();
        if records.is_empty() && warnings.is_empty() {
            return;
        }
        let d = result
            .diagnostics
            .get_or_insert_with(QueryDiagnostics::default);
        d.retrieval.extend(records);
        d.warnings.extend(warnings);
    }
    fn retrieval_option_is_constant(expression: &Expression) -> bool {
        match expression {
            Expression::MapLiteral(items) => items
                .iter()
                .all(|(_, value)| Self::retrieval_option_is_constant(value)),
            _ => Self::is_row_independent(expression),
        }
    }
    pub(super) fn requested_retrieval_policy(&self, args: &[Expression]) -> Result<String, String> {
        if !args[3..].iter().all(Self::retrieval_option_is_constant) {
            return Ok("per_row".into());
        }
        let tail = args[3..]
            .iter()
            .map(|expr| self.evaluate_expression(expr, &ResultRow::new()))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(if vector_options::parse(&tail)?.exact {
            "exact"
        } else {
            "auto"
        }
        .into())
    }
    pub(super) fn record_exact_ordering(
        &self,
        keys: &[FusedSortKey],
        rows: &ResultSet,
        limit: usize,
    ) -> Result<(), String> {
        if rows.rows.is_empty() || limit == 0 {
            return Ok(());
        }
        for key in keys {
            if let Expression::FunctionCall { name, args, .. } = &key.expression {
                if name == "vector_score" && (3..=5).contains(&args.len()) {
                    let mut record = RetrievalDiagnostics::exact("ordering_requires_exact");
                    record.requested_policy = self.requested_retrieval_policy(args)?;
                    self.record_retrieval(record);
                }
            }
        }
        Ok(())
    }
    /// Record a non-fatal execution-time warning, and echo it to stderr for
    /// interactive users (the same one-computation/two-consumers split
    /// `session::execute::prepare` applies to the schema warnings).
    ///
    /// Repeats are dropped: a correlated `CALL {}` body re-executes its CALL
    /// clause once per outer row, and the same mis-spelled relationship type
    /// is one fact about the query however many rows re-discover it.
    pub(super) fn warn(&self, message: String) {
        let mut warnings = self
            .runtime_warnings
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if warnings.contains(&message) {
            return;
        }
        super::super::emit_query_warnings(std::slice::from_ref(&message));
        warnings.push(message);
    }

    /// Move a nested executor's warnings onto this one. A `CALL {}` body runs
    /// on its own executor, so without this its procedure warnings would be
    /// dropped when that executor is.
    pub(super) fn absorb_warnings(&self, nested: &CypherExecutor<'_>) {
        for record in nested.take_retrieval() {
            self.record_retrieval(record);
        }
        for warning in nested.take_runtime_warnings() {
            self.warn_absorbed(warning);
        }
    }

    /// Absorb one already-emitted warning: recorded (de-duplicated) but not
    /// re-printed — stderr saw it when the nested executor raised it.
    fn warn_absorbed(&self, message: String) {
        let mut warnings = self
            .runtime_warnings
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if !warnings.contains(&message) {
            warnings.push(message);
        }
    }

    /// Absorb the warnings a nested [`Self::execute`] parked on its result.
    pub(super) fn absorb_diagnostics(&self, nested: &CypherResult) {
        let Some(diagnostics) = nested.diagnostics.as_ref() else {
            return;
        };
        for record in &diagnostics.retrieval {
            self.record_retrieval(record.clone());
        }
        for warning in &diagnostics.warnings {
            self.warn_absorbed(warning.clone());
        }
    }

    fn take_runtime_warnings(&self) -> Vec<String> {
        std::mem::take(
            &mut *self
                .runtime_warnings
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
        )
    }
}
