// src/graph/reporting.rs
use std::collections::VecDeque;

// Maximum number of reports to keep in history
const MAX_REPORT_HISTORY: usize = 10;

#[derive(Debug, Clone)]
#[allow(clippy::enum_variant_names)]
pub enum OperationReport {
    NodeOperation(NodeOperationReport),
    ConnectionOperation(ConnectionOperationReport),
    CalculationOperation(CalculationOperationReport),
}

#[derive(Debug, Clone)]
pub struct CalculationOperationReport {
    pub operation_type: String,
    pub expression: String,
    pub nodes_processed: usize,
    pub nodes_updated: usize,
    pub nodes_with_errors: usize,
    pub processing_time_ms: f64,
    pub is_aggregation: bool,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub errors: Vec<String>, // Add error collection
}

impl CalculationOperationReport {
    pub fn new(
        operation_type: String,
        expression: String,
        nodes_processed: usize,
        nodes_updated: usize,
        nodes_with_errors: usize,
        processing_time_ms: f64,
        is_aggregation: bool,
    ) -> Self {
        Self {
            operation_type,
            expression,
            nodes_processed,
            nodes_updated,
            nodes_with_errors,
            processing_time_ms,
            is_aggregation,
            timestamp: chrono::Utc::now(),
            errors: Vec::new(),
        }
    }

    pub fn with_errors(mut self, errors: Vec<String>) -> Self {
        self.errors = errors;
        self
    }
}

#[derive(Debug, Clone)]
pub struct NodeOperationReport {
    pub operation_type: String,
    pub nodes_created: usize,
    pub nodes_updated: usize,
    pub nodes_skipped: usize,
    pub processing_time_ms: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub errors: Vec<String>, // Add error collection
}

impl NodeOperationReport {
    pub fn new(
        operation_type: String,
        nodes_created: usize,
        nodes_updated: usize,
        nodes_skipped: usize,
        processing_time_ms: f64,
    ) -> Self {
        Self {
            operation_type,
            nodes_created,
            nodes_updated,
            nodes_skipped,
            processing_time_ms,
            timestamp: chrono::Utc::now(),
            errors: Vec::new(),
        }
    }

    pub fn with_errors(mut self, errors: Vec<String>) -> Self {
        self.errors = errors;
        self
    }
}

#[derive(Debug, Clone)]
pub struct ConnectionOperationReport {
    pub operation_type: String,
    pub connections_created: usize,
    pub connections_skipped: usize,
    /// Stub nodes auto-created so an edge to a missing endpoint could
    /// still connect (see `add_connections` vivification).
    pub stubs_vivified: usize,
    pub property_fields_tracked: usize,
    pub processing_time_ms: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub errors: Vec<String>, // Add error collection
}

impl ConnectionOperationReport {
    pub fn new(
        operation_type: String,
        connections_created: usize,
        connections_skipped: usize,
        property_fields_tracked: usize,
        processing_time_ms: f64,
    ) -> Self {
        Self {
            operation_type,
            connections_created,
            connections_skipped,
            stubs_vivified: 0,
            property_fields_tracked,
            processing_time_ms,
            timestamp: chrono::Utc::now(),
            errors: Vec::new(),
        }
    }

    pub fn with_errors(mut self, errors: Vec<String>) -> Self {
        self.errors = errors;
        self
    }
}

// Create a reports container that we'll store in the KnowledgeGraph
#[derive(Debug, Clone, Default)]
pub struct OperationReports {
    reports: VecDeque<OperationReport>,
    last_operation_index: usize,
}

impl OperationReports {
    pub fn new() -> Self {
        OperationReports {
            reports: VecDeque::with_capacity(MAX_REPORT_HISTORY),
            last_operation_index: 0,
        }
    }

    pub fn add_report(&mut self, report: OperationReport) -> usize {
        // Increment the operation index
        self.last_operation_index += 1;

        // Add to the reports queue
        self.reports.push_back(report);

        // Remove oldest if we exceed the max history
        if self.reports.len() > MAX_REPORT_HISTORY {
            self.reports.pop_front();
        }

        // Return the operation index
        self.last_operation_index
    }

    pub fn get_last_report(&self) -> Option<&OperationReport> {
        self.reports.back()
    }

    pub fn get_all_reports(&self) -> &VecDeque<OperationReport> {
        &self.reports
    }

    pub fn get_last_operation_index(&self) -> usize {
        self.last_operation_index
    }
}
