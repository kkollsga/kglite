// src/graph/type_build_meta.rs
//
// Per-type metadata collected during Phase 1 for pre-allocating column files.
// Tracks row counts, column keys, data types, and string byte totals.

use crate::datatypes::values::Value;
use crate::graph::schema::InternedKey;
use std::collections::HashMap;

/// Column data type, matching block_column::ColumnType but without BlockPool dependency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColType {
    Int64,
    Float64,
    UniqueId,
    Bool,
    Date,
    Str,
}

impl ColType {
    /// Bytes per value for fixed-width types. None for Str.
    pub fn value_size(&self) -> Option<usize> {
        match self {
            ColType::Int64 | ColType::Float64 => Some(8),
            ColType::UniqueId | ColType::Date => Some(4),
            ColType::Bool => Some(1),
            ColType::Str => None,
        }
    }

    /// Infer column type from a Value variant.
    pub fn from_value(value: &Value) -> Option<Self> {
        match value {
            Value::Int64(_) => Some(ColType::Int64),
            Value::Float64(_) => Some(ColType::Float64),
            Value::UniqueId(_) => Some(ColType::UniqueId),
            Value::Boolean(_) => Some(ColType::Bool),
            Value::DateTime(_) => Some(ColType::Date),
            Value::String(_) => Some(ColType::Str),
            Value::Null => None,
            _ => Some(ColType::Str),
        }
    }

    /// Type tag string for TypedColumn compatibility.
    pub fn type_tag(&self) -> &'static str {
        match self {
            ColType::Int64 => "int64",
            ColType::Float64 => "float64",
            ColType::UniqueId => "uniqueid",
            ColType::Bool => "bool",
            ColType::Date => "date",
            ColType::Str => "string",
        }
    }
}

/// Metadata for a single column within a type.
#[derive(Debug, Clone)]
pub struct ColumnMeta {
    pub col_type: ColType,
    /// Total bytes of string data (only meaningful for Str columns).
    pub string_bytes: u64,
    /// Number of non-null values seen for this column.
    pub non_null_count: u32,
}

/// Metadata for all columns of a single node type.
#[derive(Debug, Clone)]
pub struct TypeBuildMeta {
    pub row_count: u32,
    /// Property key → column metadata.
    pub columns: HashMap<InternedKey, ColumnMeta>,
    /// Total bytes for the title string column.
    pub title_string_bytes: u64,
    /// Total bytes for the id column (only if id is stored as string).
    pub id_is_string: bool,
    pub id_string_bytes: u64,
}

impl TypeBuildMeta {
    pub fn new() -> Self {
        TypeBuildMeta {
            row_count: 0,
            columns: HashMap::new(),
            title_string_bytes: 0,
            id_is_string: false,
            id_string_bytes: 0,
        }
    }

    /// Merge another TypeBuildMeta into this one (for type relabeling merges).
    pub fn merge_from(&mut self, other: &TypeBuildMeta) {
        self.row_count += other.row_count;
        self.title_string_bytes += other.title_string_bytes;
        self.id_string_bytes += other.id_string_bytes;
        if other.id_is_string {
            self.id_is_string = true;
        }
        for (key, col) in &other.columns {
            let entry = self.columns.entry(*key).or_insert_with(|| ColumnMeta {
                col_type: col.col_type,
                string_bytes: 0,
                non_null_count: 0,
            });
            entry.non_null_count += col.non_null_count;
            entry.string_bytes += col.string_bytes;
        }
    }

    /// Record one entity's properties for this type.
    pub fn record_entity(
        &mut self,
        id: &Value,
        title: &Value,
        properties: &[(InternedKey, Value)],
    ) {
        self.row_count += 1;

        // Track id column
        if let Value::String(s) = id {
            self.id_is_string = true;
            self.id_string_bytes += s.len() as u64;
        }

        // Track title column (always string)
        if let Value::String(s) = title {
            self.title_string_bytes += s.len() as u64;
        }

        // Track property columns
        for (key, value) in properties {
            if matches!(value, Value::Null) {
                continue;
            }
            let col = self.columns.entry(*key).or_insert_with(|| {
                let col_type = ColType::from_value(value).unwrap_or(ColType::Str);
                ColumnMeta {
                    non_null_count: 0,
                    col_type,
                    string_bytes: 0,
                }
            });
            col.non_null_count += 1;
            if col.col_type == ColType::Str {
                if let Value::String(s) = value {
                    col.string_bytes += s.len() as u64;
                }
            }
        }
    }

    /// Fill rate for a column (0.0 to 1.0).
    pub fn fill_rate(&self, col: &ColumnMeta) -> f64 {
        if self.row_count == 0 {
            return 0.0;
        }
        col.non_null_count as f64 / self.row_count as f64
    }
}
