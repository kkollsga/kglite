//! Validity intervals: the declarations that name a type's two bound
//! properties, the evaluator every surface reads bounds through ([`eval`]),
//! the endpoint indexes and masks that answer an instant without reading
//! bounds, and the valid-time view and slice. The filter queries and fluent
//! steps run under is `core::graph_filter::ElementFilter`; a bound that holds
//! anything but a date, a datetime or an ISO string is an error naming the
//! element and the property, not a pass.

mod coerce_bounds;
pub(crate) mod declarations;
#[cfg(test)]
mod declarations_tests;
mod default_instant;
pub(crate) mod duplicate_ids;
#[cfg(test)]
mod empty_when_tests;
pub(crate) mod endpoint_index;
pub(crate) mod eval;
pub(crate) mod instant;
mod loader;
#[cfg(test)]
mod loader_tests;
mod merge_key;
#[cfg(test)]
mod merge_key_tests;
pub(crate) mod peer_hist;
pub(crate) mod persist;
#[cfg(test)]
mod persist_tests;
mod request;
pub(crate) mod slice;
mod validate;
pub(crate) mod vector_mask;
pub mod view;
mod write_check;
#[cfg(test)]
mod write_check_tests;

pub(crate) use coerce_bounds::{coerce_edge_bounds, coerce_node_bounds};
pub(crate) use declarations::declared;
pub(crate) use declarations::merge_start_key;
pub use declarations::{
    declare, declare_loaded, declare_loaded_with, edge_configs, list, node_config, undeclare,
    DeclarationInfo, DeclareReport, TemporalTarget, DISK_NODE_ABUTMENT_CAP,
};
pub(crate) use declarations::{declare_loaded_grouped, EntityGrouping};
pub use default_instant::ValidTimeDefault;
pub use eval::{EmptyWhen, IntervalConvention};
pub(crate) use loader::{adopt_declarations, settle_adopted, withdraw_adopted};
pub use loader::{declare_defaulted, declare_from_column_types, LoadDeclaration};
pub(crate) use merge_key::{Image, Start, StartKey};
pub use request::{
    node_request_config, node_type_has_property, relationship_request_configs,
    relationship_type_has_property, unknown_bound_message,
};
pub(crate) use validate::{edge_bound, node_bound, view_bound, EmptyIntervals};
#[cfg(test)]
pub(crate) use write_check::unchecked;
pub(crate) use write_check::{
    ancestor_labels, check_edge_load, check_edge_rows, check_labelled_node, check_new_edge,
    check_new_node, check_node_load, check_node_update, check_stored_edge, check_stored_node,
    edge_property_is_bound,
};
pub use write_check::{check_label_stamp, check_labelled_load};

use eval::{BoundSide, TemporalError};

/// `property 'vf': the from bound … is not a date …` for a declaration known
/// by its two bound property names — the element prefix is the caller's,
/// since only it knows which element the properties belong to.
pub(crate) fn describe_bound_error(err: TemporalError, from: &str, to: &str) -> String {
    let property = match &err {
        TemporalError::Bound {
            side: BoundSide::From,
            ..
        } => from,
        TemporalError::Bound {
            side: BoundSide::To,
            ..
        } => to,
        TemporalError::Instant { .. } => return err.to_string(),
    };
    format!("property '{property}': {err}")
}
