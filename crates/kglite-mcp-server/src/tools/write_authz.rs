//! Who may write what through `cypher_query`, on a write-enabled server.
//!
//! Two parties declare a write scope and neither trusts the other to be the
//! whole answer:
//!
//! - the **operator**, at boot, via `--write-scope` / `extensions.write_scope`
//!   (see `server_run::boot_write_scope`) — access control, because the agent
//!   cannot change it; and
//! - the **agent**, per call, via the `write_scope` tool argument — role
//!   hygiene, because the agent chose it itself.
//!
//! [`resolve_write_scope`] settles them with one rule: *the agent may narrow,
//! never widen*. An operator pin therefore applies even when the agent omits
//! its own scope — the historical shape, where an absent argument meant
//! "unrestricted", is exactly the fail-open an operator installs a pin to
//! prevent.

/// The write-authorisation inputs for one `cypher_query` call: the two scopes
/// plus the freshness provenance stamped on `auto_timestamp` types.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct WriteAuthz<'a> {
    /// Boot-pinned scope. `None` = the operator pinned nothing.
    pub(crate) operator_scope: Option<&'a [String]>,
    /// The agent's own `write_scope` argument. `None` = the agent asked for
    /// no restriction of its own (which is *not* the same as unrestricted).
    pub(crate) agent_scope: Option<&'a [String]>,
    pub(crate) git_sha: Option<&'a str>,
    pub(crate) modified_by: Option<&'a str>,
}

/// Combine the operator pin and the agent's scope into the set the engine
/// enforces, or the refusal a write must be answered with.
///
/// - No pin, no argument → `None`: unrestricted, today's behaviour.
/// - No pin → the agent's list verbatim (an empty list still denies every
///   mutation, which is what the engine does with an empty scope set).
/// - Pin, no argument → the pin. **Never unrestricted.**
/// - Both → their intersection.
///
/// An empty *result* under a pin is a refusal rather than an empty set handed
/// to the engine, so the agent reads why it cannot write (whose scope, and
/// what it asked for) instead of a generic per-node scope error on whatever
/// node the statement happened to touch first.
pub(crate) fn resolve_write_scope(
    operator: Option<&[String]>,
    agent: Option<&[String]>,
) -> Result<Option<std::collections::HashSet<String>>, String> {
    let Some(pinned) = operator else {
        return Ok(agent.map(|a| a.iter().cloned().collect()));
    };
    let effective: Vec<String> = match agent {
        None => pinned.to_vec(),
        Some(a) => a.iter().filter(|t| pinned.contains(t)).cloned().collect(),
    };
    if effective.is_empty() {
        return Err(refusal(pinned, agent));
    }
    Ok(Some(effective.into_iter().collect()))
}

/// The refusal text for a write with nothing left in its effective scope. It
/// always names the operator scope — the agent cannot see the server's flags,
/// so "refused" without the pin is unactionable.
fn refusal(pinned: &[String], agent: Option<&[String]>) -> String {
    if pinned.is_empty() {
        return "no writes permitted under this server's write scope (empty): the operator pinned \
                an empty write scope, so every mutation is refused"
            .to_string();
    }
    let because = match agent {
        // Unreachable while the pin is non-empty (an omitted agent scope
        // resolves to the pin itself), kept so the message is total.
        None => "the operator pinned no writable node type".to_string(),
        Some([]) => "the requested write_scope is empty, so it permits nothing".to_string(),
        Some(a) => format!(
            "the requested write_scope [{}] shares no node type with it",
            a.join(", ")
        ),
    };
    format!(
        "no writes permitted under this server's write scope [{}]: {because}",
        pinned.join(", ")
    )
}

#[cfg(test)]
mod write_authz_tests {
    use super::*;

    fn scope(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| (*s).to_string()).collect()
    }

    fn set(names: &[&str]) -> std::collections::HashSet<String> {
        names.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn no_pin_leaves_the_agent_scope_untouched() {
        assert_eq!(resolve_write_scope(None, None).unwrap(), None);
        let agent = scope(&["Plan"]);
        assert_eq!(
            resolve_write_scope(None, Some(&agent)).unwrap(),
            Some(set(&["Plan"]))
        );
        // An agent-supplied empty list keeps its historical meaning: the
        // engine's empty scope set denies every mutation.
        let none: Vec<String> = Vec::new();
        assert_eq!(
            resolve_write_scope(None, Some(&none)).unwrap(),
            Some(std::collections::HashSet::new())
        );
    }

    #[test]
    fn a_pin_applies_when_the_agent_omits_its_own_scope() {
        let pinned = scope(&["Plan", "Task"]);
        assert_eq!(
            resolve_write_scope(Some(&pinned), None).unwrap(),
            Some(set(&["Plan", "Task"])),
            "an omitted agent scope must never fall back to unrestricted"
        );
    }

    #[test]
    fn both_scopes_intersect() {
        let pinned = scope(&["Plan", "Task"]);
        let agent = scope(&["Task", "Algorithm"]);
        assert_eq!(
            resolve_write_scope(Some(&pinned), Some(&agent)).unwrap(),
            Some(set(&["Task"])),
            "the agent narrows within the pin; Algorithm is not in the pin"
        );
    }

    #[test]
    fn an_empty_intersection_refuses_and_names_the_operator_scope() {
        let pinned = scope(&["Plan", "Task"]);
        let agent = scope(&["Algorithm"]);
        let error = resolve_write_scope(Some(&pinned), Some(&agent))
            .expect_err("nothing left in scope must refuse, not fall open");
        assert!(
            error.starts_with("no writes permitted under this server's write scope"),
            "{error}"
        );
        assert!(error.contains("[Plan, Task]"), "{error}");
        assert!(error.contains("Algorithm"), "{error}");
    }

    #[test]
    fn an_empty_pin_refuses_every_write() {
        let pinned: Vec<String> = Vec::new();
        for agent in [None, Some(scope(&["Plan"]))] {
            let error = resolve_write_scope(Some(&pinned), agent.as_deref())
                .expect_err("an empty pin permits nothing");
            assert!(error.contains("(empty)"), "{error}");
        }
    }
}
