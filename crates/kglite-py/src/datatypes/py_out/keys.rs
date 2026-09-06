use std::collections::{HashMap, HashSet};

/// Reserve genuine labels before allocating suffixes, so A,A,A_2 cannot alias.
/// Stable identity/content orders colliding entries within this result; keys
/// are presentation only and do not promise identity across graph mutations.
pub(super) fn presentation_keys<K: Ord>(
    entries: Vec<(String, K)>,
    metadata: &[&str],
) -> Vec<String> {
    let reserved: HashSet<&str> = entries
        .iter()
        .map(|(label, _)| label.as_str())
        .chain(metadata.iter().copied())
        .collect();
    let mut used: HashSet<String> = metadata.iter().map(|name| (*name).to_owned()).collect();
    let mut next_suffix = HashMap::new();
    let mut order: Vec<usize> = (0..entries.len()).collect();
    order.sort_by(|&a, &b| entries[a].cmp(&entries[b]));
    let mut keys = vec![String::new(); entries.len()];
    for index in order {
        let base = &entries[index].0;
        if used.insert(base.clone()) {
            keys[index] = base.clone();
            continue;
        }
        let suffix = next_suffix.entry(base).or_insert(2);
        loop {
            let candidate = format!("{base}_{suffix}");
            *suffix += 1;
            if !reserved.contains(candidate.as_str()) && used.insert(candidate.clone()) {
                keys[index] = candidate;
                break;
            }
        }
    }
    keys
}

#[cfg(test)]
mod tests {
    use super::presentation_keys;

    #[test]
    fn reserves_real_suffixes_independent_of_input_order() {
        let input = vec![("A".into(), 2), ("A_2".into(), 3), ("A".into(), 1)];
        assert_eq!(presentation_keys(input, &[]), ["A_3", "A_2", "A"]);
    }

    #[test]
    fn reserves_metadata_and_repeated_genuine_suffixes() {
        let input = vec![
            ("parent_id".into(), 1),
            ("parent_id_2".into(), 2),
            ("parent_id".into(), 3),
        ];
        assert_eq!(
            presentation_keys(input, &["parent_id"]),
            ["parent_id_3", "parent_id_2", "parent_id_4"]
        );
    }
}
