//! Buffer-level attribution using the current MmapOrVec/MmapBytes source.
//! Column shapes/push ordering mirror the numeric/string Session CREATE fixture.
// The included production module exposes unrelated mapping APIs unused here.
#[allow(dead_code)]
#[path = "../../../crates/kglite/src/graph/storage/mapped/mmap_vec.rs"]
mod buffers;
use buffers::{MmapBytes, MmapOrVec, MmapPod};
use serde_json::{json, Value as Json};
use std::hint::black_box;
use std::time::Instant;

type Error = Box<dyn std::error::Error>;
struct Numeric {
    data: MmapOrVec<i64>,
    nulls: MmapOrVec<u8>,
}
struct Text {
    offsets: MmapOrVec<u64>,
    bytes: MmapBytes,
    nulls: MmapOrVec<u8>,
}
struct Columns {
    numeric: Vec<Numeric>,
    text: Vec<Text>,
    tombstones: Vec<u8>,
}

fn grow<T>(len: usize, added: usize) -> usize {
    if added == 0 {
        return len;
    }
    (len * 2)
        .max(len + added)
        .max(if std::mem::size_of::<T>() == 1 { 8 } else { 4 })
}
fn reserved<T: Copy>(source: &[T], added: usize) -> Vec<T> {
    let mut out = Vec::with_capacity(grow::<T>(source.len(), added));
    out.extend_from_slice(source);
    out
}
fn copy<T: MmapPod>(source: &MmapOrVec<T>, added: usize, reserve: bool) -> MmapOrVec<T> {
    if reserve {
        MmapOrVec::from_vec(reserved(source.as_slice(), added))
    } else {
        source.clone()
    }
}
fn texts(id: usize) -> [String; 3] {
    [id.to_string(), id.to_string(), "new".into()]
}

impl Columns {
    fn seed(nodes: usize, width: usize) -> Self {
        let numeric = (0..width)
            .map(|_| Numeric {
                data: MmapOrVec::from_vec((0..nodes).map(|i| i as i64).collect()),
                nulls: MmapOrVec::from_vec(vec![0; nodes]),
            })
            .collect();
        let text = (0..3)
            .map(|field| {
                let mut offsets = MmapOrVec::from_vec(vec![0]);
                let mut bytes = MmapBytes::new();
                for id in 0..nodes {
                    let s = if field == 2 {
                        "seed".into()
                    } else {
                        id.to_string()
                    };
                    bytes.extend(s.as_bytes()).unwrap();
                    offsets.try_push(bytes.len() as u64).unwrap();
                }
                Text {
                    offsets,
                    bytes,
                    nulls: MmapOrVec::from_vec(vec![0; nodes]),
                }
            })
            .collect();
        Self {
            numeric,
            text,
            tombstones: vec![0; nodes],
        }
    }
    fn copied(&self, incoming: &[String; 3], reserve: bool) -> Self {
        let numeric = self
            .numeric
            .iter()
            .map(|n| Numeric {
                data: copy(&n.data, 1, reserve),
                nulls: copy(&n.nulls, 1, reserve),
            })
            .collect();
        let text = self
            .text
            .iter()
            .zip(incoming)
            .map(|(s, value)| Text {
                offsets: copy(&s.offsets, 1, reserve),
                nulls: copy(&s.nulls, 1, reserve),
                bytes: if reserve {
                    MmapBytes::Heap {
                        data: reserved(s.bytes.slice(0, s.bytes.len()), value.len()),
                    }
                } else {
                    s.bytes.clone()
                },
            })
            .collect();
        Self {
            numeric,
            text,
            tombstones: if reserve {
                reserved(&self.tombstones, 1)
            } else {
                self.tombstones.clone()
            },
        }
    }
    fn append(&mut self, id: usize, incoming: &[String; 3]) {
        for (i, n) in self.numeric.iter_mut().enumerate() {
            n.data.try_push(if i == 0 { id as i64 } else { 0 }).unwrap();
            n.nulls.try_push(u8::from(i != 0)).unwrap();
        }
        for (s, value) in self.text.iter_mut().zip(incoming) {
            s.bytes.extend(value.as_bytes()).unwrap();
            s.offsets.try_push(s.bytes.len() as u64).unwrap();
            s.nulls.try_push(0).unwrap();
        }
        self.tombstones.push(0);
    }
    fn capacity(&self) -> Vec<usize> {
        fn cap<T: MmapPod>(b: &MmapOrVec<T>) -> usize {
            match b {
                MmapOrVec::Heap { data } => data.capacity(),
                _ => panic!("expected heap"),
            }
        }
        let mut out = vec![self.tombstones.capacity()];
        for n in &self.numeric {
            out.extend([cap(&n.data), cap(&n.nulls)]);
        }
        for s in &self.text {
            let MmapBytes::Heap { data } = &s.bytes else {
                panic!("expected heap")
            };
            out.extend([cap(&s.offsets), data.capacity(), cap(&s.nulls)]);
        }
        out
    }
    fn verify(&self, nodes: usize, appended: usize) {
        assert_eq!(self.tombstones, vec![0; nodes + appended]);
        for (field, n) in self.numeric.iter().enumerate() {
            assert_eq!(n.data.len(), nodes + appended);
            assert_eq!(n.nulls.len(), nodes + appended);
            for id in 0..nodes + appended {
                let omitted = id >= nodes && field > 0;
                assert_eq!(n.data.as_slice()[id], if omitted { 0 } else { id as i64 });
                assert_eq!(n.nulls.as_slice()[id], u8::from(omitted));
            }
        }
        for (field, s) in self.text.iter().enumerate() {
            assert_eq!(s.offsets.len(), nodes + appended + 1);
            assert_eq!(s.nulls.as_slice(), vec![0; nodes + appended]);
            for id in 0..nodes + appended {
                let value = if field < 2 {
                    id.to_string()
                } else if id < nodes {
                    "seed".into()
                } else {
                    "new".into()
                };
                let offsets = s.offsets.as_slice();
                assert_eq!(
                    s.bytes
                        .slice(offsets[id] as usize, offsets[id + 1] as usize),
                    value.as_bytes()
                );
            }
            assert_eq!(
                s.offsets.as_slice().last().copied(),
                Some(s.bytes.len() as u64)
            );
        }
    }
}
fn ns(t: Instant) -> u64 {
    t.elapsed().as_nanos() as u64
}
fn summary(mut values: Vec<u64>) -> Json {
    values.sort_unstable();
    let sum: u128 = values.iter().map(|n| *n as u128).sum();
    json!({"mean_ns":sum as f64/values.len() as f64,"sum_ns":sum.to_string(),"min_ns":values[0],"median_ns":values[values.len()/2],"p95_ns":values[(values.len()*95).div_ceil(100)-1],"max_ns":values[values.len()-1]})
}
fn main() -> Result<(), Error> {
    let args: Vec<String> = std::env::args().collect();
    let nodes: usize = args[1].parse()?;
    let width: usize = args[2].parse()?;
    let rounds: usize = args[3].parse()?;
    let warmup: usize = args[4].parse()?;
    let reverse: usize = args[5].parse()?;
    let base = Columns::seed(nodes, width);
    let first = texts(nodes);
    let second = texts(nodes + 1);
    let mut reference = base.copied(&first, false);
    reference.append(nodes, &first);
    reference.verify(nodes, 1);
    let caps = reference.capacity();
    drop(reference);
    let mut check = base.copied(&first, true);
    check.append(nodes, &first);
    check.verify(nodes, 1);
    assert_eq!(
        check.capacity(),
        caps,
        "prototype must match actual reference post-first capacities"
    );
    check.append(nodes + 1, &second);
    check.verify(nodes, 2);
    drop(check);
    let mut records = Vec::new();
    for reserve in if reverse == 0 {
        [false, true]
    } else {
        [true, false]
    } {
        let mut samples = Vec::new();
        for event in 0..warmup + rounds {
            let clock = Instant::now();
            let mut copied = black_box(&base).copied(black_box(&first), reserve);
            copied.append(nodes, &first);
            let first_ns = ns(clock);
            let clock = Instant::now();
            copied.append(nodes + 1, &second);
            let second_ns = ns(clock);
            black_box(&copied).verify(nodes, 2);
            let clock = Instant::now();
            drop(copied);
            let drop_ns = ns(clock);
            if event >= warmup {
                samples.push([first_ns, second_ns, drop_ns]);
            }
        }
        let stages:Vec<Json>=["clone_first_append","second_append","drop","lifecycle_sum"].into_iter().enumerate().map(|(i,name)| {
            json!({"name":name,"timing":summary(samples.iter().map(|s|if i==3 {s.iter().sum()} else {s[i]}).collect())})
        }).collect();
        records.push(json!({"reserve":reserve,"nodes":nodes,"width":width,"rounds":rounds,"warmup":warmup,"stages":stages,"samples_ns":samples,"post_first_capacity":caps,"oracle":{"passed":true,"full_values_nulls_offsets_checked_each_event":true}}));
    }
    base.verify(nodes, 0);
    println!("{}", json!({"records":records}));
    Ok(())
}
