use super::{JsonQueryParameterError, JsonQueryParameterErrorKind};

const MAX_ACCEPTED_CONTAINER_DEPTH: usize = 127;

#[derive(Clone, PartialEq, Eq)]
enum PathPart {
    Key(String),
    Index(usize),
}

struct NumberFailure {
    path: Vec<PathPart>,
    kind: JsonQueryParameterErrorKind,
}

/// Validate numeric lexemes below `pointer` without materializing them as f64.
///
/// An empty pointer validates the whole value; `[]` matches any array index.
/// Malformed JSON is deliberately left to the caller's ordinary serde_json
/// parse so its established syntax error remains authoritative.
pub fn validate_json_query_numbers_at(
    source: &str,
    pointer: &[&str],
) -> Result<(), JsonQueryParameterError> {
    let mut scanner = Scanner {
        source: source.as_bytes(),
        cursor: 0,
        path: Vec::new(),
        pointer,
        failures: Vec::new(),
    };
    validate(&mut scanner)
}

fn validate(scanner: &mut Scanner<'_>) -> Result<(), JsonQueryParameterError> {
    if scanner.value(0).is_err() {
        return Ok(());
    }
    scanner.whitespace();
    if scanner.cursor != scanner.source.len() {
        return Ok(());
    }
    match scanner.failures.first() {
        Some(failure) => Err(JsonQueryParameterError {
            path: render_path(&failure.path, scanner.pointer),
            kind: failure.kind,
        }),
        None => Ok(()),
    }
}

struct Scanner<'a> {
    source: &'a [u8],
    cursor: usize,
    path: Vec<PathPart>,
    pointer: &'a [&'a str],
    failures: Vec<NumberFailure>,
}

impl Scanner<'_> {
    fn value(&mut self, depth: usize) -> Result<(), ()> {
        self.whitespace();
        match self.peek().ok_or(())? {
            b'{' if depth < MAX_ACCEPTED_CONTAINER_DEPTH => self.object(depth + 1),
            b'[' if depth < MAX_ACCEPTED_CONTAINER_DEPTH => self.array(depth + 1),
            b'"' => self.string().map(|_| ()),
            b't' => self.keyword(b"true"),
            b'f' => self.keyword(b"false"),
            b'n' => self.keyword(b"null"),
            b'-' | b'0'..=b'9' => self.number(),
            _ => Err(()),
        }
    }

    fn object(&mut self, depth: usize) -> Result<(), ()> {
        self.expect(b'{')?;
        self.whitespace();
        if self.take(b'}') {
            return Ok(());
        }
        let mut seen = std::collections::HashSet::new();
        loop {
            self.whitespace();
            let key_source = self.string()?;
            let key: String = serde_json::from_slice(key_source).map_err(|_| ())?;
            self.whitespace();
            self.expect(b':')?;
            if !seen.insert(key.clone()) {
                self.discard_shadowed_failures(&key);
            }
            self.path.push(PathPart::Key(key));
            self.value(depth)?;
            self.path.pop();
            self.whitespace();
            if self.take(b'}') {
                return Ok(());
            }
            self.expect(b',')?;
        }
    }

    fn array(&mut self, depth: usize) -> Result<(), ()> {
        self.expect(b'[')?;
        self.whitespace();
        if self.take(b']') {
            return Ok(());
        }
        let mut index = 0;
        loop {
            self.path.push(PathPart::Index(index));
            self.value(depth)?;
            self.path.pop();
            index += 1;
            self.whitespace();
            if self.take(b']') {
                return Ok(());
            }
            self.expect(b',')?;
        }
    }

    fn string(&mut self) -> Result<&[u8], ()> {
        let start = self.cursor;
        self.expect(b'"')?;
        while let Some(byte) = self.peek() {
            match byte {
                b'"' => {
                    self.cursor += 1;
                    return Ok(&self.source[start..self.cursor]);
                }
                b'\\' => {
                    self.cursor += 1;
                    let escaped = self.peek().ok_or(())?;
                    self.cursor += 1;
                    if escaped == b'u' {
                        for _ in 0..4 {
                            if !self.peek().is_some_and(|b| b.is_ascii_hexdigit()) {
                                return Err(());
                            }
                            self.cursor += 1;
                        }
                    } else if !matches!(
                        escaped,
                        b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't'
                    ) {
                        return Err(());
                    }
                }
                0x00..=0x1f => return Err(()),
                _ => self.cursor += 1,
            }
        }
        Err(())
    }

    fn number(&mut self) -> Result<(), ()> {
        let start = self.cursor;
        self.take(b'-');
        match self.peek().ok_or(())? {
            b'0' => self.cursor += 1,
            b'1'..=b'9' => {
                self.cursor += 1;
                while self.peek().is_some_and(|b| b.is_ascii_digit()) {
                    self.cursor += 1;
                }
            }
            _ => return Err(()),
        }
        let mut integer = true;
        if self.take(b'.') {
            integer = false;
            self.digits()?;
        }
        if self.peek().is_some_and(|b| matches!(b, b'e' | b'E')) {
            integer = false;
            self.cursor += 1;
            if self.peek().is_some_and(|b| matches!(b, b'+' | b'-')) {
                self.cursor += 1;
            }
            self.digits()?;
        }
        if self.in_target() {
            let lexeme = std::str::from_utf8(&self.source[start..self.cursor]).map_err(|_| ())?;
            let kind = if integer {
                lexeme
                    .parse::<i64>()
                    .err()
                    .map(|_| JsonQueryParameterErrorKind::IntegerOutOfRange)
            } else {
                lexeme
                    .parse::<f64>()
                    .ok()
                    .filter(|value| value.is_finite())
                    .is_none()
                    .then_some(JsonQueryParameterErrorKind::NonFiniteFloat)
            };
            if let Some(kind) = kind {
                self.failures.push(NumberFailure {
                    path: self.path.clone(),
                    kind,
                });
            }
        }
        Ok(())
    }

    fn in_target(&self) -> bool {
        self.path.len() >= self.pointer.len()
            && self
                .pointer
                .iter()
                .zip(&self.path)
                .all(|(wanted, actual)| match actual {
                    PathPart::Key(key) => key == wanted,
                    PathPart::Index(_) => *wanted == "[]",
                })
    }

    fn discard_shadowed_failures(&mut self, key: &str) {
        let prefix_len = self.path.len();
        self.failures.retain(|failure| {
            failure.path.get(..prefix_len) != Some(self.path.as_slice())
                || !matches!(failure.path.get(prefix_len), Some(PathPart::Key(failure_key)) if failure_key == key)
        });
    }
}

fn render_path(path: &[PathPart], pointer: &[&str]) -> String {
    let mut result = "$".to_string();
    for (position, part) in path.iter().enumerate() {
        if position < pointer.len() {
            if pointer[position] == "[]" {
                if let PathPart::Index(index) = part {
                    result.push_str(&format!("[{index}]"));
                }
            }
            continue;
        }
        match part {
            PathPart::Index(index) => result.push_str(&format!("[{index}]")),
            PathPart::Key(key)
                if key
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_') =>
            {
                result.push('.');
                result.push_str(key);
            }
            PathPart::Key(key) => {
                result.push('[');
                result.push_str(&serde_json::to_string(key).expect("JSON key serialization"));
                result.push(']');
            }
        }
    }
    result
}

impl Scanner<'_> {
    fn keyword(&mut self, keyword: &[u8]) -> Result<(), ()> {
        if self.source.get(self.cursor..self.cursor + keyword.len()) == Some(keyword) {
            self.cursor += keyword.len();
            Ok(())
        } else {
            Err(())
        }
    }

    fn digits(&mut self) -> Result<(), ()> {
        let start = self.cursor;
        while self.peek().is_some_and(|b| b.is_ascii_digit()) {
            self.cursor += 1;
        }
        (self.cursor != start).then_some(()).ok_or(())
    }

    fn whitespace(&mut self) {
        while self
            .peek()
            .is_some_and(|b| matches!(b, b' ' | b'\n' | b'\r' | b'\t'))
        {
            self.cursor += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.source.get(self.cursor).copied()
    }

    fn take(&mut self, byte: u8) -> bool {
        if self.peek() == Some(byte) {
            self.cursor += 1;
            true
        } else {
            false
        }
    }

    fn expect(&mut self, byte: u8) -> Result<(), ()> {
        self.take(byte).then_some(()).ok_or(())
    }
}
