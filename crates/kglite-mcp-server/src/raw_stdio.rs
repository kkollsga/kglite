use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;

use futures::SinkExt;
use rmcp::model::{ErrorData, GetExtensions, JsonRpcMessage};
use rmcp::service::{RxJsonRpcMessage, TxJsonRpcMessage};
use rmcp::transport::async_rw::{JsonRpcMessageCodec, JsonRpcMessageCodecError};
use rmcp::transport::Transport;
use rmcp::RoleServer;
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncWrite, BufReader};
use tokio::sync::Mutex;
use tokio_util::bytes::BytesMut;
use tokio_util::codec::{Decoder, FramedWrite};

const MAX_ACCEPTED_CONTAINER_DEPTH: usize = 127;

/// Exact JSON-RPC request text retained only for the lifetime of one request.
#[derive(Clone, Debug)]
pub(crate) struct RawJsonRpcRequest {
    pub(crate) source: Arc<str>,
    pub(crate) recovered: bool,
}

struct RawLineCodec<T> {
    inner: JsonRpcMessageCodec<T>,
}

impl<T> Default for RawLineCodec<T> {
    fn default() -> Self {
        Self {
            inner: JsonRpcMessageCodec::default(),
        }
    }
}

impl<T: serde::de::DeserializeOwned> Decoder for RawLineCodec<T> {
    type Item = (T, Arc<str>);
    type Error = JsonRpcMessageCodecError;

    fn decode(&mut self, input: &mut BytesMut) -> Result<Option<Self::Item>, Self::Error> {
        loop {
            let before = input.len();
            let raw = input
                .iter()
                .position(|byte| *byte == b'\n')
                .map(|end| input[..end].to_vec());
            match self.inner.decode(input)? {
                Some(message) => {
                    let raw = raw.expect("the line codec returned a complete line");
                    return Ok(Some((message, raw_text(&raw)?)));
                }
                None if input.len() < before => continue,
                None => return Ok(None),
            }
        }
    }

    fn decode_eof(&mut self, input: &mut BytesMut) -> Result<Option<Self::Item>, Self::Error> {
        let raw = input.as_ref().strip_suffix(b"\r").unwrap_or(input.as_ref());
        let raw = raw_text(raw)?;
        self.inner
            .decode_eof(input)
            .map(|message| message.map(|message| (message, raw)))
    }
}

fn raw_text(line: &[u8]) -> Result<Arc<str>, JsonRpcMessageCodecError> {
    const UTF8_BOM: &[u8] = b"\xEF\xBB\xBF";
    let line = line.strip_suffix(b"\r").unwrap_or(line);
    let line = line.strip_prefix(UTF8_BOM).unwrap_or(line);
    std::str::from_utf8(line).map(Arc::from).map_err(|error| {
        serde_json::Error::io(std::io::Error::new(std::io::ErrorKind::InvalidData, error)).into()
    })
}

fn recover_query_message(
    raw: &str,
    query_routes: &HashMap<String, &'static [&'static str]>,
) -> Option<RxJsonRpcMessage<RoleServer>> {
    let routing = repair_nonfinite_numbers(raw, &[])?;
    let envelope: serde_json::Value = serde_json::from_slice(&routing).ok()?;
    if envelope.get("method")?.as_str()? != "tools/call" {
        return None;
    }
    let name = envelope.pointer("/params/name")?.as_str()?;
    let pointer = query_routes.get(name)?;
    let repaired = repair_nonfinite_numbers(raw, pointer)?;
    serde_json::from_slice(&repaired).ok()
}

fn repair_nonfinite_numbers(source: &str, pointer: &[&str]) -> Option<Vec<u8>> {
    let mut scanner = RepairScanner {
        source: source.as_bytes(),
        repaired: source.as_bytes().to_vec(),
        cursor: 0,
        path: Vec::new(),
        pointer,
    };
    scanner.value(0).ok()?;
    scanner.whitespace();
    (scanner.cursor == scanner.source.len()).then_some(scanner.repaired)
}

enum RepairPathPart {
    Key(String),
    Index,
}

struct RepairScanner<'a> {
    source: &'a [u8],
    repaired: Vec<u8>,
    cursor: usize,
    path: Vec<RepairPathPart>,
    pointer: &'a [&'a str],
}

impl RepairScanner<'_> {
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
        loop {
            self.whitespace();
            let key_source = self.string()?;
            let key = serde_json::from_slice(key_source).map_err(|_| ())?;
            self.whitespace();
            self.expect(b':')?;
            self.path.push(RepairPathPart::Key(key));
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
        loop {
            self.path.push(RepairPathPart::Index);
            self.value(depth)?;
            self.path.pop();
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
                            if !self.peek().is_some_and(|byte| byte.is_ascii_hexdigit()) {
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
                while self.peek().is_some_and(|byte| byte.is_ascii_digit()) {
                    self.cursor += 1;
                }
            }
            _ => return Err(()),
        }
        let mut explicit_float = false;
        if self.take(b'.') {
            explicit_float = true;
            self.digits()?;
        }
        if self.peek().is_some_and(|byte| matches!(byte, b'e' | b'E')) {
            explicit_float = true;
            self.cursor += 1;
            if self.peek().is_some_and(|byte| matches!(byte, b'+' | b'-')) {
                self.cursor += 1;
            }
            self.digits()?;
        }
        if explicit_float && self.in_target() {
            let lexeme = std::str::from_utf8(&self.source[start..self.cursor]).map_err(|_| ())?;
            if !lexeme.parse::<f64>().is_ok_and(f64::is_finite) {
                self.repaired[start] = b'0';
                self.repaired[start + 1..self.cursor].fill(b' ');
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
                    RepairPathPart::Key(key) => key == wanted,
                    RepairPathPart::Index => *wanted == "[]",
                })
    }

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
        while self.peek().is_some_and(|byte| byte.is_ascii_digit()) {
            self.cursor += 1;
        }
        (self.cursor != start).then_some(()).ok_or(())
    }

    fn whitespace(&mut self) {
        while self
            .peek()
            .is_some_and(|byte| matches!(byte, b' ' | b'\n' | b'\r' | b'\t'))
        {
            self.cursor += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.source.get(self.cursor).copied()
    }

    fn take(&mut self, expected: u8) -> bool {
        if self.peek() == Some(expected) {
            self.cursor += 1;
            true
        } else {
            false
        }
    }

    fn expect(&mut self, expected: u8) -> Result<(), ()> {
        self.take(expected).then_some(()).ok_or(())
    }
}

pub(crate) struct QueryAwareStdio<R, W> {
    read: BufReader<R>,
    line: Vec<u8>,
    decoder: RawLineCodec<RxJsonRpcMessage<RoleServer>>,
    write: Arc<Mutex<Option<FramedWrite<W, JsonRpcMessageCodec<TxJsonRpcMessage<RoleServer>>>>>>,
    query_routes: Arc<HashMap<String, &'static [&'static str]>>,
}

impl<R, W> QueryAwareStdio<R, W>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    pub(crate) fn new(
        read: R,
        write: W,
        query_routes: HashMap<String, &'static [&'static str]>,
    ) -> Self {
        Self {
            read: BufReader::new(read),
            line: Vec::new(),
            decoder: RawLineCodec::default(),
            write: Arc::new(Mutex::new(Some(FramedWrite::new(
                write,
                JsonRpcMessageCodec::default(),
            )))),
            query_routes: Arc::new(query_routes),
        }
    }
}

impl<R, W> Transport<RoleServer> for QueryAwareStdio<R, W>
where
    R: AsyncRead + Send + Unpin,
    W: AsyncWrite + Send + Unpin + 'static,
{
    type Error = std::io::Error;

    fn send(
        &mut self,
        item: TxJsonRpcMessage<RoleServer>,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send + 'static {
        let write = self.write.clone();
        async move {
            let mut guard = write.lock().await;
            match guard.as_mut() {
                Some(writer) => writer.send(item).await.map_err(Into::into),
                None => Err(std::io::Error::new(
                    std::io::ErrorKind::NotConnected,
                    "Transport is closed",
                )),
            }
        }
    }

    async fn receive(&mut self) -> Option<RxJsonRpcMessage<RoleServer>> {
        loop {
            let decoded = match self.read.read_until(b'\n', &mut self.line).await {
                Ok(0) if self.line.is_empty() => return None,
                Ok(0) => {
                    let mut final_line = BytesMut::from(self.line.as_slice());
                    self.decoder.decode_eof(&mut final_line)
                }
                Ok(_) if !self.line.ends_with(b"\n") => continue,
                Ok(_) => {
                    let mut complete_line = BytesMut::from(self.line.as_slice());
                    self.decoder.decode(&mut complete_line)
                }
                Err(error) => {
                    tracing::error!("Error reading from stream: {error}");
                    return None;
                }
            };
            let raw = raw_text(self.line.strip_suffix(b"\n").unwrap_or(&self.line)).ok();
            self.line.clear();
            match decoded {
                Ok(Some((mut message, raw))) => {
                    if let JsonRpcMessage::Request(request) = &mut message {
                        request.request.extensions_mut().insert(RawJsonRpcRequest {
                            source: raw,
                            recovered: false,
                        });
                    }
                    return Some(message);
                }
                Ok(None) => continue,
                Err(JsonRpcMessageCodecError::Serde(error)) => {
                    if let Some(raw) = raw {
                        if let Some(mut message) = recover_query_message(&raw, &self.query_routes) {
                            if let JsonRpcMessage::Request(request) = &mut message {
                                request.request.extensions_mut().insert(RawJsonRpcRequest {
                                    source: raw,
                                    recovered: true,
                                });
                            }
                            return Some(message);
                        }
                    }
                    match error.classify() {
                        serde_json::error::Category::Syntax | serde_json::error::Category::Eof => {
                            tracing::debug!("Ignoring unparsable incoming message: {error}");
                        }
                        serde_json::error::Category::Data | serde_json::error::Category::Io => {
                            let mut guard = self.write.lock().await;
                            let writer = guard.as_mut()?;
                            let response = TxJsonRpcMessage::<RoleServer>::error(
                                ErrorData::invalid_request("Invalid request", None),
                                None,
                            );
                            if writer.send(response).await.is_err() {
                                return None;
                            }
                        }
                    }
                }
                Err(error) => {
                    tracing::error!("Error reading from stream: {error}");
                    return None;
                }
            }
        }
    }

    async fn close(&mut self) -> Result<(), Self::Error> {
        self.write.lock().await.take();
        Ok(())
    }
}
