//! Streaming Agentic FSM Parser and Permission Gate (@C:AGENT).

use crate::events::{AgentBlock, ToolStatus};
use std::sync::mpsc::Sender;
use std::time::Instant;

#[derive(Debug, PartialEq, Eq)]
enum ParserState {
    Text,
    InsideThink,
    InsideTool,
}

pub struct AgentStreamParser {
    state: ParserState,
    buffer: String,
    think_start: Option<Instant>,
    tool_counter: u32,
    tx: Sender<AgentBlock>,
}

impl AgentStreamParser {
    pub fn new(tx: Sender<AgentBlock>) -> Self {
        Self {
            state: ParserState::Text,
            buffer: String::with_capacity(4096),
            think_start: None,
            tool_counter: 0,
            tx,
        }
    }

    pub fn feed_token(&mut self, token: &str) {
        self.buffer.push_str(token);
        self.process_buffer();
    }

    fn process_buffer(&mut self) {
        loop {
            match self.state {
                ParserState::Text => {
                    if let Some(idx) = self.buffer.find("<think>") {
                        let text_chunk = self.buffer[..idx].to_string();
                        if !text_chunk.is_empty() {
                            let _ = self.tx.send(AgentBlock::ResponseChunk(text_chunk));
                        }
                        self.buffer.drain(..idx + 7);
                        self.state = ParserState::InsideThink;
                        self.think_start = Some(Instant::now());
                    } else if let Some(idx) = self.buffer.find("<tool_call>") {
                        let text_chunk = self.buffer[..idx].to_string();
                        if !text_chunk.is_empty() {
                            let _ = self.tx.send(AgentBlock::ResponseChunk(text_chunk));
                        }
                        self.buffer.drain(..idx + 11);
                        self.state = ParserState::InsideTool;
                    } else {
                        // Flush all safe non-tag text
                        if self.buffer.len() > 16 {
                            let safe_len = self.buffer.len() - 16;
                            let flush_str = self.buffer.drain(..safe_len).collect::<String>();
                            let _ = self.tx.send(AgentBlock::ResponseChunk(flush_str));
                        }
                        break;
                    }
                }
                ParserState::InsideThink => {
                    if let Some(idx) = self.buffer.find("</think>") {
                        let body = self.buffer.drain(..idx).collect::<String>();
                        self.buffer.drain(..8); // skip </think>
                        let elapsed_ms = self
                            .think_start
                            .take()
                            .map(|t| t.elapsed().as_millis() as u64)
                            .unwrap_or(0);
                        let summary = body.lines().next().unwrap_or("Reasoning trace").to_string();
                        let _ = self.tx.send(AgentBlock::Thinking {
                            elapsed_ms,
                            summary,
                            body,
                        });
                        self.state = ParserState::Text;
                    } else {
                        break;
                    }
                }
                ParserState::InsideTool => {
                    if let Some(idx) = self.buffer.find("</tool_call>") {
                        let call_str = self.buffer.drain(..idx).collect::<String>();
                        self.buffer.drain(..12); // skip </tool_call>
                        self.tool_counter += 1;

                        let (tool, args, diff) = Self::parse_tool_payload(&call_str);
                        let _ = self.tx.send(AgentBlock::ToolUse {
                            id: self.tool_counter,
                            tool,
                            args,
                            status: ToolStatus::Running,
                            diff,
                        });
                        self.state = ParserState::Text;
                    } else {
                        break;
                    }
                }
            }
        }
    }

    fn parse_tool_payload(raw: &str) -> (String, String, Option<String>) {
        if let Some(pos) = raw.find('(') {
            let tool = raw[..pos].trim().to_string();
            let args_part = raw[pos + 1..].trim_end_matches(')').trim();
            let diff = if tool == "Patch" {
                Some(args_part.to_string())
            } else {
                None
            };
            (tool, args_part.to_string(), diff)
        } else {
            ("Shell".to_string(), raw.trim().to_string(), None)
        }
    }

    pub fn finish(mut self) {
        if !self.buffer.is_empty() {
            let _ = self
                .tx
                .send(AgentBlock::ResponseChunk(std::mem::take(&mut self.buffer)));
        }
    }
}
