//! Native HOPE companion decoder — byte-level I/O.
//!
//! Loads a TorchScript HOPE model (.pt) and generates companion responses
//! from raw UTF-8 byte streams.
//!
//! Input format: "[CTX] {context_text} [USR] {user_message} [CRT] "
//! Output: raw UTF-8 bytes decoded to String.
//!
//! TorchScript interface:
//!   forward(tokens: Tensor[Long, (B, S)]) -> Tensor[(B, S, 256)]
//!   generate(prompt_bytes: List[int], max_new: int) -> List[int]

use std::path::Path;
use tch::{CModule, IValue, Tensor};

pub struct CompanionDecoder {
    model: CModule,
}

impl CompanionDecoder {
    pub fn load(model_path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let model = CModule::load(model_path)?;
        Ok(Self { model })
    }

    /// Generate a response given context text and user message.
    /// Input is formatted as bytes: "[CTX] {context} [USR] {message} [CRT] "
    /// Output is decoded from bytes to UTF-8 string.
    pub fn generate(&self, context_text: &str, user_message: &str, max_tokens: usize) -> String {
        let prompt = format!("[CTX] {} [USR] {} [CRT] ", context_text, user_message);
        let prompt_bytes: Vec<i64> = prompt.bytes().map(|b| b as i64).collect();

        // Call model.generate(prompt_bytes: List[int], max_new: int) -> List[int]
        let prompt_tensor = Tensor::from_slice(&prompt_bytes);
        let max_new = IValue::Int(max_tokens as i64);

        let result = match self.model.forward_is(&[
            IValue::Tensor(prompt_tensor),
            max_new,
        ]) {
            Ok(out) => out,
            Err(_) => return String::new(),
        };

        // Extract List[int] from IValue
        let byte_ids: Vec<i64> = match result {
            IValue::IntList(v) => v,
            _ => return String::new(),
        };

        // Convert byte values to UTF-8
        let bytes: Vec<u8> = byte_ids.iter()
            .filter_map(|&b| if b >= 0 && b < 256 { Some(b as u8) } else { None })
            .collect();

        let text = String::from_utf8_lossy(&bytes).into_owned();

        // Capitalise first letter, add period if missing
        let mut text = text.trim().to_string();
        if let Some(c) = text.get_mut(0..1) {
            c.make_ascii_uppercase();
        }
        if !text.is_empty()
            && !text.ends_with('.')
            && !text.ends_with('!')
            && !text.ends_with('?')
        {
            text.push('.');
        }
        text
    }
}
