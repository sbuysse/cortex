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
use tch::{CModule, IValue};

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
        let result = match self.model.method_is("generate", &[
            IValue::IntList(prompt_bytes),
            IValue::Int(max_tokens as i64),
        ]) {
            Ok(out) => out,
            Err(e) => {
                eprintln!("[companion_decoder] generate failed: {e}");
                return String::new();
            }
        };

        // Extract List[int] from IValue
        let byte_ids: Vec<i64> = match result {
            IValue::IntList(v) => v,
            other => {
                eprintln!("[companion_decoder] unexpected IValue variant: {other:?}");
                return String::new();
            }
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

    /// Generate a response conditioned on the live Brain state vector.
    ///
    /// `brain_vec`: 512-dim unit vector from `compose_brain_state()`.
    /// Floats are packed as `round(x * 1000) as i64` for TorchScript compatibility.
    ///
    /// Falls back to `generate()` if the TorchScript model does not export
    /// `generate_grounded` (e.g., old model without grounded training).
    pub fn generate_grounded(
        &self,
        brain_vec: &[f32],
        context_text: &str,
        user_message: &str,
        max_tokens: usize,
    ) -> String {
        let brain_ints: Vec<i64> = brain_vec.iter()
            .map(|&x| (x * 1000.0).round() as i64)
            .collect();

        let prompt = format!("[CTX] {} [USR] {} [CRT] ", context_text, user_message);
        let prompt_bytes: Vec<i64> = prompt.bytes().map(|b| b as i64).collect();

        let result = match self.model.method_is("generate_grounded", &[
            IValue::IntList(brain_ints),
            IValue::IntList(prompt_bytes),
            IValue::Int(max_tokens as i64),
        ]) {
            Ok(out) => out,
            Err(e) => {
                eprintln!("[companion_decoder] generate_grounded failed ({e}) — fallback to generate()");
                return self.generate(context_text, user_message, max_tokens);
            }
        };

        let byte_ids: Vec<i64> = match result {
            IValue::IntList(v) => v,
            other => {
                eprintln!("[companion_decoder] generate_grounded unexpected output: {other:?}");
                return self.generate(context_text, user_message, max_tokens);
            }
        };

        let bytes: Vec<u8> = byte_ids.iter()
            .filter_map(|&b| if b >= 0 && b < 256 { Some(b as u8) } else { None })
            .collect();

        let text = String::from_utf8_lossy(&bytes).into_owned();
        let mut text = text.trim().to_string();
        if let Some(c) = text.get_mut(0..1) { c.make_ascii_uppercase(); }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brain_vec_packing_roundtrip() {
        // Floats in [-1, 1] packed as i64 (×1000) and unpacked (/1000) lose < 0.001
        let values: Vec<f32> = vec![-1.0, -0.5, 0.0, 0.333, 0.999, 1.0];
        for &v in &values {
            let packed = (v * 1000.0).round() as i64;
            let unpacked = packed as f32 / 1000.0;
            assert!((unpacked - v).abs() < 0.001, "v={v}, unpacked={unpacked}");
        }
    }
}
