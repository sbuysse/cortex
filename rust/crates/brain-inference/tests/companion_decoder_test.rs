#[cfg(test)]
mod tests {
    use brain_inference::CompanionDecoder;
    use std::path::Path;

    fn decoder_paths() -> Option<(std::path::PathBuf, std::path::PathBuf)> {
        let model_p = Path::new("/opt/brain/outputs/cortex/companion_decoder/companion_decoder_ts.pt");
        let vocab_p = Path::new("/opt/brain/outputs/cortex/companion_decoder/companion_vocab.json");
        if model_p.exists() && vocab_p.exists() {
            Some((model_p.to_path_buf(), vocab_p.to_path_buf()))
        } else {
            None
        }
    }

    #[test]
    #[ignore]
    fn test_load_returns_ok_when_file_exists() {
        if let Some((mp, _vp)) = decoder_paths() {
            let dec = CompanionDecoder::load(&mp);
            assert!(dec.is_ok(), "Failed to load: {:?}", dec.err());
        }
    }

    #[test]
    #[ignore]
    fn test_generate_returns_nonempty_string() {
        if let Some((mp, _vp)) = decoder_paths() {
            let dec = CompanionDecoder::load(&mp).unwrap();
            let context = "Hello";
            let message = "How are you?";
            let response = dec.generate(context, message, 30);
            assert!(response.len() < 500);
        }
    }

    #[test]
    #[ignore]
    fn test_generate_grounded_returns_nonempty_string() {
        if let Some((mp, _vp)) = decoder_paths() {
            let dec = CompanionDecoder::load(&mp).unwrap();
            let brain_vec: Vec<f32> = (0..512).map(|i| (i as f32 * 0.01_f32).sin()).collect();
            let context = "Hello";
            let message = "How are you?";
            let response = dec.generate_grounded(&brain_vec, context, message, 30);
            assert!(response.len() < 500);
        }
    }
}
