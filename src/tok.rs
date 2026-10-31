//! Qwen2 BPE tokenizer wrapper (loads the checkpoint tokenizer.json).

pub struct Tok {
    _priv: (),
}

impl Tok {
    pub fn load(model_dir: &std::path::Path) -> anyhow::Result<Self> {
        let _ = model_dir;
        todo!()
    }

    pub fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        let _ = text;
        todo!()
    }

    pub fn token_id(&self, token: &str) -> anyhow::Result<u32> {
        let _ = token;
        todo!()
    }
}
