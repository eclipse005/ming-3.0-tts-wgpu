//! Task prompt construction (port of tts_tasks.py).

pub struct Controls {
    pub speed: Option<String>,
    pub pitch: Option<String>,
    pub volume: Option<String>,
    pub emotion: Option<String>,
    pub dialect: Option<String>,
    pub style: Option<String>,
    pub ip: Option<String>,
}

pub struct SynthRequest {
    pub prompt: String,
    pub text: String,
    pub text_language: Option<String>,
    pub instruction: Option<String>,
    pub taskname: String,
    pub use_spk_emb: bool,
    pub use_zero_spk_emb: bool,
    pub prompt_wav_path: Option<String>,
    pub prompt_text: Option<String>,
    pub cfg: f64,
    pub sigma: f64,
    pub temperature: f64,
}

#[allow(clippy::too_many_arguments)]
pub fn build_task_generation_kwargs(
    task: &str,
    text: &str,
    language_id: Option<&str>,
    instruction: Option<&str>,
    controls: Option<&Controls>,
    instruction_schema: Option<&str>,
    prompt_wav_path: Option<&str>,
    prompt_text: Option<&str>,
    spk_emb_path: Option<&str>,
    use_spk_emb: bool,
    use_zero_spk_emb: bool,
    cfg: f64,
    sigma: f64,
    temperature: f64,
) -> anyhow::Result<SynthRequest> {
    let _ = (task, text, language_id, instruction, controls, instruction_schema,
             prompt_wav_path, prompt_text, spk_emb_path, use_spk_emb,
             use_zero_spk_emb, cfg, sigma, temperature);
    todo!()
}
