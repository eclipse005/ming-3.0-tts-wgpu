//! Task prompt construction (port of tts_tasks.py).
//!
//! The prompt/instruction strings and the JSON key order are part of the model
//! input protocol, so this port reproduces them verbatim: the structured
//! instruction is a JSON document with Python's json.dumps default separators
//! (", " and ": ") and the OrderedDict insertion order of the reference.
//!
//! Deliberate notes:
//! * DIALECT drops language_id/target_language exactly like the reference, but
//!   there is no warning channel here (the reference emits UserWarning).
//! * tts_tasks.build_task_generation_kwargs forwards **kwargs into the request
//!   dict. The only keyword the reference call sites use is spk_emb_path
//!   (ming_tts.synthesize passes it explicitly), so that is the one modeled;
//!   arbitrary extra kwargs are not.

// A line-for-line port of the reference helper set: not every helper is on the
// CLI path, and pruning them would make the next upstream diff harder to read.
#![allow(dead_code)]


use anyhow::{bail, Result};

/// Canonical user-side generation prompt.
pub const DEFAULT_TTS_PROMPT: &str =
    "Please generate speech based on the following description.\n";

/// Re-exported from the text frontend (tts_tasks re-exports it too).
pub const PARALINGUISTIC_TAGS: [&str; 14] = crate::text::PARALINGUISTIC_TAGS;

/// _TASK_ALIASES keys, in declaration order.
pub const TASKS: [&str; 12] = [
    "TTS",
    "SEED",
    "PLAIN",
    "MULTILINGUAL",
    "MULTILINGUAL_TTS",
    "BASIC",
    "EMOTION",
    "DIALECT",
    "IP",
    "VOICE_DESIGN",
    "STYLE",
    "FREEFORM",
];

/// The five languages added in Ming-3.0 on top of Chinese/English.
pub const MULTILINGUAL_LANGUAGE_IDS: [(&str, &str); 5] = [
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("de", "German"),
    ("fr", "French"),
    ("es", "Spanish"),
];

/// BASIC_ATTRIBUTE_VALUES, in declaration order.
pub const BASIC_ATTRIBUTE_VALUES: [(&str, [&str; 3]); 3] = [
    ("speed", ["慢速", "中速", "快速"]),
    ("pitch", ["低", "中", "高"]),
    ("volume", ["低", "中", "高"]),
];

pub const EMOTION_VALUES: [&str; 7] = ["高兴", "悲伤", "愤怒", "恐惧", "厌恶", "惊讶", "中性"];

pub const DIALECT_VALUES: [&str; 2] = ["chuanyu dialect", "cantonese"];

/// Recommended English built-in voices.
pub const EN_IP_VOICES: [&str; 3] = ["US_Emma", "UK_George", "UK_Charlotte"];

const ZH_FIELDS: [&str; 8] = ["序号", "说话人", "方言", "风格", "语速", "基频", "音量", "情感"];
const EN_FIELDS: [&str; 8] = [
    "id", "speaker", "dialect", "style", "speed", "pitch", "volume", "emotion",
];
const BGM_FIELDS: [&str; 6] = ["Genre", "Mood", "Instrument", "Theme", "ENV", "SNR"];

/// sorted(_TASK_ALIASES), used verbatim in the unsupported-task error message.
const SORTED_TASKS: [&str; 12] = [
    "BASIC",
    "DIALECT",
    "EMOTION",
    "FREEFORM",
    "IP",
    "MULTILINGUAL",
    "MULTILINGUAL_TTS",
    "PLAIN",
    "SEED",
    "STYLE",
    "TTS",
    "VOICE_DESIGN",
];

const STRUCTURED_TASKS: [&str; 5] = ["BASIC", "EMOTION", "DIALECT", "IP", "VOICE_DESIGN"];

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Controls {
    pub speed: Option<String>,
    pub pitch: Option<String>,
    pub volume: Option<String>,
    pub emotion: Option<String>,
    pub dialect: Option<String>,
    pub style: Option<String>,
    pub ip: Option<String>,
}

impl Controls {
    /// True when no control field is set (Python's empty dict).
    pub fn is_empty(&self) -> bool {
        self.speed.is_none()
            && self.pitch.is_none()
            && self.volume.is_none()
            && self.emotion.is_none()
            && self.dialect.is_none()
            && self.style.is_none()
            && self.ip.is_none()
    }

    /// (field name, is set) in declaration order.
    fn fields(&self) -> [(&'static str, bool); 7] {
        [
            ("speed", self.speed.is_some()),
            ("pitch", self.pitch.is_some()),
            ("volume", self.volume.is_some()),
            ("emotion", self.emotion.is_some()),
            ("dialect", self.dialect.is_some()),
            ("style", self.style.is_some()),
            ("ip", self.ip.is_some()),
        ]
    }
}

// ---------------------------------------------------------------------------
// SynthRequest
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
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
    /// Forwarded **kwargs of the reference (ming_tts.synthesize passes it).
    pub spk_emb_path: Option<String>,
}

impl SynthRequest {
    /// The reference's request dict as JSON. serde_json's default map is
    /// sorted, so this is byte-identical to
    /// json.dumps(request, ensure_ascii=False, sort_keys=True,
    ///            separators=(',', ':')).
    pub fn to_json_value(&self) -> serde_json::Value {
        let mut map = serde_json::Map::new();
        map.insert("prompt".into(), self.prompt.clone().into());
        map.insert("text".into(), self.text.clone().into());
        map.insert(
            "text_language".into(),
            self.text_language.clone().map(Into::into).unwrap_or(serde_json::Value::Null),
        );
        map.insert(
            "instruction".into(),
            self.instruction.clone().map(Into::into).unwrap_or(serde_json::Value::Null),
        );
        map.insert("taskname".into(), self.taskname.clone().into());
        map.insert("use_spk_emb".into(), self.use_spk_emb.into());
        map.insert("use_zero_spk_emb".into(), self.use_zero_spk_emb.into());
        map.insert(
            "prompt_wav_path".into(),
            self.prompt_wav_path.clone().map(Into::into).unwrap_or(serde_json::Value::Null),
        );
        map.insert(
            "prompt_text".into(),
            self.prompt_text.clone().map(Into::into).unwrap_or(serde_json::Value::Null),
        );
        map.insert(
            "spk_emb_path".into(),
            self.spk_emb_path.clone().map(Into::into).unwrap_or(serde_json::Value::Null),
        );
        map.insert("cfg".into(), json_float(self.cfg));
        map.insert("sigma".into(), json_float(self.sigma));
        map.insert("temperature".into(), json_float(self.temperature));
        serde_json::Value::Object(map)
    }

    pub fn to_canonical_json(&self) -> String {
        serde_json::to_string(&self.to_json_value()).expect("serde_json serialisation")
    }
}

fn json_float(value: f64) -> serde_json::Value {
    match serde_json::Number::from_f64(value) {
        Some(number) => serde_json::Value::Number(number),
        None => serde_json::Value::Null,
    }
}

// ---------------------------------------------------------------------------
// Python formatting helpers
// ---------------------------------------------------------------------------

/// json.dumps(str, ensure_ascii=False) for the body of the string.
fn json_escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{0008}' => out.push_str("\\b"),
            '\u{000C}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn json_string(text: &str) -> String {
    format!("\"{}\"", json_escape(text))
}

fn json_opt(value: Option<&str>) -> String {
    match value {
        Some(text) => json_string(text),
        None => "null".to_string(),
    }
}

/// repr(str) for the error messages (ASCII-safe subset of Python's repr).
fn py_repr(value: &str) -> String {
    let mut out = String::from("'");
    for c in value.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\'' => out.push_str("\\'"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\x{:02x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('\'');
    out
}

fn py_list_repr(values: &[&str]) -> String {
    let body: Vec<String> = values.iter().map(|value| py_repr(value)).collect();
    format!("[{}]", body.join(", "))
}

fn py_strip(text: &str) -> &str {
    crate::text::py_strip(text)
}

fn non_blank(value: Option<&str>) -> bool {
    match value {
        Some(text) => !py_strip(text).is_empty(),
        None => false,
    }
}

// ---------------------------------------------------------------------------
// Prompt helpers
// ---------------------------------------------------------------------------

/// _TARGET_LANGUAGE_NAMES lookup + fallback to the given string.
fn target_language_name(language: &str) -> String {
    let lowered = language.to_lowercase();
    let name = match lowered.as_str() {
        "zh" | "zh-cn" | "chinese" => "Chinese",
        "en" | "english" => "English",
        "ja" | "japanese" => "Japanese",
        "ko" | "korean" => "Korean",
        "de" | "german" => "German",
        "fr" | "french" => "French",
        "es" | "spanish" => "Spanish",
        "yue" | "zh-yue" | "cantonese" => "Chinese",
        _ => return language.to_string(),
    };
    name.to_string()
}

/// Return the user-side prompt; the target language line is opt-in.
pub fn build_tts_prompt(target_language: Option<&str>) -> String {
    match target_language {
        None => DEFAULT_TTS_PROMPT.to_string(),
        Some(value) => {
            let language = py_strip(value).to_string();
            if language.is_empty() {
                return DEFAULT_TTS_PROMPT.to_string();
            }
            let name = target_language_name(&language);
            format!("{DEFAULT_TTS_PROMPT}Target language: {name}.\n")
        }
    }
}

/// Treat a missing/blank prompt as the canonical generation prompt.
pub fn normalize_tts_prompt(prompt: Option<&str>) -> String {
    match prompt {
        None => DEFAULT_TTS_PROMPT.to_string(),
        Some(value) => {
            if py_strip(value).is_empty() {
                DEFAULT_TTS_PROMPT.to_string()
            } else {
                value.to_string()
            }
        }
    }
}

/// Blank means no instruction, not an empty assistant EOS prefix.
pub fn normalize_instruction(instruction: Option<&str>) -> Option<String> {
    match instruction {
        None => None,
        Some(value) => {
            let stripped = py_strip(value);
            if stripped.is_empty() {
                None
            } else {
                Some(stripped.to_string())
            }
        }
    }
}

/// True when the target text must bypass Chinese/English TN.
pub fn preserve_multilingual_text(language_id: Option<&str>) -> bool {
    let Some(value) = language_id else {
        return false;
    };
    let language = py_strip(value).to_lowercase().replace('_', "-");
    if language.is_empty() {
        return false;
    }
    !matches!(
        language.as_str(),
        "zh" | "zh-cn" | "chinese" | "en" | "english"
    )
}

// ---------------------------------------------------------------------------
// Structured instruction
// ---------------------------------------------------------------------------

/// Build the ordered assistant-side caption consumed during SFT.
#[allow(clippy::too_many_arguments)]
pub fn build_structured_instruction(
    schema: &str,
    dialect: Option<&str>,
    style: Option<&str>,
    speed: Option<&str>,
    pitch: Option<&str>,
    volume: Option<&str>,
    emotion: Option<&str>,
    ip: Option<&str>,
) -> Result<String> {
    let schema = py_strip(schema).to_lowercase();
    let (fields, ip_key): ([&str; 8], &str) = match schema.as_str() {
        "zh" | "chinese" => (ZH_FIELDS, "IP"),
        "en" | "english" => (EN_FIELDS, "ip"),
        _ => bail!(
            "Unsupported structured instruction schema: {}",
            py_repr(&schema)
        ),
    };

    let values: [Option<&str>; 8] = [None, None, dialect, style, speed, pitch, volume, emotion];

    let mut item = String::from("{");
    for (index, field) in fields.iter().enumerate() {
        if index > 0 {
            item.push_str(", ");
        }
        item.push_str(&json_string(field));
        item.push_str(": ");
        match index {
            0 => item.push('1'),
            1 => item.push_str(&json_string("speaker_1")),
            _ => item.push_str(&json_opt(values[index])),
        }
    }
    item.push_str(", \"BGM\": {");
    for (index, field) in BGM_FIELDS.iter().enumerate() {
        if index > 0 {
            item.push_str(", ");
        }
        item.push_str(&json_string(field));
        item.push_str(": null");
    }
    item.push_str("}, ");
    item.push_str(&json_string(ip_key));
    item.push_str(": ");
    item.push_str(&json_opt(ip));
    item.push('}');

    Ok(format!("{{\"audio_sequence\": [{item}]}}"))
}

/// Build a task caption while preventing unrelated controls from leaking.
pub fn build_task_instruction(
    task: &str,
    schema: Option<&str>,
    controls: &Controls,
) -> Result<String> {
    let task = py_strip(task).to_uppercase();
    let allowed: &[&str] = match task.as_str() {
        "BASIC" => &["speed", "pitch", "volume"],
        "DIALECT" => &["dialect"],
        "EMOTION" => &["emotion"],
        "VOICE_DESIGN" | "STYLE" => &["style"],
        "IP" => &["ip"],
        _ => bail!("Unsupported TTS control task: {}", py_repr(&task)),
    };

    let mut unknown: Vec<&str> = Vec::new();
    for (key, present) in controls.fields() {
        if present && !allowed.contains(&key) {
            unknown.push(key);
        }
    }
    if !unknown.is_empty() {
        unknown.sort_unstable();
        bail!(
            "{} does not accept control fields: {}",
            task,
            py_list_repr(&unknown)
        );
    }

    let schema = match schema {
        Some(value) => value,
        None => {
            if task == "DIALECT" {
                "en"
            } else {
                "zh"
            }
        }
    };

    build_structured_instruction(
        schema,
        controls.dialect.as_deref(),
        controls.style.as_deref(),
        controls.speed.as_deref(),
        controls.pitch.as_deref(),
        controls.volume.as_deref(),
        controls.emotion.as_deref(),
        controls.ip.as_deref(),
    )
}

// ---------------------------------------------------------------------------
// Task resolution
// ---------------------------------------------------------------------------

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
) -> Result<SynthRequest> {
    build_task_generation_kwargs_ext(
        task,
        text,
        language_id,
        None,
        instruction,
        controls,
        instruction_schema,
        prompt_wav_path,
        prompt_text,
        spk_emb_path,
        use_spk_emb,
        use_zero_spk_emb,
        cfg,
        sigma,
        temperature,
    )
}

/// build_task_generation_kwargs plus the target_language alternative to
/// language_id (the reference accepts either).
#[allow(clippy::too_many_arguments)]
pub fn build_task_generation_kwargs_ext(
    task: &str,
    text: &str,
    language_id: Option<&str>,
    target_language: Option<&str>,
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
) -> Result<SynthRequest> {
    let task = py_strip(task)
        .to_uppercase()
        .replace('-', "_");

    let runtime_task: &str = match task.as_str() {
        "TTS" | "SEED" | "PLAIN" | "MULTILINGUAL" | "MULTILINGUAL_TTS" => "TTS",
        "BASIC" => "BASIC",
        "EMOTION" => "EMOTION",
        "DIALECT" => "DIALECT",
        "IP" => "IP",
        "VOICE_DESIGN" | "STYLE" => "STYLE",
        "FREEFORM" => "STYLE",
        _ => bail!(
            "Unsupported Ming TTS task {}; expected one of {}",
            py_repr(&task),
            py_list_repr(&SORTED_TASKS)
        ),
    };

    // DIALECT does not use a language ID; the reference warns and drops it.
    let (language_id, target_language) = if task == "DIALECT"
        && (non_blank(language_id) || non_blank(target_language))
    {
        (None, None)
    } else {
        (language_id, target_language)
    };

    if let (Some(left), Some(right)) = (&language_id, &target_language) {
        if py_strip(left).to_lowercase() != py_strip(right).to_lowercase() {
            bail!("language_id and target_language must not conflict");
        }
    }
    let language_id = match language_id {
        Some(value) => Some(value),
        None => target_language,
    };

    if matches!(task.as_str(), "MULTILINGUAL" | "MULTILINGUAL_TTS") && !non_blank(language_id) {
        bail!("MULTILINGUAL requires language_id");
    }

    let text = py_strip(text);
    if text.is_empty() {
        bail!("text must not be empty");
    }
    if use_spk_emb && use_zero_spk_emb {
        bail!("use_spk_emb and use_zero_spk_emb cannot both be true");
    }
    if use_spk_emb
        && !(prompt_wav_path.is_some_and(|path| !path.is_empty())
            || spk_emb_path.is_some_and(|path| !path.is_empty()))
    {
        bail!("use_spk_emb requires a prompt wav or a speaker embedding");
    }

    let controls = controls.cloned().unwrap_or_default();
    let task_instruction: Option<String>;

    if task == "FREEFORM" {
        if !controls.is_empty() {
            bail!("FREEFORM uses instruction, not controls");
        }
        task_instruction = normalize_instruction(instruction);
        if task_instruction.is_none() {
            bail!("FREEFORM requires a non-empty instruction");
        }
    } else if STRUCTURED_TASKS.contains(&task.as_str()) || task == "STYLE" {
        if instruction.is_some() {
            bail!("{task} builds its instruction from controls; do not pass instruction");
        }
        if controls.is_empty() {
            bail!("{task} requires task controls");
        }
        let prompt_task = if task == "STYLE" { "VOICE_DESIGN" } else { task.as_str() };
        task_instruction = Some(build_task_instruction(prompt_task, instruction_schema, &controls)?);
    } else {
        if !controls.is_empty() {
            bail!("{task} does not accept controls");
        }
        task_instruction = normalize_instruction(instruction);
    }

    Ok(SynthRequest {
        prompt: build_tts_prompt(language_id),
        text: text.to_string(),
        text_language: language_id.map(str::to_string),
        instruction: task_instruction,
        taskname: runtime_task.to_string(),
        use_spk_emb,
        use_zero_spk_emb,
        prompt_wav_path: prompt_wav_path.map(str::to_string),
        prompt_text: prompt_text.map(str::to_string),
        spk_emb_path: spk_emb_path.map(str::to_string),
        cfg,
        sigma,
        temperature,
    })
}
