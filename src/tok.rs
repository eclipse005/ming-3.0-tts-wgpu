//! Qwen2 BPE tokenizer wrapper (loads the checkpoint tokenizer.json).
//!
//! This module also owns the *token-level input contract* of the released
//! checkpoint, ported from the reference repo's prompt_contract.py. The
//! checkpoint learned one exact token layout; a stock Qwen2 tokenizer with the
//! extra tokens added in a different order gives every name the right spelling
//! and the wrong embedding row, which degrades audio without raising anything.
//! So the pinned ids are asserted at load time and the layout is built here
//! rather than in every caller.

// Parts of this module are exercised only by the parity tests (they read
// goldens) or kept as public API; the release binary does not call them.

use anyhow::{bail, Context, Result};
use std::path::Path;

// --- Dense (Qwen2-based 0.5b) vocabulary --------------------------------
pub const DENSE_SYSTEM_PREFIX: &str =
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";
pub const DENSE_USER_PREFIX: &str = "<|im_start|>user\n";
pub const DENSE_USER_SUFFIX: &str = "<|im_end|>\n";
pub const DENSE_ASSISTANT_PREFIX: &str = "<|im_start|>assistant\n";
pub const DENSE_SPEAKER_START: &str = "<|vision_start|>";
pub const DENSE_SPEAKER_PAD: &str = "<|vision_pad|>";
pub const DENSE_SPEAKER_END: &str = "<|vision_end|>\n";
/// The checkpoint was trained with the assistant-side caption terminated by
/// `<|im_end|>`, not `<|endoftext|>`.
pub const DENSE_INSTRUCTION_EOS: &str = "<|im_end|>";

// --- Shared -------------------------------------------------------------
pub const TEXT_INPUT_PREFIX: &str = " Text input:\n";
pub const AUDIO_START: &str = "<audio>";
pub const AUDIO_PATCH: &str = "<audioPatch>";

/// A BGM caption carries its own field list and must not get the text prefix.
pub const BGM_MARKERS: [&str; 5] = [
    "Genre: ",
    "Mood: ",
    "Instrument: ",
    "Theme: ",
    "Duration: ",
];

/// Token ids the released 0.5b checkpoint was trained with (prompt_contract.py
/// `DENSE_TOKEN_IDS`). These index embedding rows, so a name-only match is not
/// enough.
pub const DENSE_TOKEN_IDS: [(&str, u32); 8] = [
    ("<|endoftext|>", 151643),
    ("<|im_start|>", 151644),
    ("<|im_end|>", 151645),
    ("<|vision_start|>", 151652),
    ("<|vision_end|>", 151653),
    ("<|vision_pad|>", 151654),
    ("<audioPatch>", 151705),
    ("<audio>", 151706),
];

/// Per-checkpoint token layout. Only the dense layout exists: falling back to
/// the legacy MoE constants for an unrecognised checkpoint would produce a
/// sequence that tokenizes without error and conditions the model on something
/// it never saw, so an unsupported `model_type` is refused instead.
#[derive(Debug, Clone, Copy)]
pub struct Dialect {
    pub speaker_start: &'static str,
    pub speaker_pad: &'static str,
    pub speaker_end: &'static str,
    pub instruction_eos: &'static str,
}

pub const DENSE: Dialect = Dialect {
    speaker_start: DENSE_SPEAKER_START,
    speaker_pad: DENSE_SPEAKER_PAD,
    speaker_end: DENSE_SPEAKER_END,
    instruction_eos: DENSE_INSTRUCTION_EOS,
};

/// Tokens that must exist as single ids for the layout to be expressible.
pub fn required_tokens(d: Dialect) -> [&'static str; 5] {
    [
        AUDIO_START,
        AUDIO_PATCH,
        d.speaker_start,
        d.speaker_pad,
        d.instruction_eos,
    ]
}

pub struct Tok {
    tokenizer: tokenizers::Tokenizer,
}

impl Tok {
    /// Load `<model_dir>/tokenizer.json` and validate it against the trained
    /// layout, failing loudly when the ids do not match.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("tokenizer.json");
        let tok = Self::from_tokenizer_json(&path)
            .with_context(|| format!("load tokenizer {}", path.display()))?;
        tok.validate_contract(DENSE)?;
        Ok(tok)
    }

    pub fn from_tokenizer_json(path: &Path) -> Result<Self> {
        let tokenizer = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("read {}: {e}", path.display()))?;
        Ok(Self { tokenizer })
    }

    /// `add_special_tokens=False` encoding, exactly like
    /// `AutoTokenizer.from_pretrained(...).encode(text, add_special_tokens=False)`.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("tokenizer encode failed: {e}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// `convert_tokens_to_ids` for a single token string.
    pub fn token_id(&self, token: &str) -> Result<u32> {
        self.tokenizer
            .token_to_id(token)
            .with_context(|| format!("token {token:?} is not in the vocabulary"))
    }

    /// Reject a tokenizer that cannot express the trained layout: a tokenizer
    /// that silently splits `<audio>` into sub-words still produces audio, just
    /// conditioned on a sequence the model never saw.
    pub fn validate_contract(&self, d: Dialect) -> Result<()> {
        let mut broken: Vec<String> = Vec::new();
        for token in required_tokens(d) {
            let id = self.token_id(token)?;
            let encoded = self.encode(token)?;
            if encoded != [id] {
                broken.push(format!("{token:?} -> {encoded:?}"));
            }
        }
        if !broken.is_empty() {
            bail!(
                "tokenizer does not match the Ming-3.0-tts dense contract; these \
                 tokens are missing or split: {}",
                broken.join(", ")
            );
        }
        let mut mismatched: Vec<String> = Vec::new();
        for (token, want) in DENSE_TOKEN_IDS {
            let got = self.token_id(token)?;
            if got != want {
                mismatched.push(format!("{token:?}: expected {want}, got {got}"));
            }
        }
        if !mismatched.is_empty() {
            bail!(
                "tokenizer ids do not match Ming-3.0-tts training: {}",
                mismatched.join(", ")
            );
        }
        Ok(())
    }
}

/// Arguments of `prompt_contract.build_input_ids`.
#[derive(Debug, Clone)]
pub struct InputSeqArgs<'a> {
    pub prompt: &'a str,
    pub text: &'a str,
    pub num_speakers: usize,
    pub instruction: Option<&'a str>,
    pub prompt_text: Option<&'a str>,
    pub prompt_latent_steps: usize,
}

/// The assembled sequence plus the rows the caller must overwrite with
/// features.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputIds {
    pub input_ids: Vec<u32>,
    /// Index of each `speaker_pad` row.
    pub speaker_positions: Vec<usize>,
    /// Index of the first prompt-latent row, `-1` without an acoustic prompt.
    pub latent_position: i64,
}

/// Port of `prompt_contract.build_input_ids` for the dense checkpoint.
pub fn build_input_ids(tok: &Tok, d: Dialect, args: &InputSeqArgs) -> Result<InputIds> {
    let mut speaker_block: Vec<u32> = Vec::new();
    for index in 0..args.num_speakers {
        speaker_block.extend(tok.encode(&format!("  speaker_{}:", index + 1))?);
        speaker_block.extend(tok.encode(d.speaker_start)?);
        speaker_block.extend(tok.encode(d.speaker_pad)?);
        speaker_block.extend(tok.encode(d.speaker_end)?);
    }

    let mut instruction_block: Vec<u32> = Vec::new();
    if let Some(instruction) = args.instruction {
        instruction_block.extend(tok.encode(instruction)?);
        instruction_block.extend(tok.encode(d.instruction_eos)?);
    }

    let mut prompt_text_block: Vec<u32> = Vec::new();
    let mut latent_block: Vec<u32> = Vec::new();
    if args.prompt_latent_steps > 0 {
        if let Some(prompt_text) = args.prompt_text {
            prompt_text_block = tok.encode(prompt_text)?;
            let patch = tok.encode(AUDIO_PATCH)?;
            latent_block = patch.repeat(args.prompt_latent_steps);
        }
    }

    let text_prefix = if BGM_MARKERS.iter().all(|m| args.text.contains(m)) {
        Vec::new()
    } else {
        tok.encode(TEXT_INPUT_PREFIX)?
    };

    let mut head = tok.encode(DENSE_SYSTEM_PREFIX)?;
    head.extend(tok.encode(DENSE_USER_PREFIX)?);
    head.extend(tok.encode(args.prompt)?);

    let mut turn_end = tok.encode(DENSE_USER_SUFFIX)?;
    turn_end.extend(tok.encode(DENSE_ASSISTANT_PREFIX)?);

    let mut input_ids: Vec<u32> = Vec::new();
    input_ids.extend(head);
    input_ids.extend(speaker_block);
    input_ids.extend(text_prefix);
    input_ids.extend(prompt_text_block);
    input_ids.extend(tok.encode(args.text)?);
    input_ids.extend(turn_end);
    input_ids.extend(instruction_block);
    input_ids.extend(tok.encode(AUDIO_START)?);

    let speaker_pad_id = tok.token_id(d.speaker_pad)?;
    let speaker_start_id = tok.token_id(d.speaker_start)?;
    let speaker_positions: Vec<usize> = input_ids
        .iter()
        .enumerate()
        .filter(|(_, &id)| id == speaker_start_id)
        .map(|(position, _)| position + 1)
        .collect();
    if speaker_positions.len() != args.num_speakers {
        bail!(
            "expected {} speaker slots, found {}",
            args.num_speakers,
            speaker_positions.len()
        );
    }
    for &position in &speaker_positions {
        if input_ids.get(position) != Some(&speaker_pad_id) {
            bail!("speaker slot is not followed by its pad token");
        }
    }

    let latent_position = if latent_block.is_empty() {
        -1
    } else {
        input_ids.len() as i64
    };
    input_ids.extend(latent_block);
    Ok(InputIds {
        input_ids,
        speaker_positions,
        latent_position,
    })
}


// ---------------------------------------------------------------- tests
