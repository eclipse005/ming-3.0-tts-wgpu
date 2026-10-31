//! Port of prompt_contract.py: the exact token layout the checkpoint was
//! trained with. Kept as a separate module so a layout regression is a single
//! file to bisect.

use crate::tok::Tok;
use anyhow::{bail, Result};

pub const DENSE_SYSTEM_PREFIX: &str = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";
pub const DENSE_USER_PREFIX: &str = "<|im_start|>user\n";
pub const DENSE_USER_SUFFIX: &str = "<|im_end|>\n";
pub const DENSE_ASSISTANT_PREFIX: &str = "<|im_start|>assistant\n";
pub const DENSE_SPEAKER_START: &str = "<|vision_start|>";
pub const DENSE_SPEAKER_PAD: &str = "<|vision_pad|>";
pub const DENSE_SPEAKER_END: &str = "<|vision_end|>\n";
pub const DENSE_INSTRUCTION_EOS: &str = "<|im_end|>";
pub const TEXT_INPUT_PREFIX: &str = " Text input:\n";
pub const AUDIO_START: &str = "<audio>";
pub const AUDIO_PATCH: &str = "<audioPatch>";

const BGM_MARKERS: [&str; 5] = ["Genre: ", "Mood: ", "Instrument: ", "Theme: ", "Duration: "];

pub struct InputLayout {
    pub ids: Vec<u32>,
    pub speaker_positions: Vec<usize>,
    pub latent_position: i64,
}

/// Assemble the input sequence and the rows to overwrite with features.
#[allow(clippy::too_many_arguments)]
pub fn build_input_ids(
    tok: &Tok,
    prompt: &str,
    text: &str,
    num_speakers: usize,
    instruction: Option<&str>,
    prompt_text: Option<&str>,
    prompt_latent_steps: usize,
) -> Result<InputLayout> {
    let enc = |s: &str| tok.encode(s);

    let mut speaker_block: Vec<u32> = Vec::new();
    for index in 0..num_speakers {
        speaker_block.extend(enc(&format!("  speaker_{}:", index + 1))?);
        speaker_block.extend(enc(DENSE_SPEAKER_START)?);
        speaker_block.extend(enc(DENSE_SPEAKER_PAD)?);
        speaker_block.extend(enc(DENSE_SPEAKER_END)?);
    }

    let mut instruction_block: Vec<u32> = Vec::new();
    if let Some(i) = instruction {
        instruction_block = enc(i)?;
        instruction_block.extend(enc(DENSE_INSTRUCTION_EOS)?);
    }

    let mut prompt_text_block: Vec<u32> = Vec::new();
    let mut latent_block: Vec<u32> = Vec::new();
    if prompt_latent_steps > 0 {
        if let Some(pt) = prompt_text {
            prompt_text_block = enc(pt)?;
            let patch = enc(AUDIO_PATCH)?;
            for _ in 0..prompt_latent_steps {
                latent_block.extend_from_slice(&patch);
            }
        }
    }

    let is_bgm = BGM_MARKERS.iter().all(|m| text.contains(m));
    let text_prefix = if is_bgm { Vec::new() } else { enc(TEXT_INPUT_PREFIX)? };

    let mut head = enc(DENSE_SYSTEM_PREFIX)?;
    head.extend(enc(DENSE_USER_PREFIX)?);
    head.extend(enc(prompt)?);

    let mut turn_end = enc(DENSE_USER_SUFFIX)?;
    turn_end.extend(enc(DENSE_ASSISTANT_PREFIX)?);

    let mut ids = head;
    ids.extend(speaker_block);
    ids.extend(text_prefix);
    ids.extend(prompt_text_block);
    ids.extend(enc(text)?);
    ids.extend(turn_end);
    ids.extend(instruction_block);
    ids.extend(enc(AUDIO_START)?);

    let speaker_start_id = tok.token_id(DENSE_SPEAKER_START)?;
    let speaker_pad_id = tok.token_id(DENSE_SPEAKER_PAD)?;
    let speaker_positions: Vec<usize> = ids
        .iter()
        .enumerate()
        .filter(|(_, &t)| t == speaker_start_id)
        .map(|(p, _)| p + 1)
        .collect();
    if speaker_positions.len() != num_speakers {
        bail!(
            "expected {} speaker slots, found {}",
            num_speakers,
            speaker_positions.len()
        );
    }
    for &p in &speaker_positions {
        if ids[p] != speaker_pad_id {
            bail!("speaker slot is not followed by its pad token");
        }
    }

    let latent_position = if latent_block.is_empty() { -1 } else { ids.len() as i64 };
    ids.extend(latent_block);
    Ok(InputLayout { ids, speaker_positions, latent_position })
}
