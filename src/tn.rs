//! Port of the reference text normaliser (`frontend/talker_tn`): the compiled
//! pynini FSTs plus the Python glue (pre-processor, token parser).
//!
//! Reference pipeline:
//!   TalkerPreProcessor -> (contains CJK) TalkerRE -> digit-hyphen -> zh tagger
//!   -> TokenParser reorder -> zh verbalizer; otherwise the en_* FSTs.
//!
//! The FSTs ship as OpenFST binary files; `rustfst` reads and composes them, so
//! the tag/verbalise step runs the same automaton the Python code runs.
//!
//! Still missing: the TalkerRE regex rule layer (re_utils/*) and number_en.py,
//! which the reference applies before the FSTs for Chinese / non-Chinese text
//! respectively. See PROGRESS 0.12.


use anyhow::{Context, Result};
use rustfst::algorithms::compose::compose;
use rustfst::algorithms::tr_sort;
use rustfst::algorithms::tr_compares::ILabelCompare;
use rustfst::prelude::*;

/// Byte-string FST: one state per byte, ilabel == olabel == the byte.
fn string_fst(text: &str) -> VectorFst<TropicalWeight> {
    let mut fst: VectorFst<TropicalWeight> = VectorFst::new();
    let mut prev = fst.add_state();
    fst.set_start(prev).unwrap();
    for b in text.as_bytes() {
        let next = fst.add_state();
        fst.add_tr(
            prev,
            Tr::new(*b as u32, *b as u32, TropicalWeight::one(), next),
        )
        .unwrap();
        prev = next;
    }
    fst.set_final(prev, TropicalWeight::one()).unwrap();
    fst
}

/// Output-label string of the cheapest (unique) path.
fn best_path_output(fst: &VectorFst<TropicalWeight>) -> Vec<u8> {
    let mut out = Vec::new();
    let mut state = match fst.start() {
        Some(s) => s,
        None => return out,
    };
    loop {
        let trs = fst.get_trs(state).unwrap();
        let trs = trs.trs();
        if trs.is_empty() {
            break;
        }
        let (olabel, nextstate) = (trs[0].olabel, trs[0].nextstate);
        if olabel != 0 {
            out.push(olabel as u8);
        }
        state = nextstate;
    }
    out
}

pub struct TnFst {
    tagger: VectorFst<TropicalWeight>,
    verbalizer: VectorFst<TropicalWeight>,
    ordertype: &'static str,
}

impl TnFst {
    fn load(prefix: &str, ordertype: &'static str) -> Result<Self> {
        let mut tagger = VectorFst::<TropicalWeight>::read(format!("{prefix}_tagger.fst"))
            .with_context(|| format!("read {prefix}_tagger.fst"))?;
        let mut verbalizer = VectorFst::<TropicalWeight>::read(format!("{prefix}_verbalizer.fst"))
            .with_context(|| format!("read {prefix}_verbalizer.fst"))?;
        // Composition matches the first operand's olabel against the second's
        // ilabel through arc-sorted matchers; pynini ships these FSTs sorted,
        // but sorting again is cheap and removes any dependency on that.
        tr_sort(&mut tagger, ILabelCompare {});
        tr_sort(&mut verbalizer, ILabelCompare {});
        Ok(Self {
            tagger,
            verbalizer,
            ordertype,
        })
    }

    fn apply(fst: &VectorFst<TropicalWeight>, input: &str) -> Result<String> {
        let s = string_fst(input);
        let lattice: VectorFst<TropicalWeight> = compose::<
            TropicalWeight,
            VectorFst<TropicalWeight>,
            VectorFst<TropicalWeight>,
            VectorFst<TropicalWeight>,
            _,
            _,
        >(&s, fst)?;
        let best: VectorFst<TropicalWeight> = shortest_path_with_config(
            &lattice,
            ShortestPathConfig {
                nshortest: 1,
                unique: true,
                ..Default::default()
            },
        )?;
        let bytes = best_path_output(&best);
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    pub fn tag(&self, input: &str) -> Result<String> {
        Self::apply(&self.tagger, input)
    }

    /// `reorder` is applied by the caller (pure string rewriting).
    pub fn verbalize(&self, reordered: &str) -> Result<String> {
        Self::apply(&self.verbalizer, reordered)
    }

    pub fn ordertype(&self) -> &'static str {
        self.ordertype
    }
}

// ---------------------------------------------------------------------------
// TokenParser (1:1 with frontend/talker_tn/token_parser.py)
// ---------------------------------------------------------------------------

const TN_ORDERS: &[(&str, &[&str])] = &[
    ("date", &["year", "month", "day"]),
    ("fraction", &["denominator", "numerator"]),
    ("measure", &["denominator", "numerator", "value"]),
    ("money", &["value", "currency"]),
    ("time", &["noon", "hour", "minute", "second"]),
];
const EN_TN_ORDERS: &[(&str, &[&str])] = &[
    ("date", &["preserve_order", "text", "day", "month", "year"]),
    (
        "money",
        &[
            "integer_part",
            "fractional_part",
            "quantity",
            "currency_maj",
        ],
    ),
];

fn order_for(ordertype: &str, name: &str) -> Option<&'static [&'static str]> {
    let table = match ordertype {
        "en_tn" => EN_TN_ORDERS,
        _ => TN_ORDERS,
    };
    table
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, order)| *order)
}

#[derive(Default)]
struct TagToken {
    name: String,
    order: Vec<String>,
    members: Vec<(String, String)>,
}

impl TagToken {
    fn string(&self, ordertype: &str) -> String {
        let mut order = self.order.clone();
        if let Some(want) = order_for(ordertype, &self.name) {
            let preserve = self
                .members
                .iter()
                .any(|(k, v)| k == "preserve_order" && v == "true");
            if !preserve {
                order = want.iter().map(|s| s.to_string()).collect();
            }
        }
        let mut out = format!("{} {{", self.name);
        for key in &order {
            if let Some((_, value)) = self.members.iter().find(|(k, _)| k == key) {
                out.push_str(&format!(" {}: \"{}\"", key, value));
            }
        }
        out.push_str(" }");
        out
    }
}

/// Parse the tagger output. Grammar (token_parser.py): `name { key: "value" }`.
fn parse_tags(input: &str) -> Vec<TagToken> {
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0usize;
    let mut tokens = Vec::new();
    loop {
        while i < chars.len() && chars[i] == ' ' {
            i += 1;
        }
        if i >= chars.len() {
            break;
        }
        let mut name = String::new();
        while i < chars.len() && (chars[i].is_ascii_alphabetic() || chars[i] == '_') {
            name.push(chars[i]);
            i += 1;
        }
        if name.is_empty() {
            break;
        }
        while i < chars.len() && matches!(chars[i], ' ' | '{') {
            i += 1;
        }
        let mut token = TagToken {
            name,
            ..Default::default()
        };
        loop {
            while i < chars.len() && chars[i] == ' ' {
                i += 1;
            }
            if i >= chars.len() {
                break;
            }
            if chars[i] == '}' {
                i += 1;
                break;
            }
            let mut key = String::new();
            while i < chars.len() && (chars[i].is_ascii_alphabetic() || chars[i] == '_') {
                key.push(chars[i]);
                i += 1;
            }
            // token_parser.parse_chars(': "'): consume at most one of each,
            // in that order. A greedy skip eats the value when it starts with
            // a space (char { value: " " }).
            for sep in [':', ' ', '"'] {
                if i < chars.len() && chars[i] == sep {
                    i += 1;
                }
            }
            let mut value = String::new();
            while i < chars.len() && chars[i] != '"' {
                let ch = chars[i];
                value.push(ch);
                i += 1;
                // A backslash escapes the next character (token_parser.py).
                if ch == '\\' && i < chars.len() {
                    value.push(chars[i]);
                    i += 1;
                }
            }
            if i < chars.len() {
                i += 1;
            }
            token.order.push(key.clone());
            token.members.push((key, value));
        }
        tokens.push(token);
    }
    tokens
}

pub fn reorder(tagged: &str, ordertype: &str) -> String {
    parse_tags(tagged)
        .iter()
        .map(|t| t.string(ordertype))
        .collect::<Vec<_>>()
        .join(" ")
}

/// `frontend/talker_tn/talker_pre_processor.py`, rule for rule.
pub fn preprocess(text: &str) -> String {
    use fancy_regex::Regex;
    let rules: [(&str, &str); 9] = [
        (r"\$\$.*?\$\$|\$.*?\$", ""),
        (r"https?://[a-zA-Z0-9./?=&_:-]+|www\.[a-zA-Z0-9./?=&_:-]+", ""),
        (r"`.*?`", ""),
        (r"```.*?```", ""),
        (r"!\[.*?\]\(.*?\)", ""),
        (r"(?<=\d),(?=\d{3})", ""),
        (r"(?<=[^a-zA-Z0-9.])[ ]+(?=[^a-zA-Z0-9.])", ""),
        (r"(?<=[a-zA-Z])[ ]+(?=[a-zA-Z])", " "),
        (r"\*\*\*|\*\*|###|---", ""),
    ];
    let mut out = text.to_string();
    for (pat, rep) in rules {
        let re = Regex::new(pat).expect("preprocess regex");
        out = re.replace_all(&out, rep).into_owned();
    }
    out.chars()
        .filter(|c| {
            let u = *c as u32;
            let emoji = (0x1F600..=0x1F64F).contains(&u);
            let symbols = "▼●★☆■□▲△◆◇○◎√×✔✖";
            let controls = matches!(c, '\t' | '\n' | '\r' | '\x0c');
            !(emoji || symbols.contains(*c) || controls)
        })
        .collect()
}

/// `is_chinese`: contains a CJK unified ideograph (text_segment_cut.py).
pub fn has_cjk(text: &str) -> bool {
    text.chars()
        .any(|c| ('\u{4e00}'..='\u{9fff}').contains(&c))
}

/// `(?<=\d)-(?=\d)` -> `，` (talker_tn.py).
fn digit_hyphen(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    for (i, c) in chars.iter().enumerate() {
        let between_digits = *c == '-'
            && i > 0
            && i + 1 < chars.len()
            && chars[i - 1].is_ascii_digit()
            && chars[i + 1].is_ascii_digit();
        out.push(if between_digits { '，' } else { *c });
    }
    out
}

/// Where the shipped FSTs live: `tn_data/` next to the crate (copied from the
/// reference's `frontend/talker_tn`), overridable with `MING_TN_DIR`.
pub fn default_dir() -> std::path::PathBuf {
    std::env::var("MING_TN_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tn_data")
        })
}

/// The whole TalkerTN chain (minus the TalkerRE rule layer, see module docs).
pub mod number_en;
pub mod re;

pub struct TalkerTn {
    zh: TnFst,
    en: TnFst,
    re: re::ReConfig,
    number_en: number_en::NumberEn,
}

impl TalkerTn {
    pub fn load(dir: &std::path::Path) -> Result<Self> {
        Ok(Self {
            zh: TnFst::load(&dir.join("zh_tn").to_string_lossy(), "tn")?,
            en: TnFst::load(&dir.join("en_tn").to_string_lossy(), "en_tn")?,
            re: re::ReConfig::load(&dir.join("talker_re.json"))?,
            number_en: number_en::NumberEn::new()?,
        })
    }

    /// `ming_tts._normalize_fragment`: English text runs through
    /// `normalize_numbers` *before* TalkerTN (which preprocesses and tags it);
    /// Chinese text goes straight to the regex layer + zh FSTs.
    pub fn normalize_fragment(&self, text: &str) -> Result<String> {
        let text = if has_cjk(text) {
            text.to_string()
        } else {
            self.number_en.normalize(text)
        };
        self.normalize(&text)
    }

    pub fn normalize(&self, text: &str) -> Result<String> {
        let text = preprocess(text);
        if text.is_empty() {
            return Ok(text);
        }
        let has_cjk = has_cjk(&text);
        if has_cjk {
            let text = re::talker_re(&self.re, &text);
            let text = digit_hyphen(&text);
            let tagged = self.zh.tag(&text)?;
            let reordered = reorder(&tagged, self.zh.ordertype());
            self.zh.verbalize(&reordered)
        } else {
            let tagged = self.en.tag(&text)?;
            let reordered = reorder(&tagged, self.en.ordertype());
            self.en.verbalize(&reordered)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    

    /// Reference text goldens generated by tools/_tn_goldens.py with the real
    /// pynini chain (Chinese only).
    #[test]
    fn matches_reference_goldens() {
        crate::need_goldens!();
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("goldens/tn_cases.json");
        let raw = std::fs::read_to_string(&path).expect("read tn_cases.json");
        let cases: Vec<serde_json::Value> = serde_json::from_str(&raw).expect("parse tn_cases.json");
        let tn = TalkerTn::load(&default_dir()).unwrap();
        let mut bad = 0usize;
        for case in &cases {
            let input = case["input"].as_str().unwrap();
            let want = case["expected"].as_str().unwrap();
            let mut got = tn
                .normalize_fragment(input)
                .unwrap_or_else(|e| format!("<err {e}>"));
            got = crate::text::normalize_text_kit(&got);
            if got.starts_with('，') {
                got = got.chars().skip(1).collect();
            }
            if got != want {
                bad += 1;
                println!("MISMATCH {input:?}\n  want {want:?}\n  got  {got:?}");
            }
        }
        println!("tn goldens: {} cases, {bad} mismatches", cases.len());
        assert_eq!(bad, 0, "{bad} of {} TN goldens differ", cases.len());
    }

    #[test]
    fn reads_and_normalizes() {
        let tn = TalkerTn::load(&crate::tn::default_dir()).unwrap();
        for input in [
            "今天天气3度",
            "价格350元",
            "下午2点30分",
            "2026年11月1日",
            "Hello 123 world",
            "增长12.5%",
            "GPT-4",
            "电话13800138000",
            "1/2",
            "3.5 万元",
        ] {
            let tagged = tn.zh.tag(input).unwrap_or_default();
            let out = tn.normalize(input).unwrap();
            println!("{input:?} -> {out:?}");
            if !tagged.is_empty() {
                println!("   tagged: {tagged:?}");
            }
        }
    }
}
