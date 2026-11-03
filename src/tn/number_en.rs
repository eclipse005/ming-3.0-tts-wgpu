//! Port of `frontend/number_en.py`: English number expansion before the en FSTs.
//!
//! The reference leans on `inflect.engine().number_to_words`; this module
//! reproduces its wording (comma-separated scale groups, "and" before a final
//! group below one hundred, hyphenated tens) and `ordinal`, then runs the same
//! nine regex passes in the same order. Goldens come from inflect 7.6.0
//! (`tools/_en_goldens.py`), so the wording is checked value by value rather
//! than by rule-of-thumb.


use anyhow::{Context, Result};
use fancy_regex::Regex;

const ONES: [&str; 20] = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen",
];
const TENS: [&str; 10] = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
];
const SCALES: [&str; 13] = [
    "",
    "thousand",
    "million",
    "billion",
    "trillion",
    "quadrillion",
    "quintillion",
    "sextillion",
    "septillion",
    "octillion",
    "nonillion",
    "decillion",
    "undecillion",
];

/// One group of up to three digits, e.g. 123 -> "one hundred and twenty-three".
fn three_digits(mut n: u128, andword: bool) -> String {
    if n == 0 {
        return String::new();
    }
    if n < 20 {
        return ONES[n as usize].to_string();
    }
    if n < 100 {
        let t = TENS[(n / 10) as usize];
        let r = n % 10;
        return if r == 0 {
            t.to_string()
        } else {
            format!("{t}-{}", ONES[r as usize])
        };
    }
    let h = ONES[(n / 100) as usize];
    n %= 100;
    let mut out = format!("{h} hundred");
    if n > 0 {
        out.push_str(if andword { " and " } else { " " });
        out.push_str(&three_digits(n, andword));
    }
    out
}

/// inflect's `number_to_words` for a non-negative integer.
pub fn number_to_words(n: u128, andword: bool) -> String {
    if n == 0 {
        return "zero".to_string();
    }
    let mut rest = n;
    let mut groups: Vec<u128> = Vec::new();
    while rest > 0 {
        groups.push(rest % 1000);
        rest /= 1000;
    }
    if groups.len() > SCALES.len() {
        return n.to_string();
    }
    let mut parts: Vec<String> = Vec::new();
    for i in (0..groups.len()).rev() {
        let g = groups[i];
        if g == 0 {
            continue;
        }
        let mut word = three_digits(g, andword);
        if i > 0 {
            word.push(' ');
            word.push_str(SCALES[i]);
        }
        parts.push(word);
    }
    if parts.len() <= 1 {
        return parts.pop().unwrap_or_default();
    }
    // The final group below one hundred attaches with "and" (or a bare space
    // when andword is suppressed); every other group is comma separated.
    let tail = groups[0];
    let mut out = parts[..parts.len() - 1].join(", ");
    if tail > 0 && tail < 100 {
        out.push_str(if andword { " and " } else { " " });
    } else {
        out.push_str(", ");
    }
    out.push_str(parts.last().unwrap());
    out
}

/// inflect's `ordinal`: transform the last word of a written number.
pub fn ordinal(word: &str) -> String {
    let (head, last) = match word.rfind([' ', '-']) {
        Some(i) => (&word[..=i], &word[i + 1..]),
        None => ("", word),
    };
    let tail = if last == "one" {
        "first"
    } else if last == "two" {
        "second"
    } else if last == "three" {
        "third"
    } else if last == "five" {
        "fifth"
    } else if last == "eight" {
        "eighth"
    } else if last == "nine" {
        "ninth"
    } else if last == "twelve" {
        "twelfth"
    } else if last == "hundred" {
        "hundredth"
    } else if last == "thousand" {
        "thousandth"
    } else if last == "million" {
        "millionth"
    } else if last == "billion" {
        "billionth"
    } else if let Some(stem) = last.strip_suffix('y') {
        return format!("{head}{stem}ieth");
    } else {
        return format!("{head}{last}th");
    };
    format!("{head}{tail}")
}

fn parse_u128(s: &str) -> Option<u128> {
    s.parse::<u128>().ok()
}

fn number_words(s: &str, andword: bool) -> String {
    parse_u128(s)
        .map(|n| number_to_words(n, andword))
        .unwrap_or_else(|| s.to_string())
}

fn with_minus(negative: bool, word: String) -> String {
    if negative {
        format!("minus {word}")
    } else {
        word
    }
}

/// Split a cleaned decimal string into (integer part, fractional digits).
fn split_decimal(clean: &str) -> (&str, Option<&str>) {
    match clean.split_once('.') {
        Some((i, f)) => (i, Some(f)),
        None => (clean, None),
    }
}

/// `_inflect.number_to_words(int_part, andword="")` + " point " + digits,
/// shared by every expander in number_en.py.
fn decimal_words(clean: &str) -> Option<String> {
    let (int_part, frac) = split_decimal(clean);
    let int_word = if int_part == "0" || int_part.is_empty() {
        "zero".to_string()
    } else {
        number_words(int_part, false)
    };
    match frac {
        None => Some(int_word),
        Some(f) if f.chars().all(|c| c.is_ascii_digit()) => {
            let mut words = vec![int_word];
            words.push("point".to_string());
            for c in f.chars() {
                words.push(number_to_words(c.to_digit(10).unwrap() as u128, false));
            }
            Some(words.join(" "))
        }
        Some(_) => None,
    }
}

fn unit_mapping(unit: &str) -> Option<&'static str> {
    Some(match unit {
        "ms" => "milliseconds",
        "s" => "seconds",
        "hz" => "hertz",
        "khz" => "kilohertz",
        "mhz" => "megahertz",
        "ghz" => "gigahertz",
        "kb" => "kilobytes",
        "mb" => "megabytes",
        "gb" => "gigabytes",
        "tb" => "terabytes",
        "kbps" => "kilobits per second",
        "mbps" => "megabits per second",
        "gbps" => "gigabits per second",
        "bps" => "bits per second",
        "cm" => "centimeters",
        "km" => "kilometers",
        "kg" => "kilograms",
        "v" => "volts",
        "a" => "amperes",
        "w" => "watts",
        "°c" => "degrees celsius",
        "°f" => "degrees fahrenheit",
        _ => return None,
    })
}

fn fraction_words(num: u128, den: u128) -> String {
    if num == 1 && den == 2 {
        return " one half ".to_string();
    }
    if num == 1 && den == 4 {
        return " one quarter ".to_string();
    }
    if den == 2 {
        let plural = if num == 1 { " half" } else { " halves" };
        return format!(" {}{plural} ", number_to_words(num, true));
    }
    if den == 4 {
        let plural = if num == 1 { " quarter" } else { " quarters" };
        return format!(" {}{plural} ", number_to_words(num, true));
    }
    format!(
        " {} {} ",
        number_to_words(num, true),
        ordinal(&number_to_words(den, true))
    )
}

struct Passes {
    comma: Regex,
    unit: Regex,
    pounds: Regex,
    dollars: Regex,
    fraction: Regex,
    percent: Regex,
    ordinal: Regex,
    version: Regex,
    number: Regex,
}

impl Passes {
    fn new() -> Result<Self> {
        let mk = |p: &str| Regex::new(p).map_err(|e| anyhow::anyhow!("number_en regex {p}: {e}"));
        Ok(Self {
            comma: mk(r"([0-9][0-9,]+[0-9])")?,
            unit: mk(r"(?i)\b(-?\d+(?:\.\d+)?)\s*(ms|s|Hz|kHz|MHz|GHz|kb|mb|gb|tb|KB|MB|GB|TB|bps|kbps|Mbps|Gbps|cm|km|kg|V|A|W|°C|°F)\b")?,
            pounds: mk(r"£(-?[0-9,]*[0-9]+(?:\.[0-9]+)?)")?,
            dollars: mk(r"\$(-?[0-9.,]*[0-9]+(?:\.[0-9]+)?)")?,
            fraction: mk(r"([0-9]+)\/([0-9]+)")?,
            percent: mk(r"(-?[0-9.,]*[0-9]+)%")?,
            ordinal: mk(r"\b[0-9]+(st|nd|rd|th)\b")?,
            version: mk(r"\b([a-zA-Z]+)([-]?)([0-9]+(?:\.[0-9]+)?)\b")?,
            number: mk(r"\b-?[0-9]+(?:\.[0-9]+)?\b")?,
        })
    }
}

fn replace_all<F>(re: &Regex, text: &str, f: F) -> String
where
    F: Fn(&str) -> String,
{
    let mut out = String::with_capacity(text.len());
    let mut last = 0usize;
    for m in re.find_iter(text) {
        let Ok(m) = m else { continue };
        out.push_str(&text[last..m.start()]);
        out.push_str(&f(m.as_str()));
        last = m.end();
    }
    out.push_str(&text[last..]);
    out
}

/// `_expand_unit`: "5.5 kg" -> " five point five kilograms ".
fn expand_unit(num_str: &str, unit: &str) -> String {
    let negative = num_str.starts_with('-');
    let clean = num_str.trim_start_matches('-');
    let clean = if clean.is_empty() { "0" } else { clean };
    let Some(words) = decimal_words(clean) else {
        return format!(" {num_str} {unit} ");
    };
    let unit_word = unit_mapping(&unit.to_lowercase()).unwrap_or({
        // The caller lower-cases; unknown units pass through in lower case.
        ""
    });
    let unit_word = if unit_word.is_empty() {
        unit.to_lowercase()
    } else {
        unit_word.to_string()
    };
    format!(" {} {} ", with_minus(negative, words), unit_word)
}

/// `_expand_percent` / `_expand_dollars` / `_expand_pounds` share this shape.
fn expand_money(num_str: &str, singular: &str, plural: &str) -> String {
    let negative = num_str.starts_with('-');
    let clean = num_str.trim_start_matches('-');
    let clean = if clean.is_empty() { "0" } else { clean };
    let plain = clean.replace(',', "");
    let Some(words) = decimal_words(&plain) else {
        return format!(" {clean} {plural} ");
    };
    let value = plain.parse::<f64>().unwrap_or(f64::NAN);
    let unit = if value.abs() == 1.0 { singular } else { plural };
    format!(" {} {} ", with_minus(negative, words), unit)
}

/// `normalize_numbers` from frontend/number_en.py, pass for pass.
fn normalize_numbers(text: &str, p: &Passes) -> String {
    // 1. thousands separators
    let text = replace_all(&p.comma, text, |m| m.replace(',', ""));
    // 2. quantities with units
    let text = replace_all(&p.unit, &text, |m| {
        let caps = p.unit.captures(m).ok().flatten();
        match caps {
            Some(c) => expand_unit(
                c.get(1).map(|g| g.as_str()).unwrap_or(""),
                c.get(2).map(|g| g.as_str()).unwrap_or(""),
            ),
            None => m.to_string(),
        }
    });
    // 3. pounds, 4. dollars
    let text = replace_all(&p.pounds, &text, |m| {
        let num = m.trim_start_matches('£');
        expand_money(num, "pound", "pounds")
    });
    let text = replace_all(&p.dollars, &text, |m| {
        let num = m.trim_start_matches('$');
        expand_money(num, "dollar", "dollars")
    });
    // 5. fractions
    let text = replace_all(&p.fraction, &text, |m| {
        let caps = p.fraction.captures(m).ok().flatten();
        match caps {
            Some(c) => {
                let num = c.get(1).map(|g| g.as_str()).unwrap_or("0");
                let den = c.get(2).map(|g| g.as_str()).unwrap_or("0");
                match (parse_u128(num), parse_u128(den)) {
                    (Some(n), Some(d)) => fraction_words(n, d),
                    _ => m.to_string(),
                }
            }
            None => m.to_string(),
        }
    });
    // 6. percent
    let text = replace_all(&p.percent, &text, |m| {
        let num = m.trim_end_matches('%');
        let negative = num.starts_with('-');
        let clean = num.trim_start_matches('-');
        let clean = if clean.is_empty() { "0" } else { clean };
        match decimal_words(clean) {
            Some(words) => format!(" {} percent ", with_minus(negative, words)),
            None => format!(" {num} percent "),
        }
    });
    // 7. ordinals turn into their plain number words
    let text = replace_all(&p.ordinal, &text, |m| {
        let digits: String = m.chars().filter(|c| c.is_ascii_digit()).collect();
        format!(" {} ", number_words(&digits, true))
    });
    // 8. versions: v2.1 -> v two point one, gpt-4 -> gpt four
    let text = replace_all(&p.version, &text, |m| {
        let caps = p.version.captures(m).ok().flatten();
        let Some(c) = caps else { return m.to_string() };
        let prefix = c.get(1).map(|g| g.as_str()).unwrap_or("");
        let num_str = c.get(3).map(|g| g.as_str()).unwrap_or("");
        let (int_part, frac) = split_decimal(num_str);
        let word = match frac {
            Some(f) if f.chars().all(|ch| ch.is_ascii_digit()) => {
                let mut words = vec![number_words(int_part, true), "point".to_string()];
                for ch in f.chars() {
                    words.push(number_to_words(ch.to_digit(10).unwrap() as u128, true));
                }
                words.join(" ")
            }
            Some(_) => return m.to_string(),
            None => number_words(int_part, true),
        };
        format!("{prefix} {word}")
    });
    // 9. leftover bare numbers
    let text = replace_all(&p.number, &text, |m| {
        let negative = m.starts_with('-');
        let clean = m.trim_start_matches('-');
        let clean = if clean.is_empty() { "0" } else { clean };
        match decimal_words(clean) {
            Some(words) => format!(" {} ", with_minus(negative, words)),
            None => format!(" {m} "),
        }
    });
    // 10. whitespace
    let collapsed: Vec<&str> = text.split_whitespace().collect();
    collapsed.join(" ").trim().to_string()
}

pub struct NumberEn {
    passes: Passes,
}

impl NumberEn {
    pub fn new() -> Result<Self> {
        Ok(Self {
            passes: Passes::new().context("build number_en regexes")?,
        })
    }

    pub fn normalize(&self, text: &str) -> String {
        normalize_numbers(text, &self.passes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn words_match_inflect_samples() {
        assert_eq!(number_to_words(0, true), "zero");
        assert_eq!(number_to_words(21, true), "twenty-one");
        assert_eq!(number_to_words(101, true), "one hundred and one");
        assert_eq!(number_to_words(101, false), "one hundred one");
        assert_eq!(
            number_to_words(1234, true),
            "one thousand, two hundred and thirty-four"
        );
        assert_eq!(number_to_words(2026, true), "two thousand and twenty-six");
        assert_eq!(number_to_words(2026, false), "two thousand twenty-six");
        assert_eq!(
            number_to_words(101000, true),
            "one hundred and one thousand"
        );
        assert_eq!(number_to_words(100010, true), "one hundred thousand and ten");
        assert_eq!(number_to_words(100100, true), "one hundred thousand, one hundred");
        assert_eq!(number_to_words(1000001, true), "one million and one");
    }

    fn golden(name: &str) -> serde_json::Value {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("goldens").join(name);
        serde_json::from_str(&std::fs::read_to_string(path).expect("read golden")).expect("parse golden")
    }

    /// Every sampled value against inflect 7.6.0 (tools/_en_goldens.py).
    #[test]
    fn words_match_inflect_goldens() {
        let v = golden("number_en_words.json");
        let mut bad = 0;
        for row in v["words"].as_array().unwrap() {
            let n = row["n"].as_u64().unwrap() as u128;
            let want_and = row["and"].as_str().unwrap();
            let want_no = row["noand"].as_str().unwrap();
            let got_and = number_to_words(n, true);
            let got_no = number_to_words(n, false);
            if got_and != want_and {
                bad += 1;
                println!("n={n} and: want {want_and:?} got {got_and:?}");
            }
            if got_no != want_no {
                bad += 1;
                println!("n={n} noand: want {want_no:?} got {got_no:?}");
            }
        }
        for row in v["ordinals"].as_array().unwrap() {
            let w = row["word"].as_str().unwrap();
            let want = row["ordinal"].as_str().unwrap();
            let got = ordinal(w);
            if got != want {
                bad += 1;
                println!("ordinal {w:?}: want {want:?} got {got:?}");
            }
        }
        assert_eq!(bad, 0, "{bad} wording mismatches");
    }

    #[test]
    fn normalize_numbers_matches_reference() {
        let v = golden("number_en_cases.json");
        let eng = NumberEn::new().unwrap();
        let mut bad = 0;
        for row in v.as_array().unwrap() {
            let input = row["input"].as_str().unwrap();
            let want = row["expected"].as_str().unwrap();
            let got = eng.normalize(input);
            if got != want {
                bad += 1;
                println!("IN  {input:?}\n  want {want:?}\n  got  {got:?}");
            }
        }
        assert_eq!(bad, 0, "{bad} normalize_numbers mismatches");
    }

    #[test]
    fn ordinals_match_inflect_samples() {
        assert_eq!(ordinal("one"), "first");
        assert_eq!(ordinal("two"), "second");
        assert_eq!(ordinal("three"), "third");
        assert_eq!(ordinal("five"), "fifth");
        assert_eq!(ordinal("eight"), "eighth");
        assert_eq!(ordinal("twelve"), "twelfth");
        assert_eq!(ordinal("twenty"), "twentieth");
        assert_eq!(ordinal("twenty-one"), "twenty-first");
        assert_eq!(ordinal("one hundred and one"), "one hundred and first");
        assert_eq!(ordinal("one thousand"), "one thousandth");
    }
}
