//! Port of `frontend/talker_tn/talker_re.py` + `re_utils/*`: the regex rule
//! layer that rewrites Chinese text before the FSTs.
//!
//! The Python code builds a token stream (chars, digit runs, punctuation runs),
//! scores digit runs against a list of shape rules (date, decimal, time,
//! version, IP, fraction, phone, formula), deals the winning rule, then binds
//! digits to money/measure/percent/serial/order units around them, and finally
//! folds range punctuation into 至/负/大约. This module reproduces that pipeline
//! token for token; the rules are data (talker_re.json), the scoring and the
//! folding are the code below.

#![allow(dead_code)]

use std::collections::HashMap;

use anyhow::{Context, Result};
use fancy_regex::Regex;

const D2C: [char; 10] = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九'];
const D2C_LIANG: [char; 10] = ['零', '一', '两', '三', '四', '五', '六', '七', '八', '九'];
const D2C_YAO: [char; 10] = ['零', '幺', '二', '三', '四', '五', '六', '七', '八', '九'];
const NUM_UNIT: [&str; 13] = [
    "", "万", "亿", "兆", "京", "垓", "秭", "穰", "沟", "涧", "正", "载", "极",
];
const DIGIT_PUNC: &[char] = &[
    '-', '~', '～', ':', '：', '.', '+', 'x', '*', '/', '÷', '=', '≥', '≤', '>', '<',
];

fn digit(ch: char) -> Option<usize> {
    ch.to_digit(10).map(|d| d as usize).filter(|_| ch.is_ascii_digit())
}

fn d4_to_chi(num: &str) -> String {
    let mut s = num.trim_start_matches('0').to_string();
    if s.len() == 2 && s.starts_with('1') {
        let d = &s[1..];
        if d == "0" {
            return "十".to_string();
        }
        return format!("十{}", D2C[digit(d.chars().next().unwrap()).unwrap()]);
    }
    let mut res = String::new();
    while !s.is_empty() {
        if s.starts_with('0') {
            let rest = s.trim_start_matches('0');
            if rest.is_empty() {
                break;
            }
            s = rest.to_string();
            res.push('零');
            continue;
        }
        let d0 = digit(s.chars().next().unwrap()).unwrap();
        match s.len() {
            4 => res.push_str(&format!("{}千", D2C_LIANG[d0])),
            3 => res.push_str(&format!("{}百", D2C_LIANG[d0])),
            2 => res.push_str(&format!("{}十", D2C[d0])),
            _ => res.push(D2C[d0]),
        }
        s = s[1..].to_string();
    }
    res
}

fn to_string_number(num_str: &str) -> String {
    if num_str == "0" {
        return "零".to_string();
    }
    let mut num = num_str.to_string();
    let mut res = String::new();
    for unit in NUM_UNIT {
        if num.is_empty() {
            break;
        }
        let split = num.len().saturating_sub(4);
        let cur = &num[split..];
        let cur_d4 = d4_to_chi(cur);
        if !cur_d4.is_empty() {
            res = format!("{cur_d4}{unit}{res}");
        }
        num = num[..split].to_string();
    }
    if !num.is_empty() {
        let yao: String = num.chars().map(|c| D2C_YAO[digit(c).unwrap()]).collect();
        res = format!("{yao}{res}");
    }
    if res.is_empty() {
        "零".to_string()
    } else {
        res
    }
}

fn to_string_rm0(num_str: &str) -> String {
    num_str
        .trim_start_matches('0')
        .chars()
        .map(|c| D2C[digit(c).unwrap()])
        .collect()
}

fn to_string_direct(num_str: &str) -> String {
    num_str
        .chars()
        .map(|c| D2C[digit(c).unwrap()])
        .collect()
}

fn to_string_yao(num_str: &str) -> String {
    num_str
        .chars()
        .map(|c| D2C_YAO[digit(c).unwrap()])
        .collect()
}

// ---------------------------------------------------------------------------
// Tokens
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub enum Tok {
    Char(String),
    Punc(String),
    Digit(String),
    Date {
        yyyy: Option<Box<Tok>>,
        mm: Option<Box<Tok>>,
        dd: Option<Box<Tok>>,
        day: Option<char>,
    },
    Decimal {
        int: Box<Tok>,
        frac: Box<Tok>,
        range: Option<Box<Tok>>,
    },
    Time {
        hour: Box<Tok>,
        minute: Box<Tok>,
        second: Option<Box<Tok>>,
        noon: Option<String>,
    },
    Version(Vec<Tok>),
    Ip(Vec<Tok>),
    Frac {
        num: Box<Tok>,
        den: Box<Tok>,
        range: Option<Box<Tok>>,
    },
    PhoneDash(Vec<Tok>),
    Formula(Vec<Tok>),
    MoneyBefore {
        d: Box<Tok>,
        tag: String,
        range: Option<Box<Tok>>,
    },
    MoneyAfter {
        d: Box<Tok>,
        tag: String,
        range: Option<Box<Tok>>,
    },
    MeasureBefore {
        d: Box<Tok>,
        tag: String,
        range: Option<Box<Tok>>,
    },
    MeasureAfter {
        d: Box<Tok>,
        tag: String,
        range: Option<Box<Tok>>,
    },
    Percent {
        d: Box<Tok>,
        range: Option<Box<Tok>>,
    },
    SerialBefore {
        d: Box<Tok>,
        tag: String,
    },
    SerialAfter {
        d: Box<Tok>,
        tag: String,
    },
    Order(Box<Tok>),
    Sn {
        txt: String,
        det: bool,
    },
}

impl Tok {
    fn char(s: impl Into<String>) -> Self {
        Tok::Char(s.into())
    }

    fn name(&self) -> &'static str {
        match self {
            Tok::Char(_) => "char",
            Tok::Punc(_) => "punc",
            Tok::Digit(_) => "digit",
            Tok::Date { .. } => "date",
            Tok::Decimal { .. } => "decimal",
            Tok::Time { .. } => "time",
            Tok::Version(_) => "version",
            Tok::Ip(_) => "ip",
            Tok::Frac { .. } => "frac",
            Tok::PhoneDash(_) => "phone_dash",
            Tok::Formula(_) => "formula",
            Tok::MoneyBefore { .. } | Tok::MoneyAfter { .. } => "money",
            Tok::MeasureBefore { .. } => "measure_before",
            Tok::MeasureAfter { .. } => "measure_after",
            Tok::Percent { .. } => "percent",
            Tok::SerialBefore { .. } => "serial_before",
            Tok::SerialAfter { .. } => "serial_after",
            Tok::Order(_) => "order",
            Tok::Sn { .. } => "serial_number",
        }
    }

    fn txt(&self) -> &str {
        match self {
            Tok::Char(s) | Tok::Punc(s) | Tok::Digit(s) => s,
            Tok::Sn { txt, .. } => txt,
            _ => "",
        }
    }

    fn digit_txt(&self) -> Option<&str> {
        match self {
            Tok::Digit(s) => Some(s),
            _ => None,
        }
    }

    fn digit_to_string(&self) -> String {
        self.to_string()
    }

    fn digit_to_string_rm0(&self) -> String {
        to_string_rm0(self.digit_txt().unwrap_or(""))
    }

    fn to_string_yao(&self) -> String {
        match self {
            Tok::Digit(s) => to_string_yao(s),
            Tok::Ip(parts) => parts
                .iter()
                .map(|p| to_string_yao(p.txt()))
                .collect::<Vec<_>>()
                .join("点"),
            Tok::PhoneDash(parts) => parts.iter().map(|p| to_string_yao(p.txt())).collect(),
            _ => self.to_string(),
        }
    }

    pub fn to_string(&self) -> String {
        match self {
            Tok::Char(s) | Tok::Punc(s) => s.clone(),
            Tok::Digit(s) => to_string_number(s),
            Tok::Date { yyyy, mm, dd, day } => {
                let mut res = String::new();
                if let Some(y) = yyyy {
                    res.push_str(&y.digit_to_string_rm0());
                    res.push('年');
                }
                if let Some(m) = mm {
                    res.push_str(&m.digit_to_string());
                    res.push('月');
                }
                if let Some(d) = dd {
                    res.push_str(&d.digit_to_string());
                    match day {
                        Some(c) => res.push(*c),
                        None => res.push('号'),
                    }
                }
                res
            }
            Tok::Decimal { int, frac, range } => {
                let core = format!("{}点{}", int.digit_to_string(), to_string_direct(frac.txt()));
                match range {
                    Some(r) => format!("{}至{}", r.digit_to_string(), core),
                    None => core,
                }
            }
            Tok::Time { hour, minute, second, noon } => {
                let mut res = noon.clone().unwrap_or_default();
                if hour.txt() == "02" || hour.txt() == "2" {
                    res.push_str("两点");
                } else {
                    res.push_str(&hour.digit_to_string());
                    res.push('点');
                }
                let mt = minute.txt();
                let zero_minute = mt == "0" || mt == "00";
                if zero_minute && second.as_ref().map(|s| s.txt() == "0" || s.txt() == "00").unwrap_or(true) {
                    return format!("{res}整");
                }
                if zero_minute {
                    res.push_str("零分");
                } else if mt.len() == 1 || mt.starts_with('0') {
                    res.push_str(&format!("零{}分", minute.digit_to_string()));
                } else {
                    res.push_str(&format!("{}分", minute.digit_to_string()));
                }
                if let Some(sec) = second {
                    let st = sec.txt();
                    if st == "0" || st == "00" {
                        res.push_str("零秒");
                    } else if st.len() == 1 || st.starts_with('0') {
                        res.push_str(&format!("零{}秒", sec.digit_to_string()));
                    } else {
                        res.push_str(&format!("{}秒", sec.digit_to_string()));
                    }
                }
                res
            }
            Tok::Version(parts) => parts
                .iter()
                .map(|p| p.digit_to_string())
                .collect::<Vec<_>>()
                .join("点"),
            Tok::Ip(parts) => parts
                .iter()
                .map(|p| to_string_yao(p.txt()))
                .collect::<Vec<_>>()
                .join("点"),
            Tok::Frac { num, den, range } => match range {
                Some(r) => format!(
                    "{}至{}分之{}",
                    r.digit_to_string(),
                    den.digit_to_string(),
                    num.digit_to_string()
                ),
                None => format!("{}分之{}", den.digit_to_string(), num.digit_to_string()),
            },
            Tok::PhoneDash(parts) => parts.iter().map(|p| to_string_yao(p.txt())).collect(),
            Tok::Formula(parts) => {
                let mut res = String::new();
                for (i, p) in parts.iter().enumerate() {
                    if i % 2 == 0 {
                        res.push_str(&p.digit_to_string());
                    } else {
                        res.push_str(operator(p.txt()));
                    }
                }
                res
            }
            // REMoney.to_string always appends the tag; only try_digit differs
            // between the before/after subclasses.
            Tok::MoneyBefore { d, tag, range } => match range {
                Some(r) => format!("{}至{}{}", r.digit_to_string(), d.digit_to_string(), tag),
                None => format!("{}{}", d.digit_to_string(), tag),
            },
            Tok::MoneyAfter { d, tag, range } => match range {
                Some(r) => format!("{}至{}{}", r.digit_to_string(), d.digit_to_string(), tag),
                None => format!("{}{}", d.digit_to_string(), tag),
            },
            Tok::MeasureBefore { d, tag, range } => match range {
                Some(r) => format!("{}{}至{}", tag, r.digit_to_string(), d.digit_to_string()),
                None => format!("{}{}", tag, d.digit_to_string()),
            },
            Tok::MeasureAfter { d, tag, range } => match range {
                Some(r) => format!("{}至{}{}", r.digit_to_string(), d.digit_to_string(), tag),
                None => format!("{}{}", d.digit_to_string(), tag),
            },
            Tok::Percent { d, range } => match range {
                Some(r) => format!("百分之{}至{}", r.digit_to_string(), d.digit_to_string()),
                None => format!("百分之{}", d.digit_to_string()),
            },
            Tok::SerialBefore { d, tag } => format!("{}{}", tag, to_string_yao(d.txt())),
            Tok::SerialAfter { d, tag } => format!("{}{}", to_string_yao(d.txt()), tag),
            Tok::Order(d) => d.digit_to_string(),
            Tok::Sn { txt, det } => {
                if !*det {
                    return txt.clone();
                }
                txt.chars()
                    .map(|c| match c {
                        '0'..='9' => D2C_YAO[digit(c).unwrap()].to_string(),
                        '-' => "杠".to_string(),
                        _ => c.to_string(),
                    })
                    .collect()
            }
        }
    }
}

fn operator(txt: &str) -> &'static str {
    match txt {
        "+" => "加",
        "-" => "减",
        "x" | "*" => "乘",
        "/" | "÷" => "除",
        "=" => "等于",
        ">=" | "≥" => "大于等于",
        "<=" | "≤" => "小于等于",
        ">" => "大于",
        "<" => "小于",
        _ => "",
    }
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

pub struct ReConfig {
    pub money: HashMap<String, String>,
    pub measure_before: HashMap<String, String>,
    pub measure_after: HashMap<String, String>,
    pub serial_before: HashMap<String, String>,
    pub serial_after: HashMap<String, String>,
    char_map0: HashMap<char, String>,
    proper_noun: HashMap<String, String>,
    proper_re: Regex,
}

fn dict(v: &serde_json::Value, key: &str) -> HashMap<String, String> {
    v.get(key)
        .and_then(|x| x.as_object())
        .map(|o| {
            o.iter()
                .map(|(k, val)| (k.clone(), val.as_str().unwrap_or("").to_string()))
                .collect()
        })
        .unwrap_or_default()
}

impl ReConfig {
    pub fn load(path: &std::path::Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let v: serde_json::Value = serde_json::from_str(&raw).context("parse talker_re.json")?;
        let char_map0 = v
            .get("char_map0")
            .and_then(|x| x.as_object())
            .map(|o| {
                o.iter()
                    .filter_map(|(k, val)| {
                        k.chars().next().map(|c| (c, val.as_str().unwrap_or("").to_string()))
                    })
                    .collect()
            })
            .unwrap_or_default();
        let proper_noun = dict(&v, "proper_noun");
        let mut keys: Vec<&String> = proper_noun.keys().collect();
        keys.sort_by_key(|k| std::cmp::Reverse(k.chars().count()));
        let alternation = keys
            .iter()
            .map(|k| fancy_regex::escape(k))
            .collect::<Vec<_>>()
            .join("|");
        let proper_re = Regex::new(&format!(r"(?<!\d)({alternation})(?!\d)"))
            .context("build proper-noun regex")?;
        Ok(Self {
            money: dict(&v, "money"),
            measure_before: dict(&v, "measure_before"),
            measure_after: dict(&v, "measure_after"),
            serial_before: dict(&v, "serial_before"),
            serial_after: dict(&v, "serial_after"),
            char_map0,
            proper_noun,
            proper_re,
        })
    }

    fn lens(map: &HashMap<String, String>) -> Vec<usize> {
        let mut v: Vec<usize> = map.keys().map(|k| k.chars().count()).collect();
        v.sort_unstable_by(|a, b| b.cmp(a));
        v.dedup();
        v
    }
}

// ---------------------------------------------------------------------------
// Token stream
// ---------------------------------------------------------------------------

fn tokenize(text: &str) -> Vec<Tok> {
    let mut out: Vec<Tok> = Vec::new();
    for c in text.chars() {
        let tok = if c.is_ascii_digit() {
            Ok(Tok::Digit(c.to_string()))
        } else if DIGIT_PUNC.contains(&c) {
            Ok(Tok::Punc(c.to_string()))
        } else {
            Err(Tok::Char(c.to_string()))
        };
        let tok = match tok {
            Ok(t) => t,
            Err(t) => t,
        };
        match (out.last_mut(), &tok) {
            (Some(Tok::Digit(s)), Tok::Digit(d)) => s.push_str(d),
            (Some(Tok::Punc(s)), Tok::Punc(p)) => s.push_str(p),
            (Some(Tok::Char(s)), Tok::Char(d)) => s.push_str(d),
            _ => out.push(tok),
        }
    }
    out
}

#[derive(Clone)]
enum Elem {
    Tok(Tok),
    Group(Vec<Tok>),
}

fn elem_txt(e: &Elem) -> &str {
    match e {
        Elem::Tok(t) => t.txt(),
        Elem::Group(g) => g.first().map(|t| t.txt()).unwrap_or(""),
    }
}

fn elem_name(e: &Elem) -> &'static str {
    match e {
        Elem::Tok(t) => t.name(),
        Elem::Group(_) => "group",
    }
}

// ---------------------------------------------------------------------------
// Scoring (only the classes that score digit runs)
// ---------------------------------------------------------------------------

const AVA_MM: &[&str] = &[
    "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "1", "2", "3", "4",
    "5", "6", "7", "8", "9",
];
const AVA_DD: &[&str] = &[
    "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15",
    "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30",
    "31", "1", "2", "3", "4", "5", "6", "7", "8", "9",
];

fn noon_map(prefix: &str) -> Option<&'static str> {
    match prefix {
        "a m" | "a.m." | "am" | "A M" | "AM" => Some("上午"),
        "p m" | "p.m." | "pm" | "P M" | "PM" => Some("下午"),
        _ => None,
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ScoreClass {
    Date,
    Decimal,
    Time,
    Version,
    Ip,
    Frac,
    PhoneDash,
    Formula,
}

fn score(class: ScoreClass, stack: &[Elem], idx: usize) -> i32 {
    let g = match &stack[idx] {
        Elem::Group(g) => g,
        _ => return -1,
    };
    let prev_txt = if idx > 0 { elem_txt(&stack[idx - 1]).to_string() } else { String::new() };
    let next_txt = stack.get(idx + 1).map(|e| elem_txt(e).to_string()).unwrap_or_default();
    match class {
        ScoreClass::Date => {
            if g.len() == 5
                && g[1].txt() == g[3].txt()
                && [".", "-", "/"].contains(&g[1].txt())
                && g[0].txt().len() == 4
                && AVA_MM.contains(&g[2].txt())
                && AVA_DD.contains(&g[4].txt())
            {
                80
            } else {
                -1
            }
        }
        ScoreClass::Decimal => {
            if g.len() == 3 && g[1].txt() == "." {
                50
            } else {
                -1
            }
        }
        ScoreClass::Time => {
            let pre_txt = idx > 0
                && ["早上", "上午", "中午", "下午", "晚上", "傍晚", "凌晨", "半夜"]
                    .iter()
                    .any(|w| prev_txt.ends_with(w));
            let mut noon = None;
            if !next_txt.is_empty() {
                for len in 2..=4 {
                    let prefix: String = next_txt.chars().take(len).collect();
                    if let Some(n) = noon_map(&prefix) {
                        noon = Some(n);
                        break;
                    }
                }
            }
            let two = g.len() == 3
                && [":", "："].contains(&g[1].txt())
                && g[0].txt().len() < 3
                && g[2].txt().len() < 3;
            let three = g.len() == 5
                && [":", "："].contains(&g[1].txt())
                && [":", "："].contains(&g[3].txt())
                && g[0].txt().len() < 3
                && g[2].txt().len() < 3
                && g[4].txt().len() < 3;
            if two && (pre_txt || noon.is_some()) {
                90
            } else if two {
                60
            } else if three && (pre_txt || noon.is_some()) {
                90
            } else if three {
                80
            } else {
                -1
            }
        }
        ScoreClass::Version => {
            if g.len() == 5 && g[1].txt() == "." && g[3].txt() == "." {
                if idx > 0 && (prev_txt.ends_with('v') || prev_txt.ends_with('V')) {
                    90
                } else if idx > 0 && prev_txt.contains("版本") {
                    90
                } else {
                    50
                }
            } else {
                -1
            }
        }
        ScoreClass::Ip => {
            if g.len() == 7
                && g[1].txt() == g[3].txt()
                && g[3].txt() == g[5].txt()
                && g[1].txt() == "."
            {
                50
            } else {
                -1
            }
        }
        ScoreClass::Frac => {
            if g.len() == 3 && g[1].txt() == "/" {
                50
            } else if g.len() == 5
                && ["-", "~", "～"].contains(&g[1].txt())
                && g[3].txt() == "/"
            {
                50
            } else {
                -1
            }
        }
        ScoreClass::PhoneDash => {
            if g.len() == 3
                && g[0].txt().len() == 3
                && g[1].txt() == "-"
                && g[2].txt().len() == 8
            {
                80
            } else if g.len() == 3
                && g[0].txt().len() == 4
                && g[1].txt() == "-"
                && [7, 8].contains(&g[2].txt().len())
            {
                80
            } else if g.len() == 5
                && (g[0].txt() == "400"
                    || ([3, 4].contains(&g[0].txt().len()) && g[0].txt().starts_with('0')))
                && g[1].txt() == "-"
                && g[3].txt() == "-"
                && [3, 4].contains(&g[2].txt().len())
                && g[4].txt().len() == 4
            {
                90
            } else {
                -1
            }
        }
        ScoreClass::Formula => {
            if g.len() < 3 {
                return -1;
            }
            let ops: Vec<&str> = (1..g.len()).step_by(2).map(|i| g[i].txt()).collect();
            if ops.iter().all(|o| *o == "-") {
                return -1;
            }
            if ops.iter().all(|o| operator(o) != "") {
                if ops.contains(&"=") {
                    85
                } else {
                    10
                }
            } else {
                -1
            }
        }
    }
}

fn deal(class: ScoreClass, stack: &mut Vec<Elem>, idx: usize) -> usize {
    match class {
        ScoreClass::Date => {
            if let Elem::Group(g) = &stack[idx] {
                stack[idx] = Elem::Tok(Tok::Date {
                    yyyy: Some(Box::new(g[0].clone())),
                    mm: Some(Box::new(g[2].clone())),
                    dd: Some(Box::new(g[4].clone())),
                    day: None,
                });
            }
        }
        ScoreClass::Decimal => {
            if let Elem::Group(g) = &stack[idx] {
                stack[idx] = Elem::Tok(Tok::Decimal {
                    int: Box::new(g[0].clone()),
                    frac: Box::new(g[2].clone()),
                    range: None,
                });
            }
        }
        ScoreClass::Time => {
            let mut noon = None;
            if let Some(Elem::Tok(next)) = stack.get_mut(idx + 1) {
                let t = next.txt().to_string();
                for len in 2..=4 {
                    let prefix: String = t.chars().take(len).collect();
                    if let Some(n) = noon_map(&prefix) {
                        noon = Some(n);
                        let rest: String = t.chars().skip(len).collect();
                        *next = Tok::Char(rest);
                        break;
                    }
                }
            }
            if let Elem::Group(g) = &stack[idx] {
                stack[idx] = Elem::Tok(Tok::Time {
                    hour: Box::new(g[0].clone()),
                    minute: Box::new(g[2].clone()),
                    second: if g.len() == 5 { Some(Box::new(g[4].clone())) } else { None },
                    noon: noon.map(|s| s.to_string()),
                });
            }
        }
        ScoreClass::Version => {
            if let Elem::Group(g) = &stack[idx] {
                stack[idx] = Elem::Tok(Tok::Version(vec![g[0].clone(), g[2].clone(), g[4].clone()]));
            }
        }
        ScoreClass::Ip => {
            if let Elem::Group(g) = &stack[idx] {
                stack[idx] = Elem::Tok(Tok::Ip(vec![
                    g[0].clone(),
                    g[2].clone(),
                    g[4].clone(),
                    g[6].clone(),
                ]));
            }
        }
        ScoreClass::Frac => {
            if let Elem::Group(g) = &stack[idx] {
                stack[idx] = Elem::Tok(if g.len() == 3 {
                    Tok::Frac {
                        num: Box::new(g[0].clone()),
                        den: Box::new(g[2].clone()),
                        range: None,
                    }
                } else {
                    Tok::Frac {
                        num: Box::new(g[2].clone()),
                        den: Box::new(g[4].clone()),
                        range: Some(Box::new(g[0].clone())),
                    }
                });
            }
        }
        ScoreClass::PhoneDash => {
            if let Elem::Group(g) = &stack[idx] {
                stack[idx] = Elem::Tok(Tok::PhoneDash(if g.len() == 3 {
                    vec![g[0].clone(), g[2].clone()]
                } else {
                    vec![g[0].clone(), g[2].clone(), g[4].clone()]
                }));
            }
        }
        ScoreClass::Formula => {
            if let Elem::Group(g) = &stack[idx] {
                stack[idx] = Elem::Tok(Tok::Formula(g.clone()));
            }
        }
    }
    idx + 1
}

// ---------------------------------------------------------------------------
// digit -> unit binding helpers
// ---------------------------------------------------------------------------

fn try_digit_before(
    stack: &mut Vec<Elem>,
    idx: usize,
    dict: &HashMap<String, String>,
    lens: &[usize],
    mk: fn(Box<Tok>, String) -> Tok,
) -> bool {
    if idx == 0 {
        return false;
    }
    let prev_name = elem_name(&stack[idx - 1]);
    if prev_name != "char" && prev_name != "punc" {
        return false;
    }
    let (prev_txt, prev_is_punc) = match &stack[idx - 1] {
        Elem::Tok(t) => (t.txt().to_string(), matches!(t, Tok::Punc(_))),
        _ => return false,
    };
    let _ = prev_is_punc;
    let chars: Vec<char> = prev_txt.chars().collect();
    for &len in lens {
        if len > chars.len() {
            continue;
        }
        let start = chars.len() - len;
        let suffix: String = chars[start..].iter().collect();
        if let Some(tag) = dict.get(&suffix) {
            if suffix.chars().next().map(|c| c.is_ascii_alphabetic()).unwrap_or(false)
                && len < chars.len()
                && chars[start - 1].is_ascii_alphabetic()
            {
                continue;
            }
            let rest: String = chars[..start].iter().collect();
            if let Elem::Tok(p) = &mut stack[idx - 1] {
                *p = Tok::Char(rest);
            }
            let d = match &stack[idx] {
                Elem::Tok(t) => t.clone(),
                _ => return false,
            };
            stack[idx] = Elem::Tok(mk(Box::new(d), tag.clone()));
            return true;
        }
    }
    false
}

fn try_digit_after(
    stack: &mut Vec<Elem>,
    idx: usize,
    dict: &HashMap<String, String>,
    lens: &[usize],
    mk: fn(Box<Tok>, String) -> Tok,
) -> bool {
    if idx + 1 >= stack.len() {
        return false;
    }
    let next_name = elem_name(&stack[idx + 1]);
    if next_name != "char" && next_name != "punc" {
        return false;
    }
    let next_txt = match &stack[idx + 1] {
        Elem::Tok(t) => t.txt().to_string(),
        _ => return false,
    };
    let chars: Vec<char> = next_txt.chars().collect();
    for &len in lens {
        if len > chars.len() {
            continue;
        }
        let prefix: String = chars[..len].iter().collect();
        if let Some(tag) = dict.get(&prefix) {
            if prefix.chars().last().map(|c| c.is_ascii_alphabetic()).unwrap_or(false)
                && len < chars.len()
                && chars[len].is_ascii_alphabetic()
            {
                continue;
            }
            let rest: String = chars[len..].iter().collect();
            if let Elem::Tok(p) = &mut stack[idx + 1] {
                *p = Tok::Char(rest);
            }
            let d = match &stack[idx] {
                Elem::Tok(t) => t.clone(),
                _ => return false,
            };
            stack[idx] = Elem::Tok(mk(Box::new(d), tag.clone()));
            return true;
        }
    }
    false
}

fn with_range(t: &Tok, start: Tok) -> Tok {
    match t.clone() {
        Tok::MeasureAfter { d, tag, .. } => Tok::MeasureAfter {
            d,
            tag,
            range: Some(Box::new(start)),
        },
        Tok::MeasureBefore { d, tag, .. } => Tok::MeasureBefore {
            d,
            tag,
            range: Some(Box::new(start)),
        },
        Tok::MoneyAfter { d, tag, .. } => Tok::MoneyAfter {
            d,
            tag,
            range: Some(Box::new(start)),
        },
        Tok::MoneyBefore { d, tag, .. } => Tok::MoneyBefore {
            d,
            tag,
            range: Some(Box::new(start)),
        },
        Tok::Percent { d, .. } => Tok::Percent {
            d,
            range: Some(Box::new(start)),
        },
        Tok::Decimal { int, frac, .. } => Tok::Decimal {
            int,
            frac,
            range: Some(Box::new(start)),
        },
        Tok::Frac { num, den, .. } => Tok::Frac {
            num,
            den,
            range: Some(Box::new(start)),
        },
        other => other,
    }
}

fn merge_punc_measure(stack: &mut Vec<Elem>, idx: usize) -> usize {
    if idx == 0 || idx >= stack.len() {
        return idx + 1;
    }
    if elem_name(&stack[idx - 1]) != "punc" {
        return idx + 1;
    }
    let p = elem_txt(&stack[idx - 1]).to_string();
    if !["-", "~", "～"].contains(&p.as_str()) {
        return idx + 1;
    }
    if idx >= 2 {
        let a = elem_name(&stack[idx - 2]).to_string();
        let b = elem_name(&stack[idx]).to_string();
        if a == b
            || (["measure_before", "money"].contains(&a.as_str())
                && ["measure_after", "money"].contains(&b.as_str()))
        {
            stack[idx - 1] = Elem::Tok(Tok::char("至"));
            return idx + 1;
        }
        if a == "digit" {
            let start = match &stack[idx - 2] {
                Elem::Tok(t) => t.clone(),
                _ => return idx + 1,
            };
            let unit_tok = match &stack[idx] {
                Elem::Tok(t) => with_range(t, start),
                _ => return idx + 1,
            };
            let mut new_stack: Vec<Elem> = stack[..idx - 2].to_vec();
            new_stack.push(Elem::Tok(unit_tok));
            new_stack.extend(stack[idx + 1..].iter().cloned());
            *stack = new_stack;
            return idx - 1;
        }
        if a == "char" {
            let rep = if p == "-" { "负" } else { "大约" };
            stack[idx - 1] = Elem::Tok(Tok::char(rep));
        }
    } else {
        let rep = if p == "-" { "负" } else { "大约" };
        stack[idx - 1] = Elem::Tok(Tok::char(rep));
    }
    idx + 1
}

fn merge_punc_range(stack: &mut Vec<Elem>, idx: usize) -> usize {
    if idx >= 2 && idx - 1 < stack.len() && elem_name(&stack[idx - 1]) == "punc" {
        let p = elem_txt(&stack[idx - 1]).to_string();
        if ["-", "~", "～"].contains(&p.as_str()) && elem_name(&stack[idx - 2]) == elem_name(&stack[idx]) {
            stack[idx - 1] = Elem::Tok(Tok::char("至"));
        }
    }
    idx + 1
}

fn merge_punc(stack: &mut Vec<Elem>, idx: usize) -> usize {
    match elem_name(&stack[idx]) {
        "measure_before" | "measure_after" | "money" | "frac" | "decimal" | "percent" => {
            merge_punc_measure(stack, idx)
        }
        "date" | "time" | "ip" => merge_punc_range(stack, idx),
        _ => idx + 1,
    }
}

// ---------------------------------------------------------------------------
// Date matcher (the multi-token form)
// ---------------------------------------------------------------------------

fn date_try_digit(stack: &mut Vec<Elem>, idx: usize) -> bool {
    let txt = |e: &Elem| elem_txt(e).to_string();
    let name = |e: &Elem| elem_name(e);
    let n = stack.len();
    if idx + 5 < n
        && name(&stack[idx]) == "digit"
        && name(&stack[idx + 2]) == "digit"
        && name(&stack[idx + 4]) == "digit"
        && txt(&stack[idx + 1]) == "年"
        && txt(&stack[idx + 3]) == "月"
        && txt(&stack[idx + 5]).starts_with(['日', '号'])
        && AVA_MM.contains(&txt(&stack[idx + 2]).as_str())
        && AVA_DD.contains(&txt(&stack[idx + 4]).as_str())
    {
        let yyyy = stack.remove(idx);
        let _year = stack.remove(idx);
        let mm = stack.remove(idx);
        let _month = stack.remove(idx);
        let dd = stack.remove(idx);
        let day_txt = txt(&stack[idx]);
        let day = day_txt.chars().next();
        let rest: String = day_txt.chars().skip(1).collect();
        if let Elem::Tok(t) = &mut stack[idx] {
            *t = Tok::Char(rest);
        }
        let mk = |e: Elem| match e {
            Elem::Tok(t) => t,
            _ => Tok::Char(String::new()),
        };
        stack.insert(
            idx,
            Elem::Tok(Tok::Date {
                yyyy: Some(Box::new(mk(yyyy))),
                mm: Some(Box::new(mk(mm))),
                dd: Some(Box::new(mk(dd))),
                day,
            }),
        );
        return true;
    }
    if idx + 3 < n
        && name(&stack[idx]) == "digit"
        && name(&stack[idx + 2]) == "digit"
        && txt(&stack[idx + 1]) == "年"
        && txt(&stack[idx + 3]).starts_with('月')
        && AVA_MM.contains(&txt(&stack[idx + 2]).as_str())
    {
        let yyyy = stack.remove(idx);
        let _year = stack.remove(idx);
        let mm = stack.remove(idx);
        let mt = txt(&stack[idx]);
        let rest: String = mt.chars().skip(1).collect();
        if let Elem::Tok(t) = &mut stack[idx] {
            *t = Tok::Char(rest);
        }
        let mk = |e: Elem| match e {
            Elem::Tok(t) => t,
            _ => Tok::Char(String::new()),
        };
        stack.insert(
            idx,
            Elem::Tok(Tok::Date {
                yyyy: Some(Box::new(mk(yyyy))),
                mm: Some(Box::new(mk(mm))),
                dd: None,
                day: None,
            }),
        );
        return true;
    }
    if idx + 3 < n
        && name(&stack[idx]) == "digit"
        && name(&stack[idx + 2]) == "digit"
        && txt(&stack[idx + 1]) == "月"
        && txt(&stack[idx + 3]).starts_with(['日', '号'])
        && AVA_MM.contains(&txt(&stack[idx]).as_str())
        && AVA_DD.contains(&txt(&stack[idx + 2]).as_str())
    {
        let mm = stack.remove(idx);
        let _month = stack.remove(idx);
        let dd = stack.remove(idx);
        let day_txt = txt(&stack[idx]);
        let day = day_txt.chars().next();
        let rest: String = day_txt.chars().skip(1).collect();
        if let Elem::Tok(t) = &mut stack[idx] {
            *t = Tok::Char(rest);
        }
        let mk = |e: Elem| match e {
            Elem::Tok(t) => t,
            _ => Tok::Char(String::new()),
        };
        stack.insert(
            idx,
            Elem::Tok(Tok::Date {
                yyyy: None,
                mm: Some(Box::new(mk(mm))),
                dd: Some(Box::new(mk(dd))),
                day,
            }),
        );
        return true;
    }
    if idx + 1 < n
        && txt(&stack[idx + 1]).starts_with('年')
        && txt(&stack[idx]).chars().count() == 4
        && name(&stack[idx]) == "digit"
    {
        let y = stack[idx].clone();
        let mk = |e: Elem| match e {
            Elem::Tok(t) => t,
            _ => Tok::Char(String::new()),
        };
        stack[idx] = Elem::Tok(Tok::Date {
            yyyy: Some(Box::new(mk(y))),
            mm: None,
            dd: None,
            day: None,
        });
        let nt = txt(&stack[idx + 1]);
        let rest: String = nt.chars().skip(1).collect();
        if let Elem::Tok(t) = &mut stack[idx + 1] {
            *t = Tok::Char(rest);
        }
        return true;
    }
    false
}

// ---------------------------------------------------------------------------
// The pipeline
// ---------------------------------------------------------------------------

pub fn talker_re(cfg: &ReConfig, text: &str) -> String {
    // 1. character substitutions
    let mut mapped = String::with_capacity(text.len());
    for c in text.chars() {
        match cfg.char_map0.get(&c) {
            Some(v) => mapped.push_str(v),
            None => mapped.push(c),
        }
    }
    // 2. proper nouns
    let mut proper = String::with_capacity(mapped.len());
    let mut last = 0usize;
    for m in cfg.proper_re.find_iter(&mapped) {
        let Ok(m) = m else { continue };
        proper.push_str(&mapped[last..m.start()]);
        if let Some(v) = cfg.proper_noun.get(m.as_str()) {
            proper.push_str(v);
        }
        last = m.end();
    }
    proper.push_str(&mapped[last..]);
    let text = proper;

    // 3. serial-number spans
    let mut segs: Vec<Tok> = Vec::new();
    let sn_re = Regex::new(r"(?=[A-Z0-9-]*[A-Z])(?=[A-Z0-9-]*[0-9])[A-Z0-9-]+").unwrap();
    let mut last = 0usize;
    for m in sn_re.find_iter(&text) {
        let Ok(m) = m else { continue };
        if m.start() > last {
            segs.push(Tok::Char(text[last..m.start()].to_string()));
        }
        segs.push(Tok::Sn { txt: m.as_str().to_string(), det: false });
        last = m.end();
    }
    if last < text.len() {
        segs.push(Tok::Char(text[last..].to_string()));
    }
    if segs.is_empty() {
        segs.push(Tok::Char(text.clone()));
    }
    // RESN.det
    for i in 0..segs.len() {
        if let Tok::Sn { txt, det } = &segs[i] {
            let prev = if i > 0 { segs[i - 1].txt().to_string() } else { String::new() };
            let mut is_det = !prev.is_empty()
                && ["SN", "sn", "序列号", "序列"]
                    .iter()
                    .any(|k| prev.contains(k));
            if !is_det {
                let mut interleave: Vec<char> = Vec::new();
                for c in txt.chars() {
                    let ty = if c.is_ascii_digit() {
                        '0'
                    } else if c.is_ascii_alphabetic() {
                        'A'
                    } else {
                        '-'
                    };
                    if interleave.last() != Some(&ty) {
                        interleave.push(ty);
                    }
                }
                is_det = interleave.len() > 5;
            }
            if is_det {
                if let Tok::Sn { det, .. } = &mut segs[i] {
                    *det = true;
                }
            }
        }
    }
    let seg_text: String = segs.iter().map(|t| t.to_string()).collect();

    // 4. tokenize
    let mut base_s = tokenize(&seg_text);
    base_s.push(Tok::Char(String::new()));

    // 5. merge runs around characters
    let mut stack: Vec<Elem> = Vec::new();
    let mut last_char: i64 = -1;
    let mut idx = 0usize;
    while idx < base_s.len() {
        if base_s[idx].name() == "char" {
            let li = (last_char + 1) as usize;
            if li < base_s.len() && base_s[li].name() == "punc" {
                stack.push(Elem::Tok(base_s[li].clone()));
                last_char += 1;
            }
            let lo = (last_char + 1) as usize;
            if (idx as i64 - last_char) > 3 {
                if idx > 0 && base_s[idx - 1].name() == "punc" {
                    stack.push(Elem::Group(base_s[lo..idx - 1].to_vec()));
                    stack.push(Elem::Tok(base_s[idx - 1].clone()));
                } else {
                    stack.push(Elem::Group(base_s[lo..idx].to_vec()));
                }
            } else {
                for t in &base_s[lo..idx] {
                    stack.push(Elem::Tok(t.clone()));
                }
            }
            stack.push(Elem::Tok(base_s[idx].clone()));
            last_char = idx as i64;
        }
        idx += 1;
    }

    // 6. score and deal digit runs
    let classes = [
        ScoreClass::Date,
        ScoreClass::Decimal,
        ScoreClass::Time,
        ScoreClass::Version,
        ScoreClass::Ip,
        ScoreClass::Frac,
        ScoreClass::PhoneDash,
        ScoreClass::Formula,
    ];
    let mut idx = 0usize;
    while idx < stack.len() {
        if matches!(stack[idx], Elem::Group(_)) {
            let mut best = -1i32;
            let mut best_cls = None;
            for cls in classes {
                let s = score(cls, &stack, idx);
                if s >= best {
                    best = s;
                    best_cls = Some(cls);
                }
            }
            if best < 0 {
                let g = match &stack[idx] {
                    Elem::Group(g) => g.clone(),
                    _ => unreachable!(),
                };
                let mut ranges: Vec<usize> = g
                    .iter()
                    .enumerate()
                    .filter(|(_, t)| ["-", "~", "～"].contains(&t.txt()))
                    .map(|(i, _)| i)
                    .collect();
                let mid = g.len() as f64 / 2.0;
                ranges.sort_by(|a, b| {
                    let da = (*a as f64 - mid).abs();
                    let db = (*b as f64 - mid).abs();
                    da.partial_cmp(&db).unwrap()
                });
                if let Some(&ri) = ranges.first() {
                    let mut stack2: Vec<Elem> = Vec::new();
                    stack2.extend(stack[..idx].iter().map(|e| match e {
                        Elem::Tok(t) => Elem::Tok(t.clone()),
                        Elem::Group(g) => Elem::Group(g.clone()),
                    }));
                    stack2.push(Elem::Group(g[..ri].to_vec()));
                    stack2.push(Elem::Tok(Tok::char("至")));
                    stack2.push(Elem::Group(g[ri + 1..].to_vec()));
                    stack2.extend(stack[idx + 1..].iter().map(|e| match e {
                        Elem::Tok(t) => Elem::Tok(t.clone()),
                        Elem::Group(g) => Elem::Group(g.clone()),
                    }));
                    let mut b0 = -1i32;
                    let mut c0 = None;
                    let mut b1 = -1i32;
                    let mut c1 = None;
                    for cls in classes {
                        let s0 = score(cls, &stack2, idx);
                        if s0 >= b0 {
                            b0 = s0;
                            c0 = Some(cls);
                        }
                        let s1 = score(cls, &stack2, idx + 2);
                        if s1 >= b1 {
                            b1 = s1;
                            c1 = Some(cls);
                        }
                    }
                    if b0 >= 0 && b1 >= 0 && c0 == c1 {
                        let cls = c0.unwrap();
                        stack = stack2;
                        let mut i = deal(cls, &mut stack, idx);
                        i = deal(cls, &mut stack, i + 1);
                        idx = i;
                        continue;
                    }
                    // flatten
                    let g = match &stack[idx] {
                        Elem::Group(g) => g.clone(),
                        _ => unreachable!(),
                    };
                    stack.remove(idx);
                    for (off, t) in g.iter().enumerate() {
                        stack.insert(idx + off, Elem::Tok(t.clone()));
                    }
                    idx += g.len();
                } else {
                    let g = match &stack[idx] {
                        Elem::Group(g) => g.clone(),
                        _ => unreachable!(),
                    };
                    stack.remove(idx);
                    for (off, t) in g.iter().enumerate() {
                        stack.insert(idx + off, Elem::Tok(t.clone()));
                    }
                    idx += g.len();
                }
            } else {
                idx = deal(best_cls.unwrap(), &mut stack, idx);
            }
        } else {
            idx += 1;
        }
    }

    // 7. bind units around digits
    let money_len = ReConfig::lens(&cfg.money);
    let mb_len = ReConfig::lens(&cfg.measure_before);
    let ma_len = ReConfig::lens(&cfg.measure_after);
    let sb_len = ReConfig::lens(&cfg.serial_before);
    let sa_len = ReConfig::lens(&cfg.serial_after);
    let mut idx = 0usize;
    while idx < stack.len() {
        let name = elem_name(&stack[idx]).to_string();
        if name == "char" && elem_txt(&stack[idx]).is_empty() {
            stack.remove(idx);
            continue;
        }
        if name != "digit" && name != "decimal" {
            idx += 1;
            continue;
        }
        let money = &cfg.money;
        let mb = &cfg.measure_before;
        let ma = &cfg.measure_after;
        let sb = &cfg.serial_before;
        let sa = &cfg.serial_after;
        let done = try_digit_before(&mut stack, idx, money, &money_len, |d, tag| Tok::MoneyBefore {
            d,
            tag,
            range: None,
        });
        if done {
            idx += 1;
            continue;
        }
        if date_try_digit(&mut stack, idx) {
            idx += 1;
            continue;
        }
        let done = try_digit_after(&mut stack, idx, money, &money_len, |d, tag| Tok::MoneyAfter {
            d,
            tag,
            range: None,
        });
        if done {
            idx += 1;
            continue;
        }
        let done = try_digit_before(&mut stack, idx, mb, &mb_len, |d, tag| Tok::MeasureBefore {
            d,
            tag,
            range: None,
        });
        if done {
            idx += 1;
            continue;
        }
        // measure_before special case: previous measure_after with tag 等于
        if idx > 0 {
            if let Elem::Tok(Tok::MeasureAfter { tag, .. }) = &stack[idx - 1] {
                if tag == "等于" {
                    let d = match &stack[idx] {
                        Elem::Tok(t) => t.clone(),
                        _ => unreachable!(),
                    };
                    stack[idx] = Elem::Tok(Tok::MeasureBefore {
                        d: Box::new(d),
                        tag: String::new(),
                        range: None,
                    });
                    idx += 1;
                    continue;
                }
            }
        }
        let done = try_digit_after(&mut stack, idx, ma, &ma_len, |d, tag| Tok::MeasureAfter {
            d,
            tag,
            range: None,
        });
        if done {
            idx += 1;
            continue;
        }
        // percent: next token starts with '%'
        if idx + 1 < stack.len() {
            let nt = elem_txt(&stack[idx + 1]).to_string();
            if let Some(rest) = nt.strip_prefix('%') {
                if let Elem::Tok(t) = &mut stack[idx + 1] {
                    *t = Tok::Char(rest.to_string());
                }
                let d = match &stack[idx] {
                    Elem::Tok(t) => t.clone(),
                    _ => unreachable!(),
                };
                stack[idx] = Elem::Tok(Tok::Percent {
                    d: Box::new(d),
                    range: None,
                });
                idx += 1;
                continue;
            }
        }
        let done = try_digit_before(&mut stack, idx, sb, &sb_len, |d, tag| Tok::SerialBefore {
            d,
            tag,
        });
        if done {
            idx += 1;
            continue;
        }
        let done = try_digit_after(&mut stack, idx, sa, &sa_len, |d, tag| Tok::SerialAfter { d, tag });
        if done {
            idx += 1;
            continue;
        }
        // order: "N. " at the start of a sentence
        let is_order = {
            let d = elem_txt(&stack[idx]).to_string();
            d.chars().count() < 2
                && idx + 2 < stack.len()
                && elem_txt(&stack[idx + 1]) == "."
                && elem_txt(&stack[idx + 2]).starts_with(' ')
        };
        if is_order {
            let d = match &stack[idx] {
                Elem::Tok(t) => t.clone(),
                _ => unreachable!(),
            };
            stack[idx] = Elem::Tok(Tok::Order(Box::new(d)));
        }
        idx += 1;
    }

    // 8. fold punctuation into ranges / signs
    let mut idx = 0usize;
    while idx < stack.len() {
        idx = merge_punc(&mut stack, idx);
        if idx > stack.len() {
            break;
        }
    }

    // 9. join
    stack
        .iter()
        .map(|e| match e {
            // Unclaimed digits keep their ASCII form (talker_re returns ii.txt).
            Elem::Tok(Tok::Digit(t)) => t.clone(),
            Elem::Tok(t) => t.to_string(),
            Elem::Group(g) => g
                .iter()
                .map(|t| match t {
                    Tok::Digit(d) => d.clone(),
                    _ => t.to_string(),
                })
                .collect(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn number_reading() {
        assert_eq!(to_string_number("11"), "十一");
        assert_eq!(to_string_number("350"), "三百五十");
        assert_eq!(to_string_number("1024"), "一千零二十四");
        assert_eq!(to_string_number("2026"), "两千零二十六");
        assert_eq!(to_string_number("100000"), "十万");
        assert_eq!(to_string_rm0("2026"), "二零二六");
        assert_eq!(to_string_yao("1380"), "幺三八零");
    }
}
