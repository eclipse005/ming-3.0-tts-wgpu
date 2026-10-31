//! Text front end (port of frontend/toolkit.py + frontend/text_segment_cut.py).
//!
//! The port is character-for-character: every branch of the Python reference
//! has a counterpart here, including its quirks (the \x60，\x60-for-… substitution
//! in the fallback whitelist, the empty-final-sentence branch in the sentence
//! splitter, the DOTPLACEHOLDER round trip, ...).
//!
//! Python's Unicode-aware regex classes are reproduced by hand-written scanners
//! over chars, which keeps the behaviour independent of a regex engine's
//! Unicode tables; the patterns whose behaviour depends on backtracking
//! (Markdown code spans) and Python's \d (Unicode Nd, not merely ASCII) use
//! the regex crate.
//!
//! Known, deliberate limits (see the notes in the individual functions):
//! * NFC composition in get_semantic_length is algorithmic: the Latin
//!   composition pairs are generated from unicodedata.decomposition and Hangul
//!   L+V(+T) is composed arithmetically. Canonical reordering of several
//!   combining marks on one starter is not implemented; it cannot change the
//!   counted characters for this corpus.
//! * cut_text_by_semantic_length returns the fragments; the reference also
//!   returns calibrated positions, which do not influence the fragments.

#![allow(dead_code)] // wired into the pipeline with another milestone

// ---------------------------------------------------------------------------
// Paralinguistic tags
// ---------------------------------------------------------------------------

/// Inline paralinguistic tags understood by Ming-3.0-tts (frontend/toolkit.py).
pub const PARALINGUISTIC_TAGS: [&str; 14] = [
    "[Breathing]",
    "[Cough]",
    "[Sigh]",
    "[Laughter]",
    "[Surprise-ah]",
    "[Surprise-oh]",
    "[Dissatisfaction-hnn]",
    "[Uhm]",
    "[Confirmation-en]",
    "[Question-ei]",
    "[Question-ah]",
    "[Question-en]",
    "[Question-oh]",
    "[Question-huh]",
];

// ---------------------------------------------------------------------------
// Python-compatible character predicates and string helpers
// ---------------------------------------------------------------------------

/// str.isspace(): Unicode White_Space plus U+001C..U+001F (bidi class B/S).
fn py_isspace(c: char) -> bool {
    matches!(c,
        '\u{0009}'..='\u{000D}'
        | '\u{001C}'..='\u{001F}'
        | '\u{0020}'
        | '\u{0085}'
        | '\u{00A0}'
        | '\u{1680}'
        | '\u{2000}'..='\u{200A}'
        | '\u{2028}'
        | '\u{2029}'
        | '\u{202F}'
        | '\u{205F}'
        | '\u{3000}')
}

/// str.strip().
pub fn py_strip(s: &str) -> &str {
    let start = s
        .char_indices()
        .find(|(_, c)| !py_isspace(*c))
        .map(|(i, _)| i)
        .unwrap_or(s.len());
    let end = s
        .char_indices()
        .rev()
        .find(|(_, c)| !py_isspace(*c))
        .map(|(i, c)| i + c.len_utf8())
        .unwrap_or(start);
    &s[start..end]
}

/// str.rstrip().
fn py_rstrip(s: &str) -> &str {
    let end = s
        .char_indices()
        .rev()
        .find(|(_, c)| !py_isspace(*c))
        .map(|(i, c)| i + c.len_utf8())
        .unwrap_or(0);
    &s[..end]
}

/// str.lstrip(chars) for an explicit character set.
fn py_lstrip_set<'a>(s: &'a str, set: &str) -> &'a str {
    let mut start = 0;
    for (i, c) in s.char_indices() {
        if set.contains(c) {
            start = i + c.len_utf8();
        } else {
            break;
        }
    }
    &s[start..]
}

/// string.punctuation.
const PY_PUNCTUATION: &str = "!\"#$%&'()*+,-./:;<=>?@[\\]^_\x60{|}~";

/// string.whitespace.
const PY_ASCII_WHITESPACE: &str = " \t\n\r\x0b\x0c";

/// Python str.isalnum() code points, as ranges.
const PY_ALNUM_RANGES: [(u32, u32); 706] = [
    (0x30, 0x39), (0x41, 0x5A), (0x61, 0x7A), (0xAA, 0xAA), (0xB2, 0xB3), (0xB5, 0xB5),
    (0xB9, 0xBA), (0xBC, 0xBE), (0xC0, 0xD6), (0xD8, 0xF6), (0xF8, 0x2C1), (0x2C6, 0x2D1),
    (0x2E0, 0x2E4), (0x2EC, 0x2EC), (0x2EE, 0x2EE), (0x370, 0x374), (0x376, 0x377),
    (0x37A, 0x37D), (0x37F, 0x37F), (0x386, 0x386), (0x388, 0x38A), (0x38C, 0x38C),
    (0x38E, 0x3A1), (0x3A3, 0x3F5), (0x3F7, 0x481), (0x48A, 0x52F), (0x531, 0x556),
    (0x559, 0x559), (0x560, 0x588), (0x5D0, 0x5EA), (0x5EF, 0x5F2), (0x620, 0x64A),
    (0x660, 0x669), (0x66E, 0x66F), (0x671, 0x6D3), (0x6D5, 0x6D5), (0x6E5, 0x6E6),
    (0x6EE, 0x6FC), (0x6FF, 0x6FF), (0x710, 0x710), (0x712, 0x72F), (0x74D, 0x7A5),
    (0x7B1, 0x7B1), (0x7C0, 0x7EA), (0x7F4, 0x7F5), (0x7FA, 0x7FA), (0x800, 0x815),
    (0x81A, 0x81A), (0x824, 0x824), (0x828, 0x828), (0x840, 0x858), (0x860, 0x86A),
    (0x8A0, 0x8B4), (0x8B6, 0x8C7), (0x904, 0x939), (0x93D, 0x93D), (0x950, 0x950),
    (0x958, 0x961), (0x966, 0x96F), (0x971, 0x980), (0x985, 0x98C), (0x98F, 0x990),
    (0x993, 0x9A8), (0x9AA, 0x9B0), (0x9B2, 0x9B2), (0x9B6, 0x9B9), (0x9BD, 0x9BD),
    (0x9CE, 0x9CE), (0x9DC, 0x9DD), (0x9DF, 0x9E1), (0x9E6, 0x9F1), (0x9F4, 0x9F9),
    (0x9FC, 0x9FC), (0xA05, 0xA0A), (0xA0F, 0xA10), (0xA13, 0xA28), (0xA2A, 0xA30),
    (0xA32, 0xA33), (0xA35, 0xA36), (0xA38, 0xA39), (0xA59, 0xA5C), (0xA5E, 0xA5E),
    (0xA66, 0xA6F), (0xA72, 0xA74), (0xA85, 0xA8D), (0xA8F, 0xA91), (0xA93, 0xAA8),
    (0xAAA, 0xAB0), (0xAB2, 0xAB3), (0xAB5, 0xAB9), (0xABD, 0xABD), (0xAD0, 0xAD0),
    (0xAE0, 0xAE1), (0xAE6, 0xAEF), (0xAF9, 0xAF9), (0xB05, 0xB0C), (0xB0F, 0xB10),
    (0xB13, 0xB28), (0xB2A, 0xB30), (0xB32, 0xB33), (0xB35, 0xB39), (0xB3D, 0xB3D),
    (0xB5C, 0xB5D), (0xB5F, 0xB61), (0xB66, 0xB6F), (0xB71, 0xB77), (0xB83, 0xB83),
    (0xB85, 0xB8A), (0xB8E, 0xB90), (0xB92, 0xB95), (0xB99, 0xB9A), (0xB9C, 0xB9C),
    (0xB9E, 0xB9F), (0xBA3, 0xBA4), (0xBA8, 0xBAA), (0xBAE, 0xBB9), (0xBD0, 0xBD0),
    (0xBE6, 0xBF2), (0xC05, 0xC0C), (0xC0E, 0xC10), (0xC12, 0xC28), (0xC2A, 0xC39),
    (0xC3D, 0xC3D), (0xC58, 0xC5A), (0xC60, 0xC61), (0xC66, 0xC6F), (0xC78, 0xC7E),
    (0xC80, 0xC80), (0xC85, 0xC8C), (0xC8E, 0xC90), (0xC92, 0xCA8), (0xCAA, 0xCB3),
    (0xCB5, 0xCB9), (0xCBD, 0xCBD), (0xCDE, 0xCDE), (0xCE0, 0xCE1), (0xCE6, 0xCEF),
    (0xCF1, 0xCF2), (0xD04, 0xD0C), (0xD0E, 0xD10), (0xD12, 0xD3A), (0xD3D, 0xD3D),
    (0xD4E, 0xD4E), (0xD54, 0xD56), (0xD58, 0xD61), (0xD66, 0xD78), (0xD7A, 0xD7F),
    (0xD85, 0xD96), (0xD9A, 0xDB1), (0xDB3, 0xDBB), (0xDBD, 0xDBD), (0xDC0, 0xDC6),
    (0xDE6, 0xDEF), (0xE01, 0xE30), (0xE32, 0xE33), (0xE40, 0xE46), (0xE50, 0xE59),
    (0xE81, 0xE82), (0xE84, 0xE84), (0xE86, 0xE8A), (0xE8C, 0xEA3), (0xEA5, 0xEA5),
    (0xEA7, 0xEB0), (0xEB2, 0xEB3), (0xEBD, 0xEBD), (0xEC0, 0xEC4), (0xEC6, 0xEC6),
    (0xED0, 0xED9), (0xEDC, 0xEDF), (0xF00, 0xF00), (0xF20, 0xF33), (0xF40, 0xF47),
    (0xF49, 0xF6C), (0xF88, 0xF8C), (0x1000, 0x102A), (0x103F, 0x1049), (0x1050, 0x1055),
    (0x105A, 0x105D), (0x1061, 0x1061), (0x1065, 0x1066), (0x106E, 0x1070), (0x1075, 0x1081),
    (0x108E, 0x108E), (0x1090, 0x1099), (0x10A0, 0x10C5), (0x10C7, 0x10C7), (0x10CD, 0x10CD),
    (0x10D0, 0x10FA), (0x10FC, 0x1248), (0x124A, 0x124D), (0x1250, 0x1256), (0x1258, 0x1258),
    (0x125A, 0x125D), (0x1260, 0x1288), (0x128A, 0x128D), (0x1290, 0x12B0), (0x12B2, 0x12B5),
    (0x12B8, 0x12BE), (0x12C0, 0x12C0), (0x12C2, 0x12C5), (0x12C8, 0x12D6), (0x12D8, 0x1310),
    (0x1312, 0x1315), (0x1318, 0x135A), (0x1369, 0x137C), (0x1380, 0x138F), (0x13A0, 0x13F5),
    (0x13F8, 0x13FD), (0x1401, 0x166C), (0x166F, 0x167F), (0x1681, 0x169A), (0x16A0, 0x16EA),
    (0x16EE, 0x16F8), (0x1700, 0x170C), (0x170E, 0x1711), (0x1720, 0x1731), (0x1740, 0x1751),
    (0x1760, 0x176C), (0x176E, 0x1770), (0x1780, 0x17B3), (0x17D7, 0x17D7), (0x17DC, 0x17DC),
    (0x17E0, 0x17E9), (0x17F0, 0x17F9), (0x1810, 0x1819), (0x1820, 0x1878), (0x1880, 0x1884),
    (0x1887, 0x18A8), (0x18AA, 0x18AA), (0x18B0, 0x18F5), (0x1900, 0x191E), (0x1946, 0x196D),
    (0x1970, 0x1974), (0x1980, 0x19AB), (0x19B0, 0x19C9), (0x19D0, 0x19DA), (0x1A00, 0x1A16),
    (0x1A20, 0x1A54), (0x1A80, 0x1A89), (0x1A90, 0x1A99), (0x1AA7, 0x1AA7), (0x1B05, 0x1B33),
    (0x1B45, 0x1B4B), (0x1B50, 0x1B59), (0x1B83, 0x1BA0), (0x1BAE, 0x1BE5), (0x1C00, 0x1C23),
    (0x1C40, 0x1C49), (0x1C4D, 0x1C7D), (0x1C80, 0x1C88), (0x1C90, 0x1CBA), (0x1CBD, 0x1CBF),
    (0x1CE9, 0x1CEC), (0x1CEE, 0x1CF3), (0x1CF5, 0x1CF6), (0x1CFA, 0x1CFA), (0x1D00, 0x1DBF),
    (0x1E00, 0x1F15), (0x1F18, 0x1F1D), (0x1F20, 0x1F45), (0x1F48, 0x1F4D), (0x1F50, 0x1F57),
    (0x1F59, 0x1F59), (0x1F5B, 0x1F5B), (0x1F5D, 0x1F5D), (0x1F5F, 0x1F7D), (0x1F80, 0x1FB4),
    (0x1FB6, 0x1FBC), (0x1FBE, 0x1FBE), (0x1FC2, 0x1FC4), (0x1FC6, 0x1FCC), (0x1FD0, 0x1FD3),
    (0x1FD6, 0x1FDB), (0x1FE0, 0x1FEC), (0x1FF2, 0x1FF4), (0x1FF6, 0x1FFC), (0x2070, 0x2071),
    (0x2074, 0x2079), (0x207F, 0x2089), (0x2090, 0x209C), (0x2102, 0x2102), (0x2107, 0x2107),
    (0x210A, 0x2113), (0x2115, 0x2115), (0x2119, 0x211D), (0x2124, 0x2124), (0x2126, 0x2126),
    (0x2128, 0x2128), (0x212A, 0x212D), (0x212F, 0x2139), (0x213C, 0x213F), (0x2145, 0x2149),
    (0x214E, 0x214E), (0x2150, 0x2189), (0x2460, 0x249B), (0x24EA, 0x24FF), (0x2776, 0x2793),
    (0x2C00, 0x2C2E), (0x2C30, 0x2C5E), (0x2C60, 0x2CE4), (0x2CEB, 0x2CEE), (0x2CF2, 0x2CF3),
    (0x2CFD, 0x2CFD), (0x2D00, 0x2D25), (0x2D27, 0x2D27), (0x2D2D, 0x2D2D), (0x2D30, 0x2D67),
    (0x2D6F, 0x2D6F), (0x2D80, 0x2D96), (0x2DA0, 0x2DA6), (0x2DA8, 0x2DAE), (0x2DB0, 0x2DB6),
    (0x2DB8, 0x2DBE), (0x2DC0, 0x2DC6), (0x2DC8, 0x2DCE), (0x2DD0, 0x2DD6), (0x2DD8, 0x2DDE),
    (0x2E2F, 0x2E2F), (0x3005, 0x3007), (0x3021, 0x3029), (0x3031, 0x3035), (0x3038, 0x303C),
    (0x3041, 0x3096), (0x309D, 0x309F), (0x30A1, 0x30FA), (0x30FC, 0x30FF), (0x3105, 0x312F),
    (0x3131, 0x318E), (0x3192, 0x3195), (0x31A0, 0x31BF), (0x31F0, 0x31FF), (0x3220, 0x3229),
    (0x3248, 0x324F), (0x3251, 0x325F), (0x3280, 0x3289), (0x32B1, 0x32BF), (0x3400, 0x4DBF),
    (0x4E00, 0x9FFC), (0xA000, 0xA48C), (0xA4D0, 0xA4FD), (0xA500, 0xA60C), (0xA610, 0xA62B),
    (0xA640, 0xA66E), (0xA67F, 0xA69D), (0xA6A0, 0xA6EF), (0xA717, 0xA71F), (0xA722, 0xA788),
    (0xA78B, 0xA7BF), (0xA7C2, 0xA7CA), (0xA7F5, 0xA801), (0xA803, 0xA805), (0xA807, 0xA80A),
    (0xA80C, 0xA822), (0xA830, 0xA835), (0xA840, 0xA873), (0xA882, 0xA8B3), (0xA8D0, 0xA8D9),
    (0xA8F2, 0xA8F7), (0xA8FB, 0xA8FB), (0xA8FD, 0xA8FE), (0xA900, 0xA925), (0xA930, 0xA946),
    (0xA960, 0xA97C), (0xA984, 0xA9B2), (0xA9CF, 0xA9D9), (0xA9E0, 0xA9E4), (0xA9E6, 0xA9FE),
    (0xAA00, 0xAA28), (0xAA40, 0xAA42), (0xAA44, 0xAA4B), (0xAA50, 0xAA59), (0xAA60, 0xAA76),
    (0xAA7A, 0xAA7A), (0xAA7E, 0xAAAF), (0xAAB1, 0xAAB1), (0xAAB5, 0xAAB6), (0xAAB9, 0xAABD),
    (0xAAC0, 0xAAC0), (0xAAC2, 0xAAC2), (0xAADB, 0xAADD), (0xAAE0, 0xAAEA), (0xAAF2, 0xAAF4),
    (0xAB01, 0xAB06), (0xAB09, 0xAB0E), (0xAB11, 0xAB16), (0xAB20, 0xAB26), (0xAB28, 0xAB2E),
    (0xAB30, 0xAB5A), (0xAB5C, 0xAB69), (0xAB70, 0xABE2), (0xABF0, 0xABF9), (0xAC00, 0xD7A3),
    (0xD7B0, 0xD7C6), (0xD7CB, 0xD7FB), (0xF900, 0xFA6D), (0xFA70, 0xFAD9), (0xFB00, 0xFB06),
    (0xFB13, 0xFB17), (0xFB1D, 0xFB1D), (0xFB1F, 0xFB28), (0xFB2A, 0xFB36), (0xFB38, 0xFB3C),
    (0xFB3E, 0xFB3E), (0xFB40, 0xFB41), (0xFB43, 0xFB44), (0xFB46, 0xFBB1), (0xFBD3, 0xFD3D),
    (0xFD50, 0xFD8F), (0xFD92, 0xFDC7), (0xFDF0, 0xFDFB), (0xFE70, 0xFE74), (0xFE76, 0xFEFC),
    (0xFF10, 0xFF19), (0xFF21, 0xFF3A), (0xFF41, 0xFF5A), (0xFF66, 0xFFBE), (0xFFC2, 0xFFC7),
    (0xFFCA, 0xFFCF), (0xFFD2, 0xFFD7), (0xFFDA, 0xFFDC), (0x10000, 0x1000B),
    (0x1000D, 0x10026), (0x10028, 0x1003A), (0x1003C, 0x1003D), (0x1003F, 0x1004D),
    (0x10050, 0x1005D), (0x10080, 0x100FA), (0x10107, 0x10133), (0x10140, 0x10178),
    (0x1018A, 0x1018B), (0x10280, 0x1029C), (0x102A0, 0x102D0), (0x102E1, 0x102FB),
    (0x10300, 0x10323), (0x1032D, 0x1034A), (0x10350, 0x10375), (0x10380, 0x1039D),
    (0x103A0, 0x103C3), (0x103C8, 0x103CF), (0x103D1, 0x103D5), (0x10400, 0x1049D),
    (0x104A0, 0x104A9), (0x104B0, 0x104D3), (0x104D8, 0x104FB), (0x10500, 0x10527),
    (0x10530, 0x10563), (0x10600, 0x10736), (0x10740, 0x10755), (0x10760, 0x10767),
    (0x10800, 0x10805), (0x10808, 0x10808), (0x1080A, 0x10835), (0x10837, 0x10838),
    (0x1083C, 0x1083C), (0x1083F, 0x10855), (0x10858, 0x10876), (0x10879, 0x1089E),
    (0x108A7, 0x108AF), (0x108E0, 0x108F2), (0x108F4, 0x108F5), (0x108FB, 0x1091B),
    (0x10920, 0x10939), (0x10980, 0x109B7), (0x109BC, 0x109CF), (0x109D2, 0x10A00),
    (0x10A10, 0x10A13), (0x10A15, 0x10A17), (0x10A19, 0x10A35), (0x10A40, 0x10A48),
    (0x10A60, 0x10A7E), (0x10A80, 0x10A9F), (0x10AC0, 0x10AC7), (0x10AC9, 0x10AE4),
    (0x10AEB, 0x10AEF), (0x10B00, 0x10B35), (0x10B40, 0x10B55), (0x10B58, 0x10B72),
    (0x10B78, 0x10B91), (0x10BA9, 0x10BAF), (0x10C00, 0x10C48), (0x10C80, 0x10CB2),
    (0x10CC0, 0x10CF2), (0x10CFA, 0x10D23), (0x10D30, 0x10D39), (0x10E60, 0x10E7E),
    (0x10E80, 0x10EA9), (0x10EB0, 0x10EB1), (0x10F00, 0x10F27), (0x10F30, 0x10F45),
    (0x10F51, 0x10F54), (0x10FB0, 0x10FCB), (0x10FE0, 0x10FF6), (0x11003, 0x11037),
    (0x11052, 0x1106F), (0x11083, 0x110AF), (0x110D0, 0x110E8), (0x110F0, 0x110F9),
    (0x11103, 0x11126), (0x11136, 0x1113F), (0x11144, 0x11144), (0x11147, 0x11147),
    (0x11150, 0x11172), (0x11176, 0x11176), (0x11183, 0x111B2), (0x111C1, 0x111C4),
    (0x111D0, 0x111DA), (0x111DC, 0x111DC), (0x111E1, 0x111F4), (0x11200, 0x11211),
    (0x11213, 0x1122B), (0x11280, 0x11286), (0x11288, 0x11288), (0x1128A, 0x1128D),
    (0x1128F, 0x1129D), (0x1129F, 0x112A8), (0x112B0, 0x112DE), (0x112F0, 0x112F9),
    (0x11305, 0x1130C), (0x1130F, 0x11310), (0x11313, 0x11328), (0x1132A, 0x11330),
    (0x11332, 0x11333), (0x11335, 0x11339), (0x1133D, 0x1133D), (0x11350, 0x11350),
    (0x1135D, 0x11361), (0x11400, 0x11434), (0x11447, 0x1144A), (0x11450, 0x11459),
    (0x1145F, 0x11461), (0x11480, 0x114AF), (0x114C4, 0x114C5), (0x114C7, 0x114C7),
    (0x114D0, 0x114D9), (0x11580, 0x115AE), (0x115D8, 0x115DB), (0x11600, 0x1162F),
    (0x11644, 0x11644), (0x11650, 0x11659), (0x11680, 0x116AA), (0x116B8, 0x116B8),
    (0x116C0, 0x116C9), (0x11700, 0x1171A), (0x11730, 0x1173B), (0x11800, 0x1182B),
    (0x118A0, 0x118F2), (0x118FF, 0x11906), (0x11909, 0x11909), (0x1190C, 0x11913),
    (0x11915, 0x11916), (0x11918, 0x1192F), (0x1193F, 0x1193F), (0x11941, 0x11941),
    (0x11950, 0x11959), (0x119A0, 0x119A7), (0x119AA, 0x119D0), (0x119E1, 0x119E1),
    (0x119E3, 0x119E3), (0x11A00, 0x11A00), (0x11A0B, 0x11A32), (0x11A3A, 0x11A3A),
    (0x11A50, 0x11A50), (0x11A5C, 0x11A89), (0x11A9D, 0x11A9D), (0x11AC0, 0x11AF8),
    (0x11C00, 0x11C08), (0x11C0A, 0x11C2E), (0x11C40, 0x11C40), (0x11C50, 0x11C6C),
    (0x11C72, 0x11C8F), (0x11D00, 0x11D06), (0x11D08, 0x11D09), (0x11D0B, 0x11D30),
    (0x11D46, 0x11D46), (0x11D50, 0x11D59), (0x11D60, 0x11D65), (0x11D67, 0x11D68),
    (0x11D6A, 0x11D89), (0x11D98, 0x11D98), (0x11DA0, 0x11DA9), (0x11EE0, 0x11EF2),
    (0x11FB0, 0x11FB0), (0x11FC0, 0x11FD4), (0x12000, 0x12399), (0x12400, 0x1246E),
    (0x12480, 0x12543), (0x13000, 0x1342E), (0x14400, 0x14646), (0x16800, 0x16A38),
    (0x16A40, 0x16A5E), (0x16A60, 0x16A69), (0x16AD0, 0x16AED), (0x16B00, 0x16B2F),
    (0x16B40, 0x16B43), (0x16B50, 0x16B59), (0x16B5B, 0x16B61), (0x16B63, 0x16B77),
    (0x16B7D, 0x16B8F), (0x16E40, 0x16E96), (0x16F00, 0x16F4A), (0x16F50, 0x16F50),
    (0x16F93, 0x16F9F), (0x16FE0, 0x16FE1), (0x16FE3, 0x16FE3), (0x17000, 0x187F7),
    (0x18800, 0x18CD5), (0x18D00, 0x18D08), (0x1B000, 0x1B11E), (0x1B150, 0x1B152),
    (0x1B164, 0x1B167), (0x1B170, 0x1B2FB), (0x1BC00, 0x1BC6A), (0x1BC70, 0x1BC7C),
    (0x1BC80, 0x1BC88), (0x1BC90, 0x1BC99), (0x1D2E0, 0x1D2F3), (0x1D360, 0x1D378),
    (0x1D400, 0x1D454), (0x1D456, 0x1D49C), (0x1D49E, 0x1D49F), (0x1D4A2, 0x1D4A2),
    (0x1D4A5, 0x1D4A6), (0x1D4A9, 0x1D4AC), (0x1D4AE, 0x1D4B9), (0x1D4BB, 0x1D4BB),
    (0x1D4BD, 0x1D4C3), (0x1D4C5, 0x1D505), (0x1D507, 0x1D50A), (0x1D50D, 0x1D514),
    (0x1D516, 0x1D51C), (0x1D51E, 0x1D539), (0x1D53B, 0x1D53E), (0x1D540, 0x1D544),
    (0x1D546, 0x1D546), (0x1D54A, 0x1D550), (0x1D552, 0x1D6A5), (0x1D6A8, 0x1D6C0),
    (0x1D6C2, 0x1D6DA), (0x1D6DC, 0x1D6FA), (0x1D6FC, 0x1D714), (0x1D716, 0x1D734),
    (0x1D736, 0x1D74E), (0x1D750, 0x1D76E), (0x1D770, 0x1D788), (0x1D78A, 0x1D7A8),
    (0x1D7AA, 0x1D7C2), (0x1D7C4, 0x1D7CB), (0x1D7CE, 0x1D7FF), (0x1E100, 0x1E12C),
    (0x1E137, 0x1E13D), (0x1E140, 0x1E149), (0x1E14E, 0x1E14E), (0x1E2C0, 0x1E2EB),
    (0x1E2F0, 0x1E2F9), (0x1E800, 0x1E8C4), (0x1E8C7, 0x1E8CF), (0x1E900, 0x1E943),
    (0x1E94B, 0x1E94B), (0x1E950, 0x1E959), (0x1EC71, 0x1ECAB), (0x1ECAD, 0x1ECAF),
    (0x1ECB1, 0x1ECB4), (0x1ED01, 0x1ED2D), (0x1ED2F, 0x1ED3D), (0x1EE00, 0x1EE03),
    (0x1EE05, 0x1EE1F), (0x1EE21, 0x1EE22), (0x1EE24, 0x1EE24), (0x1EE27, 0x1EE27),
    (0x1EE29, 0x1EE32), (0x1EE34, 0x1EE37), (0x1EE39, 0x1EE39), (0x1EE3B, 0x1EE3B),
    (0x1EE42, 0x1EE42), (0x1EE47, 0x1EE47), (0x1EE49, 0x1EE49), (0x1EE4B, 0x1EE4B),
    (0x1EE4D, 0x1EE4F), (0x1EE51, 0x1EE52), (0x1EE54, 0x1EE54), (0x1EE57, 0x1EE57),
    (0x1EE59, 0x1EE59), (0x1EE5B, 0x1EE5B), (0x1EE5D, 0x1EE5D), (0x1EE5F, 0x1EE5F),
    (0x1EE61, 0x1EE62), (0x1EE64, 0x1EE64), (0x1EE67, 0x1EE6A), (0x1EE6C, 0x1EE72),
    (0x1EE74, 0x1EE77), (0x1EE79, 0x1EE7C), (0x1EE7E, 0x1EE7E), (0x1EE80, 0x1EE89),
    (0x1EE8B, 0x1EE9B), (0x1EEA1, 0x1EEA3), (0x1EEA5, 0x1EEA9), (0x1EEAB, 0x1EEBB),
    (0x1F100, 0x1F10C), (0x1FBF0, 0x1FBF9), (0x20000, 0x2A6DD), (0x2A700, 0x2B734),
    (0x2B740, 0x2B81D), (0x2B820, 0x2CEA1), (0x2CEB0, 0x2EBE0), (0x2F800, 0x2FA1D),
    (0x30000, 0x3134A),
];

/// Python str.isdecimal() code points (Unicode Nd), as ranges.
const PY_DECIMAL_RANGES: [(u32, u32); 61] = [
    (0x30, 0x39), (0x660, 0x669), (0x6F0, 0x6F9), (0x7C0, 0x7C9), (0x966, 0x96F),
    (0x9E6, 0x9EF), (0xA66, 0xA6F), (0xAE6, 0xAEF), (0xB66, 0xB6F), (0xBE6, 0xBEF),
    (0xC66, 0xC6F), (0xCE6, 0xCEF), (0xD66, 0xD6F), (0xDE6, 0xDEF), (0xE50, 0xE59),
    (0xED0, 0xED9), (0xF20, 0xF29), (0x1040, 0x1049), (0x1090, 0x1099), (0x17E0, 0x17E9),
    (0x1810, 0x1819), (0x1946, 0x194F), (0x19D0, 0x19D9), (0x1A80, 0x1A89), (0x1A90, 0x1A99),
    (0x1B50, 0x1B59), (0x1BB0, 0x1BB9), (0x1C40, 0x1C49), (0x1C50, 0x1C59), (0xA620, 0xA629),
    (0xA8D0, 0xA8D9), (0xA900, 0xA909), (0xA9D0, 0xA9D9), (0xA9F0, 0xA9F9), (0xAA50, 0xAA59),
    (0xABF0, 0xABF9), (0xFF10, 0xFF19), (0x104A0, 0x104A9), (0x10D30, 0x10D39),
    (0x11066, 0x1106F), (0x110F0, 0x110F9), (0x11136, 0x1113F), (0x111D0, 0x111D9),
    (0x112F0, 0x112F9), (0x11450, 0x11459), (0x114D0, 0x114D9), (0x11650, 0x11659),
    (0x116C0, 0x116C9), (0x11730, 0x11739), (0x118E0, 0x118E9), (0x11950, 0x11959),
    (0x11C50, 0x11C59), (0x11D50, 0x11D59), (0x11DA0, 0x11DA9), (0x16A60, 0x16A69),
    (0x16B50, 0x16B59), (0x1D7CE, 0x1D7FF), (0x1E140, 0x1E149), (0x1E2F0, 0x1E2F9),
    (0x1E950, 0x1E959), (0x1FBF0, 0x1FBF9),
];

/// Canonical composition pairs (base, combining mark) -> composed, from
/// unicodedata.decomposition, restricted to the pairs that can touch the
/// counted Latin ranges. Sorted by (base, mark) for binary search.
const NFC_COMPOSE: [(u32, u32, u32); 497] = [
    (0x41, 0x300, 0xC0), (0x41, 0x301, 0xC1), (0x41, 0x302, 0xC2), (0x41, 0x303, 0xC3),
    (0x41, 0x304, 0x100), (0x41, 0x306, 0x102), (0x41, 0x307, 0x226), (0x41, 0x308, 0xC4),
    (0x41, 0x309, 0x1EA2), (0x41, 0x30A, 0xC5), (0x41, 0x30C, 0x1CD), (0x41, 0x30F, 0x200),
    (0x41, 0x311, 0x202), (0x41, 0x323, 0x1EA0), (0x41, 0x325, 0x1E00), (0x41, 0x328, 0x104),
    (0x42, 0x307, 0x1E02), (0x42, 0x323, 0x1E04), (0x42, 0x331, 0x1E06), (0x43, 0x301, 0x106),
    (0x43, 0x302, 0x108), (0x43, 0x307, 0x10A), (0x43, 0x30C, 0x10C), (0x43, 0x327, 0xC7),
    (0x44, 0x307, 0x1E0A), (0x44, 0x30C, 0x10E), (0x44, 0x323, 0x1E0C), (0x44, 0x327, 0x1E10),
    (0x44, 0x32D, 0x1E12), (0x44, 0x331, 0x1E0E), (0x45, 0x300, 0xC8), (0x45, 0x301, 0xC9),
    (0x45, 0x302, 0xCA), (0x45, 0x303, 0x1EBC), (0x45, 0x304, 0x112), (0x45, 0x306, 0x114),
    (0x45, 0x307, 0x116), (0x45, 0x308, 0xCB), (0x45, 0x309, 0x1EBA), (0x45, 0x30C, 0x11A),
    (0x45, 0x30F, 0x204), (0x45, 0x311, 0x206), (0x45, 0x323, 0x1EB8), (0x45, 0x327, 0x228),
    (0x45, 0x328, 0x118), (0x45, 0x32D, 0x1E18), (0x45, 0x330, 0x1E1A), (0x46, 0x307, 0x1E1E),
    (0x47, 0x301, 0x1F4), (0x47, 0x302, 0x11C), (0x47, 0x304, 0x1E20), (0x47, 0x306, 0x11E),
    (0x47, 0x307, 0x120), (0x47, 0x30C, 0x1E6), (0x47, 0x327, 0x122), (0x48, 0x302, 0x124),
    (0x48, 0x307, 0x1E22), (0x48, 0x308, 0x1E26), (0x48, 0x30C, 0x21E), (0x48, 0x323, 0x1E24),
    (0x48, 0x327, 0x1E28), (0x48, 0x32E, 0x1E2A), (0x49, 0x300, 0xCC), (0x49, 0x301, 0xCD),
    (0x49, 0x302, 0xCE), (0x49, 0x303, 0x128), (0x49, 0x304, 0x12A), (0x49, 0x306, 0x12C),
    (0x49, 0x307, 0x130), (0x49, 0x308, 0xCF), (0x49, 0x309, 0x1EC8), (0x49, 0x30C, 0x1CF),
    (0x49, 0x30F, 0x208), (0x49, 0x311, 0x20A), (0x49, 0x323, 0x1ECA), (0x49, 0x328, 0x12E),
    (0x49, 0x330, 0x1E2C), (0x4A, 0x302, 0x134), (0x4B, 0x301, 0x1E30), (0x4B, 0x30C, 0x1E8),
    (0x4B, 0x323, 0x1E32), (0x4B, 0x327, 0x136), (0x4B, 0x331, 0x1E34), (0x4C, 0x301, 0x139),
    (0x4C, 0x30C, 0x13D), (0x4C, 0x323, 0x1E36), (0x4C, 0x327, 0x13B), (0x4C, 0x32D, 0x1E3C),
    (0x4C, 0x331, 0x1E3A), (0x4D, 0x301, 0x1E3E), (0x4D, 0x307, 0x1E40), (0x4D, 0x323, 0x1E42),
    (0x4E, 0x300, 0x1F8), (0x4E, 0x301, 0x143), (0x4E, 0x303, 0xD1), (0x4E, 0x307, 0x1E44),
    (0x4E, 0x30C, 0x147), (0x4E, 0x323, 0x1E46), (0x4E, 0x327, 0x145), (0x4E, 0x32D, 0x1E4A),
    (0x4E, 0x331, 0x1E48), (0x4F, 0x300, 0xD2), (0x4F, 0x301, 0xD3), (0x4F, 0x302, 0xD4),
    (0x4F, 0x303, 0xD5), (0x4F, 0x304, 0x14C), (0x4F, 0x306, 0x14E), (0x4F, 0x307, 0x22E),
    (0x4F, 0x308, 0xD6), (0x4F, 0x309, 0x1ECE), (0x4F, 0x30B, 0x150), (0x4F, 0x30C, 0x1D1),
    (0x4F, 0x30F, 0x20C), (0x4F, 0x311, 0x20E), (0x4F, 0x31B, 0x1A0), (0x4F, 0x323, 0x1ECC),
    (0x4F, 0x328, 0x1EA), (0x50, 0x301, 0x1E54), (0x50, 0x307, 0x1E56), (0x52, 0x301, 0x154),
    (0x52, 0x307, 0x1E58), (0x52, 0x30C, 0x158), (0x52, 0x30F, 0x210), (0x52, 0x311, 0x212),
    (0x52, 0x323, 0x1E5A), (0x52, 0x327, 0x156), (0x52, 0x331, 0x1E5E), (0x53, 0x301, 0x15A),
    (0x53, 0x302, 0x15C), (0x53, 0x307, 0x1E60), (0x53, 0x30C, 0x160), (0x53, 0x323, 0x1E62),
    (0x53, 0x326, 0x218), (0x53, 0x327, 0x15E), (0x54, 0x307, 0x1E6A), (0x54, 0x30C, 0x164),
    (0x54, 0x323, 0x1E6C), (0x54, 0x326, 0x21A), (0x54, 0x327, 0x162), (0x54, 0x32D, 0x1E70),
    (0x54, 0x331, 0x1E6E), (0x55, 0x300, 0xD9), (0x55, 0x301, 0xDA), (0x55, 0x302, 0xDB),
    (0x55, 0x303, 0x168), (0x55, 0x304, 0x16A), (0x55, 0x306, 0x16C), (0x55, 0x308, 0xDC),
    (0x55, 0x309, 0x1EE6), (0x55, 0x30A, 0x16E), (0x55, 0x30B, 0x170), (0x55, 0x30C, 0x1D3),
    (0x55, 0x30F, 0x214), (0x55, 0x311, 0x216), (0x55, 0x31B, 0x1AF), (0x55, 0x323, 0x1EE4),
    (0x55, 0x324, 0x1E72), (0x55, 0x328, 0x172), (0x55, 0x32D, 0x1E76), (0x55, 0x330, 0x1E74),
    (0x56, 0x303, 0x1E7C), (0x56, 0x323, 0x1E7E), (0x57, 0x300, 0x1E80), (0x57, 0x301, 0x1E82),
    (0x57, 0x302, 0x174), (0x57, 0x307, 0x1E86), (0x57, 0x308, 0x1E84), (0x57, 0x323, 0x1E88),
    (0x58, 0x307, 0x1E8A), (0x58, 0x308, 0x1E8C), (0x59, 0x300, 0x1EF2), (0x59, 0x301, 0xDD),
    (0x59, 0x302, 0x176), (0x59, 0x303, 0x1EF8), (0x59, 0x304, 0x232), (0x59, 0x307, 0x1E8E),
    (0x59, 0x308, 0x178), (0x59, 0x309, 0x1EF6), (0x59, 0x323, 0x1EF4), (0x5A, 0x301, 0x179),
    (0x5A, 0x302, 0x1E90), (0x5A, 0x307, 0x17B), (0x5A, 0x30C, 0x17D), (0x5A, 0x323, 0x1E92),
    (0x5A, 0x331, 0x1E94), (0x61, 0x300, 0xE0), (0x61, 0x301, 0xE1), (0x61, 0x302, 0xE2),
    (0x61, 0x303, 0xE3), (0x61, 0x304, 0x101), (0x61, 0x306, 0x103), (0x61, 0x307, 0x227),
    (0x61, 0x308, 0xE4), (0x61, 0x309, 0x1EA3), (0x61, 0x30A, 0xE5), (0x61, 0x30C, 0x1CE),
    (0x61, 0x30F, 0x201), (0x61, 0x311, 0x203), (0x61, 0x323, 0x1EA1), (0x61, 0x325, 0x1E01),
    (0x61, 0x328, 0x105), (0x62, 0x307, 0x1E03), (0x62, 0x323, 0x1E05), (0x62, 0x331, 0x1E07),
    (0x63, 0x301, 0x107), (0x63, 0x302, 0x109), (0x63, 0x307, 0x10B), (0x63, 0x30C, 0x10D),
    (0x63, 0x327, 0xE7), (0x64, 0x307, 0x1E0B), (0x64, 0x30C, 0x10F), (0x64, 0x323, 0x1E0D),
    (0x64, 0x327, 0x1E11), (0x64, 0x32D, 0x1E13), (0x64, 0x331, 0x1E0F), (0x65, 0x300, 0xE8),
    (0x65, 0x301, 0xE9), (0x65, 0x302, 0xEA), (0x65, 0x303, 0x1EBD), (0x65, 0x304, 0x113),
    (0x65, 0x306, 0x115), (0x65, 0x307, 0x117), (0x65, 0x308, 0xEB), (0x65, 0x309, 0x1EBB),
    (0x65, 0x30C, 0x11B), (0x65, 0x30F, 0x205), (0x65, 0x311, 0x207), (0x65, 0x323, 0x1EB9),
    (0x65, 0x327, 0x229), (0x65, 0x328, 0x119), (0x65, 0x32D, 0x1E19), (0x65, 0x330, 0x1E1B),
    (0x66, 0x307, 0x1E1F), (0x67, 0x301, 0x1F5), (0x67, 0x302, 0x11D), (0x67, 0x304, 0x1E21),
    (0x67, 0x306, 0x11F), (0x67, 0x307, 0x121), (0x67, 0x30C, 0x1E7), (0x67, 0x327, 0x123),
    (0x68, 0x302, 0x125), (0x68, 0x307, 0x1E23), (0x68, 0x308, 0x1E27), (0x68, 0x30C, 0x21F),
    (0x68, 0x323, 0x1E25), (0x68, 0x327, 0x1E29), (0x68, 0x32E, 0x1E2B), (0x68, 0x331, 0x1E96),
    (0x69, 0x300, 0xEC), (0x69, 0x301, 0xED), (0x69, 0x302, 0xEE), (0x69, 0x303, 0x129),
    (0x69, 0x304, 0x12B), (0x69, 0x306, 0x12D), (0x69, 0x308, 0xEF), (0x69, 0x309, 0x1EC9),
    (0x69, 0x30C, 0x1D0), (0x69, 0x30F, 0x209), (0x69, 0x311, 0x20B), (0x69, 0x323, 0x1ECB),
    (0x69, 0x328, 0x12F), (0x69, 0x330, 0x1E2D), (0x6A, 0x302, 0x135), (0x6A, 0x30C, 0x1F0),
    (0x6B, 0x301, 0x1E31), (0x6B, 0x30C, 0x1E9), (0x6B, 0x323, 0x1E33), (0x6B, 0x327, 0x137),
    (0x6B, 0x331, 0x1E35), (0x6C, 0x301, 0x13A), (0x6C, 0x30C, 0x13E), (0x6C, 0x323, 0x1E37),
    (0x6C, 0x327, 0x13C), (0x6C, 0x32D, 0x1E3D), (0x6C, 0x331, 0x1E3B), (0x6D, 0x301, 0x1E3F),
    (0x6D, 0x307, 0x1E41), (0x6D, 0x323, 0x1E43), (0x6E, 0x300, 0x1F9), (0x6E, 0x301, 0x144),
    (0x6E, 0x303, 0xF1), (0x6E, 0x307, 0x1E45), (0x6E, 0x30C, 0x148), (0x6E, 0x323, 0x1E47),
    (0x6E, 0x327, 0x146), (0x6E, 0x32D, 0x1E4B), (0x6E, 0x331, 0x1E49), (0x6F, 0x300, 0xF2),
    (0x6F, 0x301, 0xF3), (0x6F, 0x302, 0xF4), (0x6F, 0x303, 0xF5), (0x6F, 0x304, 0x14D),
    (0x6F, 0x306, 0x14F), (0x6F, 0x307, 0x22F), (0x6F, 0x308, 0xF6), (0x6F, 0x309, 0x1ECF),
    (0x6F, 0x30B, 0x151), (0x6F, 0x30C, 0x1D2), (0x6F, 0x30F, 0x20D), (0x6F, 0x311, 0x20F),
    (0x6F, 0x31B, 0x1A1), (0x6F, 0x323, 0x1ECD), (0x6F, 0x328, 0x1EB), (0x70, 0x301, 0x1E55),
    (0x70, 0x307, 0x1E57), (0x72, 0x301, 0x155), (0x72, 0x307, 0x1E59), (0x72, 0x30C, 0x159),
    (0x72, 0x30F, 0x211), (0x72, 0x311, 0x213), (0x72, 0x323, 0x1E5B), (0x72, 0x327, 0x157),
    (0x72, 0x331, 0x1E5F), (0x73, 0x301, 0x15B), (0x73, 0x302, 0x15D), (0x73, 0x307, 0x1E61),
    (0x73, 0x30C, 0x161), (0x73, 0x323, 0x1E63), (0x73, 0x326, 0x219), (0x73, 0x327, 0x15F),
    (0x74, 0x307, 0x1E6B), (0x74, 0x308, 0x1E97), (0x74, 0x30C, 0x165), (0x74, 0x323, 0x1E6D),
    (0x74, 0x326, 0x21B), (0x74, 0x327, 0x163), (0x74, 0x32D, 0x1E71), (0x74, 0x331, 0x1E6F),
    (0x75, 0x300, 0xF9), (0x75, 0x301, 0xFA), (0x75, 0x302, 0xFB), (0x75, 0x303, 0x169),
    (0x75, 0x304, 0x16B), (0x75, 0x306, 0x16D), (0x75, 0x308, 0xFC), (0x75, 0x309, 0x1EE7),
    (0x75, 0x30A, 0x16F), (0x75, 0x30B, 0x171), (0x75, 0x30C, 0x1D4), (0x75, 0x30F, 0x215),
    (0x75, 0x311, 0x217), (0x75, 0x31B, 0x1B0), (0x75, 0x323, 0x1EE5), (0x75, 0x324, 0x1E73),
    (0x75, 0x328, 0x173), (0x75, 0x32D, 0x1E77), (0x75, 0x330, 0x1E75), (0x76, 0x303, 0x1E7D),
    (0x76, 0x323, 0x1E7F), (0x77, 0x300, 0x1E81), (0x77, 0x301, 0x1E83), (0x77, 0x302, 0x175),
    (0x77, 0x307, 0x1E87), (0x77, 0x308, 0x1E85), (0x77, 0x30A, 0x1E98), (0x77, 0x323, 0x1E89),
    (0x78, 0x307, 0x1E8B), (0x78, 0x308, 0x1E8D), (0x79, 0x300, 0x1EF3), (0x79, 0x301, 0xFD),
    (0x79, 0x302, 0x177), (0x79, 0x303, 0x1EF9), (0x79, 0x304, 0x233), (0x79, 0x307, 0x1E8F),
    (0x79, 0x308, 0xFF), (0x79, 0x309, 0x1EF7), (0x79, 0x30A, 0x1E99), (0x79, 0x323, 0x1EF5),
    (0x7A, 0x301, 0x17A), (0x7A, 0x302, 0x1E91), (0x7A, 0x307, 0x17C), (0x7A, 0x30C, 0x17E),
    (0x7A, 0x323, 0x1E93), (0x7A, 0x331, 0x1E95), (0xC2, 0x300, 0x1EA6), (0xC2, 0x301, 0x1EA4),
    (0xC2, 0x303, 0x1EAA), (0xC2, 0x309, 0x1EA8), (0xC4, 0x304, 0x1DE), (0xC5, 0x301, 0x1FA),
    (0xC6, 0x301, 0x1FC), (0xC6, 0x304, 0x1E2), (0xC7, 0x301, 0x1E08), (0xCA, 0x300, 0x1EC0),
    (0xCA, 0x301, 0x1EBE), (0xCA, 0x303, 0x1EC4), (0xCA, 0x309, 0x1EC2), (0xCF, 0x301, 0x1E2E),
    (0xD4, 0x300, 0x1ED2), (0xD4, 0x301, 0x1ED0), (0xD4, 0x303, 0x1ED6), (0xD4, 0x309, 0x1ED4),
    (0xD5, 0x301, 0x1E4C), (0xD5, 0x304, 0x22C), (0xD5, 0x308, 0x1E4E), (0xD6, 0x304, 0x22A),
    (0xD8, 0x301, 0x1FE), (0xDC, 0x300, 0x1DB), (0xDC, 0x301, 0x1D7), (0xDC, 0x304, 0x1D5),
    (0xDC, 0x30C, 0x1D9), (0xE2, 0x300, 0x1EA7), (0xE2, 0x301, 0x1EA5), (0xE2, 0x303, 0x1EAB),
    (0xE2, 0x309, 0x1EA9), (0xE4, 0x304, 0x1DF), (0xE5, 0x301, 0x1FB), (0xE6, 0x301, 0x1FD),
    (0xE6, 0x304, 0x1E3), (0xE7, 0x301, 0x1E09), (0xEA, 0x300, 0x1EC1), (0xEA, 0x301, 0x1EBF),
    (0xEA, 0x303, 0x1EC5), (0xEA, 0x309, 0x1EC3), (0xEF, 0x301, 0x1E2F), (0xF4, 0x300, 0x1ED3),
    (0xF4, 0x301, 0x1ED1), (0xF4, 0x303, 0x1ED7), (0xF4, 0x309, 0x1ED5), (0xF5, 0x301, 0x1E4D),
    (0xF5, 0x304, 0x22D), (0xF5, 0x308, 0x1E4F), (0xF6, 0x304, 0x22B), (0xF8, 0x301, 0x1FF),
    (0xFC, 0x300, 0x1DC), (0xFC, 0x301, 0x1D8), (0xFC, 0x304, 0x1D6), (0xFC, 0x30C, 0x1DA),
    (0x102, 0x300, 0x1EB0), (0x102, 0x301, 0x1EAE), (0x102, 0x303, 0x1EB4),
    (0x102, 0x309, 0x1EB2), (0x103, 0x300, 0x1EB1), (0x103, 0x301, 0x1EAF),
    (0x103, 0x303, 0x1EB5), (0x103, 0x309, 0x1EB3), (0x112, 0x300, 0x1E14),
    (0x112, 0x301, 0x1E16), (0x113, 0x300, 0x1E15), (0x113, 0x301, 0x1E17),
    (0x14C, 0x300, 0x1E50), (0x14C, 0x301, 0x1E52), (0x14D, 0x300, 0x1E51),
    (0x14D, 0x301, 0x1E53), (0x15A, 0x307, 0x1E64), (0x15B, 0x307, 0x1E65),
    (0x160, 0x307, 0x1E66), (0x161, 0x307, 0x1E67), (0x168, 0x301, 0x1E78),
    (0x169, 0x301, 0x1E79), (0x16A, 0x308, 0x1E7A), (0x16B, 0x308, 0x1E7B),
    (0x17F, 0x307, 0x1E9B), (0x1A0, 0x300, 0x1EDC), (0x1A0, 0x301, 0x1EDA),
    (0x1A0, 0x303, 0x1EE0), (0x1A0, 0x309, 0x1EDE), (0x1A0, 0x323, 0x1EE2),
    (0x1A1, 0x300, 0x1EDD), (0x1A1, 0x301, 0x1EDB), (0x1A1, 0x303, 0x1EE1),
    (0x1A1, 0x309, 0x1EDF), (0x1A1, 0x323, 0x1EE3), (0x1AF, 0x300, 0x1EEA),
    (0x1AF, 0x301, 0x1EE8), (0x1AF, 0x303, 0x1EEE), (0x1AF, 0x309, 0x1EEC),
    (0x1AF, 0x323, 0x1EF0), (0x1B0, 0x300, 0x1EEB), (0x1B0, 0x301, 0x1EE9),
    (0x1B0, 0x303, 0x1EEF), (0x1B0, 0x309, 0x1EED), (0x1B0, 0x323, 0x1EF1),
    (0x1B7, 0x30C, 0x1EE), (0x1EA, 0x304, 0x1EC), (0x1EB, 0x304, 0x1ED), (0x226, 0x304, 0x1E0),
    (0x227, 0x304, 0x1E1), (0x228, 0x306, 0x1E1C), (0x229, 0x306, 0x1E1D),
    (0x22E, 0x304, 0x230), (0x22F, 0x304, 0x231), (0x292, 0x30C, 0x1EF),
    (0x1E36, 0x304, 0x1E38), (0x1E37, 0x304, 0x1E39), (0x1E5A, 0x304, 0x1E5C),
    (0x1E5B, 0x304, 0x1E5D), (0x1E62, 0x307, 0x1E68), (0x1E63, 0x307, 0x1E69),
    (0x1EA0, 0x302, 0x1EAC), (0x1EA0, 0x306, 0x1EB6), (0x1EA1, 0x302, 0x1EAD),
    (0x1EA1, 0x306, 0x1EB7), (0x1EB8, 0x302, 0x1EC6), (0x1EB9, 0x302, 0x1EC7),
    (0x1ECC, 0x302, 0x1ED8), (0x1ECD, 0x302, 0x1ED9),
];

/// The combining marks used by NFC_COMPOSE.
const NFC_MARKS: [(u32, u32); 8] = [
    (0x300, 0x304), (0x306, 0x30C), (0x30F, 0x30F), (0x311, 0x311), (0x31B, 0x31B), (0x323, 0x328), (0x32D, 0x32E), (0x330, 0x331),
];


/// Membership test for a sorted, non-overlapping code point range table.
fn in_code_point_ranges(table: &[(u32, u32)], cp: u32) -> bool {
    table
        .binary_search_by(|&(start, end)| {
            if cp < start {
                std::cmp::Ordering::Greater
            } else if cp > end {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_ok()
}

/// Python \d (Unicode decimal digits, category Nd), from the reference's
/// Unicode tables.
fn py_is_decimal(c: char) -> bool {
    in_code_point_ranges(&PY_DECIMAL_RANGES, c as u32)
}

/// Python str.isalnum() (categories L* plus N* and the Other_Alphabetic
/// marks Python excludes); used by add_period.
fn py_isalnum(c: char) -> bool {
    in_code_point_ranges(&PY_ALNUM_RANGES, c as u32)
}

fn is_han_4e00_9fff(c: char) -> bool {
    ('\u{4E00}'..='\u{9FFF}').contains(&c)
}

/// [一-鿿] (U+4E00..U+9FFF), used by add_period.
fn has_han_4e00_9fff(text: &str) -> bool {
    text.chars().any(is_han_4e00_9fff)
}

/// [\u4e00-\u9fa5], used by chinese_punc_en2cn (note: 9FA5, not 9FFF).
fn has_han_4e00_9fa5(text: &str) -> bool {
    text.chars().any(|c| ('\u{4E00}'..='\u{9FA5}').contains(&c))
}

/// [\u4e00-\u9fff\u3041-\u3096\u309d-\u309f\u30a1-\u30fa\u30fc-\u30ff\uac00-\ud7a3]
/// from get_semantic_length: one semantic unit per Han/kana/Hangul character.
fn is_cjk_semantic(c: char) -> bool {
    matches!(c,
        '\u{4E00}'..='\u{9FFF}'
        | '\u{3041}'..='\u{3096}'
        | '\u{309D}'..='\u{309F}'
        | '\u{30A1}'..='\u{30FA}'
        | '\u{30FC}'..='\u{30FF}'
        | '\u{AC00}'..='\u{D7A3}')
}

/// [A-Za-z\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u024f\u1e00-\u1eff] from
/// get_semantic_length: the letters that make up one Latin "word".
fn is_latin_letter(c: char) -> bool {
    c.is_ascii_alphabetic()
        || matches!(c,
            '\u{00C0}'..='\u{00D6}'
            | '\u{00D8}'..='\u{00F6}'
            | '\u{00F8}'..='\u{024F}'
            | '\u{1E00}'..='\u{1EFF}')
}

fn is_apostrophe(c: char) -> bool {
    c == '\'' || c == '\u{2019}'
}

/// string.punctuation + string.whitespace membership, as used by
/// has_valid_content. Note that CJK punctuation is *not* in this set.
fn is_py_punct_or_ws(c: char) -> bool {
    PY_PUNCTUATION.contains(c) || PY_ASCII_WHITESPACE.contains(c)
}

// ---------------------------------------------------------------------------
// toolkit.py: markdown / text cleanup
// ---------------------------------------------------------------------------

fn strip_markdown_code(text: &str) -> String {
    let double = regex::Regex::new(r"\x60\x60(.+?)\x60\x60").unwrap();
    let single = regex::Regex::new(r"\x60(.+?)\x60").unwrap();
    let text = double.replace_all(text, "${1}").into_owned();
    let text = single.replace_all(&text, "${1}").into_owned();
    text.replace('\x60', "")
}

/// Pre-process text: strip Markdown inline-code markers.
pub fn preprosess_text(text: &str) -> String {
    strip_markdown_code(text)
}

/// frontend/text_segment_cut.py is_chinese: any U+4E00..U+9FFF.
pub fn is_chinese(text: &str) -> bool {
    has_han_4e00_9fff(text)
}

/// Python re.search(r'[a-zA-Z]', text) based language sniffing.
pub fn get_lang_type(text: &str) -> &'static str {
    let has_chinese = is_chinese(text);
    let has_english = text.chars().any(|c| c.is_ascii_alphabetic());
    match (has_chinese, has_english) {
        (true, true) => "mixed",
        (true, false) => "zh",
        (false, true) => "en",
        (false, false) => "other",
    }
}

/// has_valid_content: true when some character is neither ASCII punctuation
/// nor ASCII whitespace.
pub fn has_valid_content(text: &str) -> bool {
    text.chars().any(|c| !is_py_punct_or_ws(c))
}

/// Whitespace runs (\s+) collapsed to one ASCII space.
fn compress_py_whitespace(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_ws = false;
    for c in text.chars() {
        if py_isspace(c) {
            if !in_ws {
                out.push(' ');
                in_ws = true;
            }
        } else {
            out.push(c);
            in_ws = false;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// text_segment_cut.py: semantic length
// ---------------------------------------------------------------------------

/// Canonical composition lookup: (base, combining mark) -> composed.
fn nfc_lookup(base: char, mark: char) -> Option<char> {
    let key = (base as u32, mark as u32);
    NFC_COMPOSE
        .binary_search_by(|&(b, m, _)| (b, m).cmp(&key))
        .ok()
        .and_then(|index| char::from_u32(NFC_COMPOSE[index].2))
}

fn is_combining_mark(c: char) -> bool {
    in_code_point_ranges(&NFC_MARKS, c as u32)
}

/// NFC as used by get_semantic_length.
///
/// Implemented algorithmically: canonical composition of the Latin pairs that
/// can touch the counted ranges (NFC_COMPOSE, generated from
/// unicodedata.decomposition) plus Hangul L+V(+T) syllable composition.
/// Canonical reordering of several combining marks on one starter is not
/// implemented: it cannot change the counted characters for this corpus, and
/// the corpus golden pins the decomposed-Latin and decomposed-Hangul cases.
fn nfc(text: &str) -> std::borrow::Cow<'_, str> {
    let needs = text.chars().any(|c| {
        is_combining_mark(c)
            || ('\u{1100}'..='\u{1112}').contains(&c)
            || ('\u{11A8}'..='\u{11C2}').contains(&c)
    });
    if !needs {
        return std::borrow::Cow::Borrowed(text);
    }
    let mut out: Vec<char> = Vec::with_capacity(text.len());
    for c in text.chars() {
        if let Some(&last) = out.last() {
            // Hangul L jamo + V jamo
            if ('\u{1100}'..='\u{1112}').contains(&last)
                && ('\u{1161}'..='\u{1175}').contains(&c)
            {
                let syllable = 0xAC00
                    + ((last as u32 - 0x1100) * 21 + (c as u32 - 0x1161)) * 28;
                out.pop();
                out.push(char::from_u32(syllable).unwrap());
                continue;
            }
            // Hangul LV syllable + T jamo
            if ('\u{AC00}'..='\u{D7A3}').contains(&last)
                && (last as u32 - 0xAC00) % 28 == 0
                && ('\u{11A8}'..='\u{11C2}').contains(&c)
            {
                out.pop();
                out.push(char::from_u32(last as u32 + (c as u32 - 0x11A7)).unwrap());
                continue;
            }
            if let Some(composed) = nfc_lookup(last, c) {
                out.pop();
                out.push(composed);
                continue;
            }
        }
        out.push(c);
    }
    std::borrow::Cow::Owned(out.into_iter().collect())
}

/// Semantic length: one per CJK character, one per Latin word.
pub fn get_semantic_length(text: &str) -> usize {
    let normalized = nfc(text);
    let chars: Vec<char> = normalized.chars().collect();
    let mut count = 0usize;
    let mut i = 0usize;
    while i < chars.len() {
        if is_cjk_semantic(chars[i]) {
            count += 1;
            i += 1;
            continue;
        }
        if is_latin_letter(chars[i]) {
            i += 1;
            while i < chars.len() && is_latin_letter(chars[i]) {
                i += 1;
            }
            // (?:['’][latin]+)*
            loop {
                if i + 1 < chars.len() && is_apostrophe(chars[i]) && is_latin_letter(chars[i + 1]) {
                    i += 1;
                    while i < chars.len() && is_latin_letter(chars[i]) {
                        i += 1;
                    }
                } else {
                    break;
                }
            }
            count += 1;
            continue;
        }
        i += 1;
    }
    count
}

// ---------------------------------------------------------------------------
// toolkit.py: add_period
// ---------------------------------------------------------------------------

const WEAK_TAIL_PUNCT: &str = "，,、;；";

/// Append a sentence-final period unless the text already ends in punctuation.
/// A weak trailing punctuation mark (，,、;；) is promoted in place.
pub fn add_period(text: &str) -> String {
    let stripped = py_rstrip(text);
    if stripped.is_empty() {
        return ".".to_string();
    }
    let last_char = stripped.chars().last().unwrap();
    if WEAK_TAIL_PUNCT.contains(last_char) {
        let has_chinese = has_han_4e00_9fff(stripped);
        let mut out: String = stripped.chars().collect();
        out.pop();
        out.push(if has_chinese { '。' } else { '.' });
        return out;
    }
    // "not last_char.isalnum()": already punctuation.
    if !py_isalnum(last_char) {
        return stripped.to_string();
    }
    let is_chinese = is_han_4e00_9fff(last_char) || ('\u{3400}'..='\u{4DBF}').contains(&last_char);
    stripped.to_string() + if is_chinese { "。" } else { "." }
}

// ---------------------------------------------------------------------------
// toolkit.py: fallback_replace_non_alnum / chinese_punc_en2cn /
// clean_text_formatting / normalize_text_kit
// ---------------------------------------------------------------------------

/// The whitelist of fallback_replace_non_alnum, spelled out character by
/// character: [^\u4e00-\u9fffA-Za-z0-9，。！？；：、,.!?;:\. \u2019\u2018'"()（）\u201C\u201D].
fn is_fallback_allowed(c: char) -> bool {
    is_han_4e00_9fff(c)
        || c.is_ascii_alphanumeric()
        || matches!(c,
            '，' | '。' | '！' | '？' | '；' | '：' | '、'
            | ',' | '.' | '!' | '?' | ';' | ':' | ' '
            | '\u{2019}' | '\u{2018}' | '\'' | '"'
            | '(' | ')' | '（' | '）' | '\u{201C}' | '\u{201D}')
}

/// Collapse runs of an exact character to one.
fn collapse_char_runs(text: &str, ch: char) -> String {
    let mut out = String::with_capacity(text.len());
    let mut prev: Option<char> = None;
    for c in text.chars() {
        if c == ch && prev == Some(ch) {
            continue;
        }
        out.push(c);
        prev = Some(c);
    }
    out
}

/// Replace every character outside the TTS whitelist with ，, collapse the
/// resulting runs, then strip.
pub fn fallback_replace_non_alnum(text: &str) -> String {
    let mut replaced = String::with_capacity(text.len());
    for c in text.chars() {
        replaced.push(if is_fallback_allowed(c) { c } else { '，' });
    }
    py_strip(&collapse_char_runs(&replaced, '，')).to_string()
}

/// Chinese punctuation as understood by chinese_punc_en2cn
/// (，。！？；：“”‘’（）【】《》——……、, duplicates removed).
const CH_PUNCS: &str = "，。！？；：“”‘’（）【】《》—…、";
const STRONG_PUNCT: &str = "。！？";
const WEAK_PUNCT: &str = "，；、";

fn is_ch_punct(c: char) -> bool {
    CH_PUNCS.contains(c)
}

/// re.sub(r'(?<=\d)\.(?=\d)', 'DOTPLACEHOLDER', text).
fn protect_decimal_points(text: &str, placeholder: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len() + 8);
    let mut i = 0usize;
    while i < chars.len() {
        if chars[i] == '.'
            && i > 0
            && i + 1 < chars.len()
            && py_is_decimal(chars[i - 1])
            && py_is_decimal(chars[i + 1])
        {
            out.push_str(placeholder);
        } else {
            out.push(chars[i]);
        }
        i += 1;
    }
    out
}

/// re.sub(r'(\d)\.(\d)', r'\1##DOT###\2', text) (non-overlapping, left to right).
fn protect_decimal_pairs(text: &str, placeholder: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len() + 8);
    let mut i = 0usize;
    while i < chars.len() {
        if i + 2 < chars.len()
            && py_is_decimal(chars[i])
            && chars[i + 1] == '.'
            && py_is_decimal(chars[i + 2])
        {
            out.push(chars[i]);
            out.push_str(placeholder);
            out.push(chars[i + 2]);
            i += 3;
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

/// re.sub(r'([A-Z])\.([A-Z])', r'\1##DOT###\2', text).
fn protect_acronym_pairs(text: &str, placeholder: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len() + 8);
    let mut i = 0usize;
    while i < chars.len() {
        if i + 2 < chars.len()
            && chars[i].is_ascii_uppercase()
            && chars[i + 1] == '.'
            && chars[i + 2].is_ascii_uppercase()
        {
            out.push(chars[i]);
            out.push_str(placeholder);
            out.push(chars[i + 2]);
            i += 3;
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

/// re.findall(r'https?://[^\s，。！？]+', text).
fn find_urls(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut urls = Vec::new();
    let mut i = 0usize;
    while i < chars.len() {
        let mut consumed = 0usize;
        for scheme in ["https://", "http://"] {
            let n = scheme.chars().count();
            if i + n <= chars.len() && chars[i..i + n].iter().copied().eq(scheme.chars()) {
                consumed = n;
                break;
            }
        }
        if consumed == 0 {
            i += 1;
            continue;
        }
        let mut j = i + consumed;
        while j < chars.len()
            && !py_isspace(chars[j])
            && !matches!(chars[j], '，' | '。' | '！' | '？')
        {
            j += 1;
        }
        if j > i + consumed {
            urls.push(chars[i..j].iter().collect());
            i = j;
        } else {
            i += 1;
        }
    }
    urls
}

/// punc_map substitution: ASCII punctuation to its full-width counterpart.
fn map_ascii_punc(text: &str) -> String {
    let mut out = String::with_capacity(text.len() * 2);
    for c in text.chars() {
        match c {
            ',' => out.push('，'),
            '.' => out.push('。'),
            ';' => out.push('；'),
            ':' => out.push('：'),
            '!' => out.push('！'),
            '?' => out.push('？'),
            '(' => out.push('（'),
            ')' => out.push('）'),
            '[' => out.push('【'),
            ']' => out.push('】'),
            '<' => out.push('《'),
            '>' => out.push('》'),
            '_' => out.push_str("——"),
            other => out.push(other),
        }
    }
    out
}

/// '"' and "'" to their curly counterparts, alternating open/close.
fn alternate_quotes(text: &str) -> String {
    let mut dq = 0usize;
    let mut sq = 0usize;
    let mut out = String::with_capacity(text.len() * 2);
    for c in text.chars() {
        match c {
            '"' => {
                dq += 1;
                out.push(if dq % 2 != 0 { '“' } else { '”' });
            }
            '\'' => {
                sq += 1;
                out.push(if sq % 2 != 0 { '‘' } else { '’' });
            }
            other => out.push(other),
        }
    }
    out
}

fn is_ideographic_space(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\u{00A0}' | '\u{3000}')
}

/// re.sub(r'[ \t\u00A0\u3000]*([ch_puncs])[ \t\u00A0\u3000]*', r'\1', text).
fn strip_spaces_around_ch_punct(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    // Drop runs that immediately precede a CJK punctuation mark.
    let mut pass1 = String::with_capacity(text.len());
    let mut i = 0usize;
    while i < chars.len() {
        if is_ideographic_space(chars[i]) {
            let mut j = i;
            while j < chars.len() && is_ideographic_space(chars[j]) {
                j += 1;
            }
            if j < chars.len() && is_ch_punct(chars[j]) {
                i = j; // drop the run
                continue;
            }
            for &c in &chars[i..j] {
                pass1.push(c);
            }
            i = j;
            continue;
        }
        pass1.push(chars[i]);
        i += 1;
    }
    // Drop runs that immediately follow a CJK punctuation mark.
    let p1: Vec<char> = pass1.chars().collect();
    let mut out = String::with_capacity(pass1.len());
    let mut i = 0usize;
    while i < p1.len() {
        out.push(p1[i]);
        if is_ch_punct(p1[i]) {
            let mut j = i + 1;
            while j < p1.len() && is_ideographic_space(p1[j]) {
                j += 1;
            }
            i = j;
            continue;
        }
        i += 1;
    }
    out
}

/// The reference writes this rule as a plain raw string while its neighbours
/// are f-strings, so the character class is the literal text \x60{ch_puncs}\x60
/// and not the punctuation set. Only these characters are collapsed when
/// repeated; a run of "……" survives. The port keeps the behaviour.
const CH_PUNCS_LITERAL_TYPO: &str = "{ch_puncs}";

/// re.sub(r'([{ch_puncs}])\1+', r'\1', text) -- the pattern of the reference,
/// i.e. a class of the literal characters { c h _ p u n s }.
fn squash_identical_ch_punct(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut prev: Option<char> = None;
    for c in text.chars() {
        if CH_PUNCS_LITERAL_TYPO.contains(c) && prev == Some(c) {
            continue;
        }
        out.push(c);
        prev = Some(c);
    }
    out
}

/// re.sub(r'[，；、]+([。！？])', r'\1', text).
fn drop_weak_before_strong(text: &str) -> String {
    let mut out: Vec<char> = Vec::with_capacity(text.chars().count());
    for c in text.chars() {
        if STRONG_PUNCT.contains(c) {
            while out.last().map_or(false, |p| WEAK_PUNCT.contains(*p)) {
                out.pop();
            }
        }
        out.push(c);
    }
    out.into_iter().collect()
}

/// re.sub(r'([。！？])[，；、]+', r'\1', text).
fn drop_weak_after_strong(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        if WEAK_PUNCT.contains(c) && out.chars().last().map_or(false, |p| STRONG_PUNCT.contains(p)) {
            continue;
        }
        out.push(c);
    }
    out
}

/// re.sub(r'[，；]+[、]|[、][，；]+', '、', text).
fn merge_weak_runs_into_dunhao(text: &str) -> String {
    let mut out: Vec<char> = Vec::with_capacity(text.chars().count());
    for c in text.chars() {
        if c == '、' {
            while out.last().map_or(false, |p| *p == '，' || *p == '；') {
                out.pop();
            }
            out.push('、');
            continue;
        }
        if (c == '，' || c == '；') && out.last() == Some(&'、') {
            continue;
        }
        out.push(c);
    }
    out.into_iter().collect()
}

/// Keep the first character of a run of set characters.
fn squash_runs_to_first(text: &str, set: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        if set.contains(c) && out.chars().last().map_or(false, |p| set.contains(p)) {
            continue;
        }
        out.push(c);
    }
    out
}

/// clean_text_formatting: drop a leading run of "noise" punctuation, strip.
pub fn clean_text_formatting(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    const LEADING: &str = "，。！？；：、….!?-—";
    py_strip(py_lstrip_set(text, LEADING)).to_string()
}

/// Chinese-text punctuation normalisation (the steps and their order are the
/// Python ones; see toolkit.py for the rationale).
pub fn chinese_punc_en2cn(text: &str) -> String {
    if text.is_empty() || !has_han_4e00_9fa5(text) {
        return clean_text_formatting(text);
    }

    // URL placeholder protection.
    let urls = find_urls(text);
    let mut current = text.to_string();
    for (i, url) in urls.iter().enumerate() {
        current = current.replace(url.as_str(), &format!("URLHOLDER{i}"));
    }

    // Protect the decimal points between digits.
    current = protect_decimal_points(&current, "DOTPLACEHOLDER");
    current = current.replace("...", "……");

    // Base symbol conversion.
    current = map_ascii_punc(&current);

    // Restore decimal points.
    current = current.replace("DOTPLACEHOLDER", ".");

    // Quote conversion with open/close counting.
    current = alternate_quotes(&current);

    // Remove spaces around CJK punctuation.
    current = strip_spaces_around_ch_punct(&current);

    // Recursive punctuation cleanup.
    loop {
        let old_text = current.clone();
        current = squash_identical_ch_punct(&current);
        current = drop_weak_before_strong(&current);
        current = drop_weak_after_strong(&current);
        current = merge_weak_runs_into_dunhao(&current);
        current = squash_runs_to_first(&current, STRONG_PUNCT);
        current = squash_runs_to_first(&current, WEAK_PUNCT);
        if old_text == current {
            break;
        }
    }

    // Collapse runs of spaces/tabs.
    let mut squeezed = String::with_capacity(current.len());
    let chars: Vec<char> = current.chars().collect();
    let mut i = 0usize;
    while i < chars.len() {
        if is_ideographic_space(chars[i]) {
            let mut j = i;
            while j < chars.len() && is_ideographic_space(chars[j]) {
                j += 1;
            }
            squeezed.push(' ');
            i = j;
            continue;
        }
        squeezed.push(chars[i]);
        i += 1;
    }
    current = squeezed;

    // Restore URLs.
    for (i, url) in urls.iter().enumerate() {
        current = current.replace(&format!("URLHOLDER{i}"), url.as_str());
    }
    clean_text_formatting(&current)
}

/// Full normalisation pipeline: fallback whitelist, punctuation conversion,
/// whitespace squeeze.
pub fn normalize_text_kit(text: &str) -> String {
    let text = fallback_replace_non_alnum(text);
    let text = chinese_punc_en2cn(&text);
    py_strip(&compress_py_whitespace(&text)).to_string()
}

// ---------------------------------------------------------------------------
// toolkit.py: protect_paralinguistic_tags
// ---------------------------------------------------------------------------

/// Byte spans of every paralinguistic tag occurrence, leftmost-first.
fn find_tag_spans(text: &str) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut pos = 0usize;
    while pos < text.len() {
        let rest = &text[pos..];
        let mut found: Option<usize> = None;
        for tag in PARALINGUISTIC_TAGS.iter() {
            if rest.starts_with(tag) {
                found = Some(tag.len());
                break;
            }
        }
        match found {
            Some(len) => {
                spans.push((pos, pos + len));
                pos += len;
            }
            None => {
                let step = rest.chars().next().map(|c| c.len_utf8()).unwrap_or(1);
                pos += step;
            }
        }
    }
    spans
}

/// Apply normalize to the text around paralinguistic tags only, keeping the
/// tags verbatim.
pub fn protect_paralinguistic_tags<F>(text: &str, normalize: F) -> String
where
    F: Fn(&str) -> String,
{
    if text.is_empty() {
        return String::new();
    }
    let spans = find_tag_spans(text);
    if spans.is_empty() {
        return normalize(text);
    }

    let mut segments: Vec<&str> = Vec::with_capacity(spans.len() + 1);
    let mut last = 0usize;
    for &(start, end) in &spans {
        segments.push(&text[last..start]);
        last = end;
    }
    segments.push(&text[last..]);

    let mut pieces = String::with_capacity(text.len());
    for (index, segment) in segments.iter().enumerate() {
        if !py_strip(segment).is_empty() {
            // Reinstate the boundary whitespace so an English word is not glued
            // onto the neighbouring tag.
            let lead = if segment.chars().next().map_or(false, py_isspace) {
                " "
            } else {
                ""
            };
            let trail = if segment.chars().last().map_or(false, py_isspace) {
                " "
            } else {
                ""
            };
            pieces.push_str(lead);
            let normalized = normalize(segment);
            pieces.push_str(py_strip(&normalized));
            pieces.push_str(trail);
        } else {
            pieces.push_str(segment);
        }
        if index < spans.len() {
            let (start, end) = spans[index];
            pieces.push_str(&text[start..end]);
        }
    }
    pieces
}

// ---------------------------------------------------------------------------
// text_segment_cut.py: fragment cutting
// ---------------------------------------------------------------------------

/// Append one clause to the fragment list, merging when it fits.
pub fn append_text_fragment(
    fragments: &mut Vec<String>,
    new_text: &str,
    max_len: usize,
    min_tail_length: usize,
) {
    let new_text = py_lstrip_set(new_text, "，,:; \t\n\r\x0b\x0c");
    if !has_valid_content(new_text) {
        return;
    }
    if fragments.is_empty() {
        fragments.push(new_text.to_string());
        return;
    }
    let last_fragment = fragments.last().unwrap().clone();
    let last_semantic_len = get_semantic_length(&last_fragment);
    let new_semantic_len = get_semantic_length(new_text);

    if last_semantic_len + new_semantic_len <= max_len {
        // A sentence-final fragment followed by a very short clause starts a
        // new fragment instead of being glued on.
        if (last_fragment.ends_with('。')
            || last_fragment.ends_with('！')
            || last_fragment.ends_with('？'))
            && new_semantic_len < min_tail_length
        {
            fragments.push(new_text.to_string());
        } else {
            let mut separator = "";
            if !last_fragment.ends_with(' ')
                && new_text
                    .chars()
                    .next()
                    .map_or(false, |c| c.is_ascii_alphanumeric())
            {
                separator = " ";
            }
            let last = fragments.len() - 1;
            fragments[last].push_str(separator);
            fragments[last].push_str(new_text);
        }
    } else {
        fragments.push(new_text.to_string());
    }
}

const CLAUSE_PUNCT: &str = "，、；。！？,;!?";

/// re.fullmatch(clause_pattern, part) where
/// clause_pattern = r'([，、；。！？,;!?]|\s+[-•·])'.
fn is_clause_delimiter(part: &str) -> bool {
    let chars: Vec<char> = part.chars().collect();
    if chars.len() == 1 && CLAUSE_PUNCT.contains(chars[0]) {
        return true;
    }
    if chars.len() >= 2 {
        let n = chars.len();
        let tail = chars[n - 1];
        if tail == '-' || tail == '•' || tail == '·' {
            return chars[..n - 1].iter().all(|c| py_isspace(*c));
        }
    }
    false
}

/// re.split(r'([，、；。！？,;!?]|\s+[-•·])', text) — keeps the delimiters.
fn split_clause_delimiters(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut parts: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut i = 0usize;
    while i < chars.len() {
        let c = chars[i];
        if CLAUSE_PUNCT.contains(c) {
            parts.push(std::mem::take(&mut current));
            parts.push(c.to_string());
            i += 1;
            continue;
        }
        if py_isspace(c) {
            let mut j = i;
            while j < chars.len() && py_isspace(chars[j]) {
                j += 1;
            }
            if j < chars.len() && matches!(chars[j], '-' | '•' | '·') {
                parts.push(std::mem::take(&mut current));
                parts.push(chars[i..=j].iter().collect());
                i = j + 1;
                continue;
            }
        }
        current.push(c);
        i += 1;
    }
    parts.push(current);
    parts
}

/// Split an over-long clause along punctuation boundaries. A complete
/// punctuation-free clause is kept whole even when it exceeds max_len.
pub fn split_long_fragment(text_fragment: &str, max_len: usize) -> Vec<String> {
    if get_semantic_length(text_fragment) <= max_len {
        return vec![text_fragment.to_string()];
    }

    let parts = split_clause_delimiters(text_fragment);
    let mut merged_clauses: Vec<String> = Vec::new();
    for part in parts {
        if part.is_empty() {
            continue;
        }
        if !merged_clauses.is_empty() && is_clause_delimiter(&part) {
            merged_clauses.last_mut().unwrap().push_str(&part);
        } else {
            merged_clauses.push(part);
        }
    }

    let mut fragments: Vec<String> = Vec::new();
    let mut current_fragment = String::new();
    for clause in merged_clauses {
        let clause_semantic_len = get_semantic_length(&clause);
        // Over-long sub-clauses are kept whole: never split words or
        // punctuation-free semantic units.
        if clause_semantic_len > max_len {
            if !current_fragment.is_empty() {
                fragments.push(std::mem::take(&mut current_fragment));
            }
            fragments.push(clause);
            continue;
        }
        let current_semantic_len = get_semantic_length(&current_fragment);
        if current_semantic_len + clause_semantic_len <= max_len {
            current_fragment.push_str(&clause);
        } else {
            if !current_fragment.is_empty() {
                fragments.push(std::mem::take(&mut current_fragment));
            }
            current_fragment = clause;
        }
    }
    if !current_fragment.is_empty() {
        fragments.push(current_fragment);
    }
    fragments
}

/// Cut text into fragments of about max_length semantic units.
pub fn cut_text_by_semantic_length(text: &str, max_length: usize) -> Vec<String> {
    cut_text_by_semantic_length_full(text, max_length, 5)
}

/// cut_text_by_semantic_length(text, max_semantic_length, min_tail_length):
/// fragments only. The Python return value also carries calibrated positions;
/// they do not influence the fragments, so the port returns just the strings.
pub fn cut_text_by_semantic_length_full(
    text: &str,
    max_semantic_length: usize,
    min_tail_length: usize,
) -> Vec<String> {
    if !has_valid_content(text) {
        return Vec::new();
    }

    const DOT_PLACEHOLDER: &str = "##DOT##";

    // Protect decimal points and dotted abbreviations.
    let mut processed = protect_decimal_pairs(text, DOT_PLACEHOLDER);
    for _ in 0..3 {
        processed = protect_acronym_pairs(&processed, DOT_PLACEHOLDER);
    }
    let processed = processed.replace('\n', " ").replace("。，", "。");

    if get_semantic_length(&processed) <= max_semantic_length {
        return vec![processed.replace(DOT_PLACEHOLDER, ".")];
    }

    // Only used while processing; it does not affect the fragments.
    let normalized = processed
        .replace('.', "。")
        .replace('!', "！")
        .replace('?', "？")
        .replace(',', "，");

    let normalized_chars: Vec<char> = normalized.chars().collect();

    // Stage 1: split into sentences.
    let mut sentences: Vec<String> = Vec::new();
    let mut sentence_positions: Vec<(usize, usize)> = Vec::new();
    let mut current_sentence = String::new();
    let mut start_idx = 0usize;
    for (i, &ch) in normalized_chars.iter().enumerate() {
        current_sentence.push(ch);
        if matches!(ch, '。' | '！' | '？') {
            let sentence = py_strip(&current_sentence).to_string();
            if !sentence.is_empty() {
                sentences.push(sentence);
                sentence_positions.push((start_idx, i + 1));
            }
            current_sentence.clear();
            start_idx = i + 1;
        }
    }
    if !current_sentence.is_empty() {
        sentences.push(py_strip(&current_sentence).to_string());
        sentence_positions.push((start_idx, normalized_chars.len()));
        let last = sentences.len() - 1;
        if !(sentences[last].ends_with('。')
            || sentences[last].ends_with('！')
            || sentences[last].ends_with('？'))
        {
            sentences[last].push('。');
        }
    }

    // Stage 2/3: clauses, then fragments.
    let mut result_fragments: Vec<String> = Vec::new();
    for (sentence, &(sent_start, _sent_end)) in sentences.iter().zip(sentence_positions.iter()) {
        let sentence_chars: Vec<char> = sentence.chars().collect();
        let mut clauses: Vec<String> = Vec::new();
        let mut clause_positions: Vec<(usize, usize)> = Vec::new();
        let mut current_clause = String::new();
        let mut clause_start = 0usize;

        for (i, &ch) in sentence_chars.iter().enumerate() {
            current_clause.push(ch);
            if matches!(ch, '，' | ';' | '；') {
                let clause = py_strip(&current_clause).to_string();
                if !clause.is_empty() && has_valid_content(&clause) {
                    clauses.push(clause);
                    clause_positions.push((clause_start, i + 1));
                } else if !clause.is_empty() && !clauses.is_empty() {
                    let last = clauses.len() - 1;
                    clauses[last].push_str(&clause);
                    clause_positions[last] = (clause_positions[last].0, i + 1);
                }
                current_clause.clear();
                clause_start = i + 1;
            }
        }
        if !current_clause.is_empty() {
            let clause = py_strip(&current_clause).to_string();
            if !clause.is_empty() && has_valid_content(&clause) {
                clauses.push(clause);
                clause_positions.push((clause_start, sentence_chars.len()));
            } else if !clause.is_empty() && !clauses.is_empty() {
                let last = clauses.len() - 1;
                clauses[last].push_str(&clause);
                clause_positions[last] = (clause_positions[last].0, sentence_chars.len());
            }
        }

        let mut i = 0usize;
        while i < clauses.len() {
            let clause = clauses[i].clone();
            let (clause_start_in_sent, _clause_end_in_sent) = clause_positions[i];
            let _abs_start = sent_start + clause_start_in_sent;

            let clause_semantic_len = get_semantic_length(&clause);

            // Short clauses are merged with the next one when it fits.
            if clause_semantic_len < min_tail_length && i + 1 < clauses.len() {
                let next_clause = clauses[i + 1].clone();
                let combined_clause = format!("{clause}{next_clause}");
                let combined_semantic_len = get_semantic_length(&combined_clause);
                if combined_semantic_len <= max_semantic_length {
                    append_text_fragment(
                        &mut result_fragments,
                        &combined_clause,
                        max_semantic_length,
                        min_tail_length,
                    );
                    i += 2;
                    continue;
                }
            }

            if clause_semantic_len > max_semantic_length {
                let sub_fragments = split_long_fragment(&clause, max_semantic_length);
                for frag in sub_fragments {
                    append_text_fragment(
                        &mut result_fragments,
                        &frag,
                        max_semantic_length,
                        min_tail_length,
                    );
                }
            } else {
                append_text_fragment(
                    &mut result_fragments,
                    &clause,
                    max_semantic_length,
                    min_tail_length,
                );
            }
            i += 1;
        }
    }

    result_fragments
        .into_iter()
        .map(|frag| frag.replace(DOT_PLACEHOLDER, "."))
        .collect()
}

// ---------------------------------------------------------------------------
// Tests: assert the port against goldens/m5_text/text.json (Python reference)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn load(name: &str) -> Value {
        let dir = std::env::var("MING_GOLDEN").unwrap_or_else(|_| {
            concat!(env!("CARGO_MANIFEST_DIR"), "/goldens").to_string()
        });
        let path = std::path::Path::new(&dir).join("m5_text").join(name);
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
        serde_json::from_str(&text)
            .unwrap_or_else(|error| panic!("cannot parse {}: {error}", path.display()))
    }

    fn strings(value: &Value) -> Vec<String> {
        value
            .as_array()
            .unwrap_or_else(|| panic!("expected an array, got {value}"))
            .iter()
            .map(|item| item.as_str().expect("string").to_string())
            .collect()
    }

    /// Keep console output ASCII even when a corpus label is Chinese.
    fn ascii_label(label: &str) -> String {
        let mut out = String::new();
        let mut in_non_ascii = false;
        for c in label.chars() {
            if c.is_ascii() {
                out.push(c);
                in_non_ascii = false;
            } else if !in_non_ascii {
                out.push('#');
                in_non_ascii = true;
            }
        }
        out
    }

    fn ranges(value: &Value) -> Vec<(u32, u32)> {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|pair| {
                let pair = pair.as_array().unwrap();
                (
                    pair[0].as_u64().unwrap() as u32,
                    pair[1].as_u64().unwrap() as u32,
                )
            })
            .collect()
    }

    fn in_ranges(table: &[(u32, u32)], cp: u32) -> bool {
        table
            .binary_search_by(|&(start, end)| {
                if cp < start {
                    std::cmp::Ordering::Greater
                } else if cp > end {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Equal
                }
            })
            .is_ok()
    }

    #[test]
    fn corpus_matches_python() {
        let golden = load("text.json");
        let corpus = golden["corpus"].as_array().unwrap();
        let fuzz = golden["fuzz"].as_array().unwrap();
        assert!(
            corpus.len() >= 200,
            "corpus should hold the params_test texts plus the edge cases"
        );
        assert!(fuzz.len() >= 300, "deterministic fuzz corpus missing");
        let fuzz_long = golden["fuzz_long"].as_array().unwrap();
        assert!(fuzz_long.len() >= 100, "long fuzz corpus missing");
        let entries: Vec<&Value> = corpus
            .iter()
            .chain(fuzz.iter())
            .chain(fuzz_long.iter())
            .collect();

        let mut cases = 0usize;
        let mut comparisons = 0usize;
        let mut fragment_total = 0usize;

        for (index, entry) in entries.iter().enumerate() {
            let raw = entry["raw"].as_str().unwrap();
            let label = ascii_label(entry["label"].as_str().unwrap());
            let pipe = add_period(&preprosess_text(raw));

            assert_eq!(
                is_chinese(raw),
                entry["is_chinese"].as_bool().unwrap(),
                "corpus[{index}] ({label}) is_chinese"
            );
            assert_eq!(
                get_lang_type(raw),
                entry["lang_type"].as_str().unwrap(),
                "corpus[{index}] ({label}) lang_type"
            );
            assert_eq!(
                has_valid_content(raw),
                entry["has_valid_content"].as_bool().unwrap(),
                "corpus[{index}] ({label}) has_valid_content"
            );
            assert_eq!(
                get_semantic_length(raw),
                entry["semantic_length"].as_u64().unwrap() as usize,
                "corpus[{index}] ({label}) semantic_length"
            );
            comparisons += 4;

            macro_rules! check_str {
                ($key:expr, $got:expr) => {{
                    assert_eq!(
                        $got,
                        entry[$key].as_str().unwrap(),
                        "corpus[{}] ({}) field {}",
                        index,
                        label,
                        $key
                    );
                    comparisons += 1;
                }};
            }
            macro_rules! check_vec {
                ($key:expr, $got:expr) => {{
                    let got = $got;
                    fragment_total += got.len();
                    assert_eq!(
                        got,
                        strings(&entry[$key]),
                        "corpus[{}] ({}) field {}",
                        index,
                        label,
                        $key
                    );
                    comparisons += 1;
                }};
            }

            check_str!("preprosess", preprosess_text(raw));
            check_str!("add_period", add_period(raw));
            check_str!("normalize_text_kit", normalize_text_kit(raw));
            check_str!("fallback_replace_non_alnum", fallback_replace_non_alnum(raw));
            check_str!("chinese_punc_en2cn", chinese_punc_en2cn(raw));
            check_str!("clean_text_formatting", clean_text_formatting(raw));
            check_str!(
                "protect_normalize_text_kit",
                protect_paralinguistic_tags(raw, normalize_text_kit)
            );
            check_vec!("fragments_24", cut_text_by_semantic_length(raw, 24));
            check_vec!("fragments_40", cut_text_by_semantic_length(raw, 40));
            check_vec!("pipe_fragments_24", cut_text_by_semantic_length(&pipe, 24));
            check_vec!("pipe_fragments_40", cut_text_by_semantic_length(&pipe, 40));

            cases += 1;
            if cases % 50 == 0 {
                println!("text corpus: {cases} entries compared");
            }
        }

        assert_eq!(cases, entries.len());
        println!(
            "text corpus: {cases} entries, {comparisons} field comparisons, {fragment_total} fragments, all equal to Python"
        );
    }

    #[test]
    fn split_long_fragment_matches_python() {
        let golden = load("text.json");
        let fixed = golden["split_long_fragment"].as_array().unwrap();
        let fuzz = golden["fuzz_split_long_fragment"].as_array().unwrap();
        let cases: Vec<&Value> = fixed.iter().chain(fuzz.iter()).collect();
        let mut comparisons = 0usize;
        for (index, case) in cases.iter().enumerate() {
            let raw = case["raw"].as_str().unwrap();
            for (key, max_len) in [("fragments_8", 8usize), ("fragments_12", 12), ("fragments_24", 24)]
            {
                let got = split_long_fragment(raw, max_len);
                assert_eq!(
                    got,
                    strings(&case[key]),
                    "split_long_fragment case {index} max_len {max_len}"
                );
                comparisons += 1;
            }
        }
        println!("split_long_fragment: {comparisons} cases match Python");
    }

    #[test]
    fn append_text_fragment_matches_python() {
        let golden = load("text.json");
        let fixed = golden["append_text_fragment"].as_array().unwrap();
        let fuzz = golden["fuzz_append_text_fragment"].as_array().unwrap();
        let cases: Vec<&Value> = fixed.iter().chain(fuzz.iter()).collect();
        for (index, case) in cases.iter().enumerate() {
            let mut fragments = strings(&case["seed"]);
            append_text_fragment(
                &mut fragments,
                case["new_text"].as_str().unwrap(),
                case["max_len"].as_u64().unwrap() as usize,
                case["min_tail"].as_u64().unwrap() as usize,
            );
            assert_eq!(
                fragments,
                strings(&case["fragments"]),
                "append_text_fragment case {index}"
            );
        }
        println!("append_text_fragment: {} cases match Python", cases.len());
    }

    #[test]
    fn character_classes_match_python() {
        let golden = load("text.json");
        let classes = &golden["char_classes"];

        assert_eq!(
            PY_PUNCTUATION,
            classes["string_punctuation"].as_str().unwrap(),
            "string.punctuation"
        );
        assert_eq!(
            PY_ASCII_WHITESPACE,
            classes["string_whitespace"].as_str().unwrap(),
            "string.whitespace"
        );

        let spaces = ranges(&classes["isspace"]);
        let decimals = ranges(&classes["isdecimal"]);
        let alnum = ranges(&classes["isalnum"]);
        let cjk = ranges(&classes["semantic_cjk"]);
        let latin = ranges(&classes["semantic_latin"]);

        let mut space_bad = 0usize;
        let mut decimal_bad = 0usize;
        let mut alnum_bad = 0usize;
        let mut cjk_bad = 0usize;
        let mut latin_bad = 0usize;
        let mut checked = 0usize;
        for cp in 0..0x110000u32 {
            if (0xD800..=0xDFFF).contains(&cp) {
                continue;
            }
            let Some(c) = char::from_u32(cp) else {
                continue;
            };
            checked += 1;
            if py_isspace(c) != in_ranges(&spaces, cp) {
                space_bad += 1;
            }
            if py_is_decimal(c) != in_ranges(&decimals, cp) {
                decimal_bad += 1;
            }
            if py_isalnum(c) != in_ranges(&alnum, cp) {
                alnum_bad += 1;
            }
            if is_cjk_semantic(c) != in_ranges(&cjk, cp) {
                cjk_bad += 1;
            }
            if is_latin_letter(c) != in_ranges(&latin, cp) {
                latin_bad += 1;
            }
        }

        assert_eq!(space_bad, 0, "str.isspace mismatches: {space_bad}");
        assert_eq!(decimal_bad, 0, "regex \\d mismatches: {decimal_bad}");
        assert_eq!(alnum_bad, 0, "str.isalnum mismatches: {alnum_bad}");
        assert_eq!(cjk_bad, 0, "semantic CJK mismatches: {cjk_bad}");
        assert_eq!(latin_bad, 0, "latin letter mismatches: {latin_bad}");
        println!(
            "character classes: {checked} code points, isspace={} isdecimal={} isalnum={} cjk={} latin={} ranges, 0 mismatches",
            spaces.len(),
            decimals.len(),
            alnum.len(),
            cjk.len(),
            latin.len()
        );
    }
}
