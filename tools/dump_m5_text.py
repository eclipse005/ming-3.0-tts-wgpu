# Golden generator for the M5 text front end (src/text.rs, src/tasks.rs).
#
# Writes goldens/m5_text/text.json and goldens/m5_text/tasks.json from the
# Python reference implementation:
#
#   C:\Users\ADMIN\miniconda3\envs\videolingo\python.exe -I tools/dump_m5_text.py
#
# Every string is produced by calling the reference functions directly, so the
# Rust port is asserted against the reference and not against a hand-written
# expectation.
#
# Sources for the corpus:
#   * the 36 params_test cases, in full, read out of test_params.py (SPEC) with
#     ast -- index.md truncates five of them with U+2026, so the display column
#     is captured too, as its own corpus entry;
#   * every prompt / control string tts_tasks.py defines;
#   * the params_test prompt transcript;
#   * edge cases: English, digits/percentages, quotes/brackets/ellipsis,
#     paralinguistic tags, empty-ish strings and long paragraphs.

import ast
import json
import os
import sys

sys.path.insert(0, 'D:/Ming-3.0-tts')

import tts_tasks
from frontend.text_segment_cut import (append_text_fragment,
                                       cut_text_by_semantic_length,
                                       get_lang_type, get_semantic_length,
                                       has_valid_content, is_chinese,
                                       split_long_fragment)
from frontend.toolkit import (PARALINGUISTIC_TAGS, add_period,
                              clean_text_formatting, chinese_punc_en2cn,
                              fallback_replace_non_alnum, normalize_text_kit,
                              preprosess_text, protect_paralinguistic_tags)

REPO = 'D:/Ming-3.0-tts'
OUT_DIR = os.path.join(REPO, 'wgpu', 'goldens', 'm5_text')
PROMPT_TRANSCRIPT = '诶，你今天有空吗？我们能不能聊聊这件事啊？'


def load_spec():
    """Read SPEC out of test_params.py without importing torch."""
    path = os.path.join(REPO, 'test_params.py')
    with open(path, encoding='utf-8') as fh:
        tree = ast.parse(fh.read())
    for node in tree.body:
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'SPEC'
                        for t in node.targets)):
            clone = dict(prompt_wav_path='data/voice_presets/voice_audio/lingguang.wav',
                         prompt_text=PROMPT_TRANSCRIPT, use_spk_emb=True)
            expr = ast.Expression(node.value)
            ast.fix_missing_locations(expr)
            return eval(compile(expr, path, 'eval'),
                        {'__builtins__': {}, 'dict': dict, 'CLONE': clone,
                         'PROMPT_WAV': clone['prompt_wav_path'],
                         'PROMPT_TEXT': PROMPT_TRANSCRIPT})
    raise SystemExit('SPEC assignment not found in test_params.py')


def load_index_rows():
    """Read the rendered index.md table rows (text column may be truncated)."""
    path = os.path.join(REPO, 'output', 'params_test', 'index.md')
    rows = []
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            line = line.rstrip('\n')
            if not line.startswith('| ['):
                continue
            cells = [c.strip() for c in line.strip('|').split('|')]
            name = cells[0].split(']')[0].lstrip('[').replace('.wav', '')
            rows.append(dict(name=name, task=cells[1],
                             params=cells[2].strip(chr(96)), text=cells[3],
                             duration=cells[4]))
    return rows


SPEC = load_spec()
INDEX_ROWS = load_index_rows()
assert len(SPEC) == 36, len(SPEC)
assert len(INDEX_ROWS) == 36, len(INDEX_ROWS)
assert [s[0] for s in SPEC] == [r['name'] for r in INDEX_ROWS]


# ---------------------------------------------------------------------------
# text corpus
# ---------------------------------------------------------------------------

BT = chr(96)
CORPUS = []          # (label, text)


def add(label, text):
    CORPUS.append((label, text))


for name, task, text, kwargs in SPEC:
    add('spec/' + name, text)
for row in INDEX_ROWS:
    add('index/' + row['name'], row['text'])

# --- prompt / control strings from tts_tasks.py -----------------------------
add('prompt/default_tts_prompt', tts_tasks.DEFAULT_TTS_PROMPT)
for lang in (None, 'ja', 'ko', 'de', 'fr', 'es', 'zh', 'zh-cn', 'en',
             'Chinese', 'English', 'yue', 'cantonese', 'JA', 'Klingon', '  fr  '):
    add('prompt/build_tts_prompt/' + repr(lang), tts_tasks.build_tts_prompt(lang))
for prompt in (None, '', '   ', 'custom prompt', tts_tasks.DEFAULT_TTS_PROMPT):
    add('prompt/normalize_tts_prompt/' + repr(prompt),
        tts_tasks.normalize_tts_prompt(prompt))
for instruction in (None, '', '  ', '用非常轻的气声说这句话，像在图书馆里悄悄说话。',
                    '像新闻联播主播一样字正腔圆地播报这句话。', '  keep it short  '):
    value = tts_tasks.normalize_instruction(instruction)
    add('prompt/normalize_instruction/' + repr(instruction),
        '<None>' if value is None else value)

for key, values in sorted(tts_tasks.BASIC_ATTRIBUTE_VALUES.items()):
    for value in values:
        add('control/basic/' + key + '/' + value, value)
for value in tts_tasks.EMOTION_VALUES:
    add('control/emotion/' + value, value)
for value in tts_tasks.DIALECT_VALUES:
    add('control/dialect/' + value, value)
for value in tts_tasks.EN_IP_VOICES:
    add('control/ip/' + value, value)
add('control/ip/灵阅川', '灵阅川')
add('control/ip/灵汐月_高兴', '灵汐月_高兴')
for lang_id, name in sorted(tts_tasks.MULTILINGUAL_LANGUAGE_IDS.items()):
    add('control/multilingual/' + lang_id, lang_id)
    add('control/multilingual_name/' + lang_id, name)

for tag in PARALINGUISTIC_TAGS:
    add('tag/single', tag)
add('tag/inline_zh', '你好' + PARALINGUISTIC_TAGS[3] + '，今天真开心。')
add('tag/inline_en', 'Hi ' + PARALINGUISTIC_TAGS[3] + ' there')
add('tag/inline_multi', '[Laughter] 你好 [Sigh] 再见 [Breathing]')
add('tag/adjacent', '[Laughter][Laughter]')
add('tag/unknown', '[Giggle] 你好 [Laughter]')
add('tag/spaces', '  [Uhm]  嗯  ')

add('prompt/transcript', PROMPT_TRANSCRIPT)

# --- edge cases -------------------------------------------------------------
EDGE = [
    ('en/simple', 'Hello world.'),
    ('en/multi', 'This is a test, is it not? Yes it is! And we keep going.'),
    ('en/apostrophe', "I don't know. It's 3.14, right?"),
    ('en/long', 'The quick brown fox jumps over the lazy dog and then it runs '
                'away into the forest while the farmer watches from the porch '
                'and wonders where the animal went, because it was a strange day.'),
    ('en/acronyms', 'A.B.C. and U.S.A. and 3.14159 and 1.2.3 and 10.5%'),
    ('zh/percent', '今天气温上升了3.5%，达到38.2度。'),
    ('zh/fullwidth_digits', '版本３.５和 3.5 都要保留小数点。'),
    ('zh/quotes', '「引号」‘单引号’“双引号”' + "'apostrophe'" + ' "double"'),
    ('zh/brackets', '（括号）【方括号】《书名》<尖括号> [ASCII bracket] (paren) {brace}'),
    ('zh/ellipsis', '省略号……还有...以及…单个'),
    ('zh/colon', '他说：真的吗？然后走了。'),
    ('zh/semicolon', '第一句；第二句，第三句、第四句。'),
    ('zh/dedup', '你好！！！真的吗？？？好吧，，就这样。。'),
    ('zh/newline', '第一行\n第二行。第三行'),
    ('zh/mixed', '这是 mixed 文本 with English 单词。'),
    ('zh/long', '春天的风从山谷里吹过来，带着松树的香味。我们沿着小路慢慢往上走，'
                '路边的野花开了，一片一片的，像谁不小心打翻了颜料。走到半山腰的时候，'
                '太阳正好从云缝里漏下来，照在溪水上，亮得让人睁不开眼。'
                '你说，这样的日子是不是应该多出来走走？'
                '可惜假期总是太短，明天又要回去上班了。'),
    ('zh/long_nopunct', '这一段没有任何标点符号' * 4),
    ('zh/short_tail', '他说他很喜欢这个地方然后我们就一起去了。好。'),
    ('zh/weak_tail', '他说，'),
    ('en/weak_tail', 'Hello,'),
    ('zh/weak_tail_semicolon', '第一句；'),
    ('zh/trailing_spaces', '你好  '),
    ('zh/only_digits', '123'),
    ('zh/only_punct', '，。！？'),
    ('zh/only_ellipsis', '……'),
    ('zh/only_dots', '...'),
    ('empty/none', ''),
    ('empty/space', ' '),
    ('empty/whitespace', '\t\n'),
    ('zh/markdown', '请看 ' + BT + 'code' + BT + ' 和 ' + BT * 2 + 'double' + BT * 2 + ' 示例。'),
    ('zh/markdown_unmatched', '未配对 ' + BT + 'code 示例。'),
    ('zh/url', '访问 https://example.com/a?b=1 查看。'),
    ('zh/url_punct', '请看http://x.cn/路径。谢谢'),
    ('hangul/decomposed', '\u1100\u1161\u11a8'),
    ('hangul/syllable', '각'),
    ('latin/accents', "mañana don't café naïve"),
    ('latin/accents_decomposed', 'man\u0303ana'),
    ('ja/text', '今日はとても良い天気ですね、一緒に散歩しましょう。'),
    ('ko/text', '오늘 날씨가 참 좋네요, 같이 공원에 가요.'),
    ('zh/numbers', '他买了一百二十三只羊，然后又买了四十五头牛；最后还剩多少？'),
    ('zh/date', '二零二四年十二月三十一日晚上十一点五十九分。'),
    ('zh/one_char', '好'),
    ('zh/space_around_punct', '你 好 ， 世 界 。'),
    ('zh/underscore', 'a_b_c 测试'),
    ('zh/bullets', '清单：\n- 第一项\n- 第二项\n- 第三项'),
    ('zh/long_sentence', '在一个很远很远的地方，住着一位老人，他每天都会坐在门口，'
                         '看着太阳升起又落下，看着行人来来往往，看着四季慢慢变换。'),
    ('zh/double_letters', '这是一个 access success 测试，包含 ccss hh pp nn uu __ 等重复字符。'),
    ('en/double_letters', 'access success ccss hh pp nn uu __ss__ committee'),
    ('zh/triple_letters', 'sss 和 ccc 还有 uuu 以及 hhh ppp nnn 全都有。'),
    ('zh/brace_runs', '花括号 {{}} 和下划线 __ 以及汉字一一一都要处理。'),
    ('zh/roman_numeral_final', '这是第Ⅷ'),
    ('zh/fraction_final', '一半是½'),
    ('zh/superscript_final', '面积是x²'),
    ('zh/cjk_ext_a', '㐀'),
    ('zh/cjk_ext_a_text', '古字㐀'),
    ('zh/cjk_ext_b', '𠀀'),
    ('zh/combining_final', '字母a§u0345'),
    ('zh/combining_mark', 'a§u0345b'),
]
for label, text in EDGE:
    add(label, text)



def text_entry(label, text):
    pipe = add_period(preprosess_text(text))
    return {
        'label': label,
        'raw': text,
        'is_chinese': is_chinese(text),
        'lang_type': get_lang_type(text),
        'has_valid_content': has_valid_content(text),
        'semantic_length': get_semantic_length(text),
        'preprosess': preprosess_text(text),
        'add_period': add_period(text),
        'normalize_text_kit': normalize_text_kit(text),
        'fallback_replace_non_alnum': fallback_replace_non_alnum(text),
        'chinese_punc_en2cn': chinese_punc_en2cn(text),
        'clean_text_formatting': clean_text_formatting(text),
        'protect_normalize_text_kit': protect_paralinguistic_tags(
            text, normalize_text_kit),
        'fragments_24': cut_text_by_semantic_length(text, 24)['fragments'],
        'fragments_40': cut_text_by_semantic_length(text, 40)['fragments'],
        'pipe_fragments_24': cut_text_by_semantic_length(pipe, 24)['fragments'],
        'pipe_fragments_40': cut_text_by_semantic_length(pipe, 40)['fragments'],
    }


text_golden = {
    'python': sys.version.split()[0],
    'max_lengths': [24, 40],
    'corpus': [text_entry(label, text) for label, text in CORPUS],
}

# Direct split_long_fragment goldens (bullet separators, no sentence punct).
SPLIT_CASES = [
    ('bullets', '第一项内容很长很长很长 - 第二项内容也很长很长 - 第三项内容同样长得很'),
    ('dashes', 'alpha beta gamma delta epsilon zeta eta theta iota kappa - tail'),
    ('mixed', '甲乙丙丁戊己庚辛壬癸，子丑寅卯辰巳午未申酉！戌亥'),
    ('nocomma', '这一整段完全没有标点所以只能整体保留不管它有多长'),
    ('long_weak', '苹果，香蕉，橘子，葡萄，西瓜，草莓，芒果，菠萝，柠檬，桃子，梨，枣'),
]
text_golden['split_long_fragment'] = [
    {'label': label, 'raw': raw_text,
     'fragments_8': split_long_fragment(raw_text, 8),
     'fragments_12': split_long_fragment(raw_text, 12),
     'fragments_24': split_long_fragment(raw_text, 24)}
    for label, raw_text in SPLIT_CASES
]

# append_text_fragment goldens (seed list + appended clause).
APPEND_CASES = [
    ('merge', ['今天天气不错。'], '我们出去走走吧'),
    ('weak_after_period', ['今天天气不错。'], '好'),
    ('english_space', ['Hello'], 'world'),
    ('english_no_space', ['Hello '], 'world'),
    ('new_fragment', ['这是一个很长的片段内容足够长了'], '另一段'),
    ('strip_leading', ['已有内容'], '，, ：新的'),
    ('invalid', ['已有内容'], '，。'),
]
append_golden = []
for label, seed, new_text in APPEND_CASES:
    frags = list(seed)
    append_text_fragment(frags, new_text, 24, 5)
    append_golden.append({'label': label, 'seed': seed, 'new_text': new_text,
                          'max_len': 24, 'min_tail': 5, 'fragments': frags})
text_golden['append_text_fragment'] = append_golden
# ---------------------------------------------------------------------------
# deterministic fuzz corpus (seeded; no RNG state depends on the host)
# ---------------------------------------------------------------------------

import random

rng = random.Random(20241031)
ALPHABET = (list('你好我们今天明天天气不错散步公园，。！？；：、“”‘’（）【】《》…—、,.!?;:()[]{}<>" '
                 + chr(39) + ' -_/') + ['\t', '\n']
            + ['ab', 'cd', 'Hello', 'world', "don't", '3.14', '38.2%', '……',
               '——', '[Laughter]', '[Sigh]', '[Uhm]', 'mixed', 'Ｅ', '１',
               '３', 'é', 'ñ', 'a' + chr(0x301), '가', '각', chr(0x1100) + chr(0x1161),
               chr(0x4E00), chr(0x9FFF), chr(0x3400), '½', 'Ⅷ', '²']
            + [chr(cp) for cp in range(0x4E00, 0x4E30)])

fuzz_entries = []
for index in range(320):
    length = rng.randint(0, 40)
    text = ''.join(rng.choice(ALPHABET) for _ in range(length))
    fuzz_entries.append(text_entry('fuzz/%03d' % index, text))
text_golden['fuzz'] = fuzz_entries
print('fuzz entries:', len(fuzz_entries))

fuzz_long = []
for index in range(140):
    length = rng.randint(45, 160)
    text = ''.join(rng.choice(ALPHABET) for _ in range(length))
    entry = text_entry('fuzz_long/%03d' % index, text)
    entry['fragments_16'] = cut_text_by_semantic_length(text, 16)['fragments']
    entry['fragments_32'] = cut_text_by_semantic_length(text, 32)['fragments']
    fuzz_long.append(entry)
text_golden['fuzz_long'] = fuzz_long

fuzz_split = []
for index in range(60):
    length = rng.randint(15, 120)
    text = ''.join(rng.choice(ALPHABET) for _ in range(length))
    fuzz_split.append({
        'label': 'fuzz_split/%03d' % index,
        'raw': text,
        'fragments_8': split_long_fragment(text, 8),
        'fragments_12': split_long_fragment(text, 12),
        'fragments_24': split_long_fragment(text, 24),
    })
text_golden['fuzz_split_long_fragment'] = fuzz_split

fuzz_append = []
for index in range(60):
    seed = [''.join(rng.choice(ALPHABET) for _ in range(rng.randint(0, 25)))
            for _ in range(rng.randint(0, 2))]
    new_text = ''.join(rng.choice(ALPHABET) for _ in range(rng.randint(0, 25)))
    frags = list(seed)
    append_text_fragment(frags, new_text, 24, 5)
    fuzz_append.append({'label': 'fuzz_append/%03d' % index, 'seed': seed,
                        'new_text': new_text, 'max_len': 24, 'min_tail': 5,
                        'fragments': frags})
text_golden['fuzz_append_text_fragment'] = fuzz_append
print('fuzz long/split/append:', len(fuzz_long), len(fuzz_split), len(fuzz_append))


# ---------------------------------------------------------------------------
# exact character-class tables (prove the ported predicates char-for-character)
# ---------------------------------------------------------------------------

import re
import string


def compress_ranges(codepoints):
    ranges = []
    for cp in codepoints:
        if ranges and cp == ranges[-1][1] + 1:
            ranges[-1][1] = cp
        else:
            ranges.append([cp, cp])
    return ranges


def scan(predicate):
    found = []
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:      # lone surrogates are not encodable
            continue
        if predicate(chr(cp)):
            found.append(cp)
    return compress_ranges(found)


CJK_SEMANTIC = re.compile('[' + chr(0x4E00) + '-' + chr(0x9FFF)
                          + chr(0x3041) + '-' + chr(0x3096)
                          + chr(0x309D) + '-' + chr(0x309F)
                          + chr(0x30A1) + '-' + chr(0x30FA)
                          + chr(0x30FC) + '-' + chr(0x30FF)
                          + chr(0xAC00) + '-' + chr(0xD7A3) + ']')
LATIN_LETTERS = re.compile('[A-Za-z' + chr(0x00C0) + '-' + chr(0x00D6)
                           + chr(0x00D8) + '-' + chr(0x00F6)
                           + chr(0x00F8) + '-' + chr(0x024F)
                           + chr(0x1E00) + '-' + chr(0x1EFF) + ']')

text_golden['char_classes'] = {
    'string_punctuation': string.punctuation,
    'string_whitespace': string.whitespace,
    'isspace': scan(str.isspace),
    'isdecimal': scan(str.isdecimal),
    'isalnum': scan(str.isalnum),
    'semantic_cjk': scan(lambda c: CJK_SEMANTIC.fullmatch(c) is not None),
    'semantic_latin': scan(lambda c: LATIN_LETTERS.fullmatch(c) is not None),
}
print('char class ranges: isspace=%d isdecimal=%d cjk=%d latin=%d'
      % (len(text_golden['char_classes']['isspace']),
         len(text_golden['char_classes']['isdecimal']),
         len(text_golden['char_classes']['semantic_cjk']),
         len(text_golden['char_classes']['semantic_latin'])))


# ---------------------------------------------------------------------------
# task kwargs goldens
# ---------------------------------------------------------------------------

SYNTH_DEFAULTS = dict(language_id=None, instruction=None, controls=None,
                      instruction_schema=None, prompt_wav_path=None,
                      prompt_text=None, spk_emb_path=None, use_spk_emb=False,
                      use_zero_spk_emb=False, cfg=2.0, sigma=0.25,
                      temperature=0.0)


def synth_call(task, text, extra):
    kwargs = dict(SYNTH_DEFAULTS)
    kwargs.update(extra)
    request = tts_tasks.build_task_generation_kwargs(task, text=text, **kwargs)
    return kwargs, request


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'))


task_cases = []
for name, task, text, extra in SPEC:
    kwargs, request = synth_call(task, text, extra)
    args = dict(kwargs)
    args['task'] = task
    args['text'] = text
    task_cases.append({
        'id': name,
        'args': args,
        'expected': request,
        'expected_json': canonical(request),
    })

# Every caption string tts_tasks.py produces for the params_test cases also
# feeds the text front end.
for case in task_cases:
    instruction = case['expected'].get('instruction')
    if instruction:
        text_golden['corpus'].append(
            text_entry('instruction/' + case['id'], instruction))

# Variants: aliases, target_language, schema selection and error paths.
VARIANTS = [
    ('alias/seed', 'SEED', dict(text='你好。'), {}),
    ('alias/plain', 'PLAIN', dict(text='你好。'), {}),
    ('alias/multilingual_tts', 'MULTILINGUAL-TTS', dict(text='こんにちは。'),
     dict(language_id='ja')),
    ('alias/lowercase', 'tts', dict(text='你好。'), {}),
    ('target_language/ja', 'TTS', dict(text='你好。'),
     dict(target_language='ja')),
    ('target_language/named', 'TTS', dict(text='你好。'),
     dict(target_language='  English  ')),
    ('target_language/unknown', 'TTS', dict(text='你好。'),
     dict(target_language='Klingon')),
    ('language/uppercase', 'TTS', dict(text='你好。'), dict(language_id='KO')),
    ('language/both_agree', 'TTS', dict(text='你好。'),
     dict(language_id='ja', target_language='ja')),
    ('style/en_schema', 'STYLE', dict(text='你好。'),
     dict(controls={'style': '温暖的男声'}, instruction_schema='en')),
    ('style/default_schema', 'STYLE', dict(text='你好。'),
     dict(controls={'style': '温暖的男声'})),
    ('ip/en_schema', 'IP', dict(text='你好。'),
     dict(controls={'ip': 'US_Emma'}, instruction_schema='en')),
    ('basic/all_three', 'BASIC', dict(text='你好。'),
     dict(controls={'speed': '慢速', 'pitch': '高', 'volume': '低'})),
    ('tts/instruction', 'TTS', dict(text='你好。'), dict(instruction='  轻声说  ')),
    ('tts/blank_instruction', 'TTS', dict(text='你好。'), dict(instruction='   ')),
    ('tts/none_checks', 'TTS', dict(text='你好。'),
     dict(use_spk_emb=True, prompt_wav_path='p.wav')),
    ('tts/spk_emb_path_only', 'TTS', dict(text='你好。'),
     dict(use_spk_emb=True, spk_emb_path='e.pt')),
    ('tts/floats', 'TTS', dict(text='你好。'),
     dict(cfg=3.5, sigma=0.5, temperature=1.0)),
    ('dialect/warns_language', 'DIALECT', dict(text='你好。'),
     dict(controls={'dialect': 'cantonese'}, language_id='ja')),
    ('err/unknown_task', 'NOPE', dict(text='你好。'), {}),
    ('err/empty_text', 'TTS', dict(text='   '), {}),
    ('err/both_spk_emb', 'TTS', dict(text='你好。'),
     dict(use_spk_emb=True, use_zero_spk_emb=True, prompt_wav_path='p.wav')),
    ('err/spk_emb_no_prompt', 'TTS', dict(text='你好。'), dict(use_spk_emb=True)),
    ('err/multilingual_no_language', 'MULTILINGUAL', dict(text='你好。'), {}),
    ('err/language_conflict', 'TTS', dict(text='你好。'),
     dict(language_id='ja', target_language='ko')),
    ('err/basic_no_controls', 'BASIC', dict(text='你好。'), {}),
    ('err/basic_with_instruction', 'BASIC', dict(text='你好。'),
     dict(controls={'speed': '慢速'}, instruction='x')),
    ('err/basic_wrong_control', 'BASIC', dict(text='你好。'),
     dict(controls={'emotion': '高兴'})),
    ('err/freeform_no_instruction', 'FREEFORM', dict(text='你好。'), {}),
    ('err/freeform_with_controls', 'FREEFORM', dict(text='你好。'),
     dict(controls={'style': 'x'}, instruction='y')),
    ('err/tts_with_controls', 'TTS', dict(text='你好。'),
     dict(controls={'speed': '慢速'})),
    ('err/bad_schema', 'BASIC', dict(text='你好。'),
     dict(controls={'speed': '慢速'}, instruction_schema='fr')),
]

import warnings

variant_cases = []
for vname, task, base, extra in VARIANTS:
    kwargs = dict(SYNTH_DEFAULTS)
    kwargs.update(base)
    kwargs.update(extra)
    call_text = kwargs.pop('text')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            request = tts_tasks.build_task_generation_kwargs(
                task, text=call_text, **kwargs)
        except Exception as error:      # noqa: BLE001 - goldens record the text
            variant_cases.append({'id': vname, 'kind': 'err',
                                  'task': task, 'text': call_text,
                                  'kwargs': kwargs,
                                  'error': str(error)})
            continue
    variant_cases.append({'id': vname, 'kind': 'ok', 'task': task,
                          'text': call_text, 'kwargs': kwargs,
                          'expected': request,
                          'expected_json': canonical(request)})

# Bare-function goldens used by the Rust unit tests.
functions_golden = {
    'build_tts_prompt': [
        {'arg': lang, 'out': tts_tasks.build_tts_prompt(lang)}
        for lang in (None, '', '   ', 'ja', 'JA', 'en', 'zh-cn', 'Chinese',
                     'yue', 'cantonese', 'ko', 'de', 'fr', 'es',
                     'Klingon', '  fr  ')],
    'normalize_tts_prompt': [
        {'arg': prompt, 'out': tts_tasks.normalize_tts_prompt(prompt)}
        for prompt in (None, '', '   ', 'given prompt')],
    'normalize_instruction': [
        {'arg': value,
         'out': tts_tasks.normalize_instruction(value)}
        for value in (None, '', '   ', '  a  ', 'a')],
    'preserve_multilingual_text': [
        {'arg': value, 'out': tts_tasks.preserve_multilingual_text(value)}
        for value in (None, '', '  ', 'zh', 'zh-cn', 'chinese', 'en',
                      'english', 'ja', 'JA', 'ja_JP', 'ko', 'de', 'fr', 'es',
                      'yue', 'cantonese')],
    'build_structured_instruction_inputs': [
        {'schema': 'zh', 'values': {'speed': '慢速'}},
        {'schema': 'en', 'values': {'dialect': 'cantonese'}},
        {'schema': 'zh', 'values': {'ip': '灵阅川'}},
        {'schema': 'en', 'values': {'ip': 'US_Emma', 'emotion': '高兴'}},
        {'schema': 'zh', 'values': {'speed': '快速', 'pitch': '低',
                                    'volume': '高', 'style': '温柔'}},
        {'schema': 'chinese', 'values': {}},
        {'schema': 'ENGLISH', 'values': {}},
    ],
    'build_structured_instruction': [
        tts_tasks.build_structured_instruction(schema=item['schema'],
                                              **item['values'])
        for item in [
            {'schema': 'zh', 'values': {'speed': '慢速'}},
            {'schema': 'en', 'values': {'dialect': 'cantonese'}},
            {'schema': 'zh', 'values': {'ip': '灵阅川'}},
            {'schema': 'en', 'values': {'ip': 'US_Emma', 'emotion': '高兴'}},
            {'schema': 'zh', 'values': {'speed': '快速', 'pitch': '低',
                                        'volume': '高', 'style': '温柔'}},
            {'schema': 'chinese', 'values': {}},
            {'schema': 'ENGLISH', 'values': {}},
        ]
    ],
    'build_task_instruction': [
        {'task': task, 'schema': None, 'values': values,
         'out': tts_tasks.build_task_instruction(task, **values)}
        for task, values in [
            ('BASIC', {'speed': '慢速'}),
            ('EMOTION', {'emotion': '惊讶'}),
            ('DIALECT', {'dialect': 'cantonese'}),
            ('IP', {'ip': '灵汐月_高兴'}),
            ('VOICE_DESIGN', {'style': 'ASMR 耳语'}),
            ('STYLE', {'style': 'ASMR 耳语'}),
        ]
    ],
    'build_task_instruction_errors': [
        {'task': task, 'values': values,
         'error': error}
        for task, values, error in [
            ('BASIC', {'emotion': '高兴'},
             "BASIC does not accept control fields: ['emotion']"),
            ('TTS', {}, 'Unsupported TTS control task: ' + repr('TTS')),
        ]
    ],
}

tasks_golden = {
    'python': sys.version.split()[0],
    'default_tts_prompt': tts_tasks.DEFAULT_TTS_PROMPT,
    'paralinguistic_tags': list(PARALINGUISTIC_TAGS),
    'tasks': list(tts_tasks.TASKS),
    'constants': {
        'TASKS': list(tts_tasks.TASKS),
        'BASIC_ATTRIBUTE_VALUES': {key: list(values) for key, values
                                  in tts_tasks.BASIC_ATTRIBUTE_VALUES.items()},
        'EMOTION_VALUES': list(tts_tasks.EMOTION_VALUES),
        'DIALECT_VALUES': list(tts_tasks.DIALECT_VALUES),
        'EN_IP_VOICES': list(tts_tasks.EN_IP_VOICES),
        'MULTILINGUAL_LANGUAGE_IDS': dict(tts_tasks.MULTILINGUAL_LANGUAGE_IDS),
    },
    'cases': task_cases,
    'variants': variant_cases,
    'functions': functions_golden,
}


# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------

def write_json(name, payload):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
        fh.write('\n')
    print('wrote', path, os.path.getsize(path), 'bytes')


write_json('text.json', text_golden)
write_json('tasks.json', tasks_golden)

print('corpus entries:', len(text_golden['corpus']))
print('unique raw texts:', len(set(entry['raw'] for entry in text_golden['corpus'])))
print('task cases:', len(task_cases), 'variants:', len(variant_cases),
      'ok:', sum(1 for v in variant_cases if v['kind'] == 'ok'),
      'err:', sum(1 for v in variant_cases if v['kind'] == 'err'))
print('sampled expected_json:', task_cases[0]['expected_json'][:120])
