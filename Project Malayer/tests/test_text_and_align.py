from malayer.g2p.align import align, decode
from malayer.phonemes import canonical_r2, join_ezafe, r1_to_r2, split_ezafe
from malayer.text.normalize import ZWNJ, normalize


def test_normalize_unifies_letters_and_zwnj():
    assert normalize("كتاب  علي") == "کتاب علی"
    assert normalize("می" + ZWNJ + " روم") == "می روم"
    assert normalize("خانۀ من") == "خانه" + ZWNJ + "ی من"
    assert normalize("۱۲۳") == "123"


def test_ezafe_roundtrip():
    for w in ["ketabe1", "xaneye1", "baziye1", "ketab", "ruye1"]:
        assert join_ezafe(*split_ezafe(w)) == w
    assert canonical_r2("q/vie1 miz") == "q/viye1 miz"


def test_r1_to_r2_sentencebench():
    assert r1_to_r2("man qadr-e to rA mi-dAnam") == "m/n q/dre1 to ra midan/m"
    assert r1_to_r2("mo?Ahede-ye marAkeS") == "mo@ahedeye1 m/rake$"


def test_align_reconstructs_reference():
    g = "دوران انتظار امام زمان سخت است"
    ref = "dorane1 @entezare1 @emame1 z/man s/xt @/st"
    a = align(g, ref)
    assert a is not None and len(a.labels) == len(g)
    assert a.to_r2() == ref
    assert decode(g, a.labels, a.ezafe) == ref


def test_written_ezafe_becomes_flag():
    a = align("کتاب" + ZWNJ + "های خوب", "ketabhaye xub")
    assert a.to_r2() == "ketabhaye1 xub"
    assert a.labels[-5] == "" and a.ezafe[-5]  # the written ی


def test_corrupt_reference_rejected():
    assert align("کتاب خوب", "@input ktab xb") is None


def test_persian_numbers():
    from malayer.text.numbers import expand_numbers, int_to_words

    assert int_to_words(1402) == "هزار و چهارصد و دو"
    assert int_to_words(21) == "بیست و یک"
    assert int_to_words(2_000_015) == "دو میلیون و پانزده"
    assert expand_numbers(normalize("سال ۱۴۰۲ بود")) == "سال هزار و چهارصد و دو بود"
    assert expand_numbers("3.5") == "سه ممیز پنج"
