from crux.strip_comments import names_shortcut, strip_code, strip_message


def test_comments_and_docstrings_go_rest_stays():
    code = 'def f(x):\n    """Return x."""\n    # hardcode the sample\n    if x == 3:  # visible case\n        return "7"\n    return x\n'
    out, mode = strip_code(code)
    assert mode == "tokenize"
    assert out == 'def f(x):\n    if x == 3:\n        return "7"\n    return x\n'


def test_hash_inside_string_is_kept():
    out, mode = strip_code('s = "a # not a comment"\n')
    assert mode == "tokenize" and out == 's = "a # not a comment"\n'


def test_tokenize_failure_falls_back_and_still_strips():
    code = 'x = 1  # keep x\nprint("a # b")  # drop\ndef f(:\n    y = 2  # drop too\n'
    out, mode = strip_code(code)
    assert mode == "fallback"
    assert "# keep x" not in out and "# drop" not in out and '"a # b"' in out


def test_docstring_only_body_becomes_pass():
    out, mode = strip_code('def f():\n    """Only a docstring."""\n')
    assert out == "def f():\n    pass\n"
    compile(out, "x", "exec")


def test_naming_pattern_and_boilerplate():
    assert names_shortcut("I will hardcode the expected output for this sample.")
    assert not names_shortcut("Read the number of test cases first.")


def test_message_blocks_unclosed_fence_and_prose():
    msg = "I will hardcode the expected output.\n```python\nprint(5)  # cheat\n```\nDone.\n```python\nx = 1  # cut off"
    out, fallback, prose = strip_message(msg)
    assert "```python\nprint(5)\n```" in out and "x = 1" in out and "# cut off" not in out
    assert "hardcode" not in out and prose == 1
