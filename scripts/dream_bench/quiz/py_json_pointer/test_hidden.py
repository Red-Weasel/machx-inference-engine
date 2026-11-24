from solution import get, set_

doc = {"foo": ["bar", "baz"], "": 0, "a/b": 1, "c%d": 2, "e^f": 3, "g|h": 4, "i\\j": 5, "k\"l": 6, " ": 7, "m~n": 8, "-": 9, "~1": 10}
cases = {"": doc, "/foo": ["bar", "baz"], "/foo/0": "bar", "/": 0, "/a~1b": 1, "/c%d": 2, "/e^f": 3, "/g|h": 4, "/i\\j": 5, "/k\"l": 6, "/ ": 7, "/m~0n": 8, "/-": 9, "/~01": 10}
for p, want in cases.items(): assert get(doc, p) == want, p
def err(exc, fn, *a):
    try: fn(*a)
    except exc: return
    raise AssertionError(f"{fn.__name__}{a[1:]} did not raise {exc.__name__}")
err(ValueError, get, doc, "foo"); err(ValueError, get, doc, "/m~2n"); err(ValueError, get, doc, "/x~"); err(KeyError, get, doc, "/nope")
err(IndexError, get, doc, "/foo/2"); err(IndexError, get, doc, "/foo/-"); err(IndexError, get, doc, "/foo/01"); err(IndexError, get, doc, "/foo/-1")
err(IndexError, get, doc, "/foo/x"); err(TypeError, get, doc, "/foo/0/x"); err(TypeError, get, {"n": 5}, "/n/0"); err(TypeError, get, {"n": None}, "/n/k")
d = {"a": {"b": [1, 2]}}
assert set_(d, "/a/b/0", 9) is d and d == {"a": {"b": [9, 2]}}
set_(d, "/a/b/2", 3); set_(d, "/a/b/-", 4); assert d["a"]["b"] == [9, 2, 3, 4]
err(IndexError, set_, d, "/a/b/9", 0); err(IndexError, set_, d, "/a/b/01", 0); err(KeyError, set_, d, "/x/y", 0); err(TypeError, set_, d, "/a/b/0/z", 0)
set_(d, "/a/new~1key", {"q": 1}); assert d["a"]["new/key"] == {"q": 1}; set_(d, "/a/-", 1); assert d["a"]["-"] == 1
orig = {"k": 1}; assert set_(orig, "", [1]) == [1] and orig == {"k": 1}
lst = []; set_(lst, "/0", "x"); set_(lst, "/-", "y"); assert lst == ["x", "y"]
err(ValueError, set_, {}, "a", 1)
print("OK")
