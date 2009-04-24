import functools
import numpy as np
import rpy2.rinterface as ri

__all__ = ("r", "rcopy", "rarray", "rzeros", "rones")

ri.initr()

class R(object):
    def __init__(self, ri_environment):
        object.__setattr__(self, "_env", ri_environment)

    ### Forward dictionary-like accesses to the global environment:
    def __getitem__(self, name):
        return ri2py(object.__getattribute__(self, "_env").get(name), name)

    def __setitem__(self, name, value):
        object.__getattribute__(self, "_env")[name] = py2ri(value)

    def __delitem__(self, name):
        self.remove(name)

    def __contains__(self, name):
        try:
            self[name]
        except LookupError:
            return False
        else:
            return True

    ### Also forward attribute-like accesses there, as a convenience, with
    ### somewhat quirkier rules:
    def __getattribute__(self, name):
        if name.endswith("__"):
            return object.__getattribute__(self, name)
        try:
            return self[_munge_py_name_to_r(name)]
        except LookupError:
            raise AttributeError, name

    def __setattr__(self, *args, **kwargs):
        raise NotImplementedError, "assign to r[foo] instead"

    def __delattr__(self, *args, **kwargs):
        raise NotImplementedError, "delete r[foo] instead"

    def __call__(self, string, env=None):
        p = self.parse(text=string,
                       srcfile=self.srcfilecopy("<Python r(...)>", string))
        self_env = object.__getattribute__(self, "_env")
        if env is None:
            env = self_env
        try:
            return r.eval(p, envir=py2ri(env), enclos=self_env)
        except ri.RRuntimeError, e:
            raise RRuntimeError(e, string)

r = R(ri.globalEnv)

def test_r_mapping_interface():
    assert "lm" in r
    assert isinstance(r["lm"], RClosure)
    assert "foo" not in r
    r["foo"] = 1
    assert "foo" in r
    assert r["foo"][0] == 1
    del r["foo"]
    assert "foo" not in r

def test_r_attribute_interface():
    assert r.lm is not None
    assert isinstance(r.class_, RClosure)
    assert isinstance(r.dollar, RClosure)
    assert isinstance(r.as_integer, RClosure)

def test_r_call():
    assert r("1 + 1")[0] == 2
    # Check that it converts errors into our preferred exception type:
    try:
        r("stop('asdf')")
    except RRuntimeError, e:
        assert "asdf" in str(e.exc)
    except:
        assert False
    else:
        assert False

def rcopy(obj):
    return ri2py(py2ri(obj), None)

def test_rcopy():
    assert isinstance(rcopy("foo"), RWrapper)
    assert rcopy(1)[0] == 1

def _capture_r_output(f, *args, **kwargs):
    if hasattr(ri, "getWriteConsole"):
        prev = ri.getWriteConsole()
    else:
        import sys
        prev = sys.stdout.write
    buf = []
    def capture(s):
        buf.append(s)
    try:
        ri.setWriteConsole(capture)
        f(*args, **kwargs)
    finally:
        ri.setWriteConsole(prev)
    return "".join(buf)
    
def test__capture_r_output():
    assert _capture_r_output(r.print_, 1) == "[1] 1\n"

# For interactive front-ends:
def is_complete_expression(string):
    try:
        # Don't actually want the output, but this suppresses any
        # direct-to-console printing of errors:
        _capture_r_output(r.parse, text=string)
    except RRuntimeError, e:
        # XX FIXME: Surely there is a better way to do this...
        if "unexpected end of input" in e.exc.message:
            return False
    return True

def complete(line, cursor_idx):
    # cursor_idx is the 0-based position of the cursor within the line
    # Magic stolen from ESS source code:
    r("utils:::.assignLinebuffer")(line)
    r("utils:::.assignEnd")(cursor_idx)
    token = r("utils:::.guessTokenFromLine()")
    r("utils:::.completeToken()")
    completions = r("utils:::.retrieveCompletions()")
    return [completion[len(token[0]):] for completion in list(completions)]

def test_is_complete_expression():
    assert not is_complete_expression("1 +")
    assert is_complete_expression("1 + 1")
    assert is_complete_expression("1 1")

class RRuntimeError(ri.RRuntimeError):
    "An enhanced RRuntimeError that includes the R-side exception."
    def __init__(self, exc, call):
        # Prevent double-wrapping:
        while isinstance(exc, RRuntimeError):
            exc = exc.exc
        self.exc = exc
        self.call = call
        try:
            self.tb = _capture_r_output(r.traceback)
        except ri.RRuntimeError:
            self.tb = "<error generating R traceback>"
        if not self.tb.endswith("\n"):
            self.tb += "\n"

    def __str__(self):
        call = self.call
        if call is None:
            call = "<unknown R call>"
        return ("Error in %r: %s\nTraceback:\n%s"
                % (call, self.exc.message, self.tb))

def _r_repr(sexp):
    output = _capture_r_output(r["show"], sexp)
    while output.endswith("\n"):
        output = output[:-1]
    return output

def test__r_repr():
    assert _r_repr(ri.SexpVector([1, 2, 3], ri.INTSXP)) == "[1] 1 2 3"

class RWrapper(object):
    def __init__(self, sexp):
        self._sexp = sexp
        self.r = RBinder(self._sexp)

    # We'd just assign this in __init__ except that there ends up being a
    # circularity: the entries in the _NA_for_sexp_type dictionary are
    # themselves RWrappers/RArrays, and if we access _NA_for_sexp_type in
    # __init__ then we can't use __init__ to construct the entries in
    # _NA_for_sexp_type! Delaying computation of the .NA attribute until use
    # breaks this cycle:
    @property
    def NA(self):
        return _NA_for_sexp_type.get(self._sexp.typeof, NA_LOGICAL)

    # Make asarray() do the right thing (can happen for things like string
    # arrays, which do not get mapped to RArray):
    @property
    def __array_interface__(self):
        return self._sexp.__array_interface__

    def __as_r_sexp__(self):
        return self._sexp

    def __repr__(self):
        return _r_repr(self._sexp)

    def __len__(self):
        return len(self._sexp)

    def __getitem__(self, idx):
        item = self._sexp[idx]
        if isinstance(item, ri.Sexp):
            return ri2py(item)
        else:
            return item

def test_rwrapper():
    sexp = ri.SexpVector([1, 2, 3], ri.INTSXP)
    wrapper = RWrapper(sexp)
    assert repr(wrapper) == "[1] 1 2 3"
    assert r.is_na(wrapper.NA)
    assert repr(r.class_(wrapper.NA)) == '[1] "integer"'
    assert wrapper.__as_r_sexp__() is sexp
    assert isinstance(wrapper.r, RBinder)
    assert len(wrapper) == 3
    assert wrapper[1] == 2

class RClosure(RWrapper):
    def __init__(self, name, sexp):
        RWrapper.__init__(self, sexp)
        self._name = name

    def __call__(self, *args, **kwargs):
        rcall_args = [(None, value) for value in args]
        rcall_args += [(_munge_py_name_to_r(key), value)
                       for key, value in kwargs.iteritems()]
        return self.rcall(rcall_args)

    def rcall(self, args):
        "Takes an iterable like [(name, value), (name, value)]. names may be None."
        converted_args = tuple([(name, py2ri(value)) for name, value in args])
        try:
            return ri2py(self._sexp.rcall(converted_args),
                         "<return value from %s>" % (self._name,))
        except ri.RRuntimeError, e:
            raise RRuntimeError(e, self._name)

    def __repr__(self):
        return ("%s with name %s:\n%s"
                % (self.__class__.__name__, self._name, RWrapper.__repr__(self)))

    @property
    def __doc__(self):
        lookup = r.help(self._name)
        assert lookup.r.class_()[0] == "help_files_with_topic"
        if len(lookup) == 0:
            return "<no R help file found>"
        return open(lookup[0]).read()
        
def test_rclosure():
    sexp = ri.globalEnv.get("list")
    closure = RClosure("list", sexp)
    # Check __call__:
    frame1 = closure(1, k=2)
    assert repr(frame1) == "[[1]]\n[1] 1\n\n$k\n[1] 2"
    # Check rcall:
    frame2 = closure.rcall([("k", 2), (None, 1)])
    assert repr(frame2) == "$k\n[1] 2\n\n[[2]]\n[1] 1"
    # Check mangling of keyword arguments:
    frame3 = closure(class_=1, is_set=2)
    assert repr(frame3) == "$class\n[1] 1\n\n$is.set\n[1] 2"
    # Check docstring is loaded:
    assert "R Documentation" in closure.__doc__

def test_rclosure_error():
    stop = ri.globalEnv.get("stop")
    closure = RClosure("stop", stop)
    try:
        closure()
    except RRuntimeError, e:
        assert e.call == "stop"
    except:
        assert False
    else:
        assert False

class RBinder(object):
    def __init__(self, sexp):
        object.__setattr__(self, "_sexp", sexp)

    def __repr__(self):
        return _r_repr(object.__getattribute__(self, "_sexp"))

    # __getitem__ has a funny calling convention:
    #   foo[1] -> foo.__getitem__(1)
    #   foo[1, 2] -> foo.__getitem__((1, 2))
    # Note that it always receives a single argument, that may or may not be a
    # tuple!
    def __getitem__(self, args):
        if not isinstance(args, tuple):
            args = (args,)
        return self(*args)

    # This is functionally the same as [], but in Python [] cannot take
    # kwargs, while this can:
    def __call__(self, *args, **kwargs):
        sexp = object.__getattribute__(self, "_sexp")
        if sexp is None:
            raise ValueError, "this object has no corresponding R object"
        return r["["](sexp, *args, **kwargs)

    def __setitem__(self, *args, **kwargs):
        raise NotImplementedError, "R does not support mutating arbitrary arrays in place"

    def __getattribute__(self, name):
        name = _munge_py_name_to_r(name)
        sexp = object.__getattribute__(self, "_sexp")
        if sexp is None:
            raise NotImplementedError, "this object has no corresponding R object"
        return functools.partial(r[name], sexp)

def test_rbinder():
    sexp = ri.SexpVector([1, 2, 3, 4], ri.INTSXP)
    sexp.do_slot_assign("dim", ri.SexpVector([2, 2], ri.INTSXP))
    binder = RBinder(sexp)
    assert repr(binder) == _r_repr(sexp)
    # simple indexing:
    assert binder[1] == 1
    assert binder(1) == 1
    # multidimensional indexing:
    assert binder[1, 2] == 3
    # slice indexing:
    assert (binder[1, :] == [1, 3]).all()
    assert (binder[1, 1:2] == [1, 3]).all()
    # kwargs indexing:
    assert binder(1, 1, drop=True).shape == (1,)
    assert binder(1, 1, drop=False).shape == (1, 1)
    # "method" calls:
    assert (binder.dim() == np.array([2, 2])).all()
    
def _munge_py_name_to_r(name):
    # help.search("dollar") returns nothing, so I think we're safe here:
    if name == "dollar":
        name = "$"
    # Strip trailing underscores to let people use "class_" and "print_":
    if name.endswith("_"):
        name = name[:-1]
    # R names basically always use dots where Python names would use
    # underscores, so for a convenience API we might as well map the one to
    # the other:
    name = name.replace("_", ".")
    return name

def test__munge_py_name_to_r():
    assert _munge_py_name_to_r("dollar") == "$"
    assert _munge_py_name_to_r("class_") == "class"
    assert _munge_py_name_to_r("foo_bar_") == "foo.bar"

# This is a bit subtle, because ndarray's have tricky behavior wrt
# subclasses. We want to define a subclass that we can use to wrap an ri.Sexp,
# provide direct access to it, and keep track of the original ri.Sexp so that
# we if this array is passed to an R function then no further copying will be
# necessary.
# 
# However, we can't just say "instances of this subclass wrap R vectors, non-R
# vectors are represented by classic ndarray instances". This is because numpy
# magically makes sure that if you have an array subclass, then things like
# slices will automagically also be of that subclass. You cannot avoid
# this. There are three cases: 
#   1) A new array is constructed from scratch: __new__ is called (following
#      the usual Python rules). Basically this happens iff someone calls your
#      constructor explicitly -- someone has to actually type RArray(...).
#   2) A new array is constructed implicitly as a view on some other array:
#      numpy creates an object *that is an instance of your class, but one
#      without any of Python's normal object creation codepaths called on
#      it*. On this half-initialized object, then, it calls the method
#      __array_finalize__ (giving it the base array as an argument), to do any
#      constructor-ish stuff you want to do.
#   3) A new array is constructed implicitly as the result of a ufunc call:
#      this works similarly to (2), except that numpy calls __ufunc_wrap__
#      instead, and it gets a little more info on what the ufunc was. By
#      default __ufunc_wrap__ just calls __array_finalize__.
# (See http://docs.scipy.org/doc/numpy/user/basics.subclassing.html )
# The details don't matter so much for us -- the point is that we cannot hook
# into the actual memory allocation, and thus while we *will* end up with
# implicitly created instances of our class, we *cannot* construct them in "R
# space".
#
# Therefore, some RArray's exist in R space, and some do not. We distinguish
# these cases by the value of the _sexp attribute. __new__ sets _sexp to a
# non-None value. Implicitly created instances never have __new__ called, so
# their _sexp variable is never set. We set _sexp to None on the class object,
# so when _sexp is not set on an instance, attempting to access _sexp will
# fall back on the class value, i.e., it "defaults" to None.

class ArrayWrapImpossible(Exception):
    pass

def _set_sexp_shape(sexp, new_dim):
    # We don't convert 1-d (or 0-d) arrays to R arrays, but leave them as
    # vectors. This is because 1-d arrays in R are not very useful -- for
    # instance, if you have a length-n vector then matrix multiplication
    # will automatically coerce it to either nx1 or 1xn, but if you have a
    # length-n 1d array then matrix multiplication will not accept it at
    # all. Which breaks real code:
    if len(new_dim) < 2:
        return
    assert not sexp.named
    sexp.do_slot_assign("dim", ri.SexpVector(new_dim, ri.INTSXP))

class RArray(np.ndarray):
    _sexp = None
    _typeof = None

    def __new__(cls, sexp, sealed=True):
        assert isinstance(sexp, ri.Sexp)
        if sexp.typeof not in (ri.LGLSXP, ri.REALSXP, ri.INTSXP, ri.CPLXSXP):
            raise ArrayWrapImpossible
        # This calls the implicit view construction logic mentioned above to
        # do the actual creation:
        self = np.asarray(sexp).view(cls)
        self._sexp = sexp
        self._typeof = self._sexp.typeof
        self.r = RBinder(self._sexp)
        if sealed:
            self.seal()
        return self

    # In multi-array operations, prefer not to return the result as this type
    # (though many numpy operations ignore this):
    __array_priority__ = -1.0

    # Don't return our type from ufuncs:
    def __array_wrap__(self, a):
        return a

    def __array_finalize__(self, parent):
        if hasattr(parent, "_typeof") and not hasattr(self, "_typeof"):
            self._typeof = parent._typeof

    # See comment on RWrapper for why this is a property:
    @property
    def NA(self):
        return _NA_for_sexp_type[self._typeof]

    def is_r(self):
        return self._sexp is not None

    def is_sealed(self):
        return self.is_r() and not self.flags["WRITEABLE"]

    def seal(self):
        assert self.is_r(), "Attempt to seal an RArray view"
        self.flags["WRITEABLE"] = False

    def unseal(self, assert_no_copy=False):
        assert self.is_r(), "Attempt to unseal an RArray view"
        if self._sexp.named:
            assert self.is_sealed()
            assert not assert_no_copy, "Unsealing this RArray requires a copy, but assert_no_copy=True was given"
            sexp = self._sexp.duplicate()
        else:
            # Should just set self.flags["WRITEABLE"] = True in this case, but
            # that hits a numpy bug.
            sexp = self._sexp
        return self.__class__(sexp, sealed=False)

    def _replace_sexp(self, new_sexp):
        assert len(new_sexp) == len(self._sexp)
        assert new_sexp.typeof == self._sexp.typeof
        new_array = self.__class__(new_sexp)
        _nasty_hack_swap_view_targets(self, new_array)
        # new_array is now in an inconsistent state; let it go to a peaceful
        # grave.
        self._sexp = new_sexp

    def _hacky_unseal(self, assert_no_copy=False):
        assert self.is_r(), "Cannot unseal an RArray view"
        assert self.is_sealed()
        if self._sexp.named > 1:
            assert not assert_no_copy, "Unsealing this RArray requires a copy, but assert_no_copy=True was given"
            self._replace_sexp(self._sexp.__deepcopy__())
            assert self._sexp.named == 0
        _nasty_hack_though_not_nearly_so_bad_as_that_last_one_set_writeable(self)

    def __repr__(self):
        if not self.is_r():
            msg = "(R-style, view)"
        elif self.is_sealed():
            msg = "(R-style, sealed)"
        else:
            msg = "(R-style, UNsealed)"
        return "%s\n%s" % (np.ndarray.__repr__(np.asarray(self)), msg)

    def __as_r_sexp__(self):
        if self.is_r():
            assert self.is_sealed(), "Attempt to pass unsealed RArray into R; call .seal() first (or use an ordinary array)"
            return self._sexp
        else:
            return rarray(self, seal=True).__as_r_sexp__()

    def resize(self, new_shape):
        if self.is_sealed():
            raise NotImplementedError, "Cannot resize a sealed array"
        if self.is_r():
            _set_sexp_shape(self._sexp, new_shape)
        np.ndarray.resize(self, new_shape)

## Noted for potential future use:
import ctypes
class PyArrayObject(ctypes.Structure):
    _fields_ = [
        # PyObject_HEAD:
        ("ob_refcnt", ctypes.c_int64),
        ("ob_type", ctypes.c_void_p),
        # PyArrayObject fields, from numpy/ndarrayobject.h:
        # char *data;             /* pointer to raw data buffer */
        ("data", ctypes.c_void_p),
        # int nd;                 /* number of dimensions, also called
        #                            ndim */
        ("nd", ctypes.c_int),
        # npy_intp *dimensions;   /* size in each dimension */
        ("dimensions", ctypes.c_void_p),
        # npy_intp *strides;      /* bytes to jump to get to the
        #                            next element in each dimension */
        ("strides", ctypes.c_void_p),
        # PyObject *base;         /* This object should be decref'd
        #                            upon deletion of array */
        #                         /* For views it points to the original
        #                            array */
        #                         /* For creation from buffer object it
        #                            points to an object that shold be
        #                            decref'd on deletion */
        #                         /* For UPDATEIFCOPY flag this is an
        #                            array to-be-updated upon deletion of
        #                            this one */
        ("base", ctypes.py_object),
        # PyArray_Descr *descr;   /* Pointer to type structure */
        ("descr", ctypes.c_void_p),
        # int flags;              /* Flags describing array -- see below*/
        ("flags", ctypes.c_int),
        # PyObject *weakreflist;  /* For weakreferences */
        ("weakreflist", ctypes.c_void_p),
        ]
def _sanity_check_ctypes_hack(a):
    # Sanity checks to guard against, e.g., our structure definition above
    # going out-of-sync with numpy:
    ct_a = PyArrayObject.from_address(id(a))
    assert id(ct_a.base) == id(a.base)
    assert a.ctypes.data == ct_a.data
    
# If you have two ndarrays, 'a' and 'b', and they are each a view on "similar"
# underlying arrays, then this swaps them around so that 'a' becomes a view on
# 'b.base' and 'b' becomes a view on 'a.base'. Swapping instead of
# simple-assignment avoids a lot of ref-counting problems. (I bet you can
# still break ref-counting with this, though, if you're not careful.)
def _nasty_hack_swap_view_targets(a, b):
    assert a.size == b.size
    assert a.dtype == b.dtype
    assert a.shape == b.shape
    assert a.strides == b.strides
    assert a.base is not None
    assert b.base is not None
    _sanity_check_ctypes_hack(a)
    ct_a = PyArrayObject.from_address(id(a))
    ct_b = PyArrayObject.from_address(id(b))
    # Swap the base objects:
    tmp = ct_a.base
    ct_a.base = ct_b.base
    ct_b.base = tmp
    # Swap the data pointer:
    tmp = ct_a.data
    ct_a.data = ct_b.data
    ct_b.data = tmp

# Work around an obnoxious bug in numpy where if you set an array
# non-WRITEABLE, and it is (eventually) a view on a non-numpy array, you
# cannot set it back again:
def _nasty_hack_though_not_nearly_so_bad_as_that_last_one_set_writeable(a):
    _sanity_check_ctypes_hack(a)
    WRITEABLE = 0x400
    assert not a.flags["WRITEABLE"]
    assert a.base.flags["WRITEABLE"]
    ct_a = PyArrayObject.from_address(id(a))
    ct_a.flags |= WRITEABLE
    assert a.flags["WRITEABLE"]

def ri2py(obj, name):
    assert isinstance(obj, ri.Sexp)
    if isinstance(obj, ri.SexpClosure):
        return RClosure(name, obj)
    if isinstance(obj, ri.SexpVector):
        try:
            return RArray(obj)
        except ArrayWrapImpossible:
            pass
    return RWrapper(obj)

# The possible kind codes are listed at
#   http://numpy.scipy.org/array_interface.shtml
_kind_to_sexp = {
    # "t" -> not really supported by numpy
    "b": ri.LGLSXP,
    "i": ri.INTSXP,
    # "u" -> special-cased below
    "f": ri.REALSXP,
    "c": ri.CPLXSXP,
    "S": ri.STRSXP,
    "U": ri.STRSXP,
    # "V" -> special-cased below
    "O": ri.VECSXP,
    }
def _arraylike_to_sexp(object, recurse, **kwargs):
    if isinstance(object, ri.Sexp):
        assert not kwargs
        return object
    array = np.asarray(object, **kwargs)
    sexp_seq = array.ravel("F")
    shape = array.shape
    if array.dtype.kind == "O":
        if not recurse:
            raise ValueError, "Cannot convert object arrays without recursing"
        else:
            sexp_seq = map(py2ri, sexp_seq)
    # Most types map directly to R arrays (or in one case an R list):
    if array.dtype.kind in _kind_to_sexp:
        sexp = ri.SexpVector(sexp_seq,
                             _kind_to_sexp[array.dtype.kind])
        _set_sexp_shape(sexp, shape)
        return sexp
    # R does not support unsigned types:
    elif array.dtype.kind == "u":
        raise ValueError, "Cannot convert numpy array of unsigned values -- R does not have unsigned integers."
    # Record arrays map onto R data frames:
    elif array.dtype.kind == "V":
        if len(array.shape) != 1:
            raise ValueError, "Only unidimensional record arrays can be converted to data frames"
        if array.dtype.names is None:
            raise ValueError, "Cannot convert void array of type %r to data.frame -- it has no field names"
        df_args = []
        for field_name in array.dtype.names:
            df_args.append((field_name, py2ri(array[field_name])))
        # XX FIXME: data.frame returns a data frame with .named == 2, even
        # though in fact we know it is not aliased anywhere else. So it might
        # be worth hacking its NAMED value back:
        return ri.baseNameSpaceEnv["data.frame"].rcall(tuple(df_args))
    # It should be impossible to get here:
    else:
        raise ValueError, "Unknown numpy array type."

def test__arraylike_to_sexp():
    # Existing sexps are passed through without change:
    sexp = ri.SexpVector([10], ri.INTSXP)
    assert _arraylike_to_sexp(sexp, False) is sexp
    # Types map through to the right kind of SEXP:
    s = _arraylike_to_sexp(np.array([True], dtype=bool), False)
    assert s.typeof == ri.LGLSXP
    assert s[0] == True
    s = _arraylike_to_sexp(np.array([1], dtype=int), False)
    assert s.typeof == ri.INTSXP
    assert s[0] == 1
    s = _arraylike_to_sexp(np.array([1], dtype=float), False)
    assert s.typeof == ri.REALSXP
    assert s[0] == 1.0
    s = _arraylike_to_sexp(np.array([1 + 2j], dtype=complex), False)
    assert s.typeof == ri.CPLXSXP
    assert s[0] == 1 + 2j
    s = _arraylike_to_sexp(np.array(["hi"], dtype=str), False)
    assert s.typeof == ri.STRSXP
    assert s[0] == "hi"
    s = _arraylike_to_sexp(np.array(["hi"], dtype=unicode), False)
    assert s.typeof == ri.STRSXP
    assert s[0] == "hi"
    s = _arraylike_to_sexp(np.array(["hi"], dtype=object), True)
    assert s.typeof == ri.VECSXP
    assert s[0][0] == "hi"
    try:
        _arraylike_to_sexp(np.array(["hi"], dtype=object), False)
    except ValueError:
        pass
    else:
        assert False

    # Record array:
    dt = np.dtype([("a", int), ("b", float), ("c", bool)])
    ra = np.array([(1, 3.2, True), (-10, 0.7, False)], dtype=dt)
    frame = rcopy(ra)
    assert frame.r.class_()[0] == "data.frame"
    assert frame.r.nrow()[0] == 2
    assert list(frame.r.names()) == ["a", "b", "c"]
    a = frame.r.dollar("a")
    assert a[0] == 1 and a[1] == -10
    b = frame.r.dollar("b")
    assert b[0] == 3.2 and b[1] == 0.7
    c = frame.r.dollar("c")
    assert c[0] == 1 and c[1] == 0
    
    # Type autodetection:
    assert _arraylike_to_sexp([1], False).typeof == ri.INTSXP
    assert _arraylike_to_sexp([True], False).typeof == ri.LGLSXP
    assert _arraylike_to_sexp([0.1], False).typeof == ri.REALSXP
    assert _arraylike_to_sexp([1j], False).typeof == ri.CPLXSXP
    assert _arraylike_to_sexp(["asdf", "hi"], False).typeof == ri.STRSXP

def _iterable(obj):
    try:
        iter(obj)
    except TypeError:
        return False
    else:
        return True

def test__iterable():
    assert _iterable([])
    assert _iterable((1, 2, 3))
    assert not _iterable(1)
    assert not _iterable(None)
    def generator():
        yield None
    assert not _iterable(generator)
    assert _iterable(generator())

def py2ri(obj):
    if isinstance(obj, ri.Sexp):
        return obj

    if hasattr(obj, "__as_r_sexp__"):
        return obj.__as_r_sexp__()

    if isinstance(obj, dict):
        return r["list"].rcall(obj.iteritems()).__as_r_sexp__()

    if isinstance(obj, (list, tuple, np.ndarray)):
        return _arraylike_to_sexp(obj, True)

    # See if numpy recognizes it (this catches lots of basic python types
    # like int and bool, plus more exotic things like numpy.float32's,
    # etc.). But disable recursion, because if it *isn't* recognized then
    # _arraylike_to_sexp will call back to us to try and convert the
    # individual items...
    try:
        return _arraylike_to_sexp([obj], False)
    except ValueError:
        pass

    # This needs to after the "does numpy recognize it?" check, because
    # otherwise it will catch things like strings:
    if _iterable(obj):
        return _arraylike_to_sexp(list(obj), True)

    if obj is None:
        return NULL.__as_r_sexp__()

    if isinstance(obj, slice):
        # Simple ":" maps to R_MissingArg, because in R, leaving out an
        # indexing argument means "everything":
        if obj == slice(None):
            #assert hasattr(ri, "getMissingArgSexp")
            #return ri.getMissingArgSexp()
            # But by the recycling rule, simple TRUE also means "everything":
            return r("TRUE").__as_r_sexp__()
        # Otherwise, though, we only accept the slice syntax supported by R,
        # i.e., start:stop exactly.
        if obj.step not in (None, 1) or obj.start is None or obj.stop is None:
            raise (ValueError,
                   "R-style slices must be either ':' or 'start:stop'")
        return r.seq_int(obj.start, obj.stop).__as_r_sexp__()

    raise ValueError, "cannot convert object %r to R" % (obj,)

def rarray(object, **kwargs):
    seal = kwargs.get("seal", False)
    if "seal" in kwargs:
        del kwargs["seal"]
    array = RArray(_arraylike_to_sexp(object, True, **kwargs), sealed=seal)
    return array

def _r_filled_array(filler, shape, dtype, seal):
    if np.dtype(dtype).kind == "f":
        filler = r.as_numeric(filler)
    elif np.dtype(dtype).kind == "i":
        filler = r.as_integer(filler)
    elif np.dtype(dtype).kind == "b":
        filler = r.as_logical(filler)
    elif np.dtype(dtype).kind == "c":
        filler = r.as_complex(filler)
    else:
        raise ValueError, "Not sure how to make an array of type %r" % (dtype,)
    array = r.rep_int(filler, np.asarray(shape).prod())
    array = array.unseal(assert_no_copy=True)
    array.resize(shape)
    if seal:
        array.seal()
    return array

def rzeros(shape, dtype=float, seal=False):
    return _r_filled_array(0, shape, dtype, seal)

def rones(shape, dtype=float, seal=False):
    return _r_filled_array(1, shape, dtype, seal)

# These do immediate calls into all the machinery above, so we can't do them
# until all the definitions have been executed, i.e. they have to go at the
# end of the file:
NULL = r("NULL")

_NA_for_sexp_type = {}
NA_NUMERIC = r("as.numeric(NA)")
_NA_for_sexp_type[ri.REALSXP] = NA_NUMERIC
NA_COMPLEX = r("as.complex(NA)")
_NA_for_sexp_type[ri.CPLXSXP] = NA_COMPLEX
NA_INTEGER = r("as.integer(NA)")
_NA_for_sexp_type[ri.INTSXP] = NA_INTEGER
NA_LOGICAL = r("as.logical(NA)")
_NA_for_sexp_type[ri.LGLSXP] = NA_LOGICAL
NA_CHARACTER = r("as.character(NA)")
_NA_for_sexp_type[ri.STRSXP] = NA_CHARACTER

def NA_for(obj):
    # Don't copy the whole thing:
    if _iterable(obj):
        obj = iter(obj).next()
    return rcopy(obj).NA

import threading
import time
_r_polling_thread = None
class RPollerThread(threading.Thread):
    def __init__(self, hz):
        threading.Thread.__init__(self, name="R-polling-thread")
        self.setDaemon(True)
        self._want_run = True
        self.set_hz(hz)

    def stop(self):
        self._want_run = False

    def set_hz(self, hz):
        self._sleep_time = 1. / hz

    def run(self):
        while self._want_run:
            time.sleep(self._sleep_time)
            try:
                ri.process_revents()
            except:
                print "error in process_revents: ignored"

def set_interactive(interactive=True, hz=10):
    global _r_polling_thread
    if interactive:
        if _r_polling_thread is None:
            _r_polling_thread = RPollerThread(hz)
            _r_polling_thread.start()
        _r_polling_thread.set_hz(hz)
    else:
        if _r_polling_thread is not None:
            _r_polling_thread.stop()
            _r_polling_thread.join()
            _r_polling_thread = None

if __name__ == "__main__":
    import nose
    nose.runmodule()
