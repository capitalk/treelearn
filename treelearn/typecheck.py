
def check_type(x, t):
    if not isinstance(x,t):
        msg = "Expected %s : %s to be %s" % (x, type(x), t)
        raise RuntimeError(msg)

def check_field(x,f):
    if not hasattr(x,f):
        msg = "Expected %s : %s to have field %s" % (x, type(x), f)
        raise RuntimeError(msg)

def check_fields(x,fs):
    for f in fs:
        check_field(x,f)

def check_estimator(x):
    check_fields(x, ['fit', 'predict'])

def check_int(x):
    check_type(x, int)

def check_bool(x):
    check_type(x, bool)

def check_dict(x):
    check_type(x, dict)
