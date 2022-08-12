import tensorflow as tf


def debug(fn_name, do_debug=False):
    if do_debug:
        print("Tracing", fn_name)
        tf.print("Executing", fn_name)
