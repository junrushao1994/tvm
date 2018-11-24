import tvm
import os
from tvm.contrib import nvcc
import numpy as np


# Quick knobs
TASK="lstm"
USE_MANUAL_CODE = False
PERSIST_KERNEL = True
DETECT_GLOBAL_BARRIER = PERSIST_KERNEL
SKIP_CHECK = False
UNROLL_WLOAD = True
NUM_SM = 60   # 80 streaming processors in V100

NUM_THREAD_Y = 8
NUM_THREAD_X = 16 * 3 // 2


@tvm.register_func
def tvm_callback_cuda_compile(code):
    """Use nvcc compiler for better perf."""
    ptx =  nvcc.compile_cuda(code, target="ptx")
    return ptx


@tvm.register_func
def tvm_callback_cuda_postproc(code):
    if not os.path.exists("perf"):
        os.mkdir("perf")
    with open("perf/%s_generated.cu" % TASK, "w") as f:
        f.write(code)
    if USE_MANUAL_CODE:
        code = open("perf/%s_manual.cu" % TASK).read()
    return code


def _linear(x, w, b, name):
    # x: [seq_len, batch_size, input_dim]
    # w: [num_gemm, num_hidden, input_dim]
    # b: [num_gemm, num_hidden]
    # wxb: [seq_len, batch_size, num_gemm, num_hidden]
    # k: IterVar(min=0, extent=input_dim)
    assert w is not None or b is not None
    seq_len = x.shape[0]
    batch_size = x.shape[1]
    input_dim = x.shape[2]
    num_gemm = (w if w is not None else b).shape[0]
    num_hidden = (w if w is not None else b).shape[1]
    if w is None:
        k = None
        wx = tvm.compute((seq_len, batch_size, num_gemm, num_hidden),
                          lambda t, m, a, j: x[t, m, j],
                          name=name + "_w")
    else:
        k = tvm.reduce_axis((0, input_dim), name=name + "_k")
        wx = tvm.compute((seq_len, batch_size, num_gemm, num_hidden),
                          lambda t, m, a, j: tvm.sum(w[a, j, k] * x[t - 1, m, k], axis=k),
                          name=name + "_w")
    if b is None:
        wxb = tvm.compute((seq_len, batch_size, num_gemm, num_hidden),
                           lambda t, m, a, j: wx[t, m, a, j],
                           name=name + "_b")
    else:
        wxb = tvm.compute((seq_len, batch_size, num_gemm, num_hidden),
                           lambda t, m, a, j: wx[t, m, a, j] + b[a, j],
                           name=name + "_b")
    return wxb, k


def gru(n_seq_len=128, n_num_hidden=128, n_input_dim=128, n_batch_size=1):
    # This GRU includes input projection and all bias terms, should exactly match that of cuDNN
    seq_len = tvm.var("seq_len")
    batch_size = tvm.var("batch_size")
    num_hidden = tvm.var("num_hidden")
    input_dim = tvm.var("input_dim")
    # Define the input
    x = tvm.placeholder((seq_len, batch_size, input_dim), name="x")
    # Define weight and bias for input projection and hidden state transformation
    w_i2h = tvm.placeholder((3, num_hidden, input_dim), name="w_i2h")
    b_i2h = tvm.placeholder((3, num_hidden), name="b_h2h")
    w_h2h = tvm.placeholder((3, num_hidden, num_hidden), name="w_h2h")
    b_h2h = tvm.placeholder((3, num_hidden), name="b_h2h")
    # Define hidden state and cell state
    s_h = tvm.placeholder((seq_len, batch_size, num_hidden), name="s_h")
    s_h_init = tvm.compute((1, batch_size, num_hidden), lambda *i: 0.0, name="s_h_init")
    # Do the transformation, the bias for input projection can be fused into that of hidden transformation
    l_i2h, k_i2h = _linear(x, w_i2h, b_i2h, name="l_i2h")
    l_h2h, k_h2h = _linear(s_h, w_h2h, b_h2h, name="l_h2h")
    # Computation inside an GRU cell
    g_r = tvm.compute((seq_len, batch_size, num_hidden), lambda t, i, j: l_i2h[t, i, 0, j] + l_h2h[t, i, 0, j], name="g_r")
    g_i = tvm.compute((seq_len, batch_size, num_hidden), lambda t, i, j: l_i2h[t, i, 1, j] + l_h2h[t, i, 1, j], name="g_i")
    g_h = tvm.compute((seq_len, batch_size, num_hidden), lambda t, i, j: l_i2h[t, i, 2, j] + g_r[t, i, j] * l_h2h[t, i, 2, j], name="g_h")
    n_h = tvm.compute((seq_len, batch_size, num_hidden), lambda *i: (1.0 - g_i(*i)) * g_h(*i) + g_i(*i) * s_h(*i), name="n_h")
    # Define update rules explicitly
    u_h = tvm.compute((seq_len, batch_size, num_hidden), lambda *i: n_h(*i), name="u_h")
    # Finally, define the scanning itself
    scan_h = tvm.scan(
        init=[s_h_init],
        update=[u_h],
        state_placeholder=[s_h],
        inputs=[x],
        name="gru_scan")
    # schedule
    s = tvm.create_schedule(scan_h.op)


def lstm(n_seq_len=128, n_num_hidden=128, n_input_dim=128, n_batch_size=1):
    # This LSTM includes input projection and all bias terms, should exactly match that of cuDNN
    seq_len = tvm.var("seq_len")
    batch_size = tvm.var("batch_size")
    num_hidden = tvm.var("num_hidden")
    input_dim = tvm.var("input_dim")
    # Define the input
    x = tvm.placeholder((seq_len, batch_size, input_dim), name="x")
    # Define weight and bias for input projection and hidden state transformation
    w_i2h = tvm.placeholder((4, num_hidden, input_dim), name="w_i2h")
    w_h2h = tvm.placeholder((4, num_hidden, num_hidden), name="w_h2h")
    b_h2h = tvm.placeholder((4, num_hidden), name="b_h2h")
    # Define hidden state and cell state
    s_h = tvm.placeholder((seq_len, batch_size, num_hidden), name="s_h")
    s_c = tvm.placeholder((seq_len, batch_size, num_hidden), name="s_c")
    s_h_init = tvm.compute((1, batch_size, num_hidden), lambda *i: 0.0, name="s_h_init")
    s_c_init = tvm.compute((1, batch_size, num_hidden), lambda *i: 0.0, name="s_c_init")
    # Do the transformation, the bias for input projection can be fused into that of hidden transformation
    l_i2h, k_i2h = _linear(x, w_i2h, None, name="l_i2h")
    l_h2h, k_h2h = _linear(s_h, w_h2h, b_h2h, name="l_h2h")
    # Sum up the transformed inputs and previous hidden states
    g = tvm.compute((seq_len, batch_size, 4, num_hidden), lambda *i: l_i2h(*i) + l_h2h(*i), name="g")
    # Computation inside an LSTM cell
    g_i = tvm.compute((seq_len, batch_size, num_hidden), lambda t, i, j: tvm.sigmoid(g[t, i, 0, j]), name="g_i")
    g_f = tvm.compute((seq_len, batch_size, num_hidden), lambda t, i, j: tvm.sigmoid(g[t, i, 1, j]), name="g_f")
    g_c = tvm.compute((seq_len, batch_size, num_hidden), lambda t, i, j: tvm.tanh   (g[t, i, 2, j]), name="g_c")
    g_o = tvm.compute((seq_len, batch_size, num_hidden), lambda t, i, j: tvm.sigmoid(g[t, i, 3, j]), name="g_o")
    n_c = tvm.compute((seq_len, batch_size, num_hidden), lambda t, i, j: g_f[t, i, j] * s_c[t - 1, i, j] + g_i[t, i, j] * g_c[t, i, j], name="n_c")
    n_h = tvm.compute((seq_len, batch_size, num_hidden), lambda t, i, j: g_o[t, i, j] * tvm.tanh(n_c[t, i, j]), name="n_h")
    # Define update rules explicitly
    u_h = tvm.compute((seq_len, batch_size, num_hidden), lambda *i: n_h(*i), name="u_h")
    u_c = tvm.compute((seq_len, batch_size, num_hidden), lambda *i: n_c(*i), name="u_c")
    # Finally, define the scanning itself
    scan_h, scan_c = tvm.scan(
        init=[s_h_init, s_c_init],
        update=[u_h, u_c],
        state_placeholder=[s_h, s_c],
        inputs=[x],
        name="lstm_scan")
    # schedule
    s = tvm.create_schedule([scan_h.op, scan_c.op])


def main():
    gru()
    lstm()


if __name__ == "__main__":
    main()