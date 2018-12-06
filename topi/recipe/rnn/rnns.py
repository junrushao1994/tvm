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
  # b: None or [num_gemm, num_hidden]
  # k: IterVar(min=0, extent=input_dim)
  assert w is not None
  seq_len = x.shape[0]
  batch_size = x.shape[1]
  input_dim = x.shape[2]
  num_gemm = w.shape[0]
  num_hidden = w.shape[1]

  k = tvm.reduce_axis((0, input_dim), name=name + "_k")
  if b is None:
    wx = tvm.compute((seq_len, batch_size, num_gemm, num_hidden),
                      lambda t, n, a, c: tvm.sum(w[a, c, k] * x[t - 1, n, k], axis=k),
                      name=name + "_w")
  else:
    wx = tvm.compute((seq_len, batch_size, num_gemm, num_hidden),
                      lambda t, n, a, c: tvm.sum(w[a, c, k] * x[t - 1, n, k] + b[a][c], axis=k),
                      name=name + "_w")
  return wx, k


def vanilla(n_seq_len=128, n_num_hidden=128, n_input_dim=128, n_batch_size=1):
  # This vanilla RNN includes input projection and all bias terms, should exactly match that of cuDNN
  seq_len = tvm.var("seq_len")
  batch_size = tvm.var("batch_size")
  num_hidden = tvm.var("num_hidden")
  input_dim = tvm.var("input_dim")
  # Define the input
  x = tvm.placeholder((seq_len, batch_size, input_dim), name="x")
  # Define weight and bias for input projection and hidden state transformation
  w_i2h = tvm.placeholder((1, num_hidden, input_dim), name="w_i2h")
  b_i2h = tvm.placeholder((1, num_hidden), name="b_i2h")
  w_h2h = tvm.placeholder((1, num_hidden, num_hidden), name="w_h2h")
  b_h2h = tvm.placeholder((1, num_hidden), name="b_h2h")
  # Define hidden state and cell state
  s_h = tvm.placeholder((seq_len, batch_size, num_hidden), name="s_h")
  s_h_init = tvm.compute((1, batch_size, num_hidden), lambda *i: 0.0, name="s_h_init")
  # Do the transformation, the bias for input projection can be fused into that of hidden transformation
  bias = tvm.compute((1, num_hidden), lambda *i: b_i2h(*i) + b_h2h(*i), name="bias")
  l_i2h, k_i2h = _linear(x, w_i2h, None, name="l_i2h")
  l_h2h, k_h2h = _linear(s_h, w_h2h, bias, name="l_h2h")
  # Sum up the transformed inputs and previous hidden states
  g = tvm.compute((seq_len, batch_size, 1, num_hidden), lambda *i: l_i2h(*i) + l_h2h(*i), name="g")
  # Computation inside an LSTM cell
  n_h = tvm.compute((seq_len, batch_size, num_hidden), lambda t, n, c: tvm.tanh(g[t, n, 0, c]), name="g_i")
  # Define update rules explicitly
  u_h = tvm.compute((seq_len, batch_size, num_hidden), lambda *i: n_h(*i), name="u_h")
  # Finally, define the scanning itself
  scan_h = tvm.scan(
    init=[s_h_init],
    update=[u_h],
    state_placeholder=[s_h],
    inputs=[x],
    name="lstm_scan")
  # schedule
  s = tvm.create_schedule([scan_h.op])


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
  g_r = tvm.compute((seq_len, batch_size, num_hidden), lambda t, n, c: l_i2h[t, n, 0, c] + l_h2h[t, n, 0, c], name="g_r")
  g_i = tvm.compute((seq_len, batch_size, num_hidden), lambda t, n, c: l_i2h[t, n, 1, c] + l_h2h[t, n, 1, c], name="g_i")
  g_h = tvm.compute((seq_len, batch_size, num_hidden), lambda t, n, c: l_i2h[t, n, 2, c] + g_r[t, n, c] * l_h2h[t, n, 2, c], name="g_h")
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
  b_i2h = tvm.placeholder((4, num_hidden), name="b_i2h")
  w_h2h = tvm.placeholder((4, num_hidden, num_hidden), name="w_h2h")
  b_h2h = tvm.placeholder((4, num_hidden), name="b_h2h")
  # Define hidden state and cell state
  s_h = tvm.placeholder((seq_len, batch_size, num_hidden), name="s_h")
  s_c = tvm.placeholder((seq_len, batch_size, num_hidden), name="s_c")
  s_h_init = tvm.compute((1, batch_size, num_hidden), lambda *i: 0.0, name="s_h_init")
  s_c_init = tvm.compute((1, batch_size, num_hidden), lambda *i: 0.0, name="s_c_init")
  # Do the transformation, the bias for input projection can be fused into that of hidden transformation
  bias = tvm.compute((1, num_hidden), lambda *i: b_i2h(*i) + b_h2h(*i), name="bias")
  l_i2h, k_i2h = _linear(x, w_i2h, None, name="l_i2h")
  l_h2h, k_h2h = _linear(s_h, w_h2h, bias, name="l_h2h")
  # Sum up the transformed inputs and previous hidden states
  g = tvm.compute((seq_len, batch_size, 4, num_hidden), lambda *i: l_i2h(*i) + l_h2h(*i), name="g")
  # Computation inside an LSTM cell
  g_i = tvm.compute((seq_len, batch_size, num_hidden), lambda t, n, c: tvm.sigmoid(g[t, n, 0, c]), name="g_i")
  g_f = tvm.compute((seq_len, batch_size, num_hidden), lambda t, n, c: tvm.sigmoid(g[t, n, 1, c]), name="g_f")
  g_c = tvm.compute((seq_len, batch_size, num_hidden), lambda t, n, c: tvm.tanh   (g[t, n, 2, c]), name="g_c")
  g_o = tvm.compute((seq_len, batch_size, num_hidden), lambda t, n, c: tvm.sigmoid(g[t, n, 3, c]), name="g_o")
  n_c = tvm.compute((seq_len, batch_size, num_hidden), lambda t, n, c: g_f[t, n, c] * s_c[t - 1, n, c] + g_i[t, n, c] * g_c[t, n, c], name="n_c")
  n_h = tvm.compute((seq_len, batch_size, num_hidden), lambda t, n, c: g_o[t, n, c] * tvm.tanh(n_c[t, n, c]), name="n_h")
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
  vanilla()
  gru()
  lstm()


if __name__ == "__main__":
  main()
