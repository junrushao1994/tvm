import tvm
import os
from tvm.contrib import nvcc
import numpy as np


# Quick knobs
TASK="rnn"
USE_MANUAL_CODE = False
DETECT_GLOBAL_BARRIER = True
UNROLL = True
NUM_SM = 80   # 80 streaming processors in V100

NUM_THREAD_Y = 32
NUM_THREAD_X = 4


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


def _input(seq_len, batch_size, input_dim, name, **kwargs):
  x = tvm.placeholder((seq_len, batch_size, input_dim), name=name)
  def sch(s, scan, readers, n_tx, n_ty, tid_x, tid_y, **kwargs):
    # fetch inputs
    xL = s.cache_read(x, "shared", readers=readers)
    s[xL].compute_at(s[scan], s[scan].op.scan_axis)
    # split `nc` into threadIdx.y * threadIdx.x
    _, n, c = s[xL].op.axis
    nc = s[xL].fuse(n, c)
    _, xi = s[xL].split(nc, factor=n_ty * n_tx)
    ty, xi = s[xL].split(xi, nparts=n_ty)
    tx, _ = s[xL].split(xi, nparts=n_tx)
    # bind with threads
    s[xL].bind(ty, tid_y)
    s[xL].bind(tx, tid_x)
    s[xL].double_buffer()
  return (x, ), sch


def _state(seq_len, batch_size, num_hidden, name, **kwargs):
  state = tvm.placeholder((seq_len, batch_size, num_hidden), name=name)
  s_init = tvm.compute((1, batch_size, num_hidden), lambda _, n, c: 0.0, name="init_" + name)
  def sch(s, scan, readers, **kwargs):
    def _state(n_ty, n_tx, tid_y, tid_x, **kwargs):
      # fetching the state
      sS = s.cache_read(state, "shared", readers=readers)
      s[sS].compute_at(s[scan], s[scan].op.scan_axis)
      # split `nc` into threadIdx.y * threadIdx.x
      _, n, c = s[sS].op.axis
      nc = s[sS].fuse(n, c)
      _, xi = s[sS].split(nc, factor=n_ty * n_tx)
      ty, xi = s[sS].split(xi, nparts=n_ty)
      tx, _ = s[sS].split(xi, nparts=n_tx)
      # bind with threads
      s[sS].bind(ty, tid_y)
      s[sS].bind(tx, tid_x)
      s[sS].double_buffer()
    def _s_init(n_bx, n_tx, bid_x, tid_x, tid_y, **kwargs):
      # split `nc` into blockIdx.x * threadIdx.x
      _, n, c = s[s_init].op.axis
      nc = s[s_init].fuse(n, c)
      _, xi = s[s_init].split(nc, factor=n_bx * n_tx)
      bx, xi = s[s_init].split(xi, nparts=n_bx)
      tx, _ = s[s_init].split(xi, nparts=n_tx)
      # bind with threads
      s[s_init].bind(bx, bid_x)
      s[s_init].bind(tx, tid_x)
      s[s_init].set_store_predicate(tid_y.equal(0))
    _state(**kwargs)
    _s_init(**kwargs)
  return (state, s_init), sch


def _linear(x, num_gemm, num_hidden, name, **kwargs):
  seq_len, batch_size, input_dim = x.shape
  w = tvm.placeholder((num_gemm, num_hidden, input_dim), name="w_" + name)
  b = tvm.placeholder((num_gemm, num_hidden), name="b_" + name)
  k = tvm.reduce_axis((0, input_dim), name="k_" + name)
  g = tvm.compute((seq_len, batch_size, num_gemm, num_hidden),
                   lambda t, n, a, c: tvm.sum(x[t - 1, n, k] * w[a, c, k], axis=k),
                   name="g_" + name)
  f = tvm.compute((seq_len, batch_size, num_gemm, num_hidden),
                   lambda t, n, a, c: g[t, n, a, c] + b[a, c],
                   name=name)
  def sch(s, scan, **kwargs):
    def _w(bid_x, tid_y, tid_x, n_bx, n_ty, n_tx, **kwargs):
      # w[a, c, i] => (blockIdx.x, *, threadIdx.x)``
      wS = s.cache_read(w, "shared", readers=[g])
      s[wS].compute_at(s[scan], tid_x)
      a, c, i = s[wS].op.axis
      ac = s[wS].fuse(a, c)
      bx, _ = s[wS].split(ac, nparts=n_bx)
      _, tx = s[wS].split(i, factor=n_tx)
      s[wS].bind(bx, bid_x)
      s[wS].bind(tx, tid_x)
      s[wS].set_store_predicate(tid_y.equal(0))
    def _b(bid_x, tid_y, tid_x, n_bx, n_ty, n_tx, **kwargs):
      # b[a, c] => (blockIdx.x, *, threadIdx.x)
      bS = s.cache_read(b, "shared", readers=[f])
      s[bS].compute_at(s[scan], tid_x)
      a, c = s[bS].op.axis
      ac = s[bS].fuse(a, c)
      bx, _ = s[bS].split(ac, nparts=n_bx)
      s[bS].bind(bx, bid_x)
      s[bS].set_store_predicate(tvm.all(tid_x.equal(0), tid_y.equal(0)))
    def _mm(bid_x, tid_y, tid_x, n_bx, n_ty, n_tx, **kwargs):
      # inline the bias add
      s[f].compute_inline()
      # do reduction on the last dimension
      k, = s[g].op.reduce_axis
      ko, _ = s[g].split(k, nparts=n_ty)
      rf = s.rfactor(g, ko)
      k, = s[g].op.reduce_axis
      s[rf].compute_at(s[g], k)
      # bind with threads
      _, _, a, c = s[g].op.axis
      ac = s[g].fuse(a, c)
      bx, _ = s[g].split(ac, nparts=n_bx)
      ty, = s[g].op.reduce_axis
      s[g].bind(bx, bid_x)
      s[g].bind(ty, tid_y)
      s[g].set_store_predicate(tid_x.equal(0))
    _w(**kwargs)
    _b(**kwargs)
    _mm(**kwargs)
  return (f, g, w, b), sch


def _update_state(x, seq_len, batch_size, num_hidden, name, **kwargs):
  u = tvm.compute((seq_len, batch_size, num_hidden), lambda t, n, c: x[t, n, c], name=name)
  def sch(s, n_bx, n_tx, bid_x, tid_x, tid_y, **kwargs):
    s[x].compute_inline()
    _, _, c = s[u].op.axis
    _, xi = s[u].split(c, factor=n_bx * n_tx)
    bx, xi = s[u].split(xi, nparts=n_bx)
    tx, _ = s[u].split(xi, nparts=n_tx)
    # bind with threads
    s[u].bind(bx, bid_x)
    s[u].bind(tx, tid_x)
    s[u].set_store_predicate(tid_y.equal(0))
  return (u, ), sch


def vanilla(n_seq_len=128, n_num_hidden=128, n_input_dim=128, n_batch_size=1):
  n_seq_len += 1
  seq_len = tvm.convert(n_seq_len)
  batch_size = tvm.convert(n_batch_size)
  num_hidden = tvm.convert(n_num_hidden)
  input_dim = tvm.convert(n_input_dim)
  # batch_size = tvm.var("batch_size")
  # num_hidden = tvm.var("num_hidden")
  # input_dim = tvm.var("input_dim")
  config = {
    "seq_len": seq_len,
    "batch_size": batch_size,
    "num_hidden": num_hidden,
    "input_dim": input_dim,
    "num_gemm": 1,
  }
  # Define weight and bias for input projection and hidden state transformation
  (x_i, ), sch_x_i = _input(name="x_i", **config)
  (s_h, init_s_h), sch_s_h = _state(name="s_h", **config)
  (l_i2h, g_i2h, w_i2h, b_i2h), sch_i2h = _linear(x_i, name="l_i2h", **config)
  (l_h2h, g_h2h, w_h2h, b_h2h), sch_h2h = _linear(s_h, name="l_h2h", **config)
  # Computation inside the RNN cell
  n_h = tvm.compute((seq_len, batch_size, num_hidden), lambda t, n, c: tvm.tanh(l_i2h[t, n, 0, c] + l_h2h[t, n, 0, c]), name="n_h")
  # Define update rules explicitly
  (u_h, ), sch_u_h = _update_state(n_h, name="u_h", **config)
  # Finally, define the scanning itself
  scan_h = tvm.scan(
    state_placeholder=[s_h],
    init=[init_s_h],
    update=[u_h],
    inputs=[x_i],
    name="scan_h")
  # schedule
  s = tvm.create_schedule([scan_h.op])
  sch_cfg = {
    "n_bx": NUM_SM,
    "n_tx": NUM_THREAD_X,
    "n_ty": NUM_THREAD_Y,
    "bid_x": tvm.thread_axis((0, NUM_SM), "blockIdx.x"),
    "tid_x": tvm.thread_axis((0, NUM_THREAD_X), "threadIdx.x"),
    "tid_y": tvm.thread_axis((0, NUM_THREAD_Y), "threadIdx.y"),
  }
  # some collections and auxiliary definitions
  def lower(simple_mode=True):
    return tvm.lower(s, [w_i2h, b_i2h, w_h2h, b_h2h, x_i, scan_h], simple_mode=simple_mode)
  # do the scheduling
  s[scan_h.op].env_threads([sch_cfg["bid_x"], sch_cfg["tid_y"], sch_cfg["tid_x"]])
  sch_x_i(s, scan=scan_h, readers=[g_i2h], **sch_cfg)
  sch_s_h(s, scan=scan_h, readers=[g_h2h], **sch_cfg)
  sch_i2h(s, scan=scan_h, **sch_cfg)
  sch_h2h(s, scan=scan_h, **sch_cfg)
  sch_u_h(s, **sch_cfg)

  def check_device(target="cuda"):
    assert target == "cuda"
    frnn = tvm.build(s, [w_i2h, b_i2h, w_h2h, b_h2h, x_i, scan_h], target)
    ctx = tvm.gpu(0)
    # prepare data
    w_i2h_np = np.random.normal(size=(1, n_num_hidden, n_input_dim)).astype("float32")
    w_h2h_np = np.random.normal(size=(1, n_num_hidden, n_input_dim)).astype("float32")
    b_i2h_np = np.random.normal(size=(1, n_num_hidden)).astype("float32")
    b_h2h_np = np.random.normal(size=(1, n_num_hidden)).astype("float32")
    x_i_np = np.random.normal(size=(n_seq_len, n_batch_size, n_input_dim)).astype("float32")
    s_h_np = np.random.normal(size=(n_seq_len, n_batch_size, n_num_hidden)).astype("float32")
    # convert to NDArray
    w_i2h_nd = tvm.nd.array(w_i2h_np, ctx)
    w_h2h_nd = tvm.nd.array(w_h2h_np, ctx)
    b_i2h_nd = tvm.nd.array(b_i2h_np, ctx)
    b_h2h_nd = tvm.nd.array(b_h2h_np, ctx)
    x_i_nd = tvm.nd.array(x_i_np, ctx)
    s_h_nd = tvm.nd.array(s_h_np, ctx)
    # launch the kernel.
    print("Running the kernel...")
    frnn(w_i2h_nd, b_i2h_nd, w_h2h_nd, b_h2h_nd, x_i_nd, s_h_nd)
    ctx.sync()
    # measure time cost of second step.
    evaluator = frnn.time_evaluator(frnn.entry_name, ctx, 1, repeat=1000)
    eval_result = evaluator(w_i2h_nd, b_i2h_nd, w_h2h_nd, b_h2h_nd, x_i_nd, s_h_nd)
    print("Time cost=%g ms" % (eval_result.mean * 1000.0))
  with tvm.build_config(
    detect_global_barrier=DETECT_GLOBAL_BARRIER,
    auto_unroll_max_step=128,
    unroll_explicit=False):
    check_device()


def main():
  vanilla()
  # gru()
  # lstm()


if __name__ == "__main__":
  main()
