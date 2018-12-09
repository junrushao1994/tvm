import tvm
import os
from tvm.contrib import nvcc
import numpy as np


# Quick knobs
TASK="rnn"
DOUBLE_BUFFER = False
USE_MANUAL_CODE = False
DETECT_GLOBAL_BARRIER = False
NUM_SM = 80   # 80 streaming processors in V100

NUM_THREAD_X = 32


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
  def sch(s, scan, readers, n_tx, tid_x, **kwargs):
    xL = s.cache_read(x, "shared", readers=readers)
    s[xL].compute_at(s[scan], s[scan].op.scan_axis)
    _t, _n, c = s[xL].op.axis
    _co, tx = s[xL].split(c, factor=n_tx)
    s[xL].bind(tx, tid_x)
    if DOUBLE_BUFFER:
      s[xL].double_buffer()
  return (x, ), sch


def _state(seq_len, batch_size, num_hidden, name, **kwargs):
  state = tvm.placeholder((seq_len, batch_size, num_hidden), name=name)
  s_init = tvm.compute((1, batch_size, num_hidden), lambda _, n, c: 0.0, name="init_" + name)
  def sch(s, scan, readers, **kwargs):
    def _state(n_tx, tid_x, **kwargs):
      if not readers:
        return
      # fetching the state
      sS = s.cache_read(state, "shared", readers=readers)
      s[sS].compute_at(s[scan], s[scan].op.scan_axis)
      _t, _n, c = s[sS].op.axis
      _co, tx = s[sS].split(c, factor=n_tx)
      s[sS].bind(tx, tid_x)
      if DOUBLE_BUFFER:
        s[sS].double_buffer()
    def _s_init(n_bx, tid_x, bid_x, **kwargs):
      _1, _n, c = s[s_init].op.axis
      bx, _ci = s[s_init].split(c, nparts=n_bx)
      s[s_init].bind(bx, bid_x)
      s[s_init].set_store_predicate(tid_x.equal(0))
    _state(**kwargs)
    _s_init(**kwargs)
  return (state, s_init), sch


def _linear(x, num_gemm, num_hidden, name, **kwargs):
  # TODO
  pass


def _update_state(x, seq_len, batch_size, num_hidden, name, **kwargs):
  u = tvm.compute((seq_len, batch_size, num_hidden), lambda t, n, c: x[t, n, c], name=name)
  def sch(s, n_bx, bid_x, tid_x, **kwargs):
    s[x].compute_inline()
    _t, _n, c = s[u].op.axis
    _co, bx = s[u].split(c, factor=n_bx)
    s[u].bind(bx, bid_x)
    s[u].set_store_predicate(tid_x.equal(0))
  return (u, ), sch


def vanilla(n_seq_len=128, n_num_hidden=128, n_input_dim=128, n_batch_size=1):
  n_seq_len += 1
  # seq_len = tvm.convert(n_seq_len)
  batch_size = tvm.convert(n_batch_size)
  num_hidden = tvm.convert(n_num_hidden)
  # input_dim = tvm.convert(n_input_dim)
  seq_len = tvm.var("seq_len")
  # batch_size = tvm.var("batch_size")
  # num_hidden = tvm.var("num_hidden")
  input_dim = num_hidden
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
  # Computation inside the RNN cell
  n_h = tvm.compute((seq_len, batch_size, input_dim), lambda t, n, c: x_i[t - 1, n, c] + s_h[t - 1, n, c], name="n_h")
  # Define update rules explicitly
  (u_h, ), sch_u_h = _update_state(n_h, name="u_h", **config)
  # Finally, define the scanning itself
  scan_h = tvm.scan(
    inputs=[x_i],
    state_placeholder=[s_h],
    init=[init_s_h],
    update=[u_h],
    name="scan_h")
  # schedule
  s = tvm.create_schedule([scan_h.op])
  sch_cfg = {
    "n_bx": NUM_SM,
    "n_tx": NUM_THREAD_X,
    "bid_x": tvm.thread_axis((0, NUM_SM), "blockIdx.x"),
    "tid_x": tvm.thread_axis((0, NUM_THREAD_X), "threadIdx.x"),
  }
  # some collections and auxiliary definitions
  def lower(simple_mode=True):
    return tvm.lower(s, [x_i, scan_h], simple_mode=simple_mode)
  s[scan_h.op].env_threads([sch_cfg["bid_x"], sch_cfg["tid_x"]])
  # do the scheduling
  sch_x_i(s, scan=scan_h, readers=[n_h], **sch_cfg)
  sch_s_h(s, scan=scan_h, readers=[n_h], **sch_cfg)
  sch_u_h(s, **sch_cfg)
  print(lower())

  def check_device(target="cuda -arch=sm_70 -O3"):
    print("Compiling the kernel...")
    frnn = tvm.build(s, [x_i, scan_h], target)
    ctx = tvm.gpu(0)
    # prepare data
    x_i_np = np.full(fill_value=1,shape=(n_seq_len, n_batch_size, n_input_dim), dtype="float32")
    s_h_np = np.full(fill_value=0,shape=(n_seq_len, n_batch_size, n_input_dim), dtype="float32")
    # convert to NDArray
    x_i_nd = tvm.nd.array(x_i_np, ctx)
    s_h_nd = tvm.nd.array(s_h_np, ctx)
    # launch the kernel.
    print("Running the kernel...")
    for _ in range(10):
      frnn(x_i_nd, s_h_nd)
    ctx.sync()
    # measure time cost of second step.
    print("Benchmarking...")
    evaluator = frnn.time_evaluator(frnn.entry_name, ctx, 1, repeat=10)
    eval_result = evaluator(x_i_nd, s_h_nd)
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
