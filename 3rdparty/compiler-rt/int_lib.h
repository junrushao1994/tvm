/*
 * Copyright (c) 2009-2015 by llvm/compiler-rt contributors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 * \file int_lib.h
 * \brief Functions for conversion between fp32 and fp16, adopted from compiler-rt.
 */
#ifndef COMPILER_RT_INT_LIB_H_
#define COMPILER_RT_INT_LIB_H_

// Definitions for builtins unavailable on MSVC
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>

uint32_t __inline __builtin_ctz(uint32_t value) {
  unsigned long trailing_zero = 0;
  if (_BitScanForward(&trailing_zero, value)) return trailing_zero;
  return 32;
}

uint32_t __inline __builtin_clz(uint32_t value) {
  unsigned long leading_zero = 0;
  if (_BitScanReverse(&leading_zero, value)) return 31 - leading_zero;
  return 32;
}

#if defined(_M_ARM) || defined(_M_X64)
uint32_t __inline __builtin_clzll(uint64_t value) {
  unsigned long leading_zero = 0;
  if (_BitScanReverse64(&leading_zero, value)) return 63 - leading_zero;
  return 64;
}
#else
uint32_t __inline __builtin_clzll(uint64_t value) {
  if (value == 0) return 64;
  uint32_t msh = (uint32_t)(value >> 32);
  uint32_t lsh = (uint32_t)(value & 0xFFFFFFFF);
  if (msh != 0) return __builtin_clz(msh);
  return 32 + __builtin_clz(lsh);
}
#endif

#define __builtin_clzl __builtin_clzll
#endif  // defined(_MSC_VER) && !defined(__clang__)

#endif  // COMPILER_RT_INT_LIB_H_
