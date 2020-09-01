/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/relay/analysis/count_var_appearance.cc
 * \brief Implementation of CountVarAppearance and AllVarsDistinct
 */

#include <tvm/relay/expr_functor.h>

namespace tvm {
namespace relay {

class VarAppearanceCounter final : public ExprVisitor {
 public:
  void VisitExpr_(const VarNode* var) override { ++counter[var]; }

  Map<Var, Integer> Run(const Expr& expr) {
    VisitExpr(expr);
    Map<Var, Integer> result;
    for (const auto& kv : counter) {
      result.Set(GetRef<Var>(kv.first), kv.second);
    }
    return result;
  }

  std::unordered_map<const VarNode*, int> counter;
};

Map<Var, Integer> CountVarAppearance(const Expr& body) { return VarAppearanceCounter().Run(body); }

bool AllVarsDistinct(const Expr& body) {
  for (const auto& kv : CountVarAppearance(body)) {
    if (kv.second > 1) {
      return false;
    }
  }
  return true;
}

}  // namespace relay
}  // namespace tvm
