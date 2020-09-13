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
 * \file simplify_inference.cc
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/pattern_functor.h>
#include <tvm/relay/transform.h>

#include "pattern_util.h"

namespace tvm {
namespace relay {

class VarToIdMutator final : public ExprMutator, public PatternMutator {
 public:
  using TIdMap = std::unordered_map<Var, Id, ObjectPtrHash, ObjectPtrEqual>;

  explicit VarToIdMutator(const TIdMap& id_map) : id_map(id_map) {}

  Expr VisitExpr(const Expr& expr) override {
    if (const auto* var = expr.as<VarNode>()) {
      return Var(id_map.at(GetRef<Var>(var)), var->type_annotation, expr->span);
    } else {
      return ExprMutator::VisitExpr(expr);
    }
  }

  Pattern VisitPattern(const Pattern& p) final { return PatternMutator::VisitPattern(p); }

  Pattern VisitPattern_(const PatternVarNode* pattern_var) override {
    const Var& var = pattern_var->var;
    PatternVar result(Var(id_map.at(var), var->type_annotation, var->span));
    result->span = pattern_var->span;
    return std::move(result);
  }

  const TIdMap& id_map;
};

Expr VarToId(const Expr& e) {
  Map<Var, Integer> counter = CountVarAppearance(e);
  bool all_unique = true;
  for (const auto& kv : counter) {
    int cnt = kv.second;
    if (cnt > 1) {
      all_unique = false;
      break;
    }
  }
  if (all_unique) {
    return e;
  }
  VarToIdMutator::TIdMap id_map;
  for (const auto& kv : counter) {
    const Var& var = kv.first;
    id_map[var] = Id(var->vid->name_hint);
  }
  return VarToIdMutator(id_map).VisitExpr(e);
}

namespace transform {

Pass VarToId() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) { return Downcast<Function>(VarToId(f)); };
  return CreateFunctionPass(pass_func, 0, "VarToId", {});
}

TVM_REGISTER_GLOBAL("relay._transform.VarToId").set_body_typed(VarToId);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
