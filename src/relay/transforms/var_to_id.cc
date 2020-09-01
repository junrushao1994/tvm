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
#include <tvm/relay/transform.h>

#include "pattern_util.h"

namespace tvm {
namespace relay {

class VarToIdMutator final : public ExprMutator {
 public:
  using TIdMap = std::unordered_map<Var, Id, ObjectPtrHash, ObjectPtrEqual>;

  explicit VarToIdMutator(const TIdMap& id_map) : id_map(id_map) {}

  Expr VisitExpr_(const VarNode* var) override {
    return Var(id_map.at(GetRef<Var>(var)), var->type_annotation, {});
  }

  const TIdMap& id_map;
};

Expr VarToId(const Expr& e) {
  Map<Var, Integer> counter = CountVarAppearance(e);
  VarToIdMutator::TIdMap id_map;
  for (const auto& kv : counter) {
    Var var = kv.first;
    id_map[var] = Id(var->vid->name_hint);
  }
  return VarToIdMutator(id_map).Mutate(e);
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
