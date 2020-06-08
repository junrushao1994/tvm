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
 * \file tvm/node/container.h
 * \brief Array/Map container in the DSL graph.
 */
#ifndef TVM_NODE_CONTAINER_H_
#define TVM_NODE_CONTAINER_H_

#include <tvm/runtime/container.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>

#include <string>

namespace tvm {

using runtime::Array;
using runtime::ArrayNode;
using runtime::Downcast;
using runtime::IterAdapter;
using runtime::make_object;
using runtime::Map;
using runtime::MapNode;
using runtime::Object;
using runtime::ObjectEqual;
using runtime::ObjectHash;
using runtime::ObjectPtr;
using runtime::ObjectPtrEqual;
using runtime::ObjectPtrHash;
using runtime::ObjectRef;
using runtime::String;
using runtime::StringObj;

}  // namespace tvm

namespace tvm {
namespace runtime {
// Additional overloads for PackedFunc checking.
template <typename T>
struct ObjectTypeChecker<Array<T> > {
  static bool Check(const Object* ptr) {
    if (ptr == nullptr) return true;
    if (!ptr->IsInstance<ArrayNode>()) return false;
    const ArrayNode* n = static_cast<const ArrayNode*>(ptr);
    for (const ObjectRef& p : *n) {
      if (!ObjectTypeChecker<T>::Check(p.get())) {
        return false;
      }
    }
    return true;
  }
  static std::string TypeName() { return "List[" + ObjectTypeChecker<T>::TypeName() + "]"; }
};

template <typename K, typename V>
struct ObjectTypeChecker<Map<K, V> > {
  static bool Check(const Object* ptr) {
    if (ptr == nullptr) return true;
    if (!ptr->IsInstance<MapNode>()) return false;
    const MapNode* n = static_cast<const MapNode*>(ptr);
    for (const auto& kv : *n) {
      if (!ObjectTypeChecker<K>::Check(kv.k.get())) return false;
      if (!ObjectTypeChecker<V>::Check(kv.v.get())) return false;
    }
    return true;
  }
  static std::string TypeName() {
    return "Map[" + ObjectTypeChecker<K>::TypeName() + ", " + ObjectTypeChecker<V>::TypeName() +
           ']';
  }
};
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_NODE_CONTAINER_H_
