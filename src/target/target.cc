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
 *  Compile executable modules.
 * \file src/target/target.cc
 */
#include <dmlc/thread_local.h>
#include <tvm/node/repr_printer.h>
#include <tvm/runtime/registry.h>
#include <tvm/target/target.h>
#include <tvm/target/target_id.h>
#include <tvm/tir/expr.h>

#include <algorithm>
#include <stack>

namespace tvm {

using runtime::PackedFunc;
using runtime::TVMArgs;
using runtime::TVMRetValue;

inline int FindUnique(const std::string& str, const std::string& substr) {
  size_t pos = str.find_first_of(substr);
  if (pos == std::string::npos) {
    return -1;
  }
  size_t next_pos = pos + substr.size();
  CHECK(next_pos >= str.size() || str.find_first_of(substr, next_pos) == std::string::npos)
      << "ValueError: At most one \"" << substr << "\" is allowed in "
      << "the the given string \"" << str << "\"";
  return pos;
}

inline size_t CalcNumPrefixDashes(const std::string& s) {
  size_t i = 0;
  for (; i < s.length() && s[i] == '-'; ++i)
    ;
  return i;
}

Map<String, ObjectRef> ParseTargetAttrs(
    const std::vector<std::string>& options,
    const std::unordered_map<String, TargetIdNode::ValueTypeInfo>& key_vtype,
    const std::unordered_map<String, ObjectRef>& key_default) {
  std::unordered_map<String, ObjectRef> attrs;
  for (size_t iter = 0, end = options.size(); iter < end;) {
    std::string s = options[iter++];
    // remove the prefix dashes
    size_t n_dashes = CalcNumPrefixDashes(s);
    CHECK(0 < n_dashes && n_dashes < s.size())
        << "ValueError: Not an attribute key \"" << s << "\"";
    s = s.substr(n_dashes);
    // parse name-obj pair
    std::string name;
    std::string obj;
    int pos;
    if ((pos = FindUnique(s, "=")) != -1) {
      // case 1. --key=value
      name = s.substr(0, pos);
      obj = s.substr(pos + 1);
      CHECK(!name.empty()) << "ValueError: Empty attribute key in \"" << options[iter - 1] << "\"";
      CHECK(!obj.empty()) << "ValueError: Empty attribute in \"" << options[iter - 1] << "\"";
    } else if (iter < end && options[iter][0] != '-') {
      // case 2. --key value
      name = s;
      obj = options[iter++];
    } else {
      // case 3. --boolean-key
      name = s;
      obj = "1";
    }
    // check if `name` is invalid
    auto it = key_vtype.find(name);
    if (it == key_vtype.end()) {
      std::ostringstream os;
      os << "AttributeError: Invalid config option, cannot recognize \'" << name
         << "\'. Candidates are:";
      for (const auto& kv : key_vtype) {
        os << "\n  " << kv.first;
      }
      LOG(FATAL) << os.str();
    }
    // then `name` is valid, let's parse them
    // only several types are supported when parsing raw string
    const auto& info = it->second;
    std::istringstream is(obj);
    if (info.type_index == Integer::ContainerType::_GetOrAllocRuntimeTypeIndex()) {
      int v;
      is >> v;
      attrs[name] = Integer(v);
    } else if (info.type_index == String::ContainerType::_GetOrAllocRuntimeTypeIndex()) {
      std::string v;
      is >> v;
      attrs[name] = String(v);
    } else {
      LOG(FATAL) << "TypeError: Parsing type \"" << info.type_key
                 << "\" from raw string is not supported"
                 << ", but get attribute key \"" << name << "\""
                 << ", and attribute \"" << obj << "\"";
    }
    if (is.fail()) {
      LOG(FATAL) << "ValueError: Cannot parse type \"" << info.type_key << "\""
                 << ", where attribute key is \"" << name << "\""
                 << ", and attribute is \"" << obj << "\"";
    }
  }
  // set default attribute values if they do not exist
  for (const auto& kv : key_default) {
    if (!attrs.count(kv.first)) {
      attrs[kv.first] = kv.second;
    }
  }
  return attrs;
}

Target Target::CreateTarget(const std::string& name, const std::vector<std::string>& options) {
  TargetId id = TargetId::Get(name);
  ObjectPtr<TargetNode> target = make_object<TargetNode>();
  target->id = id;
  // parse attrs
  target->attrs = ParseTargetAttrs(options, id->key2vtype_, id->key2default_);
  String device_name = target->GetAttr<String>("device", "").value();
  // create string representation
  {
    std::ostringstream str_repr;
    str_repr << name;
    for (const auto& s : options) {
      str_repr << ' ' << s;
    }
    target->str_repr_ = str_repr.str();
  }
  // correct `thread_warp_size` for intel_graphics
  if (name == "opencl" && device_name == "intel_graphics") {
    target->attrs.Set("thread_warp_size", Integer(16));
  }
  // set up keys
  {
    Array<String> keys = target->id->default_keys;
    // add `device_name`
    if (!device_name.empty()) {
      keys.push_back(device_name);
    }
    // add user provided keys
    std::istringstream is(target->GetAttr<String>("keys", "").value());
    for (std::string item; std::getline(is, item, ',');) {
      keys.push_back(item);
    }
    target->keys = std::move(keys);
  }
  return Target(target);
}

TVM_REGISTER_NODE_TYPE(TargetNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TargetNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const TargetNode*>(node.get());
      p->stream << op->str();
    });

TVM_REGISTER_GLOBAL("target.TargetCreate").set_body([](TVMArgs args, TVMRetValue* ret) {
  std::string name = args[0];
  std::vector<std::string> options;
  for (int i = 1; i < args.num_args; ++i) {
    std::string arg = args[i];
    options.push_back(arg);
  }

  *ret = Target::CreateTarget(name, options);
});

TVM_REGISTER_GLOBAL("target.TargetFromString").set_body([](TVMArgs args, TVMRetValue* ret) {
  std::string target_str = args[0];
  *ret = Target::Create(target_str);
});

std::vector<std::string> TargetNode::GetKeys() const {
  std::vector<std::string> result;
  for (auto& expr : keys) {
    result.push_back(expr);
  }
  return result;
}

std::unordered_set<std::string> TargetNode::GetLibs() const {
  Optional<String> libs = this->GetAttr<String>("libs");
  if (!libs.defined()) {
    return {};
  }
  std::unordered_set<std::string> result;
  std::istringstream is(libs.value());
  for (std::string item; std::getline(is, item, ',');) {
    result.insert(item);
  }
  return result;
}

const std::string& TargetNode::str() const {
  CHECK(!str_repr_.empty());
  return str_repr_;
}

bool StartsWith(const std::string& str, const std::string& pattern) {
  return str.compare(0, pattern.length(), pattern) == 0;
}

Target Target::Create(const std::string& target_str) {
  std::vector<std::string> splits;
  std::istringstream is(target_str);
  for (std::string s; is >> s; splits.push_back(s))
    ;
  CHECK(!splits.empty()) << "ValueError: Cannot parse empty target string: \"" << target_str
                         << "\"";
  return CreateTarget(splits[0], {splits.begin() + 1, splits.end()});
}

/*! \brief Entry to hold the Target context stack. */
struct TVMTargetThreadLocalEntry {
  /*! \brief The current target context */
  std::stack<tvm::Target> context_stack;
};

/*! \brief Thread local store to hold the Target context stack. */
typedef dmlc::ThreadLocalStore<TVMTargetThreadLocalEntry> TVMTargetThreadLocalStore;

void Target::EnterWithScope() {
  TVMTargetThreadLocalEntry* entry = TVMTargetThreadLocalStore::Get();
  entry->context_stack.push(*this);
}

void Target::ExitWithScope() {
  TVMTargetThreadLocalEntry* entry = TVMTargetThreadLocalStore::Get();
  CHECK(!entry->context_stack.empty());
  CHECK(entry->context_stack.top().same_as(*this));
  entry->context_stack.pop();
}

tvm::Target Target::Current(bool allow_not_defined) {
  TVMTargetThreadLocalEntry* entry = TVMTargetThreadLocalStore::Get();
  if (entry->context_stack.size() > 0) {
    return entry->context_stack.top();
  }
  CHECK(allow_not_defined)
      << "Target context required. Please set it by constructing a TargetContext";

  return Target();
}

TVM_REGISTER_GLOBAL("target.GetCurrentTarget").set_body([](TVMArgs args, TVMRetValue* ret) {
  bool allow_not_defined = args[0];
  *ret = Target::Current(allow_not_defined);
});
class Target::Internal {
 public:
  static void EnterScope(Target target) { target.EnterWithScope(); }
  static void ExitScope(Target target) { target.ExitWithScope(); }
};

TVM_REGISTER_GLOBAL("target.EnterTargetScope").set_body_typed(Target::Internal::EnterScope);

TVM_REGISTER_GLOBAL("target.ExitTargetScope").set_body_typed(Target::Internal::ExitScope);

namespace target {
std::vector<std::string> MergeOptions(std::vector<std::string> opts,
                                      const std::vector<std::string>& new_opts) {
  opts.insert(opts.end(), new_opts.begin(), new_opts.end());
  return opts;
}

Target llvm(const std::vector<std::string>& options) {
  return Target::CreateTarget("llvm", options);
}

Target cuda(const std::vector<std::string>& options) {
  return Target::CreateTarget("cuda", options);
}

Target rocm(const std::vector<std::string>& options) {
  return Target::CreateTarget("rocm", options);
}

Target opencl(const std::vector<std::string>& options) {
  return Target::CreateTarget("opencl", options);
}

Target metal(const std::vector<std::string>& options) {
  return Target::CreateTarget("metal", options);
}

Target mali(const std::vector<std::string>& options) {
  return Target::CreateTarget("opencl", MergeOptions(options, {"-device=mali"}));
}

Target intel_graphics(const std::vector<std::string>& options) {
  return Target::CreateTarget("opencl", MergeOptions(options, {"-device=intel_graphics"}));
}

Target stackvm(const std::vector<std::string>& options) {
  return Target::CreateTarget("stackvm", options);
}

Target ext_dev(const std::vector<std::string>& options) {
  return Target::CreateTarget("ext_dev", options);
}

Target hexagon(const std::vector<std::string>& options) {
  return Target::CreateTarget("hexagon", options);
}
}  // namespace target
}  // namespace tvm
