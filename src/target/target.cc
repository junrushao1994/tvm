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

inline Target Target::NewCreate(const std::string& target_str) {
  std::unordered_map<String, ObjectRef> attrs;
  std::vector<std::string> splits;
  {
    std::istringstream is(target_str);
    for (std::string s; is >> s; splits.push_back(s))
      ;
    CHECK(!splits.empty()) << "ValueError: Cannot parse empty string: \"" << target_str << "\"";
  }
  TargetId id = TargetId::Get(splits[0]);
  const auto& key_vtype = id->key2vtype_;
  for (size_t iter = 1, end = splits.size(); iter < end;) {
    std::string s = splits[iter++];
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
      CHECK(!name.empty()) << "ValueError: Empty attribute key in \"" << splits[iter - 1] << "\"";
      CHECK(!obj.empty()) << "ValueError: Empty attribute in \"" << splits[iter - 1] << "\"";
    } else if (iter < end && splits[iter][0] != '-') {
      // case 2. --key value
      name = s;
      obj = splits[iter++];
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
  const auto& key_default = id->key2default_;
  for (const auto &kv : key_default) {
    if (!attrs.count(kv.first)) {
      attrs[kv.first] = kv.second;
    }
  }
  ObjectPtr<TargetNode> target = make_object<TargetNode>();
  target->attrs = attrs;
  return Target(target);
}

TVM_REGISTER_NODE_TYPE(TargetNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TargetNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const TargetNode*>(node.get());
      p->stream << op->str();
    });

/*!
 * \brief Construct a Target node from the given name and options.
 * \param name The major target name. Should be one of
 * {"aocl", "aocl_sw_emu", "c", "cuda", "ext_dev", "hexagon", "hybrid", "llvm",
 *  "metal", "nvptx", "opencl", "rocm", "sdaccel", "stackvm", "vulkan"}
 * \param options Additional options appended to the target
 * \return The constructed Target
 */
Target CreateTarget(const std::string& name, const std::vector<std::string>& options) {
  {
    std::ostringstream os;
    os << name;
    for (const auto &s : options) {
      os << ' ' << s;
    }
    Target::NewCreate(os.str());
  }
  auto t = make_object<TargetNode>();
  t->id = TargetId::Get(name);

  std::string libs_flag = "-libs=";
  std::string device_flag = "-device=";
  std::string keys_flag = "-keys=";
  for (auto& item : options) {
    t->options_array.push_back(item);

    if (item.find(libs_flag) == 0) {
      std::stringstream ss(item.substr(libs_flag.length()));
      std::string lib_item;
      while (std::getline(ss, lib_item, ',')) {
        t->libs_array.push_back(lib_item);
      }
    } else if (item.find(device_flag) == 0) {
      t->device_name = item.substr(device_flag.length());
      t->keys_array.push_back(t->device_name);
    } else if (item.find(keys_flag) == 0) {
      std::stringstream ss(item.substr(keys_flag.length()));
      std::string key_item;
      while (std::getline(ss, key_item, ',')) {
        t->keys_array.push_back(key_item);
      }
    }
  }

  if (t->device_name.length() > 0) {
    t->keys_array.push_back(t->device_name);
  }
  t->device_type = kDLCPU;
  t->thread_warp_size = 1;
  if (name == "c" && t->device_name == "micro_dev") {
    t->device_type = kDLMicroDev;
  } else if (name == "c" || name == "llvm") {
    t->keys_array.push_back("cpu");
  } else if (name == "cuda" || name == "nvptx") {
    t->device_type = kDLGPU;
    t->keys_array.push_back("cuda");
    t->keys_array.push_back("gpu");
    t->max_num_threads = 1024;
    t->thread_warp_size = 32;
  } else if (name == "rocm" || name == "opencl") {
    // For now assume rocm schedule for opencl
    if (name == "opencl") {
      t->device_type = kDLOpenCL;
    } else {  // rocm
      t->device_type = kDLROCM;
      t->thread_warp_size = 64;
    }
    t->keys_array.push_back(name);
    t->keys_array.push_back("gpu");
    t->max_num_threads = 256;
    if (t->device_name == "intel_graphics") {
      t->thread_warp_size = 16;
    }
  } else if (name == "metal" || name == "vulkan" || name == "webgpu") {
    if (name == "metal") {
      t->device_type = kDLMetal;
    } else if (name == "vulkan") {
      t->device_type = kDLVulkan;
    } else {
      t->device_type = kDLWebGPU;
    }
    t->keys_array.push_back(name);
    t->keys_array.push_back("gpu");
    t->max_num_threads = 256;
  } else if (name == "sdaccel") {
    t->device_type = kDLOpenCL;
    t->keys_array.push_back("sdaccel");
    t->keys_array.push_back("hls");
  } else if (name == "aocl" || name == "aocl_sw_emu") {
    t->device_type = kDLAOCL;
    t->keys_array.push_back("aocl");
    t->keys_array.push_back("hls");
  } else if (name == "stackvm") {
    t->device_type = kDLCPU;
  } else if (name == "ext_dev") {
    t->device_type = kDLExtDev;
  } else if (name == "hybrid") {
    t->device_type = kDLCPU;
  } else if (name == "hexagon") {
    t->keys_array.push_back("hexagon");
    t->device_type = kDLHexagon;
  } else if (name == "webgpu") {
    t->keys_array.push_back("webgpu");
    t->device_type = kDLWebGPU;
  } else {
    LOG(ERROR) << "Unknown target name " << name << "; falling back to stackvm";
    return target::stackvm();
  }

  return Target(t);
}

TVM_REGISTER_GLOBAL("target.TargetCreate").set_body([](TVMArgs args, TVMRetValue* ret) {
  std::string name = args[0];
  std::vector<std::string> options;
  for (int i = 1; i < args.num_args; ++i) {
    std::string arg = args[i];
    options.push_back(arg);
  }

  *ret = CreateTarget(name, options);
});

TVM_REGISTER_GLOBAL("target.TargetFromString").set_body([](TVMArgs args, TVMRetValue* ret) {
  std::string target_str = args[0];
  *ret = Target::Create(target_str);
});

std::vector<std::string> TargetNode::keys() const {
  std::vector<std::string> result;
  for (auto& expr : keys_array) {
    result.push_back(expr);
  }
  return result;
}

std::vector<std::string> TargetNode::options() const {
  std::vector<std::string> result;
  for (auto& expr : options_array) {
    result.push_back(expr);
  }
  return result;
}

std::unordered_set<std::string> TargetNode::libs() const {
  std::unordered_set<std::string> result;
  for (auto& expr : libs_array) {
    result.insert(expr);
  }
  return result;
}

const std::string& TargetNode::str() const {
  if (str_repr_.length() != 0) return str_repr_;
  std::ostringstream result;
  result << id->name;
  for (const auto& x : options()) {
    result << " " << x;
  }
  str_repr_ = result.str();
  return str_repr_;
}

bool StartsWith(const std::string& str, const std::string& pattern) {
  return str.compare(0, pattern.length(), pattern) == 0;
}

std::string GetDeviceName(const std::string& target_str) {
  std::istringstream ss(target_str);
  std::string name;
  ss >> name;

  std::string item;
  while (ss >> item) {
    if (StartsWith(item, "-device=")) {
      return item.substr(std::string("-device=").length());
    }
  }

  return "";
}

Target Target::Create(const std::string& target_str) {
  if (target_str.length() == 0) {
    LOG(ERROR) << "target_str must not be empty";
  }

  std::istringstream ss(target_str);
  std::string name;

  ss >> name;
  auto device_name = GetDeviceName(target_str);

  std::vector<std::string> options;
  std::string item;
  while (ss >> item) {
    options.push_back(item);
  }

  return CreateTarget(name, options);
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

Target llvm(const std::vector<std::string>& options) { return CreateTarget("llvm", options); }

Target cuda(const std::vector<std::string>& options) { return CreateTarget("cuda", options); }

Target rocm(const std::vector<std::string>& options) { return CreateTarget("rocm", options); }

Target opencl(const std::vector<std::string>& options) { return CreateTarget("opencl", options); }

Target metal(const std::vector<std::string>& options) { return CreateTarget("metal", options); }

Target mali(const std::vector<std::string>& options) {
  return CreateTarget("opencl", MergeOptions(options, {"-device=mali"}));
}

Target intel_graphics(const std::vector<std::string>& options) {
  return CreateTarget("opencl", MergeOptions(options, {"-device=intel_graphics"}));
}

Target stackvm(const std::vector<std::string>& options) { return CreateTarget("stackvm", options); }

Target ext_dev(const std::vector<std::string>& options) { return CreateTarget("ext_dev", options); }

Target hexagon(const std::vector<std::string>& options) { return CreateTarget("hexagon", options); }
}  // namespace target
}  // namespace tvm
