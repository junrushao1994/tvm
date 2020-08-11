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

#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <tvm/target/target.h>

#include <cmath>
#include <string>

using namespace tvm;

TVM_REGISTER_TARGET_KIND("TestTargetKind")
    .set_attr<std::string>("Attr1", "Value1")
    .add_attr_option<Bool>("my_bool")
    .add_attr_option<Array<String>>("your_names")
    .add_attr_option<Map<String, Integer>>("her_maps");

TEST(TargetKind, GetAttrMap) {
  auto map = tvm::TargetKind::GetAttrMap<std::string>("Attr1");
  auto target_kind = tvm::TargetKind::Get("TestTargetKind");
  std::string result = map[target_kind];
  CHECK_EQ(result, "Value1");
}

TEST(TargetKind, CreationFromConfig) {
  Map<String, ObjectRef> config = {
      {"my_bool", Bool(true)},
      {"your_names", Array<String>{"junru", "jian"}},
      {"kind", String("TestTargetKind")},
      {
          "her_maps",
          Map<String, Integer>{
              {"a", 1},
              {"b", 2},
          },
      },
  };
  Target target = Target::FromConfig(config);
  CHECK_EQ(target->kind, TargetKind::Get("TestTargetKind"));
  CHECK_EQ(target->tag, "");
  CHECK(target->keys.empty());
  Bool my_bool = target->GetAttr<Bool>("my_bool").value();
  CHECK_EQ(my_bool.operator bool(), true);
  Array<String> your_names = target->GetAttr<Array<String>>("your_names").value();
  CHECK_EQ(your_names.size(), 2U);
  CHECK_EQ(your_names[0], "junru");
  CHECK_EQ(your_names[1], "jian");
  Map<String, Integer> her_maps = target->GetAttr<Map<String, Integer>>("her_maps").value();
  CHECK_EQ(her_maps.size(), 2U);
  CHECK_EQ(her_maps["a"], 1);
  CHECK_EQ(her_maps["b"], 2);
}

TEST(TargetKind, CreationFromConfigFail) {
  Map<String, ObjectRef> config = {
      {"my_bool", Bool(true)},
      {"your_names", Array<String>{"junru", "jian"}},
      {"kind", String("TestTargetKind")},
      {"bad", ObjectRef(nullptr)},
      {
          "her_maps",
          Map<String, Integer>{
              {"a", 1},
              {"b", 2},
          },
      },
  };
  bool failed = false;
  try {
    Target::FromConfig(config);
  } catch (...) {
    failed = true;
  }
  ASSERT_EQ(failed, true);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
