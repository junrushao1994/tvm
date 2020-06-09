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
 * \file tvm/runtime/container.h
 * \brief Common POD(plain old data) container types.
 */
#ifndef TVM_RUNTIME_CONTAINER_H_
#define TVM_RUNTIME_CONTAINER_H_

#include <dmlc/logging.h>
#include <int_lib.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>

#include <algorithm>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <string>
#include <unordered_map>
// We use c++14 std::experimental::string_view for optimizing hash computation
// only right now, its usage is limited in this file. Any broader usage of
// std::experiment in our core codebase is discouraged and needs community
// discussion for each use case. Reference for feature test macros of
// string_view:
// https://isocpp.org/std/standing-documents/sd-6-sg10-feature-test-recommendations
// https://en.cppreference.com/w/User:D41D8CD98F/feature_testing_macros
#if defined(__cpp_lib_experimental_string_view) && __cpp_lib_experimental_string_view >= 201411
#define TVM_USE_CXX14_STRING_VIEW_HASH 1
#else
#define TVM_USE_CXX14_STRING_VIEW_HASH 0
#endif

// Tested with clang version 9.0.1 and c++17. It will detect string_view support
// correctly.
#if defined(__cpp_lib_string_view) && __cpp_lib_string_view >= 201606
#define TVM_USE_CXX17_STRING_VIEW_HASH 1
#else
#define TVM_USE_CXX17_STRING_VIEW_HASH 0
#endif

#if TVM_USE_CXX17_STRING_VIEW_HASH
#include <string_view>
#elif TVM_USE_CXX14_STRING_VIEW_HASH
#include <experimental/string_view>
#endif

#include <type_traits>
#include <utility>
#include <vector>

#if defined(_MSC_VER) && !defined(__clang__)
uint32_t __builtin_ctz(uint32_t value);
uint32_t __builtin_clz(uint32_t value);
uint32_t __builtin_clzll(uint64_t value);
#define __builtin_clzl __builtin_clzll
#endif  // defined(_MSC_VER) && !defined(__clang__)

namespace tvm {
namespace runtime {

/*! \brief String-aware ObjectRef equal functor */
struct ObjectHash {
  /*!
   * \brief Calculate the hash code of an ObjectRef
   * \param a The given ObjectRef
   * \return Hash code of a, string hash for strings and pointer address otherwise.
   */
  size_t operator()(const ObjectRef& a) const;
};

/*! \brief String-aware ObjectRef hash functor */
struct ObjectEqual {
  /*!
   * \brief Check if the two ObjectRef are equal
   * \param a One ObjectRef
   * \param b The other ObjectRef
   * \return String equality if both are strings, pointer address equality otherwise.
   */
  bool operator()(const ObjectRef& a, const ObjectRef& b) const;
};

/*!
 * \brief Base template for classes with array like memory layout.
 *
 *        It provides general methods to access the memory. The memory
 *        layout is ArrayType + [ElemType]. The alignment of ArrayType
 *        and ElemType is handled by the memory allocator.
 *
 * \tparam ArrayType The array header type, contains object specific metadata.
 * \tparam ElemType The type of objects stored in the array right after
 * ArrayType.
 *
 * \code
 * // Example usage of the template to define a simple array wrapper
 * class ArrayObj : public InplaceArrayBase<ArrayObj, Elem> {
 * public:
 *  // Wrap EmplaceInit to initialize the elements
 *  template <typename Iterator>
 *  void Init(Iterator begin, Iterator end) {
 *   size_t num_elems = std::distance(begin, end);
 *   auto it = begin;
 *   this->size = 0;
 *   for (size_t i = 0; i < num_elems; ++i) {
 *     InplaceArrayBase::EmplaceInit(i, *it++);
 *     this->size++;
 *   }
 *  }
 * }
 *
 * void test_function() {
 *   vector<Elem> fields;
 *   auto ptr = make_inplace_array_object<ArrayObj, Elem>(fields.size());
 *   ptr->Init(fields.begin(), fields.end());
 *
 *   // Access the 0th element in the array.
 *   assert(ptr->operator[](0) == fields[0]);
 * }
 *
 * \endcode
 */
template <typename ArrayType, typename ElemType>
class InplaceArrayBase {
 public:
  /*!
   * \brief Access element at index
   * \param idx The index of the element.
   * \return Const reference to ElemType at the index.
   */
  const ElemType& operator[](size_t idx) const {
    size_t size = Self()->GetSize();
    CHECK_LT(idx, size) << "Index " << idx << " out of bounds " << size << "\n";
    return *(reinterpret_cast<ElemType*>(AddressOf(idx)));
  }

  /*!
   * \brief Access element at index
   * \param idx The index of the element.
   * \return Reference to ElemType at the index.
   */
  ElemType& operator[](size_t idx) {
    size_t size = Self()->GetSize();
    CHECK_LT(idx, size) << "Index " << idx << " out of bounds " << size << "\n";
    return *(reinterpret_cast<ElemType*>(AddressOf(idx)));
  }

  /*!
   * \brief Destroy the Inplace Array Base object
   */
  ~InplaceArrayBase() {
    if (!(std::is_standard_layout<ElemType>::value && std::is_trivial<ElemType>::value)) {
      size_t size = Self()->GetSize();
      for (size_t i = 0; i < size; ++i) {
        ElemType* fp = reinterpret_cast<ElemType*>(AddressOf(i));
        fp->ElemType::~ElemType();
      }
    }
  }

 protected:
  /*!
   * \brief Construct a value in place with the arguments.
   *
   * \tparam Args Type parameters of the arguments.
   * \param idx Index of the element.
   * \param args Arguments to construct the new value.
   *
   * \note Please make sure ArrayType::GetSize returns 0 before first call of
   * EmplaceInit, and increment GetSize by 1 each time EmplaceInit succeeds.
   */
  template <typename... Args>
  void EmplaceInit(size_t idx, Args&&... args) {
    void* field_ptr = AddressOf(idx);
    new (field_ptr) ElemType(std::forward<Args>(args)...);
  }

  /*!
   * \brief Return the self object for the array.
   *
   * \return Pointer to ArrayType.
   */
  inline ArrayType* Self() const {
    return static_cast<ArrayType*>(const_cast<InplaceArrayBase*>(this));
  }

  /*!
   * \brief Return the raw pointer to the element at idx.
   *
   * \param idx The index of the element.
   * \return Raw pointer to the element.
   */
  void* AddressOf(size_t idx) const {
    static_assert(
        alignof(ArrayType) % alignof(ElemType) == 0 && sizeof(ArrayType) % alignof(ElemType) == 0,
        "The size and alignment of ArrayType should respect "
        "ElemType's alignment.");

    size_t kDataStart = sizeof(ArrayType);
    ArrayType* self = Self();
    char* data_start = reinterpret_cast<char*>(self) + kDataStart;
    return data_start + idx * sizeof(ElemType);
  }
};

/*!
 * \brief iterator adapter that adapts TIter to return another type.
 * \tparam Converter a struct that contains converting function
 * \tparam TIter the content iterator type.
 */
template <typename Converter, typename TIter>
class IterAdapter {
 public:
  using difference_type = typename std::iterator_traits<TIter>::difference_type;
  using value_type = typename Converter::ResultType;
  using pointer = typename Converter::ResultType*;
  using reference = typename Converter::ResultType&;
  using iterator_category = typename std::iterator_traits<TIter>::iterator_category;

  explicit IterAdapter(TIter iter) : iter_(iter) {}
  IterAdapter& operator++() {
    ++iter_;
    return *this;
  }
  IterAdapter& operator--() {
    --iter_;
    return *this;
  }
  IterAdapter operator++(int) {
    IterAdapter copy = *this;
    ++iter_;
    return copy;
  }
  IterAdapter operator--(int) {
    IterAdapter copy = *this;
    --iter_;
    return copy;
  }

  IterAdapter operator+(difference_type offset) const { return IterAdapter(iter_ + offset); }

  template <typename T = IterAdapter>
  typename std::enable_if<std::is_same<iterator_category, std::random_access_iterator_tag>::value,
                          typename T::difference_type>::type inline
  operator-(const IterAdapter& rhs) const {
    return iter_ - rhs.iter_;
  }

  bool operator==(IterAdapter other) const { return iter_ == other.iter_; }
  bool operator!=(IterAdapter other) const { return !(*this == other); }
  const value_type operator*() const { return Converter::convert(*iter_); }

 private:
  TIter iter_;
};

/*!
 * \brief iterator adapter that adapts TIter to return another type.
 * \tparam Converter a struct that contains converting function
 * \tparam TIter the content iterator type.
 */
template <typename Converter, typename TIter>
class ReverseIterAdapter {
 public:
  using difference_type = typename std::iterator_traits<TIter>::difference_type;
  using value_type = typename Converter::ResultType;
  using pointer = typename Converter::ResultType*;
  using reference = typename Converter::ResultType&;  // NOLINT(*)
  using iterator_category = typename std::iterator_traits<TIter>::iterator_category;

  explicit ReverseIterAdapter(TIter iter) : iter_(iter) {}
  ReverseIterAdapter& operator++() {
    --iter_;
    return *this;
  }
  ReverseIterAdapter& operator--() {
    ++iter_;
    return *this;
  }
  ReverseIterAdapter& operator++(int) {
    ReverseIterAdapter copy = *this;
    --iter_;
    return copy;
  }
  ReverseIterAdapter& operator--(int) {
    ReverseIterAdapter copy = *this;
    ++iter_;
    return copy;
  }
  ReverseIterAdapter operator+(difference_type offset) const {
    return ReverseIterAdapter(iter_ - offset);
  }

  template <typename T = ReverseIterAdapter>
  typename std::enable_if<std::is_same<iterator_category, std::random_access_iterator_tag>::value,
                          typename T::difference_type>::type inline
  operator-(const ReverseIterAdapter& rhs) const {
    return rhs.iter_ - iter_;
  }

  bool operator==(ReverseIterAdapter other) const { return iter_ == other.iter_; }
  bool operator!=(ReverseIterAdapter other) const { return !(*this == other); }
  const value_type operator*() const { return Converter::convert(*iter_); }

 private:
  TIter iter_;
};

/*! \brief array node content in array */
class ArrayNode : public Object, public InplaceArrayBase<ArrayNode, ObjectRef> {
 public:
  /*! \return The size of the array */
  size_t size() const { return this->size_; }

  /*!
   * \brief Read i-th element from array.
   * \param i The index
   * \return the i-th element.
   */
  const ObjectRef at(int64_t i) const { return this->operator[](i); }

  /*! \return begin constant iterator */
  const ObjectRef* begin() const { return static_cast<ObjectRef*>(InplaceArrayBase::AddressOf(0)); }

  /*! \return end constant iterator */
  const ObjectRef* end() const { return begin() + size_; }

  /*! \brief Release reference to all the elements */
  void clear() { ShrinkBy(size_); }

  /*!
   * \brief Set i-th element of the array in-place
   * \param i The index
   * \param item The value to be set
   */
  void SetItem(int64_t i, ObjectRef item) { this->operator[](i) = std::move(item); }

  /*!
   * \brief Constructs a container and copy from another
   * \param cap The capacity of the container
   * \param from Source of the copy
   * \return Ref-counted ArrayNode requested
   */
  static ObjectPtr<ArrayNode> CopyFrom(int64_t cap, ArrayNode* from) {
    int64_t size = from->size_;
    CHECK_GE(cap, size) << "ValueError: not enough capacity";
    ObjectPtr<ArrayNode> p = ArrayNode::Empty(cap);
    ObjectRef* write = p->MutableBegin();
    ObjectRef* read = from->MutableBegin();
    // To ensure exception safety, size is only incremented after the initialization succeeds
    for (int64_t& i = p->size_ = 0; i < size; ++i) {
      new (write++) ObjectRef(*read++);
    }
    return p;
  }

  /*!
   * \brief Constructs a container and move from another
   * \param cap The capacity of the container
   * \param from Source of the move
   * \return Ref-counted ArrayNode requested
   */
  static ObjectPtr<ArrayNode> MoveFrom(int64_t cap, ArrayNode* from) {
    int64_t size = from->size_;
    CHECK_GE(cap, size) << "ValueError: not enough capacity";
    ObjectPtr<ArrayNode> p = ArrayNode::Empty(cap);
    ObjectRef* write = p->MutableBegin();
    ObjectRef* read = from->MutableBegin();
    // To ensure exception safety, size is only incremented after the initialization succeeds
    for (int64_t& i = p->size_ = 0; i < size; ++i) {
      new (write++) ObjectRef(std::move(*read++));
    }
    from->size_ = 0;
    return p;
  }

  /*!
   * \brief Constructs a container with n elements. Each element is a copy of val
   * \param n The size of the container
   * \param val The init value
   * \return Ref-counted ArrayNode requested
   */
  static ObjectPtr<ArrayNode> CreateRepeated(int64_t n, const ObjectRef& val) {
    ObjectPtr<ArrayNode> p = ArrayNode::Empty(n);
    ObjectRef* itr = p->MutableBegin();
    for (int64_t& i = p->size_ = 0; i < n; ++i) {
      new (itr++) ObjectRef(val);
    }
    return p;
  }

  static constexpr const uint32_t _type_index = TypeIndex::kRuntimeArray;
  static constexpr const char* _type_key = "Array";
  TVM_DECLARE_FINAL_OBJECT_INFO(ArrayNode, Object);

 private:
  /*! \return Size of initialized memory, used by InplaceArrayBase. */
  size_t GetSize() const { return this->size_; }

  /*! \return begin mutable iterator */
  ObjectRef* MutableBegin() const {
    return static_cast<ObjectRef*>(InplaceArrayBase::AddressOf(0));
  }

  /*! \return end mutable iterator */
  ObjectRef* MutableEnd() const { return MutableBegin() + size_; }

  /*!
   * \brief Create an ArrayNode with the given capacity.
   * \param n Required capacity
   * \return Ref-counted ArrayNode requested
   */
  static ObjectPtr<ArrayNode> Empty(int64_t n = kInitSize) {
    CHECK_GE(n, 0);
    ObjectPtr<ArrayNode> p = make_inplace_array_object<ArrayNode, ObjectRef>(n);
    p->capacity_ = n;
    p->size_ = 0;
    return p;
  }

  /*!
   * \brief Inplace-initialize the elements starting idx from [first, last)
   * \param idx The starting point
   * \param first Begin of iterator
   * \param last End of iterator
   * \tparam IterType The type of iterator
   * \return Self
   */
  template <typename IterType>
  ArrayNode* InitRange(int64_t idx, IterType first, IterType last) {
    ObjectRef* itr = MutableBegin() + idx;
    for (; first != last; ++first) {
      ObjectRef ref = *first;
      new (itr++) ObjectRef(std::move(ref));
    }
    return this;
  }

  /*!
   * \brief Move elements from right to left, requires src_begin > dst
   * \param dst Destination
   * \param src_begin The start point of copy (inclusive)
   * \param src_end The end point of copy (exclusive)
   * \return Self
   */
  ArrayNode* MoveElementsLeft(int64_t dst, int64_t src_begin, int64_t src_end) {
    ObjectRef* from = MutableBegin() + src_begin;
    ObjectRef* to = MutableBegin() + dst;
    while (src_begin++ != src_end) {
      *to++ = std::move(*from++);
    }
    return this;
  }

  /*!
   * \brief Move elements from left to right, requires src_begin < dst
   * \param dst Destination
   * \param src_begin The start point of move (inclusive)
   * \param src_end The end point of move (exclusive)
   * \return Self
   */
  ArrayNode* MoveElementsRight(int64_t dst, int64_t src_begin, int64_t src_end) {
    ObjectRef* from = MutableBegin() + src_end;
    ObjectRef* to = MutableBegin() + (src_end - src_begin + dst);
    while (src_begin++ != src_end) {
      *--to = std::move(*--from);
    }
    return this;
  }

  /*!
   * \brief Enlarges the size of the array
   * \param delta Size enlarged, should be positive
   * \param val Default value
   * \return Self
   */
  ArrayNode* EnlargeBy(int64_t delta, const ObjectRef& val = ObjectRef(nullptr)) {
    ObjectRef* itr = MutableEnd();
    while (delta-- > 0) {
      new (itr++) ObjectRef(val);
      ++size_;
    }
    return this;
  }

  /*!
   * \brief Shrinks the size of the array
   * \param delta Size shrinked, should be positive
   * \return Self
   */
  ArrayNode* ShrinkBy(int64_t delta) {
    ObjectRef* itr = MutableEnd();
    while (delta-- > 0) {
      (--itr)->ObjectRef::~ObjectRef();
      --size_;
    }
    return this;
  }

  /*! \brief Number of elements used */
  int64_t size_;

  /*! \brief Number of elements allocated */
  int64_t capacity_;

  /*! \brief Initial size of ArrayNode */
  static constexpr int64_t kInitSize = 4;

  /*! \brief Expansion factor of the Array */
  static constexpr int64_t kIncFactor = 2;

  // CRTP parent class
  friend InplaceArrayBase<ArrayNode, ObjectRef>;

  // Reference class
  template <typename, typename>
  friend class Array;

  // To specialize make_object<ArrayNode>
  friend ObjectPtr<ArrayNode> make_object<>();
};

/*!
 * \brief Array container of ObjectRef in DSL graph.
 *  Array implements copy-on-write semantics, which means array is mutable
 *  but copy will happen when array is referenced in more than two places.
 *
 * operator[] only provide const access, use Set to mutate the content.
 * \tparam T The content ObjectRef type.
 */
template <typename T,
          typename = typename std::enable_if<std::is_base_of<ObjectRef, T>::value>::type>
class Array : public ObjectRef {
 public:
  // constructors
  /*!
   * \brief default constructor
   */
  Array() { data_ = ArrayNode::Empty(); }

  /*!
   * \brief move constructor
   * \param other source
   */
  Array(Array<T>&& other) : ObjectRef() {  // NOLINT(*)
    data_ = std::move(other.data_);
  }

  /*!
   * \brief copy constructor
   * \param other source
   */
  Array(const Array<T>& other) : ObjectRef() {  // NOLINT(*)
    data_ = other.data_;
  }

  /*!
   * \brief constructor from pointer
   * \param n the container pointer
   */
  explicit Array(ObjectPtr<Object> n) : ObjectRef(n) {}

  /*!
   * \brief Constructor from iterator
   * \param first begin of iterator
   * \param last end of iterator
   * \tparam IterType The type of iterator
   */
  template <typename IterType>
  Array(IterType first, IterType last) {
    Assign(first, last);
  }

  /*!
   * \brief constructor from initializer list
   * \param init The initializer list
   */
  Array(std::initializer_list<T> init) {  // NOLINT(*)
    Assign(init.begin(), init.end());
  }

  /*!
   * \brief constructor from vector
   * \param init The vector
   */
  Array(const std::vector<T>& init) {  // NOLINT(*)
    Assign(init.begin(), init.end());
  }

  /*!
   * \brief Constructs a container with n elements. Each element is a copy of val
   * \param n The size of the container
   * \param val The init value
   */
  explicit Array(const size_t n, const T& val) { data_ = ArrayNode::CreateRepeated(n, val); }

  /*!
   * \brief move assign operator
   * \param other The source of assignment
   * \return reference to self.
   */
  Array<T>& operator=(Array<T>&& other) {
    data_ = std::move(other.data_);
    return *this;
  }

  /*!
   * \brief copy assign operator
   * \param other The source of assignment
   * \return reference to self.
   */
  Array<T>& operator=(const Array<T>& other) {
    data_ = other.data_;
    return *this;
  }
  /*!
   * \brief Create a runtime::Array and expose it as ObjectPtr
   * \tparam Args Type of argument list
   * \param args Argument list
   * \return The result ObjectPtr
   */
  template <typename... Args>
  static ObjectPtr<Object> CreateObjectPtr(Args&&... args) {
    Array<T> array(std::forward<Args>(args)...);
    ObjectPtr<Object> data = std::move(array.data_);
    return data;
  }

 public:
  // iterators
  struct ValueConverter {
    using ResultType = T;
    static T convert(const ObjectRef& n) { return DowncastNoCheck<T>(n); }
  };

  using iterator = IterAdapter<ValueConverter, const ObjectRef*>;
  using reverse_iterator = ReverseIterAdapter<ValueConverter, const ObjectRef*>;

  /*! \return begin iterator */
  iterator begin() const { return iterator(GetArrayNode()->begin()); }

  /*! \return end iterator */
  iterator end() const { return iterator(GetArrayNode()->end()); }

  /*! \return rbegin iterator */
  reverse_iterator rbegin() const {
    // ArrayNode::end() is never nullptr
    return reverse_iterator(GetArrayNode()->end() - 1);
  }

  /*! \return rend iterator */
  reverse_iterator rend() const {
    // ArrayNode::begin() is never nullptr
    return reverse_iterator(GetArrayNode()->begin() - 1);
  }

 public:
  // const methods in std::vector
  /*!
   * \brief Immutably read i-th element from array.
   * \param i The index
   * \return the i-th element.
   */
  const T operator[](int64_t i) const {
    ArrayNode* p = GetArrayNode();
    CHECK(p != nullptr) << "ValueError: cannot index a null array";
    CHECK(0 <= i && i < p->size_) << "IndexError: indexing " << i << " on an array of size "
                                  << p->size_;
    return DowncastNoCheck<T>(*(p->begin() + i));
  }

  /*! \return The size of the array */
  size_t size() const {
    ArrayNode* p = GetArrayNode();
    return p == nullptr ? 0 : GetArrayNode()->size_;
  }

  /*! \return The capacity of the array */
  size_t capacity() const {
    ArrayNode* p = GetArrayNode();
    return p == nullptr ? 0 : GetArrayNode()->capacity_;
  }

  /*! \return Whether array is empty */
  bool empty() const { return size() == 0; }

  /*! \return The first element of the array */
  const T front() const {
    ArrayNode* p = GetArrayNode();
    CHECK(p != nullptr) << "ValueError: cannot index a null array";
    CHECK_GT(p->size_, 0) << "IndexError: cannot index an empty array";
    return DowncastNoCheck<T>(*(p->begin()));
  }

  /*! \return The last element of the array */
  const T back() const {
    ArrayNode* p = GetArrayNode();
    CHECK(p != nullptr) << "ValueError: cannot index a null array";
    CHECK_GT(p->size_, 0) << "IndexError: cannot index an empty array";
    return DowncastNoCheck<T>(*(p->end() - 1));
  }

 public:
  // mutation in std::vector, implements copy-on-write

  /*!
   * \brief push a new item to the back of the list
   * \param item The item to be pushed.
   */
  void push_back(const T& item) {
    ArrayNode* p = CopyOnWrite(1);
    p->EmplaceInit(p->size_++, item);
  }

  /*!
   * \brief Insert an element into the given position
   * \param position An iterator pointing to the insertion point
   * \param val The element to insert
   */
  void insert(iterator position, const T& val) {
    CHECK(data_ != nullptr) << "ValueError: cannot insert a null array";
    int64_t idx = std::distance(begin(), position);
    int64_t size = GetArrayNode()->size_;
    auto addr = CopyOnWrite(1)                               //
                    ->EnlargeBy(1)                           //
                    ->MoveElementsRight(idx + 1, idx, size)  //
                    ->MutableBegin();
    new (addr + idx) ObjectRef(val);
  }

  /*!
   * \brief Insert a range of elements into the given position
   * \param position An iterator pointing to the insertion point
   * \param first The begin iterator of the range
   * \param last The end iterator of the range
   */
  template <typename IterType>
  void insert(iterator position, IterType first, IterType last) {
    if (first == last) {
      return;
    }
    CHECK(data_ != nullptr) << "ValueError: cannot insert a null array";
    int64_t idx = std::distance(begin(), position);
    int64_t size = GetArrayNode()->size_;
    int64_t numel = std::distance(first, last);
    CopyOnWrite(numel)
        ->EnlargeBy(numel)
        ->MoveElementsRight(idx + numel, idx, size)
        ->InitRange(idx, first, last);
  }

  /*! \brief Remove the last item of the list */
  void pop_back() {
    CHECK(data_ != nullptr) << "ValueError: cannot pop_back because array is null";
    int64_t size = GetArrayNode()->size_;
    CHECK_GT(size, 0) << "ValueError: cannot pop_back because array is empty";
    CopyOnWrite()->ShrinkBy(1);
  }

  /*!
   * \brief Erase an element on the given position
   * \param position An iterator pointing to the element to be erased
   */
  void erase(iterator position) {
    CHECK(data_ != nullptr) << "ValueError: cannot erase a null array";
    int64_t st = std::distance(begin(), position);
    int64_t size = GetArrayNode()->size_;
    CHECK(0 <= st && st < size) << "ValueError: cannot erase at index " << st
                                << ", because Array size is " << size;
    CopyOnWrite()                             //
        ->MoveElementsLeft(st, st + 1, size)  //
        ->ShrinkBy(1);
  }

  /*!
   * \brief Erase a given range of elements
   * \param first The begin iterator of the range
   * \param last The end iterator of the range
   */
  void erase(iterator first, iterator last) {
    if (first == last) {
      return;
    }
    CHECK(data_ != nullptr) << "ValueError: cannot erase a null array";
    int64_t size = GetArrayNode()->size_;
    int64_t st = std::distance(begin(), first);
    int64_t ed = std::distance(begin(), last);
    CHECK_LT(st, ed) << "ValueError: cannot erase array in range [" << st << ", " << ed << ")";
    CHECK(0 <= st && st <= size && 0 <= ed && ed <= size)
        << "ValueError: cannot erase array in range [" << st << ", " << ed << ")"
        << ", because array size is " << size;
    CopyOnWrite()                         //
        ->MoveElementsLeft(st, ed, size)  //
        ->ShrinkBy(ed - st);
  }

  /*!
   * \brief Resize the array.
   * \param n The new size.
   */
  void resize(int64_t n) {
    CHECK_GE(n, 0) << "ValueError: cannot resize an Array to negative size";
    if (data_ == nullptr) {
      SwitchContainer(n);
      return;
    }
    int64_t size = GetArrayNode()->size_;
    if (size < n) {
      CopyOnWrite(n - size)->EnlargeBy(n - size);
    } else if (size > n) {
      CopyOnWrite()->ShrinkBy(size - n);
    }
  }

  /*!
   * \brief Make sure the list has the capacity of at least n
   * \param n lower bound of the capacity
   */
  void reserve(int64_t n) {
    if (data_ == nullptr || n > GetArrayNode()->capacity_) {
      SwitchContainer(n);
    }
  }

  /*! \brief Release reference to all the elements */
  void clear() {
    if (data_ != nullptr) {
      ArrayNode* p = CopyOnWrite();
      p->clear();
    }
  }

 public:
  // Array's own methods

  /*!
   * \brief set i-th element of the array.
   * \param i The index
   * \param value The value to be setted.
   */
  void Set(int64_t i, T value) {
    ArrayNode* p = this->CopyOnWrite();
    CHECK(0 <= i && i < p->size_) << "IndexError: indexing " << i << " on an array of size "
                                  << p->size_;
    *(p->MutableBegin() + i) = std::move(value);
  }

  /*! \return The underlying ArrayNode */
  ArrayNode* GetArrayNode() const { return static_cast<ArrayNode*>(data_.get()); }

  /*!
   * \brief Helper function to apply fmutate to mutate an array.
   * \param fmutate The transformation function T -> T.
   * \tparam F the type of the mutation function.
   * \note This function performs copy on write optimization.
   */
  template <typename F>
  void MutateByApply(F fmutate) {
    if (data_ == nullptr) {
      return;
    }
    struct StackFrame {
      ArrayNode* p;
      ObjectRef* itr;
      int64_t i;
      int64_t size;
    };
    std::unique_ptr<StackFrame> s = std::make_unique<StackFrame>();
    s->p = GetArrayNode();
    s->itr = s->p->MutableBegin();
    s->i = 0;
    s->size = s->p->size_;
    if (!data_.unique()) {
      // Loop invariant: keeps iterating when
      // 1) data is not unique
      // 2) no elements are actually mutated yet
      for (; s->i < s->size; ++s->i, ++s->itr) {
        T new_elem = fmutate(DowncastNoCheck<T>(*s->itr));
        // do nothing when there is no mutation
        if (new_elem.same_as(*s->itr)) {
          continue;
        }
        // loop invariant breaks when the first real mutation happens
        // we copy the elements into a new unique array
        ObjectPtr<ArrayNode> copy = ArrayNode::CopyFrom(s->p->capacity_, s->p);
        s->itr = copy->MutableBegin() + (s->i++);
        *s->itr++ = std::move(new_elem);
        data_ = std::move(copy);
        // make sure `data_` is unique and break
        break;
      }
    }
    // when execution comes to this line, it is guaranteed that either
    //    1) i == size
    // or 2) data_.unique() is true
    for (; s->i < s->size; ++s->i, ++s->itr) {
      *s->itr = std::move(fmutate(std::move(DowncastNoCheck<T>(std::move(*s->itr)))));
    }
  }

  /*!
   * \brief reset the array to content from iterator.
   * \param first begin of iterator
   * \param last end of iterator
   * \tparam IterType The type of iterator
   */
  template <typename IterType>
  void Assign(IterType first, IterType last) {
    int64_t cap = std::distance(first, last);
    CHECK_GE(cap, 0) << "ValueError: cannot construct an Array of negative size";
    ArrayNode* p = GetArrayNode();
    if (p != nullptr && data_.unique() && p->capacity_ >= cap) {
      // do not have to make new space
      p->clear();
    } else {
      // create new space
      data_ = ArrayNode::Empty(cap);
      p = GetArrayNode();
    }
    // To ensure exception safety, size is only incremented after the initialization succeeds
    ObjectRef* itr = p->MutableBegin();
    for (int64_t& i = p->size_ = 0; i < cap; ++i, ++first, ++itr) {
      new (itr) ObjectRef(*first);
    }
  }

  /*!
   * \brief Copy on write semantics
   *  Do nothing if current handle is the unique copy of the array.
   *  Otherwise make a new copy of the array to ensure the current handle
   *  hold a unique copy.
   *
   * \return Handle to the internal node container(which ganrantees to be unique)
   */
  ArrayNode* CopyOnWrite() {
    if (data_ == nullptr) {
      return SwitchContainer(ArrayNode::kInitSize);
    }
    if (!data_.unique()) {
      return SwitchContainer(capacity());
    }
    return static_cast<ArrayNode*>(data_.get());
  }

  /*! \brief specify container node */
  using ContainerType = ArrayNode;

 private:
  /*!
   * \brief Implement copy-on-write semantics, and ensures capacity is enough for extra elements.
   * \param reserve_extra Number of extra slots needed
   * \return ArrayNode pointer to the unique copy
   */
  ArrayNode* CopyOnWrite(int64_t reserve_extra) {
    ArrayNode* p = GetArrayNode();
    if (p == nullptr) {
      // necessary to get around the constexpr address issue before c++17
      const int64_t kInitSize = ArrayNode::kInitSize;
      return SwitchContainer(std::max(kInitSize, reserve_extra));
    }
    if (p->capacity_ >= p->size_ + reserve_extra) {
      return CopyOnWrite();
    }
    int64_t cap = p->capacity_ * ArrayNode::kIncFactor;
    cap = std::max(cap, p->size_ + reserve_extra);
    return SwitchContainer(cap);
  }

  /*!
   * \brief Move or copy the ArrayNode to new address with the given capacity
   * \param capacity The capacity requirement of the new address
   */
  ArrayNode* SwitchContainer(int64_t capacity) {
    if (data_ == nullptr) {
      data_ = ArrayNode::Empty(capacity);
    } else if (data_.unique()) {
      data_ = ArrayNode::MoveFrom(capacity, GetArrayNode());
    } else {
      data_ = ArrayNode::CopyFrom(capacity, GetArrayNode());
    }
    return static_cast<ArrayNode*>(data_.get());
  }
};

// Specialize make_object<ArrayNode> to make sure it is correct.
template <>
inline ObjectPtr<ArrayNode> make_object() {
  return ArrayNode::Empty();
}

/*! \brief An object representing a structure or enumeration. */
class ADTObj : public Object, public InplaceArrayBase<ADTObj, ObjectRef> {
 public:
  /*! \brief The tag representing the constructor used. */
  int32_t tag;
  /*! \brief Number of fields in the ADT object. */
  uint32_t size;
  // The fields of the structure follows directly in memory.

  static constexpr const uint32_t _type_index = TypeIndex::kRuntimeADT;
  static constexpr const char* _type_key = "runtime.ADT";
  TVM_DECLARE_FINAL_OBJECT_INFO(ADTObj, Object);

 private:
  /*!
   * \return The number of elements in the array.
   */
  size_t GetSize() const { return size; }

  /*!
   * \brief Initialize the elements in the array.
   *
   * \tparam Iterator Iterator type of the array.
   * \param begin The begin iterator.
   * \param end The end iterator.
   */
  template <typename Iterator>
  void Init(Iterator begin, Iterator end) {
    size_t num_elems = std::distance(begin, end);
    this->size = 0;
    auto it = begin;
    for (size_t i = 0; i < num_elems; ++i) {
      InplaceArrayBase::EmplaceInit(i, *it++);
      // Only increment size after the initialization succeeds
      this->size++;
    }
  }

  friend class ADT;
  friend InplaceArrayBase<ADTObj, ObjectRef>;
};

/*! \brief reference to algebraic data type objects. */
class ADT : public ObjectRef {
 public:
  /*!
   * \brief construct an ADT object reference.
   * \param tag The tag of the ADT object.
   * \param fields The fields of the ADT object.
   * \return The constructed ADT object reference.
   */
  ADT(int32_t tag, std::vector<ObjectRef> fields) : ADT(tag, fields.begin(), fields.end()){};

  /*!
   * \brief construct an ADT object reference.
   * \param tag The tag of the ADT object.
   * \param begin The begin iterator to the start of the fields array.
   * \param end The end iterator to the end of the fields array.
   * \return The constructed ADT object reference.
   */
  template <typename Iterator>
  ADT(int32_t tag, Iterator begin, Iterator end) {
    size_t num_elems = std::distance(begin, end);
    auto ptr = make_inplace_array_object<ADTObj, ObjectRef>(num_elems);
    ptr->tag = tag;
    ptr->Init(begin, end);
    data_ = std::move(ptr);
  }

  /*!
   * \brief construct an ADT object reference.
   * \param tag The tag of the ADT object.
   * \param init The initializer list of fields.
   * \return The constructed ADT object reference.
   */
  ADT(int32_t tag, std::initializer_list<ObjectRef> init) : ADT(tag, init.begin(), init.end()){};

  /*!
   * \brief Access element at index.
   *
   * \param idx The array index
   * \return const ObjectRef
   */
  const ObjectRef& operator[](size_t idx) const { return operator->()->operator[](idx); }

  /*!
   * \brief Return the ADT tag.
   */
  int32_t tag() const { return operator->()->tag; }

  /*!
   * \brief Return the number of fields.
   */
  size_t size() const { return operator->()->size; }

  /*!
   * \brief Construct a tuple object.
   *
   * \tparam Args Type params of tuple feilds.
   * \param args Tuple fields.
   * \return ADT The tuple object reference.
   */
  template <typename... Args>
  static ADT Tuple(Args&&... args) {
    return ADT(0, std::forward<Args>(args)...);
  }

  TVM_DEFINE_OBJECT_REF_METHODS(ADT, ObjectRef, ADTObj);
};

/*! \brief An object representing string. It's POD type. */
class StringObj : public Object {
 public:
  /*! \brief The pointer to string data. */
  const char* data;

  /*! \brief The length of the string object. */
  uint64_t size;

  static constexpr const uint32_t _type_index = TypeIndex::kRuntimeString;
  static constexpr const char* _type_key = "runtime.String";
  TVM_DECLARE_FINAL_OBJECT_INFO(StringObj, Object);

 private:
  /*! \brief String object which is moved from std::string container. */
  class FromStd;

  friend class String;
};

/*!
 * \brief Reference to string objects.
 *
 * \code
 *
 * // Example to create runtime String reference object from std::string
 * std::string s = "hello world";
 *
 * // You can create the reference from existing std::string
 * String ref{std::move(s)};
 *
 * // You can rebind the reference to another string.
 * ref = std::string{"hello world2"};
 *
 * // You can use the reference as hash map key
 * std::unordered_map<String, int32_t> m;
 * m[ref] = 1;
 *
 * // You can compare the reference object with other string objects
 * assert(ref == "hello world", true);
 *
 * // You can convert the reference to std::string again
 * string s2 = (string)ref;
 *
 * \endcode
 */
class String : public ObjectRef {
 public:
  /*!
   * \brief Construct an empty string.
   */
  String() : String(std::string()) {}
  /*!
   * \brief Construct a new String object
   *
   * \param other The moved/copied std::string object
   *
   * \note If user passes const reference, it will trigger copy. If it's rvalue,
   * it will be moved into other.
   */
  String(std::string other);  // NOLINT(*)

  /*!
   * \brief Construct a new String object
   *
   * \param other a char array.
   */
  String(const char* other)  // NOLINT(*)
      : String(std::string(other)) {}

  /*!
   * \brief Change the value the reference object points to.
   *
   * \param other The value for the new String
   *
   */
  inline String operator=(std::string other);

  /*!
   * \brief Compare is less than other std::string
   *
   * \param other The other string
   *
   * \return the comparison result
   */
  bool operator<(const std::string& other) const { return this->compare(other) < 0; }
  bool operator<(const String& other) const { return this->compare(other) < 0; }
  bool operator<(const char* other) const { return this->compare(other) < 0; }

  /*!
   * \brief Compare is greater than other std::string
   *
   * \param other The other string
   *
   * \return the comparison result
   */
  bool operator>(const std::string& other) const { return this->compare(other) > 0; }
  bool operator>(const String& other) const { return this->compare(other) > 0; }
  bool operator>(const char* other) const { return this->compare(other) > 0; }

  /*!
   * \brief Compare is less than or equal to other std::string
   *
   * \param other The other string
   *
   * \return the comparison result
   */
  bool operator<=(const std::string& other) const { return this->compare(other) <= 0; }
  bool operator<=(const String& other) const { return this->compare(other) <= 0; }
  bool operator<=(const char* other) const { return this->compare(other) <= 0; }

  /*!
   * \brief Compare is greater than or equal to other std::string
   *
   * \param other The other string
   *
   * \return the comparison result
   */
  bool operator>=(const std::string& other) const { return this->compare(other) >= 0; }
  bool operator>=(const String& other) const { return this->compare(other) >= 0; }
  bool operator>=(const char* other) const { return this->compare(other) >= 0; }

  /*!
   * \brief Compare is equal to other std::string
   *
   * \param other The other string
   *
   * \return the comparison result
   */
  bool operator==(const std::string& other) const { return this->compare(other) == 0; }
  bool operator==(const String& other) const { return this->compare(other) == 0; }
  bool operator==(const char* other) const { return compare(other) == 0; }

  /*!
   * \brief Compare is not equal to other std::string
   *
   * \param other The other string
   *
   * \return the comparison result
   */
  bool operator!=(const std::string& other) const { return this->compare(other) != 0; }
  bool operator!=(const String& other) const { return this->compare(other) != 0; }
  bool operator!=(const char* other) const { return this->compare(other) != 0; }

  /*!
   * \brief Compares this String object to other
   *
   * \param other The String to compare with.
   *
   * \return zero if both char sequences compare equal. negative if this appear
   * before other, positive otherwise.
   */
  int compare(const String& other) const {
    return memncmp(data(), other.data(), size(), other.size());
  }

  /*!
   * \brief Compares this String object to other
   *
   * \param other The string to compare with.
   *
   * \return zero if both char sequences compare equal. negative if this appear
   * before other, positive otherwise.
   */
  int compare(const std::string& other) const {
    return memncmp(data(), other.data(), size(), other.size());
  }

  /*!
   * \brief Compares this to other
   *
   * \param other The character array to compare with.
   *
   * \return zero if both char sequences compare equal. negative if this appear
   * before other, positive otherwise.
   */
  int compare(const char* other) const {
    return memncmp(data(), other, size(), std::strlen(other));
  }

  /*!
   * \brief Returns a pointer to the char array in the string.
   *
   * \return const char*
   */
  const char* c_str() const { return get()->data; }

  /*!
   * \brief Return the length of the string
   *
   * \return size_t string length
   */
  size_t size() const {
    const auto* ptr = get();
    return ptr->size;
  }

  /*!
   * \brief Return the length of the string
   *
   * \return size_t string length
   */
  size_t length() const { return size(); }

  /*!
   * \brief Retun if the string is empty
   *
   * \return true if empty, false otherwise.
   */
  bool empty() const { return size() == 0; }

  /*!
   * \brief Return the data pointer
   *
   * \return const char* data pointer
   */
  const char* data() const { return get()->data; }

  /*!
   * \brief Convert String to an std::sting object
   *
   * \return std::string
   */
  operator std::string() const { return std::string{get()->data, size()}; }

  /*!
   * \brief Check if a TVMArgValue can be converted to String, i.e. it can be std::string or String
   * \param val The value to be checked
   * \return A boolean indicating if val can be converted to String
   */
  static bool CanConvertFrom(const TVMArgValue& val) {
    return val.type_code() == kTVMStr || val.IsObjectRef<tvm::runtime::String>();
  }

  /*!
   * \brief Hash the binary bytes
   * \param data The data pointer
   * \param size The size of the bytes.
   * \return the hash value.
   */
  static size_t HashBytes(const char* data, size_t size) {
    // This function falls back to string copy with c++11 compiler and is
    // recommended to be compiled with c++14
#if TVM_USE_CXX17_STRING_VIEW_HASH
    return std::hash<std::string_view>()(std::string_view(data, size));
#elif TVM_USE_CXX14_STRING_VIEW_HASH
    return std::hash<std::experimental::string_view>()(std::experimental::string_view(data, size));
#else
    return std::hash<std::string>()(std::string(data, size));
#endif
  }

  /*! \return the internal StringObj pointer */
  const StringObj* get() const { return operator->(); }

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(String, ObjectRef, StringObj);

 private:
  /*!
   * \brief Compare two char sequence
   *
   * \param lhs Pointers to the char array to compare
   * \param rhs Pointers to the char array to compare
   * \param lhs_count Length of the char array to compare
   * \param rhs_count Length of the char array to compare
   * \return int zero if both char sequences compare equal. negative if this
   * appear before other, positive otherwise.
   */
  static int memncmp(const char* lhs, const char* rhs, size_t lhs_count, size_t rhs_count);

  friend struct tvm::runtime::ObjectEqual;
};

/*! \brief An object representing string moved from std::string. */
class StringObj::FromStd : public StringObj {
 public:
  /*!
   * \brief Construct a new FromStd object
   *
   * \param other The moved/copied std::string object
   *
   * \note If user passes const reference, it will trigger copy. If it's rvalue,
   * it will be moved into other.
   */
  explicit FromStd(std::string other) : data_container{other} {}

 private:
  /*! \brief Container that holds the memory. */
  std::string data_container;

  friend class String;
};

inline String::String(std::string other) {
  auto ptr = make_object<StringObj::FromStd>(std::move(other));
  ptr->size = ptr->data_container.size();
  ptr->data = ptr->data_container.data();
  data_ = std::move(ptr);
}

inline String String::operator=(std::string other) {
  String replace{std::move(other)};
  data_.swap(replace.data_);
  return Downcast<String>(*this);
}

inline String operator+(const std::string lhs, const String& rhs) {
  return lhs + rhs.operator std::string();
}

inline std::ostream& operator<<(std::ostream& out, const String& input) {
  out.write(input.data(), input.size());
  return out;
}

inline int String::memncmp(const char* lhs, const char* rhs, size_t lhs_count, size_t rhs_count) {
  if (lhs == rhs && lhs_count == rhs_count) return 0;

  for (size_t i = 0; i < lhs_count && i < rhs_count; ++i) {
    if (lhs[i] < rhs[i]) return -1;
    if (lhs[i] > rhs[i]) return 1;
  }
  if (lhs_count < rhs_count) {
    return -1;
  } else if (lhs_count > rhs_count) {
    return 1;
  } else {
    return 0;
  }
}

inline size_t ObjectHash::operator()(const ObjectRef& a) const {
  if (const auto* str = a.as<StringObj>()) {
    return String::HashBytes(str->data, str->size);
  }
  return ObjectPtrHash()(a);
}

inline bool ObjectEqual::operator()(const ObjectRef& a, const ObjectRef& b) const {
  if (a.same_as(b)) {
    return true;
  }
  if (const auto* str_a = a.as<StringObj>()) {
    if (const auto* str_b = b.as<StringObj>()) {
      return String::memncmp(str_a->data, str_b->data, str_a->size, str_b->size) == 0;
    }
  }
  return false;
}

template <>
struct PackedFuncValueConverter<::tvm::runtime::String> {
  static String From(const TVMArgValue& val) {
    if (val.IsObjectRef<tvm::runtime::String>()) {
      return val.AsObjectRef<tvm::runtime::String>();
    } else {
      return tvm::runtime::String(val.operator std::string());
    }
  }

  static String From(const TVMRetValue& val) {
    if (val.IsObjectRef<tvm::runtime::String>()) {
      return val.AsObjectRef<tvm::runtime::String>();
    } else {
      return tvm::runtime::String(val.operator std::string());
    }
  }
};

/*! \brief Helper to represent nullptr for optional. */
struct NullOptType {};

/*!
 * \brief Optional container that to represent to a Nullable variant of T.
 * \tparam T The original ObjectRef.
 *
 * \code
 *
 *  Optional<String> opt0 = nullptr;
 *  Optional<String> opt1 = String("xyz");
 *  CHECK(opt0 == nullptr);
 *  CHECK(opt1 == "xyz");
 *
 * \endcode
 */
template <typename T>
class Optional : public ObjectRef {
 public:
  using ContainerType = typename T::ContainerType;
  static_assert(std::is_base_of<ObjectRef, T>::value, "Optional is only defined for ObjectRef.");
  // default constructors.
  Optional() = default;
  Optional(const Optional<T>&) = default;
  Optional(Optional<T>&&) = default;
  Optional<T>& operator=(const Optional<T>&) = default;
  Optional<T>& operator=(Optional<T>&&) = default;
  /*!
   * \brief Construct from an ObjectPtr
   *        whose type already matches the ContainerType.
   * \param ptr
   */
  explicit Optional(ObjectPtr<Object> ptr) : ObjectRef(ptr) {}
  /*! \brief Nullopt handling */
  Optional(NullOptType) {}  // NOLINT(*)
  // nullptr handling.
  // disallow implicit conversion as 0 can be implicitly converted to nullptr_t
  explicit Optional(std::nullptr_t) {}
  Optional<T>& operator=(std::nullptr_t) {
    data_ = nullptr;
    return *this;
  }
  // normal value handling.
  Optional(T other)  // NOLINT(*)
      : ObjectRef(std::move(other)) {}
  Optional<T>& operator=(T other) {
    ObjectRef::operator=(std::move(other));
    return *this;
  }
  // delete the int constructor
  // since Optional<Integer>(0) is ambiguious
  // 0 can be implicitly casted to nullptr_t
  explicit Optional(int val) = delete;
  Optional<T>& operator=(int val) = delete;
  /*!
   * \return A not-null container value in the optional.
   * \note This function performs not-null checking.
   */
  T value() const {
    CHECK(data_ != nullptr);
    return T(data_);
  }
  /*!
   * \return The contained value if the Optional is not null
   *         otherwise return the default_value.
   */
  T value_or(T default_value) const { return data_ != nullptr ? T(data_) : default_value; }
  /*! \return Whether the container is not nullptr.*/
  explicit operator bool() const { return *this != nullptr; }
  // operator overloadings
  bool operator==(std::nullptr_t) const { return data_ == nullptr; }
  bool operator!=(std::nullptr_t) const { return data_ != nullptr; }
  auto operator==(const Optional<T>& other) const {
    // support case where sub-class returns a symbolic ref type.
    using RetType = decltype(value() == other.value());
    if (same_as(other)) return RetType(true);
    if (*this != nullptr && other != nullptr) {
      return value() == other.value();
    } else {
      // one of them is nullptr.
      return RetType(false);
    }
  }
  auto operator!=(const Optional<T>& other) const {
    // support case where sub-class returns a symbolic ref type.
    using RetType = decltype(value() != other.value());
    if (same_as(other)) return RetType(false);
    if (*this != nullptr && other != nullptr) {
      return value() != other.value();
    } else {
      // one of them is nullptr.
      return RetType(true);
    }
  }
  auto operator==(const T& other) const {
    using RetType = decltype(value() == other);
    if (same_as(other)) return RetType(true);
    if (*this != nullptr) return value() == other;
    return RetType(false);
  }
  auto operator!=(const T& other) const { return !(*this == other); }
  template <typename U>
  auto operator==(const U& other) const {
    using RetType = decltype(value() == other);
    if (*this == nullptr) return RetType(false);
    return value() == other;
  }
  template <typename U>
  auto operator!=(const U& other) const {
    using RetType = decltype(value() != other);
    if (*this == nullptr) return RetType(true);
    return value() != other;
  }
  static constexpr bool _type_is_nullable = true;
};

template <typename T>
struct PackedFuncValueConverter<Optional<T>> {
  static Optional<T> From(const TVMArgValue& val) {
    if (val.type_code() == kTVMNullptr) return Optional<T>(nullptr);
    return PackedFuncValueConverter<T>::From(val);
  }
  static Optional<T> From(const TVMRetValue& val) {
    if (val.type_code() == kTVMNullptr) return Optional<T>(nullptr);
    return PackedFuncValueConverter<T>::From(val);
  }
};

/*! \brief map node content */
class MapNode : public Object {
  /*! \brief The number of elements in a memory block */
  static constexpr int kBlockCap = 16;
  /*! \brief Maximum load factor of the hash map */
  static constexpr double kMaxLoadFactor = 0.99;
  /*! \brief Binary representation of the metadata of an empty slot */
  static constexpr uint8_t kEmptySlot = uint8_t(0b11111111);
  /*! \brief Binary representation of the metadata of a protected slot */
  static constexpr uint8_t kProtectedSlot = uint8_t(0b11111110);
  /*! \brief Number of probing choices available */
  static constexpr int kNumJumpDists = 126;
  /* clang-format off */
  /*! \brief Candidates of probing distance */
  TVM_DLL static constexpr uint64_t kJumpDists[kNumJumpDists] {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    // Quadratic probing with triangle numbers. See also:
    // 1) https://en.wikipedia.org/wiki/Quadratic_probing
    // 2) https://fgiesen.wordpress.com/2015/02/22/triangular-numbers-mod-2n/
    // 3) https://github.com/skarupke/flat_hash_map
    21, 28, 36, 45, 55, 66, 78, 91, 105, 120,
    136, 153, 171, 190, 210, 231, 253, 276, 300, 325,
    351, 378, 406, 435, 465, 496, 528, 561, 595, 630,
    666, 703, 741, 780, 820, 861, 903, 946, 990, 1035,
    1081, 1128, 1176, 1225, 1275, 1326, 1378, 1431, 1485, 1540,
    1596, 1653, 1711, 1770, 1830, 1891, 1953, 2016, 2080, 2145,
    2211, 2278, 2346, 2415, 2485, 2556, 2628,
    // larger triangle numbers
    8515, 19110, 42778, 96141, 216153,
    486591, 1092981, 2458653, 5532801, 12442566,
    27993903, 62983476, 141717030, 318844378, 717352503,
    1614057336, 3631522476, 8170957530, 18384510628, 41364789378,
    93070452520, 209408356380, 471168559170, 1060128894105, 2385289465695,
    5366898840628, 12075518705635, 27169915244790, 61132312065111, 137547689707000,
    309482283181501, 696335127828753, 1566753995631385, 3525196511162271, 7931691992677701,
    17846306936293605, 40154190677507445, 90346928918121501, 203280589587557251, 457381325854679626,
    1029107982097042876, 2315492959180353330, 5209859154120846435,
  };
  /* clang-format on */

 public:
  /*! \brief Type of the keys in the hash map */
  using key_type = ObjectRef;
  /*! \brief Type of the values in the hash map */
  using mapped_type = ObjectRef;

  static constexpr const uint32_t _type_index = TypeIndex::kRuntimeMap;
  static constexpr const char* _type_key = "Map";
  TVM_DECLARE_FINAL_OBJECT_INFO(MapNode, Object);

 private:
  struct KVType;
  struct Block;
  struct ListNode;

 public:
  class iterator;

  /*!
   * \brief Destroy the MapNode
   */
  ~MapNode() { this->Reset(); }

  /*!
   * \brief Number of elements in the MapNode
   * \return The result
   */
  size_t size() const { return size_; }

  /*!
   * \brief Count the number of times a key exists in the MapNode
   * \param key The indexing key
   * \return The result, 0 or 1
   */
  size_t count(const key_type& key) const { return !Search(key).IsNone(); }

  /*!
   * \brief Index value associated with a key, throw exception if the key does not exist
   * \param key The indexing key
   * \return The const reference to the value
   */
  const mapped_type& at(const key_type& key) const { return At(key); }

  /*!
   * \brief Index value associated with a key, throw exception if the key does not exist
   * \param key The indexing key
   * \return The mutable reference to the value
   */
  mapped_type& at(const key_type& key) { return At(key); }

  /*! \return begin iterator */
  iterator begin() const { return size_ == 0 ? iterator() : iterator(0, this); }

  /*! \return end iterator */
  iterator end() const { return size_ == 0 ? iterator() : iterator(slots_ + 1, this); }

  /*!
   * \brief Index value associated with a key
   * \param key The indexing key
   * \return The iterator of the entry associated with the key, end iterator if not exists
   */
  iterator find(const key_type& key) const {
    ListNode n = Search(key);
    return n.IsNone() ? end() : iterator(n.i, this);
  }

  /*!
   * \brief Erase the entry associated with the key, do nothing if not exists
   * \param key The indexing key
   */
  void erase(const key_type& key) { Erase(key); }

  /*!
   * \brief Erase the entry associated with the iterator
   * \param position The iterator
   */
  void erase(const iterator& position) {
    uint64_t i = position.i;
    if (position.self != nullptr && i <= this->slots_) {
      Erase(ListNode(i, this));
    }
  }

 private:
  /*!
   * \brief Insert and construct in-place with the given args, do nothing if key already exists
   * \tparam Args Type of the args forwarded to the constructor
   */
  template <typename... Args>
  void emplace(Args&&... args) {
    Emplace(std::forward<Args>(args)...);
  }

  /*!
   * \brief Index value associated with a key, create new entry if the key does not exist
   * \param key The indexing key
   * \return The mutable reference to the value
   */
  mapped_type& operator[](const key_type& key) { return Emplace(key, mapped_type()).Val(); }

  /*!
   * \brief reset the array to content from iterator.
   * \param first begin of iterator
   * \param last end of iterator
   * \tparam IterType The type of iterator
   */
  template <typename IterType>
  void Assign(IterType first, IterType last) {
    int64_t cap = std::distance(first, last);
    this->ReleaseItems();
    this->Reserve(cap);
    for (; first != last; ++first) {
      this->Emplace(*first);
    }
  }

  /*!
   * \brief Search for the given key
   * \param key The key
   * \return ListNode that associated with the key
   */
  ListNode Search(const key_type& key) const {
    if (this->size_ == 0) {
      return ListNode();
    }
    for (ListNode n = ListNode::GetHead(ObjectHash()(key), this); !n.IsNone(); n.MoveToNext(this)) {
      if (ObjectEqual()(key, n.Key())) {
        return n;
      }
    }
    return ListNode();
  }

  /*!
   * \brief Search for the given key, throw exception if not exists
   * \param key The key
   * \return ListNode that associated with the key
   */
  mapped_type& At(const key_type& key) const {
    ListNode n = Search(key);
    CHECK(!n.IsNone()) << "IndexError: key is not in Map";
    return n.Val();
  }

  /*!
   * \brief In-place construct an entry, or do nothing if already exists
   * \tparam Item Type of arguments forwarded to the constructor
   * \param arg Arguments fed to the constructor
   * \return ListNode that associated with the key, no matter whether it already exists
   */
  template <typename Item>
  ListNode Emplace(Item&& arg) {
    KVType item(std::forward<Item>(arg));
    return Emplace(std::move(item.k), std::move(item.v));
  }

  /*!
   * \brief In-place construct an entry, or do nothing if already exists
   * \tparam Key Type of the key
   * \tparam Args Type of the rest of the arguments fed to the constructor
   * \param key The indexing key
   * \param args Other arguments
   * \return ListNode that associated with the key, no matter whether it already exists
   */
  template <typename Key, typename... Args>
  ListNode Emplace(Key&& key, Args&&... args) {
    ReHashIfNone();
    // required that `m` to be the head of a linked list through which we can iterator
    ListNode m = ListNode::FromHash(ObjectHash()(key), this);
    // `m` can be: 1) empty; 2) body of an irrelevant list; 3) head of the relevant list
    // Case 1: empty
    if (m.IsEmpty()) {
      KVType v(std::forward<Key>(key), std::forward<Args>(args)...);
      m.NewHead(std::move(v));
      this->size_ += 1;
      return m;
    }
    // Case 2: body of an irrelevant list
    if (!m.IsHead()) {
      // we move the elements around and construct the single-elements linked list
      return SpareListHead(std::move(m), std::forward<Key>(key), std::forward<Args>(args)...);
    }
    // Case 3: head of the relevant list
    // we iterate through the linked list until the end
    ListNode n = m;
    do {
      // find equal item, do not insert
      if (ObjectEqual()(key, n.Key())) {
        return n;
      }
      // make sure `m` is the previous element of `n`
      m = n;
    } while (n.MoveToNext(this));
    // `m` is the tail of the linked list
    // always check capacity before insertion
    if (ReHashIfFull()) {
      return Emplace(std::forward<Key>(key), std::forward<Args>(args)...);
    }
    uint8_t jump;
    ListNode empty;
    // rehash if there is no empty space after `m`
    if (ReHashIfNoNextEmpty(m, &empty, &jump)) {
      return Emplace(std::forward<Key>(key), std::forward<Args>(args)...);
    }
    KVType v(std::forward<Key>(key), std::forward<Args>(args)...);
    empty.NewTail(std::move(v));
    // link `n` to `empty`, and move forward
    m.SetJump(jump);
    this->size_ += 1;
    return empty;
  }

  /*!
   * \brief Spare an entry to be the head of a linked list
   * \tparam Args Type of the arguments fed to the constructor
   * \param n The given entry to be spared
   * \param args The arguments
   * \return ListNode that associated with the head
   */
  template <typename... Args>
  ListNode SpareListHead(ListNode n, Args&&... args) {
    // `n` is not the head of the linked list
    // move the original item of `n` (if any)
    // and construct new item on the position `n`
    if (ReHashIfFull()) {
      // always check capacity before insertion
      return Emplace(std::forward<Args>(args)...);
    }
    // To make `n` empty, we
    // 1) find `w` the previous element of `n` in the linked list
    // 2) copy the linked list starting from `r = n`
    // 3) paste them after `w`
    // read from the linked list after `r`
    ListNode r = n;
    // write to the tail of `w`
    ListNode w = n.GetPrev(this);
    // after `n` is moved, we disallow writing to the slot
    bool is_first = true;
    uint8_t r_meta, jump;
    ListNode empty;
    do {
      // `jump` describes how `w` is jumped to `empty`
      // rehash if there is no empty space after `w`
      if (ReHashIfNoNextEmpty(w, &empty, &jump)) {
        return Emplace(std::forward<Args>(args)...);
      }
      // move `r` to `empty`
      empty.NewTail(std::move(r.Data()));
      // clear the metadata of `r`
      r_meta = r.Meta();
      if (is_first) {
        is_first = false;
        r.SetProtected();
      } else {
        r.SetEmpty();
      }
      // link `w` to `empty`, and move forward
      w.SetJump(jump);
      w = empty;
      // move `r` forward as well
    } while (r.MoveToNext(this, r_meta));
    // finally we have done moving the linked list
    // fill data_ into `n`
    KVType v(std::forward<Args>(args)...);
    n.NewHead(std::move(v));
    this->size_ += 1;
    return n;
  }

  /*!
   * \brief Remove a ListNode
   * \param n The node to be removed
   */
  void Erase(const ListNode& n) {
    this->size_ -= 1;
    if (!n.HasNext()) {
      // `n` is the last
      if (!n.IsHead()) {
        // cut the link if there is any
        n.GetPrev(this).SetJump(0);
      }
      n.Data().KVType::~KVType();
      n.SetEmpty();
    } else {
      ListNode last = n, prev = n;
      for (last.MoveToNext(this); last.HasNext(); prev = last, last.MoveToNext(this)) {
      }
      n.Data() = std::move(last.Data());
      last.SetEmpty();
      prev.SetJump(0);
    }
  }

  /*!
   * \brief Remove an entry associated with the given key
   * \param key The node to be removed
   */
  void Erase(const key_type& key) {
    ListNode n = Search(key);
    if (!n.IsNone()) {
      Erase(n);
    }
  }

  /*!
   * \brief Reserve some space
   * \param count The space to be reserved
   */
  void Reserve(uint64_t count) {
    if (slots_ < count * 2) {
      ReHash(count * 2);
    }
  }

  /*! \brief Clear the container to empty, release all memory acquired */
  void Reset() {
    this->ReleaseItems();
    delete[] data_;
    data_ = nullptr;
    slots_ = 0;
    size_ = 0;
    fib_ = 63;
  }

  /*! \brief Clear the container to empty, release all entries */
  void ReleaseItems() {
    uint64_t n_blocks = CalcNumBlocks(this->slots_);
    MapNode* m = this;
    for (uint64_t bi = 0; bi < n_blocks; ++bi) {
      uint8_t* m_m = m->data_[bi].b;
      KVType* m_d = reinterpret_cast<KVType*>(m->data_[bi].b + kBlockCap);
      for (int j = 0; j < kBlockCap; ++j, ++m_m, ++m_d) {
        uint8_t& meta = *m_m;
        if (meta != uint8_t(kProtectedSlot) && meta != uint8_t(kEmptySlot)) {
          meta = uint8_t(kEmptySlot);
          m_d->KVType::~KVType();
        }
      }
    }
    this->size_ = 0;
  }

  /*!
   * \brief Re-hashing with the given required capacity
   * \param required The lower bound of capacity required
   */
  void ReHash(uint64_t required) {
    constexpr uint64_t one = 1;
    uint64_t new_n_slots = static_cast<uint64_t>(required / kMaxLoadFactor) + 1;
    new_n_slots = std::max(new_n_slots, required);
    if (new_n_slots <= 0) {
      return;
    }
    uint8_t new_fib = __builtin_clzll(new_n_slots);
    new_n_slots = one << (64 - new_fib);
    if (new_n_slots <= slots_ + 1) {
      return;
    }
    ObjectPtr<MapNode> p = MoveFrom(new_fib, new_n_slots, this);
    std::swap(p->data_, this->data_);
    std::swap(p->slots_, this->slots_);
    std::swap(p->size_, this->size_);
    std::swap(p->fib_, this->fib_);
  }

  /*!
   * \brief Re-hashing if the container is empty
   * \return If re-hashing happens
   */
  bool ReHashIfNone() {
    constexpr uint64_t min_size = 7;
    if (slots_ == 0) {
      ReHash(min_size);
      return true;
    }
    return false;
  }

  /*!
   * \brief Re-hashing if the container achieves max load factor
   * \return If re-hashing happens
   */
  bool ReHashIfFull() {
    constexpr uint64_t min_size = 7;
    if (slots_ == 0 || size_ + 1 > (slots_ + 1) * kMaxLoadFactor) {
      ReHash(std::max(min_size, size_ + 1));
      return true;
    }
    return false;
  }

  /*!
   * \brief Re-hashing if cannot find space for the linked-list
   * \param n Find the next empty element of this node
   * \param empty The resulting empty element
   * \param jump Jump required to the empty element
   * \return If re-hashing happens
   */
  bool ReHashIfNoNextEmpty(const ListNode& n, ListNode* empty, uint8_t* jump) {
    constexpr uint64_t min_size = 7;
    *empty = n.GetNextEmpty(this, jump);
    if (empty->IsNone()) {
      ReHash(std::max(min_size, slots_ * 2 + 1));
      return true;
    }
    return false;
  }

 public:
  /*!
   * \brief Create an empty container
   * \return The object created
   */
  static ObjectPtr<MapNode> Empty() {
    ObjectPtr<MapNode> p = make_object<MapNode>();
    p->data_ = nullptr;
    p->slots_ = 0;
    p->size_ = 0;
    p->fib_ = 63;
    return p;
  }

 private:
  /*!
   * \brief Create an empty container
   * \param fib The fib shift provided
   * \param n_slots Number of slots required
   * \return The object created
   */
  static ObjectPtr<MapNode> Empty(uint8_t fib, uint64_t n_slots) {
    if (n_slots == 0) {
      return Empty();
    }
    ObjectPtr<MapNode> p = make_object<MapNode>();
    uint64_t n_blocks = CalcNumBlocks(n_slots - 1);
    Block* block = p->data_ = new Block[n_blocks];
    p->slots_ = n_slots - 1;
    p->size_ = 0;
    p->fib_ = fib;
    for (uint64_t i = 0; i < n_blocks; ++i, ++block) {
      std::fill(block->b, block->b + kBlockCap, uint8_t(kEmptySlot));
    }
    return p;
  }

  /*!
   * \brief Create an empty container with elements moving from another MapNode
   * \param fib The fib shift provided
   * \param n_slots Number of slots required
   * \param m The source container
   * \return The object created
   */
  static ObjectPtr<MapNode> MoveFrom(uint8_t fib, uint64_t n_slots, MapNode* m) {
    ObjectPtr<MapNode> p = MapNode::Empty(fib, n_slots);
    uint64_t n_blocks = CalcNumBlocks(m->slots_);
    for (uint64_t bi = 0; bi < n_blocks; ++bi) {
      uint8_t* m_m = m->data_[bi].b;
      KVType* m_d = reinterpret_cast<KVType*>(m->data_[bi].b + kBlockCap);
      for (int j = 0; j < kBlockCap; ++j, ++m_m, ++m_d) {
        uint8_t& meta = *m_m;
        if (meta != uint8_t(kProtectedSlot) && meta != uint8_t(kEmptySlot)) {
          meta = uint8_t(kEmptySlot);
          p->Emplace(std::move(m_d->k), std::move(m_d->v));
        }
      }
    }
    delete[] m->data_;
    m->data_ = nullptr;
    m->slots_ = 0;
    m->size_ = 0;
    m->fib_ = 0;
    return p;
  }

  /*!
   * \brief Create an empty container with elements copying from another MapNode
   * \param m The source container
   * \return The object created
   */
  static ObjectPtr<MapNode> CopyFrom(MapNode* m) {
    ObjectPtr<MapNode> p = make_object<MapNode>();
    uint64_t n_blocks = CalcNumBlocks(m->slots_);
    p->data_ = new Block[n_blocks];
    p->slots_ = m->slots_;
    p->size_ = m->size_;
    p->fib_ = m->fib_;
    for (uint64_t bi = 0; bi < n_blocks; ++bi) {
      uint8_t* m_m = m->data_[bi].b;
      uint8_t* p_m = p->data_[bi].b;
      KVType* m_d = reinterpret_cast<KVType*>(m->data_[bi].b + kBlockCap);
      KVType* p_d = reinterpret_cast<KVType*>(p->data_[bi].b + kBlockCap);
      for (int j = 0; j < kBlockCap; ++j, ++m_m, ++m_d, ++p_m, ++p_d) {
        uint8_t& meta = *p_m = *m_m;
        CHECK(meta != kProtectedSlot);
        if (meta != uint8_t(kEmptySlot)) {
          new (p_d) KVType(*m_d);
        }
      }
    }
    return p;
  }

  static uint64_t CalcNumBlocks(uint64_t n_slots_m1) {
    uint64_t n_slots = n_slots_m1 > 0 ? n_slots_m1 + 1 : 0;
    return (n_slots + kBlockCap - 1) / kBlockCap;
  }

  /*! \brief Alternative to std::pair with standard layout */
  struct KVType {
    template <class K, class V>
    KVType(const K& k, const V& v) : k(k), v(v) {}
    template <class K, class V>
    KVType(K&& k, V&& v) : k(std::forward<K>(k)), v(std::forward<V>(v)) {}
    template <class K, class V>
    KVType(const K& k, V&& v) : k(k), v(std::forward<V>(v)) {}
    template <class K, class V>
    KVType(K&& k, const V& v) : k(std::forward<K>(k)), v(v) {}
    /*! \brief The STL type */
    using TStl = std::pair<key_type, mapped_type>;
    /*! \brief Converting from STL type */
    KVType(const TStl& kv) : k(kv.first), v(kv.second) {}  // NOLINT(*)
    /*! \brief Converting to STL type */
    operator TStl() const { return std::make_pair(k, v); }
    /*! \brief The key, or std::pair::first */
    key_type k;
    /*! \brief The value, or std::pair::second */
    mapped_type v;
  };

  /*! \brief POD type of a chunk of memory used to */
  struct Block {
    uint8_t b[kBlockCap + kBlockCap * sizeof(KVType)];
  };

  /*! \brief The implicit in-place linked list used to index a chain */
  struct ListNode {
    /*! \brief Construct None */
    ListNode() : i(0), cur(nullptr) {}
    /*! \brief Construct from position */
    ListNode(uint64_t i, const MapNode* self) : i(i), cur(self->data_ + (i / kBlockCap)) {}
    /*! \brief Construct from hash code */
    static ListNode FromHash(uint64_t h, const MapNode* self) {
      // Fibonacci Hashing. See also:
      // https://programmingpraxis.com/2018/06/19/fibonacci-hash/
      constexpr uint64_t coeff = 11400714819323198485ull;
      uint64_t i = (coeff * h) >> (self->fib_);
      return ListNode(i, self);
    }
    /*! \brief Construct from hash code if the position is head of list */
    static ListNode GetHead(uint64_t h, const MapNode* self) {
      ListNode n = ListNode::FromHash(h, self);
      return n.IsHead() ? n : ListNode();
    }
    /*! \brief Metadata on the entry */
    uint8_t& Meta() const { return *(cur->b + i % kBlockCap); }
    /*! \brief Data on the entry */
    KVType& Data() const {
      return *(reinterpret_cast<KVType*>(cur->b + kBlockCap + (i % kBlockCap) * sizeof(KVType)));
    }
    /*! \brief Key on the entry */
    key_type& Key() const { return Data().k; }
    /*! \brief Value on the entry */
    mapped_type& Val() const { return Data().v; }
    /*! \brief If the entry is head of linked list */
    bool IsHead() const { return (Meta() & 0b10000000) == 0b00000000; }
    /*! \brief If the entry is none */
    bool IsNone() const { return cur == nullptr; }
    /*! \brief If the entry is empty slot */
    bool IsEmpty() const { return Meta() == uint8_t(kEmptySlot); }
    /*! \brief If the entry is protected slot */
    bool IsProtected() const { return Meta() == uint8_t(kProtectedSlot); }
    /*! \brief Set the entry to be empty */
    void SetEmpty() const { Meta() = uint8_t(kEmptySlot); }
    /*! \brief Set the entry to be protected */
    void SetProtected() const { Meta() = uint8_t(kProtectedSlot); }
    /*! \brief Set the entry's jump to its next entry */
    void SetJump(uint8_t jump) const { (Meta() &= 0b10000000) |= jump; }
    /*! \brief Construct a head of linked list in-place */
    void NewHead(KVType&& v) const {
      Meta() = 0b00000000;
      new (&Data()) KVType(std::move(v));
    }
    /*! \brief Construct a tail of linked list in-place */
    void NewTail(KVType&& v) const {
      Meta() = 0b10000000;
      new (&Data()) KVType(std::move(v));
    }
    /*! \brief If the entry has next entry on the linked list */
    bool HasNext() const { return kJumpDists[Meta() & 0b01111111] != 0; }
    /*! \brief Move the entry to the next entry on the linked list */
    bool MoveToNext(const MapNode* self, uint8_t meta) {
      uint64_t d = kJumpDists[meta & 0b01111111];
      if (d == 0) {
        i = 0;
        cur = nullptr;
        return false;
      }
      i = (i + d) & (self->slots_);
      cur = self->data_ + (i / kBlockCap);
      return true;
    }
    /*! \brief Move the entry to the next entry on the linked list */
    bool MoveToNext(const MapNode* self) { return MoveToNext(self, Meta()); }
    /*! \brief Get the previous entry on the linked list */
    ListNode GetPrev(const MapNode* self) const {
      // start from the head of the linked list, which must exist
      ListNode n = FromHash(ObjectHash()(Key()), self);
      // `m` is always the previous item of `n`
      ListNode m = n;
      for (n.MoveToNext(self); i != n.i; m = n, n.MoveToNext(self)) {
      }
      return m;
    }
    /*! \brief Get the next empty jump */
    ListNode GetNextEmpty(const MapNode* self, uint8_t* jump) const {
      for (uint8_t idx = 1; idx < kNumJumpDists; ++idx) {
        ListNode n((i + kJumpDists[idx]) & (self->slots_), self);
        if (n.IsEmpty()) {
          *jump = idx;
          return n;
        }
      }
      *jump = 0;
      return ListNode();
    }
    /*! \brief Index on the real array */
    uint64_t i;
    /*! \brief Pointer to the actual block */
    Block* cur;
  };

  /*!
   * \brief The base implementation of hash map iterator
   * \tparam T The child class in CRTP
   */
  template <class T>
  class IteratorBase {
   public:
    using iterator_category = std::bidirectional_iterator_tag;
    using difference_type = int64_t;
    using value_type = KVType;
    using pointer = KVType*;
    using reference = KVType&;
    /*! \brief Default constructor */
    IteratorBase() : i(0), self(nullptr) {}
    /*! \brief Compare iterators */
    bool operator==(const T& other) const { return i == other.i && self == other.self; }
    /*! \brief Compare iterators */
    bool operator!=(const T& other) const { return !(*this == other); }
    /*! \brief De-reference iterators */
    reference operator*() const { return ListNode(i, self).Data(); }
    /*! \brief Pointer-to of iterators */
    pointer operator->() const { return &ListNode(i, self).Data(); }
    /*! \brief Prefix self increment */
    T& operator++() {
      if (self == nullptr || i > self->slots_) {
        return static_cast<T&>(*this);
      }
      for (++i; i <= self->slots_; ++i) {
        if (!ListNode(i, self).IsEmpty()) {
          return static_cast<T&>(*this);
        }
      }
      return static_cast<T&>(*this);
    }
    /*! \brief Prefix self decrement */
    T& operator--() {
      if (self == nullptr || i > self->slots_ + 1) {
        return static_cast<T&>(*this);
      }
      while (i-- != 0) {
        if (!ListNode(i, self).IsEmpty()) {
          return static_cast<T&>(*this);
        }
      }
      i = self->slots_ + 1;
      return static_cast<T&>(*this);
    }
    /*! \brief Suffix self increment */
    T operator++(int) {
      T copy = static_cast<T&>(*this);
      ++(*this);
      return copy;
    }
    /*! \brief Suffix self decrement */
    T operator--(int) {
      T copy = static_cast<T&>(*this);
      --(*this);
      return copy;
    }
    /*! \brief Constructor */
    IteratorBase(uint64_t _i, const MapNode* _self) : i(_i), self(_self) {
      if (self != nullptr) {
        for (; i <= self->slots_; ++i) {
          if (!ListNode(i, self).IsEmpty()) {
            break;
          }
        }
      }
    }
    /*! \brief The position on the array */
    uint64_t i;
    /*! \brief The container it points to */
    const MapNode* self;
  };

  static_assert(sizeof(KVType) == 16 || sizeof(KVType) == 8, "sizeof(KVType) incorrect");
  static_assert(sizeof(Block) == kBlockCap * (sizeof(KVType) + 1), "sizeof(Block) incorrect");
  static_assert(std::is_standard_layout<KVType>::value, "KVType is not standard layout");
  static_assert(std::is_standard_layout<Block>::value, "Block is not standard layout");

 public:
  /*! \brief The iterator of hash map */
  class iterator : public IteratorBase<iterator> {
   public:
    using iterator_category = std::bidirectional_iterator_tag;
    using difference_type = int64_t;
    using value_type = typename KVType::TStl;
    using pointer = value_type*;
    using reference = value_type&;

    iterator() = default;

   private:
    iterator(uint64_t i, const MapNode* self) : IteratorBase(i, self) {}
    friend class MapNode;
  };

 protected:
  /*! \brief array of data blocks */
  Block* data_;
  /*! \brief number of slots minus 1 */
  uint64_t slots_;
  /*! \brief number of entries in the container */
  uint64_t size_;
  /*! \brief fib shift in Fibonacci Hashing */
  uint8_t fib_;

  // Reference class
  template <typename, typename, typename, typename>
  friend class Map;
};

/*!
 * \brief Map container of NodeRef->NodeRef in DSL graph.
 *  Map implements copy on write semantics, which means map is mutable
 *  but copy will happen when array is referenced in more than two places.
 *
 * operator[] only provide const acces, use Set to mutate the content.
 * \tparam K The key NodeRef type.
 * \tparam V The value NodeRef type.
 */
template <typename K, typename V,
          typename = typename std::enable_if<std::is_base_of<ObjectRef, K>::value>::type,
          typename = typename std::enable_if<std::is_base_of<ObjectRef, V>::value>::type>
class Map : public ObjectRef {
 public:
  class iterator;
  /*!
   * \brief default constructor
   */
  Map() { data_ = MapNode::Empty(); }
  /*!
   * \brief move constructor
   * \param other source
   */
  Map(Map<K, V>&& other) { data_ = std::move(other.data_); }
  /*!
   * \brief copy constructor
   * \param other source
   */
  Map(const Map<K, V>& other) : ObjectRef(other.data_) {}
  /*!
   * \brief copy assign operator
   * \param other The source of assignment
   * \return reference to self.
   */
  Map<K, V>& operator=(Map<K, V>&& other) {
    data_ = std::move(other.data_);
    return *this;
  }
  /*!
   * \brief move assign operator
   * \param other The source of assignment
   * \return reference to self.
   */
  Map<K, V>& operator=(const Map<K, V>& other) {
    data_ = other.data_;
    return *this;
  }
  /*!
   * \brief constructor from pointer
   * \param n the container pointer
   */
  explicit Map(ObjectPtr<Object> n) : ObjectRef(n) {}
  /*!
   * \brief constructor from iterator
   * \param begin begin of iterator
   * \param end end of iterator
   * \tparam IterType The type of iterator
   */
  template <typename IterType>
  Map(IterType begin, IterType end) {
    ObjectPtr<MapNode> n = MapNode::Empty();
    n->Assign(begin, end);
    data_ = std::move(n);
  }
  /*!
   * \brief constructor from initializer list
   * \param init The initalizer list
   */
  Map(std::initializer_list<std::pair<K, V>> init) {
    ObjectPtr<MapNode> n = MapNode::Empty();
    n->Assign(init.begin(), init.end());
    data_ = std::move(n);
  }
  /*!
   * \brief constructor from unordered_map
   * \param init The unordered_map
   */
  template <typename Hash, typename Equal>
  Map(const std::unordered_map<K, V, Hash, Equal>& init) {  // NOLINT(*)
    ObjectPtr<MapNode> n = MapNode::Empty();
    n->Assign(init.begin(), init.end());
    data_ = std::move(n);
  }
  /*!
   * \brief Create a runtime::Map and expose it as ObjectPtr
   * \tparam Args Type of argument list
   * \param args Argument list
   * \return The result ObjectPtr
   */
  template <typename... Args>
  static ObjectPtr<Object> CreateObjectPtr(Args&&... args) {
    Map<K, V> map(std::forward<Args>(args)...);
    ObjectPtr<Object> data = std::move(map.data_);
    return data;
  }
  /*!
   * \brief Read element from map.
   * \param key The key
   * \return the corresonding element.
   */
  const V at(const K& key) const { return DowncastNoCheck<V>(GetMapNode()->at(key)); }
  /*!
   * \brief Read element from map.
   * \param key The key
   * \return the corresonding element.
   */
  const V operator[](const K& key) const { return this->at(key); }
  /*! \return The size of the array */
  size_t size() const {
    MapNode* n = GetMapNode();
    return n == nullptr ? 0 : n->size();
  }
  /*! \return The number of elements of the key */
  size_t count(const K& key) const {
    MapNode* n = GetMapNode();
    return n == nullptr ? 0 : GetMapNode()->count(key);
  }
  /*! \return whether array is empty */
  bool empty() const { return size() == 0; }
  /*!
   * \brief set the Map.
   * \param key The index key.
   * \param value The value to be setted.
   */
  void Set(const K& key, const V& value) { CopyOnWrite()->operator[](key) = value; }
  /*! \return begin iterator */
  iterator begin() const { return iterator(GetMapNode()->begin()); }
  /*! \return end iterator */
  iterator end() const { return iterator(GetMapNode()->end()); }
  /*! \return find the key and returns the associated iterator */
  iterator find(const K& key) const { return iterator(GetMapNode()->find(key)); }
  /*!
   * \brief copy on write semantics
   *  Do nothing if current handle is the unique copy of the array.
   *  Otherwise make a new copy of the array to ensure the current handle
   *  hold a unique copy.
   *
   * \return Handle to the internal node container(which ganrantees to be unique)
   */
  MapNode* CopyOnWrite() {
    if (data_.get() == nullptr || !data_.unique()) {
      data_ = MapNode::CopyFrom(GetMapNode());
    }
    return GetMapNode();
  }
  /*! \brief specify container node */
  using ContainerType = MapNode;

  /*! \brief Iterator of the hash map */
  class iterator : public MapNode::IteratorBase<iterator> {
   public:
    using iterator_category = std::bidirectional_iterator_tag;
    using difference_type = int64_t;
    using value_type = const std::pair<K, V>;
    using pointer = void;
    using reference = value_type;

    iterator() : MapNode::IteratorBase<iterator>() {}

    value_type operator*() const {
      MapNode::ListNode n(this->i, this->self);
      return std::make_pair(DowncastNoCheck<K>(n.Key()), DowncastNoCheck<V>(n.Val()));
    }

   private:
    iterator(const MapNode::iterator& itr)  // NOLINT(*)
        : MapNode::IteratorBase<iterator>(itr.i, itr.self) {}

    template <typename, typename, typename, typename>
    friend class Map;
  };

 private:
  /*! \brief Return data_ as type of pointer of MapNode */
  MapNode* GetMapNode() const { return static_cast<MapNode*>(data_.get()); }
};

}  // namespace runtime

// expose the functions to the root namespace.
using runtime::Optional;
using runtime::String;
constexpr runtime::NullOptType NullOpt{};
}  // namespace tvm

namespace std {

template <>
struct hash<::tvm::runtime::String> {
  std::size_t operator()(const ::tvm::runtime::String& str) const {
    return ::tvm::runtime::String::HashBytes(str.data(), str.size());
  }
};
}  // namespace std

#endif  // TVM_RUNTIME_CONTAINER_H_
