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
#include <utility>

namespace tvm {

using runtime::Array;
using runtime::ArrayNode;
using runtime::Downcast;
using runtime::IterAdapter;
using runtime::make_object;
using runtime::Object;
using runtime::ObjectEqual;
using runtime::ObjectHash;
using runtime::ObjectPtr;
using runtime::ObjectPtrEqual;
using runtime::ObjectPtrHash;
using runtime::ObjectRef;
using runtime::String;
using runtime::StringObj;

/*! \brief map node content */
class MapNode : public Object {
 public:
  /*! \brief Type of the keys in the hash map */
  using key_type = ObjectRef;
  /*! \brief Type of the values in the hash map */
  using mapped_type = ObjectRef;
  /*! \brief Type of value stored in the hash map */
  using KVType = std::pair<ObjectRef, ObjectRef>;
  /*! \brief Iterator class */
  class iterator;

  static_assert(std::is_standard_layout<KVType>::value, "KVType is not standard layout");
  static_assert(sizeof(KVType) == 16 || sizeof(KVType) == 8, "sizeof(KVType) incorrect");

  static constexpr const uint32_t _type_index = runtime::TypeIndex::kRuntimeMap;
  static constexpr const char* _type_key = "Map";
  TVM_DECLARE_FINAL_OBJECT_INFO(MapNode, Object);

  /*!
   * \brief Number of elements in the DenseMapNode
   * \return The result
   */
  size_t size() const { return size_; }
  /*!
   * \brief Count the number of times a key exists in the DenseMapNode
   * \param key The indexing key
   * \return The result, 0 or 1
   */
  size_t count(const key_type& key) const;
  /*!
   * \brief Index value associated with a key, throw exception if the key does not exist
   * \param key The indexing key
   * \return The const reference to the value
   */
  const mapped_type& at(const key_type& key) const;
  /*!
   * \brief Index value associated with a key, throw exception if the key does not exist
   * \param key The indexing key
   * \return The mutable reference to the value
   */
  mapped_type& at(const key_type& key);
  /*! \return begin iterator */
  iterator begin() const;
  /*! \return end iterator */
  iterator end() const;
  /*!
   * \brief Index value associated with a key
   * \param key The indexing key
   * \return The iterator of the entry associated with the key, end iterator if not exists
   */
  iterator find(const key_type& key) const;
  /*!
   * \brief Erase the entry associated with the iterator
   * \param position The iterator
   */
  void erase(const iterator& position);
  /*!
   * \brief Erase the entry associated with the key, do nothing if not exists
   * \param key The indexing key
   */
  void erase(const key_type& key) { erase(find(key)); }

  class iterator {
   public:
    using iterator_category = std::bidirectional_iterator_tag;
    using difference_type = int64_t;
    using value_type = KVType;
    using pointer = KVType*;
    using reference = KVType&;
    /*! \brief Default constructor */
    iterator() : i(0), self(nullptr) {}
    /*! \brief Compare iterators */
    bool operator==(const iterator& other) const { return i == other.i && self == other.self; }
    /*! \brief Compare iterators */
    bool operator!=(const iterator& other) const { return !(*this == other); }
    /*! \brief De-reference iterators */
    pointer operator->() const;
    /*! \brief De-reference iterators */
    reference operator*() const { return *((*this).operator->()); }
    /*! \brief Prefix self increment, e.g. ++iter */
    iterator& operator++();
    /*! \brief Prefix self decrement, e.g. --iter */
    iterator& operator--();
    /*! \brief Suffix self increment */
    iterator operator++(int) {
      iterator copy = *this;
      ++(*this);
      return copy;
    }
    /*! \brief Suffix self decrement */
    iterator operator--(int) {
      iterator copy = *this;
      --(*this);
      return copy;
    }

   protected:
    /*! \brief Construct by value */
    iterator(uint64_t i, const MapNode* self) : i(i), self(self) {}
    /*! \brief The position on the array */
    uint64_t i;
    /*! \brief The container it points to */
    const MapNode* self;

    friend class DenseMapNode;
  };

 protected:
  /*!
   * \brief Create an empty container
   * \return The object created
   */
  static ObjectPtr<MapNode> Empty();
  /*!
   * \brief Create the map using contents from the given iterators.
   * \param first Begin of iterator
   * \param last End of iterator
   * \tparam IterType The type of iterator
   * \return ObjectPtr to the map created
   */
  template <typename IterType>
  static ObjectPtr<Object> CreateFromRange(IterType first, IterType last);
  /*!
   * \brief Insert an entry into the given hash map
   * \param kv The entry to be inserted
   * \param map The pointer to the map, can be changed if re-hashing happens
   */
  static void Insert(const KVType& kv, ObjectPtr<Object>* map);
  /*!
   * \brief Create an empty container with elements copying from another DenseMapNode
   * \param m The source container
   * \return The object created
   */
  static ObjectPtr<MapNode> CopyFrom(MapNode* m);
  /*! \brief number of slots minus 1 */
  uint64_t slots_;
  /*! \brief number of entries in the container */
  uint64_t size_;
  // Reference class
  template <typename, typename, typename, typename>
  friend class Map;
};

/*! \brief map node content */
class DenseMapNode : public MapNode {
 private:
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

  struct Block;
  struct ListNode;

 public:
  using MapNode::iterator;

  /*!
   * \brief Destroy the DenseMapNode
   */
  ~DenseMapNode() { this->Reset(); }

  size_t count(const key_type& key) const { return !Search(key).IsNone(); }

  const mapped_type& at(const key_type& key) const { return At(key); }

  mapped_type& at(const key_type& key) { return At(key); }

  iterator begin() const {
    if (slots_ == 0) {
      return iterator(0, this);
    }
    for (uint64_t i = 0; i <= slots_; ++i) {
      if (!ListNode(i, this).IsEmpty()) {
        return iterator(i, this);
      }
    }
    return iterator(slots_ + 1, this);
  }

  iterator end() const { return slots_ == 0 ? iterator(0, this) : iterator(slots_ + 1, this); }

  iterator find(const key_type& key) const {
    ListNode n = Search(key);
    return n.IsNone() ? end() : iterator(n.i, this);
  }

  void erase(const iterator& position) {
    uint64_t i = position.i;
    if (position.self != nullptr && i <= this->slots_) {
      Erase(ListNode(i, this));
    }
  }

 private:
  /*!
   * \brief Search for the given key
   * \param key The key
   * \return ListNode that associated with the key
   */
  ListNode Search(const key_type& key) const {
    if (this->size_ == 0) {
      return ListNode();
    }
    for (ListNode n = GetHead(ObjectHash()(key)); !n.IsNone(); n.MoveToNext(this)) {
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
   * \brief Try to insert a key, or do nothing if already exists
   * \param k The indexing key
   * \param result The linked-list entry found or just constructed
   * \return A boolean, indicating if actual insertion happens
   */
  bool TryInsert(const key_type& k, ListNode* result) {
    if (slots_ == 0) {
      return false;
    }
    // required that `m` to be the head of a linked list through which we can iterator
    ListNode m = FromHash(ObjectHash()(k));
    // `m` can be: 1) empty; 2) body of an irrelevant list; 3) head of the relevant list
    // Case 1: empty
    if (m.IsEmpty()) {
      m.NewHead(KVType(k, ObjectRef(nullptr)));
      this->size_ += 1;
      *result = m;
      return true;
    }
    // Case 2: body of an irrelevant list
    if (!m.IsHead()) {
      // we move the elements around and construct the single-element linked list
      return IsFull() ? false : TrySpareListHead(m, k, result);
    }
    // Case 3: head of the relevant list
    // we iterate through the linked list until the end
    ListNode n = m;
    do {
      // find equal item, do not insert
      if (ObjectEqual()(k, n.Key())) {
        *result = n;
        return true;
      }
      // make sure `m` is the previous element of `n`
      m = n;
    } while (n.MoveToNext(this));
    // `m` is the tail of the linked list
    // always check capacity before insertion
    if (IsFull()) {
      return false;
    }
    // find the next empty slot
    uint8_t jump;
    if (!m.GetNextEmpty(this, &jump, result)) {
      return false;
    }
    result->NewTail(KVType(k, ObjectRef(nullptr)));
    // link `n` to `empty`, and move forward
    m.SetJump(jump);
    this->size_ += 1;
    return true;
  }

  /*!
   * \brief Spare an entry to be the head of a linked list
   * \param n The given entry to be spared
   * \param k The indexing key
   * \param result The linked-list entry constructed as the head
   * \return A boolean, if actual insertion happens
   */
  bool TrySpareListHead(ListNode n, const key_type& k, ListNode* result) {
    // `n` is not the head of the linked list
    // move the original item of `n` (if any)
    // and construct new item on the position `n`
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
      if (!w.GetNextEmpty(this, &jump, &empty)) {
        return false;
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
    n.NewHead(KVType(k, ObjectRef(nullptr)));
    this->size_ += 1;
    *result = n;
    return true;
  }

  /*!
   * \brief Check whether the hash table is full
   * \return A boolean indicating whether hash table is full
   */
  bool IsFull() const { return size_ + 1 > (slots_ + 1) * kMaxLoadFactor; }

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

  /*! \brief Clear the container to empty, release all entries and memory acquired */
  void Reset() {
    uint64_t n_blocks = CalcNumBlocks(this->slots_);
    DenseMapNode* m = this;
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
    delete[] data_;
    data_ = nullptr;
    slots_ = 0;
    size_ = 0;
    fib_shift_ = 63;
  }

  /*!
   * \brief Create an empty container
   * \return The object created
   */
  static ObjectPtr<DenseMapNode> Empty() {
    ObjectPtr<DenseMapNode> p = make_object<DenseMapNode>();
    p->data_ = nullptr;
    p->slots_ = 0;
    p->size_ = 0;
    p->fib_shift_ = 63;
    return p;
  }

  /*!
   * \brief Create an empty container
   * \param fib_shift The fib shift provided
   * \param n_slots Number of slots required, should be power-of-two
   * \return The object created
   */
  static ObjectPtr<DenseMapNode> Empty(uint32_t fib_shift, uint64_t n_slots) {
    CHECK((n_slots & -(n_slots)) == n_slots);
    if (n_slots == 0) {
      return Empty();
    }
    ObjectPtr<DenseMapNode> p = make_object<DenseMapNode>();
    uint64_t n_blocks = CalcNumBlocks(n_slots - 1);
    Block* block = p->data_ = new Block[n_blocks];
    p->slots_ = n_slots - 1;
    p->size_ = 0;
    p->fib_shift_ = fib_shift;
    for (uint64_t i = 0; i < n_blocks; ++i, ++block) {
      std::fill(block->b, block->b + kBlockCap, uint8_t(kEmptySlot));
    }
    return p;
  }

  /*!
   * \brief Create an empty container with elements copying from another DenseMapNode
   * \param m The source container
   * \return The object created
   */
  static ObjectPtr<DenseMapNode> CopyFrom(DenseMapNode* m) {
    if (m == nullptr) {
      return Empty();
    }
    ObjectPtr<DenseMapNode> p = make_object<DenseMapNode>();
    uint64_t n_blocks = CalcNumBlocks(m->slots_);
    p->data_ = new Block[n_blocks];
    p->slots_ = m->slots_;
    p->size_ = m->size_;
    p->fib_shift_ = m->fib_shift_;
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

  /*!
   * \brief Insert an entry into the given hash map
   * \param kv The entry to be inserted
   * \param map The pointer to the map, can be changed if re-hashing happens
   */
  static void Insert(const KVType& kv, ObjectPtr<Object>* map) {
    DenseMapNode* m = static_cast<DenseMapNode*>(map->get());
    ListNode n;
    if (m->TryInsert(kv.first, &n)) {
      n.Val() = kv.second;
      return;
    }
    ObjectPtr<Object> p =
        m->slots_ == 0 ? Empty(59, 32) : Empty(m->fib_shift_ - 1, m->slots_ * 2 + 2);
    Insert(kv, &p);
    uint64_t n_blocks = CalcNumBlocks(m->slots_);
    for (uint64_t bi = 0; bi < n_blocks; ++bi) {
      uint8_t* m_m = m->data_[bi].b;
      KVType* m_d = reinterpret_cast<KVType*>(m->data_[bi].b + kBlockCap);
      for (int j = 0; j < kBlockCap; ++j, ++m_m, ++m_d) {
        uint8_t& meta = *m_m;
        if (meta != uint8_t(kProtectedSlot) && meta != uint8_t(kEmptySlot)) {
          meta = uint8_t(kEmptySlot);
          KVType kv = std::move(*m_d);
          Insert(kv, &p);
        }
      }
    }
    delete[] m->data_;
    CHECK((static_cast<DenseMapNode*>(p.get()))->size_ == m->size_ + 1);
    m->data_ = nullptr;
    m->slots_ = 0;
    m->size_ = 0;
    m->fib_shift_ = 63;
    *map = p;
  }

  /*! \brief Construct from hash code */
  ListNode FromHash(uint64_t hash_value) const {
    return ListNode(FibHash(hash_value, fib_shift_), this);
  }

  /*! \brief Construct from hash code if the position is head of list */
  ListNode GetHead(uint64_t hash_value) const {
    ListNode n = FromHash(hash_value);
    return n.IsHead() ? n : ListNode();
  }

  /*! \brief Construct the number of blocks in the hash table */
  static uint64_t CalcNumBlocks(uint64_t n_slots_m1) {
    uint64_t n_slots = n_slots_m1 > 0 ? n_slots_m1 + 1 : 0;
    return (n_slots + kBlockCap - 1) / kBlockCap;
  }

  /*! \brief Fibonacci Hashing, maps a hash code to an index in a power-of-2-sized table. */
  static uint64_t FibHash(uint64_t hash_value, uint32_t fib_shift) {
    // See also: https://programmingpraxis.com/2018/06/19/fibonacci-hash/
    constexpr uint64_t coeff = 11400714819323198485ull;
    return (coeff * hash_value) >> fib_shift;
  }

  /*! \brief POD type of a chunk of memory used to */
  struct Block {
    uint8_t b[kBlockCap + kBlockCap * sizeof(KVType)];
  };

  static_assert(sizeof(Block) == kBlockCap * (sizeof(KVType) + 1), "sizeof(Block) incorrect");
  static_assert(std::is_standard_layout<Block>::value, "Block is not standard layout");

  /*! \brief The implicit in-place linked list used to index a chain */
  struct ListNode {
    /*! \brief Construct None */
    ListNode() : i(0), cur(nullptr) {}
    /*! \brief Construct from position */
    ListNode(uint64_t i, const DenseMapNode* self) : i(i), cur(self->data_ + (i / kBlockCap)) {}
    /*! \brief Metadata on the entry */
    uint8_t& Meta() const { return *(cur->b + i % kBlockCap); }
    /*! \brief Data on the entry */
    KVType& Data() const {
      return *(reinterpret_cast<KVType*>(cur->b + kBlockCap + (i % kBlockCap) * sizeof(KVType)));
    }
    /*! \brief Key on the entry */
    key_type& Key() const { return Data().first; }
    /*! \brief Value on the entry */
    mapped_type& Val() const { return Data().second; }
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
    void NewHead(KVType v) const {
      Meta() = 0b00000000;
      new (&Data()) KVType(std::move(v));
    }
    /*! \brief Construct a tail of linked list in-place */
    void NewTail(KVType v) const {
      Meta() = 0b10000000;
      new (&Data()) KVType(std::move(v));
    }
    /*! \brief If the entry has next entry on the linked list */
    bool HasNext() const { return kJumpDists[Meta() & 0b01111111] != 0; }
    /*! \brief Move the entry to the next entry on the linked list */
    bool MoveToNext(const DenseMapNode* self, uint8_t meta) {
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
    bool MoveToNext(const DenseMapNode* self) { return MoveToNext(self, Meta()); }
    /*! \brief Get the previous entry on the linked list */
    ListNode GetPrev(const DenseMapNode* self) const {
      // start from the head of the linked list, which must exist
      ListNode n = self->FromHash(ObjectHash()(Key()));
      // `m` is always the previous item of `n`
      ListNode m = n;
      for (n.MoveToNext(self); i != n.i; m = n, n.MoveToNext(self)) {
      }
      return m;
    }
    /*! \brief Get the next empty jump */
    bool GetNextEmpty(const DenseMapNode* self, uint8_t* jump, ListNode* result) const {
      for (uint8_t idx = 1; idx < kNumJumpDists; ++idx) {
        ListNode n((i + kJumpDists[idx]) & (self->slots_), self);
        if (n.IsEmpty()) {
          *jump = idx;
          *result = n;
          return true;
        }
      }
      return false;
    }
    /*! \brief Index on the real array */
    uint64_t i;
    /*! \brief Pointer to the actual block */
    Block* cur;
  };

  uint64_t IncItr(uint64_t i) const {
    for (++i; i <= slots_; ++i) {
      if (!ListNode(i, this).IsEmpty()) {
        return i;
      }
    }
    return slots_ + 1;
  }

  uint64_t DecItr(uint64_t i) const {
    while (i-- != 0) {
      if (!ListNode(i, this).IsEmpty()) {
        return i;
      }
    }
    return slots_ + 1;
  }

  KVType* DeRefItr(uint64_t i) const { return &ListNode(i, this).Data(); }

 protected:
  /*! \brief fib shift in Fibonacci Hashing */
  uint32_t fib_shift_;
  /*! \brief array of data blocks */
  Block* data_;

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
  friend class MapNode;
  friend ObjectPtr<MapNode> runtime::make_object<>();
};

namespace runtime {
template <>
inline ObjectPtr<MapNode> make_object<>() {
  return DenseMapNode::Empty();
}
}  // namespace runtime

inline MapNode::iterator::pointer MapNode::iterator::operator->() const {
  const DenseMapNode* p = static_cast<const DenseMapNode*>(self);
  return p->DeRefItr(i);
}

inline MapNode::iterator& MapNode::iterator::operator++() {
  const DenseMapNode* p = static_cast<const DenseMapNode*>(self);
  i = p->IncItr(i);
  return *this;
}

inline MapNode::iterator& MapNode::iterator::operator--() {
  const DenseMapNode* p = static_cast<const DenseMapNode*>(self);
  i = p->DecItr(i);
  return *this;
}

inline size_t MapNode::count(const key_type& key) const {
  const DenseMapNode* p = static_cast<const DenseMapNode*>(this);
  return p->count(key);
}

inline const MapNode::mapped_type& MapNode::at(const MapNode::key_type& key) const {
  const DenseMapNode* p = static_cast<const DenseMapNode*>(this);
  return p->at(key);
}

inline MapNode::mapped_type& MapNode::at(const MapNode::key_type& key) {
  DenseMapNode* p = static_cast<DenseMapNode*>(this);
  return p->at(key);
}

inline MapNode::iterator MapNode::begin() const {
  const DenseMapNode* p = static_cast<const DenseMapNode*>(this);
  return p->begin();
}

inline MapNode::iterator MapNode::end() const {
  const DenseMapNode* p = static_cast<const DenseMapNode*>(this);
  return p->end();
}

inline MapNode::iterator MapNode::find(const MapNode::key_type& key) const {
  const DenseMapNode* p = static_cast<const DenseMapNode*>(this);
  return p->find(key);
}

inline void MapNode::erase(const MapNode::iterator& position) {
  DenseMapNode* p = static_cast<DenseMapNode*>(this);
  return p->erase(position);
}

inline ObjectPtr<MapNode> MapNode::Empty() { return DenseMapNode::Empty(); }

template <typename IterType>
inline ObjectPtr<Object> MapNode::CreateFromRange(IterType first, IterType last) {
  int64_t cap = std::distance(first, last);
  if (cap <= 0) {
    return DenseMapNode::Empty();
  }
  uint32_t fib_shift = 64;
  uint64_t n_slots = 1;
  for (; cap; fib_shift -= 1, n_slots <<= 1, cap >>= 1) {
  }
  ObjectPtr<Object> n = DenseMapNode::Empty(fib_shift - 1, n_slots << 1);
  for (; first != last; ++first) {
    KVType kv(*first);
    DenseMapNode::Insert(kv, &n);
  }
  return n;
}

inline void MapNode::Insert(const KVType& kv, ObjectPtr<Object>* map) {
  DenseMapNode::Insert(kv, map);
}

inline ObjectPtr<MapNode> MapNode::CopyFrom(MapNode* m) {
  return DenseMapNode::CopyFrom(static_cast<DenseMapNode*>(m));
}

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
    data_ = MapNode::CreateFromRange(begin, end);
  }
  /*!
   * \brief constructor from initializer list
   * \param init The initalizer list
   */
  Map(std::initializer_list<std::pair<K, V>> init) {
    data_ = MapNode::CreateFromRange(init.begin(), init.end());
  }
  /*!
   * \brief constructor from unordered_map
   * \param init The unordered_map
   */
  template <typename Hash, typename Equal>
  Map(const std::unordered_map<K, V, Hash, Equal>& init) {  // NOLINT(*)
    data_ = MapNode::CreateFromRange(init.begin(), init.end());
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
  const V at(const K& key) const { return DowncastNoCheck<V>(GetDenseMapNode()->at(key)); }
  /*!
   * \brief Read element from map.
   * \param key The key
   * \return the corresonding element.
   */
  const V operator[](const K& key) const { return this->at(key); }
  /*! \return The size of the array */
  size_t size() const {
    MapNode* n = GetDenseMapNode();
    return n == nullptr ? 0 : n->size();
  }
  /*! \return The number of elements of the key */
  size_t count(const K& key) const {
    MapNode* n = GetDenseMapNode();
    return n == nullptr ? 0 : GetDenseMapNode()->count(key);
  }
  /*! \return whether array is empty */
  bool empty() const { return size() == 0; }
  /*!
   * \brief set the Map.
   * \param key The index key.
   * \param value The value to be setted.
   */
  void Set(const K& key, const V& value) {
    CopyOnWrite();
    MapNode::Insert(MapNode::KVType(key, value), &data_);
  }
  /*! \return begin iterator */
  iterator begin() const { return iterator(GetDenseMapNode()->begin()); }
  /*! \return end iterator */
  iterator end() const { return iterator(GetDenseMapNode()->end()); }
  /*! \return find the key and returns the associated iterator */
  iterator find(const K& key) const { return iterator(GetDenseMapNode()->find(key)); }
  /*!
   * \brief copy on write semantics
   *  Do nothing if current handle is the unique copy of the array.
   *  Otherwise make a new copy of the array to ensure the current handle
   *  hold a unique copy.
   *
   * \return Handle to the internal node container(which ganrantees to be unique)
   */
  MapNode* CopyOnWrite() {
    if (data_.get() == nullptr) {
      data_ = MapNode::Empty();
    } else if (!data_.unique()) {
      data_ = MapNode::CopyFrom(GetDenseMapNode());
    }
    return GetDenseMapNode();
  }
  /*! \brief specify container node */
  using ContainerType = DenseMapNode;

  /*! \brief Iterator of the hash map */
  class iterator {
   public:
    using iterator_category = std::bidirectional_iterator_tag;
    using difference_type = int64_t;
    using value_type = const std::pair<K, V>;
    using pointer = value_type*;
    using reference = value_type;

    iterator() : itr() {}

    /*! \brief Compare iterators */
    bool operator==(const iterator& other) const { return itr == other.itr; }
    /*! \brief Compare iterators */
    bool operator!=(const iterator& other) const { return itr != other.itr; }
    /*! \brief De-reference iterators is not allowed */
    pointer operator->() const = delete;
    /*! \brief De-reference iterators */
    reference operator*() const {
      MapNode::KVType& kv = *itr;
      return std::make_pair(DowncastNoCheck<K>(kv.first), DowncastNoCheck<V>(kv.second));
    }
    /*! \brief Prefix self increment, e.g. ++iter */
    iterator& operator++() {
      ++itr;
      return *this;
    }
    /*! \brief Prefix self decrement, e.g. --iter */
    iterator& operator--() {
      --itr;
      return *this;
    }
    /*! \brief Suffix self increment */
    iterator operator++(int) {
      iterator copy = *this;
      ++(*this);
      return copy;
    }
    /*! \brief Suffix self decrement */
    iterator operator--(int) {
      iterator copy = *this;
      --(*this);
      return copy;
    }

   private:
    iterator(const MapNode::iterator& itr)  // NOLINT(*)
        : itr(itr) {}

    template <typename, typename, typename, typename>
    friend class Map;

    MapNode::iterator itr;
  };

 private:
  /*! \brief Return data_ as type of pointer of DenseMapNode */
  MapNode* GetDenseMapNode() const { return static_cast<MapNode*>(data_.get()); }
};

}  // namespace tvm

namespace tvm {
namespace runtime {
// Additional overloads for PackedFunc checking.
template <typename T>
struct ObjectTypeChecker<Array<T>> {
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
struct ObjectTypeChecker<Map<K, V>> {
  static bool Check(const Object* ptr) {
    if (ptr == nullptr) return true;
    if (!ptr->IsInstance<MapNode>()) return false;
    const MapNode* n = static_cast<const MapNode*>(ptr);
    for (const auto& kv : *n) {
      if (!ObjectTypeChecker<K>::Check(kv.first.get())) return false;
      if (!ObjectTypeChecker<V>::Check(kv.second.get())) return false;
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
