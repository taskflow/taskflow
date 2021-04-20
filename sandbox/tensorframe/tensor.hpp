#pragma once

#include "../taskflow.hpp"

#include <variant>
#include <filesystem>

namespace tf {

enum StorageLevel {
  MEMORY = 0,
  MEMORY_AND_DISK = 1
};

/** 

@class Tensor

@brief a tensor contains arithmetic data in N dimensions

*/
template <typename T>
class Tensor {
  
  template <typename U>
  friend class TensorNode;

  template <typename U>
  friend class TensorExpr;
  
  template <typename U>
  friend class TensorFrame;

  struct Chunk {
    std::vector<T> data;
    std::string location;
  };

  public:

    Tensor(const Tensor& tensor) = delete;
    Tensor(Tensor&& tensor) = delete;

    Tensor(std::vector<size_t> shape);
    Tensor(std::vector<size_t> shape, size_t max_chunk_size);

    const std::vector<size_t>& shape() const;
    const std::vector<size_t>& chunk_shape() const;

    size_t size() const;
    size_t rank() const;
    size_t chunk_size() const;
    size_t num_chunks() const;

    StorageLevel storage_level() const;

    void dump(std::ostream& ostream) const;
    
    template <typename... Is>
    size_t flat_chunk_index(Is... indices) const;

    template <typename... Is>
    size_t flat_index(Is... indices) const;

  private:
    
    StorageLevel _storage_level;

    std::vector<size_t> _shape;
    std::vector<size_t> _chunk_shape;
    std::vector<size_t> _chunk_grid;
    std::vector<Chunk> _chunks;

    void _make_chunks(size_t = 65536*1024);  // 65MB per chunk
    
    size_t _flat_chunk_index(size_t&, size_t) const;

    template <typename... Is>
    size_t _flat_chunk_index(size_t&, size_t, Is...) const;
    
    size_t _flat_index(size_t&, size_t) const;

    template <typename... Is>
    size_t _flat_index(size_t&, size_t, Is...) const;
};

template <typename T>
Tensor<T>::Tensor(std::vector<size_t> shape) : 
  _shape       {std::move(shape)},
  _chunk_shape (_shape.size()),
  _chunk_grid  (_shape.size()) {

  _make_chunks();
}

template <typename T>
Tensor<T>::Tensor(std::vector<size_t> shape, size_t max_chunk_size) :
  _shape       {std::move(shape)},
  _chunk_shape (_shape.size()),
  _chunk_grid  (_shape.size()) {

  _make_chunks(std::max(1ul, max_chunk_size));
}

template <typename T>
size_t Tensor<T>::size() const {
  return std::accumulate(
    _shape.begin(), _shape.end(), 1, std::multiplies<size_t>()
  );
}

template <typename T>
size_t Tensor<T>::num_chunks() const {
  return _chunks.size();
}

template <typename T>
size_t Tensor<T>::chunk_size() const {
  return _chunks[0].data.size();
}

template <typename T>
size_t Tensor<T>::rank() const {
  return _shape.size();
}

template <typename T>
const std::vector<size_t>& Tensor<T>::shape() const {
  return _shape;
}

template <typename T>
const std::vector<size_t>& Tensor<T>::chunk_shape() const {
  return _chunk_shape;
}

template <typename T>
template <typename... Is>
size_t Tensor<T>::flat_chunk_index(Is... rest) const {

  if(sizeof...(Is) != rank()) {
    TF_THROW("index rank dose not match tensor rank");
  }

  size_t offset;
  return _flat_chunk_index(offset, rest...);
}

template <typename T>
size_t Tensor<T>::_flat_chunk_index(size_t& offset, size_t id) const {
  offset = 1;
  return id/_chunk_shape.back();
}

template <typename T>
template <typename... Is>
size_t Tensor<T>::_flat_chunk_index(
  size_t& offset, size_t id, Is... rest
) const {
  auto i = _flat_chunk_index(offset, rest...);
  offset *= _chunk_grid[_chunk_shape.size() - (sizeof...(Is))];
  return (id/_chunk_shape[_chunk_shape.size() - sizeof...(Is) - 1])*offset + i;
}

template <typename T>
template <typename... Is>
size_t Tensor<T>::flat_index(Is... rest) const {

  if(sizeof...(Is) != rank()) {
    TF_THROW("index rank dose not match tensor rank");
  }

  size_t offset;
  return _flat_index(offset, rest...);
}

template <typename T>
size_t Tensor<T>::_flat_index(size_t& offset, size_t id) const {
  offset = 1;
  return id;
}

template <typename T>
template <typename... Is>
size_t Tensor<T>::_flat_index(size_t& offset, size_t id, Is... rest) const {
  auto i = _flat_index(offset, rest...);
  offset *= _shape[_shape.size() - (sizeof...(Is))];
  return id*offset + i;
}

template <typename T>
void Tensor<T>::dump(std::ostream& os) const {

  os << "Tensor<" << typeid(T).name() << "> {\n"
     << "  shape=[";

  for(size_t i=0; i<_shape.size(); ++i) {
    if(i) os << 'x';
    os << _shape[i];
  }

  os << "], chunk=[";

  for(size_t i=0; i<_chunk_shape.size(); ++i) {
    if(i) os << 'x';
    os << _chunk_shape[i];
  }

  os << "], pgrid=[";

  for(size_t i=0; i<_chunk_grid.size(); ++i) {
    if(i) os << 'x';
    os << _chunk_grid[i];
  }

  os << "]\n}\n";
}

template <typename T>
void Tensor<T>::_make_chunks(size_t M) {  

  size_t P = 1;
  size_t N = 1;

  for(int i=_shape.size()-1; i>=0; i--) {
    if(M >= _shape[i]) {
      _chunk_shape[i] = _shape[i];
      _chunk_grid[i] = 1;
      N *= _chunk_shape[i];
      M /= _shape[i];
    }
    else {
      _chunk_shape[i] = M;
      _chunk_grid[i] = (_shape[i] + _chunk_shape[i] - 1) / _chunk_shape[i];
      P *= _chunk_grid[i];
      N *= _chunk_shape[i];
      for(i--; i>=0; i--) {
        _chunk_shape[i] = 1;
        _chunk_grid[i] = _shape[i];
        P *= _chunk_grid[i];
      }
      break;
    }
  }

  _chunks.resize(P);

  // we allocate the first data in memory
  _chunks[0].data.resize(N);

  // TODO: the rest sits in the disk
  for(size_t i=1; i<_chunks.size(); ++i) {
  }

  // assign the storage level
  _storage_level = (_chunks.size() <= 1) ? MEMORY : MEMORY_AND_DISK;
}

}  // end of namespace tf -----------------------------------------------------









