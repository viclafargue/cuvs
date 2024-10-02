/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * NOTE: this file is generated by generate_iface.py
 *
 * Make changes there and run in this directory:
 *
 * > python generate_iface.py
 *
 */

#include "iface.hpp"

namespace cuvs::neighbors {

#define CUVS_INST_MG_CAGRA(T, IdxT)                                                                \
  using T_ha    = raft::host_device_accessor<std::experimental::default_accessor<const T>,         \
                                          raft::memory_type::device>;                           \
  using T_da    = raft::host_device_accessor<std::experimental::default_accessor<const T>,         \
                                          raft::memory_type::host>;                             \
  using IdxT_ha = raft::host_device_accessor<std::experimental::default_accessor<const IdxT>,      \
                                             raft::memory_type::device>;                           \
  using IdxT_da = raft::host_device_accessor<std::experimental::default_accessor<const IdxT>,      \
                                             raft::memory_type::host>;                             \
                                                                                                   \
  template void build(                                                                             \
    const raft::device_resources& handle,                                                          \
    cuvs::neighbors::iface<cagra::index<T, IdxT>, T, IdxT>& interface,                             \
    const cuvs::neighbors::index_params* index_params,                                             \
    raft::mdspan<const T, matrix_extent<int64_t>, row_major, T_ha> index_dataset);                 \
                                                                                                   \
  template void build(                                                                             \
    const raft::device_resources& handle,                                                          \
    cuvs::neighbors::iface<cagra::index<T, IdxT>, T, IdxT>& interface,                             \
    const cuvs::neighbors::index_params* index_params,                                             \
    raft::mdspan<const T, matrix_extent<int64_t>, row_major, T_da> index_dataset);                 \
                                                                                                   \
  template void extend(                                                                            \
    const raft::device_resources& handle,                                                          \
    cuvs::neighbors::iface<cagra::index<T, IdxT>, T, IdxT>& interface,                             \
    raft::mdspan<const T, matrix_extent<int64_t>, row_major, T_ha> new_vectors,                    \
    std::optional<raft::mdspan<const IdxT, vector_extent<int64_t>, layout_c_contiguous, IdxT_ha>>  \
      new_indices);                                                                                \
                                                                                                   \
  template void extend(                                                                            \
    const raft::device_resources& handle,                                                          \
    cuvs::neighbors::iface<cagra::index<T, IdxT>, T, IdxT>& interface,                             \
    raft::mdspan<const T, matrix_extent<int64_t>, row_major, T_da> new_vectors,                    \
    std::optional<raft::mdspan<const IdxT, vector_extent<int64_t>, layout_c_contiguous, IdxT_da>>  \
      new_indices);                                                                                \
                                                                                                   \
  template void search(const raft::device_resources& handle,                                       \
                       const cuvs::neighbors::iface<cagra::index<T, IdxT>, T, IdxT>& interface,    \
                       const cuvs::neighbors::search_params* search_params,                        \
                       raft::device_matrix_view<const T, int64_t, row_major> h_queries,            \
                       raft::device_matrix_view<IdxT, int64_t, row_major> d_neighbors,             \
                       raft::device_matrix_view<float, int64_t, row_major> d_distances);           \
                                                                                                   \
  template void search(const raft::device_resources& handle,                                       \
                       const cuvs::neighbors::iface<cagra::index<T, IdxT>, T, IdxT>& interface,    \
                       const cuvs::neighbors::search_params* search_params,                        \
                       raft::host_matrix_view<const T, int64_t, row_major> h_queries,              \
                       raft::device_matrix_view<IdxT, int64_t, row_major> d_neighbors,             \
                       raft::device_matrix_view<float, int64_t, row_major> d_distances);           \
                                                                                                   \
  template void serialize(const raft::device_resources& handle,                                    \
                          const cuvs::neighbors::iface<cagra::index<T, IdxT>, T, IdxT>& interface, \
                          std::ostream& os);                                                       \
                                                                                                   \
  template void deserialize(const raft::device_resources& handle,                                  \
                            cuvs::neighbors::iface<cagra::index<T, IdxT>, T, IdxT>& interface,     \
                            std::istream& is);                                                     \
                                                                                                   \
  template void deserialize(const raft::device_resources& handle,                                  \
                            cuvs::neighbors::iface<cagra::index<T, IdxT>, T, IdxT>& interface,     \
                            const std::string& filename);
CUVS_INST_MG_CAGRA(half, uint32_t);

#undef CUVS_INST_MG_CAGRA

}  // namespace cuvs::neighbors
