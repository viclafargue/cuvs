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

#include <cuvs/neighbors/brute_force.hpp>

namespace cuvs::neighbors::brute_force {

#define CUVS_INST_BFKNN(T, IdxT)                                                                 \
  auto build(raft::resources const& res,                                                         \
             raft::device_matrix_view<const T, IdxT, raft::row_major> dataset,                   \
             cuvs::distance::DistanceType metric,                                                \
             float metric_arg)                                                                   \
    ->cuvs::neighbors::brute_force::index<T>                                                     \
  {                                                                                              \
    return cuvs::neighbors::brute_force::index<T>(std::move(raft::neighbors::brute_force::build( \
      res, dataset, static_cast<raft::distance::DistanceType>(metric), metric_arg)));            \
  }                                                                                              \
                                                                                                 \
  void search(raft::resources const& res,                                                        \
              const cuvs::neighbors::brute_force::index<T>& idx,                                 \
              raft::device_matrix_view<const T, IdxT, raft::row_major> queries,                  \
              raft::device_matrix_view<IdxT, IdxT, raft::row_major> neighbors,                   \
              raft::device_matrix_view<T, IdxT, raft::row_major> distances)                      \
  {                                                                                              \
    raft::neighbors::brute_force::search(                                                        \
      res, *idx.get_raft_index(), queries, neighbors, distances);                                \
  }

CUVS_INST_BFKNN(float, int64_t);

#undef CUVS_INST_BFKNN

}  // namespace cuvs::neighbors::brute_force