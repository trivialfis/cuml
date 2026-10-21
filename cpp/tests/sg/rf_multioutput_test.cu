/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuml/ensemble/randomforest.hpp>

#include <raft/core/handle.hpp>

#include <rmm/cuda_stream_pool.hpp>

#include <cuda/stream>
#include <thrust/device_vector.h>

#include <gtest/gtest.h>
#include <treelite/gtil.h>
#include <treelite/tree.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace ML {

template <typename T>
void test_multi_output_regression(bool weighted, float min_impurity_decrease)
{
  // Only the second target varies. A split must use both targets, and each child
  // must contain two samples, not four target observations.
  thrust::device_vector<T> X            = std::vector<T>{0, 0, 1, 1};
  thrust::device_vector<T> y            = std::vector<T>{5, 0, 5, 4, 5, 10, 5, 14};
  thrust::device_vector<double> weights = std::vector<double>{1, 3, 1, 3};
  auto stream_pool                      = std::make_shared<rmm::cuda_stream_pool>(1);
  raft::handle_t handle(cuda::stream_ref{cudaStreamPerThread}, stream_pool);
  auto params =
    set_rf_params(2, -1, 1.0, 4, 2, 2, min_impurity_decrease, false, 1, 1.0, 0, MSE, 1, 128);
  RandomForestMetaData<T, T> forest;
  fit(handle,
      &forest,
      X.data().get(),
      4,
      1,
      y.data().get(),
      params,
      rapids_logger::level_enum::info,
      nullptr,
      weighted ? weights.data().get() : nullptr,
      false,
      2);

  ASSERT_EQ(forest.trees.size(), 1);
  const auto& tree = *forest.trees[0];
  EXPECT_EQ(tree.num_outputs, 2);
  const auto& root = tree.sparsetree[0];
  EXPECT_EQ(root.InstanceCount(), 4);
  if (min_impurity_decrease > 12.5f) {
    // The nonconstant target's gain is 25; the mean over both targets is 12.5.
    ASSERT_TRUE(root.IsLeaf());
    EXPECT_EQ(tree.vector_leaf[0], T(5));
    EXPECT_EQ(tree.vector_leaf[1], weighted ? T(8) : T(7));
  } else {
    ASSERT_EQ(tree.sparsetree.size(), 3);
    ASSERT_FALSE(root.IsLeaf());
    EXPECT_EQ(root.QueryValue(), T(0));
    EXPECT_NEAR(root.BestMetric(), 12.5, 1e-6);
    auto left  = root.LeftChildId();
    auto right = root.RightChildId();
    EXPECT_EQ(tree.sparsetree[left].InstanceCount(), 2);
    EXPECT_EQ(tree.sparsetree[right].InstanceCount(), 2);
    EXPECT_TRUE(tree.sparsetree[left].IsLeaf());
    EXPECT_TRUE(tree.sparsetree[right].IsLeaf());
    EXPECT_EQ(tree.vector_leaf[2 * left], T(5));
    EXPECT_EQ(tree.vector_leaf[2 * left + 1], weighted ? T(3) : T(2));
    EXPECT_EQ(tree.vector_leaf[2 * right], T(5));
    EXPECT_EQ(tree.vector_leaf[2 * right + 1], weighted ? T(13) : T(12));
  }

  TreeliteModelHandle model_handle;
  build_treelite_forest(&model_handle, &forest, 1);
  std::unique_ptr<treelite::Model> model(static_cast<treelite::Model*>(model_handle));
  EXPECT_EQ(model->task_type, treelite::TaskType::kRegressor);
  EXPECT_EQ(model->num_target, 2);
  EXPECT_EQ(model->num_class.AsVector(), (std::vector<std::int32_t>{1, 1}));
  EXPECT_EQ(model->leaf_vector_shape.AsVector(), (std::vector<std::int32_t>{2, 1}));
  EXPECT_EQ(model->target_id.AsVector(), (std::vector<std::int32_t>{-1}));
  EXPECT_EQ(model->class_id.AsVector(), (std::vector<std::int32_t>{0}));
  std::visit(
    [&](const auto& preset) {
      const auto& exported = preset.trees[0];
      for (int i = 0; i < exported.num_nodes; ++i) {
        if (exported.IsLeaf(i)) { EXPECT_EQ(exported.LeafVector(i).size(), 2); }
      }
    },
    model->variant_);

  std::vector<T> input{0, 1};
  std::vector<T> predictions(4);
  treelite::gtil::Configuration config;
  config.nthread = 1;
  treelite::gtil::Predict(*model, input.data(), 2, predictions.data(), config);
  EXPECT_EQ(predictions[0], T(5));
  EXPECT_EQ(predictions[2], T(5));
  if (min_impurity_decrease > 12.5f) {
    EXPECT_EQ(predictions[1], weighted ? T(8) : T(7));
    EXPECT_EQ(predictions[3], predictions[1]);
  } else {
    EXPECT_EQ(predictions[1], weighted ? T(3) : T(2));
    EXPECT_EQ(predictions[3], weighted ? T(13) : T(12));
  }
}

TEST(RfMultiOutputTest, SharedSplitsAndVectorLeaves)
{
  for (bool weighted : {false, true}) {
    for (float threshold : {0.0f, 12.6f}) {
      test_multi_output_regression<float>(weighted, threshold);
      test_multi_output_regression<double>(weighted, threshold);
    }
  }
}

TEST(RfMultiOutputTest, TargetHistogramsDoNotMultiplySampleCounts)
{
  thrust::device_vector<float> X = std::vector<float>{0, 1, 2, 3};
  thrust::device_vector<float> y = std::vector<float>{0, 0, 0, 0, 0, 0, 10, 10};
  auto stream_pool               = std::make_shared<rmm::cuda_stream_pool>(1);
  raft::handle_t handle(cuda::stream_ref{cudaStreamPerThread}, stream_pool);
  auto params = set_rf_params(1, -1, 1.0, 4, 2, 2, 0.0, false, 1, 1.0, 0, MSE, 1, 128);
  RandomForestRegressorF forest;
  fit(handle,
      &forest,
      X.data().get(),
      4,
      1,
      y.data().get(),
      params,
      rapids_logger::level_enum::info,
      nullptr,
      nullptr,
      false,
      2);
  // Isolating the last sample has higher gain, but would violate min_samples_leaf.
  const auto& tree = *forest.trees[0];
  ASSERT_EQ(tree.sparsetree.size(), 3);
  const auto& root = tree.sparsetree[0];
  EXPECT_EQ(root.QueryValue(), 1.0f);
  EXPECT_EQ(tree.sparsetree[root.LeftChildId()].InstanceCount(), 2);
  EXPECT_EQ(tree.sparsetree[root.RightChildId()].InstanceCount(), 2);
}

TEST(RfMultiOutputTest, DuplicateTargetsPreserveGainAndUseGlobalHistograms)
{
  auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(1);
  raft::handle_t handle(cuda::stream_ref{cudaStreamPerThread}, stream_pool);
  constexpr int n_rows = 128;
  std::vector<float> host_X(n_rows);
  for (int i = 0; i < n_rows; ++i) {
    host_X[i] = i;
  }
  thrust::device_vector<float> X = host_X;
  auto params = set_rf_params(1, -1, 1.0, n_rows, 64, 2, 0.0, false, 1, 1.0, 0, MSE, 1, 128);
  // 64 outputs exceed shared-memory histogram capacity with 128 bins.
  for (int n_outputs : {1, 2, 64}) {
    std::vector<float> host_y(n_rows * n_outputs);
    for (int i = 0; i < n_rows; ++i) {
      for (int k = 0; k < n_outputs; ++k) {
        host_y[i * n_outputs + k] = (i < 64 ? 0 : 10) + k;
      }
    }
    thrust::device_vector<float> y = host_y;
    RandomForestRegressorF forest;
    fit(handle,
        &forest,
        X.data().get(),
        n_rows,
        1,
        y.data().get(),
        params,
        rapids_logger::level_enum::info,
        nullptr,
        nullptr,
        false,
        n_outputs);
    const auto& tree = *forest.trees[0];
    ASSERT_EQ(tree.sparsetree.size(), 3);
    const auto& root = tree.sparsetree[0];
    EXPECT_EQ(root.QueryValue(), 63.0f);
    EXPECT_NEAR(root.BestMetric(), 25.0f, 1e-6);
    for (int k = 0; k < n_outputs; ++k) {
      EXPECT_EQ(tree.vector_leaf[root.LeftChildId() * n_outputs + k], k);
      EXPECT_EQ(tree.vector_leaf[root.RightChildId() * n_outputs + k], 10 + k);
    }
  }
}

TEST(RfMultiOutputTest, ClassificationStillCountsEachSampleOnce)
{
  thrust::device_vector<float> X = std::vector<float>{0, 0, 1, 1};
  thrust::device_vector<int> y   = std::vector<int>{0, 1, 2, 2};
  auto stream_pool               = std::make_shared<rmm::cuda_stream_pool>(1);
  raft::handle_t handle(cuda::stream_ref{cudaStreamPerThread}, stream_pool);
  auto params = set_rf_params(1, -1, 1.0, 4, 2, 2, 0.0, false, 1, 1.0, 0, GINI, 1, 128);
  RandomForestClassifierF forest;
  fit(handle, &forest, X.data().get(), 4, 1, y.data().get(), 3, params);
  const auto& tree = *forest.trees[0];
  ASSERT_EQ(tree.sparsetree.size(), 3);
  const auto& root = tree.sparsetree[0];
  EXPECT_EQ(root.InstanceCount(), 4);
  EXPECT_EQ(tree.sparsetree[root.LeftChildId()].InstanceCount(), 2);
  EXPECT_EQ(tree.sparsetree[root.RightChildId()].InstanceCount(), 2);
  auto left  = root.LeftChildId() * 3;
  auto right = root.RightChildId() * 3;
  EXPECT_EQ(tree.vector_leaf[left], 0.5f);
  EXPECT_EQ(tree.vector_leaf[left + 1], 0.5f);
  EXPECT_EQ(tree.vector_leaf[left + 2], 0.0f);
  EXPECT_EQ(tree.vector_leaf[right], 0.0f);
  EXPECT_EQ(tree.vector_leaf[right + 1], 0.0f);
  EXPECT_EQ(tree.vector_leaf[right + 2], 1.0f);
}

}  // namespace ML
