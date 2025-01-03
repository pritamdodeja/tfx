# Copyright 2019 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TFX Transform component definition."""

from typing import List, Optional

from tfx import types
from tfx.components.filter_gen import executor
from tfx.dsl.components.base import base_beam_component
from tfx.dsl.components.base import executor_spec
from tfx.types import standard_component_specs
from tfx.types import standard_artifacts


class Filter(base_beam_component.BaseBeamComponent):
  """A TFX component to filter the input examples."""
  

  SPEC_CLASS = standard_component_specs.FilterSpec
  EXECUTOR_SPEC = executor_spec.BeamExecutorSpec(executor.Executor)

  def __init__(
      self,
      examples: types.BaseChannel,
      statistics: types.BaseChannel,
      schema: types.BaseChannel,
      exclude_splits: Optional[List[str]] = None):
    """Construct a Transform component.

    Args:
      examples: A BaseChannel of type `standard_artifacts.Examples` (required).
        This should contain custom splits specified in splits_config. If custom
        split is not provided, this should contain two splits 'train' and
        'eval'.
      schema: A BaseChannel of type `standard_artifacts.Schema`. This should
        contain a single schema artifact.
      module_file: The file path to a python module file, from which the
        'preprocessing_fn' function will be loaded.
        Exactly one of 'module_file' or 'preprocessing_fn' must be supplied.

        The function needs to have the following signature:
        ```
        def preprocessing_fn(inputs: Dict[Text, Any]) -> Dict[Text, Any]:
          ...
        ```
        where the values of input and returned Dict are either tf.Tensor or
        tf.SparseTensor.

        If additional inputs are needed for preprocessing_fn, they can be passed
        in custom_config:

        ```
        def preprocessing_fn(inputs: Dict[Text, Any], custom_config:
                             Dict[Text, Any]) -> Dict[Text, Any]:
          ...
        ```
        To update the stats options used to compute the pre-transform or
        post-transform statistics, optionally define the
        'stats-options_updater_fn' within the same module. If implemented,
        this function needs to have the following signature:
        ```
        def stats_options_updater_fn(stats_type: tfx.components.transform
          .stats_options_util.StatsType, stats_options: tfdv.StatsOptions)
          -> tfdv.StatsOptions:
          ...
        ```
        Use of a RuntimeParameter for this argument is experimental.
      preprocessing_fn: The path to python function that implements a
        'preprocessing_fn'. See 'module_file' for expected signature of the
        function. Exactly one of 'module_file' or 'preprocessing_fn' must be
        supplied. Use of a RuntimeParameter for this argument is experimental.
      splits_config: A transform_pb2.SplitsConfig instance, providing splits
        that should be analyzed and splits that should be transformed. Note
        analyze and transform splits can have overlap. Default behavior (when
        splits_config is not set) is analyze the 'train' split and transform all
        splits. If splits_config is set, analyze cannot be empty.
      analyzer_cache: Optional input 'TransformCache' channel containing cached
        information from previous Transform runs. When provided, Transform will
        try use the cached calculation if possible.
      materialize: If True, write transformed examples as an output.
      disable_analyzer_cache: If False, Transform will use input cache if
        provided and write cache output. If True, `analyzer_cache` must not be
        provided.
      force_tf_compat_v1: (Optional) If True and/or TF2 behaviors are disabled
        Transform will use Tensorflow in compat.v1 mode irrespective of
        installed version of Tensorflow. Defaults to `False`.
      custom_config: A dict which contains additional parameters that will be
        passed to preprocessing_fn.
      disable_statistics: If True, do not invoke TFDV to compute pre-transform
        and post-transform statistics. When statistics are computed, they will
        will be stored in the `pre_transform_feature_stats/` and
        `post_transform_feature_stats/` subfolders of the `transform_graph`
        export.
      stats_options_updater_fn: The path to a python function that implements a
        'stats_options_updater_fn'. See 'module_file' for expected signature of
        the function. 'stats_options_updater_fn' cannot be defined if
        'module_file' is specified.

    Raises:
      ValueError: When both or neither of 'module_file' and 'preprocessing_fn'
        is supplied.
    """

    example_artifacts = types.Channel(type=standard_artifacts.Examples)
    spec = standard_component_specs.FilterSpec(
        examples=examples,
        schema=schema,
        statistics=statistics,
        exclude_splits=exclude_splits,
        output_examples=example_artifacts,
    )
    super().__init__(spec=spec)
