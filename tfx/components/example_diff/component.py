# Copyright 2022 Google LLC. All Rights Reserved.
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
"""TFX ExampleDiff component definition."""
from typing import List, Optional, Tuple

from absl import logging
from tfx import types
from tfx.components.example_diff import executor
from tfx.dsl.components.base import base_beam_component
from tfx.dsl.components.base import executor_spec
from tfx.proto import example_diff_pb2
from tfx.types import standard_artifacts
from tfx.types import standard_component_specs
from tfx.utils import json_utils


class ExampleDiff(base_beam_component.BaseBeamComponent):
  """TFX ExampleDiff component.

  Computes example level diffs according to an ExampleDiffConfig. See TFDV
  [feature_skew_detector.py](https://github.com/tensorflow/data-validation/blob/master/tensorflow_data_validation/skew/feature_skew_detector.py)
  for more details.

  This executor is under development and may change.
  """

  SPEC_CLASS = standard_component_specs.ExampleDiffSpec
  EXECUTOR_SPEC = executor_spec.BeamExecutorSpec(executor.Executor)

  def __init__(self,
               examples_test: types.BaseChannel,
               examples_base: types.BaseChannel,
               config: example_diff_pb2.ExampleDiffConfig,
               include_split_pairs: Optional[List[Tuple[str, str]]] = None):
    """Construct an ExampleDiff component.

    Args:
      examples_test: A [BaseChannel][tfx.v1.types.BaseChannel] of `ExamplesPath` type, as generated by the
        [ExampleGen component](../../../guide/examplegen).
        This needs to contain any splits referenced in `include_split_pairs`.
      examples_base: A second [BaseChannel][tfx.v1.types.BaseChannel] of `ExamplesPath` type to which
        `examples` should be compared. This needs to contain any splits
        referenced in `include_split_pairs`.
      config: A ExampleDiffConfig that defines configuration for the skew
        detection pipeline.
      include_split_pairs: Pairs of split names that ExampleDiff should be run
        on. Default behavior if not supplied is to run on all pairs. Order is
        (test, base) with respect to examples_test, examples_base.
    """
    if include_split_pairs is None:
      logging.info('Including all split pairs because include_split_pairs is '
                   'not set.')
    diffs = types.Channel(type=standard_artifacts.ExamplesDiff)
    spec = standard_component_specs.ExampleDiffSpec(
        **{
            standard_component_specs.EXAMPLES_KEY:
                examples_test,
            standard_component_specs.BASELINE_EXAMPLES_KEY:
                examples_base,
            standard_component_specs.INCLUDE_SPLIT_PAIRS_KEY:
                json_utils.dumps(include_split_pairs),
            standard_component_specs.EXAMPLE_DIFF_RESULT_KEY:
                diffs,
            standard_component_specs.EXAMPLE_DIFF_CONFIG_KEY: config
        })
    super().__init__(spec=spec)