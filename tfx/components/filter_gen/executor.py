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
"""TFX statistics_gen executor."""
import tensorflow as tf
import os
from typing import Any, Dict, List
import apache_beam as beam
from tfx_bsl.public import tfxio
from absl import logging
from tfx.components.transform import labels
import tensorflow_data_validation as tfdv
from tensorflow_data_validation.statistics import stats_options as options
from tensorflow_data_validation.utils import dashboard_util
from tfx import types
from tfx.components.statistics_gen import stats_artifact_utils
from tfx.components.util import examples_utils
from tfx.components.util import tfxio_utils
from tfx.dsl.components.base import base_beam_executor
from tfx.types import artifact_utils
from tfx.types import standard_component_specs
from tfx.utils import io_utils
from tfx.utils import json_utils
from local.filter.filter import FilterRecordsDoFn

# Default file name for stats generated.
DEFAULT_FILE_NAME = 'data_tfrecord'
DEFAULT_TF_RECORD_FILE_NAME = 'data_tfrecord'

_TELEMETRY_DESCRIPTORS = ['Filter']


class Executor(base_beam_executor.BaseBeamExecutor):
  """Computes statistics over input training data for example validation.

  The StatisticsGen component generates features statistics and random samples
  over training data, which can be used for visualization and validation.
  StatisticsGen uses Beam and appropriate algorithms to scale to large datasets.

  To include StatisticsGen in a TFX pipeline, configure your pipeline similar to
  https://github.com/tensorflow/tfx/blob/master/tfx/examples/chicago_taxi_pipeline/taxi_pipeline_simple.py#L75.
  """

  def Do(
      self,
      input_dict: Dict[str, List[types.Artifact]],
      output_dict: Dict[str, List[types.Artifact]],
      exec_properties: Dict[str, Any],
  ) -> None:
    """Computes stats for each split of input using tensorflow_data_validation.

    Args:
      input_dict: Input dict from input key to a list of Artifacts.
        - examples: A list of type `standard_artifacts.Examples`. This should
          contain both 'train' and 'eval' split.
        - schema: Optionally, a list of type `standard_artifacts.Schema`. When
          the stats_options exec_property also contains a schema, this input
          should not be provided.
      output_dict: Output dict from output key to a list of Artifacts.
        - statistics: A list of type `standard_artifacts.ExampleStatistics`.
          This should contain both the 'train' and 'eval' splits.
      exec_properties: A dict of execution properties.
        - stats_options_json: Optionally, a JSON representation of StatsOptions.
          When a schema is provided as an input, the StatsOptions value should
          not also contain a schema.
        - exclude_splits: JSON-serialized list of names of splits where
          statistics and sample should not be generated.
        - sample_rate_by_split: Optionally, A dict mapping split_name to sample
          rate, which is used to apply a different sample rate to the
          corresponding split. When this is supplied, it will overwrite the
          single sample rate on stats_options_json.

    Raises:
      ValueError when a schema is provided both as an input and as part of the
      StatsOptions exec_property, or if execution properties specify
      write_sharded_output when unsupported.

    Returns:
      None
    """
    self._log_startup(input_dict, output_dict, exec_properties)
    exclude_splits = (
        json_utils.loads(
            exec_properties.get(
                standard_component_specs.EXCLUDE_SPLITS_KEY, 'null'
            )
        )
        or []
    )
    if not isinstance(exclude_splits, list):
      raise ValueError(
          'exclude_splits in execution properties needs to be a '
          'list. Got %s instead.'
          % type(exclude_splits)
      )

    examples = artifact_utils.get_single_instance(
        input_dict[standard_component_specs.EXAMPLES_KEY]
    )

    if examples.has_custom_property(
        examples_utils.CUSTOM_SPLIT_PATTERN_PROPERTY_NAME
    ):
      split_to_pattern = json_utils.loads(
          examples.get_string_custom_property(
              examples_utils.CUSTOM_SPLIT_PATTERN_PROPERTY_NAME
          )
      )
      splits = list(split_to_pattern.keys())
    else:
      splits = artifact_utils.decode_split_names(examples.split_names)

    split_names = [split for split in splits if split not in exclude_splits]


    output_examples_artifact = artifact_utils.get_single_instance(
        output_dict[standard_component_specs.OUTPUT_EXAMPLES_KEY]
    )
    output_examples_artifact.split_names = artifact_utils.encode_split_names(
        split_names
    )
    # set the span property of the statistics artifact equal to
    # the span of the input examples artifact.
    output_examples_artifact.span = examples.span
    stats_artifact = artifact_utils.get_single_instance(input_dict[standard_component_specs.STATISTICS_KEY])
    if input_dict.get(standard_component_specs.SCHEMA_KEY):
      schema = io_utils.SchemaReader().read(
          io_utils.get_only_uri_in_dir(
              artifact_utils.get_single_uri(
                  input_dict[standard_component_specs.SCHEMA_KEY]
              )
          )
      )

    train_stats = stats_artifact_utils.load_statistics(stats_artifact, 'train').proto()
    with self._make_beam_pipeline() as p:
      for split in split_names:
        logging.info('Filtering examples for split %s.', split)
        input_uri = os.path.join(examples.uri, 'Split-' + split)
        file_pattern = os.path.join(input_uri, '*')
        tfrecords = beam.io.ReadFromTFRecord(file_pattern)
        tfrecord_tfxio = tfxio.TFExampleRecord(file_pattern=file_pattern, schema=schema)
        output_uri = artifact_utils.get_split_uri(
            output_dict[standard_component_specs.OUTPUT_EXAMPLES_KEY], split
        )
        data = p | 'tfrecords[%s]' % split >> tfrecords
        filtered_tfrecords = beam.ParDo(FilterRecordsDoFn(schema=schema, train_stats=train_stats))
        output_split_path = os.path.join(output_examples_artifact.uri, 'Split-' + split)
        write_transform = beam.io.WriteToTFRecord(os.path.join(output_split_path, DEFAULT_TF_RECORD_FILE_NAME), compression_type='gzip', file_name_suffix='.gz')
        _ = (data | f"Filter{split}" >> filtered_tfrecords | f"Write{split}" >> write_transform)
        logging.info(
            'Filter for split %s written to %s.', split, output_split_path
        )
