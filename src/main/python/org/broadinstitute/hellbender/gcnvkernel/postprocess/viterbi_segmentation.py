from typing import List, Tuple
from ..structs.metadata import SampleMetadataCollection
import logging
import os
import numpy as np
import theano as th
from .. import types
from ..io import io_consts, io_commons, io_denoising_calling, io_intervals_and_counts
from ..structs.interval import Interval
from ..structs.metadata import IntervalListMetadata
from ..models.model_denoising_calling import DenoisingModelConfig, CopyNumberCallingConfig,\
    DenoisingCallingWorkspace

_logger = logging.getLogger(__name__)


class ViterbiSegmentationEngine:
    """

    Note:
        It is assumed that the model and calls shards are provided in the ascending ordering according
        to the SAM sequence dictionary. It is not checked or enforced here.
    """
    def __init__(self,
                 scattered_model_paths: List[str],
                 scattered_calls_paths: List[str],
                 sample_metadata_collection: SampleMetadataCollection):
        self._validate_args(scattered_model_paths, scattered_calls_paths, sample_metadata_collection)
        self.scattered_calls_paths = scattered_calls_paths
        self.sample_metadata_collection = sample_metadata_collection
        self.denoising_config = self._get_denoising_config(scattered_model_paths[0])
        self.calling_config = self._get_calling_config(scattered_model_paths[0])

        # assemble scattered global entities (interval list, log_q_tau_tk)
        self.interval_list: List[Interval] = []
        log_q_tau_tk_shards: Tuple[np.ndarray] = ()
        for model_path in scattered_model_paths:
            self.interval_list += self._get_interval_list_from_model_shard(model_path)
            log_q_tau_tk_shards += (self._get_log_q_tau_tk_from_model_shard(model_path),)
        self.log_q_tau_tk = np.concatenate(log_q_tau_tk_shards, axis=0)

        # sample names
        self.sample_names = self._get_sample_names_from_calls_shard(scattered_calls_paths[0])
        self.num_samples = len(self.sample_names)

        # interval list metadata
        self.interval_list_metadata: IntervalListMetadata = IntervalListMetadata(self.interval_list)

    def _generate_workspace_for_viterbi(self,
                                        sample_index: int,
                                        contig: str,
                                        full_copy_number_log_emission_tc: np.ndarray) -> DenoisingCallingWorkspace:
        """todo.

        Note:
            The read count array will be set to 0 in the workspace. It is not need in the Viterbi
            segmentation process, however, the initializer requires it.

        Args:
            sample_index:
            contig:
            full_copy_number_log_emission_tc:

        Returns:
            an instance of `DenoisingCallingWorkspace` containing intervals for the specified contig,
            the associated class posterior probabilities, and log emission probabilities
        """
        assert 0 <= sample_index < self.num_samples, "Sample index is out of range."

        contig_interval_indices = self.interval_list_metadata.contig_interval_indices[contig]

        single_sample_workspace = DenoisingCallingWorkspace(
            self.denoising_config,
            self.calling_config,
            [self.interval_list[ti] for ti in contig_interval_indices],
            np.zeros((1, len(contig_interval_indices)), dtype=types.med_uint),
            [self.sample_names[sample_index]],
            self.sample_metadata_collection,
            posterior_initializer=None)

        # set copy number class log posteriors
        single_sample_workspace.log_q_tau_tk = th.shared(
            self.log_q_tau_tk[contig_interval_indices, :], name='log_q_tau_tk')

        # set emission probabilities
        single_sample_workspace.log_copy_number_emission_stc.set_value(
            full_copy_number_log_emission_tc[np.newaxis, contig_interval_indices, :])

        return single_sample_workspace

    def perform_viterbi_segmentation_for_single_sample(self,
                                                       sample_index: int):
        pass

    @staticmethod
    def _validate_args(scattered_model_paths: List[str],
                       scattered_calls_paths: List[str],
                       sample_metadata_collection: SampleMetadataCollection):
        assert len(scattered_model_paths) > 0, "At least one model shard must be provided."
        assert len(scattered_calls_paths) == len(scattered_model_paths),\
            "The number of model shards ({0}) and calls shards ({1}) must match.".format(
                len(scattered_model_paths), len(scattered_calls_paths))

        scattered_sample_names: List[Tuple[str]] = []
        for model_path, calls_path in zip(scattered_model_paths, scattered_calls_paths):
            # assert interval lists are identical
            model_interval_list_file = os.path.join(model_path, io_consts.default_interval_list_filename)
            calls_interval_list_file = os.path.join(calls_path, io_consts.default_interval_list_filename)
            io_commons.assert_files_are_identical(model_interval_list_file, calls_interval_list_file)

            # assert gcnvkernel versions are identical
            model_gcnvkernel_version_file = os.path.join(model_path, io_consts.default_gcnvkernel_version_json_filename)
            calls_gcnvkernel_version_file = os.path.join(calls_path, io_consts.default_gcnvkernel_version_json_filename)
            io_commons.assert_files_are_identical(model_gcnvkernel_version_file, calls_gcnvkernel_version_file)

            # assert denoising configs are identical
            model_denoising_config_file = os.path.join(model_path, io_consts.default_denoising_config_json_filename)
            calls_denoising_config_file = os.path.join(calls_path, io_consts.default_denoising_config_json_filename)
            io_commons.assert_files_are_identical(model_denoising_config_file, calls_denoising_config_file)

            # assert callings configs are identical
            model_calling_config_file = os.path.join(model_path, io_consts.default_calling_config_json_filename)
            calls_calling_config_file = os.path.join(calls_path, io_consts.default_calling_config_json_filename)
            io_commons.assert_files_are_identical(model_calling_config_file, calls_calling_config_file)

            # extract and store sample names for the current shard
            scattered_sample_names.append(ViterbiSegmentationEngine._get_sample_names_from_calls_shard(calls_path))

        # all scattered calls have the same set of samples and in the same order
        assert len(set(scattered_sample_names)) == 1,\
            "The scattered calls contain different sample names and/or different number of samples."

        # all samples have ploidy calls in the metadata collection
        sample_names = list(scattered_sample_names[0])
        sample_metadata_collection.all_samples_have_ploidy_metadata(sample_names)

    @staticmethod
    def _get_sample_names_from_calls_shard(calls_path: str) -> Tuple[str]:
        sample_names: Tuple[str] = ()
        sample_index = 0
        while True:
            sample_posteriors_path = io_denoising_calling.get_sample_posterior_path(calls_path, sample_index)
            if not os.path.isdir(sample_posteriors_path):
                break
            sample_names += (io_commons.get_sample_name_from_txt_file(sample_posteriors_path),)
            sample_index += 1
        if len(sample_names) == 0:
            raise Exception("Could not file any sample posterior calls in {0}.".format(calls_path))
        else:
            return tuple(sample_names)

    @staticmethod
    def _get_denoising_config(input_path: str) -> DenoisingModelConfig:
        return DenoisingModelConfig.from_json_file(os.path.join(
            input_path, io_consts.default_denoising_config_json_filename))

    @staticmethod
    def _get_calling_config(input_path: str) -> CopyNumberCallingConfig:
        return CopyNumberCallingConfig.from_json_file(os.path.join(
            input_path, io_consts.default_calling_config_json_filename))

    @staticmethod
    def _get_interval_list_from_model_shard(model_path: str) -> List[Interval]:
        interval_list_file = os.path.join(model_path, io_consts.default_interval_list_filename)
        return io_intervals_and_counts.load_interval_list_tsv_file(interval_list_file)

    @staticmethod
    def _get_log_q_tau_tk_from_model_shard(model_path: str) -> np.ndarray:
        return io_commons.read_ndarray_from_tsv(os.path.join(
            model_path, io_consts.default_class_log_posterior_tsv_filename))

    @staticmethod
    def _get_log_copy_number_emission_tc_from_calls_shard(calls_path: str, sample_index: int):
        return io_denoising_calling.SampleDenoisingAndCallingPosteriorsImporter.\
            import_ndarray_tc_with_copy_number_header(
                io_denoising_calling.get_sample_posterior_path(calls_path, sample_index),
                io_consts.default_copy_number_log_emission_tsv_filename)
