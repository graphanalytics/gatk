from typing import List, Tuple, Dict, Set, Optional, TypeVar
from ..structs.metadata import SampleMetadataCollection
import itertools
import logging
import os
import numpy as np
import theano as th
import theano.tensor as tt
import pymc3 as pm
from .. import types, config
from ..io import io_consts, io_commons, io_denoising_calling, io_intervals_and_counts
from ..structs.interval import Interval
from ..structs.metadata import IntervalListMetadata
from ..models.model_denoising_calling import DenoisingModelConfig, CopyNumberCallingConfig,\
    HHMMClassAndCopyNumberBasicCaller
from ..models.theano_hmm import TheanoForwardBackward, TheanoViterbi
from scipy.misc import logsumexp

_logger = logging.getLogger(__name__)


class HMMSegmentationQualityCalculator:
    """Calculates quality metrics for hidden state segments for a given HMM.

    Note:
        The initializer requires the emission and transition probabilities, as well as the forward
        and backward tables and the log posterior probability.
    """

    # 10 / ln(10)
    INV_LN_10_TIMES_10 = 4.342944819032518

    # ln(1/2)
    LN_HALF = -0.6931471805599453

    def __init__(self,
                 log_emission_tc: np.ndarray,
                 log_trans_tcc: np.ndarray,
                 alpha_tc: np.ndarray,
                 beta_tc: np.ndarray,
                 log_posterior_prob_tc: np.ndarray,
                 log_data_likelihood: float):
        assert isinstance(log_emission_tc, np.ndarray)
        assert log_emission_tc.ndim == 2
        self.num_sites, self.num_states = log_emission_tc.shape

        assert isinstance(log_trans_tcc, np.ndarray)
        assert log_trans_tcc.shape == (self.num_sites - 1, self.num_states, self.num_states)

        assert isinstance(alpha_tc, np.ndarray)
        assert alpha_tc.shape == (self.num_sites, self.num_states)

        assert isinstance(beta_tc, np.ndarray)
        assert beta_tc.shape == (self.num_sites, self.num_states)

        assert isinstance(log_posterior_prob_tc, np.ndarray)
        assert log_posterior_prob_tc.shape == (self.num_sites, self.num_states)

        self.log_emission_tc = log_emission_tc
        self.log_trans_tcc = log_trans_tcc
        self.alpha_tc = alpha_tc
        self.beta_tc = beta_tc
        self.log_posterior_prob_tc = log_posterior_prob_tc
        self.log_data_likelihood = log_data_likelihood

        self.all_states_set = set(range(self.num_states))

    @staticmethod
    @th.configparser.change_flags(compute_test_value="ignore")
    def _get_compiled_constrained_path_log_likelihood_theano_func():
        alpha_first_c = tt.vector('alpha_first_c')
        beta_last_c = tt.vector('beta_last_c')
        log_emission_tc = tt.matrix('log_emission_tc')
        log_trans_tcc = tt.tensor3('log_trans_tcc')

        def update_alpha(c_log_emission_c: tt.vector,
                         c_log_trans_cc: tt.matrix,
                         p_alpha_c: tt.vector):
            return c_log_emission_c + pm.math.logsumexp(p_alpha_c.dimshuffle(0, 'x') +
                                                        c_log_trans_cc, axis=0).dimshuffle(1)

        alpha_seg_iters, _ = th.scan(
            fn=update_alpha,
            sequences=[log_emission_tc, log_trans_tcc],
            outputs_info=[alpha_first_c])
        alpha_seg_end_c = alpha_seg_iters[-1, :]

        inputs = [alpha_first_c, beta_last_c, log_emission_tc, log_trans_tcc]
        output = pm.math.logsumexp(alpha_seg_end_c + beta_last_c)
        return th.function(inputs=inputs, outputs=output)

    # make a private static instance
    _constrained_path_log_likelihood_theano_func = _get_compiled_constrained_path_log_likelihood_theano_func.__func__()

    def get_log_constrained_posterior_prob(self,
                                           start_index: int, end_index: int,
                                           allowed_states: Set[int]) -> float:
        """Calculates the constrained log posterior probability for contiguous set of sites in a Markov chain.
        At each site, only a subset of all states (as set by `allowed_states`) are allowed and the other states
        are strictly avoided.

        Args:
            start_index: first site index (inclusive)
            end_index: last site index (inclusive)
            allowed_states: the list of allowed states in the segment

        Returns:
            log constrained posterior probability (float)
        """
        assert start_index >= 0
        assert end_index < self.num_sites
        assert end_index >= start_index
        assert all(isinstance(item, int) and 0 <= item < self.num_states for item in allowed_states), \
            "The set of allowed states must be integers and in range [0, {0}]".format(self.num_states - 1)
        allowed_states_list = sorted(allowed_states)

        constrained_alpha_first_c = self.alpha_tc[start_index, allowed_states_list]
        constrained_beta_last_c = self.beta_tc[end_index, allowed_states_list]
        if end_index == start_index:  # single-site segment
            log_constrained_data_likelihood: float = logsumexp(constrained_alpha_first_c + constrained_beta_last_c)
            return log_constrained_data_likelihood - self.log_data_likelihood
        else:
            # calculate the required slices of the log emission and log transition representing
            # paths that only contain the allowed states
            constrained_log_emission_tc = \
                self.log_emission_tc[(start_index + 1):(end_index + 1), allowed_states_list]
            constrained_log_trans_tcc = \
                self.log_trans_tcc[start_index:end_index, allowed_states_list, :][:, :, allowed_states_list]
            return np.asscalar(self._constrained_path_log_likelihood_theano_func(
                constrained_alpha_first_c, constrained_beta_last_c,
                constrained_log_emission_tc, constrained_log_trans_tcc) - self.log_data_likelihood)

    def get_segment_some_quality(self, start_index: int, end_index: int, call_state: int) -> float:
        """Calculates the phred-scaled posterior probability that one or more ("some") sites in a segment have
        the same hidden state ("call").

        Args:
            start_index: first site index (inclusive)
            end_index: last site index (inclusive)
            call_state: segment call state index

        Returns:
            a phred-scaled probability
        """
        assert call_state in self.all_states_set
        other_states = self.all_states_set.copy()
        other_states.remove(call_state)
        all_other_states_log_prob = self.get_log_constrained_posterior_prob(start_index, end_index, other_states)
        return self.log_prob_to_phred(all_other_states_log_prob, complement=False)

    def get_segment_exact_quality(self, start_index: int, end_index: int, call_state: int) -> float:
        """Calculates the phred-scaled posterior probability that "all" sites in a segment have the same
        hidden state ("call").

        Args:
            start_index: first site index (inclusive)
            end_index: last site index (inclusive)
            call_state: segment call state index

        Returns:
            a phred-scaled probability
        """
        assert call_state in self.all_states_set
        all_called_state_log_prob = self.get_log_constrained_posterior_prob(start_index, end_index, {call_state})
        return self.log_prob_to_phred(all_called_state_log_prob, complement=True)

    def get_segment_start_quality(self, start_index: int, call_state: int) -> float:
        """Calculates the phred-scaled posterior probability that a site marks the start of a segment.

        Args:
            start_index: left breakpoint index of a segment
            call_state: segment call state index

        Returns:
            a phred-scaled probability
        """
        assert 0 <= start_index < self.num_sites
        if start_index == 0:
            log_prob = self.log_posterior_prob_tc[0, call_state]
        else:
            # calculate the probability of all paths that start from other states and end up with the called state
            all_other_states = self.all_states_set.copy()
            all_other_states.remove(call_state)
            all_other_states_list = list(all_other_states)
            prev_alpha_c = self.alpha_tc[start_index - 1, all_other_states_list]
            current_beta = self.beta_tc[start_index, call_state]
            current_log_emission = self.log_emission_tc[start_index, call_state]
            log_trans_c = self.log_trans_tcc[start_index - 1, all_other_states_list, call_state]
            log_breakpoint_likelihood = logsumexp(prev_alpha_c + log_trans_c + current_log_emission) + current_beta
            log_prob = log_breakpoint_likelihood - self.log_data_likelihood

        return self.log_prob_to_phred(log_prob, complement=True)

    def get_segment_end_quality(self, end_index: int, call_state: int) -> float:
        """Calculates the phred-scaled posterior probability that a site marks the end of a segment.

        Args:
            end_index: right breakpoint index of a segment
            call_state: segment call state index

        Returns:

        """
        assert 0 <= end_index < self.num_sites
        if end_index == self.num_sites - 1:
            log_prob = self.log_posterior_prob_tc[self.num_sites - 1, call_state]
        else:
            # calculate the probability of all paths that start from call state and end up with other states
            all_other_states = self.all_states_set.copy()
            all_other_states.remove(call_state)
            all_other_states_list = list(all_other_states)
            current_alpha = self.alpha_tc[end_index, call_state]
            next_beta_c = self.beta_tc[end_index + 1, all_other_states_list]
            next_log_emission_c = self.log_emission_tc[end_index + 1, all_other_states_list]
            log_trans_c = self.log_trans_tcc[end_index, call_state, all_other_states_list]
            log_breakpoint_likelihood = logsumexp(current_alpha + log_trans_c + next_log_emission_c + next_beta_c)
            log_prob = log_breakpoint_likelihood - self.log_data_likelihood

        return self.log_prob_to_phred(log_prob, complement=True)

    @staticmethod
    def log_prob_to_phred(log_prob: float, complement: bool = False) -> float:
        """Converts probabilities in natural log scale to phred scale.

        Args:
            log_prob: a probability in the natural log scale
            complement: invert the probability

        Returns:
            phred-scaled probability
        """
        final_log_prob = log_prob if not complement else HMMSegmentationQualityCalculator.log_prob_complement(log_prob)
        return -final_log_prob * HMMSegmentationQualityCalculator.INV_LN_10_TIMES_10

    @staticmethod
    def log_prob_complement(log_prob: float) -> float:
        """Calculates the complement of a probability in the natural log scale.

        Args:
            log_prob: a probability in the natural log scale

        Returns:
            complement of the the probability in the natural log scale
        """
        log_prob_zero_capped = min(0., log_prob)
        if log_prob_zero_capped >= HMMSegmentationQualityCalculator.LN_HALF:
            return np.log(-np.expm1(log_prob_zero_capped))
        else:
            return np.log1p(-np.exp(log_prob_zero_capped))


class ConstantCopyNumberSegment:
    def __init__(self, contig: str, start: int, end: int, num_spanning_intervals: int, copy_number_call: int):
        self.contig = contig
        self.start = start
        self.end = end
        self.copy_number_call = copy_number_call
        self.num_spanning_intervals = num_spanning_intervals
        self.some_quality: Optional[float] = None
        self.exact_quality: Optional[float] = None
        self.start_quality: Optional[float] = None
        self.end_quality: Optional[float] = None

    @staticmethod
    def get_header_column_string():
        return '\t'.join([io_consts.contig_column_name,
                          io_consts.start_column_name,
                          io_consts.end_column_name,
                          io_consts.num_spanning_intervals_column_name,
                          io_consts.copy_number_call_column_name,
                          io_consts.some_quality_column_name,
                          io_consts.exact_quality_column_name,
                          io_consts.start_quality_column_name,
                          io_consts.end_quality_column_name])

    @staticmethod
    def _repr_quality(quality):
        return '{0:.{1}f}'.format(quality, config.phred_decimals) if quality is not None else 'n/a'

    def __repr__(self):
        return '\t'.join([self.contig,
                          repr(self.start),
                          repr(self.end),
                          repr(self.num_spanning_intervals),
                          repr(self.copy_number_call),
                          self._repr_quality(self.some_quality),
                          self._repr_quality(self.exact_quality),
                          self._repr_quality(self.start_quality),
                          self._repr_quality(self.end_quality)])


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
        self.log_q_tau_tk: np.ndarray = np.concatenate(log_q_tau_tk_shards, axis=0)

        # sample names
        self.sample_names = self._get_sample_names_from_calls_shard(scattered_calls_paths[0])
        self.num_samples = len(self.sample_names)

        # interval list metadata
        interval_list_metadata: IntervalListMetadata = IntervalListMetadata(self.interval_list)
        self.ordered_contig_list = interval_list_metadata.ordered_contig_list
        self.contig_interval_indices = interval_list_metadata.contig_interval_indices
        self.contig_interval_lists: Dict[str, List[Interval]] = {
            contig: [self.interval_list[ti] for ti in self.contig_interval_indices[contig]]
            for contig in self.ordered_contig_list}

        # cnv stay probability for each contig
        self.cnv_stay_prob_t_j: Dict[str, np.ndarray] = dict()
        for contig in self.ordered_contig_list:
            contig_interval_list = self.contig_interval_lists[contig]
            dist_t = np.asarray([contig_interval_list[ti + 1].distance(contig_interval_list[ti])
                                 for ti in range(len(contig_interval_list) - 1)], dtype=types.floatX)
            self.cnv_stay_prob_t_j[contig] = np.exp(-dist_t / self.calling_config.cnv_coherence_length)

        # forward-backward algorithm
        self.theano_forward_backward = TheanoForwardBackward(log_posterior_output=None, include_alpha_beta_output=True)

        # viterbi algorithm
        self.theano_viterbi = TheanoViterbi()

        # copy-number HMM specs generator
        self.get_copy_number_hmm_specs = HHMMClassAndCopyNumberBasicCaller\
            .get_compiled_copy_number_hmm_specs_theano_func()

    def perform_viterbi_segmentation_for_single_sample(self, sample_index: int):
        # load copy number log emission for the sample
        copy_number_log_emission_tc_shards = ()
        for calls_path in self.scattered_calls_paths:
            copy_number_log_emission_tc_shards += (self._get_log_copy_number_emission_tc_from_calls_shard(
                calls_path, sample_index),)
        copy_number_log_emission_tc = np.concatenate(copy_number_log_emission_tc_shards, axis=0)

        # iterate over contigs and perform segmentation
        sample_name = self.sample_names[sample_index]
        sample_segments: List[ConstantCopyNumberSegment] = []
        for contig in self.ordered_contig_list:
            # copy-number prior probabilities for each class
            contig_baseline_copy_number_state = self.sample_metadata_collection.get_sample_ploidy_metadata(sample_name) \
                .get_contig_ploidy(contig)
            pi_jkc = HHMMClassAndCopyNumberBasicCaller.get_copy_number_prior_for_sample_jkc(
                self.calling_config.num_copy_number_states,
                self.calling_config.p_alt,
                np.asarray([contig_baseline_copy_number_state], dtype=types.med_uint))

            # contig interval list and indices
            contig_interval_list = self.contig_interval_lists[contig]
            contig_interval_indices = self.contig_interval_indices[contig]

            # mapping from intervals to contig index (since we have a single contig, all intervals map to index=0)
            t_to_j_map = np.zeros((len(contig_interval_list),), dtype=types.med_uint)

            # copy-number class log probability
            log_q_tau_tk = self.log_q_tau_tk[contig_interval_indices, :]

            # copy-number log emission probability for contig intervals
            copy_number_log_emission_contig_tc = copy_number_log_emission_tc[contig_interval_indices, :]

            # get HMM specs
            hmm_specs = self.get_copy_number_hmm_specs(
                pi_jkc, self.cnv_stay_prob_t_j[contig], log_q_tau_tk, t_to_j_map)
            log_prior_c = hmm_specs[0]
            log_trans_tcc = hmm_specs[1]

            # run forward-back algorithm
            fb_result = self.theano_forward_backward.perform_forward_backward_no_admix(
                log_prior_c, log_trans_tcc, copy_number_log_emission_contig_tc)
            log_posterior_prob_tc = fb_result[0]
            log_data_likelihood = fb_result[1]
            alpha_tc = fb_result[2]
            beta_tc = fb_result[3]

            # run viterbi algorithm
            viterbi_path_t_contig = self.theano_viterbi.get_viterbi_path(
                log_prior_c, log_trans_tcc, copy_number_log_emission_contig_tc)

            # initialize the segment quality calculator
            segment_quality_calculator = HMMSegmentationQualityCalculator(
                copy_number_log_emission_contig_tc, log_trans_tcc,
                alpha_tc, beta_tc, log_posterior_prob_tc, log_data_likelihood)

            # coalesce into piecewise constant copy-number segments, calculate qualities
            for copy_number_call, start_index, end_index in self.coalesce_seq_into_segments(viterbi_path_t_contig):
                segment = ConstantCopyNumberSegment(contig,
                                                    contig_interval_list[start_index].start,
                                                    contig_interval_list[end_index].end,
                                                    end_index - start_index + 1,
                                                    copy_number_call)
                segment.some_quality = segment_quality_calculator.get_segment_some_quality(
                    start_index, end_index, copy_number_call)
                segment.exact_quality = segment_quality_calculator.get_segment_exact_quality(
                    start_index, end_index, copy_number_call)
                segment.start_quality = segment_quality_calculator.get_segment_start_quality(
                    start_index, copy_number_call)
                segment.end_quality = segment_quality_calculator.get_segment_end_quality(
                    end_index, copy_number_call)
                sample_segments.append(segment)

        return sample_segments

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

    @staticmethod
    def coalesce_seq_into_segments(seq: List[TypeVar('_T')]) -> List[Tuple[TypeVar('_T'), int, int]]:
        """Coalesces a sequence of objects into piecewise constant segments, along with start and end indices
        for each constant segment.

        Example:
            seq = ['a', 'a', 'a', 'a', 'b', 'c', 'c', 'a', 'a', 'a']
            result = [('a', 0, 3), ('b', 4, 4), ('c', 5, 6), ('a', 7, 9)]

        Args:
            seq: a sequence of objects that implement __equals__

        Returns:
            a generator for (object, start_index, end_index)
        """
        for seg in itertools.groupby(enumerate(seq), key=lambda elem: elem[1]):
            seg_const = seg[0]
            grouper = seg[1]
            start_index = grouper.__next__()[0]
            end_index = start_index
            try:
                while True:
                    end_index = grouper.__next__()[0]
            except StopIteration:
                pass
            yield (seg_const, start_index, end_index)
