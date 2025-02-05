import logging
import numpy as np
import pandas as pd
import wfdb
import wfdb.processing

from enum import Enum


logger = logging.getLogger(__name__)


class ECG():
    """
    Class, based on the WFDB package, for easier handling of ECG signals.
    """
    class Lead(Enum):
        """
        Enumeration of valid ECG leads.
        """
        AVF = "avf"
        AVL = "avl"
        AVR = "avr"
        I   = "i"
        II  = "ii"
        III = "iii"
        V1  = "v1"
        V2  = "v2"
        V3  = "v3"
        V4  = "v4"
        V5  = "v5"
        V6  = "v6"


    def __init__(self, filepath :str, filename :str, leads :list['ECG.Lead']|None=None):
        """
        Parameters
        ----------
        filepath : str
            Path to the folder containing the data files.
        filename : str
            Name of the data file to read without any file extensions.
        leads : list[ECG.Lead] | None, optional
            Annotations of ECG leads to be considered, by default ``None``.
            If ``None``, all leads are considered.
        """
        if leads is None:
            leads = [lead for lead in ECG.Lead]

        self.filepath = filepath
        self.filename = filename

        self.signals :wfdb.Record = None
        self.leads :list[ECG.Lead] = list()
        self.annotations :dict[ECG.Lead, wfdb.Annotation] = dict()

        try:
            self.signals = wfdb.rdrecord(filepath + filename)
        except FileNotFoundError:
            logger.error(f"Data file '{filepath + filename}' not found")
            raise FileNotFoundError(f"Data file '{filepath + filename}' not found") from None

        for lead in leads:
            try:
                self.annotations[lead] = wfdb.rdann(filepath + filename, lead.value, return_label_elements=["symbol", "label_store"], summarize_labels=True)
                self.leads.append(lead)
            except FileNotFoundError:
                logger.warning(f"Annotation file '{filepath + filename}.{lead.value}' not found")

        self.fs = self.signals.fs
        self.size = self.signals.sig_len
        return


    def get_annotation(self, lead :'ECG.Lead') -> wfdb.Annotation:
        """
        Returns the annotation of the given lead.

        Parameters
        ----------
        lead : ECG.Lead
            The lead of which to recover the annotation.

        Returns
        -------
        : wfdb.Annotation
            Annotation of the given lead.

        Raises
        ------
        : ValueError
            If the given lead is not known for this ECG.
        """
        self._check_lead(lead)
        return self.annotations[lead]


    def get_annotation_desc(self, lead :'ECG.Lead') -> pd.DataFrame:
        """
        Returns the description of the annotation of the given lead.

        Parameters
        ----------
        lead : ECG.Lead
            The lead of which to recover the annotation.

        Returns
        -------
        : pd.DataFrame
            Description of the annotation of the given lead.

        Raises
        ------
        : ValueError
            If the given lead is not known for this ECG.
        """
        self._check_lead(lead)
        return self.annotations[lead].contained_labels


    def get_mean_hr(self, lead :'ECG.Lead') -> int:
        """
        This function computes the mean heart rate of the given lead,
        in particular it returns the average duration of a beat measured in number of samples.

        Parameters
        ----------
        lead : ECG.Lead
            The lead of ECG considered.

        Returns
        -------
        : int
            Mean heart beat in number of samples.

        Raises
        ------
        : ValueError
            If the given lead is not known for this ECG.
        """
        self._check_lead(lead)
        hr = wfdb.processing.calc_mean_hr(
            wfdb.processing.ann2rr(self.filepath + self.filename, lead.value, as_array=True),
            fs=self.fs
        )
        return int(round(hr, 0))


    def get_p_signal(self, lead :'ECG.Lead') -> np.ndarray:
        """
        Returns the p-signal of the given lead.

        Parameters
        ----------
        lead : ECG.Lead
            The lead of which to recover the p-signal.

        Returns
        -------
        : np.ndarray of shape \(N)
            P-signal of the given lead.

        Raises
        ------
        : ValueError
            If the given lead is not known for this ECG.
        """
        self._check_lead(lead)
        return self.signals.p_signal[:, self.signals.sig_name.index(lead.value)]


    def get_record(self) -> wfdb.Record:
        """
        Returns the record of the signal.

        Returns
        -------
        : wfdb.Record
            Record of the signal.
        """
        return self.signals


    def signals_to_df(self) -> pd.DataFrame:
        """
        Returns the signals as a pandas DataFrame.

        Returns
        -------
        : pd.DataFrame
            ECG leads as a pandas DataFrame.
        """
        return self.signals.to_dataframe()


    def t_waves(self, lead :'ECG.Lead', peak_symbol :str="t", onset_symbol :str="(", end_symbol=")") -> np.ndarray:
        """
        Returns the indeces of the samples corresponding to the start, peak and end points
        of each T-wave present in the signal of the lead considered.

        Parameters
        ----------
        lead : ECG.Lead
            The lead considered.
        peak_symbol : str, optional
            Symbol of the wave peak, by default is ``'t'``.
        onset_symbol : str, optional
            Symbol of the wave onset, by default is ``'('``.
        end_symbol : str, optional
            Symbol of the wave end, by default is ``')'``.

        Returns
        -------
        : np.ndarray of shape \(W, 3)
            Indices of the samples corresponding to the start, peak and end points
            of the ``W`` T-waves present in the signal.

        Raises
        ------
        : ValueError
            If the given lead is not known for this ECG.
        """
        self._check_lead(lead)
        return self._find_subseqs(lead, onset_symbol, peak_symbol, end_symbol)


    def _check_lead(self, lead :'ECG.Lead') -> None:
        """
        Checks if the given lead is known for this ECG.

        Parameters
        ----------
        lead : ECG.Lead
            Lead to check.

        Raises
        ------
        : ValueError
            If the given lead is not known for this ECG.
        """
        if lead not in self.leads:
            logger.error(f"Lead '{lead.name}' not found for record '{self.filepath + self.filename}'")
            raise ValueError(f"Lead '{lead.name}' not found for record '{self.filepath + self.filename}'")
        return None


    def _find_subseqs(self, lead :'ECG.Lead', symbol_1 :str, symbol_2 :str, symbol_3) -> np.ndarray:
        """
        Finds all the occurrences of the sub-sequence formed by the three given symbols
        within the annotation of the given lead and returns the corresponding samples (their indeces).

        Parameters
        ----------
        lead : ECG.Lead
            Lead considered.
        symbol_1 : str
            First symbol of the sub-sequence.
        symbol_2 : str
            Second symbol of the sub-sequence.
        symbol_3 : str
            Third symbol of the sub-sequence.

        Returns
        -------
        : np.ndarray of shape \(N, 3)
            Indeces of the samples associated with the ``N`` sub-sequences found
            in the annotation matching the given sub-sequence.
        """
        assert(lead in self.leads)

        ann = self.get_annotation(lead)
        ann_desc = self.get_annotation_desc(lead)

        if (symbol_1 not in ann_desc["symbol"].values) or (symbol_2 not in ann_desc["symbol"].values) or (symbol_3 not in ann_desc["symbol"].values):
            logger.error(f"Symbols '{symbol_1}', '{symbol_2}' and '{symbol_3}' not found in ecg '{self.filename}.{lead.value}'")
            raise FileNotFoundError(f"Symbols '{symbol_1}', '{symbol_2}' and '{symbol_3}' not found in ecg '{self.filename}.{lead.value}'") from None

        sym1_id = ann_desc[ann_desc["symbol"] == symbol_1]["label_store"].iloc[0]
        sym2_id = ann_desc[ann_desc["symbol"] == symbol_2]["label_store"].iloc[0]
        sym3_id = ann_desc[ann_desc["symbol"] == symbol_3]["label_store"].iloc[0]

        # second symbol's indeces in the matching annotation's subsequences
        sym2_idxs = np.argwhere(
            (np.roll(ann.label_store, 1)[1:-1] == sym1_id) & (ann.label_store[1:-1] == sym2_id) & (np.roll(ann.label_store, -1)[1:-1] == sym3_id)
        ).flatten() + 1

        subseq_idxs = np.zeros((len(sym2_idxs), 3), dtype=np.uint)
        subseq_idxs[:, 0], subseq_idxs[:, 1], subseq_idxs[:, 2]  = sym2_idxs-1, sym2_idxs, sym2_idxs+1

        samples = ann.sample[subseq_idxs]
        return samples