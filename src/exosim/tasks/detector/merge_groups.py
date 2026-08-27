from tqdm.auto import tqdm

from exosim.models.signal import Adu
from exosim.tasks.task import Task
from exosim.utils.iterators import iterate_over_chunks


class MergeGroups(Task):
    """
    It averages the NDRs of the same group.
    """

    def __init__(self):
        """
        Parameters
        ----------
        subexposures: :class:`~exosim.models.signal.Counts`
            sub-exposures cached signal
        n_groups: int
            number of groups in the same exposure
        n_ndrs: int
            number of ndrs to merge in the same group
        ndrs: :class:`~exosim.models.signal.Counts`
            ndrs cached signal'
        """
        self.add_task_param("subexposures", " ")
        self.add_task_param("n_groups", " ")
        self.add_task_param("n_ndrs", " ")
        self.add_task_param("output", "output file", None)

    def execute(self):
        self.info("merging groups")
        subexposures = self.get_task_param("subexposures")
        n_groups = self.get_task_param("n_groups")
        n_ndrs = self.get_task_param("n_ndrs")
        output = self.get_task_param("output")

        merged_ndrs = Adu(
            spectral=subexposures.spectral,
            time=subexposures.time[0::n_ndrs],
            data=None,
            spatial=subexposures.spatial,
            shape=(
                subexposures.dataset.shape[0] // n_ndrs,
                subexposures.dataset.shape[1],
                subexposures.dataset.shape[2],
            ),
            cached=True,
            output=output,
            dataset_name="NDRs",
            output_path=None,
            metadata=subexposures.metadata,
            dtype=subexposures.dataset[0, 0, 0].dtype,
        )

        if n_ndrs > 1:
            j = 0
            for i in tqdm(range(n_groups), desc="merging groups"):
                merged_ndrs.dataset[i] = (
                    subexposures.dataset[j : j + n_ndrs].sum(axis=0) / n_ndrs
                )
                j += n_ndrs
                merged_ndrs.output.flush()
        else:
            for chunk in iterate_over_chunks(
                subexposures.dataset, desc="merging groups"
            ):
                merged_ndrs.dataset[chunk] = subexposures.dataset[chunk]

        self.set_output(merged_ndrs)
