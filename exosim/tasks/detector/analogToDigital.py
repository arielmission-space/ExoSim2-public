from copy import deepcopy

import numpy as np

from exosim.models.signal import Adu
from exosim.tasks.task import Task
from exosim.utils.iterators import iterate_over_chunks


class AnalogToDigital(Task):
    """
    It converts the :math:`counts` units of sub-exposures into :math:`adu` units of NDRs.
    The conversion is related to the user defined number of bits of the ADC.
    In the number of bits is not defined, 32 is the default value.
    This Task select the smallest ``dtype`` to represent the desired data type:

    + if the input number of bits is smaller than 16, a ``int16`` data type is used,
    + if the input number of bits is smaller than 8, a ``int8`` data type is used.

    Otherwise, a ``int32`` data type is used.

    The user should also specify the ADC gain factor to apply to the focal plane before the conversion.

    Finally, the user should specify the rounding method to use to cast the float into integers.
    The `ADC_round_method` keyword indicates which method the ADC should use to cast the float into integers. Three options are available:

    - `floor` which uses :class:`numpy.floor`;
    - `ceil` which uses :class:`numpy.ceil`;
    - `round` which uses :class:`numpy.round`;

    Default is `floor`.


    Returns
    ---------
    :class:`~exosim.models.signal.Adu`
        sub-exposures converted into NDRs :math:`adu` units
    """

    def __init__(self):
        """
        Parameters
        ----------
        """
        self.add_task_param("subexposures", " ")
        self.add_task_param(
            "parameters",
            " ",
        )
        self.add_task_param("output", "output file", None)

    def execute(self):
        self.info("converting to digital")
        subexposures = self.get_task_param("subexposures")
        parameters = self.get_task_param("parameters")
        output = self.get_task_param("output")

        # select the rounding method. Default is 'floor'
        rounders = {"floor": np.floor, "ceil": np.ceil, "round": np.round}
        round_method = "floor"
        if "ADC_round_method" in parameters["detector"].keys():
            round_method = parameters["detector"]["ADC_round_method"]
            if round_method not in ["round", "floor", "ceil"]:
                self.error("round method should be round, floor or ceil")
                raise ValueError("round method should be round, floor or ceil")
        rounder = rounders[round_method]

        if "ADC_num_bit" in parameters["detector"].keys():
            bits_num = parameters["detector"]["ADC_num_bit"]
            if isinstance(bits_num, float):
                if not bits_num.is_integer():
                    self.error("number of ADC bits should be integer")
                    raise TypeError("number of ADC bits should be integer")
            self.debug(
                "ADC set to {} bits.".format(
                    parameters["detector"]["ADC_num_bit"]
                )
            )
        else:
            bits_num = 32
            self.debug("ADC set to 32 bit by dfault.")

        if bits_num > 32:
            self.error("max bits number supported is 32.")
            raise ValueError("max bits number supported is 32.")

        dtype_range = 2**bits_num - 1

        int_type = np.dtype("uint32")
        if bits_num <= 16:
            int_type = np.dtype("uint16")
        if bits_num <= 8:
            int_type = np.dtype("uint8")

        gain_factor = 1.0
        if "ADC_gain" in parameters["detector"]:
            if isinstance(parameters["detector"]["ADC_gain"], str):
                gain_factor = parameters["detector"]["ADC_gain"].lower()
            else:
                gain_factor = parameters["detector"]["ADC_gain"]

        offset = "auto"
        if "ADC_offset" in parameters["detector"]:
            if isinstance(parameters["detector"]["ADC_offset"], str):
                offset = parameters["detector"]["ADC_offset"].lower()
            else:
                offset = parameters["detector"]["ADC_offset"]

        if offset == "auto":
            offset = np.min(subexposures.dataset)

        if gain_factor == "auto":
            max_ = np.max(subexposures.dataset)

            gain_factor = dtype_range / (max_ - offset)
            gain_factor = float(gain_factor)

        self.info("ADC gain:{}".format(gain_factor))
        self.info("ADC offset:{}".format(offset))
        self.info("ADC range: {}".format(dtype_range))

        ndrs = Adu(
            spectral=subexposures.spectral,
            time=subexposures.time,
            data=None,
            spatial=subexposures.spatial,
            shape=subexposures.shape,
            cached=True,
            output=output,
            dataset_name="ADC_ndrs",
            metadata=subexposures.metadata,
            output_path=None,
            dtype=int_type,
        )

        for chunk in iterate_over_chunks(
            subexposures.dataset, desc="converting to digital"
        ):
            data = (
                deepcopy(subexposures.dataset[chunk]) - offset
            ) * gain_factor
            ndrs.dataset[chunk] = rounder(data).astype(int_type)
            ndrs.output.flush()

        ndrs.metadata["ADC"] = {"gain": gain_factor, "offset": offset}

        self.set_output(ndrs)
