import numpy as np

from .exosimTool import ExoSimTool


class ADCGainEstimator(ExoSimTool):
    """
    It computes the desired ADC gain, given the number of bits used by the ADC
    and the maximum number of counts in the pixel to represent.

    Returns
    --------
    dict
        ADC gain factors

    Examples
    ----------

    >>> import exosim.tools as tools
    >>>
    >>> tools.ADCGainEstimator(options_file='tools_input_example.xml')
    """

    def __init__(self, options_file, output=None):
        """
        Parameters
        __________
        parameters: dict
            dictionary containing the parameters. This is usually parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`
        """
        super().__init__(options_file)

        self.info("creating dead pixel map")

        for ch in self.ch_list:
            gain_factor, dtype_max_val, int_type = self.model(
                self.ch_param[ch]
            )

            self.info("---------- {} --------".format(ch))
            self.info("gain factor: {}".format(gain_factor))
            self.info("max adc value: {}".format(dtype_max_val))
            self.info("integer dtype: {}".format(int_type))

            # prepare scheme output
            read_dict = {
                "gain factor": gain_factor,
                "max adc value": dtype_max_val,
                "integer dtype": int_type,
            }
            self.results.update({ch: read_dict})

    def model(self, parameters):
        """

        Parameters
        ----------
        parameters: dict
            dictionary contained the sources parameters. This is usually parsed from :class:`~exosim.tasks.load.loadOptions.LoadOptions`

        Returns
        --------
        float:
            adc gain factor
        int:
            max adc value
        str:
            data type used in ExoSim

        """
        self.info("converting to digital")

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

        dtype_max_val = 2**bits_num - 1

        int_type = np.dtype("int32")
        if bits_num <= 16:
            int_type = np.dtype("int16")
        if bits_num <= 8:
            int_type = np.dtype("int8")

        if "ADC_max_value" in parameters["detector"].keys():
            desired_max = parameters["detector"]["ADC_max_value"]
        else:
            desired_max = parameters["detector"]["well_depth"] * 1.1

        gain_factor = dtype_max_val / desired_max

        return gain_factor, dtype_max_val, int_type
