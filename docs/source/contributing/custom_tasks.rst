.. _Custom Tasks:

===================================
Custom Tasks
===================================

`ExoSim` allow the user to replace some of the default :class:`~exosim.tasks.task.Task` with custom versions of the same process.
Before writing a custom task, make you sure to read :ref:`tasks`.

To write a custom :class:`~exosim.tasks.task.Task` we need to create a class that first inherits from the default one,
and then we can overwrite the `model` method.

Let's do an example. Suppose we want to write our own version of :class:`~exosim.tasks.instrument.loadResponsivity.LoadResponsivity`.
This task estimate the detector responsivity and shall be indicated in the channel description, as described in :ref:`responsivity`.

The default task simply reads the right column from the file:

.. code-block:: xml

    <channel> channel_name

        <qe>
            <responsivity_task>LoadResponsivity</responsivity_task>
            <datafile>__ConfigPath__/qe.ecsv</datafile>
        </qe>

Using the :func:`~exosim.tasks.instrument.loadResponsivity.LoadResponsivity.model` method:

.. code-block:: python

    def model(self, parameters, wavelength, time):
        """
        Parameters
        ----------
        parameters: dict
            dictionary contained the sources parameters.
        wavelength: :class:`~astropy.units.Quantity`
            wavelength grid.
        time: :class:`~astropy.units.Quantity`
            time grid.

        Returns
        --------
        :class:`~exosim.models.signal.Signal`
            channel responsivity

        """
        qe_data = parameters['qe']['data']
        wl_ = qe_data['Wavelength']
        qe_ = qe_data[parameters['value']]
        qe = signal.Dimensionless(data=qe_, spectral=wl_)
        qe.spectral_rebin(wavelength)
        qe.temporal_rebin(time)

        responsivity = signal.Signal(spectral=wavelength, time=time,
                                     data=qe.data * wavelength.to(
                                         u.m) / const.c / const.h * u.count)
        return responsivity

The input of the model is the `parameter` dictionary that contains the description of the full channel.
So, let's imagine that instead of reading the data from a file, we want to estimate the quantum efficiency from a quadratic equation:

.. math::

    qe(\lambda) = A \cdot (\frac{\lambda}{\lambda_0})^2 + B \cdot \frac{\lambda}{\lambda_0} + C

where :math:`\lambda_0` is a reference wavelength.
This equation for the quantum efficiency obviously has no physical justification.
This example has been chosen specifically because it is not representative of any physical process, just to focus the attention on the code capabilities.

Then we need to include this model parameters in the channel description:

.. code-block:: xml

    <channel> channel_name

        <qe>
            <A> 1 </A>
            <B> 2 </B>
            <C> 3 </C>
            <wl_0 unit=`micron`> 3.0 </wl_0>
        </qe>

and then we can write our own :class:`~exosim.tasks.task.Task` as

.. code-block:: python

    import exosim.tasks.load as load

    class CustomResponsivity(load.LoadResponsivity):
        """
        Custom responsivity class
        """

        def model(self, parameters, wavelength, time):
            """
            Parameters
            ----------
            parameters: dict
                dictionary contained the sources parameters.
            wavelength: :class:`~astropy.units.Quantity`
                wavelength grid.
            time: :class:`~astropy.units.Quantity`
                time grid.

            Returns
            --------
            :class:`~exosim.models.signal.Signal`
                channel responsivity

            """
            A = parameters['qe']['A']
            B = parameters['qe']['B']
            C = parameters['qe']['C']
            wl_0 = parameters['qe']['wl_0']
            qe_ = A * (wavelength/wl_0)**2 + B * (wavelength/wl_0) + c
            qe = signal.Dimensionless(data=qe_, spectral=wavelength)
            qe.temporal_rebin(time)

            responsivity = signal.Signal(spectral=wavelength, time=time,
                                         data=qe.data * wavelength.to(
                                             u.m) / const.c / const.h * u.count)
            return responsivity

It's important that the custom model returns an object of the same kind of the default one or an error will be raised.

Now we need to store this class in a dedicated file. Assume the file is `your/path/customResponsivity.py`, then you have to indicate it in the `.xml` description as

.. code-block:: xml

    <channel> channel_name

        <qe>
            <responsivity_task> your/path/customResponsivity.py </responsivity_task>
            <A> 1 </A>
            <B> 2 </B>
            <C> 3 </C>
            <wl_0 unit=`micron`> 3.0 </wl_0>
        </qe>

Now `ExoSim` will run your task instead of the default one.
