.. _Custom Tasks:

============
Custom tasks
============

`ExoSim` lets you replace some of the default :class:`~exosim.tasks.task.Task`
implementations with your own version of the same step. Before writing one, read
:ref:`tasks`.

A custom task is a class that inherits from the default one and overrides its
``model`` method.

An example
----------

Suppose we want our own version of
:class:`~exosim.tasks.instrument.load_responsivity.LoadResponsivity`. This task
estimates the detector responsivity and is named in the channel description, as
described in :ref:`responsivity`.

The default task just reads the right column from a file:

.. code-block:: xml

    <channel> channel_name

        <qe>
            <responsivity_task>LoadResponsivity</responsivity_task>
            <datafile>__ConfigPath__/qe.ecsv</datafile>
        </qe>

through its
:func:`~exosim.tasks.instrument.load_responsivity.LoadResponsivity.model` method:

.. code-block:: python

    def model(self, parameters, wavelength, time):
        """
        Parameters
        ----------
        parameters: dict
            dictionary containing the source parameters.
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

The input to ``model`` is the ``parameters`` dictionary, which holds the full
channel description. Suppose that instead of reading the data from a file we want
to estimate the quantum efficiency from a quadratic law:

.. math::

    qe(\lambda) = A \cdot (\frac{\lambda}{\lambda_0})^2 + B \cdot \frac{\lambda}{\lambda_0} + C

where :math:`\lambda_0` is a reference wavelength. This law has no physical
justification; it is chosen precisely because it does not represent any real
process, so the example stays focused on the code.

We add the model parameters to the channel description:

.. code-block:: xml

    <channel> channel_name

        <qe>
            <A> 1 </A>
            <B> 2 </B>
            <C> 3 </C>
            <wl_0 unit="micron"> 3.0 </wl_0>
        </qe>

and write the task:

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
            qe_ = A * (wavelength/wl_0)**2 + B * (wavelength/wl_0) + C
            qe = signal.Dimensionless(data=qe_, spectral=wavelength)
            qe.temporal_rebin(time)

            responsivity = signal.Signal(spectral=wavelength, time=time,
                                         data=qe.data * wavelength.to(
                                             u.m) / const.c / const.h * u.count)
            return responsivity

The custom ``model`` must return an object of the same type as the default one,
or `ExoSim` raises an error.

Save the class in its own file, say ``your/path/customResponsivity.py``, and
point the description at it:

.. code-block:: xml

    <channel> channel_name

        <qe>
            <responsivity_task> your/path/customResponsivity.py </responsivity_task>
            <A> 1 </A>
            <B> 2 </B>
            <C> 3 </C>
            <wl_0 unit="micron"> 3.0 </wl_0>
        </qe>

`ExoSim` now runs your task instead of the default one.
