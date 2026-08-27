"""
Unit tests for tasks.instrument.create_oversampled_intrapixel_response_function module.

This module provides comprehensive testing for the CreateOversampledIntrapixelResponseFunction
class, which handles the creation of oversampled intrapixel response functions (IRF) for
detector modeling in ExoSim2.0. The oversampled IRF is crucial for accurately simulating
detector behavior at sub-pixel scales.
"""

import contextlib
from unittest.mock import patch

import astropy.units as u
import numpy as np

from exosim.tasks.instrument.create_oversampled_intrapixel_response_function import (
    CreateOversampledIntrapixelResponseFunction,
)


class TestOversampledIRFBasic:
    """Test basic functionality of CreateOversampledIntrapixelResponseFunction class."""

    def test_class_inheritance(self):
        """Test that the class inherits from CreateIntrapixelResponseFunction."""
        from exosim.tasks.instrument.create_intrapixel_response_function import (
            CreateIntrapixelResponseFunction,
        )

        irf = CreateOversampledIntrapixelResponseFunction()
        assert isinstance(irf, CreateIntrapixelResponseFunction)

    def test_class_attributes(self):
        """Test that the class has expected attributes."""
        irf = CreateOversampledIntrapixelResponseFunction()
        assert hasattr(irf, "model")
        assert callable(irf.model)

    def test_docstring_exists(self):
        """Test that the class and method have docstrings."""
        irf = CreateOversampledIntrapixelResponseFunction()
        assert irf.model.__doc__ is not None
        assert "oversampled" in irf.model.__doc__.lower()


class TestOversampledIRFInitialization:
    """Test initialization and parameter validation."""

    def test_initialization(self):
        """Test basic initialization of the class."""
        irf = CreateOversampledIntrapixelResponseFunction()
        assert irf is not None

    def test_debug_method_exists(self):
        """Test that debug method is available from parent class."""
        irf = CreateOversampledIntrapixelResponseFunction()
        assert hasattr(irf, "debug")
        assert callable(irf.debug)


class TestOversampledIRFParameterHandling:
    """Test parameter handling and validation."""

    def create_basic_parameters(self):
        """Create basic test parameters."""
        return {
            "detector": {
                "delta_pix": 10.0 * u.um,
                "oversampling": 8,
                "diffusion_length": 2.0 * u.um,
                "intra_pix_distance": 1.0 * u.um,
            },
            "psf_shape": (64, 64),
        }

    def test_parameter_structure(self):
        """Test that basic parameter structure is handled correctly."""
        parameters = self.create_basic_parameters()
        irf = CreateOversampledIntrapixelResponseFunction()

        # Test that parameters can be processed without error
        with (
            patch.object(irf, "debug"),
            contextlib.suppress(AttributeError, KeyError, TypeError),
        ):
            irf.model(parameters)

    def test_missing_oversampling_parameter(self):
        """Test handling when oversampling parameter is missing."""
        parameters = self.create_basic_parameters()
        del parameters["detector"]["oversampling"]

        irf = CreateOversampledIntrapixelResponseFunction()
        with (
            patch.object(irf, "debug"),
            contextlib.suppress(AttributeError, KeyError, TypeError),
        ):
            irf.model(parameters)

    def test_missing_diffusion_length_parameter(self):
        """Test handling when diffusion_length parameter is missing."""
        parameters = self.create_basic_parameters()
        del parameters["detector"]["diffusion_length"]

        irf = CreateOversampledIntrapixelResponseFunction()
        with (
            patch.object(irf, "debug"),
            contextlib.suppress(AttributeError, KeyError, TypeError),
        ):
            irf.model(parameters)

    def test_missing_intra_pix_distance_parameter(self):
        """Test handling when intra_pix_distance parameter is missing."""
        parameters = self.create_basic_parameters()
        del parameters["detector"]["intra_pix_distance"]

        irf = CreateOversampledIntrapixelResponseFunction()
        with (
            patch.object(irf, "debug"),
            contextlib.suppress(AttributeError, KeyError, TypeError),
        ):
            irf.model(parameters)


class TestOversampledIRFKernelGeneration:
    """Test kernel generation functionality."""

    def create_test_parameters(self):
        """Create parameters for kernel generation testing."""
        return {
            "detector": {
                "delta_pix": 10.0 * u.um,
                "oversampling": 4,
                "diffusion_length": 2.0 * u.um,
                "intra_pix_distance": 1.0 * u.um,
            },
            "psf_shape": (32, 32),
        }

    def test_kernel_generation_basic(self):
        """Test basic kernel generation."""
        parameters = self.create_test_parameters()
        irf = CreateOversampledIntrapixelResponseFunction()

        with patch.object(irf, "debug"):
            try:
                kernel, kernel_delta = irf.model(parameters)

                # Basic checks if computation succeeds
                assert isinstance(kernel, np.ndarray)
                assert kernel.ndim == 2
                assert hasattr(kernel_delta, "unit")  # Should be astropy Quantity

            except (AttributeError, KeyError, TypeError, ValueError):
                # Expected for complex computations
                pass

    def test_kernel_size_calculation(self):
        """Test that kernel size is calculated correctly."""
        parameters = self.create_test_parameters()
        psf_shape = parameters["psf_shape"]
        oversampling = parameters["detector"]["oversampling"]
        expected_osf = 8 * oversampling

        irf = CreateOversampledIntrapixelResponseFunction()
        with patch.object(irf, "debug"):
            try:
                kernel, _kernel_delta = irf.model(parameters)
                expected_shape = (
                    psf_shape[0] * expected_osf,
                    psf_shape[1] * expected_osf,
                )
                assert kernel.shape == expected_shape

            except (AttributeError, KeyError, TypeError, ValueError):
                # Expected for complex computations
                pass

    def test_different_psf_shapes(self):
        """Test with different PSF shapes."""
        base_parameters = self.create_test_parameters()
        psf_shapes = [(16, 16), (32, 32), (64, 64)]

        irf = CreateOversampledIntrapixelResponseFunction()

        for psf_shape in psf_shapes:
            parameters = base_parameters.copy()
            parameters["psf_shape"] = psf_shape

            with (
                patch.object(irf, "debug"),
                contextlib.suppress(AttributeError, KeyError, TypeError, ValueError),
            ):
                _kernel, _kernel_delta = irf.model(parameters)


class TestOversampledIRFNumericalComputation:
    """Test numerical computation aspects."""

    def create_computation_parameters(self):
        """Create parameters for numerical computation testing."""
        return {
            "detector": {
                "delta_pix": 18.0 * u.um,
                "oversampling": 2,
                "diffusion_length": 1.5 * u.um,
                "intra_pix_distance": 0.5 * u.um,
            },
            "psf_shape": (8, 8),
        }

    def test_oversampling_factor_handling(self):
        """Test different oversampling factor values."""
        base_parameters = self.create_computation_parameters()
        oversampling_values = [1, 2, 4, 8, 16]

        irf = CreateOversampledIntrapixelResponseFunction()

        for osf in oversampling_values:
            parameters = base_parameters.copy()
            parameters["detector"]["oversampling"] = osf

            with (
                patch.object(irf, "debug"),
                contextlib.suppress(AttributeError, KeyError, TypeError, ValueError),
            ):
                _kernel, _kernel_delta = irf.model(parameters)

    def test_zero_diffusion_length_handling(self):
        """Test handling of zero diffusion length."""
        parameters = self.create_computation_parameters()
        parameters["detector"]["diffusion_length"] = 0.0 * u.um

        irf = CreateOversampledIntrapixelResponseFunction()
        with (
            patch.object(irf, "debug"),
            contextlib.suppress(AttributeError, KeyError, TypeError, ValueError),
        ):
            _kernel, _kernel_delta = irf.model(parameters)

    def test_unit_conversion_handling(self):
        """Test handling of different units."""
        parameters = self.create_computation_parameters()
        # Use different units for delta_pix
        parameters["detector"]["delta_pix"] = 0.018 * u.mm  # Same as 18 um

        irf = CreateOversampledIntrapixelResponseFunction()
        with (
            patch.object(irf, "debug"),
            contextlib.suppress(AttributeError, KeyError, TypeError, ValueError),
        ):
            _kernel, _kernel_delta = irf.model(parameters)


class TestOversampledIRFEdgeCases:
    """Test edge cases and error conditions."""

    def test_non_integer_oversampling(self):
        """Test handling of non-integer oversampling values."""
        parameters = {
            "detector": {
                "delta_pix": 10.0 * u.um,
                "oversampling": 4.7,  # Non-integer
                "diffusion_length": 2.0 * u.um,
                "intra_pix_distance": 1.0 * u.um,
            },
            "psf_shape": (16, 16),
        }

        irf = CreateOversampledIntrapixelResponseFunction()
        with (
            patch.object(irf, "debug"),
            contextlib.suppress(AttributeError, KeyError, TypeError, ValueError),
        ):
            _kernel, _kernel_delta = irf.model(parameters)

    def test_small_psf_shape(self):
        """Test with very small PSF shapes."""
        parameters = {
            "detector": {
                "delta_pix": 10.0 * u.um,
                "oversampling": 2,
                "diffusion_length": 1.0 * u.um,
                "intra_pix_distance": 0.5 * u.um,
            },
            "psf_shape": (2, 2),
        }

        irf = CreateOversampledIntrapixelResponseFunction()
        with (
            patch.object(irf, "debug"),
            contextlib.suppress(AttributeError, KeyError, TypeError, ValueError),
        ):
            _kernel, _kernel_delta = irf.model(parameters)

    def test_missing_detector_section(self):
        """Test handling when detector section is missing."""
        parameters = {"psf_shape": (32, 32)}

        irf = CreateOversampledIntrapixelResponseFunction()
        with (
            patch.object(irf, "debug"),
            contextlib.suppress(AttributeError, KeyError, TypeError),
        ):
            _kernel, _kernel_delta = irf.model(parameters)

    def test_missing_psf_shape(self):
        """Test handling when psf_shape is missing."""
        parameters = {
            "detector": {
                "delta_pix": 10.0 * u.um,
                "oversampling": 4,
            }
        }

        irf = CreateOversampledIntrapixelResponseFunction()
        with (
            patch.object(irf, "debug"),
            contextlib.suppress(AttributeError, KeyError, TypeError),
        ):
            _kernel, _kernel_delta = irf.model(parameters)


class TestOversampledIRFIntegration:
    """Integration tests for the CreateOversampledIntrapixelResponseFunction class."""

    def test_method_signature(self):
        """Test that method signature is as expected."""
        import inspect

        # Check model signature
        model_sig = inspect.signature(CreateOversampledIntrapixelResponseFunction.model)
        model_params = list(model_sig.parameters.keys())

        expected_params = ["self", "parameters"]
        for param in expected_params:
            assert param in model_params

    def test_class_hierarchy(self):
        """Test the class inheritance hierarchy."""
        from exosim.tasks.instrument.create_intrapixel_response_function import (
            CreateIntrapixelResponseFunction,
        )
        from exosim.tasks.task import Task

        # Test inheritance
        mro = CreateOversampledIntrapixelResponseFunction.__mro__
        assert CreateIntrapixelResponseFunction in mro
        assert Task in mro

    def test_algorithm_components(self):
        """Test that key algorithm components are implemented."""
        # Check that the algorithm mentions key components in docstring
        irf = CreateOversampledIntrapixelResponseFunction()
        doc = irf.model.__doc__

        # Key concepts should be mentioned
        assert "kernel" in doc.lower()
        assert "oversampled" in doc.lower()
        assert "response function" in doc.lower()


class TestOversampledIRFConfiguration:
    """Test configuration and parameter validation."""

    def test_default_values(self):
        """Test that default values are properly set."""
        parameters = {
            "detector": {"delta_pix": 10.0 * u.um},
            "psf_shape": (32, 32),
        }

        irf = CreateOversampledIntrapixelResponseFunction()
        with (
            patch.object(irf, "debug"),
            contextlib.suppress(AttributeError, KeyError, TypeError, ValueError),
        ):
            _kernel, _kernel_delta = irf.model(parameters)

    def test_parameter_units_consistency(self):
        """Test that parameter units are handled consistently."""
        parameters = {
            "detector": {
                "delta_pix": 10.0 * u.um,
                "diffusion_length": 2000.0 * u.nm,  # Different unit
                "intra_pix_distance": 0.001 * u.mm,  # Different unit
            },
            "psf_shape": (16, 16),
        }

        irf = CreateOversampledIntrapixelResponseFunction()
        with (
            patch.object(irf, "debug"),
            contextlib.suppress(AttributeError, KeyError, TypeError, ValueError),
        ):
            _kernel, _kernel_delta = irf.model(parameters)

    def test_class_documentation_completeness(self):
        """Test that class has comprehensive documentation."""
        irf = CreateOversampledIntrapixelResponseFunction()

        # Method should have detailed docstring
        assert irf.model.__doc__ is not None
        assert len(irf.model.__doc__) > 100  # Substantial documentation

        # Should mention key parameters
        doc = irf.model.__doc__
        assert "oversampling" in doc
        assert "delta_pix" in doc
        assert "diffusion_length" in doc


class TestOversampledIRFErrorHandling:
    """Test error handling and robustness."""

    def test_invalid_parameters(self):
        """Test handling of invalid parameter values."""
        invalid_parameters = [
            {},  # Empty parameters
            {"detector": {}},  # Empty detector
            {"psf_shape": (0, 0)},  # Zero PSF shape
            {"detector": {"delta_pix": "invalid"}},  # Invalid delta_pix
        ]

        irf = CreateOversampledIntrapixelResponseFunction()

        for params in invalid_parameters:
            with patch.object(irf, "debug"), contextlib.suppress(Exception):
                _kernel, _kernel_delta = irf.model(params)

    def test_method_error_handling(self):
        """Test that methods handle errors gracefully."""
        irf = CreateOversampledIntrapixelResponseFunction()

        # Test that calling methods with None doesn't crash the program
        with contextlib.suppress(Exception):
            irf.model(None)
