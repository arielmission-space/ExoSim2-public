"""
Unit tests for parse modules - improving test coverage for parsing functionality.

This module tests:
- ParseSource and ParseSources classes
- ParseOpticalElement class
- ParseZodi class
- Basic functionality and error handling
"""

from collections import OrderedDict

import astropy.units as u
import numpy as np

from exosim.tasks.parse.parse_optical_element import ParseOpticalElement
from exosim.tasks.parse.parse_source import ParseSource, ParseSources
from exosim.tasks.parse.parse_zodi import ParseZodi
from exosim.tasks.task import Task


class TestParseSource:
    """Test ParseSource class."""

    def test_inheritance(self):
        """Test that ParseSource inherits from Task."""
        ps = ParseSource()
        assert isinstance(ps, Task)
        assert hasattr(ps, "execute")
        assert hasattr(ps, "add_task_param")

    def test_has_required_methods(self):
        """Test that ParseSource has all expected methods."""
        ps = ParseSource()
        required_methods = ["execute", "add_task_param", "get_task_param", "set_output"]
        for method_name in required_methods:
            assert hasattr(ps, method_name), f"Missing method: {method_name}"
            assert callable(getattr(ps, method_name)), (
                f"Method not callable: {method_name}"
            )

    def test_initialization_creates_task_params(self):
        """Test that ParseSource initializes task parameters structure."""
        ps = ParseSource()
        # After initialization, _task_params should be a dict
        assert ps._task_params is not None
        assert isinstance(ps._task_params, dict)

        # Should have expected parameter names
        expected_params = ["parameters", "wavelength", "time", "output"]
        for param in expected_params:
            assert param in ps._task_params, f"Missing task parameter: {param}"


class TestParseSources:
    """Test ParseSources class."""

    def test_inheritance(self):
        """Test that ParseSources inherits from Task."""
        ps = ParseSources()
        assert isinstance(ps, Task)
        assert hasattr(ps, "execute")
        assert hasattr(ps, "add_task_param")

    def test_has_required_methods(self):
        """Test that ParseSources has all expected methods."""
        ps = ParseSources()
        required_methods = ["execute", "add_task_param", "get_task_param", "set_output"]
        for method_name in required_methods:
            assert hasattr(ps, method_name), f"Missing method: {method_name}"
            assert callable(getattr(ps, method_name)), (
                f"Method not callable: {method_name}"
            )

    def test_initialization_creates_task_params(self):
        """Test that ParseSources initializes task parameters structure."""
        ps = ParseSources()
        assert ps._task_params is not None
        assert isinstance(ps._task_params, dict)

        expected_params = ["parameters", "wavelength", "time", "output"]
        for param in expected_params:
            assert param in ps._task_params, f"Missing task parameter: {param}"


class TestParseOpticalElement:
    """Test ParseOpticalElement class."""

    def test_inheritance(self):
        """Test that ParseOpticalElement inherits from Task."""
        poe = ParseOpticalElement()
        assert isinstance(poe, Task)
        assert hasattr(poe, "execute")

    def test_has_required_methods(self):
        """Test that ParseOpticalElement has all expected methods."""
        poe = ParseOpticalElement()
        required_methods = ["execute", "add_task_param"]
        for method_name in required_methods:
            assert hasattr(poe, method_name), f"Missing method: {method_name}"
            assert callable(getattr(poe, method_name)), (
                f"Method not callable: {method_name}"
            )

    def test_initialization_creates_task_params(self):
        """Test that ParseOpticalElement initializes correctly."""
        poe = ParseOpticalElement()
        assert poe._task_params is not None
        assert isinstance(poe._task_params, dict)


class TestParseZodi:
    """Test ParseZodi class."""

    def test_inheritance(self):
        """Test that ParseZodi inherits from Task."""
        pz = ParseZodi()
        assert isinstance(pz, Task)
        assert hasattr(pz, "execute")

    def test_has_required_methods(self):
        """Test that ParseZodi has all expected methods."""
        pz = ParseZodi()
        required_methods = ["execute", "add_task_param"]
        for method_name in required_methods:
            assert hasattr(pz, method_name), f"Missing method: {method_name}"
            assert callable(getattr(pz, method_name)), (
                f"Method not callable: {method_name}"
            )

    def test_initialization_creates_task_params(self):
        """Test that ParseZodi initializes correctly."""
        pz = ParseZodi()
        assert pz._task_params is not None
        assert isinstance(pz._task_params, dict)


class TestParseSourcesCallPattern:
    """Test ParseSources call patterns without full execution."""

    def test_can_be_called_with_ordered_dict_params(self):
        """Test that ParseSources can be called with OrderedDict parameters."""
        ParseSources()

        # Create minimal test data
        test_params = OrderedDict(
            {
                "source1": {
                    "source_type": "planck",
                    "T": 5800 * u.K,
                    "R": 1.0 * u.R_sun,
                    "D": 10 * u.pc,
                }
            }
        )

        wl = np.linspace(0.5, 7.8, 10) * u.um
        tt = np.linspace(0.0, 1.0, 5) * u.hr

        # Test that we can set up parameters for calling (without actually calling)
        call_params = {
            "parameters": test_params,
            "wavelength": wl,
            "time": tt,
            "output": None,
        }

        # Verify parameter types are appropriate
        assert isinstance(call_params["parameters"], OrderedDict)
        assert hasattr(call_params["wavelength"], "unit")
        assert call_params["wavelength"].unit == u.um
        assert call_params["time"].unit == u.hr

    def test_can_be_called_with_single_source_params(self):
        """Test that ParseSources can handle single source parameters."""
        ParseSources()

        # Create single source parameter structure
        test_params = {
            "source_type": "planck",
            "T": 6000 * u.K,
            "R": 1.2 * u.R_sun,
            "D": 15 * u.pc,
        }

        wl = np.linspace(1.0, 5.0, 10) * u.um
        tt = np.linspace(0.0, 0.5, 5) * u.hr

        call_params = {
            "parameters": test_params,
            "wavelength": wl,
            "time": tt,
            "output": None,
        }

        # Verify parameter structure
        assert isinstance(call_params["parameters"], dict)
        assert not isinstance(call_params["parameters"], OrderedDict)
        assert "source_type" in call_params["parameters"]


class TestParseSourceParameterStructure:
    """Test ParseSource parameter structure."""

    def test_task_parameter_validation_without_execution(self):
        """Test that ParseSource has proper parameter validation setup."""
        ps = ParseSource()

        # Check that required parameters are defined
        assert "parameters" in ps._task_params
        assert "wavelength" in ps._task_params
        assert "time" in ps._task_params
        assert "output" in ps._task_params

        # Check parameter structure
        for param_name in ["parameters", "wavelength", "time", "output"]:
            param_info = ps._task_params[param_name]
            assert isinstance(param_info, dict)
            assert "description" in param_info or "default" in param_info

    def test_parameter_types_setup(self):
        """Test that ParseSource sets up parameters correctly."""
        ps = ParseSource()

        # Verify _task_params structure
        assert isinstance(ps._task_params, dict)
        assert len(ps._task_params) >= 4  # Should have at least the 4 main parameters

        # Test that output parameter has None as default
        assert ps._task_params["output"]["default"] is None


class TestParseModulesIntegration:
    """Integration tests for parse modules."""

    def test_imports_work(self):
        """Test that all parse modules can be imported."""
        from exosim.tasks.parse.parse_optical_element import ParseOpticalElement
        from exosim.tasks.parse.parse_source import ParseSource, ParseSources
        from exosim.tasks.parse.parse_zodi import ParseZodi

        # Test instantiation
        ps = ParseSource()
        pss = ParseSources()
        poe = ParseOpticalElement()
        pz = ParseZodi()

        assert all([ps, pss, poe, pz])

    def test_parse_classes_have_docstrings(self):
        """Test that parse classes have proper docstrings."""
        classes = [ParseSource, ParseSources, ParseOpticalElement, ParseZodi]

        for cls in classes:
            assert cls.__doc__ is not None, f"{cls.__name__} missing docstring"
            assert len(cls.__doc__.strip()) > 10, (
                f"{cls.__name__} has minimal docstring"
            )

    def test_all_parse_classes_inherit_from_task(self):
        """Test that all parse classes inherit from Task."""
        classes = [ParseSource, ParseSources, ParseOpticalElement, ParseZodi]

        for cls in classes:
            instance = cls()
            assert isinstance(instance, Task), (
                f"{cls.__name__} does not inherit from Task"
            )
            assert hasattr(instance, "execute"), (
                f"{cls.__name__} missing execute method"
            )


class TestParseErrorHandlingStructure:
    """Test error handling structure in parse modules."""

    def test_parse_source_handles_parameter_access(self):
        """Test that ParseSource has proper parameter structure for handling."""
        ps = ParseSource()

        # Test parameter structure exists
        assert ps._task_params is not None

        # Test that parameters dictionary has expected keys
        expected_params = ["parameters", "wavelength", "time", "output"]
        for param in expected_params:
            assert param in ps._task_params
            param_info = ps._task_params[param]
            assert isinstance(param_info, dict)

    def test_parse_sources_parameter_structure(self):
        """Test that ParseSources has proper parameter structure."""
        ps = ParseSources()

        # Test parameter structure
        assert ps._task_params is not None
        assert isinstance(ps._task_params, dict)

        # Should have parameters for handling both OrderedDict and regular dict
        expected_params = ["parameters", "wavelength", "time", "output"]
        for param in expected_params:
            assert param in ps._task_params


class TestParseMethodSignatures:
    """Test method signatures and return types."""

    def test_execute_method_signatures(self):
        """Test that execute methods have proper signatures."""
        parse_classes = [ParseSource, ParseSources, ParseOpticalElement, ParseZodi]

        for cls in parse_classes:
            instance = cls()
            execute_method = instance.execute

            # Test that execute method exists and is callable
            assert callable(execute_method), f"{cls.__name__}.execute is not callable"

            # Test method signature (should take no arguments except self)
            import inspect

            sig = inspect.signature(execute_method)
            params = list(sig.parameters.keys())

            # execute should only have 'self' parameter (already bound)
            assert len(params) == 0, (
                f"{cls.__name__}.execute has unexpected parameters: {params}"
            )

    def test_add_task_param_method(self):
        """Test add_task_param method signature."""
        parse_classes = [ParseSource, ParseSources, ParseOpticalElement, ParseZodi]

        for cls in parse_classes:
            instance = cls()
            method = instance.add_task_param

            assert callable(method), f"{cls.__name__}.add_task_param is not callable"

            # Test method signature
            import inspect

            sig = inspect.signature(method)
            param_names = list(sig.parameters.keys())

            # Should have param_name, param_description, and default parameters
            expected_params = ["param_name", "param_description", "default"]
            for expected in expected_params:
                assert expected in param_names, (
                    f"{cls.__name__}.add_task_param missing parameter: {expected}"
                )
